"""Deterministic syllabus grounding reduction and historical citation resolution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .schema import ValidationError, canonical_json, validate_event


def _instant(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp[:-1] + "+00:00")


def _receipt_path(source_version_id: str) -> str:
    return f"syllabus/{source_version_id}.md"


def _source_summary(event: dict[str, Any]) -> dict[str, Any]:
    source = event["source"]
    return {
        "assertion_set_sha256": event["assertion_set_sha256"],
        "filename": source["original_filename"],
        "ingested_at": event["occurred_at"],
        "page_count": source["page_count"],
        "receipt": _receipt_path(source["source_version_id"]),
        "sha256": source["sha256"],
        "source_id": source["source_id"],
        "source_version_id": source["source_version_id"],
    }


def _document_effective(assertion: dict[str, Any]) -> dict[str, Any]:
    result = {
        "assertion_id": assertion["assertion_id"],
        "citations": deepcopy(assertion["evidence"]),
        "field": assertion["field"],
        "normalized_value": deepcopy(assertion["normalized_value"]),
        "origin": "document",
        "source_assertion_id": assertion["assertion_id"],
        "status": assertion["status"],
    }
    if "note" in assertion:
        result["note"] = assertion["note"]
    return result


def _corrected_effective(correction: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    result = {
        "assertion_id": correction["correction_assertion_id"],
        "citations": deepcopy(target["evidence"]),
        "document_status": target["status"],
        "document_value": deepcopy(target["normalized_value"]),
        "field": correction["field"],
        "normalized_value": deepcopy(correction["normalized_value"]),
        "origin": "learner_correction",
        "rationale": correction["rationale"],
        "source_assertion_id": target["assertion_id"],
        "status": correction["status"],
    }
    if "note" in target:
        result["document_note"] = target["note"]
    return result


def _approval_view(approval: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    accepted = set(approval["accepted_assertion_ids"])
    deferred = set(approval["deferred_assertion_ids"])
    corrections = {item["target_assertion_id"]: item for item in approval["corrections"]}
    effective: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    eligible: dict[str, dict[str, Any]] = {}

    for source_assertion in source["assertions"]:
        source_id = source_assertion["assertion_id"]
        disposition: str
        if source_id in corrections:
            entry = _corrected_effective(corrections[source_id], source_assertion)
            disposition = "corrected"
        else:
            entry = _document_effective(source_assertion)
            disposition = "accepted" if source_id in accepted else "deferred"
        if source_id in deferred or entry["status"] == "unresolved":
            unresolved_entry = deepcopy(entry)
            unresolved_entry["disposition"] = disposition
            unresolved.append(unresolved_entry)
            continue
        effective.append(entry)
        eligible[entry["assertion_id"]] = entry

    return {
        "approval": approval,
        "effective_assertions": effective,
        "eligible_by_id": eligible,
        "source": source,
        "unresolved_assertions": unresolved,
    }


@dataclass
class GroundingTimeline:
    """Reduced source/approval history plus immutable historical resolution indexes."""

    source_order: list[str] = field(default_factory=list)
    sources_by_version: dict[str, dict[str, Any]] = field(default_factory=dict)
    sources_by_digest: dict[str, dict[str, Any]] = field(default_factory=dict)
    approvals_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    approval_views: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_approval_before_event: dict[str, str | None] = field(default_factory=dict)
    current_approval_event_id: str | None = None
    latest_source_version_id: str | None = None

    def current_view(self) -> dict[str, Any] | None:
        if self.current_approval_event_id is None:
            return None
        return self.approval_views[self.current_approval_event_id]

    def resolve_assertion(self, approval_event_id: str, assertion_id: str) -> dict[str, Any]:
        """Resolve one historically eligible assertion through its pinned approval."""
        view = self.approval_views.get(approval_event_id)
        if view is None:
            raise ValidationError(f"unknown syllabus approval event {approval_event_id!r}")
        assertion = view["eligible_by_id"].get(assertion_id)
        if assertion is None:
            raise ValidationError(
                f"assertion {assertion_id!r} was not settled and effective in approval "
                f"{approval_event_id!r}"
            )
        return deepcopy(assertion)

    def projected_state(self, legacy_syllabus: list[str]) -> dict[str, Any]:
        """Return the bounded current grounding bundle used by later orchestration."""
        current = self.current_view()
        active_source = _source_summary(current["source"]) if current else None
        active_approval = None
        if current:
            approval = current["approval"]
            active_approval = {
                "approval_set_sha256": approval["approval_set_sha256"],
                "event_id": approval["event_id"],
                "occurred_at": approval["occurred_at"],
            }

        if current is None:
            pending_ids = list(self.source_order)
        else:
            active_index = self.source_order.index(current["source"]["source"]["source_version_id"])
            pending_ids = self.source_order[active_index + 1 :]
        pending_sources = [
            _source_summary(self.sources_by_version[version_id]) for version_id in pending_ids
        ]

        if not self.source_order:
            status = "ungrounded"
        elif current is None:
            status = "approval_required"
        elif pending_sources:
            status = "approved_update_pending"
        else:
            status = "approved"

        if current:
            planning_topics: list[dict[str, Any]] = []
            topics_by_label: dict[str, dict[str, Any]] = {}
            for assertion in current["effective_assertions"]:
                if assertion["field"] != "coverage.topic" or assertion["status"] != "specified":
                    continue
                label = assertion["normalized_value"]
                topic = topics_by_label.get(label)
                if topic is None:
                    topic = {
                        "assertion_ids": [],
                        "citable": True,
                        "label": label,
                    }
                    topics_by_label[label] = topic
                    planning_topics.append(topic)
                topic["assertion_ids"].append(assertion["assertion_id"])
            effective = deepcopy(current["effective_assertions"])
            unresolved = deepcopy(current["unresolved_assertions"])
            legacy_fallback = False
        else:
            planning_topics = [
                {"assertion_ids": [], "citable": False, "label": label} for label in legacy_syllabus
            ]
            effective = []
            unresolved = []
            legacy_fallback = bool(legacy_syllabus)

        return {
            "active_approval": active_approval,
            "active_source": active_source,
            "effective_assertions": effective,
            "legacy_fallback": legacy_fallback,
            "pending_sources": pending_sources,
            "planning_topics": planning_topics,
            "status": status,
            "unresolved_assertions": unresolved,
        }


def _validate_grounding_reference(
    timeline: GroundingTimeline,
    event: dict[str, Any],
    position: int,
    active_approval: dict[str, Any] | None,
) -> None:
    grounding = event.get("grounding")
    if grounding is None:
        return
    if active_approval is None:
        raise ValidationError(
            f"events[{position}].grounding: no syllabus approval was active before this event"
        )
    approval_id = grounding["approval_event_id"]
    if approval_id != active_approval["event_id"]:
        raise ValidationError(
            f"events[{position}].grounding.approval_event_id: expected active approval "
            f"{active_approval['event_id']!r}"
        )
    if _instant(event["occurred_at"]) < _instant(active_approval["occurred_at"]):
        raise ValidationError(
            f"events[{position}].grounding: event timestamp precedes the active approval"
        )
    view = timeline.approval_views[approval_id]
    for assertion_id in grounding["assertion_ids"]:
        if assertion_id not in view["eligible_by_id"]:
            raise ValidationError(
                f"events[{position}].grounding.assertion_ids: assertion {assertion_id!r} "
                "was not settled and effective in the active approval"
            )


def reduce_grounding_timeline(events: list[dict[str, Any]]) -> GroundingTimeline:
    """Validate and reduce syllabus authority independently from learner mastery."""
    timeline = GroundingTimeline()
    latest_source: dict[str, Any] | None = None
    latest_approval: dict[str, Any] | None = None
    corrections_by_id: dict[str, dict[str, Any]] = {}
    latest_grounded_use: dict[str, datetime] = {}
    syllabus_admin_session_ids: set[str] = set()
    prior_events: dict[str, dict[str, Any]] = {}

    for position, raw_event in enumerate(events):
        event = validate_event(raw_event, f"events[{position}]")
        event_id = event["event_id"]
        if event_id in prior_events:
            raise ValidationError(f"events[{position}].event_id: duplicate event ID {event_id!r}")
        timeline.active_approval_before_event[event_id] = (
            latest_approval["event_id"] if latest_approval else None
        )
        kind = event["kind"]

        if kind in {"assessment", "session_completed"}:
            if event["session_id"] in syllabus_admin_session_ids:
                raise ValidationError(
                    f"events[{position}].session_id: learning events must not reuse a "
                    "syllabus administrative session"
                )
            _validate_grounding_reference(timeline, event, position, latest_approval)
            grounding = event.get("grounding")
            if grounding is not None:
                approval_id = grounding["approval_event_id"]
                occurred_at = _instant(event["occurred_at"])
                latest_grounded_use[approval_id] = max(
                    occurred_at,
                    latest_grounded_use.get(approval_id, occurred_at),
                )

        if kind == "syllabus_source_ingested":
            syllabus_admin_session_ids.add(event["session_id"])
            source = event["source"]
            version_id = source["source_version_id"]
            digest = source["sha256"]
            prior_digest = timeline.sources_by_digest.get(digest)
            if prior_digest is not None:
                raise ValidationError(
                    f"events[{position}].source.sha256: digest already registered by "
                    f"event {prior_digest['event_id']!r}"
                )
            if version_id in timeline.sources_by_version:
                raise ValidationError(
                    f"events[{position}].source.source_version_id: duplicate source version "
                    f"{version_id!r}"
                )
            supersedes = source["supersedes_source_version_id"]
            if latest_source is None:
                if supersedes is not None:
                    raise ValidationError(
                        f"events[{position}].source.supersedes_source_version_id: "
                        "first source version must not supersede another version"
                    )
            else:
                if supersedes is None:
                    raise ValidationError(
                        f"events[{position}].source.supersedes_source_version_id: "
                        "later source version must supersede the active source version"
                    )
                prior = timeline.sources_by_version.get(supersedes)
                if prior is None:
                    raise ValidationError(
                        f"events[{position}].source.supersedes_source_version_id: "
                        f"unknown prior source version {supersedes!r}"
                    )
                if prior is not latest_source:
                    raise ValidationError(
                        f"events[{position}].source.supersedes_source_version_id: "
                        "must cite the latest source version; forks are not allowed"
                    )
                if source["source_id"] != prior["source"]["source_id"]:
                    raise ValidationError(
                        f"events[{position}].source.source_id: must preserve the source lineage"
                    )
                if _instant(event["occurred_at"]) <= _instant(prior["occurred_at"]):
                    raise ValidationError(
                        f"events[{position}].occurred_at: must be later than the superseded source"
                    )
            timeline.sources_by_version[version_id] = event
            timeline.sources_by_digest[digest] = event
            timeline.source_order.append(version_id)
            timeline.latest_source_version_id = version_id
            latest_source = event

        elif kind == "syllabus_approval_recorded":
            if any(prior["session_id"] == event["session_id"] for prior in prior_events.values()):
                raise ValidationError(
                    f"events[{position}].session_id: approval administrative sessions "
                    "must not reuse a prior event session"
                )
            syllabus_admin_session_ids.add(event["session_id"])
            source = timeline.sources_by_version.get(event["source_version_id"])
            if source is None:
                raise ValidationError(
                    f"events[{position}].source_version_id: references an unknown prior "
                    "source version"
                )
            if source is not latest_source:
                raise ValidationError(
                    f"events[{position}].source_version_id: must approve the latest source version"
                )
            if event["source_assertion_set_sha256"] != source["assertion_set_sha256"]:
                raise ValidationError(
                    f"events[{position}].source_assertion_set_sha256: does not match source"
                )
            if _instant(event["occurred_at"]) < _instant(source["occurred_at"]):
                raise ValidationError(
                    f"events[{position}].occurred_at: approval cannot precede source ingestion"
                )
            assertions = {item["assertion_id"]: item for item in source["assertions"]}
            accepted = set(event["accepted_assertion_ids"])
            deferred = set(event["deferred_assertion_ids"])
            corrected = {item["target_assertion_id"] for item in event["corrections"]}
            disposition = accepted | deferred | corrected
            expected = set(assertions)
            unknown = disposition - expected
            missing = expected - disposition
            if unknown:
                raise ValidationError(
                    f"events[{position}]: disposition contains unknown assertion IDs: "
                    + ", ".join(sorted(unknown))
                )
            if missing:
                raise ValidationError(
                    f"events[{position}]: disposition omits assertion IDs: "
                    + ", ".join(sorted(missing))
                )
            for correction in event["corrections"]:
                correction_id = correction["correction_assertion_id"]
                if correction_id in assertions:
                    raise ValidationError(
                        f"events[{position}].corrections: correction ID {correction_id!r} "
                        "must not collide with a source assertion ID"
                    )
                target = assertions[correction["target_assertion_id"]]
                if correction["field"] != target["field"]:
                    raise ValidationError(
                        f"events[{position}].corrections: correction field must match its target"
                    )
                source_value = {
                    "normalized_value": target["normalized_value"],
                    "status": target["status"],
                }
                corrected_value = {
                    "normalized_value": correction["normalized_value"],
                    "status": correction["status"],
                }
                if canonical_json(source_value, newline=False) == canonical_json(
                    corrected_value, newline=False
                ):
                    raise ValidationError(
                        f"events[{position}].corrections: unchanged source values must be accepted"
                    )
                prior_correction = corrections_by_id.get(correction_id)
                if prior_correction is not None and canonical_json(
                    prior_correction, newline=False
                ) != canonical_json(correction, newline=False):
                    raise ValidationError(
                        f"events[{position}].corrections: correction ID {correction_id!r} "
                        "was reused with different content"
                    )
                corrections_by_id[correction_id] = correction
            supersedes = event["supersedes_approval_event_id"]
            if latest_approval is None:
                if supersedes is not None:
                    raise ValidationError(
                        f"events[{position}].supersedes_approval_event_id: "
                        "first approval must not supersede another approval"
                    )
            else:
                if supersedes != latest_approval["event_id"]:
                    raise ValidationError(
                        f"events[{position}].supersedes_approval_event_id: "
                        "must cite the active approval; forks are not allowed"
                    )
                approval_time = _instant(event["occurred_at"])
                if approval_time <= _instant(latest_approval["occurred_at"]):
                    raise ValidationError(
                        f"events[{position}].occurred_at: must be later than the "
                        "superseded approval"
                    )
                last_use = latest_grounded_use.get(latest_approval["event_id"])
                if last_use is not None and approval_time <= last_use:
                    raise ValidationError(
                        f"events[{position}].occurred_at: superseding approval must be later "
                        "than every learning event grounded by the active approval"
                    )
            view = _approval_view(event, source)
            timeline.approvals_by_id[event_id] = event
            timeline.approval_views[event_id] = view
            timeline.current_approval_event_id = event_id
            latest_approval = event

        prior_events[event_id] = event

    return timeline
