"""Deterministic generic syllabus grounding and historical citation resolution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .legacy import normalize_legacy_syllabus_source
from .schema import RESERVED_SYLLABUS_EVENT_KINDS, ValidationError, canonical_json, validate_event, validate_locator_against_document


def _instant(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp[:-1] + "+00:00")


def _receipt_path(source_version_id: str) -> str:
    return f"syllabus/{source_version_id}.md"


def _source_summary(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": source["event_id"], "filename": source["display_name"],
        "ingested_at": source["occurred_at"], "media_type": source["media_type"],
        "prepared_document_sha256": source.get("prepared_document_sha256"),
        "receipt": _receipt_path(source["source_version_id"]), "role": source["role"],
        "sha256": source["sha256"], "source_version_id": source["source_version_id"],
        "storage": source["storage"], "phase": source.get("phase", "prepared"),
    }


def _proposal_effective(proposal: dict[str, Any]) -> dict[str, Any]:
    return {
        "assertion_id": proposal["proposal_id"], "source_assertion_id": proposal["proposal_id"],
        "predicate": proposal["predicate"], "field": proposal["predicate"],
        "semantic_roles": deepcopy(proposal["semantic_roles"]), "value_type": proposal["value_type"],
        "status": proposal["status"], "value": deepcopy(proposal["value"]),
        "normalized_value": deepcopy(proposal["value"]), "origin": "document",
        "citations": deepcopy(proposal["locators"]),
    }


def _corrected_effective(correction: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    return {
        "assertion_id": correction.get("correction_id", correction.get("correction_assertion_id")),
        "source_assertion_id": target["proposal_id"], "predicate": correction.get("predicate", correction.get("field")),
        "field": correction.get("predicate", correction.get("field")),
        "semantic_roles": deepcopy(correction.get("semantic_roles", target["semantic_roles"])),
        "value_type": correction.get("value_type", target["value_type"]),
        "status": {"not_specified": "explicitly_unknown", "unresolved": "ambiguous"}.get(correction["status"], correction["status"]),
        "value": deepcopy(correction.get("value", correction.get("normalized_value"))),
        "normalized_value": deepcopy(correction.get("value", correction.get("normalized_value"))),
        "origin": "learner_correction", "rationale": correction["rationale"],
        "document_context": deepcopy(target["locators"]), "citations": [],
        "document_status": target["status"], "document_value": deepcopy(target["value"]),
    }


def _decision_view(decision: dict[str, Any], source: dict[str, Any], proposals: list[dict[str, Any]], *, legacy: bool) -> dict[str, Any]:
    proposal_ids = {item["proposal_id"] for item in proposals}
    if legacy:
        accepted = set(decision["accepted_assertion_ids"])
        deferred = set(decision["deferred_assertion_ids"])
        rejected: set[str] = set()
        correction_ids = {item["correction_assertion_id"] for item in decision["corrections"]}
        corrections = {item["target_assertion_id"]: item for item in decision["corrections"]}
        reference_field = "approval_event_id"
        digest = decision["approval_set_sha256"]
    else:
        accepted = set(decision["accepted_proposal_ids"])
        deferred = set(decision["deferred_proposal_ids"])
        rejected = set(decision["rejected_proposal_ids"])
        correction_ids = {item["correction_id"] for item in decision["corrections"]}
        corrections = {item["target_proposal_id"]: item for item in decision["corrections"]}
        reference_field = "decision_event_id"
        digest = decision["decision_set_sha256"]
    if correction_ids & proposal_ids:
        raise ValidationError("syllabus correction IDs must not collide with proposal/source assertion IDs")
    effective: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    rejected_entries: list[dict[str, Any]] = []
    eligible: dict[str, dict[str, Any]] = {}
    for proposal in proposals:
        proposal_id = proposal["proposal_id"]
        if proposal_id in corrections:
            entry = _corrected_effective(corrections[proposal_id], proposal)
            disposition = "corrected"
        else:
            entry = _proposal_effective(proposal)
            disposition = "accepted" if proposal_id in accepted else "deferred" if proposal_id in deferred else "rejected"
        entry["disposition"] = disposition
        if proposal_id in rejected:
            rejected_entries.append(entry)
        elif proposal_id in deferred or entry["status"] not in {"specified", "explicitly_unknown"}:
            unresolved.append(entry)
        else:
            effective.append(entry)
            eligible[entry["assertion_id"]] = entry
    return {"approval": decision, "decision": decision, "source": source, "proposals": proposals,
            "effective_assertions": effective, "unresolved_assertions": unresolved,
            "rejected_assertions": rejected_entries, "eligible_by_id": eligible,
            "reference_field": reference_field, "decision_digest": digest}


@dataclass
class GroundingTimeline:
    source_order: list[str] = field(default_factory=list)
    sources_by_version: dict[str, dict[str, Any]] = field(default_factory=dict)
    sources_by_digest: dict[str, dict[str, Any]] = field(default_factory=dict)
    sources_by_event_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    supplement_order: list[str] = field(default_factory=list)
    proposals_by_source_event: dict[str, dict[str, Any]] = field(default_factory=dict)
    approvals_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    approval_views: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_approval_event_id: str | None = None
    supplement_decision_ids: dict[str, str] = field(default_factory=dict)

    def current_view(self) -> dict[str, Any] | None:
        return self.approval_views.get(self.current_approval_event_id) if self.current_approval_event_id else None

    def resolve_assertion(self, authority_event_id: str, assertion_id: str) -> dict[str, Any]:
        view = self.approval_views.get(authority_event_id)
        if view is None:
            raise ValidationError(f"unknown syllabus decision/approval event {authority_event_id!r}")
        assertion = view["eligible_by_id"].get(assertion_id)
        if assertion is None:
            raise ValidationError(f"assertion {assertion_id!r} was not settled and effective in authority {authority_event_id!r}")
        return deepcopy(assertion)

    def projected_state(self, legacy_syllabus: list[str]) -> dict[str, Any]:
        current = self.current_view()
        active_source = _source_summary(current["source"]) if current else None
        active_decision = None
        if current:
            event = current["decision"]
            active_decision = {"event_id": event["event_id"], "occurred_at": event["occurred_at"],
                               "decision_set_sha256": current["decision_digest"], "reference_field": current["reference_field"]}
        active_version = current["source"]["source_version_id"] if current else None
        pending = []
        for version in self.source_order:
            if active_version is None or self.source_order.index(version) > self.source_order.index(active_version):
                pending.append(_source_summary(self.sources_by_version[version]))
        if not self.source_order:
            status = "ungrounded"
        elif current is None:
            latest = self.sources_by_version[self.source_order[-1]]
            status = "decision_required" if latest.get("phase") == "proposed" else "proposal_required"
        elif pending:
            status = "approved_update_pending"
        else:
            status = "approved"
        if current:
            planning_topics: list[dict[str, Any]] = []
            by_label: dict[str, dict[str, Any]] = {}
            for assertion in current["effective_assertions"]:
                if "planning_topic" not in assertion.get("semantic_roles", []) or assertion["status"] != "specified" or not isinstance(assertion["value"], str):
                    continue
                topic = by_label.setdefault(assertion["value"], {"assertion_ids": [], "citable": True, "label": assertion["value"]})
                if topic not in planning_topics:
                    planning_topics.append(topic)
                topic["assertion_ids"].append(assertion["assertion_id"])
            effective = deepcopy(current["effective_assertions"])
            unresolved = deepcopy(current["unresolved_assertions"])
            legacy_fallback = False
        else:
            planning_topics = [{"assertion_ids": [], "citable": False, "label": label} for label in legacy_syllabus]
            effective, unresolved, legacy_fallback = [], [], bool(legacy_syllabus)
        supplements = [_source_summary(self.sources_by_version[version]) for version in self.supplement_order]
        return {"active_approval": active_decision, "active_decision": active_decision,
                "active_source": active_source, "effective_assertions": effective,
                "legacy_fallback": legacy_fallback, "pending_sources": pending,
                "pending_authoritative_sources": pending, "planning_topics": planning_topics,
                "status": status, "supplements": supplements, "unresolved_assertions": unresolved}


def _check_locators(proposals: list[dict[str, Any]], document: dict[str, Any], prefix: str) -> None:
    for p_index, proposal in enumerate(proposals):
        for l_index, locator in enumerate(proposal["locators"]):
            validate_locator_against_document(locator, document, f"{prefix}[{p_index}].locators[{l_index}]")
        for c_index, candidate in enumerate(proposal.get("ambiguity", {}).get("candidates", [])):
            for l_index, locator in enumerate(candidate["locators"]):
                validate_locator_against_document(locator, document, f"{prefix}[{p_index}].ambiguity.candidates[{c_index}].locators[{l_index}]")


def reduce_grounding_timeline(events: list[dict[str, Any]], prepared_documents: dict[str, dict[str, Any]] | None = None) -> GroundingTimeline:
    """Validate and reduce legacy and generic syllabus authority independently from mastery."""
    prepared_documents = prepared_documents or {}
    timeline = GroundingTimeline()
    latest_authoritative: dict[str, Any] | None = None
    latest_authority_event: dict[str, Any] | None = None
    latest_grounded_use: dict[str, datetime] = {}
    seen_event_ids: set[str] = set()
    seen_sessions: set[str] = set()
    admin_sessions: set[str] = set()

    for position, raw_event in enumerate(events):
        event = validate_event(raw_event, f"events[{position}]")
        if event["event_id"] in seen_event_ids:
            raise ValidationError(f"events[{position}].event_id: duplicate event ID {event['event_id']!r}")
        kind = event["kind"]
        if kind in RESERVED_SYLLABUS_EVENT_KINDS:
            if event["session_id"] in seen_sessions:
                raise ValidationError(f"events[{position}].session_id: syllabus administrative sessions must not reuse a prior event session")
            admin_sessions.add(event["session_id"])
        elif event["session_id"] in admin_sessions:
            raise ValidationError(f"events[{position}].session_id: non-syllabus events must not reuse a syllabus administrative session")

        if kind in {"assessment", "session_completed"}:
            if event["session_id"] in admin_sessions:
                raise ValidationError(f"events[{position}].session_id: learning events must not reuse a syllabus administrative session")
            grounding = event.get("grounding")
            if grounding is not None:
                if latest_authority_event is None:
                    raise ValidationError(f"events[{position}].grounding: no syllabus decision was active before this event")
                key = "approval_event_id" if "approval_event_id" in grounding else "decision_event_id"
                if grounding[key] != latest_authority_event["event_id"]:
                    raise ValidationError(f"events[{position}].grounding.{key}: expected active authority {latest_authority_event['event_id']!r}")
                reference_field = timeline.approval_views[grounding[key]]["reference_field"]
                if key != reference_field and not (
                    key == "decision_event_id" and reference_field == "approval_event_id"
                ):
                    raise ValidationError(f"events[{position}].grounding: authority reference key does not match authority kind")
                for assertion_id in grounding["assertion_ids"]:
                    timeline.resolve_assertion(grounding[key], assertion_id)
                used = _instant(event["occurred_at"])
                if used < _instant(latest_authority_event["occurred_at"]):
                    raise ValidationError(f"events[{position}].grounding: event precedes active authority")
                latest_grounded_use[grounding[key]] = max(used, latest_grounded_use.get(grounding[key], used))

        if kind in {"syllabus_source_ingested", "syllabus_source_prepared"}:
            legacy = kind == "syllabus_source_ingested"
            if legacy:
                raw_source = event["source"]
                role, version, digest = "authoritative", raw_source["source_version_id"], raw_source["sha256"]
                predecessor = raw_source["supersedes_source_version_id"]
                document, legacy_proposals = normalize_legacy_syllabus_source(event)
                source = {"event_id": event["event_id"], "occurred_at": event["occurred_at"], "role": role,
                          "source_version_id": version, "sha256": digest, "display_name": raw_source["original_filename"],
                          "media_type": raw_source["media_type"], "storage": "legacy_text_only", "document": document,
                          "prepared_document_sha256": None, "phase": "proposed", "raw_event": event,
                          "proposals": legacy_proposals, "proposal_digest": event["assertion_set_sha256"]}
            else:
                raw_source = event["source"]
                role, version, digest = event["role"], raw_source["source_version_id"], raw_source["sha256"]
                predecessor = event["supersedes_source_version_id"]
                supplied = prepared_documents.get(event["event_id"])
                document = supplied.get("document") if isinstance(supplied, dict) and "document" in supplied else supplied
                if document is None:
                    raise ValidationError(f"events[{position}].prepared: verified prepared content is unavailable")
                source = {"event_id": event["event_id"], "occurred_at": event["occurred_at"], "role": role,
                          "source_version_id": version, "sha256": digest, "display_name": raw_source["display_name"],
                          "media_type": raw_source["media_type"], "storage": "cas", "document": document,
                          "prepared_document_sha256": event["prepared"]["prepared_document_sha256"], "phase": "prepared", "raw_event": event}
            if digest in timeline.sources_by_digest or version in timeline.sources_by_version:
                raise ValidationError(f"events[{position}].source: duplicate source digest/version")
            if role == "authoritative":
                if latest_authoritative is None and predecessor is not None:
                    raise ValidationError(f"events[{position}]: first authoritative source must not supersede another")
                if latest_authoritative is not None and predecessor != latest_authoritative["source_version_id"]:
                    raise ValidationError(f"events[{position}]: authoritative source must supersede the latest source; forks/skips are not allowed")
                if latest_authoritative and _instant(event["occurred_at"]) <= _instant(latest_authoritative["occurred_at"]):
                    raise ValidationError(f"events[{position}].occurred_at: must be later than predecessor")
                latest_authoritative = source
                timeline.source_order.append(version)
            else:
                if predecessor is not None:
                    raise ValidationError(f"events[{position}]: supplements cannot supersede authoritative sources")
                timeline.supplement_order.append(version)
            timeline.sources_by_version[version] = source
            timeline.sources_by_digest[digest] = source
            timeline.sources_by_event_id[event["event_id"]] = source
            if legacy:
                timeline.proposals_by_source_event[event["event_id"]] = {"event": event, "proposals": source["proposals"], "digest": source["proposal_digest"]}

        elif kind == "syllabus_assertions_proposed":
            source = timeline.sources_by_event_id.get(event["prepared_event_id"])
            if source is None or source["storage"] != "cas":
                raise ValidationError(f"events[{position}].prepared_event_id: unknown prepared source")
            if event["role"] != source["role"] or event["source_version_id"] != source["source_version_id"] or event["prepared_document_sha256"] != source["prepared_document_sha256"]:
                raise ValidationError(f"events[{position}]: proposal pins do not match prepared source")
            if source["role"] == "authoritative" and source is not latest_authoritative:
                raise ValidationError(f"events[{position}]: proposals must target latest authoritative source")
            if source["event_id"] in timeline.proposals_by_source_event:
                raise ValidationError(f"events[{position}]: prepared source already has a proposal set")
            if _instant(event["occurred_at"]) < _instant(source["occurred_at"]):
                raise ValidationError(f"events[{position}].occurred_at: proposals cannot precede source preparation")
            _check_locators(event["proposals"], source["document"], f"events[{position}].proposals")
            timeline.proposals_by_source_event[source["event_id"]] = {"event": event, "proposals": event["proposals"], "digest": event["proposal_set_sha256"]}
            source["phase"] = "proposed"

        elif kind in {"syllabus_approval_recorded", "syllabus_decision_recorded"}:
            legacy = kind == "syllabus_approval_recorded"
            if legacy:
                source = timeline.sources_by_version.get(event["source_version_id"])
                proposal_bundle = timeline.proposals_by_source_event.get(source["event_id"] if source else "")
                predecessor = event["supersedes_approval_event_id"]
            else:
                source = timeline.sources_by_event_id.get(event["prepared_event_id"])
                proposal_bundle = timeline.proposals_by_source_event.get(event["prepared_event_id"])
                predecessor = event["supersedes_decision_event_id"]
                if source and (event["source_version_id"] != source["source_version_id"] or event["prepared_document_sha256"] != source["prepared_document_sha256"] or event["role"] != source["role"]):
                    raise ValidationError(f"events[{position}]: decision pins do not match source")
                if proposal_bundle and (event["proposal_event_id"] != proposal_bundle["event"]["event_id"] or event["proposal_set_sha256"] != proposal_bundle["digest"]):
                    raise ValidationError(f"events[{position}]: decision pins do not match proposal set")
            if source is None or proposal_bundle is None:
                raise ValidationError(f"events[{position}]: decision references unknown source/proposals")
            proposal_event = proposal_bundle["event"]
            if _instant(event["occurred_at"]) < _instant(source["occurred_at"]) or _instant(event["occurred_at"]) < _instant(proposal_event["occurred_at"]):
                raise ValidationError(f"events[{position}].occurred_at: decision cannot precede source/proposal")
            proposals = proposal_bundle["proposals"]
            expected = {item["proposal_id"] for item in proposals}
            if legacy:
                decided = set(event["accepted_assertion_ids"]) | set(event["deferred_assertion_ids"]) | {item["target_assertion_id"] for item in event["corrections"]}
            else:
                decided = set(event["accepted_proposal_ids"]) | set(event["deferred_proposal_ids"]) | set(event["rejected_proposal_ids"]) | {item["target_proposal_id"] for item in event["corrections"]}
            if decided != expected:
                raise ValidationError(f"events[{position}]: decision must completely and exactly partition proposal IDs")
            by_id = {item["proposal_id"]: item for item in proposals}
            corrections = event["corrections"]
            for correction in corrections:
                target_id = correction.get("target_proposal_id", correction.get("target_assertion_id"))
                target = by_id[target_id]
                if correction.get("predicate", correction.get("field")) != target["predicate"]:
                    raise ValidationError(f"events[{position}].corrections: correction predicate must match target")
                if not legacy and correction["semantic_roles"] != target["semantic_roles"]:
                    raise ValidationError(f"events[{position}].corrections: correction semantic roles must match target")
            accepted = event.get("accepted_proposal_ids", event.get("accepted_assertion_ids", []))
            if not legacy and any(by_id[item]["status"] == "ambiguous" for item in accepted):
                raise ValidationError(f"events[{position}]: ambiguous proposals cannot be accepted")
            if source["role"] == "authoritative":
                if source is not latest_authoritative:
                    raise ValidationError(f"events[{position}]: decision must target latest authoritative source")
                if latest_authority_event is None:
                    if predecessor is not None:
                        raise ValidationError(f"events[{position}]: first authoritative decision must not supersede another")
                elif predecessor != latest_authority_event["event_id"]:
                    raise ValidationError(f"events[{position}]: decision must supersede active authority; forks are not allowed")
                if latest_authority_event and _instant(event["occurred_at"]) <= _instant(latest_authority_event["occurred_at"]):
                    raise ValidationError(f"events[{position}].occurred_at: must be later than superseded decision")
                last_use = latest_grounded_use.get(latest_authority_event["event_id"]) if latest_authority_event else None
                if last_use and _instant(event["occurred_at"]) <= last_use:
                    raise ValidationError(f"events[{position}].occurred_at: must follow grounded uses of prior authority")
                latest_authority_event = event
                timeline.current_approval_event_id = event["event_id"]
            else:
                prior_id = timeline.supplement_decision_ids.get(source["event_id"])
                if predecessor != prior_id:
                    raise ValidationError(f"events[{position}]: supplement decision predecessor must remain within its supplement")
                timeline.supplement_decision_ids[source["event_id"]] = event["event_id"]
            source["phase"] = "decided"
            view = _decision_view(event, source, proposals, legacy=legacy)
            timeline.approvals_by_id[event["event_id"]] = event
            timeline.approval_views[event["event_id"]] = view

        seen_event_ids.add(event["event_id"])
        seen_sessions.add(event["session_id"])
    return timeline
