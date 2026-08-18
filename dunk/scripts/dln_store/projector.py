"""Pure deterministic reducer for profile and immutable learning events."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
from typing import Any

from .grounding import reduce_grounding_timeline
from .schema import ValidationError, sha256_bytes, validate_event, validate_profile

OUTCOME_LABEL = {
    "pass": "independent-pass",
    "partial": "needs-work",
    "fail": "needs-work",
}

STAGE_OPERATIONS = {
    "acquire": {"acquire", "discriminate"},
    "relate": {"relate", "abstract"},
    "revise": {"predict"},
}

STAGE_TRANSITIONS = {
    "acquire": {"relate"},
    "relate": {"revise"},
    "revise": {"acquire", "relate"},
}


def _reference(
    event_index: dict[str, dict[str, Any]], event_id: str, expected_kind: str, context: str
) -> dict[str, Any]:
    referenced = event_index.get(event_id)
    if referenced is None:
        raise ValidationError(f"{context}: references unknown prior event {event_id!r}")
    if referenced["kind"] != expected_kind:
        raise ValidationError(
            f"{context}: event {event_id!r} must be {expected_kind}, got {referenced['kind']}"
        )
    return referenced


def _evidence_summary(event: dict[str, Any]) -> dict[str, Any]:
    result = {
        "event_id": event["event_id"],
        "occurred_at": event["occurred_at"],
        "operation": event["operation"],
        "outcome": event["outcome"],
        "novelty": event["novelty"],
    }
    if "score" in event:
        result["score"] = event["score"]
        result["max_score"] = event["max_score"]
    return result


def project_state(
    profile: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    profile_bytes: bytes,
    events_bytes: bytes,
) -> dict[str, Any]:
    """Validate references and reduce sources to a byte-stable state object."""
    validate_profile(profile)
    grounding_timeline = reduce_grounding_timeline(events)
    event_index: dict[str, dict[str, Any]] = {}
    event_generation: dict[str, int] = {}
    session_identities: dict[str, str] = {}
    completed_sessions: set[str] = set()
    completed_index: list[dict[str, Any]] = []
    archived_exams: list[dict[str, Any]] = []
    legacy_imports: list[dict[str, Any]] = []
    subjects: dict[str, dict[str, Any]] = {}
    calibration_pairs: list[tuple[float, float]] = []
    stage = "acquire"
    generation = 0
    current_model: dict[str, Any] | None = None
    next_review_date: str | None = None
    next_action: str | None = None

    for position, raw_event in enumerate(events):
        event = validate_event(raw_event, f"events[{position}]")
        event_id = event["event_id"]
        if event_id in event_index:
            raise ValidationError(f"events[{position}].event_id: duplicate event ID {event_id!r}")
        session_id = event["session_id"]
        established = session_identities.setdefault(session_id.casefold(), session_id)
        if established != session_id:
            raise ValidationError(
                f"events[{position}].session_id: {session_id!r} differs only by case from "
                f"{established!r}; session receipt paths must stay unique on "
                "case-insensitive filesystems"
            )
        if session_id in completed_sessions:
            raise ValidationError(
                f"events[{position}]: session {session_id!r} was already completed"
            )
        kind = event["kind"]

        if kind == "assessment":
            if event["operation"] not in STAGE_OPERATIONS[stage]:
                expected = ", ".join(sorted(STAGE_OPERATIONS[stage]))
                raise ValidationError(
                    f"events[{position}].operation: {event['operation']!r} is not valid in "
                    f"stage {stage!r}; expected one of {expected}"
                )
            if "retrieval" in event:
                prior = _reference(
                    event_index,
                    event["retrieval"]["prior_event_id"],
                    "assessment",
                    f"events[{position}].retrieval",
                )
                prior_id = event["retrieval"]["prior_event_id"]
                if event_generation[prior_id] != generation:
                    raise ValidationError(
                        f"events[{position}].retrieval: "
                        "prior assessment belongs to an earlier generation"
                    )
                if prior["subject"]["id"] != event["subject"]["id"]:
                    raise ValidationError(
                        f"events[{position}].retrieval: prior assessment subject differs"
                    )
                prior_at = datetime.fromisoformat(prior["occurred_at"][:-1] + "+00:00")
                current_at = datetime.fromisoformat(event["occurred_at"][:-1] + "+00:00")
                calendar_delay = (current_at.date() - prior_at.date()).days
                retrieval = event["retrieval"]
                if calendar_delay <= 0:
                    raise ValidationError(
                        f"events[{position}].retrieval: assessment must occur on a later UTC date"
                    )
                if retrieval["observed_delay_days"] != calendar_delay:
                    raise ValidationError(
                        f"events[{position}].retrieval.observed_delay_days: "
                        f"expected {calendar_delay} from timestamps"
                    )
                scheduled = date.fromisoformat(retrieval["scheduled_date"])
                if not (prior_at.date() < scheduled <= current_at.date()):
                    raise ValidationError(
                        f"events[{position}].retrieval.scheduled_date: "
                        "must be after the prior assessment and no later than this assessment"
                    )
            subject_id = event["subject"]["id"]
            subject = subjects.setdefault(
                subject_id,
                {
                    "exposure_count": 0,
                    "id": subject_id,
                    "independent": None,
                    "label": event["subject"]["label"],
                    "last_assessment_at": None,
                    "retrieval": {
                        "count": 0,
                        "latest": None,
                        "satisfied_by": None,
                        "status": "not-measured",
                    },
                    "supported": None,
                    "transfer": {"count": 0, "latest_event_id": None},
                    "type": event["subject"]["type"],
                },
            )
            if (
                subject["label"] != event["subject"]["label"]
                or subject["type"] != event["subject"]["type"]
            ):
                raise ValidationError(
                    f"events[{position}].subject: stable subject ID changed label or type"
                )
            subject["exposure_count"] += 1
            subject["last_assessment_at"] = event["occurred_at"]
            subject[event["evidence_mode"]] = _evidence_summary(event)
            if event["novelty"] == "novel":
                subject["transfer"] = {
                    "count": subject["transfer"]["count"] + 1,
                    "latest_event_id": event_id,
                }
            retrieval = event.get("retrieval")
            if retrieval and retrieval["observed_delay_days"] > 0:
                satisfies_gate = (
                    event["evidence_mode"] == "independent" and event["outcome"] == "pass"
                )
                subject["retrieval"] = {
                    "count": subject["retrieval"]["count"] + 1,
                    "latest": {
                        "delay_days": retrieval["observed_delay_days"],
                        "event_id": event_id,
                        "evidence_mode": event["evidence_mode"],
                        "outcome": event["outcome"],
                        "scheduled_date": retrieval["scheduled_date"],
                    },
                    "satisfied_by": (
                        event_id if satisfies_gate else subject["retrieval"]["satisfied_by"]
                    ),
                    "status": "measured",
                }
            if "confidence_before" in event:
                normalized_score = float(event["score"]) / float(event["max_score"])
                calibration_pairs.append((float(event["confidence_before"]), normalized_score))

        elif kind == "model_revision":
            if not event.get("initial_model", False) and stage != "revise":
                raise ValidationError(
                    f"events[{position}]: non-initial model revision requires stage 'revise', "
                    f"current stage is {stage!r}"
                )
            for trigger_id in event["triggering_prediction_event_ids"]:
                trigger = _reference(
                    event_index,
                    trigger_id,
                    "assessment",
                    f"events[{position}].triggering_prediction_event_ids",
                )
                if event_generation[trigger_id] != generation:
                    raise ValidationError(
                        f"events[{position}]: model revision trigger {trigger_id!r} "
                        "belongs to an earlier generation"
                    )
                if trigger["operation"] != "predict":
                    raise ValidationError(
                        f"events[{position}]: model revision trigger "
                        f"{trigger_id!r} is not a prediction"
                    )
            prior_id = event.get("prior_model_revision_event_id")
            if prior_id:
                _reference(
                    event_index,
                    prior_id,
                    "model_revision",
                    f"events[{position}].prior_model_revision_event_id",
                )
                if event_generation[prior_id] != generation:
                    raise ValidationError(
                        f"events[{position}].prior_model_revision_event_id: "
                        "prior model belongs to an earlier generation"
                    )
            current_model = {
                "decision": event["decision"],
                "event_id": event_id,
                "model": event["model"],
                "occurred_at": event["occurred_at"],
                "rationale": event["rationale"],
                "triggering_prediction_event_ids": event["triggering_prediction_event_ids"],
            }
            for field in ("word_count_before", "word_count_after"):
                if field in event:
                    current_model[field] = event[field]

        elif kind == "stage_transition":
            if event["from"] != stage:
                raise ValidationError(f"events[{position}].from: expected current stage {stage!r}")
            if event["to"] not in STAGE_TRANSITIONS[stage]:
                expected = ", ".join(sorted(STAGE_TRANSITIONS[stage]))
                raise ValidationError(
                    f"events[{position}].to: transition from {stage!r} "
                    f"must target one of {expected}"
                )
            assessments: list[dict[str, Any]] = []
            for assessment_id in event["assessment_event_ids"]:
                assessment = _reference(
                    event_index,
                    assessment_id,
                    "assessment",
                    f"events[{position}].assessment_event_ids",
                )
                if event_generation[assessment_id] != generation:
                    raise ValidationError(
                        f"events[{position}]: stage transition evidence "
                        "belongs to an earlier generation"
                    )
                if assessment["evidence_mode"] != "independent":
                    raise ValidationError(
                        f"events[{position}]: stage transition evidence must be independent"
                    )
                if assessment["operation"] not in STAGE_OPERATIONS[stage]:
                    raise ValidationError(
                        f"events[{position}]: transition evidence {assessment_id!r} uses "
                        f"operation {assessment['operation']!r}, not stage {stage!r} evidence"
                    )
                assessments.append(assessment)

            transition = (event["from"], event["to"])
            if transition == ("acquire", "relate"):
                if any(assessment["outcome"] != "pass" for assessment in assessments):
                    raise ValidationError(
                        f"events[{position}]: acquire-to-relate gate requires "
                        "passing independent acquire/discriminate evidence"
                    )
            elif transition == ("relate", "revise"):
                if any(assessment["outcome"] != "pass" for assessment in assessments) or not any(
                    assessment["novelty"] == "novel" for assessment in assessments
                ):
                    raise ValidationError(
                        f"events[{position}]: relate-to-revise gate requires passing "
                        "independent relate/abstract evidence including a novel task"
                    )
            elif transition[0] == "revise":
                if any(assessment["outcome"] == "pass" for assessment in assessments):
                    raise ValidationError(
                        f"events[{position}]: revise fallback requires "
                        "independent partial/failed prediction evidence"
                    )
            stage = event["to"]

        elif kind == "session_completed":
            for evidence_id in event["evidence_event_ids"]:
                evidence = event_index.get(evidence_id)
                if evidence is None:
                    raise ValidationError(
                        f"events[{position}].evidence_event_ids: "
                        f"unknown prior event {evidence_id!r}"
                    )
                if evidence["session_id"] != session_id:
                    raise ValidationError(
                        f"events[{position}].evidence_event_ids: event {evidence_id!r} "
                        "belongs to another session"
                    )
                if evidence["kind"] not in {"assessment", "model_revision", "stage_transition"}:
                    raise ValidationError(
                        f"events[{position}].evidence_event_ids: unsupported receipt evidence kind"
                    )
            completed_sessions.add(session_id)
            next_review_date = event["next_review_date"]
            next_action = event["next_action"]
            completed_index.append(
                {
                    "completed_at": event["occurred_at"],
                    "event_id": event_id,
                    "next_review_date": event["next_review_date"],
                    "receipt": f"sessions/{session_id}.md",
                    "session_id": session_id,
                }
            )

        elif kind == "domain_reset":
            generation += 1
            stage = "acquire"
            subjects = {}
            calibration_pairs = []
            current_model = None
            next_review_date = None
            next_action = None

        elif kind == "exam_cycle_closed":
            archive = {
                "closed_at": event["occurred_at"],
                "event_id": event_id,
                "exam": deepcopy(event["archived_exam"]),
            }
            if "self_reported_outcome" in event:
                archive["self_reported_outcome"] = event["self_reported_outcome"]
            archived_exams.append(archive)

        elif kind in {"syllabus_source_ingested", "syllabus_approval_recorded"}:
            pass

        elif kind == "legacy_snapshot_imported":
            legacy_imports.append(
                {
                    "event_id": event_id,
                    "source_sha256": event["source_sha256"],
                    "claims": deepcopy(event["claims"]),
                    "evidence_eligible": False,
                }
            )

        event_index[event_id] = event
        event_generation[event_id] = generation

    subject_list: list[dict[str, Any]] = []
    for subject_id in sorted(subjects):
        subject = subjects[subject_id]
        independent = subject["independent"]
        if independent is not None:
            status = OUTCOME_LABEL[independent["outcome"]]
            if independent["outcome"] == "pass" and subject["retrieval"]["satisfied_by"] is None:
                status = "needs-retrieval"
        elif subject["supported"] is not None:
            status = "supported-only"
        else:
            status = "insufficient-evidence"
        subject["status"] = status
        subject_list.append(subject)

    if calibration_pairs:
        confidence_mean = sum(pair[0] for pair in calibration_pairs) / len(calibration_pairs)
        score_mean = sum(pair[1] for pair in calibration_pairs) / len(calibration_pairs)
        calibration = {
            "count": len(calibration_pairs),
            "mean_confidence": round(confidence_mean, 6),
            "mean_gap": round(confidence_mean - score_mean, 6),
            "mean_normalized_score": round(score_mean, 6),
            "status": "measured",
        }
    else:
        calibration = {"count": 0, "status": "not-measured"}

    grounding_state = grounding_timeline.projected_state(profile["syllabus"])
    projected_syllabus = [topic["label"] for topic in grounding_state["planning_topics"]]

    return {
        "archived_exams": archived_exams,
        "calibration": calibration,
        "completed_sessions": completed_index,
        "current_model": current_model,
        "domain": profile["domain"],
        "domain_id": profile["domain_id"],
        "exam": deepcopy(profile.get("exam", {})),
        "generation": generation,
        "goal": profile["goal"],
        "grounding": grounding_state,
        "legacy_imports": legacy_imports,
        "next_action": next_action,
        "next_review_date": next_review_date,
        "revision": profile["revision"],
        "schema_version": 1,
        "source": {
            "event_count": len(events),
            "events_sha256": sha256_bytes(events_bytes),
            "profile_sha256": sha256_bytes(profile_bytes),
        },
        "stage": stage,
        "subjects": subject_list,
        "syllabus": projected_syllabus,
    }
