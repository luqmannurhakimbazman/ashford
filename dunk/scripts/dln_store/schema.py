"""Validation and canonical serialization for the local store."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
EVENT_KINDS = {
    "assessment",
    "model_revision",
    "stage_transition",
    "session_completed",
    "domain_reset",
    "exam_cycle_closed",
    "legacy_snapshot_imported",
}
STAGES = {"acquire", "relate", "revise"}
OPERATIONS = {"acquire", "discriminate", "relate", "abstract", "predict"}


class StoreError(Exception):
    """Base error with a stable CLI exit code."""

    exit_code = 1


class ValidationError(StoreError):
    exit_code = 2


class StaleRevisionError(StoreError):
    exit_code = 3


class LockError(StoreError):
    exit_code = 4


class RecoveryRequiredError(StoreError):
    exit_code = 5


def _fail(path: str, message: str) -> None:
    raise ValidationError(f"{path}: {message}")


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        _fail(path, "must be an object with string keys")
    return value


def _string(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        _fail(path, "must be a non-empty string" if not allow_empty else "must be a string")
    if "\x00" in value:
        _fail(path, "must not contain NUL")
    return value


def _identifier(value: Any, path: str) -> str:
    value = _string(value, path)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        _fail(path, "must be a portable identifier (letters, digits, dot, underscore, hyphen; max 128)")
    return value


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "must be an integer")
    if minimum is not None and value < minimum:
        _fail(path, f"must be >= {minimum}")
    return value


def _number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(path, "must be a finite number")
    if minimum is not None and result < minimum:
        _fail(path, f"must be >= {minimum}")
    return result


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "must be a boolean")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    return value


def _enum(value: Any, allowed: set[str], path: str) -> str:
    value = _string(value, path)
    if value not in allowed:
        _fail(path, f"must be one of: {', '.join(sorted(allowed))}")
    return value


def _keys(
    obj: dict[str, Any], required: set[str], optional: set[str], path: str
) -> None:
    missing = required - obj.keys()
    if missing:
        _fail(path, f"missing required field(s): {', '.join(sorted(missing))}")
    unknown = obj.keys() - required - optional
    if unknown:
        _fail(path, f"unknown field(s): {', '.join(sorted(unknown))}")


def _string_list(
    value: Any, path: str, *, unique: bool = False, nonempty: bool = False
) -> list[str]:
    items = _list(value, path)
    if nonempty and not items:
        _fail(path, "must not be empty")
    result = [_string(item, f"{path}[{index}]") for index, item in enumerate(items)]
    if unique and len(set(result)) != len(result):
        _fail(path, "must contain unique values")
    return result


def _date(value: Any, path: str, *, allow_null: bool = False) -> str | None:
    if value is None and allow_null:
        return None
    value = _string(value, path)
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValidationError(f"{path}: must be an ISO 8601 date") from exc
    return value


def _timestamp(value: Any, path: str) -> str:
    value = _string(value, path)
    if not value.endswith("Z"):
        _fail(path, "must be a UTC RFC 3339 timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValidationError(f"{path}: must be a UTC RFC 3339 timestamp") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        _fail(path, "must be UTC")
    return value


def canonical_json(value: Any, *, newline: bool = True) -> bytes:
    """Encode JSON deterministically, rejecting non-standard numeric values."""
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"value is not canonical JSON: {exc}") from exc
    return (text + ("\n" if newline else "")).encode("utf-8")


def pretty_json(value: Any) -> bytes:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"value is not JSON-compatible: {exc}") from exc
    return (text + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_domain_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", _string(name, "domain")).strip().casefold()
    return " ".join(normalized.split())


def make_domain_id(name: str) -> str:
    normalized = normalized_domain_name(name)
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-") or "domain"
    slug = slug[:48].rstrip("-") or "domain"
    return f"{slug}-{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:8]}"


def initial_profile(domain: str, goal: str) -> dict[str, Any]:
    return {
        "annotations": [],
        "domain": _string(domain, "domain"),
        "domain_id": make_domain_id(domain),
        "exam": {},
        "goal": _string(goal, "goal"),
        "review_preferences": {},
        "revision": 0,
        "schema_version": SCHEMA_VERSION,
        "syllabus": [],
    }


def validate_profile(value: Any) -> dict[str, Any]:
    profile = _object(value, "profile")
    required = {"schema_version", "domain_id", "revision", "domain", "goal", "syllabus"}
    optional = {"annotations", "review_preferences", "exam"}
    _keys(profile, required, optional, "profile")
    if _integer(profile["schema_version"], "profile.schema_version") != SCHEMA_VERSION:
        _fail("profile.schema_version", f"unsupported version; expected {SCHEMA_VERSION}")
    domain = _string(profile["domain"], "profile.domain")
    expected_id = make_domain_id(domain)
    if _string(profile["domain_id"], "profile.domain_id") != expected_id:
        _fail("profile.domain_id", f"must equal {expected_id!r} for profile.domain")
    _integer(profile["revision"], "profile.revision", minimum=0)
    _string(profile["goal"], "profile.goal")
    _string_list(profile["syllabus"], "profile.syllabus", unique=True)
    if "annotations" in profile:
        _string_list(profile["annotations"], "profile.annotations")
    if "review_preferences" in profile:
        _object(profile["review_preferences"], "profile.review_preferences")
    if "exam" in profile:
        _object(profile["exam"], "profile.exam")
    canonical_json(profile)
    return profile


def parse_profile_bytes(data: bytes, source: str = "profile.yaml") -> dict[str, Any]:
    """Parse the documented JSON-compatible YAML subset without a YAML dependency."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"{source}: must be UTF-8") from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"{source}:{exc.lineno}:{exc.colno}: unsupported YAML; use the JSON-compatible YAML subset ({exc.msg})"
        ) from exc
    return validate_profile(value)


def load_profile(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise ValidationError(f"{path}: profile is missing") from exc
    return parse_profile_bytes(data, str(path)), data


def validate_profile_patch(value: Any) -> dict[str, Any]:
    patch = _object(value, "request.profile_patch")
    allowed = {"domain", "goal", "syllabus", "annotations", "review_preferences", "exam"}
    unknown = patch.keys() - allowed
    if unknown:
        _fail(
            "request.profile_patch",
            "cannot modify system/unknown field(s): " + ", ".join(sorted(unknown)),
        )
    if "domain" in patch:
        _string(patch["domain"], "request.profile_patch.domain")
    if "goal" in patch:
        _string(patch["goal"], "request.profile_patch.goal")
    if "syllabus" in patch:
        _string_list(patch["syllabus"], "request.profile_patch.syllabus", unique=True)
    if "annotations" in patch:
        _string_list(patch["annotations"], "request.profile_patch.annotations")
    for field in ("review_preferences", "exam"):
        if field in patch:
            _object(patch[field], f"request.profile_patch.{field}")
    canonical_json(patch)
    return patch


def _validate_subject(value: Any, path: str) -> None:
    subject = _object(value, path)
    _keys(subject, {"id", "label", "type"}, set(), path)
    _string(subject["id"], f"{path}.id")
    _string(subject["label"], f"{path}.label")
    _string(subject["type"], f"{path}.type")


def _validate_assistance(value: Any, path: str) -> None:
    assistance = _object(value, path)
    _keys(assistance, {"hint_count", "level"}, set(), path)
    hint_count = _integer(assistance["hint_count"], f"{path}.hint_count", minimum=0)
    level = _enum(assistance["level"], {"none", "prompt", "worked"}, f"{path}.level")
    if (level == "none") != (hint_count == 0):
        _fail(path, "level 'none' requires zero hints; prompt/worked requires at least one")


def _validate_retrieval(value: Any, path: str) -> None:
    retrieval = _object(value, path)
    _keys(
        retrieval,
        {"prior_event_id", "scheduled_date", "observed_delay_days"},
        set(),
        path,
    )
    _string(retrieval["prior_event_id"], f"{path}.prior_event_id")
    _date(retrieval["scheduled_date"], f"{path}.scheduled_date")
    _number(retrieval["observed_delay_days"], f"{path}.observed_delay_days", minimum=0)


def validate_event(value: Any, path: str = "event") -> dict[str, Any]:
    event = _object(value, path)
    common = {"schema_version", "event_id", "session_id", "occurred_at", "kind"}
    missing = common - event.keys()
    if missing:
        _fail(path, f"missing required field(s): {', '.join(sorted(missing))}")
    if _integer(event["schema_version"], f"{path}.schema_version") != SCHEMA_VERSION:
        _fail(f"{path}.schema_version", f"unsupported version; expected {SCHEMA_VERSION}")
    _identifier(event["event_id"], f"{path}.event_id")
    _identifier(event["session_id"], f"{path}.session_id")
    _timestamp(event["occurred_at"], f"{path}.occurred_at")
    kind = _enum(event["kind"], EVENT_KINDS, f"{path}.kind")

    if kind == "assessment":
        required = common | {
            "operation", "task_id", "subject", "context_id", "novelty",
            "evidence_mode", "outcome", "rubric_id", "assistance",
        }
        optional = {"score", "max_score", "confidence_before", "retrieval", "response_time_ms"}
        _keys(event, required, optional, path)
        _enum(event["operation"], OPERATIONS, f"{path}.operation")
        _string(event["task_id"], f"{path}.task_id")
        _validate_subject(event["subject"], f"{path}.subject")
        _string(event["context_id"], f"{path}.context_id")
        _enum(event["novelty"], {"repeat", "variant", "novel"}, f"{path}.novelty")
        _enum(event["evidence_mode"], {"independent", "supported"}, f"{path}.evidence_mode")
        _enum(event["outcome"], {"pass", "partial", "fail"}, f"{path}.outcome")
        _string(event["rubric_id"], f"{path}.rubric_id")
        _validate_assistance(event["assistance"], f"{path}.assistance")
        if event["evidence_mode"] == "independent" and event["assistance"] != {
            "hint_count": 0,
            "level": "none",
        }:
            _fail(path, "independent evidence cannot include hints, prompts, or worked assistance")
        if ("score" in event) != ("max_score" in event):
            _fail(path, "score and max_score must be provided together")
        if "score" in event:
            score = _number(event["score"], f"{path}.score", minimum=0)
            maximum = _number(event["max_score"], f"{path}.max_score", minimum=0)
            if maximum <= 0 or score > maximum:
                _fail(path, "max_score must be > 0 and score must not exceed it")
        if "confidence_before" in event:
            confidence = _number(event["confidence_before"], f"{path}.confidence_before")
            if not 0 <= confidence <= 1:
                _fail(f"{path}.confidence_before", "must be between 0 and 1")
            if "score" not in event:
                _fail(path, "confidence_before requires score and max_score")
        if "retrieval" in event:
            _validate_retrieval(event["retrieval"], f"{path}.retrieval")
        if "response_time_ms" in event:
            _integer(event["response_time_ms"], f"{path}.response_time_ms", minimum=0)

    elif kind == "model_revision":
        required = common | {"triggering_prediction_event_ids", "model", "decision", "rationale"}
        optional = {
            "prior_model_revision_event_id", "word_count_before", "word_count_after", "initial_model"
        }
        _keys(event, required, optional, path)
        triggers = _string_list(
            event["triggering_prediction_event_ids"],
            f"{path}.triggering_prediction_event_ids",
            unique=True,
        )
        initial = event.get("initial_model", False)
        if "initial_model" in event:
            _boolean(initial, f"{path}.initial_model")
        if not triggers and not initial:
            _fail(path, "must cite a prediction event unless initial_model is true")
        if triggers and initial:
            _fail(path, "initial_model cannot cite triggering predictions")
        if "prior_model_revision_event_id" in event:
            _string(event["prior_model_revision_event_id"], f"{path}.prior_model_revision_event_id")
        _string(event["model"], f"{path}.model")
        _enum(event["decision"], {"exploit", "revise", "expand", "fallback-independent"}, f"{path}.decision")
        _string(event["rationale"], f"{path}.rationale")
        for field in ("word_count_before", "word_count_after"):
            if field in event:
                _integer(event[field], f"{path}.{field}", minimum=0)

    elif kind == "stage_transition":
        required = common | {"from", "to", "rubric_id", "assessment_event_ids", "decision"}
        _keys(event, required, set(), path)
        start = _enum(event["from"], STAGES, f"{path}.from")
        end = _enum(event["to"], STAGES, f"{path}.to")
        if start == end:
            _fail(path, "from and to must differ")
        _string(event["rubric_id"], f"{path}.rubric_id")
        _string_list(event["assessment_event_ids"], f"{path}.assessment_event_ids", unique=True, nonempty=True)
        _string(event["decision"], f"{path}.decision")

    elif kind == "session_completed":
        required = common | {"next_action", "next_review_date", "evidence_event_ids", "receipt_schema_version"}
        _keys(event, required, set(), path)
        _string(event["next_action"], f"{path}.next_action")
        _date(event["next_review_date"], f"{path}.next_review_date", allow_null=True)
        _string_list(event["evidence_event_ids"], f"{path}.evidence_event_ids", unique=True)
        if _integer(event["receipt_schema_version"], f"{path}.receipt_schema_version") != 1:
            _fail(f"{path}.receipt_schema_version", "unsupported version; expected 1")

    elif kind == "domain_reset":
        _keys(event, common, {"reason"}, path)
        if "reason" in event:
            _string(event["reason"], f"{path}.reason")

    elif kind == "exam_cycle_closed":
        _keys(event, common | {"archived_exam"}, {"self_reported_outcome"}, path)
        _object(event["archived_exam"], f"{path}.archived_exam")
        if "self_reported_outcome" in event:
            _string(event["self_reported_outcome"], f"{path}.self_reported_outcome")

    else:  # legacy_snapshot_imported
        _keys(event, common | {"source_sha256", "claims", "evidence_eligible"}, set(), path)
        source_hash = _string(event["source_sha256"], f"{path}.source_sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", source_hash):
            _fail(f"{path}.source_sha256", "must be a lowercase SHA-256 hex digest")
        _object(event["claims"], f"{path}.claims")
        if _boolean(event["evidence_eligible"], f"{path}.evidence_eligible"):
            _fail(f"{path}.evidence_eligible", "must be false")

    canonical_json(event)
    return event


def validate_commit_request(value: Any) -> dict[str, Any]:
    request = _object(value, "request")
    _keys(request, set(), {"events", "profile_patch"}, "request")
    if not request:
        _fail("request", "must contain events and/or profile_patch")
    events = request.get("events", [])
    _list(events, "request.events")
    for index, event in enumerate(events):
        validate_event(event, f"request.events[{index}]")
    if "profile_patch" in request:
        validate_profile_patch(request["profile_patch"])
    return request


def parse_json_file(path: Path, label: str = "JSON") -> Any:
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise ValidationError(f"{path}: {label} file is missing") from exc
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"{path}: must be UTF-8") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{path}:{exc.lineno}:{exc.colno}: invalid {label}: {exc.msg}") from exc


def parse_events_bytes(data: bytes, source: str = "events.jsonl") -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    offset = 0
    for line_number, raw_line in enumerate(data.splitlines(keepends=True), 1):
        line_offset = offset
        offset += len(raw_line)
        content = raw_line.rstrip(b"\r\n")
        if not content.strip():
            continue
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationError(
                f"{source}: line {line_number}, byte offset {line_offset + exc.start}: invalid UTF-8"
            ) from exc
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            byte_column = len(text[: exc.pos].encode("utf-8"))
            raise ValidationError(
                f"{source}: line {line_number}, byte offset {line_offset + byte_column}: malformed or truncated JSON ({exc.msg})"
            ) from exc
        events.append(validate_event(value, f"{source} line {line_number}"))
    if data and not data.endswith((b"\n", b"\r")):
        raise ValidationError(
            f"{source}: line {len(data.splitlines())}, byte offset {len(data)}: truncated JSONL (missing newline terminator)"
        )
    return events


def encode_event_lines(events: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json(validate_event(event)) for event in events)
