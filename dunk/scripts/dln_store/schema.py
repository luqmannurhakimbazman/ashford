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
RESERVED_SYLLABUS_EVENT_KINDS = {
    "syllabus_source_ingested",
    "syllabus_approval_recorded",
    "syllabus_source_prepared",
    "syllabus_assertions_proposed",
    "syllabus_decision_recorded",
}
EVENT_KINDS = {
    "assessment",
    "model_revision",
    "stage_transition",
    "session_completed",
    "domain_reset",
    "exam_cycle_closed",
    "legacy_snapshot_imported",
    *RESERVED_SYLLABUS_EVENT_KINDS,
}
STAGES = {"acquire", "relate", "revise"}
OPERATIONS = {"acquire", "discriminate", "relate", "abstract", "predict"}
SYLLABUS_ROLES = {"authoritative", "supplement"}
SYLLABUS_SEMANTIC_ROLES = {
    "planning_topic", "course_identity", "schedule", "assessment", "policy",
    "logistics", "reference", "milestone", "other",
}
SYLLABUS_VALUE_TYPES = {
    "text", "integer", "decimal", "boolean", "date", "time", "percentage",
    "list", "object", "unknown",
}


class StoreError(Exception):
    """Base error with a stable CLI exit code and machine-readable code."""

    exit_code = 1
    code = "store_error"


class SyllabusIntakeError(StoreError):
    """Expected, bounded syllabus acquisition or extraction failure."""

    exit_code = 2

    def __init__(self, code: str, message: str, *, phase: str) -> None:
        super().__init__(message)
        self.code = code
        self.phase = phase


class ValidationError(StoreError):
    """Raised when input violates the documented local-store schema."""

    exit_code = 2
    code = "validation_error"


class StaleRevisionError(StoreError):
    """Raised when a commit targets a revision other than the stored one."""

    exit_code = 3
    code = "stale_revision"


class LockError(StoreError):
    """Raised when a domain write lock cannot be acquired or safely broken."""

    exit_code = 4
    code = "lock_unavailable"


class RecoveryRequiredError(StoreError):
    """Raised when an interrupted transaction must be recovered before writing."""

    exit_code = 5
    code = "recovery_required"


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


def _unit_identifier(value: Any, path: str) -> str:
    value = _string(value, path)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value):
        _fail(path, "must be a portable unit identifier (max 128)")
    return value


def _bounded_string(value: Any, path: str, limit: int = 4096, *, allow_empty: bool = False) -> str:
    text = _string(value, path, allow_empty=allow_empty)
    if len(text.encode("utf-8")) > limit:
        _fail(path, f"must be at most {limit} UTF-8 bytes")
    return text


def _identifier(value: Any, path: str) -> str:
    value = _string(value, path)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        _fail(
            path,
            "must be a portable identifier (letters, digits, dot, underscore, hyphen; max 128)",
        )
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


def _keys(obj: dict[str, Any], required: set[str], optional: set[str], path: str) -> None:
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
    """Encode JSON as sorted, indented text for human-editable files."""
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"value is not JSON-compatible: {exc}") from exc
    return (text + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def normalized_domain_name(name: str) -> str:
    """Normalize a domain name to NFKC, casefolded, whitespace-collapsed form."""
    normalized = unicodedata.normalize("NFKC", _string(name, "domain")).strip().casefold()
    return " ".join(normalized.split())


def make_domain_id(name: str) -> str:
    """Derive the stable store-owned domain id (slug plus digest) for a domain name."""
    normalized = normalized_domain_name(name)
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-") or "domain"
    slug = slug[:48].rstrip("-") or "domain"
    return f"{slug}-{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:8]}"


def initial_profile(domain: str, goal: str) -> dict[str, Any]:
    """Build the profile that a freshly initialized domain starts from."""
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
    """Validate a decoded profile, returning it unchanged when it is well formed."""
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
            f"{source}:{exc.lineno}:{exc.colno}: unsupported YAML; "
            f"use the JSON-compatible YAML subset ({exc.msg})"
        ) from exc
    return validate_profile(value)


def load_profile(path: Path) -> tuple[dict[str, Any], bytes]:
    """Read and validate a profile file, returning the profile and its exact bytes."""
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise ValidationError(f"{path}: profile is missing") from exc
    return parse_profile_bytes(data, str(path)), data


def validate_profile_patch(value: Any) -> dict[str, Any]:
    """Validate a user profile patch, rejecting store-owned identity fields."""
    patch = _object(value, "request.profile_patch")
    allowed = {"goal", "syllabus", "annotations", "review_preferences", "exam"}
    unknown = patch.keys() - allowed
    if unknown:
        _fail(
            "request.profile_patch",
            "cannot modify system/unknown field(s): " + ", ".join(sorted(unknown)),
        )
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


def _sha256(value: Any, path: str) -> str:
    digest = _string(value, path)
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        _fail(path, "must be a lowercase SHA-256 hex digest")
    return digest


def _nullable_identifier(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, path)


def _validate_syllabus_normalized_value(value: Any, field: str, status: str, path: str) -> None:
    """Validate read-only legacy-v1 values without fixture-specific ontology rules."""
    _identifier(field, f"{path}.field")
    if status == "not_specified":
        if value is not None:
            _fail(path, "must be null when status is not_specified")
        return
    if status == "specified" and value is None:
        _fail(path, "must not be null when status is specified")
    _bounded_json(value, path)


def syllabus_assertion_set_sha256(assertions: Any) -> str:
    """Hash a complete ordered assertion snapshot using canonical JSON without a newline."""
    return sha256_bytes(canonical_json(assertions, newline=False))


def _validate_syllabus_assertions(
    assertions_value: Any, pages: dict[int, str], path: str
) -> list[dict[str, Any]]:
    assertions = _list(assertions_value, path)
    if not assertions:
        _fail(path, "must not be empty")
    identifiers: set[str] = set()
    by_id: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(assertions):
        assertion_path = f"{path}[{index}]"
        assertion = _object(value, assertion_path)
        _keys(
            assertion,
            {
                "assertion_id",
                "field",
                "status",
                "normalized_value",
                "origin",
                "confidence",
                "evidence",
            },
            {"note"},
            assertion_path,
        )
        assertion_id = _identifier(assertion["assertion_id"], f"{assertion_path}.assertion_id")
        if assertion_id in identifiers:
            _fail(f"{assertion_path}.assertion_id", "must be unique within the source")
        identifiers.add(assertion_id)
        by_id[assertion_id] = assertion
        field = _identifier(assertion["field"], f"{assertion_path}.field")
        status = _enum(
            assertion["status"],
            {"specified", "not_specified", "unresolved"},
            f"{assertion_path}.status",
        )
        _validate_syllabus_normalized_value(
            assertion["normalized_value"], field, status, f"{assertion_path}.normalized_value"
        )
        if assertion["origin"] != "document":
            _fail(f"{assertion_path}.origin", "must be document")
        _enum(
            assertion["confidence"],
            {"high", "ambiguous"},
            f"{assertion_path}.confidence",
        )
        if "note" in assertion:
            _string(assertion["note"], f"{assertion_path}.note")
        evidence_items = _list(assertion["evidence"], f"{assertion_path}.evidence")
        if not evidence_items:
            _fail(f"{assertion_path}.evidence", "must not be empty")
        for evidence_index, evidence_value in enumerate(evidence_items):
            evidence_path = f"{assertion_path}.evidence[{evidence_index}]"
            evidence = _object(evidence_value, evidence_path)
            _keys(
                evidence,
                {"page_number", "start_char", "end_char", "quote"},
                set(),
                evidence_path,
            )
            page_number = _integer(
                evidence["page_number"], f"{evidence_path}.page_number", minimum=1
            )
            if page_number not in pages:
                _fail(f"{evidence_path}.page_number", "references an unknown page")
            start = _integer(evidence["start_char"], f"{evidence_path}.start_char", minimum=0)
            end = _integer(evidence["end_char"], f"{evidence_path}.end_char", minimum=0)
            quote = _string(evidence["quote"], f"{evidence_path}.quote")
            if end <= start or end > len(pages[page_number]):
                _fail(evidence_path, "character interval is outside the page text")
            if pages[page_number][start:end] != quote:
                _fail(evidence_path, "quote must equal the exact page-text slice")
    return assertions


def syllabus_approval_set_sha256(event: dict[str, Any]) -> str:
    """Hash the complete source disposition of an approval event."""
    payload = {
        "accepted_assertion_ids": event["accepted_assertion_ids"],
        "actor": event["actor"],
        "corrections": event["corrections"],
        "deferred_assertion_ids": event["deferred_assertion_ids"],
        "source_assertion_set_sha256": event["source_assertion_set_sha256"],
        "source_version_id": event["source_version_id"],
        "supersedes_approval_event_id": event["supersedes_approval_event_id"],
    }
    return sha256_bytes(canonical_json(payload, newline=False))


def validate_sealable_items(
    value: Any, path: str, *, limit: int = 2048, minimum: int = 0
) -> list[dict[str, Any]]:
    """Type-check an externally supplied array of JSON objects before the store seals it."""
    items = _list(value, path)
    if not minimum <= len(items) <= limit:
        _fail(path, f"must contain {minimum} through {limit} items")
    return [_object(item, f"{path}[{index}]") for index, item in enumerate(items)]


def validate_proposal_id_list(value: Any, path: str, *, limit: int = 2048) -> list[str]:
    """Type-check an externally supplied proposal-ID partition before the store sorts it."""
    items = _string_list(value, path, unique=True)
    if len(items) > limit:
        _fail(path, f"must contain at most {limit} proposal IDs")
    for index, item in enumerate(items):
        _identifier(item, f"{path}[{index}]")
    return items


def _bounded_json(value: Any, path: str, limit: int = 65536) -> Any:
    data = canonical_json(value, newline=False)
    if len(data) > limit:
        _fail(path, f"canonical JSON exceeds {limit} bytes")
    return value


def _validate_typed_value(value: Any, value_type: str, path: str, *, allow_null: bool = False) -> None:
    if value is None:
        if allow_null:
            return
        _fail(path, "must not be null")
    if value_type == "text":
        _string(value, path)
    elif value_type == "integer":
        _integer(value, path)
    elif value_type in {"decimal", "percentage"}:
        number = _number(value, path)
        if value_type == "percentage" and not 0 <= number <= 100:
            _fail(path, "percentage must be between 0 and 100")
    elif value_type == "boolean":
        _boolean(value, path)
    elif value_type == "date":
        _date(value, path)
    elif value_type == "time":
        text = _string(value, path)
        if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?", text):
            _fail(path, "must be an ISO local time")
    elif value_type == "list":
        _list(value, path)
    elif value_type == "object":
        _object(value, path)
    elif value_type == "unknown":
        _bounded_json(value, path)
    _bounded_json(value, path)


def validate_prepared_document(value: Any, path: str = "prepared_document") -> dict[str, Any]:
    document = _object(value, path)
    _keys(document, {"prepared_schema_version", "media_type", "normalization", "units"}, set(), path)
    if _integer(document["prepared_schema_version"], f"{path}.prepared_schema_version") != 1:
        _fail(f"{path}.prepared_schema_version", "unsupported version; expected 1")
    _enum(document["media_type"], {"application/pdf", "text/html"}, f"{path}.media_type")
    normalization = _object(document["normalization"], f"{path}.normalization")
    _keys(normalization, {"policy_id", "unicode", "line_endings"}, set(), f"{path}.normalization")
    _identifier(normalization["policy_id"], f"{path}.normalization.policy_id")
    if normalization["unicode"] != "NFC" or normalization["line_endings"] != "LF":
        _fail(f"{path}.normalization", "must specify NFC Unicode and LF line endings")
    units = _list(document["units"], f"{path}.units")
    if not 1 <= len(units) <= 500:
        _fail(f"{path}.units", "must contain 1 through 500 units")
    ids: set[str] = set()
    total = 0
    nonblank = False
    for index, unit_value in enumerate(units):
        unit_path = f"{path}.units[{index}]"
        unit = _object(unit_value, unit_path)
        _keys(unit, {"unit_id", "kind", "text", "text_sha256"}, {"label"}, unit_path)
        unit_id = _unit_identifier(unit["unit_id"], f"{unit_path}.unit_id")
        if unit_id in ids:
            _fail(f"{unit_path}.unit_id", "must be unique")
        ids.add(unit_id)
        _enum(unit["kind"], {"page", "document", "html_block"}, f"{unit_path}.kind")
        if "label" in unit:
            if len(_string(unit["label"], f"{unit_path}.label")) > 256:
                _fail(f"{unit_path}.label", "must be at most 256 characters")
        text = _string(unit["text"], f"{unit_path}.text", allow_empty=True)
        if "\r" in text or unicodedata.normalize("NFC", text) != text:
            _fail(f"{unit_path}.text", "must be NFC text with LF line endings")
        total += len(text.encode("utf-8"))
        nonblank = nonblank or bool(text.strip())
        if _sha256(unit["text_sha256"], f"{unit_path}.text_sha256") != sha256_bytes(text.encode("utf-8")):
            _fail(f"{unit_path}.text_sha256", "does not match unit text")
    if total > 8 * 1024 * 1024:
        _fail(f"{path}.units", "normalized text exceeds 8 MiB")
    if not nonblank:
        _fail(f"{path}.units", "at least one unit must contain non-whitespace text")
    canonical_json(document, newline=False)
    return document


def prepared_document_bytes(value: Any) -> bytes:
    encoded = canonical_json(validate_prepared_document(value), newline=False)
    if len(encoded) > 16 * 1024 * 1024:
        _fail("prepared_document", "canonical prepared JSON exceeds 16 MiB")
    return encoded


def prepared_document_sha256(value: Any) -> str:
    return sha256_bytes(prepared_document_bytes(value))


def _validate_locator(value: Any, path: str) -> None:
    locator = _object(value, path)
    _keys(locator, {"unit_id", "start_char", "end_char", "quote"}, set(), path)
    _unit_identifier(locator["unit_id"], f"{path}.unit_id")
    start = _integer(locator["start_char"], f"{path}.start_char", minimum=0)
    end = _integer(locator["end_char"], f"{path}.end_char", minimum=0)
    if end <= start:
        _fail(path, "end_char must be greater than start_char")
    _bounded_string(locator["quote"], f"{path}.quote", 16384)


def validate_locator_against_document(locator: dict[str, Any], document: dict[str, Any], path: str = "locator") -> None:
    _validate_locator(locator, path)
    units = {unit["unit_id"]: unit for unit in document["units"]}
    unit = units.get(locator["unit_id"])
    if unit is None:
        _fail(f"{path}.unit_id", "references an unknown prepared unit")
    if locator["end_char"] > len(unit["text"]):
        _fail(path, "character interval is outside the prepared unit")
    if unit["text"][locator["start_char"]:locator["end_char"]] != locator["quote"]:
        _fail(path, "quote must equal the exact prepared-text slice")


def _validate_producer(value: Any, path: str, *, store_invoked: bool = False) -> None:
    producer = _object(value, path)
    _keys(producer, {"trust", "name", "version"}, set(), path)
    allowed = {"external_unverified", "store_invoked"} if store_invoked else {"external_unverified"}
    if producer["trust"] not in allowed:
        _fail(f"{path}.trust", f"must be one of {sorted(allowed)}")
    if len(_string(producer["name"], f"{path}.name")) > 256:
        _fail(f"{path}.name", "must be at most 256 characters")
    if len(_string(producer["version"], f"{path}.version")) > 256:
        _fail(f"{path}.version", "must be at most 256 characters")


def _validate_proposal(value: Any, path: str) -> None:
    proposal = _object(value, path)
    _keys(proposal, {"proposal_id", "display_order", "predicate", "label", "semantic_roles", "value_type", "status", "value", "locators"}, {"ambiguity", "note"}, path)
    _identifier(proposal["proposal_id"], f"{path}.proposal_id")
    _integer(proposal["display_order"], f"{path}.display_order", minimum=0)
    _identifier(proposal["predicate"], f"{path}.predicate")
    if len(_string(proposal["label"], f"{path}.label")) > 256:
        _fail(f"{path}.label", "must be at most 256 characters")
    roles = _string_list(proposal["semantic_roles"], f"{path}.semantic_roles", unique=True, nonempty=True)
    if not 1 <= len(roles) <= 8 or any(role not in SYLLABUS_SEMANTIC_ROLES for role in roles):
        _fail(f"{path}.semantic_roles", "contains an unsupported or excessive semantic role")
    value_type = _enum(proposal["value_type"], SYLLABUS_VALUE_TYPES, f"{path}.value_type")
    status = _enum(proposal["status"], {"specified", "explicitly_unknown", "ambiguous"}, f"{path}.status")
    if "planning_topic" in roles and value_type != "text":
        _fail(path, "planning_topic proposals must use text values")
    locators = _list(proposal["locators"], f"{path}.locators")
    if not 1 <= len(locators) <= 16:
        _fail(f"{path}.locators", "must contain 1 through 16 locators")
    for index, locator in enumerate(locators):
        _validate_locator(locator, f"{path}.locators[{index}]")
    if status == "specified":
        _validate_typed_value(proposal["value"], value_type, f"{path}.value")
        if "ambiguity" in proposal:
            _fail(f"{path}.ambiguity", "is only allowed for ambiguous proposals")
    elif status == "explicitly_unknown":
        if proposal["value"] is not None or "ambiguity" in proposal:
            _fail(path, "explicitly_unknown requires null value and no ambiguity")
    else:
        if proposal["value"] is not None or "ambiguity" not in proposal:
            _fail(path, "ambiguous requires null value and an ambiguity object")
        ambiguity = _object(proposal["ambiguity"], f"{path}.ambiguity")
        _keys(ambiguity, {"reason", "unresolved_dimensions", "candidates"}, set(), f"{path}.ambiguity")
        _bounded_string(ambiguity["reason"], f"{path}.ambiguity.reason")
        dimensions = _string_list(ambiguity["unresolved_dimensions"], f"{path}.ambiguity.unresolved_dimensions", unique=True)
        candidates = _list(ambiguity["candidates"], f"{path}.ambiguity.candidates")
        if not dimensions and not candidates:
            _fail(f"{path}.ambiguity", "requires an unresolved dimension or candidate")
        if len(dimensions) > 32 or len(candidates) > 16:
            _fail(f"{path}.ambiguity", "exceeds ambiguity bounds")
        for index, candidate_value in enumerate(candidates):
            candidate_path = f"{path}.ambiguity.candidates[{index}]"
            candidate = _object(candidate_value, candidate_path)
            _keys(candidate, {"value_type", "value", "locators"}, set(), candidate_path)
            candidate_type = _enum(candidate["value_type"], SYLLABUS_VALUE_TYPES, f"{candidate_path}.value_type")
            _validate_typed_value(candidate["value"], candidate_type, f"{candidate_path}.value")
            candidate_locators = _list(candidate["locators"], f"{candidate_path}.locators")
            if not 1 <= len(candidate_locators) <= 16:
                _fail(f"{candidate_path}.locators", "must contain 1 through 16 locators")
            for locator_index, locator in enumerate(candidate_locators):
                _validate_locator(locator, f"{candidate_path}.locators[{locator_index}]")
    if "note" in proposal:
        _bounded_string(proposal["note"], f"{path}.note")


def syllabus_proposal_set_sha256(event: dict[str, Any]) -> str:
    payload = {key: event[key] for key in ("role", "prepared_event_id", "source_version_id", "prepared_document_sha256", "producer", "proposals")}
    return sha256_bytes(canonical_json(payload, newline=False))


def syllabus_decision_set_sha256(event: dict[str, Any]) -> str:
    payload = {key: event[key] for key in ("role", "prepared_event_id", "source_version_id", "prepared_document_sha256", "proposal_event_id", "proposal_set_sha256", "actor", "accepted_proposal_ids", "deferred_proposal_ids", "rejected_proposal_ids", "corrections", "supersedes_decision_event_id")}
    return sha256_bytes(canonical_json(payload, newline=False))


def _validate_grounding_reference(value: Any, path: str) -> None:
    grounding = _object(value, path)
    _keys(grounding, {"assertion_ids"}, {"approval_event_id", "decision_event_id"}, path)
    authority_keys = [key for key in ("approval_event_id", "decision_event_id") if key in grounding]
    if len(authority_keys) != 1:
        _fail(path, "must contain exactly one of approval_event_id or decision_event_id")
    _identifier(grounding[authority_keys[0]], f"{path}.{authority_keys[0]}")
    assertion_ids = _string_list(
        grounding["assertion_ids"], f"{path}.assertion_ids", unique=True, nonempty=True
    )
    for index, assertion_id in enumerate(assertion_ids):
        _identifier(assertion_id, f"{path}.assertion_ids[{index}]")


def _validate_new_syllabus_event(event: dict[str, Any], kind: str, common: set[str], path: str) -> None:
    if kind == "syllabus_source_prepared":
        _keys(event, common | {"role", "source", "prepared", "acquisition", "extraction", "supersedes_source_version_id"}, set(), path)
        role = _enum(event["role"], SYLLABUS_ROLES, f"{path}.role")
        source = _object(event["source"], f"{path}.source")
        _keys(source, {"source_version_id", "display_name", "media_type", "byte_size", "sha256", "cas_path"}, set(), f"{path}.source")
        digest = _sha256(source["sha256"], f"{path}.source.sha256")
        if _identifier(source["source_version_id"], f"{path}.source.source_version_id") != f"sha256-{digest}":
            _fail(f"{path}.source.source_version_id", "must be sha256-<source digest>")
        name = _string(source["display_name"], f"{path}.source.display_name")
        if Path(name).name != name or name in {".", ".."}:
            _fail(f"{path}.source.display_name", "must be a display filename, not a path")
        _enum(source["media_type"], {"application/pdf", "text/html"}, f"{path}.source.media_type")
        size = _integer(source["byte_size"], f"{path}.source.byte_size", minimum=1)
        if size > 16 * 1024 * 1024:
            _fail(f"{path}.source.byte_size", "must not exceed 16 MiB")
        if source["cas_path"] != f"sources/sha256/{digest}":
            _fail(f"{path}.source.cas_path", "must be derived from source.sha256")
        prepared = _object(event["prepared"], f"{path}.prepared")
        _keys(prepared, {"prepared_document_sha256", "cas_path", "unit_count", "text_byte_size"}, set(), f"{path}.prepared")
        prepared_digest = _sha256(prepared["prepared_document_sha256"], f"{path}.prepared.prepared_document_sha256")
        if prepared["cas_path"] != f"prepared/sha256/{prepared_digest}.json":
            _fail(f"{path}.prepared.cas_path", "must be derived from prepared_document_sha256")
        _integer(prepared["unit_count"], f"{path}.prepared.unit_count", minimum=1)
        text_size = _integer(prepared["text_byte_size"], f"{path}.prepared.text_byte_size", minimum=1)
        if text_size > 8 * 1024 * 1024:
            _fail(f"{path}.prepared.text_byte_size", "must not exceed 8 MiB")
        if event["event_id"] != f"syllabus-prepared-{digest}" or event["session_id"] != f"session-syllabus-prepared-{digest}":
            _fail(path, "prepared event/session IDs must be derived from source digest")
        predecessor = _nullable_identifier(event["supersedes_source_version_id"], f"{path}.supersedes_source_version_id")
        if role == "supplement" and predecessor is not None:
            _fail(f"{path}.supersedes_source_version_id", "supplements cannot supersede authoritative sources")
        acquisition = _object(event["acquisition"], f"{path}.acquisition")
        _keys(acquisition, {"kind", "policy_id", "trust", "declared_source"}, {"provenance"}, f"{path}.acquisition")
        _identifier(acquisition["kind"], f"{path}.acquisition.kind")
        _identifier(acquisition["policy_id"], f"{path}.acquisition.policy_id")
        if acquisition["trust"] not in {"external_unverified", "store_invoked"}:
            _fail(f"{path}.acquisition.trust", "must be external_unverified or store_invoked")
        _bounded_string(acquisition["declared_source"], f"{path}.acquisition.declared_source")
        if acquisition["trust"] == "store_invoked" and "provenance" not in acquisition:
            _fail(f"{path}.acquisition.provenance", "is required for store-invoked acquisition")
        if "provenance" in acquisition:
            _bounded_json(acquisition["provenance"], f"{path}.acquisition.provenance")
        extraction = _object(event["extraction"], f"{path}.extraction")
        _keys(extraction, {"kind", "policy_id", "producer", "warnings"}, {"provenance"}, f"{path}.extraction")
        _identifier(extraction["kind"], f"{path}.extraction.kind")
        _identifier(extraction["policy_id"], f"{path}.extraction.policy_id")
        _validate_producer(extraction["producer"], f"{path}.extraction.producer", store_invoked=True)
        if extraction["producer"]["trust"] == "store_invoked" and "provenance" not in extraction:
            _fail(f"{path}.extraction.provenance", "is required for store-invoked extraction")
        if "provenance" in extraction:
            _bounded_json(extraction["provenance"], f"{path}.extraction.provenance")
        warnings = _string_list(extraction["warnings"], f"{path}.extraction.warnings")
        if any(len(item.encode("utf-8")) > 4096 for item in warnings):
            _fail(f"{path}.extraction.warnings", "each warning must be at most 4096 UTF-8 bytes")
        if len(warnings) > 32:
            _fail(f"{path}.extraction.warnings", "must contain at most 32 warnings")
    elif kind == "syllabus_assertions_proposed":
        _keys(event, common | {"role", "prepared_event_id", "source_version_id", "prepared_document_sha256", "producer", "proposals", "proposal_set_sha256"}, set(), path)
        _enum(event["role"], SYLLABUS_ROLES, f"{path}.role")
        _identifier(event["prepared_event_id"], f"{path}.prepared_event_id")
        _identifier(event["source_version_id"], f"{path}.source_version_id")
        _sha256(event["prepared_document_sha256"], f"{path}.prepared_document_sha256")
        _validate_producer(event["producer"], f"{path}.producer")
        proposals = _list(event["proposals"], f"{path}.proposals")
        if not 1 <= len(proposals) <= 2048:
            _fail(f"{path}.proposals", "must contain 1 through 2048 proposals")
        ids: set[str] = set()
        for index, proposal in enumerate(proposals):
            _validate_proposal(proposal, f"{path}.proposals[{index}]")
            if proposal["display_order"] != index:
                _fail(f"{path}.proposals[{index}].display_order", "must equal array index")
            if proposal["proposal_id"] in ids:
                _fail(f"{path}.proposals[{index}].proposal_id", "must be unique")
            ids.add(proposal["proposal_id"])
        digest = syllabus_proposal_set_sha256(event)
        if _sha256(event["proposal_set_sha256"], f"{path}.proposal_set_sha256") != digest:
            _fail(f"{path}.proposal_set_sha256", "does not match canonical proposals")
        if event["event_id"] != f"syllabus-proposals-{digest}" or event["session_id"] != f"session-syllabus-proposals-{digest}":
            _fail(path, "proposal event/session IDs must be derived from proposal-set digest")
    else:
        _keys(event, common | {"role", "prepared_event_id", "source_version_id", "prepared_document_sha256", "proposal_event_id", "proposal_set_sha256", "actor", "accepted_proposal_ids", "deferred_proposal_ids", "rejected_proposal_ids", "corrections", "supersedes_decision_event_id", "decision_set_sha256"}, set(), path)
        _enum(event["role"], SYLLABUS_ROLES, f"{path}.role")
        for key in ("prepared_event_id", "source_version_id", "proposal_event_id"):
            _identifier(event[key], f"{path}.{key}")
        _sha256(event["prepared_document_sha256"], f"{path}.prepared_document_sha256")
        _sha256(event["proposal_set_sha256"], f"{path}.proposal_set_sha256")
        actor = _object(event["actor"], f"{path}.actor")
        _keys(actor, {"type", "id"}, set(), f"{path}.actor")
        if actor != {"type": "learner", "id": "learner"}:
            _fail(f"{path}.actor", "must identify the learner")
        partitions = []
        for key in ("accepted_proposal_ids", "deferred_proposal_ids", "rejected_proposal_ids"):
            items = _string_list(event[key], f"{path}.{key}", unique=True)
            if items != sorted(items):
                _fail(f"{path}.{key}", "must be sorted")
            partitions.append(set(items))
        corrections = _list(event["corrections"], f"{path}.corrections")
        if len(corrections) > 2048 or sum(len(items) for items in partitions) + len(corrections) > 2048:
            _fail(path, "decision partitions/corrections must cover at most 2048 proposals")
        targets: set[str] = set()
        correction_ids: list[str] = []
        for index, correction_value in enumerate(corrections):
            correction_path = f"{path}.corrections[{index}]"
            correction = _object(correction_value, correction_path)
            _keys(correction, {"correction_id", "target_proposal_id", "predicate", "semantic_roles", "value_type", "status", "value", "rationale", "origin"}, set(), correction_path)
            correction_ids.append(_identifier(correction["correction_id"], f"{correction_path}.correction_id"))
            target = _identifier(correction["target_proposal_id"], f"{correction_path}.target_proposal_id")
            if target in targets:
                _fail(f"{correction_path}.target_proposal_id", "must be unique")
            targets.add(target)
            _identifier(correction["predicate"], f"{correction_path}.predicate")
            roles = _string_list(correction["semantic_roles"], f"{correction_path}.semantic_roles", unique=True, nonempty=True)
            if any(role not in SYLLABUS_SEMANTIC_ROLES for role in roles):
                _fail(f"{correction_path}.semantic_roles", "contains an unsupported role")
            value_type = _enum(correction["value_type"], SYLLABUS_VALUE_TYPES, f"{correction_path}.value_type")
            if "planning_topic" in roles and value_type != "text":
                _fail(correction_path, "planning_topic corrections must use text values")
            status = _enum(correction["status"], {"specified", "explicitly_unknown"}, f"{correction_path}.status")
            _validate_typed_value(correction["value"], value_type, f"{correction_path}.value", allow_null=status == "explicitly_unknown")
            if status == "explicitly_unknown" and correction["value"] is not None:
                _fail(f"{correction_path}.value", "must be null when explicitly_unknown")
            _bounded_string(correction["rationale"], f"{correction_path}.rationale")
            if correction["origin"] != "learner_correction":
                _fail(f"{correction_path}.origin", "must be learner_correction")
        if correction_ids != sorted(correction_ids) or len(correction_ids) != len(set(correction_ids)):
            _fail(f"{path}.corrections", "must be uniquely sorted by correction_id")
        partitions.append(targets)
        if any(partitions[i] & partitions[j] for i in range(4) for j in range(i + 1, 4)):
            _fail(path, "accepted, deferred, rejected, and corrected IDs must be disjoint")
        _nullable_identifier(event["supersedes_decision_event_id"], f"{path}.supersedes_decision_event_id")
        digest = syllabus_decision_set_sha256(event)
        if _sha256(event["decision_set_sha256"], f"{path}.decision_set_sha256") != digest:
            _fail(f"{path}.decision_set_sha256", "does not match canonical decision")
        if event["event_id"] != f"syllabus-decision-{digest}" or event["session_id"] != f"session-syllabus-decision-{digest}":
            _fail(path, "decision event/session IDs must be derived from decision-set digest")
    if len(canonical_json(event, newline=False)) > 2 * 1024 * 1024:
        _fail(path, "syllabus lifecycle event exceeds 2 MiB canonical limit")


def validate_event(value: Any, path: str = "event") -> dict[str, Any]:
    """Validate one event of any supported kind against its per-kind field contract."""
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
            "operation",
            "task_id",
            "subject",
            "context_id",
            "novelty",
            "evidence_mode",
            "outcome",
            "rubric_id",
            "assistance",
        }
        optional = {
            "score",
            "max_score",
            "confidence_before",
            "retrieval",
            "response_time_ms",
            "grounding",
        }
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
        if "grounding" in event:
            _validate_grounding_reference(event["grounding"], f"{path}.grounding")

    elif kind == "model_revision":
        required = common | {"triggering_prediction_event_ids", "model", "decision", "rationale"}
        optional = {
            "prior_model_revision_event_id",
            "word_count_before",
            "word_count_after",
            "initial_model",
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
        _enum(
            event["decision"],
            {"exploit", "revise", "expand", "fallback-independent"},
            f"{path}.decision",
        )
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
        _string_list(
            event["assessment_event_ids"],
            f"{path}.assessment_event_ids",
            unique=True,
            nonempty=True,
        )
        _string(event["decision"], f"{path}.decision")

    elif kind == "session_completed":
        required = common | {
            "next_action",
            "next_review_date",
            "evidence_event_ids",
            "receipt_schema_version",
        }
        _keys(event, required, {"grounding"}, path)
        _string(event["next_action"], f"{path}.next_action")
        _date(event["next_review_date"], f"{path}.next_review_date", allow_null=True)
        _string_list(event["evidence_event_ids"], f"{path}.evidence_event_ids", unique=True)
        if _integer(event["receipt_schema_version"], f"{path}.receipt_schema_version") != 1:
            _fail(f"{path}.receipt_schema_version", "unsupported version; expected 1")
        if "grounding" in event:
            _validate_grounding_reference(event["grounding"], f"{path}.grounding")

    elif kind == "domain_reset":
        _keys(event, common, {"reason"}, path)
        if "reason" in event:
            _string(event["reason"], f"{path}.reason")

    elif kind == "exam_cycle_closed":
        _keys(event, common | {"archived_exam"}, {"self_reported_outcome"}, path)
        _object(event["archived_exam"], f"{path}.archived_exam")
        if "self_reported_outcome" in event:
            _string(event["self_reported_outcome"], f"{path}.self_reported_outcome")

    elif kind in {"syllabus_source_prepared", "syllabus_assertions_proposed", "syllabus_decision_recorded"}:
        _validate_new_syllabus_event(event, kind, common, path)

    elif kind == "syllabus_source_ingested":
        _keys(
            event,
            common | {"source", "extraction", "pages", "assertions", "assertion_set_sha256"},
            set(),
            path,
        )
        source = _object(event["source"], f"{path}.source")
        _keys(
            source,
            {
                "source_id",
                "source_version_id",
                "original_filename",
                "media_type",
                "byte_size",
                "page_count",
                "sha256",
                "content_retention",
                "supersedes_source_version_id",
            },
            set(),
            f"{path}.source",
        )
        _identifier(source["source_id"], f"{path}.source.source_id")
        digest = _sha256(source["sha256"], f"{path}.source.sha256")
        if event["event_id"] != f"syllabus-source-{digest}":
            _fail(f"{path}.event_id", "must be deterministically derived from the source digest")
        if event["session_id"] != f"syllabus-intake-{digest}":
            _fail(f"{path}.session_id", "must be deterministically derived from the source digest")
        version_id = _identifier(source["source_version_id"], f"{path}.source.source_version_id")
        if version_id != f"sha256-{digest}":
            _fail(f"{path}.source.source_version_id", "must be sha256-<source digest>")
        filename = _string(source["original_filename"], f"{path}.source.original_filename")
        if Path(filename).name != filename or filename in {".", ".."}:
            _fail(f"{path}.source.original_filename", "must be a filename, not a path")
        _string(source["media_type"], f"{path}.source.media_type")
        _integer(source["byte_size"], f"{path}.source.byte_size", minimum=1)
        page_count = _integer(source["page_count"], f"{path}.source.page_count", minimum=1)
        if source["content_retention"] != "extracted_text_only":
            _fail(f"{path}.source.content_retention", "must be extracted_text_only")
        _nullable_identifier(
            source["supersedes_source_version_id"],
            f"{path}.source.supersedes_source_version_id",
        )

        extraction = _object(event["extraction"], f"{path}.extraction")
        _keys(
            extraction,
            {
                "adapter_id",
                "adapter_version",
                "method",
                "extractor_name",
                "extractor_version",
                "extracted_at",
                "warnings",
                "diagnostics",
            },
            set(),
            f"{path}.extraction",
        )
        _identifier(extraction["adapter_id"], f"{path}.extraction.adapter_id")
        if _integer(extraction["adapter_version"], f"{path}.extraction.adapter_version") != 1:
            _fail(f"{path}.extraction.adapter_version", "must be 1")
        _identifier(extraction["method"], f"{path}.extraction.method")
        _string(extraction["extractor_name"], f"{path}.extraction.extractor_name")
        _string(extraction["extractor_version"], f"{path}.extraction.extractor_version")
        _timestamp(extraction["extracted_at"], f"{path}.extraction.extracted_at")
        _string_list(extraction["warnings"], f"{path}.extraction.warnings")
        _string_list(extraction["diagnostics"], f"{path}.extraction.diagnostics")

        pages_value = _list(event["pages"], f"{path}.pages")
        if len(pages_value) != page_count:
            _fail(f"{path}.pages", "length must equal source.page_count")
        pages: dict[int, str] = {}
        for index, page_value in enumerate(pages_value):
            page_path = f"{path}.pages[{index}]"
            page = _object(page_value, page_path)
            _keys(page, {"page_number", "text", "text_sha256"}, set(), page_path)
            number = _integer(page["page_number"], f"{page_path}.page_number", minimum=1)
            if number != index + 1:
                _fail(f"{page_path}.page_number", "pages must be contiguous and one-based")
            text = _string(page["text"], f"{page_path}.text", allow_empty=True)
            if "\r" in text or unicodedata.normalize("NFC", text) != text:
                _fail(f"{page_path}.text", "must be NFC UTF-8 text with LF line endings")
            if _sha256(page["text_sha256"], f"{page_path}.text_sha256") != sha256_bytes(
                text.encode("utf-8")
            ):
                _fail(f"{page_path}.text_sha256", "does not match page text")
            pages[number] = text
        assertions = _validate_syllabus_assertions(event["assertions"], pages, f"{path}.assertions")
        expected_assertion_hash = syllabus_assertion_set_sha256(assertions)
        if (
            _sha256(event["assertion_set_sha256"], f"{path}.assertion_set_sha256")
            != expected_assertion_hash
        ):
            _fail(f"{path}.assertion_set_sha256", "does not match canonical assertions")

    elif kind == "syllabus_approval_recorded":
        if not event["session_id"].startswith("syllabus-approval-"):
            _fail(
                f"{path}.session_id",
                "must use the reserved syllabus-approval- administrative prefix",
            )
        _keys(
            event,
            common
            | {
                "source_version_id",
                "source_assertion_set_sha256",
                "actor",
                "accepted_assertion_ids",
                "deferred_assertion_ids",
                "corrections",
                "supersedes_approval_event_id",
                "approval_set_sha256",
            },
            set(),
            path,
        )
        _identifier(event["source_version_id"], f"{path}.source_version_id")
        _sha256(event["source_assertion_set_sha256"], f"{path}.source_assertion_set_sha256")
        actor = _object(event["actor"], f"{path}.actor")
        _keys(actor, {"type", "id"}, set(), f"{path}.actor")
        if actor["type"] != "learner":
            _fail(f"{path}.actor.type", "must be learner")
        if actor["id"] != "learner":
            _fail(f"{path}.actor.id", "must be learner")
        accepted = _string_list(
            event["accepted_assertion_ids"],
            f"{path}.accepted_assertion_ids",
            unique=True,
        )
        deferred = _string_list(
            event["deferred_assertion_ids"],
            f"{path}.deferred_assertion_ids",
            unique=True,
        )
        if accepted != sorted(accepted):
            _fail(f"{path}.accepted_assertion_ids", "must be sorted for canonical encoding")
        if deferred != sorted(deferred):
            _fail(f"{path}.deferred_assertion_ids", "must be sorted for canonical encoding")
        corrections_value = _list(event["corrections"], f"{path}.corrections")
        correction_targets: list[str] = []
        correction_ids: set[str] = set()
        for index, correction_value in enumerate(corrections_value):
            correction_path = f"{path}.corrections[{index}]"
            correction = _object(correction_value, correction_path)
            _keys(
                correction,
                {
                    "correction_assertion_id",
                    "target_assertion_id",
                    "field",
                    "status",
                    "normalized_value",
                    "rationale",
                    "origin",
                },
                set(),
                correction_path,
            )
            correction_id = _identifier(
                correction["correction_assertion_id"],
                f"{correction_path}.correction_assertion_id",
            )
            if correction_id in correction_ids:
                _fail(
                    f"{correction_path}.correction_assertion_id",
                    "must be unique within the approval",
                )
            correction_ids.add(correction_id)
            target = _identifier(
                correction["target_assertion_id"], f"{correction_path}.target_assertion_id"
            )
            correction_targets.append(target)
            field = _identifier(correction["field"], f"{correction_path}.field")
            status = _enum(
                correction["status"],
                {"specified", "not_specified", "unresolved"},
                f"{correction_path}.status",
            )
            _validate_syllabus_normalized_value(
                correction["normalized_value"],
                field,
                status,
                f"{correction_path}.normalized_value",
            )
            _bounded_string(correction["rationale"], f"{correction_path}.rationale")
            if correction["origin"] != "learner_correction":
                _fail(f"{correction_path}.origin", "must be learner_correction")
        if len(set(correction_targets)) != len(correction_targets):
            _fail(f"{path}.corrections", "target assertions must be unique")
        ordered_correction_ids = [item["correction_assertion_id"] for item in corrections_value]
        if ordered_correction_ids != sorted(ordered_correction_ids):
            _fail(f"{path}.corrections", "must be sorted by correction_assertion_id")
        decisions = [set(accepted), set(deferred), set(correction_targets)]
        if any(decisions[i] & decisions[j] for i in range(3) for j in range(i + 1, 3)):
            _fail(path, "accepted, deferred, and corrected assertion IDs must be disjoint")
        _nullable_identifier(
            event["supersedes_approval_event_id"], f"{path}.supersedes_approval_event_id"
        )
        expected_approval_hash = syllabus_approval_set_sha256(event)
        if (
            _sha256(event["approval_set_sha256"], f"{path}.approval_set_sha256")
            != expected_approval_hash
        ):
            _fail(f"{path}.approval_set_sha256", "does not match canonical disposition")

    else:  # legacy_snapshot_imported
        _keys(event, common | {"source_sha256", "claims", "evidence_eligible"}, set(), path)
        _sha256(event["source_sha256"], f"{path}.source_sha256")
        _object(event["claims"], f"{path}.claims")
        if _boolean(event["evidence_eligible"], f"{path}.evidence_eligible"):
            _fail(f"{path}.evidence_eligible", "must be false")

    canonical_json(event)
    return event


def validate_commit_request(value: Any) -> dict[str, Any]:
    """Validate a commit request's events and profile patch."""
    request = _object(value, "request")
    _keys(request, set(), {"events", "profile_patch"}, "request")
    if not request:
        _fail("request", "must contain events and/or profile_patch")
    events = request.get("events", [])
    _list(events, "request.events")
    for index, event in enumerate(events):
        validate_event(event, f"request.events[{index}]")
        grounding = event.get("grounding") if isinstance(event, dict) else None
        if isinstance(grounding, dict) and "approval_event_id" in grounding:
            _fail(f"request.events[{index}].grounding", "new writes must use decision_event_id")
    if "profile_patch" in request:
        validate_profile_patch(request["profile_patch"])
    return request


def parse_json_file(path: Path, label: str = "JSON") -> Any:
    """Read a UTF-8 JSON file, reporting the failing line and column on error."""
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
        raise ValidationError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid {label}: {exc.msg}"
        ) from exc


def parse_events_bytes(data: bytes, source: str = "events.jsonl") -> list[dict[str, Any]]:
    """Parse and validate an append-only event log, reporting byte offsets on error."""
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
                f"{source}: line {line_number}, "
                f"byte offset {line_offset + exc.start}: invalid UTF-8"
            ) from exc
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            byte_column = len(text[: exc.pos].encode("utf-8"))
            raise ValidationError(
                f"{source}: line {line_number}, "
                f"byte offset {line_offset + byte_column}: "
                f"malformed or truncated JSON ({exc.msg})"
            ) from exc
        events.append(validate_event(value, f"{source} line {line_number}"))
    if data and not data.endswith((b"\n", b"\r")):
        raise ValidationError(
            f"{source}: line {len(data.splitlines())}, byte offset {len(data)}: "
            "truncated JSONL (missing newline terminator)"
        )
    return events


def encode_event_lines(events: Iterable[dict[str, Any]]) -> bytes:
    """Encode events as canonical JSONL, revalidating each one."""
    return b"".join(canonical_json(validate_event(event)) for event in events)
