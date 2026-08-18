"""Digest-bound intake adapter for the audited ST5201X 2026 syllabus fixture."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from copy import deepcopy
from pathlib import Path
from typing import Any

from .schema import (
    SCHEMA_VERSION,
    SYLLABUS_ADAPTER_ID,
    ValidationError,
    canonical_json,
    sha256_bytes,
    syllabus_assertion_set_sha256,
    validate_event,
)

MANIFEST_PATH = Path(__file__).with_name("data") / "st5201x-2026-v1.json"
EXPECTED_SHA256 = "53909df562e2658ab3e1327eb8c33120fa12b37489178dc87bb4d632e4f15376"
EXPECTED_BYTE_SIZE = 45_185


class SyllabusUnavailableError(ValidationError):
    """The declared document path cannot be used as an intake source."""


class SyllabusUnreadableError(ValidationError):
    """The declared regular file could not be read completely."""


class SyllabusDigestMismatchError(ValidationError):
    """The supplied bytes are not the exact verified ST5201X fixture."""


class SyllabusUnsupportedError(ValidationError):
    """The requested adapter or media type is not supported."""


class SyllabusAdapterError(ValidationError):
    """The checked-in verified snapshot is internally unavailable or inconsistent."""


def _read_regular_document(path: Path) -> bytes:
    """Verify one descriptor while retaining at most the supported fixture size."""
    display = str(path)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError as exc:
        raise SyllabusUnavailableError(
            f"syllabus document unavailable: missing path {display}"
        ) from exc
    except OSError as exc:
        raise SyllabusUnavailableError(
            f"syllabus document unavailable: cannot inspect {display}: {exc}"
        ) from exc
    if stat.S_ISLNK(mode):
        raise SyllabusUnavailableError(
            f"syllabus document unavailable: symlinks are not accepted: {display}"
        )
    if not stat.S_ISREG(mode):
        raise SyllabusUnavailableError(
            f"syllabus document unavailable: not a regular file: {display}"
        )

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SyllabusUnreadableError(f"syllabus document unreadable: {display}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SyllabusUnavailableError(
                f"syllabus document unavailable: not a regular file: {display}"
            )
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        actual_size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            actual_size += len(chunk)
            digest.update(chunk)
            if actual_size <= EXPECTED_BYTE_SIZE:
                chunks.append(chunk)
            else:
                chunks.clear()
        actual_digest = digest.hexdigest()
        if actual_digest != EXPECTED_SHA256 or actual_size != EXPECTED_BYTE_SIZE:
            raise SyllabusDigestMismatchError(
                "syllabus digest mismatch: "
                f"expected sha256={EXPECTED_SHA256} size={EXPECTED_BYTE_SIZE}; "
                f"actual sha256={actual_digest} size={actual_size}"
            )
        return b"".join(chunks)
    except OSError as exc:
        raise SyllabusUnreadableError(f"syllabus document unreadable: {display}: {exc}") from exc
    finally:
        os.close(descriptor)


def _load_manifest() -> dict[str, Any]:
    try:
        value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyllabusAdapterError(
            f"verified ST5201X manifest is unavailable or invalid: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise SyllabusAdapterError("verified ST5201X manifest must be a JSON object")
    expected_keys = {
        "manifest_schema_version",
        "adapter_id",
        "adapter_version",
        "source",
        "extraction",
        "pages",
        "assertions",
        "assertion_set_sha256",
    }
    if set(value) != expected_keys:
        raise SyllabusAdapterError("verified ST5201X manifest has an unexpected field set")
    if value["manifest_schema_version"] != 1:
        raise SyllabusAdapterError("verified ST5201X manifest schema must be 1")
    if value["adapter_id"] != SYLLABUS_ADAPTER_ID or value["adapter_version"] != 1:
        raise SyllabusAdapterError("verified ST5201X manifest adapter identity is invalid")
    source = value.get("source")
    if not isinstance(source, dict):
        raise SyllabusAdapterError("verified ST5201X manifest source must be an object")
    if source.get("sha256") != EXPECTED_SHA256 or source.get("byte_size") != EXPECTED_BYTE_SIZE:
        raise SyllabusAdapterError("verified ST5201X manifest source fingerprint is invalid")
    if value.get("assertion_set_sha256") != syllabus_assertion_set_sha256(value.get("assertions")):
        raise SyllabusAdapterError("verified ST5201X manifest assertion hash is invalid")
    canonical_json(value)
    return value


def build_ingestion_event(
    document_path: Path,
    *,
    original_filename: str,
    media_type: str,
    adapter_id: str,
    occurred_at: str,
    supersedes_source_version_id: str | None = None,
) -> dict[str, Any]:
    """Verify exact bytes and emit the preverified, fixture-specific intake snapshot."""
    if adapter_id != SYLLABUS_ADAPTER_ID:
        raise SyllabusUnsupportedError(
            f"unsupported syllabus adapter {adapter_id!r}; expected {SYLLABUS_ADAPTER_ID!r}"
        )
    if media_type != "application/pdf":
        raise SyllabusUnsupportedError(
            f"unsupported syllabus media type {media_type!r}; expected 'application/pdf'"
        )
    document_bytes = _read_regular_document(document_path)
    actual_digest = sha256_bytes(document_bytes)
    if not document_bytes.startswith(b"%PDF-"):
        raise SyllabusAdapterError("verified syllabus bytes do not declare PDF media")

    manifest = _load_manifest()
    source = deepcopy(manifest["source"])
    source["original_filename"] = original_filename
    source["supersedes_source_version_id"] = supersedes_source_version_id
    event = {
        "assertion_set_sha256": manifest["assertion_set_sha256"],
        "assertions": deepcopy(manifest["assertions"]),
        "event_id": f"syllabus-source-{actual_digest}",
        "extraction": deepcopy(manifest["extraction"]),
        "kind": "syllabus_source_ingested",
        "occurred_at": occurred_at,
        "pages": deepcopy(manifest["pages"]),
        "schema_version": SCHEMA_VERSION,
        "session_id": f"syllabus-intake-{actual_digest}",
        "source": source,
    }
    return validate_event(event)
