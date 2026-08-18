"""Narrow, offline importer for a manually exported legacy Knowledge State block."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from .schema import ValidationError

START_MARKER = "<!-- KS:start -->"
END_MARKER = "<!-- KS:end -->"


def _section(block: str, heading: str) -> str:
    pattern = re.compile(rf"(?ms)^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)")
    match = pattern.search(block)
    return match.group(1).strip() if match else ""


def _split_table_line(line: str) -> list[str]:
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for character in line[1:-1]:
        if escaped:
            if character != "|":
                current.append("\\")
            current.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if escaped:
        current.append("\\")
    cells.append("".join(current).strip())
    return cells


def _table_rows(section: str, expected_columns: int) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in section.splitlines():
        line = raw.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = _split_table_line(line)
        if not cells or all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        if cells[0].casefold() in {"concept", "chain", "factor"}:
            continue
        if len(cells) < expected_columns:
            continue
        rows.append(cells[:expected_columns])
    return rows


def _syllabus(section: str) -> tuple[str | None, list[dict[str, Any]]]:
    goal: str | None = None
    topics: list[dict[str, Any]] = []
    for raw in section.splitlines():
        line = raw.strip()
        if line.casefold().startswith("goal:"):
            goal = line.split(":", 1)[1].strip() or None
            continue
        match = re.match(r"^-\s*\[([ xX])\]\s*(.+?)\s*$", line)
        if match:
            topics.append(
                {"completed_claim": match.group(1).casefold() == "x", "topic": match.group(2)}
            )
    return goal, topics


def _plain_items(section: str) -> list[str]:
    items: list[str] = []
    for raw in section.splitlines():
        value = re.sub(r"^[-*]\s+", "", raw.strip())
        if value and not value.startswith("|"):
            items.append(value)
    return items


def parse_legacy_ks(data: bytes) -> tuple[str, dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError("legacy KS input must be UTF-8") from exc
    if text.count(START_MARKER) != 1 or text.count(END_MARKER) != 1:
        raise ValidationError(
            "legacy KS input must contain exactly one KS:start and one KS:end marker"
        )
    start = text.index(START_MARKER)
    end = text.index(END_MARKER, start)
    if end <= start:
        raise ValidationError("legacy KS markers are out of order")
    block = text[start + len(START_MARKER) : end]
    if not re.search(r"(?m)^# Knowledge State\s*$", block):
        raise ValidationError("legacy KS block is missing '# Knowledge State'")

    goal, syllabus = _syllabus(_section(block, "Syllabus"))
    concepts = [
        {
            "concept": row[0],
            "status_claim": row[1],
            "syllabus_topic": row[2],
            "evidence_claim": row[3],
            "last_tested_claim": row[4],
        }
        for row in _table_rows(_section(block, "Concepts"), 5)
    ]
    chains = [
        {
            "chain": row[0],
            "status_claim": row[1],
            "evidence_claim": row[2],
            "last_tested_claim": row[3],
        }
        for row in _table_rows(_section(block, "Chains"), 4)
    ]
    factors = [
        {
            "factor": row[0],
            "status_claim": row[1],
            "evidence_claim": row[2],
            "last_tested_claim": row[3],
        }
        for row in _table_rows(_section(block, "Factors"), 4)
    ]
    claims: dict[str, Any] = {
        "chains": chains,
        "compressed_model": _section(block, "Compressed Model"),
        "concepts": concepts,
        "goal": goal,
        "open_questions": _plain_items(_section(block, "Open Questions")),
        "syllabus": syllabus,
        "factors": factors,
    }
    source_hash = hashlib.sha256(data).hexdigest()
    return source_hash, claims


def legacy_event(data: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    source_hash, claims = parse_legacy_ks(data)
    event = {
        "claims": claims,
        "evidence_eligible": False,
        "event_id": f"legacy-{source_hash[:24]}",
        "kind": "legacy_snapshot_imported",
        "occurred_at": "1970-01-01T00:00:00Z",
        "schema_version": 1,
        "session_id": f"legacy-{source_hash[:16]}",
        "source_sha256": source_hash,
    }
    return event, claims
