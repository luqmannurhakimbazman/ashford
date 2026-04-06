#!/usr/bin/env python3
"""Deterministic KS merge script.

Takes a typed JSON payload and a raw KS block (markdown between markers),
applies updates deterministically, and outputs the merged KS block to stdout.

Usage:
    python3 ks-merge.py [--dry-run] [--lenient] <payload_path> <ks_block_path>

Exit codes:
    0 — success (merged block on stdout)
    1 — failure (error on stderr)
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass

KS_START = "<!-- KS:start -->"
KS_END = "<!-- KS:end -->"
ESCAPED_KS_START = r"\<!-- KS:start --\>"
ESCAPED_KS_END = r"\<!-- KS:end --\>"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Table column schemas — maps table name to expected column headers
TABLE_SCHEMAS = {
    "concepts": ["Concept", "Status", "Syllabus Topic", "Evidence", "Last Tested"],
    "chains": ["Chain", "Status", "Evidence", "Last Tested"],
    "factors": ["Factor", "Status", "Evidence", "Last Tested"],
}

TABLE_SECTION_MAP = {
    "concepts": "Concepts",
    "chains": "Chains",
    "factors": "Factors",
}

WEAKNESS_COLUMNS = ["Priority", "Item", "Type", "Phase", "Severity", "Source", "Added"]
WEAKNESS_KEY_MAP = {
    "priority": "Priority",
    "item": "Item",
    "type": "Type",
    "phase": "Phase",
    "severity": "Severity",
    "source": "Source",
    "added": "Added",
}

SECTION_REWRITE_MAP = {
    "compressed_model": "Compressed Model",
    "open_questions": "Open Questions",
    "interleave_pool": "Interleave Pool",
}

SUBSECTION_REWRITE_MAP = {
    "calibration_trend": "Calibration Trend",
}

SECTION_APPEND_MAP = {
    "calibration_concept": "Concept-Level Confidence",
    "calibration_gate": "Gate Predictions",
    "load_session_history": "Session History",
}

LOAD_BASELINE_MAP = {
    "working_batch_size": "Observed working batch size",
    "hint_tolerance": "Hint tolerance",
    "recovery_pattern": "Recovery pattern",
}

ENGAGEMENT_MAP = {
    "momentum": "Momentum",
    "consecutive_struggles": "Consecutive struggles",
    "last_celebration": "Last celebration",
    "notes": "Notes",
}

MASTER_STATUS_VALUES = {"not-mastered", "partial", "mastered"}
WEAKNESS_TYPE_VALUES = {"concept", "chain", "factor"}
PHASE_VALUES = {"Dot", "Linear", "Network"}
SYLLABUS_STATUS_VALUES = {"checked", "unchecked"}
ENGAGEMENT_MOMENTUM_VALUES = {"positive", "neutral", "negative", "fragile"}

ALLOWED_TOP_LEVEL_KEYS = {
    "mastery_updates",
    "weakness_queue",
    "syllabus_updates",
    "section_rewrites",
    "subsection_rewrites",
    "section_appends",
    "load_baseline",
    "engagement",
}

MASTERY_ALLOWED_KEYS = {"table", "name", "status", "evidence", "last_tested", "syllabus_topic"}
WEAKNESS_ALLOWED_KEYS = {"priority", "item", "type", "phase", "severity", "source", "added"}
SYLLABUS_ALLOWED_KEYS = {"topic", "status"}
LOAD_BASELINE_ALLOWED_KEYS = set(LOAD_BASELINE_MAP.keys())
ENGAGEMENT_ALLOWED_KEYS = set(ENGAGEMENT_MAP.keys())


class MergeError(Exception):
    """Raised when payload or KS structure is invalid for deterministic merge."""


def fail(msg: str) -> None:
    """Raise a deterministic merge failure."""
    raise MergeError(msg)


def warn_or_fail(msg: str, strict: bool) -> None:
    """Warn in lenient mode; fail in strict mode."""
    if strict:
        fail(msg)
    print(f"Warning: {msg}", file=sys.stderr)


def validate_date(value: str, field_path: str) -> None:
    """Validate ISO date YYYY-MM-DD."""
    if not isinstance(value, str) or not DATE_RE.match(value):
        fail(f"{field_path} must be YYYY-MM-DD")


def canonicalize_markers(ks_block: str) -> str:
    """Normalize escaped KS markers from Notion read-back to canonical marker text."""
    return ks_block.replace(ESCAPED_KS_START, KS_START).replace(ESCAPED_KS_END, KS_END)


def validate_marker_counts(ks_block: str) -> None:
    """Validate marker balance and duplicate marker corruption."""
    start_count = ks_block.count(KS_START)
    end_count = ks_block.count(KS_END)
    if start_count != end_count:
        fail(f"Unbalanced KS markers: start={start_count}, end={end_count}")
    if start_count > 1:
        fail("KS block contains duplicate marker pairs; refusing to merge")


def validate_mastery_updates(updates: object) -> None:
    """Validate mastery_updates list schema."""
    if not isinstance(updates, list):
        fail("mastery_updates must be a list")

    for i, update in enumerate(updates):
        path = f"mastery_updates[{i}]"
        if not isinstance(update, dict):
            fail(f"{path} must be an object")
        unknown = set(update.keys()) - MASTERY_ALLOWED_KEYS
        if unknown:
            fail(f"{path} has unknown keys: {', '.join(sorted(unknown))}")

        if "table" not in update:
            fail(f"{path}.table is required")
        if update["table"] not in TABLE_SECTION_MAP:
            fail(f"{path}.table must be one of: {', '.join(TABLE_SECTION_MAP.keys())}")

        if "name" not in update:
            fail(f"{path}.name is required")
        if not isinstance(update["name"], str) or not update["name"].strip():
            fail(f"{path}.name must be a non-empty string")

        if "status" in update and update["status"] not in MASTER_STATUS_VALUES:
            fail(f"{path}.status must be one of: {', '.join(sorted(MASTER_STATUS_VALUES))}")

        if "evidence" in update and not isinstance(update["evidence"], str):
            fail(f"{path}.evidence must be a string")

        if "last_tested" in update:
            validate_date(update["last_tested"], f"{path}.last_tested")

        if "syllabus_topic" in update:
            if update["table"] != "concepts":
                fail(f"{path}.syllabus_topic is only valid for table='concepts'")
            if not isinstance(update["syllabus_topic"], str):
                fail(f"{path}.syllabus_topic must be a string")


def validate_weakness_queue(queue: object) -> None:
    """Validate weakness_queue full-rewrite schema."""
    if not isinstance(queue, list):
        fail("weakness_queue must be a list")

    for i, entry in enumerate(queue):
        path = f"weakness_queue[{i}]"
        if not isinstance(entry, dict):
            fail(f"{path} must be an object")
        unknown = set(entry.keys()) - WEAKNESS_ALLOWED_KEYS
        if unknown:
            fail(f"{path} has unknown keys: {', '.join(sorted(unknown))}")
        missing = WEAKNESS_ALLOWED_KEYS - set(entry.keys())
        if missing:
            fail(f"{path} missing required keys: {', '.join(sorted(missing))}")

        if not isinstance(entry["priority"], int) or entry["priority"] < 1:
            fail(f"{path}.priority must be an integer >= 1")
        if entry["type"] not in WEAKNESS_TYPE_VALUES:
            fail(f"{path}.type must be one of: {', '.join(sorted(WEAKNESS_TYPE_VALUES))}")
        if entry["phase"] not in PHASE_VALUES:
            fail(f"{path}.phase must be one of: {', '.join(sorted(PHASE_VALUES))}")
        for key in ("item", "severity", "source"):
            if not isinstance(entry[key], str) or not entry[key].strip():
                fail(f"{path}.{key} must be a non-empty string")
        validate_date(entry["added"], f"{path}.added")


def validate_syllabus_updates(updates: object) -> None:
    """Validate syllabus_updates schema."""
    if not isinstance(updates, list):
        fail("syllabus_updates must be a list")

    for i, update in enumerate(updates):
        path = f"syllabus_updates[{i}]"
        if not isinstance(update, dict):
            fail(f"{path} must be an object")
        unknown = set(update.keys()) - SYLLABUS_ALLOWED_KEYS
        if unknown:
            fail(f"{path} has unknown keys: {', '.join(sorted(unknown))}")
        missing = SYLLABUS_ALLOWED_KEYS - set(update.keys())
        if missing:
            fail(f"{path} missing required keys: {', '.join(sorted(missing))}")

        if not isinstance(update["topic"], str) or not update["topic"].strip():
            fail(f"{path}.topic must be a non-empty string")
        if update["status"] not in SYLLABUS_STATUS_VALUES:
            fail(f"{path}.status must be one of: {', '.join(sorted(SYLLABUS_STATUS_VALUES))}")


def validate_rewrite_map(field_name: str, value: object, allowed_keys: set) -> None:
    """Validate section rewrite maps."""
    if not isinstance(value, dict):
        fail(f"{field_name} must be an object")
    unknown = set(value.keys()) - allowed_keys
    if unknown:
        fail(f"{field_name} has unknown keys: {', '.join(sorted(unknown))}")
    for key, content in value.items():
        if not isinstance(content, str):
            fail(f"{field_name}.{key} must be a string")


def validate_section_appends(appends: object) -> None:
    """Validate section_appends map."""
    if not isinstance(appends, dict):
        fail("section_appends must be an object")
    unknown = set(appends.keys()) - set(SECTION_APPEND_MAP.keys())
    if unknown:
        fail(f"section_appends has unknown keys: {', '.join(sorted(unknown))}")

    for key, row_text in appends.items():
        if not isinstance(row_text, str):
            fail(f"section_appends.{key} must be a string")
        if "\n" in row_text:
            fail(f"section_appends.{key} must be a single table row")
        stripped = row_text.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            fail(f"section_appends.{key} must be a pipe-delimited row")


def validate_load_baseline(load_baseline: object) -> None:
    """Validate load_baseline map."""
    if not isinstance(load_baseline, dict):
        fail("load_baseline must be an object")
    unknown = set(load_baseline.keys()) - LOAD_BASELINE_ALLOWED_KEYS
    if unknown:
        fail(f"load_baseline has unknown keys: {', '.join(sorted(unknown))}")
    for key, value in load_baseline.items():
        if key == "working_batch_size":
            if not isinstance(value, int) or value < 1:
                fail("load_baseline.working_batch_size must be an integer >= 1")
        else:
            if not isinstance(value, str):
                fail(f"load_baseline.{key} must be a string")


def validate_engagement(engagement: object) -> None:
    """Validate engagement map."""
    if not isinstance(engagement, dict):
        fail("engagement must be an object")
    unknown = set(engagement.keys()) - ENGAGEMENT_ALLOWED_KEYS
    if unknown:
        fail(f"engagement has unknown keys: {', '.join(sorted(unknown))}")

    if "momentum" in engagement and engagement["momentum"] not in ENGAGEMENT_MOMENTUM_VALUES:
        fail(f"engagement.momentum must be one of: {', '.join(sorted(ENGAGEMENT_MOMENTUM_VALUES))}")
    if "consecutive_struggles" in engagement:
        value = engagement["consecutive_struggles"]
        if not isinstance(value, int) or value < 0:
            fail("engagement.consecutive_struggles must be an integer >= 0")
    for key in ("last_celebration", "notes"):
        if key in engagement and not isinstance(engagement[key], str):
            fail(f"engagement.{key} must be a string")


def validate_payload_schema(payload: dict, strict: bool) -> None:
    """Validate merge payload schema and semantic constraints."""
    unknown_top = set(payload.keys()) - ALLOWED_TOP_LEVEL_KEYS
    if unknown_top:
        warn_or_fail(f"unknown top-level payload keys: {', '.join(sorted(unknown_top))}", strict)

    if "mastery_updates" in payload:
        validate_mastery_updates(payload["mastery_updates"])
    if "weakness_queue" in payload:
        validate_weakness_queue(payload["weakness_queue"])
    if "syllabus_updates" in payload:
        validate_syllabus_updates(payload["syllabus_updates"])
    if "section_rewrites" in payload:
        validate_rewrite_map("section_rewrites", payload["section_rewrites"], set(SECTION_REWRITE_MAP.keys()))
    if "subsection_rewrites" in payload:
        validate_rewrite_map(
            "subsection_rewrites", payload["subsection_rewrites"], set(SUBSECTION_REWRITE_MAP.keys())
        )
    if "section_appends" in payload:
        validate_section_appends(payload["section_appends"])
    if "load_baseline" in payload:
        validate_load_baseline(payload["load_baseline"])
    if "engagement" in payload:
        validate_engagement(payload["engagement"])


@dataclass
class Section:
    """A parsed section of the KS block."""

    header: str  # The full header line (e.g., "## Concepts")
    level: int  # Header level (2 for ##, 3 for ###)
    body: str  # Everything after the header until the next section


def parse_args() -> tuple:
    """Parse CLI arguments. Returns (dry_run, strict, payload_path, ks_block_path)."""
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    if dry_run:
        args.remove("--dry-run")
    strict = True
    if "--lenient" in args:
        strict = False
        args.remove("--lenient")

    if len(args) != 2:
        print(
            "Usage: ks-merge.py [--dry-run] [--lenient] <payload_path> <ks_block_path>",
            file=sys.stderr,
        )
        sys.exit(1)
    return dry_run, strict, args[0], args[1]


def load_payload(path: str) -> dict:
    """Load and validate the JSON payload."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        fail(f"Failed to parse payload: {e}")
    if not isinstance(data, dict):
        fail("Payload must be a JSON object")
    return data


def load_ks_block(path: str) -> str:
    """Load the raw KS block from file."""
    try:
        with open(path) as f:
            raw = f.read()
    except OSError as e:
        fail(f"Failed to read KS block: {e}")

    canonical = canonicalize_markers(raw)
    validate_marker_counts(canonical)
    return canonical


def parse_sections(ks_block: str) -> tuple:
    """Parse KS block into preamble, sections list, and postamble.

    Returns:
        (preamble, sections, postamble) where:
        - preamble: text before the first ## header (includes <!-- KS:start --> and # Knowledge
          State)
        - sections: list of Section objects
        - postamble: text after the last section (includes <!-- KS:end -->)

    """
    lines = ks_block.split("\n")
    preamble_lines = []
    section_starts = []

    for i, line in enumerate(lines):
        match = re.match(r"^(#{2,3}) (.+)$", line)
        if match:
            section_starts.append((i, match.group(0), len(match.group(1))))
        elif not section_starts:
            preamble_lines.append(line)

    if not section_starts:
        return ks_block, [], ""

    preamble = "\n".join(preamble_lines)

    sections = []
    for idx, (start_line, header, level) in enumerate(section_starts):
        if idx + 1 < len(section_starts):
            end_line = section_starts[idx + 1][0]
        else:
            # Last section: find KS:end marker or use remaining lines
            end_line = len(lines)
            for j in range(start_line + 1, len(lines)):
                if KS_END in lines[j]:
                    end_line = j
                    break

        body = "\n".join(lines[start_line + 1 : end_line])
        sections.append(Section(header=header, level=level, body=body))

    # Postamble: find <!-- KS:end --> line and take everything from there
    postamble = ""
    for i in range(len(lines) - 1, -1, -1):
        if KS_END in lines[i]:
            postamble = "\n".join(lines[i:])
            break

    return preamble, sections, postamble


def reassemble(preamble: str, sections: list, postamble: str) -> str:
    """Reassemble parsed sections into a KS block string."""
    parts = [preamble]
    for section in sections:
        parts.append(section.header)
        parts.append(section.body)  # Always append — even empty bodies preserve blank lines
    if postamble:
        parts.append(postamble)

    result = "\n".join(parts)

    # Ensure markers are present
    if KS_START not in result:
        result = KS_START + "\n" + result
    if KS_END not in result:
        result = result.rstrip("\n") + "\n" + KS_END + "\n"

    return result


def find_section(sections: list, header_text: str, level: int | None = None) -> int | None:
    """Find a section by its header text and optional level. Returns index or None."""
    for i, section in enumerate(sections):
        if section.header.lstrip("#").strip() == header_text:
            if level is None or section.level == level:
                return i
    return None


# === Table parsing and normalization ===


def parse_html_tables(body: str) -> list:
    """Extract rows from all HTML <table> blocks in a section body.

    Returns list of (has_header, rows) tuples where rows is a list of cell lists.
    """
    tables = []
    for match in re.finditer(r"<table([^>]*)>(.*?)</table>", body, re.DOTALL):
        attrs = match.group(1)
        has_header = 'header-row="true"' in attrs or "header-row='true'" in attrs
        rows = []
        for tr in re.finditer(r"<tr>\s*(.*?)\s*</tr>", match.group(2), re.DOTALL):
            cells = [c.strip() for c in re.findall(r"<td>(.*?)</td>", tr.group(1), re.DOTALL)]
            rows.append(cells)
        if rows:
            tables.append((has_header, rows))
    return tables


def normalize_mastery_body(body: str, expected_columns: list) -> str:
    """Normalize a mastery section body into a single pipe-delimited table.

    Handles HTML <table> blocks, headerless tables, and multiple table blocks
    by merging all data rows under the expected schema.
    """
    html_tables = parse_html_tables(body)
    if not html_tables:
        return body

    data_rows = []
    for has_header, rows in html_tables:
        start = 1 if has_header else 0
        for cells in rows[start:]:
            padded = cells + [""] * max(0, len(expected_columns) - len(cells))
            row = dict(zip(expected_columns, padded[: len(expected_columns)]))
            data_rows.append(row)

    # Also collect any pipe-delimited rows outside HTML blocks
    clean = re.sub(r"<table[^>]*>.*?</table>", "", body, flags=re.DOTALL)
    if clean.strip():
        _, pipe_cols, pipe_rows = parse_table_rows(clean)
        if pipe_cols:
            data_rows.extend(pipe_rows)

    header = "| " + " | ".join(expected_columns) + " |"
    separator = "|" + "|".join("---" for _ in expected_columns) + "|"
    lines = [header, separator]
    for row in data_rows:
        cells = [row.get(col, "") for col in expected_columns]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n" + "\n".join(lines) + "\n"


# === Merge operations ===


def parse_table_rows(body: str) -> tuple:
    """Parse pipe-delimited table rows from a section body.

    Returns (header_lines, columns, data_rows) where:
    - header_lines: the header row + separator row as a single string
    - columns: list of column names
    - data_rows: list of dicts mapping column name -> value
    """
    lines = body.strip().split("\n")
    header_lines = []
    data_rows = []
    columns = []

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not columns:
            columns = cells
            header_lines.append(stripped)
        elif all(c.replace("-", "").strip() == "" for c in cells):
            header_lines.append(stripped)
        else:
            row = dict(zip(columns, cells))
            data_rows.append(row)

    return "\n".join(header_lines), columns, data_rows


def render_table(header_lines: str, columns: list, rows: list) -> str:
    """Render rows back into pipe-delimited table format."""
    lines = [header_lines]
    for row in rows:
        cells = [row.get(col, "") for col in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def apply_mastery_updates(sections: list, updates: list, dry_run: bool, strict: bool) -> list:
    """Apply mastery_updates to concept/chain/factor tables."""
    messages = []
    for update in updates:
        table = update.get("table")
        if table not in TABLE_SECTION_MAP:
            print(f"Warning: unknown mastery table '{table}', skipping", file=sys.stderr)
            continue

        section_name = TABLE_SECTION_MAP[table]
        idx = find_section(sections, section_name)
        if idx is None:
            warn_or_fail(f"section '## {section_name}' not found", strict)
            continue

        section = sections[idx]
        expected_cols = TABLE_SCHEMAS[table]
        section.body = normalize_mastery_body(section.body, expected_cols)
        header_lines, columns, data_rows = parse_table_rows(section.body)
        if not columns:
            warn_or_fail(f"no table header found in ## {section_name}", strict)
            continue
        name_col = columns[0]  # "Concept", "Chain", or "Factor"
        name = update["name"]

        # Find existing row
        existing_idx = None
        for i, row in enumerate(data_rows):
            if row.get(name_col) == name:
                existing_idx = i
                break

        if existing_idx is not None:
            row = data_rows[existing_idx]
            old_status = row.get("Status", "")
            new_status = update.get("status", old_status)
            old_evidence = row.get("Evidence", "")
            new_evidence = update.get("evidence")
            if new_evidence:
                combined = f"{old_evidence}, {new_evidence}" if old_evidence else new_evidence
            else:
                combined = old_evidence

            if dry_run:
                msgs = []
                if new_status != old_status:
                    msgs.append(f"status {old_status}\u2192{new_status}")
                if new_evidence:
                    msgs.append(f'evidence +="{new_evidence}"')
                messages.append(f'[mastery] UPDATE {table} "{name}": {", ".join(msgs)}')
            else:
                row["Status"] = new_status
                row["Evidence"] = combined
                row["Last Tested"] = update.get("last_tested", row.get("Last Tested", ""))
        else:
            new_row = {name_col: name}
            new_row["Status"] = update.get("status", "not-mastered")
            new_row["Evidence"] = update.get("evidence", "")
            new_row["Last Tested"] = update.get("last_tested", "")
            if table == "concepts":
                new_row["Syllabus Topic"] = update.get("syllabus_topic", "")
            data_rows.append(new_row)

            if dry_run:
                messages.append(f'[mastery] ADD {table} "{name}": status={new_row["Status"]}')

        if not dry_run:
            section.body = "\n" + render_table(header_lines, columns, data_rows) + "\n"

    return messages


def apply_weakness_queue(sections: list, queue: list, dry_run: bool, strict: bool) -> list:
    """Replace the entire ## Weakness Queue section."""
    messages = []
    idx = find_section(sections, "Weakness Queue")
    if idx is None:
        warn_or_fail("## Weakness Queue not found", strict)
        return messages

    if dry_run:
        old_rows = parse_table_rows(sections[idx].body)[2]
        messages.append(f"[weakness] REWRITE {len(queue)} rows (was {len(old_rows)})")
        return messages

    header = "| " + " | ".join(WEAKNESS_COLUMNS) + " |"
    separator = "|" + "|".join("---" for _ in WEAKNESS_COLUMNS) + "|"
    rows = []
    for entry in queue:
        row = {WEAKNESS_KEY_MAP.get(k, k): str(v) for k, v in entry.items()}
        cells = [row.get(col, "") for col in WEAKNESS_COLUMNS]
        rows.append("| " + " | ".join(cells) + " |")

    sections[idx].body = "\n" + header + "\n" + separator + "\n" + "\n".join(rows) + "\n"
    return messages


def apply_syllabus_updates(sections: list, updates: list, dry_run: bool, strict: bool) -> list:
    """Toggle syllabus checkboxes."""
    messages = []
    idx = find_section(sections, "Syllabus")
    if idx is None:
        warn_or_fail("## Syllabus not found", strict)
        return messages

    lines = sections[idx].body.split("\n")
    for update in updates:
        topic = update["topic"]
        status = update["status"]
        found = False
        for i, line in enumerate(lines):
            # Match "- [ ] Topic" or "- [x] Topic"
            match = re.match(r"^- \[[ x]\] (.+)$", line.strip())
            if match and match.group(1).strip() == topic:
                found = True
                new_check = "x" if status == "checked" else " "
                if dry_run:
                    action = "CHECK" if status == "checked" else "UNCHECK"
                    messages.append(f'[syllabus] {action} "{topic}"')
                else:
                    lines[i] = f"- [{new_check}] {topic}"
                break
        if not found:
            warn_or_fail(f"syllabus topic '{topic}' not found", strict)

    if not dry_run:
        sections[idx].body = "\n".join(lines)
    return messages


def apply_section_rewrites(sections: list, rewrites: dict, dry_run: bool, strict: bool) -> list:
    """Replace content of ##-level sections."""
    messages = []
    for key, content in rewrites.items():
        section_name = SECTION_REWRITE_MAP.get(key)
        if not section_name:
            print(f"Warning: unknown section_rewrites key '{key}', skipping", file=sys.stderr)
            continue
        idx = find_section(sections, section_name, level=2)
        if idx is None:
            warn_or_fail(f"## {section_name} not found", strict)
            continue
        if dry_run:
            messages.append(f"[rewrite] REPLACE ## {section_name}")
        else:
            sections[idx].body = "\n" + content + "\n"
    return messages


def apply_subsection_rewrites(sections: list, rewrites: dict, dry_run: bool, strict: bool) -> list:
    """Replace content of ###-level sections."""
    messages = []
    for key, content in rewrites.items():
        section_name = SUBSECTION_REWRITE_MAP.get(key)
        if not section_name:
            print(f"Warning: unknown subsection_rewrites key '{key}', skipping", file=sys.stderr)
            continue
        idx = find_section(sections, section_name, level=3)
        if idx is None:
            warn_or_fail(f"### {section_name} not found", strict)
            continue
        if dry_run:
            messages.append(f"[rewrite] REPLACE ### {section_name}")
        else:
            sections[idx].body = "\n" + content + "\n"
    return messages


def apply_section_appends(sections: list, appends: dict, dry_run: bool, strict: bool) -> list:
    """Append rows to table sections."""
    messages = []
    for key, row_text in appends.items():
        section_name = SECTION_APPEND_MAP.get(key)
        if not section_name:
            print(f"Warning: unknown section_appends key '{key}', skipping", file=sys.stderr)
            continue
        idx = find_section(sections, section_name, level=3)
        if idx is None:
            warn_or_fail(f"### {section_name} not found", strict)
            continue

        if dry_run:
            messages.append(f"[append] ADD row to ### {section_name}")
        else:
            # Find the last pipe-delimited row and append after it
            lines = sections[idx].body.split("\n")
            last_pipe_idx = -1
            for i, line in enumerate(lines):
                if line.strip().startswith("|"):
                    last_pipe_idx = i
            if last_pipe_idx >= 0:
                lines.insert(last_pipe_idx + 1, row_text)
            else:
                lines.append(row_text)
            sections[idx].body = "\n".join(lines)
    return messages


def apply_key_value_updates(
    sections: list,
    updates: dict,
    key_map: dict,
    section_name: str,
    dry_run: bool,
    strict: bool,
) -> list:
    """Update key-value lines (- Key: value) in a section."""
    messages = []
    idx = find_section(sections, section_name)
    if idx is None:
        warn_or_fail(f"section '{section_name}' not found", strict)
        return messages

    lines = sections[idx].body.split("\n")
    for json_key, value in updates.items():
        display_key = key_map.get(json_key)
        if not display_key:
            warn_or_fail(f"unknown key '{json_key}' for {section_name}", strict)
            continue

        found = False
        for i, line in enumerate(lines):
            if line.strip().startswith(f"- {display_key}:"):
                if dry_run:
                    old_val = line.split(":", 1)[1].strip() if ":" in line else ""
                    messages.append(
                        f"[{section_name.lower()}] {display_key}: {old_val}\u2192{value}"
                    )
                else:
                    lines[i] = f"- {display_key}: {value}"
                found = True
                break
        if not found:
            warn_or_fail(f"key '- {display_key}:' not found in {section_name}", strict)

    if not dry_run:
        sections[idx].body = "\n".join(lines)
    return messages


def main():
    """Entry point."""
    dry_run, strict, payload_path, ks_path = parse_args()
    try:
        payload = load_payload(payload_path)
        validate_payload_schema(payload, strict)
        ks_block = load_ks_block(ks_path)

        preamble, sections, postamble = parse_sections(ks_block)

        messages = []

        if "mastery_updates" in payload:
            messages.extend(
                apply_mastery_updates(
                    sections, payload["mastery_updates"], dry_run, strict
                )
            )
        if "weakness_queue" in payload:
            messages.extend(
                apply_weakness_queue(sections, payload["weakness_queue"], dry_run, strict)
            )
        if "syllabus_updates" in payload:
            messages.extend(
                apply_syllabus_updates(sections, payload["syllabus_updates"], dry_run, strict)
            )
        if "section_rewrites" in payload:
            messages.extend(
                apply_section_rewrites(sections, payload["section_rewrites"], dry_run, strict)
            )
        if "subsection_rewrites" in payload:
            messages.extend(
                apply_subsection_rewrites(
                    sections, payload["subsection_rewrites"], dry_run, strict
                )
            )
        if "section_appends" in payload:
            messages.extend(
                apply_section_appends(sections, payload["section_appends"], dry_run, strict)
            )
        if "load_baseline" in payload:
            messages.extend(
                apply_key_value_updates(
                    sections,
                    payload["load_baseline"],
                    LOAD_BASELINE_MAP,
                    "Baseline",
                    dry_run,
                    strict,
                )
            )
        if "engagement" in payload:
            messages.extend(
                apply_key_value_updates(
                    sections,
                    payload["engagement"],
                    ENGAGEMENT_MAP,
                    "Engagement Signals",
                    dry_run,
                    strict,
                )
            )

        if dry_run:
            sys.stdout.write("\n".join(messages) + "\n")
            return

        merged = reassemble(preamble, sections, postamble)
        sys.stdout.write(merged)
    except MergeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
