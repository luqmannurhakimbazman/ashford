#!/usr/bin/env python3
"""Deterministic KS merge script.

Takes a typed JSON payload and a raw KS block (markdown between markers),
applies updates deterministically, and outputs the merged KS block to stdout.

Usage:
    python3 ks-merge.py [--dry-run] <payload_path> <ks_block_path>

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

EXAM_METADATA_MAP = {
    "exam_date": "exam_date",
    "exam_format": "exam_format",
    "duration": "duration",
    "total_marks": "total_marks",
    "ai_policy": "ai_policy",
    "target_score_raw": "target_score_raw",
    "target_score_numeric": "target_score_numeric",
    "aspirational_target": "aspirational_target",
    "time_horizon_preset": "time_horizon_preset",
    "artifacts_ingested": "artifacts_ingested",
    "last_reprioritization": "last_reprioritization",
    "sessions_since_reprioritization": "sessions_since_reprioritization",
}

AGGREGATE_MAP = {
    "estimated_exam_score": "Estimated exam score",
    "hours_studied": "Hours studied",
    "phase_session_minutes": "Phase session minutes",
    "mock_session_minutes": "Mock session minutes",
    "marks_gain_rate": "Marks gain rate",
    "overall_no_ai_accuracy": "Overall no-AI accuracy",
    "overall_ai_dependence_delta": "Overall AI-dependence delta",
    "readiness": "Readiness",
}

# Table schemas for exam-related sections
TOPIC_MAP_COLUMNS = [
    "Topic",
    "Marks Weight",
    "Exam Frequency",
    "Transfer Leverage",
    "Hours To Floor",
    "Priority Score",
    "Current No-AI Score",
]
TOPIC_MAP_KEY_MAP = {
    "topic": "Topic",
    "marks_weight": "Marks Weight",
    "exam_frequency": "Exam Frequency",
    "transfer_leverage": "Transfer Leverage",
    "hours_to_floor": "Hours To Floor",
    "priority_score": "Priority Score",
    "current_no_ai_score": "Current No-AI Score",
}

PAST_PAPER_COLUMNS = ["Paper", "Year", "Topics Covered", "Avg Marks/Topic"]
PAST_PAPER_KEY_MAP = {
    "paper": "Paper",
    "year": "Year",
    "topics_covered": "Topics Covered",
    "avg_marks_per_topic": "Avg Marks/Topic",
}

PER_TOPIC_METRICS_COLUMNS = [
    "Topic",
    "Closed Book Acc",
    "Time/Question (s)",
    "Marks/Min",
    "Retention Delta",
    "AI Dep Delta",
]
PER_TOPIC_METRICS_KEY_MAP = {
    "topic": "Topic",
    "closed_book_acc": "Closed Book Acc",
    "time_per_question": "Time/Question (s)",
    "marks_per_min": "Marks/Min",
    "retention_delta": "Retention Delta",
    "ai_dep_delta": "AI Dep Delta",
}

PAST_EXAMS_COLUMNS = [
    "Exam Date",
    "Format",
    "Total Marks",
    "Target (raw)",
    "Target (numeric)",
    "Mock Count",
    "Best Mock Score",
    "Self-Reported Result",
    "Archived",
]
PAST_EXAMS_KEY_MAP = {
    "exam_date": "Exam Date",
    "exam_format": "Format",
    "total_marks": "Total Marks",
    "target_score_raw": "Target (raw)",
    "target_score_numeric": "Target (numeric)",
    "mock_count": "Mock Count",
    "best_mock_score": "Best Mock Score",
    "self_reported_result": "Self-Reported Result",
    "archived_at": "Archived",
}

QUESTION_BANK_COLUMNS = [
    "ID",
    "Topic",
    "Format",
    "Marks",
    "Difficulty",
    "Source",
    "Used In Mock",
    "Last Score",
]
QUESTION_BANK_KEY_MAP = {
    "id": "ID",
    "topic": "Topic",
    "format": "Format",
    "marks": "Marks",
    "difficulty": "Difficulty",
    "source": "Source",
    "used_in_mock": "Used In Mock",
    "last_score": "Last Score",
}

MOCK_HISTORY_COLUMNS = [
    "Mock #",
    "Date",
    "Score",
    "Time Used",
    "Marks/Min",
    "Weak Topics",
    "Notes",
]
MOCK_HISTORY_KEY_MAP = {
    "mock_number": "Mock #",
    "date": "Date",
    "score": "Score",
    "time_used": "Time Used",
    "marks_per_min": "Marks/Min",
    "weak_topics": "Weak Topics",
    "notes": "Notes",
}

ERROR_TAXONOMY_COLUMNS = [
    "Error ID",
    "Type",
    "Description",
    "Frequency",
    "Topics Affected",
    "Remediation",
]
ERROR_TAXONOMY_KEY_MAP = {
    "error_id": "Error ID",
    "type": "Type",
    "description": "Description",
    "frequency": "Frequency",
    "topics_affected": "Topics Affected",
    "remediation": "Remediation",
}

# Canonical section ordering for auto-creation of missing exam sections.
CANONICAL_SECTION_ORDER = [
    ("Syllabus", 2),
    ("Exam Metadata", 2),
    ("Exam Blueprint", 2),
    ("Topic Map", 3),
    ("High-Yield Queue", 3),
    ("Past Paper Analysis", 3),
    ("Concepts", 2),
    ("Chains", 2),
    ("Factors", 2),
    ("Compressed Model", 2),
    ("Interleave Pool", 2),
    ("Calibration Log", 2),
    ("Concept-Level Confidence", 3),
    ("Gate Predictions", 3),
    ("Calibration Trend", 3),
    ("Load Profile", 2),
    ("Baseline", 3),
    ("Session History", 3),
    ("Exam Metrics", 2),
    ("Per-Topic Metrics", 3),
    ("Aggregate", 3),
    ("Question Bank", 2),
    ("Mock History", 2),
    ("Error Taxonomy", 2),
    ("Past Exams", 2),
    ("Open Questions", 2),
    ("Weakness Queue", 2),
    ("Engagement Signals", 2),
]

# Empty body templates for auto-created sections (contains table headers where needed).
SECTION_EMPTY_BODIES = {
    "Exam Metadata": (
        "\n- exam_date:\n- exam_format:\n- duration:\n- total_marks:\n"
        "- ai_policy:\n- target_score_raw:\n- target_score_numeric:\n"
        "- aspirational_target:\n- time_horizon_preset:\n"
        "- artifacts_ingested: 0\n- last_reprioritization: \n"
        "- sessions_since_reprioritization: 0\n"
    ),
    "Topic Map": (
        "\n| Topic | Marks Weight | Exam Frequency | Transfer Leverage"
        " | Hours To Floor | Priority Score | Current No-AI Score |\n"
        "|-------|-------------|----------------|-------------------|"
        "----------------|----------------|---------------------|\n"
    ),
    "High-Yield Queue": "\n",
    "Past Paper Analysis": (
        "\n| Paper | Year | Topics Covered | Avg Marks/Topic |\n"
        "|-------|------|----------------|------------------|\n"
    ),
    "Per-Topic Metrics": (
        "\n| Topic | Closed Book Acc | Time/Question (s) | Marks/Min"
        " | Retention Delta | AI Dep Delta |\n"
        "|-------|----------------|-------------------|-----------|"
        "-----------------|-------------|\n"
    ),
    "Aggregate": (
        "\n- Estimated exam score:\n- Hours studied: 0\n"
        "- Phase session minutes: 0\n- Mock session minutes: 0\n"
        "- Marks gain rate:\n- Overall no-AI accuracy:\n"
        "- Overall AI-dependence delta:\n- Readiness: NOT READY\n"
    ),
    "Question Bank": (
        "\n| ID | Topic | Format | Marks | Difficulty | Source"
        " | Used In Mock | Last Score |\n"
        "|----|-------|--------|-------|------------|--------|"
        "-------------|------------|\n"
    ),
    "Mock History": (
        "\n| Mock # | Date | Score | Time Used | Marks/Min"
        " | Weak Topics | Notes |\n"
        "|--------|------|-------|-----------|-----------|"
        "-------------|-------|\n"
    ),
    "Error Taxonomy": (
        "\n| Error ID | Type | Description | Frequency"
        " | Topics Affected | Remediation |\n"
        "|----------|------|-------------|---------|"
        "-----------------|-------------|\n"
    ),
    "Past Exams": (
        "\n| Exam Date | Format | Total Marks | Target (raw)"
        " | Target (numeric) | Mock Count | Best Mock Score"
        " | Self-Reported Result | Archived |\n"
        "|-----------|--------|-------------|--------------|"
        "------------------|------------|-----------------|"
        "---------------------|----------|\n"
    ),
    "Exam Blueprint": "\n",
    "Exam Metrics": "\n",
}


@dataclass
class Section:
    """A parsed section of the KS block."""

    header: str  # The full header line (e.g., "## Concepts")
    level: int  # Header level (2 for ##, 3 for ###)
    body: str  # Everything after the header until the next section


def parse_args() -> tuple:
    """Parse CLI arguments. Returns (dry_run, payload_path, ks_block_path)."""
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    if dry_run:
        args.remove("--dry-run")
    if len(args) != 2:
        print("Usage: ks-merge.py [--dry-run] <payload_path> <ks_block_path>", file=sys.stderr)
        sys.exit(1)
    return dry_run, args[0], args[1]


def load_payload(path: str) -> dict:
    """Load and validate the JSON payload."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Failed to parse payload: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, dict):
        print("Payload must be a JSON object", file=sys.stderr)
        sys.exit(1)
    return data


def load_ks_block(path: str) -> str:
    """Load the raw KS block from file."""
    try:
        with open(path) as f:
            return f.read()
    except OSError as e:
        print(f"Failed to read KS block: {e}", file=sys.stderr)
        sys.exit(1)


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


def apply_mastery_updates(sections: list, updates: list, dry_run: bool) -> list:
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
            print(f"Warning: section '## {section_name}' not found, skipping", file=sys.stderr)
            continue

        section = sections[idx]
        expected_cols = TABLE_SCHEMAS[table]
        section.body = normalize_mastery_body(section.body, expected_cols)
        header_lines, columns, data_rows = parse_table_rows(section.body)
        if not columns:
            print(
                f"Warning: no table header found in ## {section_name}, skipping",
                file=sys.stderr,
            )
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


def apply_weakness_queue(sections: list, queue: list, dry_run: bool) -> list:
    """Replace the entire ## Weakness Queue section."""
    messages = []
    idx = find_section(sections, "Weakness Queue")
    if idx is None:
        print("Warning: ## Weakness Queue not found, skipping", file=sys.stderr)
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


def apply_syllabus_updates(sections: list, updates: list, dry_run: bool) -> list:
    """Toggle syllabus checkboxes."""
    messages = []
    idx = find_section(sections, "Syllabus")
    if idx is None:
        print("Warning: ## Syllabus not found, skipping", file=sys.stderr)
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
            print(f"Warning: syllabus topic '{topic}' not found, skipping", file=sys.stderr)

    if not dry_run:
        sections[idx].body = "\n".join(lines)
    return messages


def apply_section_rewrites(sections: list, rewrites: dict, dry_run: bool) -> list:
    """Replace content of ##-level sections."""
    messages = []
    for key, content in rewrites.items():
        section_name = SECTION_REWRITE_MAP.get(key)
        if not section_name:
            print(f"Warning: unknown section_rewrites key '{key}', skipping", file=sys.stderr)
            continue
        idx = find_section(sections, section_name, level=2)
        if idx is None:
            print(f"Warning: ## {section_name} not found, skipping", file=sys.stderr)
            continue
        if dry_run:
            messages.append(f"[rewrite] REPLACE ## {section_name}")
        else:
            sections[idx].body = "\n" + content + "\n"
    return messages


def apply_subsection_rewrites(sections: list, rewrites: dict, dry_run: bool) -> list:
    """Replace content of ###-level sections."""
    messages = []
    for key, content in rewrites.items():
        section_name = SUBSECTION_REWRITE_MAP.get(key)
        if not section_name:
            print(f"Warning: unknown subsection_rewrites key '{key}', skipping", file=sys.stderr)
            continue
        idx = find_section(sections, section_name, level=3)
        if idx is None:
            print(f"Warning: ### {section_name} not found, skipping", file=sys.stderr)
            continue
        if dry_run:
            messages.append(f"[rewrite] REPLACE ### {section_name}")
        else:
            sections[idx].body = "\n" + content + "\n"
    return messages


def apply_section_appends(sections: list, appends: dict, dry_run: bool) -> list:
    """Append rows to table sections."""
    messages = []
    for key, row_text in appends.items():
        section_name = SECTION_APPEND_MAP.get(key)
        if not section_name:
            print(f"Warning: unknown section_appends key '{key}', skipping", file=sys.stderr)
            continue
        idx = find_section(sections, section_name, level=3)
        if idx is None:
            print(f"Warning: ### {section_name} not found, skipping", file=sys.stderr)
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
) -> list:
    """Update key-value lines (- Key: value) in a section."""
    messages = []
    idx = find_section(sections, section_name)
    if idx is None:
        print(f"Warning: section '{section_name}' not found, skipping", file=sys.stderr)
        return messages

    lines = sections[idx].body.split("\n")
    for json_key, value in updates.items():
        display_key = key_map.get(json_key)
        if not display_key:
            print(
                f"Warning: unknown key '{json_key}' for {section_name}, skipping", file=sys.stderr
            )
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
            print(
                f"Warning: key '- {display_key}:' not found in {section_name}, skipping",
                file=sys.stderr,
            )

    if not dry_run:
        sections[idx].body = "\n".join(lines)
    return messages


def ensure_exam_sections(sections: list) -> None:
    """Auto-create missing exam sections at their canonical positions.

    Checks each section in CANONICAL_SECTION_ORDER. If missing, inserts an
    empty section at the correct position relative to existing sections.
    """
    for name, level in CANONICAL_SECTION_ORDER:
        idx = find_section(sections, name, level=level)
        if idx is not None:
            continue

        body = SECTION_EMPTY_BODIES.get(name)
        if body is None:
            continue  # Only auto-create sections we have templates for

        header = "#" * level + " " + name
        new_section = Section(header=header, level=level, body=body)

        # Find the right insertion point: after the last existing section
        # that precedes this one in canonical order.
        my_pos = CANONICAL_SECTION_ORDER.index((name, level))
        insert_at = len(sections)  # default: end
        for prev_pos in range(my_pos - 1, -1, -1):
            prev_name, prev_level = CANONICAL_SECTION_ORDER[prev_pos]
            prev_idx = find_section(sections, prev_name, level=prev_level)
            if prev_idx is not None:
                insert_at = prev_idx + 1
                break

        sections.insert(insert_at, new_section)


def apply_table_upsert(
    sections: list,
    section_name: str,
    level: int,
    columns: list,
    key_columns: list,
    key_map: dict,
    entries: list,
    dry_run: bool,
    label: str,
) -> list:
    """Generic table upsert: match rows by one or more key columns.

    Args:
        sections: Parsed section list.
        section_name: Header text of the target section.
        level: Header level (2 or 3).
        columns: Ordered list of display column names.
        key_columns: Display column name(s) used for matching (e.g. ["Topic"]).
        key_map: JSON field name -> display column name mapping.
        entries: List of dicts with JSON field names as keys.
        dry_run: If True, only report planned changes.
        label: Label prefix for dry-run messages (e.g. "topic_map").

    Returns:
        List of dry-run message strings (empty list when not dry-run).

    """
    messages: list[str] = []
    idx = find_section(sections, section_name, level=level)
    if idx is None:
        print(
            f"Warning: {('#' * level)} {section_name} not found, skipping",
            file=sys.stderr,
        )
        return messages

    header_lines, parsed_cols, data_rows = parse_table_rows(sections[idx].body)
    if not parsed_cols:
        # Build header from schema
        header_line = "| " + " | ".join(columns) + " |"
        sep_line = "|" + "|".join("---" for _ in columns) + "|"
        header_lines = header_line + "\n" + sep_line
        parsed_cols = columns
        data_rows = []

    for entry in entries:
        row_data = {}
        for json_key, value in entry.items():
            display_col = key_map.get(json_key)
            if display_col:
                row_data[display_col] = str(value)

        # Find existing row by matching ALL key columns
        existing_idx = None
        for i, row in enumerate(data_rows):
            if all(row.get(kc, "") == row_data.get(kc, "") for kc in key_columns):
                existing_idx = i
                break

        if existing_idx is not None:
            if dry_run:
                key_desc = ", ".join(f"{kc}={row_data.get(kc, '')}" for kc in key_columns)
                messages.append(f"[{label}] UPDATE row ({key_desc})")
            else:
                for col in columns:
                    if col in row_data:
                        data_rows[existing_idx][col] = row_data[col]
        else:
            new_row = {col: row_data.get(col, "") for col in columns}
            data_rows.append(new_row)
            if dry_run:
                key_desc = ", ".join(f"{kc}={row_data.get(kc, '')}" for kc in key_columns)
                messages.append(f"[{label}] ADD row ({key_desc})")

    if not dry_run:
        sections[idx].body = "\n" + render_table(header_lines, parsed_cols, data_rows) + "\n"

    return messages


def apply_table_append(
    sections: list,
    section_name: str,
    level: int,
    columns: list,
    key_map: dict,
    entries: list,
    dry_run: bool,
    label: str,
) -> list:
    """Append rows to a table without deduplication.

    Args:
        sections: Parsed section list.
        section_name: Header text of the target section.
        level: Header level (2 or 3).
        columns: Ordered list of display column names.
        key_map: JSON field name -> display column name mapping.
        entries: List of dicts with JSON field names as keys.
        dry_run: If True, only report planned changes.
        label: Label prefix for dry-run messages.

    Returns:
        List of dry-run message strings (empty list when not dry-run).

    """
    messages: list[str] = []
    idx = find_section(sections, section_name, level=level)
    if idx is None:
        print(
            f"Warning: {('#' * level)} {section_name} not found, skipping",
            file=sys.stderr,
        )
        return messages

    for entry in entries:
        row_data = {}
        for json_key, value in entry.items():
            display_col = key_map.get(json_key)
            if display_col:
                row_data[display_col] = str(value)

        cells = [row_data.get(col, "") for col in columns]
        row_text = "| " + " | ".join(cells) + " |"

        if dry_run:
            messages.append(f"[{label}] APPEND row")
        else:
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


def _has_exam_keys(payload: dict) -> bool:
    """Return True if the payload contains any exam-related keys."""
    exam_keys = {
        "exam_metadata",
        "exam_blueprint",
        "exam_metrics",
        "past_exams",
        "question_bank",
        "mock_history",
        "error_taxonomy",
    }
    return bool(exam_keys & payload.keys())


def main():
    """Entry point."""
    dry_run, payload_path, ks_path = parse_args()
    payload = load_payload(payload_path)
    ks_block = load_ks_block(ks_path)

    preamble, sections, postamble = parse_sections(ks_block)

    messages = []

    # Auto-create missing exam sections before any exam handlers run.
    if _has_exam_keys(payload):
        ensure_exam_sections(sections)

    if "mastery_updates" in payload:
        messages.extend(apply_mastery_updates(sections, payload["mastery_updates"], dry_run))
    if "weakness_queue" in payload:
        messages.extend(apply_weakness_queue(sections, payload["weakness_queue"], dry_run))
    if "syllabus_updates" in payload:
        messages.extend(apply_syllabus_updates(sections, payload["syllabus_updates"], dry_run))
    if "section_rewrites" in payload:
        messages.extend(apply_section_rewrites(sections, payload["section_rewrites"], dry_run))
    if "subsection_rewrites" in payload:
        messages.extend(
            apply_subsection_rewrites(sections, payload["subsection_rewrites"], dry_run)
        )
    if "section_appends" in payload:
        messages.extend(apply_section_appends(sections, payload["section_appends"], dry_run))
    if "load_baseline" in payload:
        messages.extend(
            apply_key_value_updates(
                sections, payload["load_baseline"], LOAD_BASELINE_MAP, "Baseline", dry_run
            )
        )
    if "engagement" in payload:
        messages.extend(
            apply_key_value_updates(
                sections, payload["engagement"], ENGAGEMENT_MAP, "Engagement Signals", dry_run
            )
        )

    # --- Exam handlers ---

    if "exam_metadata" in payload:
        messages.extend(
            apply_key_value_updates(
                sections,
                payload["exam_metadata"],
                EXAM_METADATA_MAP,
                "Exam Metadata",
                dry_run,
            )
        )

    if "exam_blueprint" in payload:
        bp = payload["exam_blueprint"]
        if "topic_map" in bp:
            messages.extend(
                apply_table_upsert(
                    sections,
                    "Topic Map",
                    3,
                    TOPIC_MAP_COLUMNS,
                    ["Topic"],
                    TOPIC_MAP_KEY_MAP,
                    bp["topic_map"],
                    dry_run,
                    "topic_map",
                )
            )
        if "high_yield_queue" in bp:
            idx = find_section(sections, "High-Yield Queue", level=3)
            if idx is not None:
                if dry_run:
                    messages.append("[exam_blueprint] REPLACE ### High-Yield Queue")
                else:
                    sections[idx].body = "\n" + bp["high_yield_queue"] + "\n"
            else:
                print(
                    "Warning: ### High-Yield Queue not found, skipping",
                    file=sys.stderr,
                )
        if "past_paper_analysis" in bp:
            messages.extend(
                apply_table_upsert(
                    sections,
                    "Past Paper Analysis",
                    3,
                    PAST_PAPER_COLUMNS,
                    ["Paper", "Year"],
                    PAST_PAPER_KEY_MAP,
                    bp["past_paper_analysis"],
                    dry_run,
                    "past_paper",
                )
            )

    if "exam_metrics" in payload:
        em = payload["exam_metrics"]
        if "per_topic" in em:
            messages.extend(
                apply_table_upsert(
                    sections,
                    "Per-Topic Metrics",
                    3,
                    PER_TOPIC_METRICS_COLUMNS,
                    ["Topic"],
                    PER_TOPIC_METRICS_KEY_MAP,
                    em["per_topic"],
                    dry_run,
                    "per_topic_metrics",
                )
            )
        if "aggregate" in em:
            messages.extend(
                apply_key_value_updates(
                    sections,
                    em["aggregate"],
                    AGGREGATE_MAP,
                    "Aggregate",
                    dry_run,
                )
            )

    if "past_exams" in payload:
        messages.extend(
            apply_table_upsert(
                sections,
                "Past Exams",
                2,
                PAST_EXAMS_COLUMNS,
                ["Exam Date"],
                PAST_EXAMS_KEY_MAP,
                payload["past_exams"],
                dry_run,
                "past_exams",
            )
        )

    if "question_bank" in payload:
        messages.extend(
            apply_table_upsert(
                sections,
                "Question Bank",
                2,
                QUESTION_BANK_COLUMNS,
                ["ID"],
                QUESTION_BANK_KEY_MAP,
                payload["question_bank"],
                dry_run,
                "question_bank",
            )
        )

    if "mock_history" in payload:
        messages.extend(
            apply_table_append(
                sections,
                "Mock History",
                2,
                MOCK_HISTORY_COLUMNS,
                MOCK_HISTORY_KEY_MAP,
                payload["mock_history"],
                dry_run,
                "mock_history",
            )
        )

    if "error_taxonomy" in payload:
        messages.extend(
            apply_table_upsert(
                sections,
                "Error Taxonomy",
                2,
                ERROR_TAXONOMY_COLUMNS,
                ["Error ID"],
                ERROR_TAXONOMY_KEY_MAP,
                payload["error_taxonomy"],
                dry_run,
                "error_taxonomy",
            )
        )

    if dry_run:
        sys.stdout.write("\n".join(messages) + "\n")
        return

    merged = reassemble(preamble, sections, postamble)
    sys.stdout.write(merged)


if __name__ == "__main__":
    main()
