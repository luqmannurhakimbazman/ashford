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


def main():
    """Entry point."""
    dry_run, payload_path, ks_path = parse_args()
    payload = load_payload(payload_path)
    ks_block = load_ks_block(ks_path)

    preamble, sections, postamble = parse_sections(ks_block)

    messages = []

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

    if dry_run:
        sys.stdout.write("\n".join(messages) + "\n")
        return

    merged = reassemble(preamble, sections, postamble)
    sys.stdout.write(merged)


if __name__ == "__main__":
    main()
