"""One-off migration script for the Docker DLN profile.

Converts legacy format (bullet-point concepts/chains, HTML tables, no KS markers)
to current format (pipe-delimited tables, KS boundary markers).

Pure text transformation — no Notion API calls.
"""

from __future__ import annotations


def rows_to_pipe_table(rows: list[dict], columns: list[str]) -> str:
    """Format a list of row dicts as a pipe-delimited Markdown table."""
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("------" for _ in columns) + "|"
    lines = [header, separator]
    for row in rows:
        cells = "| " + " | ".join(row.get(col, "") for col in columns) + " |"
        lines.append(cells)
    return "\n".join(lines)


# ---------- Task 2: parse_html_table ----------

import re


def parse_html_table(html: str) -> list[dict]:
    """Parse a Notion <table> HTML block into a list of row dicts.

    Expects <table header-row="true"> with <tr>/<td> elements.
    First <tr> is treated as headers, remaining <tr>s as data rows.
    """
    rows_raw = re.findall(r"<tr>\s*(.*?)\s*</tr>", html, re.DOTALL)
    if not rows_raw:
        return []

    def extract_cells(tr_content: str) -> list[str]:
        return [cell.strip() for cell in re.findall(r"<td>(.*?)</td>", tr_content, re.DOTALL)]

    headers = extract_cells(rows_raw[0])
    result = []
    for row_content in rows_raw[1:]:
        cells = extract_cells(row_content)
        row_dict = {}
        for i, header in enumerate(headers):
            row_dict[header] = cells[i] if i < len(cells) else ""
        result.append(row_dict)
    return result


# ---------- Task 3: parse_bullet_concepts / parse_bullet_chains ----------


def parse_bullet_concepts(section: str) -> list[dict]:
    """Parse bullet-point concept definitions into row dicts.

    Input format: `- **Name**: description text`
    Output: list of dicts with 'Concept' key.
    """
    results = []
    for match in re.finditer(r"^- \*\*(.+?)\*\*:", section, re.MULTILINE):
        results.append({"Concept": match.group(1)})
    return results


def parse_bullet_chains(section: str) -> list[dict]:
    """Parse bullet-point chain definitions into row dicts.

    Input format: `- **Name**: description text`
    Output: list of dicts with 'Chain' key.
    """
    results = []
    for match in re.finditer(r"^- \*\*(.+?)\*\*:", section, re.MULTILINE):
        results.append({"Chain": match.group(1)})
    return results


# ---------- Task 4: parse_weakness_bullets ----------


def parse_weakness_bullets(section: str) -> list[dict]:
    """Parse unstructured weakness queue bullets into table row dicts.

    Input format: `- Item description — extra context`
    Deduplicates by item text. Assigns sequential priority.
    """
    seen: set[str] = set()
    results = []
    priority = 1
    for line in section.strip().splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        # Split on em dash to separate item from context
        item_text = line[2:]  # Remove "- "
        parts = re.split(r"\s*—\s*", item_text, maxsplit=1)
        item = parts[0].strip()
        if item in seen:
            continue
        seen.add(item)
        results.append(
            {
                "Priority": str(priority),
                "Item": item,
                "Type": "concept",
                "Phase": "Network",
                "Severity": "low",
                "Source": "",
                "Added": "",
            }
        )
        priority += 1
    return results


# ---------- Task 5: infer_concept_status ----------


def infer_concept_status(
    concept_name: str,
    weakness_items: list[str],
    open_questions_text: str,
) -> str:
    """Infer mastery status for a concept by cross-referencing weaknesses and open questions.

    Uses case-insensitive word matching. Requires at least 2 significant words
    to match (or all words if the concept has fewer than 2 significant words)
    to reduce false positives from short common words like "dns" or "cmd".
    """
    name_lower = concept_name.lower()
    significant_words = [w for w in re.findall(r"\w+", name_lower) if len(w) >= 3]
    if not significant_words:
        return "mastered"

    # Require at least 2 word matches (or all words if fewer than 2)
    match_threshold = min(2, len(significant_words))

    for item in weakness_items:
        item_lower = item.lower()
        matches = sum(1 for word in significant_words if word in item_lower)
        if matches >= match_threshold:
            return "partial"

    oq_lower = open_questions_text.lower()
    matches = sum(1 for word in significant_words if word in oq_lower)
    if matches >= match_threshold:
        return "partial"

    return "mastered"


# ---------- Task 6: extract_sections ----------


def extract_sections(body: str) -> tuple[dict[str, str], str]:
    r"""Split page body into KS sections dict and session logs string.

    Sections are keyed by their H2 header name (e.g., "Concepts").
    Session logs are everything from the first `## Session \\d+` onward.
    The `# Knowledge State` H1 header is stripped.

    Returns:
        (sections_dict, session_logs_string)

    """
    # Find session log boundary: first H2 matching "## Session <number>"
    session_match = re.search(r"^(## Session \d+)", body, re.MULTILINE)
    if session_match:
        ks_body = body[: session_match.start()]
        session_logs = body[session_match.start() :]
    else:
        ks_body = body
        session_logs = ""

    # Split KS body into sections by H2 headers
    sections: dict[str, str] = {}
    h2_pattern = re.compile(r"^## (.+)$", re.MULTILINE)
    matches = list(h2_pattern.finditer(ks_body))

    for i, match in enumerate(matches):
        name = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(ks_body)
        content = ks_body[start:end].strip()
        # Strip trailing horizontal rules (--- separators from Notion)
        content = re.sub(r"\n---\s*$", "", content).strip()
        sections[name] = content

    return sections, session_logs


# ---------- Task 7: extract_factor_prose ----------


def extract_factor_prose(section: str) -> tuple[str, str]:
    """Separate factor bullet-point prose from the HTML table.

    Returns:
        (prose_text, table_html)

    """
    table_match = re.search(r"<table.*?>.*?</table>", section, re.DOTALL)
    if not table_match:
        return section, ""
    prose = section[: table_match.start()].strip()
    table_html = table_match.group(0)
    return prose, table_html


# ---------- Task 8: convert_legacy_profile ----------

# Section order matching init-template.md exactly
SECTION_ORDER = [
    "Syllabus",
    "Concepts",
    "Chains",
    "Factors",
    "Compressed Model",
    "Interleave Pool",
    "Calibration Log",
    "Load Profile",
    "Open Questions",
    "Weakness Queue",
    "Engagement Signals",
]

CONCEPT_COLUMNS = ["Concept", "Status", "Syllabus Topic", "Evidence", "Last Tested"]
CHAIN_COLUMNS = ["Chain", "Status", "Evidence", "Last Tested"]
FACTOR_COLUMNS = ["Factor", "Status", "Evidence", "Last Tested"]
WEAKNESS_COLUMNS = ["Priority", "Item", "Type", "Phase", "Severity", "Source", "Added"]
CONFIDENCE_COLUMNS = ["Concept", "Self-Rating (1-5)", "Actual Performance", "Gap", "Date"]
GATE_COLUMNS = ["Phase Gate", "Predicted Outcome", "Actual Outcome", "Date"]
SESSION_HISTORY_COLUMNS = ["Session", "Avg Batch Size", "Overload Signals", "Adjustments Made"]


def convert_legacy_profile(raw_page_body: str) -> str:
    """Convert a legacy DLN profile page body to current format.

    Parses bullet-point concepts/chains, HTML tables, and unstructured
    weakness bullets. Assembles output in init-template order with
    KS boundary markers.
    """
    sections, session_logs = extract_sections(raw_page_body)

    # Parse weakness items for status inference
    weakness_section = sections.get("Weakness Queue", "")
    weakness_rows = parse_weakness_bullets(weakness_section)
    weakness_items = [r["Item"] for r in weakness_rows]

    open_questions_text = sections.get("Open Questions", "")

    # Build each section
    output_parts: list[str] = []
    output_parts.append("<!-- KS:start -->")
    output_parts.append("# Knowledge State")
    output_parts.append("")

    for section_name in SECTION_ORDER:
        content = sections.get(section_name, "")
        output_parts.append(f"## {section_name}")

        if section_name == "Syllabus":
            # Preserve verbatim
            output_parts.append(content if content else "")

        elif section_name == "Concepts":
            output_parts.append("")
            concept_rows = parse_bullet_concepts(content)
            for row in concept_rows:
                row["Status"] = infer_concept_status(
                    row["Concept"], weakness_items, open_questions_text
                )
            output_parts.append(rows_to_pipe_table(concept_rows, CONCEPT_COLUMNS))

        elif section_name == "Chains":
            output_parts.append("")
            chain_rows = parse_bullet_chains(content)
            for row in chain_rows:
                row["Status"] = "mastered"
            output_parts.append(rows_to_pipe_table(chain_rows, CHAIN_COLUMNS))

        elif section_name == "Factors":
            prose, table_html = extract_factor_prose(content)
            if prose:
                output_parts.append(prose)
            output_parts.append("")
            if table_html:
                factor_rows = parse_html_table(table_html)
            else:
                factor_rows = []
            output_parts.append(rows_to_pipe_table(factor_rows, FACTOR_COLUMNS))

        elif section_name == "Compressed Model":
            output_parts.append(content if content else "")

        elif section_name == "Interleave Pool":
            output_parts.append(content if content else "")

        elif section_name == "Calibration Log":
            output_parts.append(_convert_calibration_log(content))

        elif section_name == "Load Profile":
            output_parts.append(_convert_load_profile(content))

        elif section_name == "Open Questions":
            output_parts.append(content if content else "")

        elif section_name == "Weakness Queue":
            output_parts.append("")
            output_parts.append(rows_to_pipe_table(weakness_rows, WEAKNESS_COLUMNS))

        elif section_name == "Engagement Signals":
            output_parts.append(content if content else "")

        output_parts.append("")

    output_parts.append("<!-- KS:end -->")

    if session_logs.strip():
        output_parts.append("")
        output_parts.append(session_logs.strip())

    return "\n".join(output_parts)


def _convert_calibration_log(content: str) -> str:
    """Convert Calibration Log section: text notes + HTML tables to pipe-delimited."""
    parts: list[str] = []

    # Split into subsections by ### headers
    sub_matches = list(re.finditer(r"^### (.+)$", content, re.MULTILINE))

    # Text before first subsection (free-form notes)
    if sub_matches:
        preamble = content[: sub_matches[0].start()].strip()
    else:
        preamble = content.strip()

    # Extract gate predictions from preamble (they may appear as bullets before subsections)
    preamble_gate_rows = _parse_gate_predictions_text(preamble) if preamble else []
    # Remove gate prediction lines from preamble text
    if preamble_gate_rows and preamble:
        preamble_lines = []
        for line in preamble.splitlines():
            normalized = line.replace("\\|", "|")
            if not re.search(r"Phase Gate\s+.+\|.*Predicted:", normalized):
                preamble_lines.append(line)
        preamble = "\n".join(preamble_lines).strip()

    if preamble:
        parts.append(preamble)

    for i, match in enumerate(sub_matches):
        sub_name = match.group(1).strip()
        start = match.end()
        end = sub_matches[i + 1].start() if i + 1 < len(sub_matches) else len(content)
        sub_content = content[start:end].strip()

        parts.append(f"\n### {sub_name}")

        if sub_name == "Concept-Level Confidence":
            table_match = re.search(r"<table.*?>.*?</table>", sub_content, re.DOTALL)
            if table_match:
                rows = parse_html_table(table_match.group(0))
                # Rename "Factor" -> "Concept", "Self-Rating" -> "Self-Rating (1-5)"
                for row in rows:
                    if "Factor" in row:
                        row["Concept"] = row.pop("Factor")
                    if "Self-Rating" in row:
                        row["Self-Rating (1-5)"] = row.pop("Self-Rating")
                parts.append(rows_to_pipe_table(rows, CONFIDENCE_COLUMNS))
            else:
                parts.append(sub_content)

        elif sub_name == "Gate Predictions":
            # Parse gate predictions from both subsection and preamble
            gate_rows = _parse_gate_predictions_text(sub_content)
            gate_rows = preamble_gate_rows + gate_rows
            parts.append(rows_to_pipe_table(gate_rows, GATE_COLUMNS))

        elif sub_name == "Calibration Trend":
            parts.append(sub_content if sub_content else "")

        else:
            parts.append(sub_content)

    # Ensure all required subsections exist
    existing_subs = {m.group(1).strip() for m in sub_matches}
    if "Concept-Level Confidence" not in existing_subs:
        parts.append("\n### Concept-Level Confidence")
        parts.append(rows_to_pipe_table([], CONFIDENCE_COLUMNS))
    if "Gate Predictions" not in existing_subs:
        parts.append("\n### Gate Predictions")
        gate_rows = preamble_gate_rows if preamble_gate_rows else []
        parts.append(rows_to_pipe_table(gate_rows, GATE_COLUMNS))
    if "Calibration Trend" not in existing_subs:
        parts.append("\n### Calibration Trend")

    return "\n".join(parts)


def _convert_load_profile(content: str) -> str:
    """Convert Load Profile section: ensure Session History table header exists."""
    if "### Session History" in content:
        # Check if it already has a pipe table
        sh_match = re.search(r"(### Session History\s*\n)(.*?)(?=\n###|\Z)", content, re.DOTALL)
        if sh_match and "|" not in sh_match.group(2):
            # Insert empty table between header and any existing content
            existing_content = sh_match.group(2).strip()
            replacement = "### Session History\n" + rows_to_pipe_table([], SESSION_HISTORY_COLUMNS)
            if existing_content:
                replacement += "\n" + existing_content
            content = content[: sh_match.start()] + replacement + content[sh_match.end() :]
    elif "### Baseline" in content:
        content = (
            content.rstrip()
            + "\n\n### Session History\n"
            + rows_to_pipe_table([], SESSION_HISTORY_COLUMNS)
        )
    else:
        # No ### Baseline or ### Session History — append both
        content = (
            content.rstrip()
            + "\n\n### Session History\n"
            + rows_to_pipe_table([], SESSION_HISTORY_COLUMNS)
        )
    return content


def _parse_gate_predictions_text(text: str) -> list[dict]:
    r"""Parse Gate Predictions from free-text format into table rows.

    Handles the known format: `Phase Gate X→Y | Predicted: ... | Actual: ... | Date: ...`
    Also handles pipe-escaped variants from Notion (\\|).
    """
    rows = []
    for line in text.strip().splitlines():
        # Normalize escaped pipes from Notion
        line = line.replace("\\|", "|")
        # Try to parse pipe-separated gate prediction
        gate_match = re.search(
            r"Phase Gate\s+(.+?)\s*\|\s*Predicted:\s*(.+?)\s*\|\s*Actual:\s*(.+?)\s*\|\s*Date:\s*(.+)",
            line,
        )
        if gate_match:
            rows.append(
                {
                    "Phase Gate": gate_match.group(1).strip(),
                    "Predicted Outcome": gate_match.group(2).strip(),
                    "Actual Outcome": gate_match.group(3).strip(),
                    "Date": gate_match.group(4).strip(),
                }
            )
    return rows


# ---------- Task 9: CLI entry point ----------

import sys
from pathlib import Path


def main() -> None:
    """CLI entry point: reads snapshot file, writes converted output."""
    if len(sys.argv) != 2:
        print("Usage: python migrate_docker.py <snapshot-file>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    raw_body = input_path.read_text()
    converted = convert_legacy_profile(raw_body)

    output_path = input_path.with_suffix(".converted.md")
    output_path.write_text(converted)
    print(f"Converted output written to: {output_path}")


if __name__ == "__main__":
    main()
