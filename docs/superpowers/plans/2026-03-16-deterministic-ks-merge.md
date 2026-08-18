# Deterministic KS Merge Implementation Plan

> **Superseded by Dunk 2.0.0.** Historical record of the removed `dln-sync`/Notion remote path; `ks-merge.py` is now legacy compatibility tooling only and the local-first filesystem store in `dunk/skills/dln/references/local-persistence-protocol.md` is authoritative.

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the LLM-driven merge in dln-sync's Step 2 with a deterministic Python script, reducing Sonnet turns per sync boundary by 2-4 and eliminating string-manipulation errors.

**Architecture:** Phase skills keep sending prose. dln-sync normalizes prose → typed JSON (1 LLM turn), then pipes JSON + KS block to `ks-merge.py` (deterministic). On script failure, hard fail and queue writes — no LLM fallback.

**Tech Stack:** Python 3.13 (managed via `uv`), stdlib only (json, sys, re), Markdown agent definitions

**Spec:** `docs/superpowers/specs/2026-03-16-deterministic-ks-merge-design.md`

---

## Task 1: Create the merge payload schema document

**Files:**
- Create: `dunk/references/merge-payload-schema.md`

- [ ] **Step 1: Create the schema document**

Create `dunk/references/merge-payload-schema.md` with the full JSON schema from the spec. This is the single source of truth for the contract between the normalizer and the merge script. Copy the schema verbatim from the spec's "Merge Payload Schema" section (lines 61-127), including the notes about `section_rewrites` being COMPLETE replacements, the `subsection_rewrites` distinction, and the `syllabus_updates` consolidation note.

```markdown
# Merge Payload Schema

JSON contract between the dln-sync normalizer and `ks-merge.py`. The normalizer produces this; the script consumes it. All fields are optional — a dispatch only includes what changed.

## Schema

\```json
{
  "mastery_updates": [
    {
      "table": "concepts | chains | factors",
      "name": "row identifier — exact match against Concept/Chain/Factor column",
      "status": "not-mastered | partial | mastered",
      "evidence": "string to append to Evidence column (comma-separated)",
      "last_tested": "YYYY-MM-DD",
      "syllabus_topic": "optional — concepts table only"
    }
  ],
  "weakness_queue": [
    {
      "priority": 1,
      "item": "string",
      "type": "concept | chain | factor",
      "phase": "Dot | Linear | Network",
      "severity": "string",
      "source": "string",
      "added": "YYYY-MM-DD"
    }
  ],
  "syllabus_updates": [
    {
      "topic": "string — must match existing syllabus line",
      "status": "checked | unchecked"
    }
  ],
  "section_rewrites": {
    "compressed_model": "COMPLETE replacement text for ## Compressed Model",
    "open_questions": "COMPLETE replacement text for ## Open Questions",
    "interleave_pool": "COMPLETE replacement text for ## Interleave Pool"
  },
  "subsection_rewrites": {
    "calibration_trend": "COMPLETE replacement text for ### Calibration Trend"
  },
  "section_appends": {
    "calibration_concept": "pipe-delimited table row (no header) to append to ### Concept-Level Confidence",
    "calibration_gate": "pipe-delimited table row (no header) to append to ### Gate Predictions",
    "load_session_history": "pipe-delimited table row (no header) to append to ### Session History"
  },
  "load_baseline": {
    "working_batch_size": "number — observed working batch size",
    "hint_tolerance": "string — e.g. 'low (needs <=1 hint per concept)'",
    "recovery_pattern": "string — e.g. 'responds well to different analogies'"
  },
  "engagement": {
    "momentum": "positive | neutral | negative",
    "consecutive_struggles": 0,
    "last_celebration": "string",
    "notes": "string"
  }
}
\```

## Rules

- **`section_rewrites`** targets `##`-level headers. Values must be COMPLETE replacement content — not deltas. The script replaces everything between the header and the next `##` header. Partial content causes data loss.
- **`subsection_rewrites`** targets `###`-level headers. Same complete-replacement semantics, but replacement ends at the next `###` or `##` header.
- **`weakness_queue`** is a full rewrite — the script replaces the entire queue, not a merge.
- **`syllabus_updates`** is consolidated into the merge payload by the normalizer. In the pre-script dln-sync, this was a standalone input field — now it's part of the merge JSON.
- **`mastery_updates`** never delete rows. Existing rows are upserted; new rows are appended.

## Field Reference

| Field | Target Section | Operation |
|-------|---------------|-----------|
| `mastery_updates` | `## Concepts`, `## Chains`, `## Factors` | Row upsert |
| `weakness_queue` | `## Weakness Queue` | Full rewrite |
| `syllabus_updates` | `## Syllabus` | Checkbox toggle |
| `section_rewrites.compressed_model` | `## Compressed Model` | Full replace |
| `section_rewrites.open_questions` | `## Open Questions` | Full replace |
| `section_rewrites.interleave_pool` | `## Interleave Pool` | Full replace |
| `subsection_rewrites.calibration_trend` | `### Calibration Trend` | Full replace |
| `section_appends.calibration_concept` | `### Concept-Level Confidence` | Row append |
| `section_appends.calibration_gate` | `### Gate Predictions` | Row append |
| `section_appends.load_session_history` | `### Session History` | Row append |
| `load_baseline` | `### Baseline` | Key-value update |
| `engagement` | `## Engagement Signals` | Key-value update |
```

- [ ] **Step 2: Commit**

```bash
git add dunk/references/merge-payload-schema.md
git commit -m "feat(dln): add merge payload schema document"
```

---

## Task 2: Build the merge script — KS block parser

**Files:**
- Create: `dunk/scripts/pyproject.toml`
- Create: `dunk/scripts/ks-merge.py`
- Create: `dunk/scripts/tests/test_ks_merge.py`

This task builds the foundation: CLI argument handling, JSON payload parsing, KS block section parsing, and the reassembly logic. No merge operations yet — just parse and reconstruct unchanged.

- [ ] **Step 1: Initialize uv project for the scripts directory**

```bash
cd /Users/luqman/Desktop/projects/my-cc-plugin/ashford/dunk/scripts && uv init --python 3.13 --no-readme
```

Then add pytest as a dev dependency:

```bash
cd /Users/luqman/Desktop/projects/my-cc-plugin/ashford/dunk/scripts && uv add --dev pytest
```

Verify the `pyproject.toml` has `requires-python = ">=3.13"` and pytest is in dev dependencies.

- [ ] **Step 2: Write test for round-trip parse and reassemble**

Create `dunk/scripts/tests/test_ks_merge.py`. This test verifies that parsing the init-template KS block into sections and reassembling produces identical output.

```python
"""Tests for ks-merge.py."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "ks-merge.py"
INIT_TEMPLATE = Path(__file__).resolve().parent.parent.parent / "skills" / "dln" / "references" / "init-template.md"


def run_merge(payload: dict, ks_block: str) -> subprocess.CompletedProcess:
    """Helper: write payload and KS block to temp files, run ks-merge.py."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as pf:
        json.dump(payload, pf)
        payload_path = pf.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as kf:
        kf.write(ks_block)
        ks_path = kf.name
    return subprocess.run(
        [sys.executable, str(SCRIPT), payload_path, ks_path],
        capture_output=True, text=True,
    )


def test_empty_payload_round_trip():
    """Empty payload should produce identical output to input."""
    ks_block = INIT_TEMPLATE.read_text()
    result = run_merge({}, ks_block)
    assert result.returncode == 0
    assert result.stdout == ks_block


def test_malformed_json_exits_1():
    """Malformed JSON payload should exit 1."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as pf:
        pf.write("not json{{{")
        payload_path = pf.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as kf:
        kf.write("<!-- KS:start -->\n# Knowledge State\n<!-- KS:end -->")
        ks_path = kf.name
    result = subprocess.run(
        [sys.executable, str(SCRIPT), payload_path, ks_path],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "json" in result.stderr.lower() or "parse" in result.stderr.lower()
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /Users/luqman/Desktop/projects/my-cc-plugin/ashford/dunk/scripts && uv run pytest tests/test_ks_merge.py -v
```

Expected: FAIL — `ks-merge.py` doesn't exist yet.

- [ ] **Step 4: Write the script skeleton**

Create `dunk/scripts/ks-merge.py` with argument parsing, JSON loading, KS block section parsing, and reassembly. No merge operations yet.

```python
#!/usr/bin/env python3
"""Deterministic KS merge script.

Takes a typed JSON payload and a raw KS block (markdown between markers),
applies updates deterministically, and outputs the merged KS block to stdout.

Usage:
    uv run python ks-merge.py [--dry-run] <payload_path> <ks_block_path>

Exit codes:
    0 — success (merged block on stdout)
    1 — failure (error on stderr)
"""
import json
import re
import sys
from dataclasses import dataclass


KS_START = "<!-- KS:start -->"
KS_END = "<!-- KS:end -->"


@dataclass
class Section:
    """A parsed section of the KS block."""

    header: str  # The full header line (e.g., "## Concepts")
    level: int   # Header level (2 for ##, 3 for ###)
    body: str    # Everything after the header until the next section


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
        - preamble: text before the first ## header (includes <!-- KS:start --> and # Knowledge State)
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


def main():
    """Entry point."""
    dry_run, payload_path, ks_path = parse_args()
    payload = load_payload(payload_path)
    ks_block = load_ks_block(ks_path)

    preamble, sections, postamble = parse_sections(ks_block)

    # TODO: Apply merge operations here (Tasks 3-5)

    merged = reassemble(preamble, sections, postamble)
    sys.stdout.write(merged)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /Users/luqman/Desktop/projects/my-cc-plugin/ashford/dunk/scripts && uv run pytest tests/test_ks_merge.py -v
```

Expected: both tests PASS. The round-trip test may need adjustment if reassembly adds/removes trailing newlines — fix until the round-trip is exact.

- [ ] **Step 6: Commit**

```bash
git add dunk/scripts/ks-merge.py dunk/scripts/tests/test_ks_merge.py dunk/scripts/pyproject.toml dunk/scripts/uv.lock
git commit -m "feat(dln): add ks-merge.py skeleton with parser, round-trip tests, and uv project"
```

---

## Task 3: Merge operations — mastery tables

**Files:**
- Modify: `dunk/scripts/ks-merge.py`
- Modify: `dunk/scripts/tests/test_ks_merge.py`

- [ ] **Step 1: Write tests for mastery table operations**

Add to `test_ks_merge.py`:

```python
# A populated KS block for testing (add at module level)
POPULATED_KS = """\
<!-- KS:start -->
# Knowledge State

## Syllabus
Goal: Learn options pricing
- [x] Options Basics
- [ ] Greeks
- [ ] Volatility

## Concepts

| Concept | Status | Syllabus Topic | Evidence | Last Tested |
|---------|--------|----------------|----------|-------------|
| Put-Call Parity | partial | Options Basics | Comprehension check pass (S2) | 2026-03-14 |
| Intrinsic Value | mastered | Options Basics | Recall pass (S1), Recall pass (S2) | 2026-03-14 |

## Chains

| Chain | Status | Evidence | Last Tested |
|-------|--------|----------|-------------|
| Pricing → Parity → Synthetics | partial | Chain trace fail (S2) | 2026-03-14 |

## Factors

| Factor | Status | Evidence | Last Tested |
|--------|--------|----------|-------------|

## Compressed Model

## Interleave Pool

## Calibration Log

### Concept-Level Confidence
| Concept | Self-Rating (1-5) | Actual Performance | Gap | Date |
|---------|-------------------|-------------------|-----|------|

### Gate Predictions
| Phase Gate | Predicted Outcome | Actual Outcome | Date |
|------------|------------------|----------------|------|

### Calibration Trend

## Load Profile

### Baseline
- Observed working batch size: 2
- Hint tolerance: low (needs <=1 hint per concept)
- Recovery pattern: responds well to different analogies

### Session History
| Session | Avg Batch Size | Overload Signals | Adjustments Made |
|---------|---------------|------------------|-----------------|

## Open Questions

## Weakness Queue

| Priority | Item | Type | Phase | Severity | Source | Added |
|----------|------|------|-------|----------|--------|-------|
| 1 | Greeks intuition | concept | Dot | not-mastered | S2 gap | 2026-03-14 |

## Engagement Signals

- Momentum: neutral
- Consecutive struggles: 0
- Last celebration: none
- Notes:
<!-- KS:end -->
"""


def test_mastery_update_existing_row():
    """Update an existing concept row — status and evidence change."""
    payload = {
        "mastery_updates": [
            {
                "table": "concepts",
                "name": "Put-Call Parity",
                "status": "mastered",
                "evidence": "Recall pass (S3)",
                "last_tested": "2026-03-16",
            }
        ]
    }
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    # Check the row was updated
    assert "| Put-Call Parity | mastered |" in result.stdout
    assert "Comprehension check pass (S2), Recall pass (S3)" in result.stdout
    assert "2026-03-16" in result.stdout
    # Intrinsic Value row should be untouched
    assert "| Intrinsic Value | mastered |" in result.stdout


def test_mastery_add_new_row():
    """Add a new concept that doesn't exist in the table."""
    payload = {
        "mastery_updates": [
            {
                "table": "concepts",
                "name": "Delta",
                "status": "not-mastered",
                "evidence": "Introduced (S3)",
                "last_tested": "2026-03-16",
                "syllabus_topic": "Greeks",
            }
        ]
    }
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    assert "| Delta | not-mastered | Greeks | Introduced (S3) | 2026-03-16 |" in result.stdout
    # Existing rows untouched
    assert "| Put-Call Parity | partial |" in result.stdout


def test_mastery_update_chain():
    """Update a chain row."""
    payload = {
        "mastery_updates": [
            {
                "table": "chains",
                "name": "Pricing → Parity → Synthetics",
                "status": "mastered",
                "evidence": "Chain trace pass (S3)",
                "last_tested": "2026-03-16",
            }
        ]
    }
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    assert "| Pricing → Parity → Synthetics | mastered |" in result.stdout
    assert "Chain trace fail (S2), Chain trace pass (S3)" in result.stdout


def test_mastery_add_new_factor():
    """Add a new factor to an empty factors table."""
    payload = {
        "mastery_updates": [
            {
                "table": "factors",
                "name": "Replication Principle",
                "status": "partial",
                "evidence": "Factor hypothesis (S5)",
                "last_tested": "2026-03-16",
            }
        ]
    }
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    assert "| Replication Principle | partial | Factor hypothesis (S5) | 2026-03-16 |" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/luqman/Desktop/projects/my-cc-plugin/ashford/dunk/scripts && uv run pytest tests/test_ks_merge.py -v -k mastery
```

Expected: FAIL — merge operations not implemented yet.

- [ ] **Step 3: Implement mastery table merging**

Add to `ks-merge.py`:

```python
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


def parse_table_rows(body: str) -> tuple:
    """Parse pipe-delimited table rows from a section body.

    Returns (header_lines, data_rows) where:
    - header_lines: the header row + separator row as a single string
    - data_rows: list of dicts mapping column name → value
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
        header_lines, columns, data_rows = parse_table_rows(section.body)
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
                    msgs.append(f"status {old_status}→{new_status}")
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
```

Then wire it into `main()`:

```python
    messages = []

    if "mastery_updates" in payload:
        messages.extend(apply_mastery_updates(sections, payload["mastery_updates"], dry_run))
```

And at the end of `main()`, before `reassemble`:

```python
    if dry_run:
        sys.stdout.write("\n".join(messages) + "\n")
        return
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/luqman/Desktop/projects/my-cc-plugin/ashford/dunk/scripts && uv run pytest tests/test_ks_merge.py -v -k mastery
```

Expected: all 4 mastery tests PASS.

- [ ] **Step 5: Commit**

```bash
git add dunk/scripts/ks-merge.py dunk/scripts/tests/test_ks_merge.py
git commit -m "feat(dln): implement mastery table merging in ks-merge.py"
```

---

## Task 4: Merge operations — weakness queue, syllabus, section rewrites

**Files:**
- Modify: `dunk/scripts/ks-merge.py`
- Modify: `dunk/scripts/tests/test_ks_merge.py`

- [ ] **Step 1: Write tests**

Add to `test_ks_merge.py`:

```python
def test_weakness_queue_full_rewrite():
    """Weakness queue should be completely replaced."""
    payload = {
        "weakness_queue": [
            {"priority": 1, "item": "Vega sensitivity", "type": "concept", "phase": "Dot", "severity": "not-mastered", "source": "S3 gap", "added": "2026-03-16"},
            {"priority": 2, "item": "Delta hedging", "type": "chain", "phase": "Dot", "severity": "partial", "source": "S3 check", "added": "2026-03-16"},
        ]
    }
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    # Old entry should be gone
    assert "Greeks intuition" not in result.stdout
    # New entries should be present
    assert "Vega sensitivity" in result.stdout
    assert "Delta hedging" in result.stdout


def test_syllabus_toggle_checked():
    """Toggle an unchecked syllabus item to checked."""
    payload = {
        "syllabus_updates": [{"topic": "Greeks", "status": "checked"}]
    }
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    assert "- [x] Greeks" in result.stdout
    # Options Basics should stay checked
    assert "- [x] Options Basics" in result.stdout
    # Volatility should stay unchecked
    assert "- [ ] Volatility" in result.stdout


def test_syllabus_toggle_unchecked():
    """Toggle a checked syllabus item to unchecked."""
    payload = {
        "syllabus_updates": [{"topic": "Options Basics", "status": "unchecked"}]
    }
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    assert "- [ ] Options Basics" in result.stdout


def test_syllabus_missing_topic_warns():
    """Missing topic should warn to stderr, not error."""
    payload = {
        "syllabus_updates": [{"topic": "Nonexistent Topic", "status": "checked"}]
    }
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    assert "nonexistent" in result.stderr.lower() or "not found" in result.stderr.lower()


def test_section_rewrite_compressed_model():
    """Replace ## Compressed Model content."""
    payload = {
        "section_rewrites": {
            "compressed_model": "Options are arbitrage-enforced replication."
        }
    }
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    assert "Options are arbitrage-enforced replication." in result.stdout
    # Other sections untouched
    assert "## Interleave Pool" in result.stdout


def test_section_rewrite_open_questions():
    """Replace ## Open Questions content."""
    payload = {
        "section_rewrites": {
            "open_questions": "- How does vol smile arise?\n- Why do OTM puts cost more?"
        }
    }
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    assert "How does vol smile arise?" in result.stdout
    assert "Why do OTM puts cost more?" in result.stdout


def test_subsection_rewrite_calibration_trend():
    """Replace ### Calibration Trend content."""
    payload = {
        "subsection_rewrites": {
            "calibration_trend": "Overconfident by 1.2 points on average. Improving."
        }
    }
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    assert "Overconfident by 1.2 points" in result.stdout
    # Concept-Level Confidence and Gate Predictions should be untouched
    assert "### Concept-Level Confidence" in result.stdout
    assert "### Gate Predictions" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/luqman/Desktop/projects/my-cc-plugin/ashford/dunk/scripts && uv run pytest tests/test_ks_merge.py -v -k "weakness or syllabus or section_rewrite or subsection_rewrite"
```

Expected: FAIL.

- [ ] **Step 3: Implement weakness queue rewrite**

Add to `ks-merge.py`:

```python
WEAKNESS_COLUMNS = ["Priority", "Item", "Type", "Phase", "Severity", "Source", "Added"]
WEAKNESS_KEY_MAP = {
    "priority": "Priority", "item": "Item", "type": "Type",
    "phase": "Phase", "severity": "Severity", "source": "Source", "added": "Added",
}


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
```

- [ ] **Step 4: Implement syllabus checkbox toggling**

Add to `ks-merge.py`:

```python
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
```

- [ ] **Step 5: Implement section_rewrites and subsection_rewrites**

Add to `ks-merge.py`:

```python
SECTION_REWRITE_MAP = {
    "compressed_model": "Compressed Model",
    "open_questions": "Open Questions",
    "interleave_pool": "Interleave Pool",
}

SUBSECTION_REWRITE_MAP = {
    "calibration_trend": "Calibration Trend",
}


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
            messages.append(f'[rewrite] REPLACE ## {section_name}')
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
            messages.append(f'[rewrite] REPLACE ### {section_name}')
        else:
            sections[idx].body = "\n" + content + "\n"
    return messages
```

Wire all four into `main()`:

```python
    if "weakness_queue" in payload:
        messages.extend(apply_weakness_queue(sections, payload["weakness_queue"], dry_run))
    if "syllabus_updates" in payload:
        messages.extend(apply_syllabus_updates(sections, payload["syllabus_updates"], dry_run))
    if "section_rewrites" in payload:
        messages.extend(apply_section_rewrites(sections, payload["section_rewrites"], dry_run))
    if "subsection_rewrites" in payload:
        messages.extend(apply_subsection_rewrites(sections, payload["subsection_rewrites"], dry_run))
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd /Users/luqman/Desktop/projects/my-cc-plugin/ashford/dunk/scripts && uv run pytest tests/test_ks_merge.py -v -k "weakness or syllabus or section_rewrite or subsection_rewrite"
```

Expected: all 7 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add dunk/scripts/ks-merge.py dunk/scripts/tests/test_ks_merge.py
git commit -m "feat(dln): implement weakness queue, syllabus, and section rewrite operations"
```

---

## Task 5: Merge operations — section appends, load baseline, engagement

**Files:**
- Modify: `dunk/scripts/ks-merge.py`
- Modify: `dunk/scripts/tests/test_ks_merge.py`

- [ ] **Step 1: Write tests**

Add to `test_ks_merge.py`:

```python
def test_section_append_calibration_concept():
    """Append a row to ### Concept-Level Confidence table."""
    payload = {
        "section_appends": {
            "calibration_concept": "| Put-Call Parity | 4 | pass | -1 | 2026-03-16 |"
        }
    }
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    assert "| Put-Call Parity | 4 | pass | -1 | 2026-03-16 |" in result.stdout
    # Table header should still be there
    assert "| Concept | Self-Rating (1-5) |" in result.stdout


def test_section_append_load_session_history():
    """Append a row to ### Session History table."""
    payload = {
        "section_appends": {
            "load_session_history": "| 3 | 3 | none | batch +1 |"
        }
    }
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    assert "| 3 | 3 | none | batch +1 |" in result.stdout


def test_load_baseline_update():
    """Update ### Baseline key-value pairs."""
    payload = {
        "load_baseline": {
            "working_batch_size": 3,
            "hint_tolerance": "medium (needs <=2 hints per concept)",
        }
    }
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    assert "- Observed working batch size: 3" in result.stdout
    assert "- Hint tolerance: medium (needs <=2 hints per concept)" in result.stdout
    # Recovery pattern should be unchanged
    assert "- Recovery pattern: responds well to different analogies" in result.stdout


def test_engagement_update():
    """Update ## Engagement Signals key-value pairs."""
    payload = {
        "engagement": {
            "momentum": "positive",
            "consecutive_struggles": 0,
            "last_celebration": "Mastered Put-Call Parity (S3)",
        }
    }
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    assert "- Momentum: positive" in result.stdout
    assert "- Consecutive struggles: 0" in result.stdout
    assert "- Last celebration: Mastered Put-Call Parity (S3)" in result.stdout
    # Notes should be unchanged
    assert "- Notes:" in result.stdout


def test_engagement_partial_update():
    """Update only some engagement fields — others stay unchanged."""
    payload = {
        "engagement": {"momentum": "negative"}
    }
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    assert "- Momentum: negative" in result.stdout
    assert "- Consecutive struggles: 0" in result.stdout  # unchanged
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/luqman/Desktop/projects/my-cc-plugin/ashford/dunk/scripts && uv run pytest tests/test_ks_merge.py -v -k "append or baseline or engagement"
```

Expected: FAIL.

- [ ] **Step 3: Implement section_appends**

Add to `ks-merge.py`:

```python
SECTION_APPEND_MAP = {
    "calibration_concept": "Concept-Level Confidence",
    "calibration_gate": "Gate Predictions",
    "load_session_history": "Session History",
}


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
            messages.append(f'[append] ADD row to ### {section_name}')
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
```

- [ ] **Step 4: Implement load_baseline and engagement**

Add to `ks-merge.py`:

```python
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


def apply_key_value_updates(
    sections: list, updates: dict, key_map: dict, section_name: str, dry_run: bool,
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
            print(f"Warning: unknown key '{json_key}' for {section_name}, skipping", file=sys.stderr)
            continue

        found = False
        for i, line in enumerate(lines):
            if line.strip().startswith(f"- {display_key}:"):
                if dry_run:
                    old_val = line.split(":", 1)[1].strip() if ":" in line else ""
                    messages.append(f"[{section_name.lower()}] {display_key}: {old_val}→{value}")
                else:
                    lines[i] = f"- {display_key}: {value}"
                found = True
                break
        if not found:
            print(f"Warning: key '- {display_key}:' not found in {section_name}, skipping", file=sys.stderr)

    if not dry_run:
        sections[idx].body = "\n".join(lines)
    return messages
```

Wire into `main()`:

```python
    if "section_appends" in payload:
        messages.extend(apply_section_appends(sections, payload["section_appends"], dry_run))
    if "load_baseline" in payload:
        messages.extend(apply_key_value_updates(
            sections, payload["load_baseline"], LOAD_BASELINE_MAP, "Baseline", dry_run,
        ))
    if "engagement" in payload:
        messages.extend(apply_key_value_updates(
            sections, payload["engagement"], ENGAGEMENT_MAP, "Engagement Signals", dry_run,
        ))
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /Users/luqman/Desktop/projects/my-cc-plugin/ashford/dunk/scripts && uv run pytest tests/test_ks_merge.py -v -k "append or baseline or engagement"
```

Expected: all 5 tests PASS.

- [ ] **Step 6: Run full test suite**

```bash
cd /Users/luqman/Desktop/projects/my-cc-plugin/ashford/dunk/scripts && uv run pytest tests/test_ks_merge.py -v
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add dunk/scripts/ks-merge.py dunk/scripts/tests/test_ks_merge.py
git commit -m "feat(dln): implement section appends, load baseline, and engagement updates"
```

---

## Task 6: Dry-run mode and edge case tests

**Files:**
- Modify: `dunk/scripts/ks-merge.py`
- Modify: `dunk/scripts/tests/test_ks_merge.py`

- [ ] **Step 1: Write tests**

Add to `test_ks_merge.py`:

```python
def run_merge_dry(payload: dict, ks_block: str) -> subprocess.CompletedProcess:
    """Helper: run ks-merge.py with --dry-run."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as pf:
        json.dump(payload, pf)
        payload_path = pf.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as kf:
        kf.write(ks_block)
        ks_path = kf.name
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run", payload_path, ks_path],
        capture_output=True, text=True,
    )


def test_dry_run_shows_operations():
    """Dry-run should output human-readable change summary."""
    payload = {
        "mastery_updates": [
            {"table": "concepts", "name": "Put-Call Parity", "status": "mastered", "evidence": "Recall pass (S3)", "last_tested": "2026-03-16"},
            {"table": "concepts", "name": "Delta", "status": "not-mastered", "evidence": "Introduced (S3)", "last_tested": "2026-03-16", "syllabus_topic": "Greeks"},
        ],
        "weakness_queue": [
            {"priority": 1, "item": "Vega", "type": "concept", "phase": "Dot", "severity": "not-mastered", "source": "S3", "added": "2026-03-16"},
        ],
        "syllabus_updates": [{"topic": "Greeks", "status": "checked"}],
        "engagement": {"momentum": "positive"},
    }
    result = run_merge_dry(payload, POPULATED_KS)
    assert result.returncode == 0
    assert "[mastery] UPDATE" in result.stdout
    assert "[mastery] ADD" in result.stdout
    assert "[weakness] REWRITE" in result.stdout
    assert "[syllabus] CHECK" in result.stdout


def test_dry_run_does_not_modify():
    """Dry-run output should NOT contain the KS block."""
    payload = {
        "mastery_updates": [
            {"table": "concepts", "name": "Put-Call Parity", "status": "mastered", "evidence": "Recall pass (S3)", "last_tested": "2026-03-16"},
        ]
    }
    result = run_merge_dry(payload, POPULATED_KS)
    assert result.returncode == 0
    # Should NOT contain KS markers or full KS content
    assert "<!-- KS:start -->" not in result.stdout
    assert "## Concepts" not in result.stdout


def test_missing_section_still_succeeds():
    """Targeting a nonexistent section should warn but not fail."""
    minimal_ks = "<!-- KS:start -->\n# Knowledge State\n\n## Concepts\n\n| Concept | Status | Syllabus Topic | Evidence | Last Tested |\n|---------|--------|----------------|----------|-------------|\n\n<!-- KS:end -->\n"
    payload = {
        "mastery_updates": [
            {"table": "factors", "name": "Test", "status": "partial", "evidence": "test", "last_tested": "2026-03-16"}
        ]
    }
    result = run_merge(payload, minimal_ks)
    assert result.returncode == 0
    assert "warning" in result.stderr.lower() or "not found" in result.stderr.lower()
    # The output should still be valid with markers
    assert "<!-- KS:start -->" in result.stdout
    assert "<!-- KS:end -->" in result.stdout


def test_combined_operations():
    """All operations in a single payload should work together."""
    payload = {
        "mastery_updates": [
            {"table": "concepts", "name": "Put-Call Parity", "status": "mastered", "evidence": "Recall pass (S3)", "last_tested": "2026-03-16"},
        ],
        "weakness_queue": [
            {"priority": 1, "item": "Delta", "type": "concept", "phase": "Dot", "severity": "not-mastered", "source": "S3", "added": "2026-03-16"},
        ],
        "syllabus_updates": [{"topic": "Greeks", "status": "checked"}],
        "section_rewrites": {"compressed_model": "Options = replication."},
        "engagement": {"momentum": "positive", "consecutive_struggles": 0},
    }
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    assert "| Put-Call Parity | mastered |" in result.stdout
    assert "Greeks intuition" not in result.stdout  # old weakness gone
    assert "| Delta |" in result.stdout  # new weakness
    assert "- [x] Greeks" in result.stdout
    assert "Options = replication." in result.stdout
    assert "- Momentum: positive" in result.stdout
```

- [ ] **Step 2: Run tests**

```bash
cd /Users/luqman/Desktop/projects/my-cc-plugin/ashford/dunk/scripts && uv run pytest tests/test_ks_merge.py -v
```

Expected: all tests PASS. Fix any failures in the dry-run wiring or edge case handling.

- [ ] **Step 3: Commit**

```bash
git add dunk/scripts/ks-merge.py dunk/scripts/tests/test_ks_merge.py
git commit -m "feat(dln): add dry-run mode and edge case tests for ks-merge.py"
```

---

## Task 7: Update dln-sync agent

**Files:**
- Modify: `dunk/agents/dln-sync.md`

This task rewires dln-sync's MERGE step to use the normalizer + script flow instead of manual string manipulation.

- [ ] **Step 1: Add Bash tool to dln-sync frontmatter**

The agent needs the Bash tool to call `ks-merge.py`. In `dunk/agents/dln-sync.md`, add `Bash` to the tools list (line 13-17):

```yaml
tools:
  - mcp__plugin_Notion_notion__notion-fetch
  - mcp__plugin_Notion_notion__notion-update-page
  - mcp__plugin_Notion_notion__notion-search
  - mcp__plugin_Notion_notion__notion-query-database-view
  - Bash
```

- [ ] **Step 2: Add the Normalizer Schema section**

Insert a new section between "## Input Format" (ends at line 55) and "## Execution Protocol" (line 57). This section contains the schema verbatim from `merge-payload-schema.md` plus the two few-shot examples from the spec.

The section should be titled `## Normalizer Schema` and contain:
1. The instruction: "After FETCH (Step 1), normalize the prose dispatch payload into this JSON structure. All fields are optional — only include fields that have updates."
2. The full JSON schema from `dunk/references/merge-payload-schema.md`
3. The two normalizer examples from the spec (Dot phase and Network phase — spec lines 255-315)
4. A note: "The normalizer receives BOTH the prose dispatch AND the fetched KS block. Full-rewrite fields (weakness_queue, section_rewrites) require looking up existing values in the KS block to produce COMPLETE replacement content."

- [ ] **Step 3: Rewrite Step 2 MERGE in the Execution Protocol**

Replace the current Step 2 (line 73-80):

```markdown
#### Step 2: MERGE

Apply the deltas from `write_payload` to the fetched KS snapshot. See the Merging Rules section below for details.

The result is the **merged KS block**. It must:
- Start with `<!-- KS:start -->` (unescaped)
- End with `<!-- KS:end -->` (unescaped)
- Contain all existing KS content with deltas applied
```

With:

```markdown
#### Step 2a: NORMALIZE

Produce a JSON object conforming to the Normalizer Schema section above. You have access to both the prose dispatch payload and the fetched KS block from Step 1.

Write the JSON to a temp file and the KS block to another temp file:

```bash
# Write the JSON payload
cat > /tmp/ks-merge-payload-<page_id_8chars>-$$-$(date +%s).json << 'PAYLOAD_EOF'
<your JSON here>
PAYLOAD_EOF

# Write the KS block
cat > /tmp/ks-merge-ks-<page_id_8chars>-$$-$(date +%s).md << 'KS_EOF'
<KS block from Step 1>
KS_EOF
```

#### Step 2b: MERGE

Call the merge script:

```bash
uv run python "${CLAUDE_PLUGIN_ROOT}/scripts/ks-merge.py" /tmp/ks-merge-payload-<...>.json /tmp/ks-merge-ks-<...>.md
```

If exit 0: the stdout is the merged KS block. Use it as `new_str` in Step 3 (REPLACE).

If exit 1: **hard fail.**
1. Read stderr for the error message.
2. Set `Status.Write` to `failed` in the re-anchor payload.
3. Include `Merge error: "<stderr>"` and the temp file paths in `Debug artifacts`.
4. Queue the writes for the next boundary.
5. Skip to compression — do NOT attempt manual merge.
```

- [ ] **Step 4: Update the Merging Rules section**

Add a note at the top of the Merging Rules section (line 152):

```markdown
## Merging Rules

> **Note:** These rules are implemented by `ks-merge.py` and executed deterministically. The agent does NOT apply these rules manually — they are documented here as the source of truth for the script's behavior and for maintenance reference.
```

- [ ] **Step 5: Update Input Format — annotate syllabus_updates**

In the Input Format section (lines 53-55), add a note that `syllabus_updates` is now folded into the merge payload JSON by the normalizer:

```markdown
- **syllabus_updates**: (optional) List of syllabus topic status changes. Each entry has:
  - `topic`: The syllabus topic name (must match a `- [ ]` or `- [x]` line in `## Syllabus`)
  - `status`: `checked` (all related concepts mastered) or `unchecked` (not all mastered)
  > **Note:** The normalizer consolidates `syllabus_updates` into the merge payload JSON. The merge script handles checkbox toggling — this field no longer bypasses the merge step.
```

- [ ] **Step 6: Add temp file cleanup to VERIFY step**

In Step 4 (VERIFY), after the "On verify failure" block, add:

```markdown
**On verify success:**
Clean up temp files from Steps 2a-2b:

```bash
rm -f /tmp/ks-merge-payload-<...>.json /tmp/ks-merge-ks-<...>.md
```

Do NOT clean up on failure — temp files persist for manual inspection.
```

- [ ] **Step 7: Update Error Handling section**

Add after the existing "If a marker format violation is blocked" block (line 199):

```markdown
If the merge script fails (exit 1):
- The script's stderr contains the error message.
- Temp files persist at `/tmp/ks-merge-*` for manual inspection.
- Include in the re-anchor payload:
  ```
  ### Status
  - Write: failed
  - Failed writes: [list of intended updates from the dispatch]
  - Merge error: "<stderr message from ks-merge.py>"
  - Debug artifacts:
    - Payload: /tmp/ks-merge-payload-<page_id>-<pid>-<timestamp>.json
    - KS block: /tmp/ks-merge-ks-<page_id>-<pid>-<timestamp>.md
  ```
- Do NOT attempt manual merge as fallback — queue writes for the next boundary.
- Session log appends are still attempted — they are independent of the KS merge.
```

- [ ] **Step 8: Read back the full file and verify coherence**

Read `dunk/agents/dln-sync.md` end-to-end. Confirm:
- Bash is in the tools list
- Normalizer Schema section is between Input Format and Execution Protocol
- Step 2 references the normalizer schema and the script
- Merging Rules have the "implemented by script" note
- Error Handling covers merge script failure with the full diagnostics template
- Input Format has the `syllabus_updates` consolidation note
- VERIFY step includes temp file cleanup on success
- No duplicate sections
- MARKER RULE is still present and unchanged

- [ ] **Step 9: Commit**

```bash
git add dunk/agents/dln-sync.md
git commit -m "feat(dln): rewire dln-sync MERGE step to use normalizer + ks-merge.py

Replaces manual LLM string manipulation with:
- Step 2a: normalize prose dispatch → typed JSON (1 LLM turn)
- Step 2b: pipe JSON + KS block to ks-merge.py (deterministic)

On script failure: hard fail, queue writes, persist temp files for
manual diagnosis. No LLM fallback."
```

---

## Task 8: Integration verification

**Files:** None (verification only)

- [ ] **Step 1: Run the full test suite**

```bash
cd /Users/luqman/Desktop/projects/my-cc-plugin/ashford/dunk/scripts && uv run pytest tests/test_ks_merge.py -v
```

Expected: all tests PASS.

- [ ] **Step 2: Manual dry-run test with realistic payload**

Create a realistic test payload and run dry-run to verify the output looks correct:

```bash
cd /Users/luqman/Desktop/projects/my-cc-plugin/ashford

cat > /tmp/test-payload.json << 'EOF'
{
  "mastery_updates": [
    {"table": "concepts", "name": "Put-Call Parity", "status": "mastered", "evidence": "Recall pass (S3)", "last_tested": "2026-03-16"},
    {"table": "concepts", "name": "Delta", "status": "not-mastered", "evidence": "Introduced (S3)", "last_tested": "2026-03-16", "syllabus_topic": "Greeks"}
  ],
  "weakness_queue": [
    {"priority": 1, "item": "Delta hedging", "type": "concept", "phase": "Dot", "severity": "not-mastered", "source": "S3 gap", "added": "2026-03-16"}
  ],
  "syllabus_updates": [{"topic": "Greeks", "status": "checked"}],
  "section_rewrites": {"compressed_model": "Options = arbitrage-enforced replication."},
  "engagement": {"momentum": "positive", "consecutive_struggles": 0, "last_celebration": "Mastered PCP (S3)"}
}
EOF

cd /Users/luqman/Desktop/projects/my-cc-plugin/ashford/dunk/scripts && uv run python ks-merge.py --dry-run /tmp/test-payload.json ../skills/dln/references/init-template.md
```

Expected: human-readable summary showing UPDATE/ADD mastery ops, REWRITE weakness, CHECK syllabus, REPLACE compressed model, engagement updates.

- [ ] **Step 3: Manual merge test with init-template**

Run the same payload without `--dry-run`:

```bash
cd /Users/luqman/Desktop/projects/my-cc-plugin/ashford/dunk/scripts && uv run python ks-merge.py /tmp/test-payload.json ../skills/dln/references/init-template.md
```

Expected: full merged KS block on stdout with:
- Put-Call Parity row updated to mastered with new evidence
- Delta row added to Concepts table
- Weakness queue has only "Delta hedging"
- Greeks syllabus item checked (note: init-template has no syllabus items, so this will warn — that's expected)
- Compressed Model has new text
- Engagement signals updated
- Markers present

- [ ] **Step 4: Verify dln-sync agent file is coherent**

Read `dunk/agents/dln-sync.md` and confirm:
- The flow is: FETCH → NORMALIZE (produce JSON) → MERGE (call script) → REPLACE → VERIFY → COMPRESS
- All section references are consistent
- The normalizer schema matches `dunk/references/merge-payload-schema.md`

- [ ] **Step 5: Clean up test artifacts**

```bash
rm -f /tmp/test-payload.json
```
