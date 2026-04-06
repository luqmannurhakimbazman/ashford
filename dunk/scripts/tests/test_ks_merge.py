"""Tests for ks-merge.py."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "ks-merge.py"
INIT_TEMPLATE = (
    Path(__file__).resolve().parent.parent.parent
    / "skills"
    / "dln"
    / "references"
    / "init-template.md"
)


def run_merge(
    payload: dict, ks_block: str, extra_args: list[str] | None = None
) -> subprocess.CompletedProcess:
    """Helper: write payload and KS block to temp files, run ks-merge.py."""
    args = extra_args or []
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as pf:
        json.dump(payload, pf)
        payload_path = pf.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as kf:
        kf.write(ks_block)
        ks_path = kf.name
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, payload_path, ks_path],
        capture_output=True,
        text=True,
    )


def run_merge_dry(
    payload: dict, ks_block: str, extra_args: list[str] | None = None
) -> subprocess.CompletedProcess:
    """Helper: run ks-merge.py with --dry-run."""
    args = extra_args or []
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as pf:
        json.dump(payload, pf)
        payload_path = pf.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as kf:
        kf.write(ks_block)
        ks_path = kf.name
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run", *args, payload_path, ks_path],
        capture_output=True,
        text=True,
    )


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
| Pricing \u2192 Parity \u2192 Synthetics | partial | Chain trace fail (S2) | 2026-03-14 |

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


# === Task 2: Round-trip and malformed JSON tests ===


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
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "json" in result.stderr.lower() or "parse" in result.stderr.lower()


def test_unknown_top_level_key_fails_strict():
    """Unknown top-level payload keys should fail in strict mode."""
    ks_block = INIT_TEMPLATE.read_text()
    result = run_merge({"unknown_top": "x"}, ks_block)
    assert result.returncode == 1
    assert "unknown top-level" in result.stderr.lower()


def test_missing_required_mastery_field_fails_cleanly():
    """Missing required mastery fields should fail with clear error (no traceback)."""
    payload = {"mastery_updates": [{"table": "concepts", "status": "mastered"}]}
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 1
    assert "mastery_updates[0].name is required" in result.stderr
    assert "traceback" not in result.stderr.lower()


# === Task 3: Mastery table tests ===


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
                "name": "Pricing \u2192 Parity \u2192 Synthetics",
                "status": "mastered",
                "evidence": "Chain trace pass (S3)",
                "last_tested": "2026-03-16",
            }
        ]
    }
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    assert "| Pricing \u2192 Parity \u2192 Synthetics | mastered |" in result.stdout
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
    assert (
        "| Replication Principle | partial | Factor hypothesis (S5) | 2026-03-16 |" in result.stdout
    )


# === Task 4: Weakness queue, syllabus, section rewrite tests ===


def test_weakness_queue_full_rewrite():
    """Weakness queue should be completely replaced."""
    payload = {
        "weakness_queue": [
            {
                "priority": 1,
                "item": "Vega sensitivity",
                "type": "concept",
                "phase": "Dot",
                "severity": "not-mastered",
                "source": "S3 gap",
                "added": "2026-03-16",
            },
            {
                "priority": 2,
                "item": "Delta hedging",
                "type": "chain",
                "phase": "Dot",
                "severity": "partial",
                "source": "S3 check",
                "added": "2026-03-16",
            },
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
    payload = {"syllabus_updates": [{"topic": "Greeks", "status": "checked"}]}
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    assert "- [x] Greeks" in result.stdout
    # Options Basics should stay checked
    assert "- [x] Options Basics" in result.stdout
    # Volatility should stay unchecked
    assert "- [ ] Volatility" in result.stdout


def test_syllabus_toggle_unchecked():
    """Toggle a checked syllabus item to unchecked."""
    payload = {"syllabus_updates": [{"topic": "Options Basics", "status": "unchecked"}]}
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    assert "- [ ] Options Basics" in result.stdout


def test_syllabus_missing_topic_fails_strict():
    """Missing topic should fail in strict mode."""
    payload = {"syllabus_updates": [{"topic": "Nonexistent Topic", "status": "checked"}]}
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 1
    assert "nonexistent" in result.stderr.lower() or "not found" in result.stderr.lower()


def test_syllabus_missing_topic_warns_lenient():
    """Missing topic should warn (not fail) in lenient mode."""
    payload = {"syllabus_updates": [{"topic": "Nonexistent Topic", "status": "checked"}]}
    result = run_merge(payload, POPULATED_KS, extra_args=["--lenient"])
    assert result.returncode == 0
    assert "warning" in result.stderr.lower() or "not found" in result.stderr.lower()


def test_section_rewrite_compressed_model():
    """Replace ## Compressed Model content."""
    payload = {
        "section_rewrites": {"compressed_model": "Options are arbitrage-enforced replication."}
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


# === Task 5: Section appends, load baseline, engagement tests ===


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
    payload = {"section_appends": {"load_session_history": "| 3 | 3 | none | batch +1 |"}}
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
    payload = {"engagement": {"momentum": "negative"}}
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    assert "- Momentum: negative" in result.stdout
    assert "- Consecutive struggles: 0" in result.stdout  # unchanged


# === Task 6: Dry-run mode and edge case tests ===


def test_dry_run_shows_operations():
    """Dry-run should output human-readable change summary."""
    payload = {
        "mastery_updates": [
            {
                "table": "concepts",
                "name": "Put-Call Parity",
                "status": "mastered",
                "evidence": "Recall pass (S3)",
                "last_tested": "2026-03-16",
            },
            {
                "table": "concepts",
                "name": "Delta",
                "status": "not-mastered",
                "evidence": "Introduced (S3)",
                "last_tested": "2026-03-16",
                "syllabus_topic": "Greeks",
            },
        ],
        "weakness_queue": [
            {
                "priority": 1,
                "item": "Vega",
                "type": "concept",
                "phase": "Dot",
                "severity": "not-mastered",
                "source": "S3",
                "added": "2026-03-16",
            },
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
            {
                "table": "concepts",
                "name": "Put-Call Parity",
                "status": "mastered",
                "evidence": "Recall pass (S3)",
                "last_tested": "2026-03-16",
            },
        ]
    }
    result = run_merge_dry(payload, POPULATED_KS)
    assert result.returncode == 0
    # Should NOT contain KS markers or full KS content
    assert "<!-- KS:start -->" not in result.stdout
    assert "## Concepts" not in result.stdout


def test_missing_section_fails_strict():
    """Targeting a nonexistent section should fail in strict mode."""
    minimal_ks = (
        "<!-- KS:start -->\n# Knowledge State\n\n## Concepts\n\n"
        "| Concept | Status | Syllabus Topic | Evidence | Last Tested |\n"
        "|---------|--------|----------------|----------|-------------|\n\n"
        "<!-- KS:end -->\n"
    )
    payload = {
        "mastery_updates": [
            {
                "table": "factors",
                "name": "Test",
                "status": "partial",
                "evidence": "test",
                "last_tested": "2026-03-16",
            }
        ]
    }
    result = run_merge(payload, minimal_ks)
    assert result.returncode == 1
    assert "not found" in result.stderr.lower()
    assert "<!-- KS:start -->" not in result.stdout


def test_missing_section_warns_lenient():
    """Targeting a nonexistent section should warn but succeed in lenient mode."""
    minimal_ks = (
        "<!-- KS:start -->\n# Knowledge State\n\n## Concepts\n\n"
        "| Concept | Status | Syllabus Topic | Evidence | Last Tested |\n"
        "|---------|--------|----------------|----------|-------------|\n\n"
        "<!-- KS:end -->\n"
    )
    payload = {
        "mastery_updates": [
            {
                "table": "factors",
                "name": "Test",
                "status": "partial",
                "evidence": "test",
                "last_tested": "2026-03-16",
            }
        ]
    }
    result = run_merge(payload, minimal_ks, extra_args=["--lenient"])
    assert result.returncode == 0
    assert "warning" in result.stderr.lower() or "not found" in result.stderr.lower()
    # The output should still be valid with markers
    assert "<!-- KS:start -->" in result.stdout
    assert "<!-- KS:end -->" in result.stdout


def test_escaped_markers_are_canonicalized():
    """Escaped Notion markers should be normalized to a single canonical marker pair."""
    ks_escaped = """\
\\<!-- KS:start --\\>
# Knowledge State

## Engagement Signals

- Momentum: neutral
- Consecutive struggles: 0
- Last celebration: none
- Notes:
\\<!-- KS:end --\\>
"""
    payload = {"engagement": {"momentum": "positive"}}
    result = run_merge(payload, ks_escaped)
    assert result.returncode == 0
    assert result.stdout.count("<!-- KS:start -->") == 1
    assert result.stdout.count("<!-- KS:end -->") == 1
    assert r"\<!-- KS:start --\>" not in result.stdout
    assert r"\<!-- KS:end --\>" not in result.stdout
    assert "- Momentum: positive" in result.stdout


def test_duplicate_markers_fail():
    """Duplicate KS marker pairs should fail to prevent marker corruption."""
    ks_dup = """\
<!-- KS:start -->
<!-- KS:start -->
# Knowledge State

## Engagement Signals
- Momentum: neutral
- Consecutive struggles: 0
- Last celebration: none
- Notes:
<!-- KS:end -->
<!-- KS:end -->
"""
    payload = {"engagement": {"momentum": "positive"}}
    result = run_merge(payload, ks_dup)
    assert result.returncode == 1
    assert "duplicate marker" in result.stderr.lower()


def test_html_table_with_header_row():
    """HTML <table header-row="true"> should be normalized to pipe-delimited."""
    ks_html = """\
<!-- KS:start -->
# Knowledge State

## Concepts
<table header-row="true">
<tr>
<td>Concept</td>
<td>Status</td>
<td>Syllabus Topic</td>
<td>Evidence</td>
<td>Last Tested</td>
</tr>
<tr>
<td>Put-Call Parity</td>
<td>partial</td>
<td>Options Basics</td>
<td>S2 pass</td>
<td>2026-03-14</td>
</tr>
</table>

## Chains

| Chain | Status | Evidence | Last Tested |
|-------|--------|----------|-------------|

## Factors

| Factor | Status | Evidence | Last Tested |
|--------|--------|----------|-------------|

<!-- KS:end -->
"""
    payload = {
        "mastery_updates": [
            {
                "table": "concepts",
                "name": "Put-Call Parity",
                "status": "mastered",
                "evidence": "Recall pass (S5)",
                "last_tested": "2026-03-19",
            }
        ]
    }
    result = run_merge(payload, ks_html)
    assert result.returncode == 0
    assert "| Put-Call Parity | mastered |" in result.stdout
    assert "S2 pass, Recall pass (S5)" in result.stdout


def test_html_duplicate_table_merged():
    """Two HTML tables under ## Concepts — one with header, one without — should merge."""
    ks_dup = """\
<!-- KS:start -->
# Knowledge State

## Concepts
<table header-row="true">
<tr>
<td>Concept</td>
<td>Status</td>
<td>Syllabus Topic</td>
<td>Evidence</td>
<td>Last Tested</td>
</tr>
<tr>
<td>Put-Call Parity</td>
<td>partial</td>
<td>Options Basics</td>
<td>S2 pass</td>
<td>2026-03-14</td>
</tr>
</table>
<table>
<tr>
<td>FastAPI uvicorn</td>
<td>partial</td>
<td>Infra</td>
<td>Wrong factor (S8)</td>
<td>2026-03-16</td>
</tr>
<tr>
<td>Otherside VM</td>
<td>mastered</td>
<td>Infra</td>
<td>Path resolution pass (S8)</td>
<td>2026-03-16</td>
</tr>
</table>

## Chains

| Chain | Status | Evidence | Last Tested |
|-------|--------|----------|-------------|

## Factors

| Factor | Status | Evidence | Last Tested |
|--------|--------|----------|-------------|

<!-- KS:end -->
"""
    payload = {
        "mastery_updates": [
            {
                "table": "concepts",
                "name": "FastAPI uvicorn",
                "status": "mastered",
                "evidence": "Correct factor (S9)",
                "last_tested": "2026-03-19",
            }
        ]
    }
    result = run_merge(payload, ks_dup)
    assert result.returncode == 0
    # Both original rows preserved
    assert "| Put-Call Parity | partial |" in result.stdout
    assert "| Otherside VM | mastered |" in result.stdout
    # Updated row from headerless table
    assert "| FastAPI uvicorn | mastered |" in result.stdout
    assert "Wrong factor (S8), Correct factor (S9)" in result.stdout
    # Output is pipe-delimited, not HTML
    assert "<table" not in result.stdout


def test_html_normalization_outputs_single_table():
    """After normalizing HTML tables, the output should have exactly one pipe table."""
    ks_dup = """\
<!-- KS:start -->
# Knowledge State

## Concepts
<table header-row="true">
<tr>
<td>Concept</td>
<td>Status</td>
<td>Syllabus Topic</td>
<td>Evidence</td>
<td>Last Tested</td>
</tr>
<tr>
<td>Alpha</td>
<td>mastered</td>
<td></td>
<td></td>
<td></td>
</tr>
</table>
<table>
<tr>
<td>Beta</td>
<td>partial</td>
<td></td>
<td></td>
<td></td>
</tr>
</table>

## Chains

| Chain | Status | Evidence | Last Tested |
|-------|--------|----------|-------------|

## Factors

| Factor | Status | Evidence | Last Tested |
|--------|--------|----------|-------------|

<!-- KS:end -->
"""
    payload = {
        "mastery_updates": [
            {
                "table": "concepts",
                "name": "Alpha",
                "status": "mastered",
                "evidence": "Confirmed (S9)",
                "last_tested": "2026-03-19",
            }
        ]
    }
    result = run_merge(payload, ks_dup)
    assert result.returncode == 0
    # No HTML tables in output
    assert "<table" not in result.stdout
    # Both rows in a single pipe table
    assert "| Alpha | mastered |" in result.stdout
    assert "| Beta | partial |" in result.stdout


def test_combined_operations():
    """All operations in a single payload should work together."""
    payload = {
        "mastery_updates": [
            {
                "table": "concepts",
                "name": "Put-Call Parity",
                "status": "mastered",
                "evidence": "Recall pass (S3)",
                "last_tested": "2026-03-16",
            },
        ],
        "weakness_queue": [
            {
                "priority": 1,
                "item": "Delta",
                "type": "concept",
                "phase": "Dot",
                "severity": "not-mastered",
                "source": "S3",
                "added": "2026-03-16",
            },
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


# === Task 7: Exam handler tests ===

# A KS block with all exam sections for testing exam handlers.
EXAM_KS = """\
<!-- KS:start -->
# Knowledge State

## Syllabus
Goal: Pass COMP3000 exam
- [x] Sorting
- [ ] Graphs

## Exam Metadata

- exam_date: 2026-06-15
- exam_format: closed-book
- duration: 3h
- total_marks: 100
- ai_policy: closed-book
- target_score_raw: 80+
- target_score_numeric: 80
- aspirational_target: false
- time_horizon_preset: 10w
- artifacts_ingested: 2
- last_reprioritization: 2026-04-01
- sessions_since_reprioritization: 3

## Exam Blueprint

### Topic Map

| Topic | Marks Weight | Exam Frequency | Transfer Leverage | Hours To Floor | Priority Score | Current No-AI Score |
|-------|-------------|----------------|-------------------|----------------|----------------|---------------------|
| Sorting | 20 | high | 0.8 | 4 | 85 | 70 |

### High-Yield Queue

1. Graphs \u2014 Dijkstra
2. Sorting \u2014 Quicksort edge cases

### Past Paper Analysis

| Paper | Year | Topics Covered | Avg Marks/Topic |
|-------|------|----------------|-----------------|
| Midterm | 2025 | Sorting, Trees | 12 |

## Concepts

| Concept | Status | Syllabus Topic | Evidence | Last Tested |
|---------|--------|----------------|----------|-------------|
| Quicksort | mastered | Sorting | Recall pass (S1) | 2026-04-01 |

## Chains

| Chain | Status | Evidence | Last Tested |
|-------|--------|----------|-------------|

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

## Exam Metrics

### Per-Topic Metrics

| Topic | Closed Book Acc | Time/Question (s) | Marks/Min | Retention Delta | AI Dep Delta |
|-------|----------------|-------------------|-----------|-----------------|-------------|
| Sorting | 0.75 | 45 | 1.2 | +0.1 | -0.05 |

### Aggregate

- Estimated exam score: 72
- Hours studied: 10
- Phase session minutes: 300
- Mock session minutes: 120
- Marks gain rate: 2.5
- Overall no-AI accuracy: 0.70
- Overall AI-dependence delta: -0.03
- Readiness: NOT READY

## Question Bank

| ID | Topic | Format | Marks | Difficulty | Source | Used In Mock | Last Score |
|----|-------|--------|-------|------------|--------|-------------|------------|
| Q1 | Sorting | short-answer | 5 | medium | Past Paper 2025 | yes | 4 |

## Mock History

| Mock # | Date | Score | Time Used | Marks/Min | Weak Topics | Notes |
|--------|------|-------|-----------|-----------|-------------|-------|
| 1 | 2026-04-01 | 65 | 2h30m | 0.9 | Graphs | First attempt |

## Error Taxonomy

| Error ID | Type | Description | Frequency | Topics Affected | Remediation |
|----------|------|-------------|-----------|-----------------|-------------|
| E1 | conceptual | Off-by-one in loop bounds | 3 | Sorting | Practice boundary conditions |

## Past Exams

| Exam Date | Format | Total Marks | Target (raw) | Target (numeric) | Mock Count | Best Mock Score | Self-Reported Result | Archived |
|-----------|--------|-------------|--------------|------------------|------------|-----------------|---------------------|----------|

## Open Questions

## Weakness Queue

| Priority | Item | Type | Phase | Severity | Source | Added |
|----------|------|------|-------|----------|--------|-------|

## Engagement Signals

- Momentum: neutral
- Consecutive struggles: 0
- Last celebration: none
- Notes:
<!-- KS:end -->
"""


def test_exam_metadata_update():
    """Update exam metadata key-value pairs."""
    payload = {
        "exam_metadata": {
            "exam_date": "2026-07-01",
            "target_score_numeric": 85,
            "sessions_since_reprioritization": 0,
        }
    }
    result = run_merge(payload, EXAM_KS)
    assert result.returncode == 0
    assert "- exam_date: 2026-07-01" in result.stdout
    assert "- target_score_numeric: 85" in result.stdout
    assert "- sessions_since_reprioritization: 0" in result.stdout
    # Untouched fields preserved
    assert "- exam_format: closed-book" in result.stdout
    assert "- duration: 3h" in result.stdout


def test_topic_map_upsert_existing():
    """Update an existing row in ### Topic Map."""
    payload = {
        "exam_blueprint": {
            "topic_map": [
                {
                    "topic": "Sorting",
                    "marks_weight": "25",
                    "exam_frequency": "very-high",
                    "transfer_leverage": "0.9",
                    "hours_to_floor": "3",
                    "priority_score": "90",
                    "current_no_ai_score": "80",
                }
            ]
        }
    }
    result = run_merge(payload, EXAM_KS)
    assert result.returncode == 0
    assert "| Sorting | 25 | very-high | 0.9 | 3 | 90 | 80 |" in result.stdout


def test_topic_map_upsert_new():
    """Add a new row to ### Topic Map."""
    payload = {
        "exam_blueprint": {
            "topic_map": [
                {
                    "topic": "Graphs",
                    "marks_weight": "30",
                    "exam_frequency": "high",
                    "transfer_leverage": "0.7",
                    "hours_to_floor": "6",
                    "priority_score": "92",
                    "current_no_ai_score": "40",
                }
            ]
        }
    }
    result = run_merge(payload, EXAM_KS)
    assert result.returncode == 0
    assert "| Graphs | 30 | high | 0.7 | 6 | 92 | 40 |" in result.stdout
    # Existing row untouched
    assert "| Sorting | 20 | high |" in result.stdout


def test_high_yield_queue_rewrite():
    """Replace ### High-Yield Queue content."""
    payload = {
        "exam_blueprint": {"high_yield_queue": "1. Graphs \u2014 BFS/DFS\n2. DP \u2014 Knapsack"}
    }
    result = run_merge(payload, EXAM_KS)
    assert result.returncode == 0
    assert "Graphs \u2014 BFS/DFS" in result.stdout
    assert "DP \u2014 Knapsack" in result.stdout
    # Old content gone
    assert "Quicksort edge cases" not in result.stdout


def test_past_paper_analysis_composite_upsert_existing():
    """Update existing row in ### Past Paper Analysis (composite key)."""
    payload = {
        "exam_blueprint": {
            "past_paper_analysis": [
                {
                    "paper": "Midterm",
                    "year": "2025",
                    "topics_covered": "Sorting, Trees, Graphs",
                    "avg_marks_per_topic": "14",
                }
            ]
        }
    }
    result = run_merge(payload, EXAM_KS)
    assert result.returncode == 0
    assert "| Midterm | 2025 | Sorting, Trees, Graphs | 14 |" in result.stdout


def test_past_paper_analysis_composite_upsert_new():
    """Add new row with different year \u2014 composite key means it's a new row."""
    payload = {
        "exam_blueprint": {
            "past_paper_analysis": [
                {
                    "paper": "Midterm",
                    "year": "2024",
                    "topics_covered": "Sorting",
                    "avg_marks_per_topic": "10",
                }
            ]
        }
    }
    result = run_merge(payload, EXAM_KS)
    assert result.returncode == 0
    # Both rows should exist
    assert "| Midterm | 2025 |" in result.stdout
    assert "| Midterm | 2024 | Sorting | 10 |" in result.stdout


def test_per_topic_metrics_upsert():
    """Update existing row in ### Per-Topic Metrics."""
    payload = {
        "exam_metrics": {
            "per_topic": [
                {
                    "topic": "Sorting",
                    "closed_book_acc": "0.85",
                    "time_per_question": "40",
                    "marks_per_min": "1.5",
                    "retention_delta": "+0.15",
                    "ai_dep_delta": "-0.08",
                }
            ]
        }
    }
    result = run_merge(payload, EXAM_KS)
    assert result.returncode == 0
    assert "| Sorting | 0.85 | 40 | 1.5 | +0.15 | -0.08 |" in result.stdout


def test_per_topic_metrics_add_new():
    """Add new topic to ### Per-Topic Metrics."""
    payload = {
        "exam_metrics": {
            "per_topic": [
                {
                    "topic": "Graphs",
                    "closed_book_acc": "0.50",
                    "time_per_question": "60",
                    "marks_per_min": "0.8",
                    "retention_delta": "+0.0",
                    "ai_dep_delta": "-0.01",
                }
            ]
        }
    }
    result = run_merge(payload, EXAM_KS)
    assert result.returncode == 0
    assert "| Graphs | 0.50 | 60 |" in result.stdout
    assert "| Sorting | 0.75 |" in result.stdout


def test_aggregate_update():
    """Update ### Aggregate key-value pairs."""
    payload = {
        "exam_metrics": {
            "aggregate": {
                "estimated_exam_score": "78",
                "readiness": "BORDERLINE",
            }
        }
    }
    result = run_merge(payload, EXAM_KS)
    assert result.returncode == 0
    assert "- Estimated exam score: 78" in result.stdout
    assert "- Readiness: BORDERLINE" in result.stdout
    # Untouched fields preserved
    assert "- Hours studied: 10" in result.stdout


def test_past_exams_upsert():
    """Add a row to ## Past Exams."""
    payload = {
        "past_exams": [
            {
                "exam_date": "2026-01-15",
                "exam_format": "closed-book",
                "total_marks": "80",
                "target_score_raw": "60",
                "target_score_numeric": "60",
                "mock_count": "2",
                "best_mock_score": "55",
                "self_reported_result": "58",
                "archived_at": "2026-02-01",
            }
        ]
    }
    result = run_merge(payload, EXAM_KS)
    assert result.returncode == 0
    assert "| 2026-01-15 | closed-book | 80 |" in result.stdout


def test_question_bank_upsert_existing():
    """Update existing question in ## Question Bank."""
    payload = {
        "question_bank": [
            {
                "id": "Q1",
                "topic": "Sorting",
                "format": "short-answer",
                "marks": "5",
                "difficulty": "hard",
                "source": "Past Paper 2025",
                "used_in_mock": "yes",
                "last_score": "5",
            }
        ]
    }
    result = run_merge(payload, EXAM_KS)
    assert result.returncode == 0
    assert "| Q1 | Sorting | short-answer | 5 | hard |" in result.stdout


def test_question_bank_add_new():
    """Add new question to ## Question Bank."""
    payload = {
        "question_bank": [
            {
                "id": "Q2",
                "topic": "Graphs",
                "format": "coding",
                "marks": "10",
                "difficulty": "hard",
                "source": "Custom",
                "used_in_mock": "no",
                "last_score": "",
            }
        ]
    }
    result = run_merge(payload, EXAM_KS)
    assert result.returncode == 0
    assert "| Q2 | Graphs | coding | 10 | hard |" in result.stdout
    assert "| Q1 |" in result.stdout  # existing preserved


def test_mock_history_append():
    """Append a row to ## Mock History \u2014 no dedup."""
    payload = {
        "mock_history": [
            {
                "mock_number": "2",
                "date": "2026-04-05",
                "score": "72",
                "time_used": "2h45m",
                "marks_per_min": "1.0",
                "weak_topics": "Graphs",
                "notes": "Improved on sorting",
            }
        ]
    }
    result = run_merge(payload, EXAM_KS)
    assert result.returncode == 0
    # Both rows present (original + appended)
    assert "| 1 | 2026-04-01 | 65 |" in result.stdout
    assert "| 2 | 2026-04-05 | 72 |" in result.stdout


def test_mock_history_append_no_dedup():
    """Appending same mock number twice creates two rows."""
    payload = {
        "mock_history": [
            {
                "mock_number": "1",
                "date": "2026-04-01",
                "score": "65",
                "time_used": "2h30m",
                "marks_per_min": "0.9",
                "weak_topics": "Graphs",
                "notes": "Duplicate",
            }
        ]
    }
    result = run_merge(payload, EXAM_KS)
    assert result.returncode == 0
    # Should have two rows with Mock # = 1
    lines = [ln for ln in result.stdout.split("\n") if "| 1 |" in ln and "2026-04-01" in ln]
    assert len(lines) == 2


def test_error_taxonomy_upsert_existing():
    """Update existing error in ## Error Taxonomy."""
    payload = {
        "error_taxonomy": [
            {
                "error_id": "E1",
                "type": "conceptual",
                "description": "Off-by-one in loop bounds",
                "frequency": "5",
                "topics_affected": "Sorting, Graphs",
                "remediation": "Practice boundary conditions + tracing",
            }
        ]
    }
    result = run_merge(payload, EXAM_KS)
    assert result.returncode == 0
    assert "| E1 | conceptual | Off-by-one in loop bounds | 5 |" in result.stdout
    assert "Sorting, Graphs" in result.stdout


def test_error_taxonomy_add_new():
    """Add new error to ## Error Taxonomy."""
    payload = {
        "error_taxonomy": [
            {
                "error_id": "E2",
                "type": "procedural",
                "description": "Forgot to mark visited",
                "frequency": "2",
                "topics_affected": "Graphs",
                "remediation": "BFS/DFS checklist",
            }
        ]
    }
    result = run_merge(payload, EXAM_KS)
    assert result.returncode == 0
    assert "| E2 | procedural | Forgot to mark visited |" in result.stdout
    assert "| E1 |" in result.stdout  # existing preserved


def test_ensure_exam_sections_auto_create():
    """Legacy KS without exam sections should auto-create them."""
    payload = {
        "exam_metadata": {
            "exam_date": "2026-06-15",
            "total_marks": "100",
        }
    }
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    assert "## Exam Metadata" in result.stdout
    assert "- exam_date: 2026-06-15" in result.stdout
    assert "- total_marks: 100" in result.stdout


def test_ensure_exam_sections_creates_tables():
    """Auto-created exam table sections should have proper headers."""
    payload = {
        "question_bank": [
            {
                "id": "Q1",
                "topic": "Sorting",
                "format": "coding",
                "marks": "10",
                "difficulty": "medium",
                "source": "Custom",
                "used_in_mock": "no",
                "last_score": "",
            }
        ]
    }
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    assert "## Question Bank" in result.stdout
    assert "| Q1 | Sorting | coding |" in result.stdout


def test_ensure_exam_sections_canonical_order():
    """Auto-created sections should appear in canonical order."""
    payload = {
        "exam_metadata": {"exam_date": "2026-06-15"},
        "mock_history": [
            {
                "mock_number": "1",
                "date": "2026-04-01",
                "score": "65",
                "time_used": "2h",
                "marks_per_min": "1.0",
                "weak_topics": "All",
                "notes": "First",
            }
        ],
    }
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    out = result.stdout
    # Exam Metadata should come before Concepts
    assert out.index("## Exam Metadata") < out.index("## Concepts")
    # Mock History should come after Compressed Model
    assert out.index("## Mock History") > out.index("## Compressed Model")


def test_exam_dry_run():
    """Dry-run should report exam operations."""
    payload = {
        "exam_metadata": {"exam_date": "2026-07-01"},
        "exam_blueprint": {
            "topic_map": [
                {"topic": "Sorting", "marks_weight": "25"},
            ],
            "high_yield_queue": "1. Graphs",
            "past_paper_analysis": [
                {"paper": "Final", "year": "2025", "topics_covered": "All"},
            ],
        },
        "exam_metrics": {
            "per_topic": [
                {"topic": "Sorting", "closed_book_acc": "0.9"},
            ],
            "aggregate": {"readiness": "READY"},
        },
        "question_bank": [{"id": "Q1", "topic": "Sorting"}],
        "mock_history": [{"mock_number": "1", "date": "2026-04-01"}],
        "error_taxonomy": [{"error_id": "E1", "type": "conceptual"}],
    }
    result = run_merge_dry(payload, EXAM_KS)
    assert result.returncode == 0
    assert "[exam metadata]" in result.stdout.lower()
    assert "[topic_map]" in result.stdout
    assert "[exam_blueprint] REPLACE ### High-Yield Queue" in result.stdout
    assert "[past_paper]" in result.stdout
    assert "[per_topic_metrics]" in result.stdout
    assert "[question_bank]" in result.stdout
    assert "[mock_history] APPEND" in result.stdout
    assert "[error_taxonomy]" in result.stdout


def test_combined_exam_and_legacy_operations():
    """Exam handlers and legacy handlers should work together."""
    payload = {
        "mastery_updates": [
            {
                "table": "concepts",
                "name": "Quicksort",
                "status": "mastered",
                "evidence": "Recall pass (S2)",
                "last_tested": "2026-04-05",
            }
        ],
        "exam_metadata": {"sessions_since_reprioritization": 4},
        "exam_metrics": {
            "aggregate": {"estimated_exam_score": "75"},
        },
    }
    result = run_merge(payload, EXAM_KS)
    assert result.returncode == 0
    assert "| Quicksort | mastered |" in result.stdout
    assert "Recall pass (S1), Recall pass (S2)" in result.stdout
    assert "- sessions_since_reprioritization: 4" in result.stdout
    assert "- Estimated exam score: 75" in result.stdout
