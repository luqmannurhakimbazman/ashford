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
        capture_output=True,
        text=True,
    )


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


def test_syllabus_missing_topic_warns():
    """Missing topic should warn to stderr, not error."""
    payload = {"syllabus_updates": [{"topic": "Nonexistent Topic", "status": "checked"}]}
    result = run_merge(payload, POPULATED_KS)
    assert result.returncode == 0
    assert "nonexistent" in result.stderr.lower() or "not found" in result.stderr.lower()


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


def test_missing_section_still_succeeds():
    """Targeting a nonexistent section should warn but not fail."""
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
    assert result.returncode == 0
    assert "warning" in result.stderr.lower() or "not found" in result.stderr.lower()
    # The output should still be valid with markers
    assert "<!-- KS:start -->" in result.stdout
    assert "<!-- KS:end -->" in result.stdout


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
