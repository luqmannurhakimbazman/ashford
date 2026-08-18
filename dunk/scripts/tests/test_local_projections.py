"""Deterministic reducer, learner projection, and narrow legacy-import tests."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
SCRIPT = SCRIPTS / "dln-store.py"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "local_store"
sys.path.insert(0, str(SCRIPTS))

from dln_store.projector import project_state  # noqa: E402
from dln_store.render import (  # noqa: E402
    markdown_text,
    render_all_receipts,
    render_dashboard,
)
from dln_store.schema import (  # noqa: E402
    ValidationError,
    parse_events_bytes,
    parse_profile_bytes,
    pretty_json,
)
from dln_store.store import LocalStore  # noqa: E402


def load_fixture() -> tuple[dict, bytes, list[dict], bytes]:
    profile_bytes = FIXTURE.joinpath("profile.yaml").read_bytes()
    events_bytes = FIXTURE.joinpath("events.jsonl").read_bytes()
    return (
        parse_profile_bytes(profile_bytes),
        profile_bytes,
        parse_events_bytes(events_bytes),
        events_bytes,
    )


def run_cli(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), args[0], "--root", str(root), *args[1:]],
        capture_output=True,
        text=True,
    )


def test_fixture_projections_match_byte_for_byte_snapshots() -> None:
    profile, profile_bytes, events, events_bytes = load_fixture()
    first = project_state(profile, events, profile_bytes=profile_bytes, events_bytes=events_bytes)
    second = project_state(profile, events, profile_bytes=profile_bytes, events_bytes=events_bytes)
    assert (
        pretty_json(first)
        == pretty_json(second)
        == FIXTURE.joinpath("expected-state.json").read_bytes()
    )
    assert render_dashboard(first) == FIXTURE.joinpath("expected-dashboard.md").read_bytes()
    receipts = render_all_receipts(profile, events)
    assert (
        receipts["sessions/session-1.md"] == FIXTURE.joinpath("expected-session-1.md").read_bytes()
    )
    assert (
        receipts["sessions/session-2.md"] == FIXTURE.joinpath("expected-session-2.md").read_bytes()
    )


def test_reducer_separates_evidence_retrieval_transfer_and_calibration() -> None:
    profile, profile_bytes, events, events_bytes = load_fixture()
    state = project_state(profile, events, profile_bytes=profile_bytes, events_bytes=events_bytes)
    subject = state["subjects"][0]
    assert subject["supported"]["event_id"] == "supported-1"
    assert subject["independent"]["event_id"] == "predict-1"
    assert subject["status"] == "needs-work"
    assert subject["retrieval"] == {
        "count": 1,
        "latest": {
            "delay_days": 7,
            "event_id": "relate-1",
            "outcome": "pass",
            "scheduled_date": "2026-08-08",
        },
        "status": "measured",
    }
    assert subject["transfer"]["count"] == 2
    assert state["calibration"] == {
        "count": 3,
        "mean_confidence": 0.733333,
        "mean_gap": -0.033333,
        "mean_normalized_score": 0.766667,
        "status": "measured",
    }
    assert state["stage"] == "revise"


def test_receipts_have_canonical_sections_and_escape_markers() -> None:
    profile, _, events, _ = load_fixture()
    receipts = render_all_receipts(profile, events)
    first = receipts["sessions/session-1.md"].decode()
    assert "## Independent Evidence" in first
    assert "## Supported Performance" in first
    assert "## Prediction Error and Model Revision" in first
    assert "## Delayed Retrieval" in first
    assert "## Calibration" in first
    assert "## Next Action and Review" in first
    assert "Parity \\| Bounds<br>&lt;!-- KS:start --&gt;" in first
    assert "Delayed retrieval was not due or not measured" in first

    second = receipts["sessions/session-2.md"].decode()
    assert "Bounds \\| parity<br>&lt;!-- KS:end --&gt;" in second
    assert "partial (6/10) prediction" in second
    assert "after 7 day(s)" in second
    assert "next review:** not scheduled" in second.casefold()


def test_markdown_escaping_neutralizes_backslash_before_pipe() -> None:
    assert markdown_text(r"C:\|next") == r"C:\\\|next"
    assert markdown_text(r"drive\path|next") == r"drive\\path\|next"
    assert markdown_text("<!-- KS:start -->\n|") == "&lt;!-- KS:start --&gt;<br>\\|"


def test_stage_operations_and_noninitial_revisions_are_enforced() -> None:
    profile, profile_bytes, events, events_bytes = load_fixture()

    wrong_operation = deepcopy(events)
    wrong_operation[0]["operation"] = "predict"
    with pytest.raises(ValidationError, match="not valid in stage 'acquire'"):
        project_state(
            profile,
            wrong_operation,
            profile_bytes=profile_bytes,
            events_bytes=events_bytes,
        )

    early_revision = [deepcopy(events[0]), deepcopy(events[8])]
    with pytest.raises(ValidationError, match="non-initial model revision requires stage 'revise'"):
        project_state(
            profile,
            early_revision,
            profile_bytes=profile_bytes,
            events_bytes=b"{}\n{}\n",
        )


def test_all_remaining_event_variants_reduce_without_becoming_evidence() -> None:
    profile, profile_bytes, events, events_bytes = load_fixture()
    additions = [
        {
            "event_id": "reset-1",
            "kind": "domain_reset",
            "occurred_at": "2026-08-09T00:00:00Z",
            "reason": "new generation",
            "schema_version": 1,
            "session_id": "admin-1",
        },
        {
            "archived_exam": {"date": "2026-08-10", "target": 80},
            "event_id": "exam-close-1",
            "kind": "exam_cycle_closed",
            "occurred_at": "2026-08-10T00:00:00Z",
            "schema_version": 1,
            "self_reported_outcome": "75",
            "session_id": "admin-2",
        },
        {
            "claims": {"concepts": [{"concept": "prior claim"}]},
            "evidence_eligible": False,
            "event_id": "legacy-000000000000000000000000",
            "kind": "legacy_snapshot_imported",
            "occurred_at": "1970-01-01T00:00:00Z",
            "schema_version": 1,
            "session_id": "legacy-0000000000000000",
            "source_sha256": "0" * 64,
        },
        {
            "decision": "exploit",
            "event_id": "model-initial-1",
            "initial_model": True,
            "kind": "model_revision",
            "model": "Initial model",
            "occurred_at": "2026-08-11T00:00:00Z",
            "rationale": "Initial capture",
            "schema_version": 1,
            "session_id": "session-3",
            "triggering_prediction_event_ids": [],
        },
    ]
    extended_bytes = events_bytes + b"".join(
        json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
        for event in additions
    )
    state = project_state(
        profile,
        events + additions,
        profile_bytes=profile_bytes,
        events_bytes=extended_bytes,
    )
    assert state["generation"] == 1
    assert state["subjects"] == []
    assert state["calibration"] == {"count": 0, "status": "not-measured"}
    assert state["legacy_imports"][0]["evidence_eligible"] is False
    assert state["archived_exams"][0]["self_reported_outcome"] == "75"
    assert state["current_model"]["event_id"] == "model-initial-1"


def test_reference_integrity_rejects_non_prediction_and_supported_gate() -> None:
    profile, profile_bytes, events, events_bytes = load_fixture()
    bad_model = {
        "decision": "revise",
        "event_id": "bad-model",
        "kind": "model_revision",
        "model": "bad",
        "occurred_at": "2026-08-09T00:00:00Z",
        "rationale": "bad ref",
        "schema_version": 1,
        "session_id": "session-3",
        "triggering_prediction_event_ids": ["relate-1"],
    }
    with pytest.raises(ValidationError, match="not a prediction"):
        project_state(
            profile,
            events + [bad_model],
            profile_bytes=profile_bytes,
            events_bytes=events_bytes + b"{}\n",
        )

    transition = {
        "assessment_event_ids": ["supported-1"],
        "decision": "unsupported",
        "event_id": "bad-transition",
        "from": "revise",
        "kind": "stage_transition",
        "occurred_at": "2026-08-09T00:00:00Z",
        "rubric_id": "gate",
        "schema_version": 1,
        "session_id": "session-3",
        "to": "acquire",
    }
    with pytest.raises(ValidationError, match="must be independent"):
        project_state(
            profile,
            events + [transition],
            profile_bytes=profile_bytes,
            events_bytes=events_bytes + b"{}\n",
        )


def test_stage_gates_reject_failed_wrong_operation_and_prior_generation_evidence() -> None:
    profile, profile_bytes, events, _ = load_fixture()

    wrong_operation = [dict(event) for event in events[:4]]
    wrong_operation[1] = {**wrong_operation[1], "operation": "predict"}
    with pytest.raises(ValidationError, match="not valid in stage 'acquire'"):
        project_state(
            profile,
            wrong_operation,
            profile_bytes=profile_bytes,
            events_bytes=b"wrong-operation",
        )

    failed = [dict(event) for event in events[:4]]
    failed[1] = {**failed[1], "outcome": "fail"}
    with pytest.raises(ValidationError, match="acquire-to-relate gate"):
        project_state(
            profile,
            failed,
            profile_bytes=profile_bytes,
            events_bytes=b"failed-gate",
        )

    reset = {
        "event_id": "reset-gate-test",
        "kind": "domain_reset",
        "occurred_at": "2026-08-02T00:00:00Z",
        "reason": "new generation",
        "schema_version": 1,
        "session_id": "admin-gate-test",
    }
    stale_gate = {
        **events[3],
        "event_id": "stale-gate-test",
        "occurred_at": "2026-08-02T00:01:00Z",
        "session_id": "session-stale-gate",
    }
    with pytest.raises(ValidationError, match="earlier generation"):
        project_state(
            profile,
            events[:4] + [reset, stale_gate],
            profile_bytes=profile_bytes,
            events_bytes=b"stale-generation",
        )


def test_retrieval_requires_coherent_calendar_delay_and_schedule() -> None:
    profile, profile_bytes, events, _ = load_fixture()
    prior = events[1]
    gate = {
        **events[3],
        "assessment_event_ids": [prior["event_id"]],
        "occurred_at": "2026-08-01T09:05:00Z",
    }
    retrieval = dict(events[5])

    same_day = {
        **retrieval,
        "occurred_at": "2026-08-01T10:00:00Z",
        "retrieval": {**retrieval["retrieval"], "observed_delay_days": 0},
    }
    with pytest.raises(ValidationError, match="later UTC date"):
        project_state(
            profile,
            [prior, gate, same_day],
            profile_bytes=profile_bytes,
            events_bytes=b"same-day",
        )

    wrong_delay = {
        **retrieval,
        "retrieval": {**retrieval["retrieval"], "observed_delay_days": 6},
    }
    with pytest.raises(ValidationError, match="expected 7"):
        project_state(
            profile,
            [prior, gate, wrong_delay],
            profile_bytes=profile_bytes,
            events_bytes=b"wrong-delay",
        )

    future_schedule = {
        **retrieval,
        "retrieval": {**retrieval["retrieval"], "scheduled_date": "2026-08-09"},
    }
    with pytest.raises(ValidationError, match="scheduled_date"):
        project_state(
            profile,
            [prior, gate, future_schedule],
            profile_bytes=profile_bytes,
            events_bytes=b"future-schedule",
        )


def test_rebuild_from_fixture_recreates_only_derived_files(tmp_path: Path) -> None:
    profile, _, _, _ = load_fixture()
    directory = tmp_path / "domains" / profile["domain_id"]
    (directory / "sessions").mkdir(parents=True)
    shutil.copyfile(FIXTURE / "profile.yaml", directory / "profile.yaml")
    shutil.copyfile(FIXTURE / "events.jsonl", directory / "events.jsonl")
    sources = {
        "profile": directory.joinpath("profile.yaml").read_bytes(),
        "events": directory.joinpath("events.jsonl").read_bytes(),
    }
    LocalStore(tmp_path).rebuild(profile["domain_id"])
    assert (
        directory.joinpath("state.json").read_bytes()
        == FIXTURE.joinpath("expected-state.json").read_bytes()
    )
    assert (
        directory.joinpath("dashboard.md").read_bytes()
        == FIXTURE.joinpath("expected-dashboard.md").read_bytes()
    )
    assert (
        directory.joinpath("sessions/session-1.md").read_bytes()
        == FIXTURE.joinpath("expected-session-1.md").read_bytes()
    )
    assert directory.joinpath("profile.yaml").read_bytes() == sources["profile"]
    assert directory.joinpath("events.jsonl").read_bytes() == sources["events"]


LEGACY = """<!-- KS:start -->
# Knowledge State

## Syllabus
Goal: Learn options
- [x] Basics
- [ ] Greeks

## Concepts
| Concept | Status | Syllabus Topic | Evidence | Last Tested |
|---|---|---|---|---|
| Parity \\| Bounds | mastered | Basics | old note | 2026-01-01 |

## Chains
| Chain | Status | Evidence | Last Tested |
|---|---|---|---|
| Spot \\| Forward -> Option | partial | old trace | 2026-01-02 |

## Factors
| Factor | Status | Evidence | Last Tested |
|---|---|---|---|
| Carry \\| Funding | candidate | old comparison | 2026-01-03 |

## Compressed Model
Prices reflect replication.

## Open Questions
- What changes with dividends?
<!-- KS:end -->
"""


def test_legacy_import_is_offline_idempotent_and_evidence_ineligible(tmp_path: Path) -> None:
    source = tmp_path / "legacy.md"
    source.write_text(LEGACY, encoding="utf-8")
    first = run_cli(
        tmp_path,
        "import-legacy-ks",
        "--domain",
        "Legacy Options",
        "--input",
        str(source),
    )
    assert first.returncode == 0, first.stderr
    first_result = json.loads(first.stdout)
    assert first_result["initialized"] is True
    domain_id = first_result["domain_id"]
    directory = tmp_path / "domains" / domain_id
    state = json.loads(directory.joinpath("state.json").read_text())
    assert state["subjects"] == []
    assert state["calibration"]["status"] == "not-measured"
    imported = state["legacy_imports"][0]
    assert imported["evidence_eligible"] is False
    assert imported["claims"]["concepts"][0]["concept"] == "Parity | Bounds"
    assert imported["claims"]["concepts"][0]["status_claim"] == "mastered"
    assert imported["claims"]["chains"][0]["chain"] == "Spot | Forward -> Option"
    assert imported["claims"]["factors"][0]["factor"] == "Carry | Funding"
    assert imported["claims"]["syllabus"][0]["completed_claim"] is True
    first_events = directory.joinpath("events.jsonl").read_bytes()

    replay = run_cli(
        tmp_path,
        "import-legacy-ks",
        "--domain",
        "Legacy Options",
        "--input",
        str(source),
    )
    assert replay.returncode == 0, replay.stderr
    assert json.loads(replay.stdout)["status"] == "noop"
    assert directory.joinpath("events.jsonl").read_bytes() == first_events

    source.write_text(LEGACY.replace("Carry \\| Funding", "Volatility"), encoding="utf-8")
    different = run_cli(
        tmp_path,
        "import-legacy-ks",
        "--domain",
        "Legacy Options",
        "--input",
        str(source),
    )
    assert different.returncode == 2
    assert "different legacy snapshot" in json.loads(different.stderr)["message"]
    assert directory.joinpath("events.jsonl").read_bytes() == first_events


def test_legacy_import_rejects_missing_or_ambiguous_markers(tmp_path: Path) -> None:
    source = tmp_path / "legacy.md"
    source.write_text("# Knowledge State\n", encoding="utf-8")
    result = run_cli(
        tmp_path,
        "import-legacy-ks",
        "--domain",
        "Broken",
        "--input",
        str(source),
    )
    assert result.returncode == 2
    assert "exactly one" in json.loads(result.stderr)["message"]
