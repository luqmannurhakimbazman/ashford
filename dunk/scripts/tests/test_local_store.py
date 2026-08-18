"""Core executable tests for the authoritative local store."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "dln-store.py"


def run_cli(root: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(SCRIPT), args[0], "--root", str(root), *args[1:]]
    return subprocess.run(command, capture_output=True, text=True, env=env)


def output(result: subprocess.CompletedProcess[str]) -> dict:
    assert result.stdout, result.stderr
    return json.loads(result.stdout)


def error(result: subprocess.CompletedProcess[str]) -> dict:
    assert result.stderr
    return json.loads(result.stderr)


def init_domain(root: Path, domain: str = "Options Pricing") -> tuple[str, Path]:
    result = run_cli(root, "init", "--domain", domain, "--goal", "Build a pricing model")
    assert result.returncode == 0, result.stderr
    domain_id = output(result)["domain_id"]
    return domain_id, root / "domains" / domain_id


def write_request(tmp_path: Path, value: dict, name: str = "request.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def assessment(
    event_id: str = "assess-1",
    session_id: str = "session-1",
    *,
    evidence_mode: str = "independent",
    outcome: str = "pass",
    operation: str = "acquire",
) -> dict:
    return {
        "assistance": {"hint_count": 0, "level": "none"},
        "context_id": "ctx-1",
        "evidence_mode": evidence_mode,
        "event_id": event_id,
        "kind": "assessment",
        "novelty": "repeat",
        "occurred_at": "2026-08-18T01:00:00Z",
        "operation": operation,
        "outcome": outcome,
        "rubric_id": "rubric-1",
        "schema_version": 1,
        "session_id": session_id,
        "subject": {"id": "parity", "label": "Put-Call Parity", "type": "concept"},
        "task_id": "task-1",
    }


def canonical_tree(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not path.name.startswith(".dln")
    }


def test_init_exact_layout_and_context_list(tmp_path: Path) -> None:
    domain_id, directory = init_domain(tmp_path)
    assert sorted(path.name for path in directory.iterdir()) == [
        "dashboard.md",
        "events.jsonl",
        "profile.yaml",
        "sessions",
        "state.json",
    ]
    assert directory.joinpath("sessions").is_dir()
    assert directory.joinpath("events.jsonl").read_bytes() == b""
    profile = json.loads(directory.joinpath("profile.yaml").read_text())
    state = json.loads(directory.joinpath("state.json").read_text())
    assert profile["revision"] == state["revision"] == 0
    assert profile["domain_id"] == domain_id
    assert state["stage"] == "acquire"
    assert state["source"]["event_count"] == 0

    context = run_cli(tmp_path, "context", "--domain-id", domain_id)
    assert context.returncode == 0, context.stderr
    assert output(context)["state"] == state
    listed = run_cli(tmp_path, "list")
    assert listed.returncode == 0
    assert output(listed)["domains"][0]["domain_id"] == domain_id


def test_root_discovery_requires_explicit_configuration(tmp_path: Path) -> None:
    env = {key: value for key, value in os.environ.items() if key not in {"DLN_VAULT_ROOT", "CLAUDE_PLUGIN_DATA"}}
    missing = subprocess.run(
        [sys.executable, str(SCRIPT), "list"], capture_output=True, text=True, env=env
    )
    assert missing.returncode == 2
    assert "vault root is not configured" in error(missing)["message"]

    env["CLAUDE_PLUGIN_DATA"] = str(tmp_path / "plugin-data")
    configured = subprocess.run(
        [sys.executable, str(SCRIPT), "list"], capture_output=True, text=True, env=env
    )
    assert configured.returncode == 0
    assert output(configured) == {"domains": []}


def test_commit_append_prefix_duplicate_noop_stale_and_conflict(tmp_path: Path) -> None:
    domain_id, directory = init_domain(tmp_path)
    event = assessment()
    request = write_request(tmp_path, {"events": [event, event]})
    before_events = directory.joinpath("events.jsonl").read_bytes()
    committed = run_cli(
        tmp_path,
        "commit",
        "--domain-id",
        domain_id,
        "--expected-revision",
        "0",
        "--request",
        str(request),
    )
    assert committed.returncode == 0, committed.stderr
    assert output(committed)["appended_events"] == 1
    after_events = directory.joinpath("events.jsonl").read_bytes()
    assert after_events.startswith(before_events)
    assert json.loads(directory.joinpath("profile.yaml").read_text())["revision"] == 1

    replay = run_cli(
        tmp_path,
        "commit",
        "--domain-id",
        domain_id,
        "--expected-revision",
        "1",
        "--request",
        str(request),
    )
    assert replay.returncode == 0
    assert output(replay)["status"] == "noop"
    assert directory.joinpath("events.jsonl").read_bytes() == after_events
    assert json.loads(directory.joinpath("profile.yaml").read_text())["revision"] == 1

    snapshot = canonical_tree(directory)
    stale = run_cli(
        tmp_path,
        "commit",
        "--domain-id",
        domain_id,
        "--expected-revision",
        "0",
        "--request",
        str(request),
    )
    assert stale.returncode == 3
    assert "current 1" in error(stale)["message"]
    assert canonical_tree(directory) == snapshot

    changed = {**event, "outcome": "fail"}
    conflict_path = write_request(tmp_path, {"events": [changed]}, "conflict.json")
    conflict = run_cli(
        tmp_path,
        "commit",
        "--domain-id",
        domain_id,
        "--expected-revision",
        "1",
        "--request",
        str(conflict_path),
    )
    assert conflict.returncode == 2
    assert "different content" in error(conflict)["message"]
    assert canonical_tree(directory) == snapshot


def test_profile_patch_ownership_and_json_compatible_yaml_diagnostic(tmp_path: Path) -> None:
    domain_id, directory = init_domain(tmp_path)
    forbidden = write_request(tmp_path, {"profile_patch": {"revision": 99}})
    result = run_cli(
        tmp_path,
        "commit",
        "--domain-id",
        domain_id,
        "--expected-revision",
        "0",
        "--request",
        str(forbidden),
    )
    assert result.returncode == 2
    assert "cannot modify system" in error(result)["message"]

    rename = write_request(tmp_path, {"profile_patch": {"domain": "Renamed domain"}}, "rename.json")
    result = run_cli(
        tmp_path,
        "commit",
        "--domain-id",
        domain_id,
        "--expected-revision",
        "0",
        "--request",
        str(rename),
    )
    assert result.returncode == 2
    assert "cannot modify system" in error(result)["message"]

    patch = write_request(
        tmp_path,
        {"profile_patch": {"goal": "Use parity", "syllabus": ["Parity", "Greeks"]}},
        "patch.json",
    )
    result = run_cli(
        tmp_path,
        "commit",
        "--domain-id",
        domain_id,
        "--expected-revision",
        "0",
        "--request",
        str(patch),
    )
    assert result.returncode == 0, result.stderr
    profile = json.loads(directory.joinpath("profile.yaml").read_text())
    assert profile["goal"] == "Use parity"
    assert profile["syllabus"] == ["Parity", "Greeks"]

    directory.joinpath("profile.yaml").write_text("goal: unsupported full YAML\n", encoding="utf-8")
    invalid = run_cli(tmp_path, "context", "--domain-id", domain_id)
    assert invalid.returncode == 2
    assert "JSON-compatible YAML subset" in error(invalid)["message"]


def test_schema_boundaries_and_portable_session_id(tmp_path: Path) -> None:
    domain_id, _ = init_domain(tmp_path)
    invalid_event = assessment()
    invalid_event["operation"] = "teach"
    request = write_request(tmp_path, {"events": [invalid_event]})
    result = run_cli(
        tmp_path,
        "commit",
        "--domain-id",
        domain_id,
        "--expected-revision",
        "0",
        "--request",
        str(request),
    )
    assert result.returncode == 2
    assert "abstract" in error(result)["message"]

    invalid_event = assessment(session_id="../../escape")
    request = write_request(tmp_path, {"events": [invalid_event]}, "traversal.json")
    result = run_cli(
        tmp_path,
        "commit",
        "--domain-id",
        domain_id,
        "--expected-revision",
        "0",
        "--request",
        str(request),
    )
    assert result.returncode == 2
    assert "portable identifier" in error(result)["message"]

    assisted = assessment()
    assisted["assistance"] = {"hint_count": 1, "level": "prompt"}
    request = write_request(tmp_path, {"events": [assisted]}, "assisted.json")
    result = run_cli(
        tmp_path,
        "commit",
        "--domain-id",
        domain_id,
        "--expected-revision",
        "0",
        "--request",
        str(request),
    )
    assert result.returncode == 2
    assert "independent evidence cannot include" in error(result)["message"]

    incoherent = assessment(evidence_mode="supported")
    incoherent["assistance"] = {"hint_count": 0, "level": "worked"}
    request = write_request(tmp_path, {"events": [incoherent]}, "incoherent.json")
    result = run_cli(
        tmp_path,
        "commit",
        "--domain-id",
        domain_id,
        "--expected-revision",
        "0",
        "--request",
        str(request),
    )
    assert result.returncode == 2
    assert "requires at least one" in error(result)["message"]

    paired = assessment()
    paired["score"] = 1
    request = write_request(tmp_path, {"events": [paired]}, "unpaired.json")
    result = run_cli(
        tmp_path,
        "commit",
        "--domain-id",
        domain_id,
        "--expected-revision",
        "0",
        "--request",
        str(request),
    )
    assert result.returncode == 2
    assert "score and max_score" in error(result)["message"]


def test_completed_session_is_terminal_and_failure_preserves_revision(tmp_path: Path) -> None:
    domain_id, directory = init_domain(tmp_path)
    event = assessment()
    completion = {
        "event_id": "complete-1",
        "evidence_event_ids": [event["event_id"]],
        "kind": "session_completed",
        "next_action": "Retrieve parity tomorrow",
        "next_review_date": "2026-08-19",
        "occurred_at": "2026-08-18T01:30:00Z",
        "receipt_schema_version": 1,
        "schema_version": 1,
        "session_id": "session-1",
    }
    request = write_request(tmp_path, {"events": [event, completion]})
    result = run_cli(
        tmp_path,
        "commit",
        "--domain-id",
        domain_id,
        "--expected-revision",
        "0",
        "--request",
        str(request),
    )
    assert result.returncode == 0, result.stderr
    assert directory.joinpath("sessions/session-1.md").is_file()
    snapshot = canonical_tree(directory)

    later = assessment("assess-2", "session-1")
    later["occurred_at"] = "2026-08-18T02:00:00Z"
    request = write_request(tmp_path, {"events": [later]}, "later.json")
    result = run_cli(
        tmp_path,
        "commit",
        "--domain-id",
        domain_id,
        "--expected-revision",
        "1",
        "--request",
        str(request),
    )
    assert result.returncode == 2
    assert "already completed" in error(result)["message"]
    assert canonical_tree(directory) == snapshot


def test_validate_fails_for_modified_missing_and_orphaned_projections(tmp_path: Path) -> None:
    domain_id, directory = init_domain(tmp_path)
    valid = run_cli(tmp_path, "validate", "--domain-id", domain_id)
    assert valid.returncode == 0, valid.stderr
    assert output(valid)["status"] == "valid"

    directory.joinpath("dashboard.md").write_text("drift\n", encoding="utf-8")
    modified = run_cli(tmp_path, "validate", "--domain-id", domain_id)
    assert modified.returncode == 2
    assert "dashboard.md" in error(modified)["message"]
    assert run_cli(tmp_path, "rebuild", "--domain-id", domain_id).returncode == 0

    directory.joinpath("state.json").unlink()
    missing = run_cli(tmp_path, "validate", "--domain-id", domain_id)
    assert missing.returncode == 2
    assert "state.json" in error(missing)["message"]
    assert run_cli(tmp_path, "rebuild", "--domain-id", domain_id).returncode == 0

    orphan = directory / "sessions" / "orphan.md"
    orphan.write_text("not canonical\n", encoding="utf-8")
    unexpected = run_cli(tmp_path, "validate", "--domain-id", domain_id)
    assert unexpected.returncode == 2
    assert "sessions/orphan.md" in error(unexpected)["message"]
    orphan.unlink()
    assert run_cli(tmp_path, "validate", "--domain-id", domain_id).returncode == 0


def test_malformed_and_truncated_jsonl_reports_line_and_offset(tmp_path: Path) -> None:
    domain_id, directory = init_domain(tmp_path)
    first = json.dumps(assessment(), separators=(",", ":")) + "\n"
    directory.joinpath("events.jsonl").write_text(first + '{"broken":', encoding="utf-8")
    result = run_cli(tmp_path, "validate", "--domain-id", domain_id)
    assert result.returncode == 2
    message = error(result)["message"]
    assert "line 2" in message
    assert "byte offset" in message
    assert "malformed or truncated JSON" in message


@pytest.mark.parametrize("bad_revision", [True, -1, 1.5])
def test_profile_revision_type_is_strict(tmp_path: Path, bad_revision: object) -> None:
    domain_id, directory = init_domain(tmp_path)
    profile = json.loads(directory.joinpath("profile.yaml").read_text())
    profile["revision"] = bad_revision
    directory.joinpath("profile.yaml").write_text(json.dumps(profile), encoding="utf-8")
    result = run_cli(tmp_path, "context", "--domain-id", domain_id)
    assert result.returncode == 2
