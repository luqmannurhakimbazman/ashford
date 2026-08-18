"""Lock, transaction failure, and crash-recovery tests for the local store."""

from __future__ import annotations

import fcntl
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
SCRIPT = SCRIPTS / "dln-store.py"
sys.path.insert(0, str(SCRIPTS))

from dln_store.schema import StaleRevisionError  # noqa: E402
from dln_store.store import LocalStore  # noqa: E402
import dln_store.store as store_module  # noqa: E402


def run_cli(root: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(SCRIPT), args[0], "--root", str(root), *args[1:]]
    return subprocess.run(command, capture_output=True, text=True, env=env)


def output(result: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(result.stdout)


def error(result: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(result.stderr)


def init_domain(root: Path) -> tuple[str, Path]:
    result = run_cli(root, "init", "--domain", "Recovery", "--goal", "Survive faults")
    assert result.returncode == 0, result.stderr
    domain_id = output(result)["domain_id"]
    return domain_id, root / "domains" / domain_id


def request_file(tmp_path: Path, goal: str = "A revised goal") -> Path:
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"profile_patch": {"goal": goal}}), encoding="utf-8")
    return path


def files(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not path.name.startswith(".dln")
    }


@pytest.mark.parametrize(
    "fail_at", ["stage:dashboard.md", "before_install", "install:state.json"]
)
def test_caught_failure_restores_prior_revision_and_removes_transaction(
    tmp_path: Path, fail_at: str
) -> None:
    domain_id, directory = init_domain(tmp_path)
    request = request_file(tmp_path)
    before = files(directory)
    env = os.environ.copy()
    env["DLN_STORE_FAIL_AT"] = fail_at
    result = run_cli(
        tmp_path,
        "commit",
        "--domain-id",
        domain_id,
        "--expected-revision",
        "0",
        "--request",
        str(request),
        env=env,
    )
    assert result.returncode == 1
    assert "injected failure" in error(result)["message"]
    assert files(directory) == before
    assert not directory.joinpath(".dln-transaction").exists()
    assert tmp_path.joinpath(".locks", f"{domain_id}.lock").read_bytes() == b""


def test_profile_edit_during_commit_is_preserved_and_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    domain_id, directory = init_domain(tmp_path)
    profile_path = directory / "profile.yaml"
    original = json.loads(profile_path.read_text())
    user_edited = {**original, "annotations": ["edited during commit"]}

    def edit_at_failpoint(name: str) -> None:
        if name == "before_install":
            profile_path.write_text(json.dumps(user_edited, indent=2) + "\n", encoding="utf-8")

    monkeypatch.setattr(store_module, "_failpoint", edit_at_failpoint)
    store = LocalStore(tmp_path)
    with pytest.raises(StaleRevisionError, match="changed during"):
        store.commit(domain_id, 0, {"profile_patch": {"goal": "writer goal"}})
    assert json.loads(profile_path.read_text()) == user_edited
    assert json.loads(directory.joinpath("state.json").read_text())["revision"] == 0
    assert not directory.joinpath(".dln-transaction").exists()


def test_events_edit_during_commit_is_preserved_and_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    domain_id, directory = init_domain(tmp_path)
    events_path = directory / "events.jsonl"
    external_event = {
        "assistance": {"hint_count": 0, "level": "none"},
        "context_id": "external-context",
        "evidence_mode": "independent",
        "event_id": "external-1",
        "kind": "assessment",
        "novelty": "repeat",
        "occurred_at": "2026-08-18T01:00:00Z",
        "operation": "acquire",
        "outcome": "pass",
        "rubric_id": "external-rubric",
        "schema_version": 1,
        "session_id": "external-session",
        "subject": {"id": "external", "label": "External", "type": "concept"},
        "task_id": "external-task",
    }
    external_bytes = (
        json.dumps(external_event, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )

    def edit_at_failpoint(name: str) -> None:
        if name == "before_install":
            events_path.write_bytes(external_bytes)

    monkeypatch.setattr(store_module, "_failpoint", edit_at_failpoint)
    with pytest.raises(StaleRevisionError, match="events.jsonl changed"):
        LocalStore(tmp_path).commit(
            domain_id, 0, {"profile_patch": {"goal": "writer goal"}}
        )
    assert events_path.read_bytes() == external_bytes
    assert json.loads(directory.joinpath("profile.yaml").read_text())["revision"] == 0
    assert not directory.joinpath(".dln-transaction").exists()


def test_prepared_crash_recovery_discards_candidate_and_preserves_profile_edit(
    tmp_path: Path,
) -> None:
    domain_id, directory = init_domain(tmp_path)
    request = request_file(tmp_path)
    env = os.environ.copy()
    env["DLN_STORE_CRASH_AT"] = "before_install"
    crashed = run_cli(
        tmp_path,
        "commit",
        "--domain-id",
        domain_id,
        "--expected-revision",
        "0",
        "--request",
        str(request),
        env=env,
    )
    assert crashed.returncode == 91
    profile_path = directory / "profile.yaml"
    edited = json.loads(profile_path.read_text())
    edited["annotations"] = ["post-crash edit"]
    profile_path.write_text(json.dumps(edited, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    recovered = run_cli(
        tmp_path,
        "doctor",
        "--domain-id",
        domain_id,
        "--break-stale-lock",
        "--recover",
    )
    assert recovered.returncode == 0, recovered.stderr
    result = output(recovered)
    assert result["lock"]["broken"] is True
    assert result["recovery"] == "discarded-prepared"
    assert json.loads(profile_path.read_text())["annotations"] == ["post-crash edit"]
    assert json.loads(directory.joinpath("state.json").read_text())["revision"] == 0


def test_lock_symlink_is_rejected_without_touching_target(tmp_path: Path) -> None:
    domain_id, _ = init_domain(tmp_path)
    target = tmp_path.parent / f"{tmp_path.name}-outside-lock-target"
    target.write_text("preserve me", encoding="utf-8")
    lock = tmp_path / ".locks" / f"{domain_id}.lock"
    lock.parent.mkdir(exist_ok=True)
    lock.symlink_to(target)

    context = run_cli(tmp_path, "context", "--domain-id", domain_id)
    assert context.returncode == 2
    assert "lock file must not be a symlink" in error(context)["message"]
    assert target.read_text(encoding="utf-8") == "preserve me"

    doctor = run_cli(tmp_path, "doctor", "--domain-id", domain_id)
    assert doctor.returncode == 2
    assert "lock file must not be a symlink" in error(doctor)["message"]
    assert target.read_text(encoding="utf-8") == "preserve me"


def test_transaction_symlink_is_rejected_without_touching_target(tmp_path: Path) -> None:
    domain_id, directory = init_domain(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-transaction"
    outside.mkdir()
    outside.joinpath("sentinel.txt").write_text("preserve me", encoding="utf-8")
    directory.joinpath(".dln-transaction").symlink_to(outside, target_is_directory=True)

    request = request_file(tmp_path)
    commit = run_cli(
        tmp_path,
        "commit",
        "--domain-id",
        domain_id,
        "--expected-revision",
        "0",
        "--request",
        str(request),
    )
    assert commit.returncode == 2
    assert "transaction directory must not be a symlink" in error(commit)["message"]

    doctor = run_cli(tmp_path, "doctor", "--domain-id", domain_id, "--recover")
    assert doctor.returncode == 2
    assert "transaction directory must not be a symlink" in error(doctor)["message"]
    assert outside.joinpath("sentinel.txt").read_text(encoding="utf-8") == "preserve me"


def test_install_failure_preserves_third_party_profile_edit_and_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    domain_id, directory = init_domain(tmp_path)
    profile_path = directory / "profile.yaml"
    external = json.loads(profile_path.read_text(encoding="utf-8"))
    external["annotations"] = ["edited after installation began"]
    external_bytes = (json.dumps(external, indent=2, sort_keys=True) + "\n").encode()

    def edit_after_install_begins(name: str) -> None:
        if name == "install:dashboard.md":
            profile_path.write_bytes(external_bytes)
            raise OSError("injected post-install edit")

    monkeypatch.setattr(store_module, "_failpoint", edit_after_install_begins)
    with pytest.raises(OSError, match="post-install edit"):
        LocalStore(tmp_path).commit(
            domain_id, 0, {"profile_patch": {"goal": "writer goal"}}
        )

    assert profile_path.read_bytes() == external_bytes
    assert directory.joinpath(".dln-transaction", "journal.json").is_file()
    recovered = run_cli(tmp_path, "doctor", "--domain-id", domain_id, "--recover")
    assert recovered.returncode == 5
    assert "changed after the crash" in error(recovered)["message"]
    assert profile_path.read_bytes() == external_bytes
    assert directory.joinpath(".dln-transaction", "journal.json").is_file()


def test_active_lock_contention_and_proven_stale_lock_break(tmp_path: Path) -> None:
    domain_id, _ = init_domain(tmp_path)
    lock = tmp_path / ".locks" / f"{domain_id}.lock"
    lock.parent.mkdir()
    lock.touch()
    request = request_file(tmp_path)
    with lock.open("r+b", buffering=0) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.write(
            json.dumps(
                {
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "started_at": "2026-08-18T00:00:00Z",
                }
            ).encode()
        )
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
        assert result.returncode == 4
        assert "status=active" in error(result)["message"]
        doctor = run_cli(
            tmp_path,
            "doctor",
            "--domain-id",
            domain_id,
            "--break-stale-lock",
        )
        assert doctor.returncode == 4
        assert "active advisory lock" in error(doctor)["message"]
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    lock.write_text(
        json.dumps(
            {
                "hostname": socket.gethostname(),
                "pid": 999_999_999,
                "started_at": "2026-08-18T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    result = run_cli(
        tmp_path,
        "doctor",
        "--domain-id",
        domain_id,
        "--break-stale-lock",
    )
    assert result.returncode == 0, result.stderr
    assert output(result)["lock"]["broken"] is True
    assert lock.read_bytes() == b""


def test_subprocess_crash_requires_doctor_then_rolls_forward(tmp_path: Path) -> None:
    domain_id, directory = init_domain(tmp_path)
    request = request_file(tmp_path)
    env = os.environ.copy()
    env["DLN_STORE_CRASH_AT"] = "install:dashboard.md"
    crashed = run_cli(
        tmp_path,
        "commit",
        "--domain-id",
        domain_id,
        "--expected-revision",
        "0",
        "--request",
        str(request),
        env=env,
    )
    assert crashed.returncode == 91
    lock = tmp_path / ".locks" / f"{domain_id}.lock"
    assert lock.read_bytes()
    assert directory.joinpath(".dln-transaction").exists()

    blocked = run_cli(tmp_path, "context", "--domain-id", domain_id)
    assert blocked.returncode == 5

    recovered = run_cli(
        tmp_path,
        "doctor",
        "--domain-id",
        domain_id,
        "--recover",
    )
    assert recovered.returncode == 0, recovered.stderr
    result = output(recovered)
    assert result["recovery"] == "rolled-forward"
    assert not directory.joinpath(".dln-transaction").exists()
    assert json.loads(directory.joinpath("profile.yaml").read_text())["revision"] == 1
    assert json.loads(directory.joinpath("state.json").read_text())["revision"] == 1

    valid = run_cli(tmp_path, "validate", "--domain-id", domain_id)
    assert valid.returncode == 0, valid.stderr
    assert output(valid)["status"] == "valid"


def test_symlinked_domain_and_sessions_are_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    domains = tmp_path / "domains"
    domains.mkdir()
    domain_link = domains / "escape-00000000"
    domain_link.symlink_to(outside, target_is_directory=True)
    result = run_cli(tmp_path, "context", "--domain-id", "escape-00000000")
    assert result.returncode == 2
    assert "domain directory must not be a symlink" in error(result)["message"]

    domain_id, directory = init_domain(tmp_path, )
    sessions = directory / "sessions"
    sessions.rmdir()
    sessions.symlink_to(outside, target_is_directory=True)
    request = request_file(tmp_path)
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
    assert "sessions directory must not be a symlink" in error(result)["message"]
    assert list(outside.iterdir()) == []


def test_rebuild_never_changes_sources_and_is_byte_deterministic(tmp_path: Path) -> None:
    domain_id, directory = init_domain(tmp_path)
    request = request_file(tmp_path)
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
    assert committed.returncode == 0
    sources = {
        "profile": directory.joinpath("profile.yaml").read_bytes(),
        "events": directory.joinpath("events.jsonl").read_bytes(),
    }
    expected_state = directory.joinpath("state.json").read_bytes()
    expected_dashboard = directory.joinpath("dashboard.md").read_bytes()
    directory.joinpath("state.json").unlink()
    directory.joinpath("dashboard.md").unlink()

    first = run_cli(tmp_path, "rebuild", "--domain-id", domain_id)
    assert first.returncode == 0, first.stderr
    assert directory.joinpath("state.json").read_bytes() == expected_state
    assert directory.joinpath("dashboard.md").read_bytes() == expected_dashboard
    first_tree = files(directory)
    second = run_cli(tmp_path, "rebuild", "--domain-id", domain_id)
    assert second.returncode == 0, second.stderr
    assert files(directory) == first_tree
    assert directory.joinpath("profile.yaml").read_bytes() == sources["profile"]
    assert directory.joinpath("events.jsonl").read_bytes() == sources["events"]


def test_transaction_fsyncs_nested_stage_and_backup_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    domain_id, directory = init_domain(tmp_path)
    store = LocalStore(tmp_path)
    session = {
        "event_id": "complete-1",
        "evidence_event_ids": [],
        "kind": "session_completed",
        "next_action": "Retrieve on a new setup",
        "next_review_date": None,
        "occurred_at": "2026-08-18T01:00:00Z",
        "receipt_schema_version": 1,
        "schema_version": 1,
        "session_id": "session-1",
    }
    store.commit(domain_id, 0, {"events": [session]})
    receipt = directory / "sessions" / "session-1.md"
    assert receipt.is_file()
    rendered = receipt.read_bytes()
    receipt.write_bytes(b"externally drifted receipt\n")

    fsynced: list[Path] = []
    original = store_module._fsync_directory

    def record(path: Path) -> None:
        fsynced.append(Path(path))
        original(path)

    monkeypatch.setattr(store_module, "_fsync_directory", record)
    store.commit(domain_id, 1, {"profile_patch": {"goal": "A revised goal"}})

    transaction = directory / store_module.TXN_NAME
    assert transaction / "stage" / "sessions" in fsynced
    assert transaction / "backups" / "sessions" in fsynced
    assert transaction / "stage" in fsynced
    assert transaction / "backups" in fsynced
    assert not transaction.exists()
    assert receipt.read_bytes() == rendered


def test_unchanged_receipts_are_not_restaged_or_reinstalled(tmp_path: Path) -> None:
    domain_id, directory = init_domain(tmp_path)
    store = LocalStore(tmp_path)
    session = {
        "event_id": "complete-1",
        "evidence_event_ids": [],
        "kind": "session_completed",
        "next_action": "Retrieve tomorrow",
        "next_review_date": None,
        "occurred_at": "2026-08-18T01:00:00Z",
        "receipt_schema_version": 1,
        "schema_version": 1,
        "session_id": "session-1",
    }
    store.commit(domain_id, 0, {"events": [session]})
    receipt = directory / "sessions" / "session-1.md"
    dashboard = directory / "dashboard.md"
    rendered = receipt.read_bytes()
    receipt_inode = receipt.stat().st_ino
    dashboard_inode = dashboard.stat().st_ino

    store.commit(domain_id, 1, {"profile_patch": {"goal": "A revised goal"}})
    assert receipt.read_bytes() == rendered
    assert receipt.stat().st_ino == receipt_inode
    assert dashboard.stat().st_ino != dashboard_inode
    assert not directory.joinpath(store_module.TXN_NAME).exists()

    assert store.rebuild(domain_id)["status"] == "rebuilt"
    assert receipt.stat().st_ino == receipt_inode
    assert not directory.joinpath(store_module.TXN_NAME).exists()
    validated = run_cli(tmp_path, "validate", "--domain-id", domain_id)
    assert validated.returncode == 0, validated.stderr


def test_concurrent_commits_admit_exactly_one_winner(tmp_path: Path) -> None:
    domain_id, directory = init_domain(tmp_path)
    writers = 5
    requests = []
    for index in range(writers):
        path = tmp_path / f"race-{index}.json"
        path.write_text(
            json.dumps({"profile_patch": {"goal": f"writer {index} won"}}), encoding="utf-8"
        )
        requests.append(path)

    command = [sys.executable, str(SCRIPT), "commit", "--root", str(tmp_path), "--domain-id", domain_id]
    processes = [
        subprocess.Popen(
            [*command, "--expected-revision", "0", "--request", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for path in requests
    ]
    results = [(process, *process.communicate()) for process in processes]
    codes = [process.returncode for process, _, _ in results]

    assert codes.count(0) == 1, codes
    # Losers either never got the single-writer lock or saw the winner's revision.
    assert all(code in {3, 4} for code in codes if code != 0), codes
    winner = next(out for process, out, _ in results if process.returncode == 0)
    assert json.loads(winner)["revision"] == 1

    validated = run_cli(tmp_path, "validate", "--domain-id", domain_id)
    assert validated.returncode == 0, validated.stderr
    assert output(validated) == {
        "derived_drift": [],
        "domain_id": domain_id,
        "event_count": 0,
        "revision": 1,
        "status": "valid",
    }
    profile = json.loads(directory.joinpath("profile.yaml").read_text())
    assert profile["revision"] == 1
    assert re.fullmatch(r"writer [0-4] won", profile["goal"])
