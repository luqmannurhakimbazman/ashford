from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
SCRIPT = SCRIPTS / "dln-store.py"
FIXTURE = Path(__file__).parent / "fixtures" / "syllabus" / "st5201x" / "syllabus2026.pdf"
EXPECTED_DIGEST = "53909df562e2658ab3e1327eb8c33120fa12b37489178dc87bb4d632e4f15376"

sys.path.insert(0, str(SCRIPTS))

projector_module = importlib.import_module("dln_store.projector")
schema_module = importlib.import_module("dln_store.schema")
adapter_module = importlib.import_module("dln_store.st5201x_syllabus")
render_module = importlib.import_module("dln_store.render")
project_state = projector_module.project_state
ValidationError = schema_module.ValidationError
build_syllabus_approval_event = schema_module.build_syllabus_approval_event
initial_profile = schema_module.initial_profile
pretty_json = schema_module.pretty_json
sha256_bytes = schema_module.sha256_bytes
build_ingestion_event = adapter_module.build_ingestion_event
render_all_receipts = render_module.render_all_receipts
render_all_syllabus_receipts = render_module.render_all_syllabus_receipts
render_dashboard = render_module.render_dashboard


def run_cli(
    root: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    command_env = os.environ.copy()
    if env:
        command_env.update(env)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--root", str(root)],
        capture_output=True,
        check=False,
        text=True,
        env=command_env,
    )


def output(result: subprocess.CompletedProcess[str]) -> dict:
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def error(result: subprocess.CompletedProcess[str]) -> dict:
    assert result.returncode != 0, result.stdout
    return json.loads(result.stderr)


def init_domain(root: Path) -> tuple[str, Path]:
    result = output(run_cli(root, "init", "--domain", "ST5201X", "--goal", "Learn statistics"))
    domain_id = result["domain_id"]
    return domain_id, root / "domains" / domain_id


def tree(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and ".dln-transaction" not in path.parts
    }


def ingest(
    root: Path, domain_id: str, revision: int, document: Path = FIXTURE, **overrides: str
) -> subprocess.CompletedProcess[str]:
    values = {
        "original_filename": "syllabus2026.pdf",
        "media_type": "application/pdf",
        "adapter": "st5201x-2026-v1",
        "occurred_at": "2026-08-19T00:00:00Z",
    }
    values.update(overrides)
    return run_cli(
        root,
        "ingest-syllabus",
        "--domain-id",
        domain_id,
        "--expected-revision",
        str(revision),
        "--document",
        str(document),
        "--original-filename",
        values["original_filename"],
        "--media-type",
        values["media_type"],
        "--adapter",
        values["adapter"],
        "--occurred-at",
        values["occurred_at"],
    )


def source_event(directory: Path) -> dict:
    return json.loads(
        directory.joinpath("events.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )


def approval_request(
    source: dict,
    *,
    event_id: str,
    occurred_at: str,
    supersedes: str | None = None,
    correction: bool = False,
) -> dict:
    all_ids = [item["assertion_id"] for item in source["assertions"]]
    alignment = "st5201x.schedule.weeks_7_13_alignment"
    title = "st5201x.course.title"
    accepted = [item for item in all_ids if item != alignment and (not correction or item != title)]
    corrections = []
    if correction:
        corrections.append(
            {
                "correction_assertion_id": "learner-correction-course-title-v1",
                "target_assertion_id": title,
                "field": "course.title",
                "status": "specified",
                "normalized_value": "Statistical Foundations for Data Science",
                "rationale": "Use the learner-confirmed official catalog wording.",
                "origin": "learner_correction",
            }
        )
    return {
        "event_id": event_id,
        "session_id": f"syllabus-approval-{event_id}",
        "occurred_at": occurred_at,
        "source_version_id": source["source"]["source_version_id"],
        "source_assertion_set_sha256": source["assertion_set_sha256"],
        "actor": {"type": "learner", "id": "learner"},
        "accepted_assertion_ids": accepted,
        "deferred_assertion_ids": [alignment],
        "corrections": corrections,
        "supersedes_approval_event_id": supersedes,
    }


def write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def mastery_view(state: dict) -> dict:
    fields = {
        "archived_exams",
        "calibration",
        "completed_sessions",
        "current_model",
        "generation",
        "legacy_imports",
        "next_action",
        "next_review_date",
        "stage",
        "subjects",
    }
    return {field: state[field] for field in fields}


def test_verified_fixture_manifest_and_noninvented_alignment() -> None:
    document = FIXTURE.read_bytes()
    assert len(document) == 45_185
    assert hashlib.sha256(document).hexdigest() == EXPECTED_DIGEST

    event = build_ingestion_event(
        FIXTURE,
        original_filename="syllabus2026.pdf",
        media_type="application/pdf",
        adapter_id="st5201x-2026-v1",
        occurred_at="2026-08-19T00:00:00Z",
    )
    assert event["source"] == {
        "source_id": "st5201x-2026-2027-sem1-syllabus",
        "source_version_id": f"sha256-{EXPECTED_DIGEST}",
        "original_filename": "syllabus2026.pdf",
        "media_type": "application/pdf",
        "byte_size": 45_185,
        "page_count": 1,
        "sha256": EXPECTED_DIGEST,
        "content_retention": "extracted_text_only",
        "supersedes_source_version_id": None,
    }
    assert event["extraction"]["extractor_version"] == "1.0 (build 1451.5.3)"
    assert len(event["assertions"]) == 52
    for assertion in event["assertions"]:
        for evidence in assertion["evidence"]:
            page = event["pages"][evidence["page_number"] - 1]["text"]
            assert page[evidence["start_char"] : evidence["end_char"]] == evidence["quote"]

    assertions = {item["assertion_id"]: item for item in event["assertions"]}
    values = {item["assertion_id"]: item["normalized_value"] for item in event["assertions"]}
    assert values["st5201x.course.code"] == "ST5201X"
    assert values["st5201x.course.title"] == "Statistical Foundations of Data Science"
    assert values["st5201x.offering.term"] == "Semester 1"
    assert values["st5201x.offering.academic_year"] == "2026/2027"
    assert values["st5201x.staff.lecturer.name"] == "Zhang Yao"
    assert values["st5201x.staff.lecturer.department"] == "Department of Statistics & Data Science"
    assert values["st5201x.staff.lecturer.room"] == "L7-106, Faculty of Science"
    assert values["st5201x.staff.lecturer.email"] == "yaozhang@nus.edu.sg"
    assert values["st5201x.class.days"] == ["Thursday", "Friday"]
    assert values["st5201x.class.time"] == "19:00-22:00"
    assert values["st5201x.class.venue"] == "Lecture Theatre 34 (LT34)"
    assert values["st5201x.tutorial.start_week"] == 3
    assert values["st5201x.reference.primary.author"] == "John Rice"
    assert values["st5201x.reference.primary.edition"] == "Third edition"
    assert {
        (item["normalized_value"]["name"], item["normalized_value"]["weight_percent"])
        for item in event["assertions"]
        if item["field"] == "assessment.component"
    } == {("Homework", 50), ("Final exam", 50)}
    policy_categories = {
        item["normalized_value"]["category"]
        for item in event["assertions"]
        if item["field"] == "policy.rule"
    }
    assert policy_categories == {
        "submission",
        "solutions",
        "lateness",
        "exam_format",
        "notes",
        "calculator",
        "devices",
        "past_exams",
        "exam_consequence",
    }
    assert assertions["st5201x.assessment.final_exam.date"]["status"] == "not_specified"
    assert assertions["st5201x.reference.primary.designation"]["status"] == "not_specified"
    alignment = assertions["st5201x.schedule.weeks_7_13_alignment"]
    assert alignment["status"] == "unresolved"
    assert alignment["normalized_value"]["alternatives"] == []
    assert alignment["normalized_value"]["unresolved_fields"] == [
        "week_to_topic",
        "week_to_milestone",
    ]
    assert [
        item["normalized_value"]["week"]
        for item in event["assertions"]
        if item["field"] == "schedule.row"
    ] == list(range(1, 7))
    assert len([item for item in event["assertions"] if item["field"] == "coverage.topic"]) == 10
    assert len([item for item in event["assertions"] if item["field"] == "milestone.homework"]) == 5


def test_cli_intake_is_digest_idempotent_and_does_not_change_mastery(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    domain_id, directory = init_domain(root)
    before = output(run_cli(root, "context", "--domain-id", domain_id))["state"]

    result = output(ingest(root, domain_id, 0))
    assert result["status"] == "committed"
    assert result["revision"] == 1
    assert result["appended_events"] == 1
    assert result["source_sha256"] == EXPECTED_DIGEST
    assert result["approval_status"] == "approval_required"

    alias = tmp_path / "renamed.pdf"
    alias.write_bytes(FIXTURE.read_bytes())
    replay = output(
        ingest(
            root,
            domain_id,
            1,
            alias,
            original_filename="department-copy.pdf",
            occurred_at="2026-08-20T00:00:00Z",
        )
    )
    assert replay["status"] == "noop"
    assert replay["revision"] == 1
    assert replay["appended_events"] == 0
    assert len(directory.joinpath("events.jsonl").read_text().splitlines()) == 1

    after = output(run_cli(root, "context", "--domain-id", domain_id))["state"]
    assert mastery_view(after) == mastery_view(before)


def test_truthful_degradation_never_changes_canonical_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "vault"
    domain_id, directory = init_domain(root)
    baseline = tree(directory)

    missing = error(ingest(root, domain_id, 0, tmp_path / "missing.pdf"))
    assert missing["error"] == "SyllabusUnavailableError"
    assert "unavailable: missing path" in missing["message"]
    assert tree(directory) == baseline

    directory_input = error(ingest(root, domain_id, 0, tmp_path))
    assert "not a regular file" in directory_input["message"]
    assert tree(directory) == baseline

    symlink = tmp_path / "syllabus-link.pdf"
    symlink.symlink_to(FIXTURE)
    linked = error(ingest(root, domain_id, 0, symlink))
    assert "symlinks are not accepted" in linked["message"]
    assert tree(directory) == baseline

    mutated = tmp_path / "mutated.pdf"
    changed = bytearray(FIXTURE.read_bytes())
    changed[-1] ^= 1
    mutated.write_bytes(changed)
    mismatch = error(ingest(root, domain_id, 0, mutated))
    assert mismatch["error"] == "SyllabusDigestMismatchError"
    assert "syllabus digest mismatch" in mismatch["message"]
    assert EXPECTED_DIGEST in mismatch["message"]
    assert tree(directory) == baseline

    wrong_media = error(ingest(root, domain_id, 0, FIXTURE, media_type="text/plain"))
    assert wrong_media["error"] == "SyllabusUnsupportedError"
    assert "unsupported syllabus media type" in wrong_media["message"]
    wrong_adapter = error(ingest(root, domain_id, 0, FIXTURE, adapter="generic-pdf"))
    assert "unsupported syllabus adapter" in wrong_adapter["message"]
    assert tree(directory) == baseline

    import dln_store.st5201x_syllabus as adapter

    def unreadable(_descriptor: int, _count: int) -> bytes:
        raise OSError("injected read failure")

    monkeypatch.setattr(adapter.os, "read", unreadable)
    with pytest.raises(ValidationError, match="syllabus document unreadable"):
        build_ingestion_event(
            FIXTURE,
            original_filename="syllabus2026.pdf",
            media_type="application/pdf",
            adapter_id="st5201x-2026-v1",
            occurred_at="2026-08-19T00:00:00Z",
        )
    assert tree(directory) == baseline


def test_context_approval_correction_supersession_is_immutable_and_replayable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "vault"
    domain_id, directory = init_domain(root)
    output(ingest(root, domain_id, 0))
    source = source_event(directory)
    before_approvals = output(run_cli(root, "context", "--domain-id", domain_id))["state"]

    first = approval_request(
        source,
        event_id="syllabus-approval-1",
        occurred_at="2026-08-19T00:10:00Z",
    )
    request_path = write_json(tmp_path / "approval-1.json", first)
    committed = output(
        run_cli(
            root,
            "approve-syllabus",
            "--domain-id",
            domain_id,
            "--expected-revision",
            "1",
            "--request",
            str(request_path),
        )
    )
    assert committed["status"] == "committed"
    assert committed["revision"] == 2
    assert committed["approval_status"] == "approved"

    source_replay = output(ingest(root, domain_id, 2, occurred_at="2026-08-20T00:00:00Z"))
    assert source_replay["status"] == "noop"
    assert source_replay["approval_status"] == "approved"
    assert source_replay["revision"] == 2

    replay = output(
        run_cli(
            root,
            "approve-syllabus",
            "--domain-id",
            domain_id,
            "--expected-revision",
            "2",
            "--request",
            str(request_path),
        )
    )
    assert replay["status"] == "noop"
    assert replay["revision"] == 2

    conflicting = deepcopy(first)
    conflicting["accepted_assertion_ids"].append(conflicting["deferred_assertion_ids"].pop())
    conflict_path = write_json(tmp_path / "approval-conflict.json", conflicting)
    conflict = error(
        run_cli(
            root,
            "approve-syllabus",
            "--domain-id",
            domain_id,
            "--expected-revision",
            "2",
            "--request",
            str(conflict_path),
        )
    )
    assert "already exists with different content" in conflict["message"]

    second = approval_request(
        source,
        event_id="syllabus-approval-2",
        occurred_at="2026-08-19T00:20:00Z",
        supersedes="syllabus-approval-1",
        correction=True,
    )
    second_path = write_json(tmp_path / "approval-2.json", second)
    superseded = output(
        run_cli(
            root,
            "approve-syllabus",
            "--domain-id",
            domain_id,
            "--expected-revision",
            "2",
            "--request",
            str(second_path),
        )
    )
    assert superseded["status"] == "committed"
    assert superseded["revision"] == 3

    events = [
        json.loads(line) for line in directory.joinpath("events.jsonl").read_text().splitlines()
    ]
    original_title = next(
        item for item in events[0]["assertions"] if item["assertion_id"] == "st5201x.course.title"
    )
    assert original_title["normalized_value"] == "Statistical Foundations of Data Science"
    assert (
        events[2]["corrections"][0]["normalized_value"]
        == "Statistical Foundations for Data Science"
    )
    assert events[2]["supersedes_approval_event_id"] == "syllabus-approval-1"

    fresh = output(run_cli(root, "context", "--domain-id", domain_id))
    assert fresh["profile"]["revision"] == 3
    assert fresh["state"]["source"]["event_count"] == 3
    assert mastery_view(fresh["state"]) == mastery_view(before_approvals)


def test_invalid_approval_and_generic_injection_preserve_tree(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    domain_id, directory = init_domain(root)
    output(ingest(root, domain_id, 0))
    source = source_event(directory)
    baseline = tree(directory)

    omitted = approval_request(
        source,
        event_id="bad-approval",
        occurred_at="2026-08-19T00:10:00Z",
    )
    omitted["accepted_assertion_ids"].pop()
    path = write_json(tmp_path / "bad.json", omitted)
    invalid = error(
        run_cli(
            root,
            "approve-syllabus",
            "--domain-id",
            domain_id,
            "--expected-revision",
            "1",
            "--request",
            str(path),
        )
    )
    assert "disposition omits assertion IDs" in invalid["message"]
    assert tree(directory) == baseline

    before_source = approval_request(
        source,
        event_id="early-approval",
        occurred_at="2026-08-18T23:59:59Z",
    )
    path = write_json(tmp_path / "early.json", before_source)
    early = error(
        run_cli(
            root,
            "approve-syllabus",
            "--domain-id",
            domain_id,
            "--expected-revision",
            "1",
            "--request",
            str(path),
        )
    )
    assert "approval cannot precede source ingestion" in early["message"]
    assert tree(directory) == baseline

    injected = write_json(tmp_path / "injected.json", {"events": [source]})
    blocked = error(
        run_cli(
            root,
            "commit",
            "--domain-id",
            domain_id,
            "--expected-revision",
            "1",
            "--request",
            str(injected),
        )
    )
    assert "reserved syllabus event kind" in blocked["message"]
    approval_event = build_syllabus_approval_event(
        approval_request(
            source,
            event_id="injected-approval",
            occurred_at="2026-08-19T00:10:00Z",
        )
    )
    injected_approval = write_json(
        tmp_path / "injected-approval.json", {"events": [approval_event]}
    )
    blocked_approval = error(
        run_cli(
            root,
            "commit",
            "--domain-id",
            domain_id,
            "--expected-revision",
            "1",
            "--request",
            str(injected_approval),
        )
    )
    assert "reserved syllabus event kind" in blocked_approval["message"]
    assert tree(directory) == baseline


def test_history_rejects_noncanonical_source_identity_and_branched_approval() -> None:
    source = build_ingestion_event(
        FIXTURE,
        original_filename="syllabus2026.pdf",
        media_type="application/pdf",
        adapter_id="st5201x-2026-v1",
        occurred_at="2026-08-19T00:00:00Z",
    )
    duplicate = deepcopy(source)
    duplicate["event_id"] = "different-source-event"
    duplicate["session_id"] = "different-source-session"
    with pytest.raises(ValidationError, match="deterministically derived"):
        project_state(
            {
                "annotations": [],
                "domain": "ST5201X",
                "domain_id": "st5201x-86824c31",
                "exam": {},
                "goal": "Learn",
                "review_preferences": {},
                "revision": 2,
                "schema_version": 1,
                "syllabus": [],
            },
            [source, duplicate],
            profile_bytes=b"profile",
            events_bytes=b"events",
        )

    first = build_syllabus_approval_event(
        approval_request(source, event_id="approval-1", occurred_at="2026-08-19T00:10:00Z")
    )
    branch = build_syllabus_approval_event(
        approval_request(source, event_id="approval-branch", occurred_at="2026-08-19T00:20:00Z")
    )
    profile = {
        "annotations": [],
        "domain": "ST5201X",
        "domain_id": "st5201x-86824c31",
        "exam": {},
        "goal": "Learn",
        "review_preferences": {},
        "revision": 3,
        "schema_version": 1,
        "syllabus": [],
    }
    with pytest.raises(ValidationError, match="must cite the active approval"):
        project_state(
            profile,
            [source, first, branch],
            profile_bytes=pretty_json(profile),
            events_bytes=b"events",
        )

    active_session = {
        "assistance": {"hint_count": 0, "level": "none"},
        "context_id": "reuse-check",
        "evidence_mode": "independent",
        "event_id": "assessment-before-approval",
        "kind": "assessment",
        "novelty": "repeat",
        "occurred_at": "2026-08-19T00:05:00Z",
        "operation": "acquire",
        "outcome": "pass",
        "rubric_id": "reuse-check",
        "schema_version": 1,
        "session_id": "syllabus-approval-approval-reuse",
        "subject": {"id": "probability", "label": "Probability", "type": "concept"},
        "task_id": "reuse-check",
    }
    reused = build_syllabus_approval_event(
        approval_request(
            source,
            event_id="approval-reuse",
            occurred_at="2026-08-19T00:10:00Z",
        )
    )
    with pytest.raises(ValidationError, match="must not reuse a prior event session"):
        project_state(
            profile,
            [source, active_session, reused],
            profile_bytes=pretty_json(profile),
            events_bytes=b"events",
        )


def projected(profile: dict, events: list[dict]) -> dict:
    return project_state(
        profile,
        events,
        profile_bytes=pretty_json(profile),
        events_bytes=b"".join(pretty_json(event) for event in events),
    )


def grounded_assessment(
    approval_event_id: str,
    assertion_ids: list[str],
    *,
    event_id: str = "grounded-assessment",
    occurred_at: str = "2026-08-19T00:20:00Z",
) -> dict:
    return {
        "assistance": {"hint_count": 0, "level": "none"},
        "context_id": event_id,
        "evidence_mode": "independent",
        "event_id": event_id,
        "grounding": {
            "approval_event_id": approval_event_id,
            "assertion_ids": assertion_ids,
        },
        "kind": "assessment",
        "novelty": "repeat",
        "occurred_at": occurred_at,
        "operation": "acquire",
        "outcome": "pass",
        "rubric_id": "grounded-rubric",
        "schema_version": 1,
        "session_id": f"session-{event_id}",
        "subject": {"id": "probability", "label": "Probability", "type": "concept"},
        "task_id": event_id,
    }


def test_grounding_projection_legacy_pending_approved_and_correction() -> None:
    profile = initial_profile("ST5201X", "Learn statistics")
    profile["syllabus"] = ["Legacy Topic"]
    legacy = projected(profile, [])
    assert legacy["syllabus"] == ["Legacy Topic"]
    assert legacy["grounding"] == {
        "active_approval": None,
        "active_source": None,
        "effective_assertions": [],
        "legacy_fallback": True,
        "pending_sources": [],
        "planning_topics": [{"assertion_ids": [], "citable": False, "label": "Legacy Topic"}],
        "status": "ungrounded",
        "unresolved_assertions": [],
    }

    source = build_ingestion_event(
        FIXTURE,
        original_filename="syllabus2026.pdf",
        media_type="application/pdf",
        adapter_id="st5201x-2026-v1",
        occurred_at="2026-08-19T00:00:00Z",
    )
    pending = projected(profile, [source])
    assert pending["grounding"]["status"] == "approval_required"
    assert pending["grounding"]["active_source"] is None
    assert pending["grounding"]["pending_sources"][0]["receipt"].startswith("syllabus/")
    assert pending["syllabus"] == ["Legacy Topic"]

    first = build_syllabus_approval_event(
        approval_request(source, event_id="approval-1", occurred_at="2026-08-19T00:10:00Z")
    )
    approved = projected(profile, [source, first])
    expected_topics = []
    for assertion in source["assertions"]:
        if (
            assertion["field"] == "coverage.topic"
            and assertion["normalized_value"] not in expected_topics
        ):
            expected_topics.append(assertion["normalized_value"])
    assert approved["grounding"]["status"] == "approved"
    assert approved["grounding"]["legacy_fallback"] is False
    assert approved["syllabus"] == expected_topics
    assert all(topic["citable"] for topic in approved["grounding"]["planning_topics"])
    assert approved["grounding"]["unresolved_assertions"][0]["assertion_id"] == (
        "st5201x.schedule.weeks_7_13_alignment"
    )
    assert "pages" not in approved["grounding"]
    approved_dashboard = render_dashboard(approved).decode()
    assert "## Course Grounding" in approved_dashboard
    assert "`approved`" in approved_dashboard
    assert "Syllabus Intake Receipt" in approved_dashboard

    update = deepcopy(source)
    update_digest = "1" * 64
    update["event_id"] = f"syllabus-source-{update_digest}"
    update["session_id"] = f"syllabus-intake-{update_digest}"
    update["occurred_at"] = "2026-08-20T00:00:00Z"
    update["source"]["sha256"] = update_digest
    update["source"]["source_version_id"] = f"sha256-{update_digest}"
    update["source"]["original_filename"] = "syllabus2026-revised.pdf"
    update["source"]["supersedes_source_version_id"] = source["source"]["source_version_id"]
    pending_update = projected(profile, [source, first, update])
    assert pending_update["grounding"]["status"] == "approved_update_pending"
    assert pending_update["grounding"]["active_source"]["sha256"] == EXPECTED_DIGEST
    assert pending_update["grounding"]["pending_sources"][0]["sha256"] == update_digest
    assert pending_update["syllabus"] == expected_topics
    assert "Pending update" in render_dashboard(pending_update).decode()

    update_approval = build_syllabus_approval_event(
        approval_request(
            update,
            event_id="approval-update",
            occurred_at="2026-08-20T00:10:00Z",
            supersedes="approval-1",
        )
    )
    updated_receipts = render_all_syllabus_receipts([source, first, update, update_approval])
    old_receipt = updated_receipts[
        f"syllabus/{source['source']['source_version_id']}.md"
    ].decode()
    assert "Historically Approved — Superseded" in old_receipt
    assert "Superseded by approval:** `approval-update`" in old_receipt
    assert update["source"]["source_version_id"] in old_receipt

    second = build_syllabus_approval_event(
        approval_request(
            source,
            event_id="approval-2",
            occurred_at="2026-08-19T00:20:00Z",
            supersedes="approval-1",
            correction=True,
        )
    )
    corrected = projected(profile, [source, first, second])
    title = next(
        item
        for item in corrected["grounding"]["effective_assertions"]
        if item["field"] == "course.title"
    )
    assert title["origin"] == "learner_correction"
    assert title["normalized_value"] == "Statistical Foundations for Data Science"
    assert title["document_value"] == "Statistical Foundations of Data Science"
    assert title["citations"]
    corrected_reference = grounded_assessment(
        "approval-2",
        ["learner-correction-course-title-v1"],
        occurred_at="2026-08-19T00:30:00Z",
    )
    assert projected(profile, [source, first, second, corrected_reference])["subjects"]

    backdated = grounded_assessment(
        "approval-1",
        ["st5201x.course.code"],
        event_id="grounded-before-backdated-approval",
        occurred_at="2026-08-19T00:30:00Z",
    )
    with pytest.raises(ValidationError, match="later than every learning event"):
        projected(profile, [source, first, backdated, second])


def test_context_intake_receipts_are_deterministic_pending_and_approved_snapshots() -> None:
    source = build_ingestion_event(
        FIXTURE,
        original_filename="syllabus2026.pdf",
        media_type="application/pdf",
        adapter_id="st5201x-2026-v1",
        occurred_at="2026-08-19T00:00:00Z",
    )
    relative = f"syllabus/{source['source']['source_version_id']}.md"
    pending = render_all_syllabus_receipts([source])[relative]
    approved_event = build_syllabus_approval_event(
        approval_request(source, event_id="approval-1", occurred_at="2026-08-19T00:10:00Z")
    )
    approved = render_all_syllabus_receipts([source, approved_event])[relative]
    snapshot_root = Path(__file__).parent / "fixtures" / "syllabus" / "st5201x"
    assert pending == snapshot_root.joinpath("expected-pending-receipt.md").read_bytes()
    assert approved == snapshot_root.joinpath("expected-approved-receipt.md").read_bytes()
    assert b"Approval Required" in pending
    assert b"Approved" in approved
    assert b"weeks_7_13_alignment" in approved
    assert b"## Canonical Page Text" in approved


def test_citation_history_is_pinned_and_mastery_neutral() -> None:
    source = build_ingestion_event(
        FIXTURE,
        original_filename="syllabus2026.pdf",
        media_type="application/pdf",
        adapter_id="st5201x-2026-v1",
        occurred_at="2026-07-31T00:00:00Z",
    )
    first = build_syllabus_approval_event(
        approval_request(source, event_id="approval-1", occurred_at="2026-07-31T00:10:00Z")
    )
    fixture_events = [
        json.loads(line)
        for line in (Path(__file__).parent / "fixtures" / "local_store" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[:5]
    ]
    title_id = "st5201x.course.title"
    fixture_events[0]["grounding"] = {
        "approval_event_id": "approval-1",
        "assertion_ids": [title_id],
    }
    fixture_events[-1]["grounding"] = {
        "approval_event_id": "approval-1",
        "assertion_ids": [title_id],
    }
    second = build_syllabus_approval_event(
        approval_request(
            source,
            event_id="approval-2",
            occurred_at="2026-08-02T00:00:00Z",
            supersedes="approval-1",
            correction=True,
        )
    )
    profile = initial_profile("ST5201X", "Learn statistics")
    grounded_events = [source, first, *fixture_events, second]
    grounded_state = projected(profile, grounded_events)
    ungrounded_learning = deepcopy(grounded_events)
    for event in ungrounded_learning:
        event.pop("grounding", None)
    ungrounded_state = projected(profile, ungrounded_learning)
    assert mastery_view(grounded_state) == mastery_view(ungrounded_state)

    receipt = render_all_receipts(profile, grounded_events)["sessions/session-1.md"].decode()
    assert "## Course Grounding" in receipt
    assert "Approval `approval-1`" in receipt
    assert "Statistical Foundations of Data Science" in receipt
    assert "Statistical Foundations for Data Science" not in receipt
    assert "page 1, chars" in receipt


def test_citation_validation_rejects_unknown_deferred_wrong_and_early() -> None:
    source = build_ingestion_event(
        FIXTURE,
        original_filename="syllabus2026.pdf",
        media_type="application/pdf",
        adapter_id="st5201x-2026-v1",
        occurred_at="2026-08-19T00:00:00Z",
    )
    first = build_syllabus_approval_event(
        approval_request(source, event_id="approval-1", occurred_at="2026-08-19T00:10:00Z")
    )
    profile = initial_profile("ST5201X", "Learn statistics")

    with pytest.raises(ValidationError, match="not settled and effective"):
        projected(profile, [source, first, grounded_assessment("approval-1", ["unknown-id"])])
    with pytest.raises(ValidationError, match="not settled and effective"):
        projected(
            profile,
            [
                source,
                first,
                grounded_assessment("approval-1", ["st5201x.schedule.weeks_7_13_alignment"]),
            ],
        )

    second = build_syllabus_approval_event(
        approval_request(
            source,
            event_id="approval-2",
            occurred_at="2026-08-19T00:15:00Z",
            supersedes="approval-1",
        )
    )
    with pytest.raises(ValidationError, match="expected active approval 'approval-2'"):
        projected(
            profile,
            [source, first, second, grounded_assessment("approval-1", ["st5201x.course.code"])],
        )
    early = grounded_assessment(
        "approval-1",
        ["st5201x.course.code"],
        event_id="early-grounding",
        occurred_at="2026-08-19T00:05:00Z",
    )
    with pytest.raises(ValidationError, match="no syllabus approval was active"):
        projected(profile, [source, early, first])


def test_validation_rejects_correction_id_collision_with_source_namespace() -> None:
    source = build_ingestion_event(
        FIXTURE,
        original_filename="syllabus2026.pdf",
        media_type="application/pdf",
        adapter_id="st5201x-2026-v1",
        occurred_at="2026-08-19T00:00:00Z",
    )
    request = approval_request(
        source,
        event_id="approval-colliding-correction",
        occurred_at="2026-08-19T00:10:00Z",
        correction=True,
    )
    request["corrections"][0]["correction_assertion_id"] = "st5201x.course.code"
    approval = build_syllabus_approval_event(request)

    with pytest.raises(ValidationError, match="must not collide with a source assertion ID"):
        projected(initial_profile("ST5201X", "Learn statistics"), [source, approval])


@pytest.mark.parametrize(
    ("admin_kind", "learning_kind"),
    [
        ("intake", "assessment"),
        ("intake", "session_completed"),
        ("approval", "assessment"),
        ("approval", "session_completed"),
    ],
)
def test_validation_rejects_later_learning_reuse_of_syllabus_admin_session(
    admin_kind: str,
    learning_kind: str,
) -> None:
    source = build_ingestion_event(
        FIXTURE,
        original_filename="syllabus2026.pdf",
        media_type="application/pdf",
        adapter_id="st5201x-2026-v1",
        occurred_at="2026-08-19T00:00:00Z",
    )
    approval = build_syllabus_approval_event(
        approval_request(
            source,
            event_id="approval-admin-session",
            occurred_at="2026-08-19T00:10:00Z",
        )
    )
    admin_session_id = (
        source["session_id"] if admin_kind == "intake" else approval["session_id"]
    )
    if learning_kind == "assessment":
        learning = grounded_assessment(
            "approval-admin-session",
            ["st5201x.course.code"],
            event_id=f"reuse-{admin_kind}-assessment",
            occurred_at="2026-08-19T00:20:00Z",
        )
        learning["session_id"] = admin_session_id
    else:
        learning = {
            "event_id": f"reuse-{admin_kind}-completion",
            "evidence_event_ids": [],
            "kind": "session_completed",
            "next_action": "Continue",
            "next_review_date": None,
            "occurred_at": "2026-08-19T00:20:00Z",
            "receipt_schema_version": 1,
            "schema_version": 1,
            "session_id": admin_session_id,
        }

    with pytest.raises(ValidationError, match="must not reuse a syllabus administrative session"):
        projected(
            initial_profile("ST5201X", "Learn statistics"),
            [source, approval, learning],
        )


def test_markdown_grounding_receipts_neutralize_active_syntax_and_page_fences() -> None:
    source = build_ingestion_event(
        FIXTURE,
        original_filename="syllabus2026.pdf",
        media_type="application/pdf",
        adapter_id="st5201x-2026-v1",
        occurred_at="2026-08-19T00:00:00Z",
    )
    source["source"]["original_filename"] = "![[note]] [link](target) `tick`"
    source["pages"][0]["text"] += "\n````\n![[page-note]]\n![image](target)"
    source["pages"][0]["text_sha256"] = sha256_bytes(
        source["pages"][0]["text"].encode("utf-8")
    )
    relative = f"syllabus/{source['source']['source_version_id']}.md"
    pending = render_all_syllabus_receipts([source])[relative].decode()
    header, appendix = pending.split("## Canonical Page Text", maxsplit=1)
    assert "![[note]]" not in header
    assert r"!\[\[note\]\] \[link\](target) \`tick\`" in header
    assert "`````text\n" in appendix
    assert "\n`````\n" in appendix
    assert "![[page-note]]" in appendix

    approval_request_value = approval_request(
        source,
        event_id="approval-markdown",
        occurred_at="2026-08-19T00:10:00Z",
        correction=True,
    )
    approval_request_value["corrections"][0]["rationale"] = "Use ![[private]] [link](x)."
    approval = build_syllabus_approval_event(approval_request_value)
    approved = render_all_syllabus_receipts([source, approval])[relative].decode()
    before_appendix = approved.split("## Canonical Page Text", maxsplit=1)[0]
    assert "![[private]]" not in before_appendix
    assert r"!\[\[private\]\] \[link\](x)." in before_appendix


def test_syllabus_generated_tree_rebuild_validation_and_symlink_safety(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    domain_id, directory = init_domain(root)
    directory.joinpath("syllabus").write_text("obstruction\n", encoding="utf-8")
    obstructed = error(run_cli(root, "validate", "--domain-id", domain_id))
    assert "syllabus path must be a directory" in obstructed["message"]
    directory.joinpath("syllabus").unlink()
    output(ingest(root, domain_id, 0))
    receipt = next(directory.joinpath("syllabus").glob("*.md"))
    expected = {
        "state.json": directory.joinpath("state.json").read_bytes(),
        "dashboard.md": directory.joinpath("dashboard.md").read_bytes(),
        receipt.relative_to(directory).as_posix(): receipt.read_bytes(),
    }
    for relative in expected:
        directory.joinpath(relative).unlink()
    rebuilt = output(run_cli(root, "rebuild", "--domain-id", domain_id))
    assert rebuilt["status"] == "rebuilt"
    for relative, content in expected.items():
        assert directory.joinpath(relative).read_bytes() == content
    assert output(run_cli(root, "validate", "--domain-id", domain_id))["status"] == "valid"

    receipt.write_text("drift\n", encoding="utf-8")
    drift = error(run_cli(root, "validate", "--domain-id", domain_id))
    assert receipt.relative_to(directory).as_posix() in drift["message"]
    output(run_cli(root, "rebuild", "--domain-id", domain_id))

    orphan = directory / "syllabus" / "orphan.md"
    orphan.write_text("orphan\n", encoding="utf-8")
    invalid = error(run_cli(root, "validate", "--domain-id", domain_id))
    assert "syllabus/orphan.md" in invalid["message"]
    orphan.unlink()

    outside = tmp_path / "outside.md"
    outside.write_bytes(receipt.read_bytes())
    receipt.unlink()
    receipt.symlink_to(outside)
    linked = error(run_cli(root, "validate", "--domain-id", domain_id))
    assert receipt.relative_to(directory).as_posix() in linked["message"]
    blocked = error(run_cli(root, "rebuild", "--domain-id", domain_id))
    assert "generated target path must not contain symlinks" in blocked["message"]

    receipt.unlink()
    directory.joinpath("syllabus").rmdir()
    outside_directory = tmp_path / "outside-syllabus"
    outside_directory.mkdir()
    directory.joinpath("syllabus").symlink_to(outside_directory, target_is_directory=True)
    linked_directory = error(run_cli(root, "validate", "--domain-id", domain_id))
    assert "syllabus directory must not be a symlink" in linked_directory["message"]
