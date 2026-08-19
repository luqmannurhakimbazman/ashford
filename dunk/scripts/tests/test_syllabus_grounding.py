"""Focused tests for the generic canonical syllabus lifecycle."""

from __future__ import annotations

import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dln_store.grounding import _decision_view, reduce_grounding_timeline
from dln_store.schema import ValidationError, canonical_json, pretty_json, sha256_bytes, syllabus_approval_set_sha256
from dln_store.store import LocalStore


def tree(directory: Path) -> dict[str, bytes]:
    return {p.relative_to(directory).as_posix(): p.read_bytes() for p in directory.rglob("*") if p.is_file() and ".dln-transaction" not in p.parts}


def initialized(tmp_path: Path) -> tuple[LocalStore, str, Path]:
    store = LocalStore(tmp_path)
    domain_id = store.init("Generic Systems", "Learn from an authoritative syllabus")["domain_id"]
    return store, domain_id, tmp_path / "domains" / domain_id


def prepared(text: str = "Course: Generic Systems\nTopic: Trees\nExam date: TBA\n", *, media_type: str = "application/pdf") -> dict:
    return {
        "prepared_schema_version": 1,
        "media_type": media_type,
        "normalization": {"policy_id": "fixture-normalization-v1", "unicode": "NFC", "line_endings": "LF"},
        "units": [{"unit_id": "page:1", "kind": "page", "label": "Page 1", "text": text, "text_sha256": sha256_bytes(text.encode())}],
    }


def located(text: str, quote: str, **values: object) -> dict:
    start = text.index(quote)
    return {**values, "locators": [{"unit_id": "page:1", "start_char": start, "end_char": start + len(quote), "quote": quote}]}


def lifecycle(store: LocalStore, domain_id: str, *, role: str = "authoritative", raw: bytes = b"generic-source-v1", predecessor: str | None = None, base_minute: int = 0) -> tuple[dict, dict, dict]:
    text = "Course: Generic Systems\nTopic: Trees\nExam date: TBA\n"
    source = store._install_prepared_syllabus(domain_id, base_minute * 3, raw_bytes=raw, prepared_document=prepared(text), role=role,
                                    display_name="generic-syllabus.pdf", occurred_at=f"2026-08-19T00:{base_minute:02d}:00Z",
                                    supersedes_source_version_id=predecessor)
    proposals = [
        located(text, "Trees", predicate="coverage.topic", label="Coverage topic", semantic_roles=["planning_topic"], value_type="text", status="specified", value="Trees"),
        located(text, "TBA", predicate="assessment.exam.date", label="Exam date", semantic_roles=["assessment"], value_type="date", status="explicitly_unknown", value=None),
    ]
    proposal = store.propose_syllabus(domain_id, base_minute * 3 + 1, prepared_event_id=source["source_event_id"],
                                      occurred_at=f"2026-08-19T00:{base_minute + 1:02d}:00Z",
                                      producer={"trust": "external_unverified", "name": "test-builder", "version": "1"}, proposals=proposals)
    decision = store.decide_syllabus(domain_id, base_minute * 3 + 2, proposal_event_id=proposal["proposal_event_id"],
                                     occurred_at=f"2026-08-19T00:{base_minute + 2:02d}:00Z",
                                     accepted_proposal_ids=proposal["proposal_ids"], deferred_proposal_ids=[], rejected_proposal_ids=[], corrections=[])
    return source, proposal, decision


def test_prepare_propose_decide_cas_content_and_offline_rebuild(tmp_path: Path) -> None:
    store, domain_id, directory = initialized(tmp_path)
    source, proposal, decision = lifecycle(store, domain_id)
    source_digest = source["source_sha256"]
    prepared_digest = source["prepared_document_sha256"]
    assert directory.joinpath("sources", "sha256", source_digest).read_bytes() == b"generic-source-v1"
    prepared_bytes = directory.joinpath("prepared", "sha256", f"{prepared_digest}.json").read_bytes()
    assert not prepared_bytes.endswith(b"\n")
    assert sha256_bytes(prepared_bytes) == prepared_digest
    content = store.syllabus_content(domain_id, source["source_event_id"])
    assert content["storage"] == "cas"
    assert content["raw_bytes"] == b"generic-source-v1"
    assert content["prepared_bytes"] == prepared_bytes
    grounding = store.context(domain_id)["state"]["grounding"]
    assert grounding["status"] == "approved"
    assert grounding["active_decision"]["event_id"] == decision["decision_event_id"]
    assert grounding["active_decision"]["reference_field"] == "decision_event_id"
    assert grounding["planning_topics"] == [{"assertion_ids": [proposal["proposal_ids"][0]], "citable": True, "label": "Trees"}]
    before_cas = {p.relative_to(directory).as_posix(): p.read_bytes() for p in (directory / "sources").rglob("*") if p.is_file()} | {p.relative_to(directory).as_posix(): p.read_bytes() for p in (directory / "prepared").rglob("*") if p.is_file()}
    store.rebuild(domain_id)
    store.rebuild(domain_id)
    after_cas = {p.relative_to(directory).as_posix(): p.read_bytes() for p in (directory / "sources").rglob("*") if p.is_file()} | {p.relative_to(directory).as_posix(): p.read_bytes() for p in (directory / "prepared").rglob("*") if p.is_file()}
    assert after_cas == before_cas
    assert store.validate(domain_id)["canonical_content"]["referenced_sources"] == 1


def test_prepare_propose_decide_exact_retries_are_idempotent(tmp_path: Path) -> None:
    store, domain_id, _ = initialized(tmp_path)
    source, proposal, decision = lifecycle(store, domain_id)
    text = "Course: Generic Systems\nTopic: Trees\nExam date: TBA\n"
    retry_source = store._install_prepared_syllabus(domain_id, 3, raw_bytes=b"generic-source-v1", prepared_document=prepared(text), role="authoritative",
                                          display_name="retry-name.pdf", occurred_at="2026-08-19T00:59:00Z")
    assert retry_source["status"] == "noop" and retry_source["revision"] == 3
    retry_proposal = store.propose_syllabus(domain_id, 3, prepared_event_id=source["source_event_id"], occurred_at="2026-08-19T00:58:00Z",
                                            producer={"trust": "external_unverified", "name": "test-builder", "version": "1"},
                                            proposals=[located(text, "Trees", predicate="coverage.topic", label="Coverage topic", semantic_roles=["planning_topic"], value_type="text", status="specified", value="Trees"),
                                                       located(text, "TBA", predicate="assessment.exam.date", label="Exam date", semantic_roles=["assessment"], value_type="date", status="explicitly_unknown", value=None)])
    assert retry_proposal["status"] == "noop" and retry_proposal["proposal_event_id"] == proposal["proposal_event_id"]
    retry_decision = store.decide_syllabus(domain_id, 3, proposal_event_id=proposal["proposal_event_id"], occurred_at="2026-08-19T00:57:00Z",
                                           accepted_proposal_ids=proposal["proposal_ids"], deferred_proposal_ids=[], rejected_proposal_ids=[], corrections=[])
    assert retry_decision["status"] == "noop" and retry_decision["decision_event_id"] == decision["decision_event_id"]


def test_all_reserved_kinds_rejected_by_generic_commit_without_writes(tmp_path: Path) -> None:
    store, domain_id, directory = initialized(tmp_path)
    source, proposal, decision = lifecycle(store, domain_id)
    events = [json.loads(line) for line in directory.joinpath("events.jsonl").read_text().splitlines()]
    before = tree(directory)
    for event in events:
        with pytest.raises(ValidationError, match="reserved syllabus"):
            store.commit(domain_id, 3, {"events": [event]})
        assert tree(directory) == before
    legacy_fixture = Path(__file__).parent / "fixtures" / "syllabus" / "legacy-v1" / "events.jsonl"
    for legacy_event in [json.loads(line) for line in legacy_fixture.read_text().splitlines()[:2]]:
        with pytest.raises(ValidationError, match="reserved syllabus"):
            store.commit(domain_id, 3, {"events": [legacy_event]})
        assert tree(directory) == before


def test_existing_nonmatching_cas_path_is_never_overwritten(tmp_path: Path) -> None:
    store, domain_id, directory = initialized(tmp_path)
    raw = b"collision-candidate"
    path = directory / "sources" / "sha256" / sha256_bytes(raw)
    path.write_bytes(b"preexisting-corrupt-bytes")
    before = tree(directory)
    with pytest.raises(ValidationError, match="collision/corruption"):
        store._install_prepared_syllabus(domain_id, 0, raw_bytes=raw, prepared_document=prepared("Topic: Trees\n"), role="authoritative",
                               display_name="generic.pdf", occurred_at="2026-08-19T00:00:00Z")
    assert tree(directory) == before


def test_locator_and_decision_failures_are_write_free(tmp_path: Path) -> None:
    store, domain_id, directory = initialized(tmp_path)
    text = "Topic: Graphs\nAmbiguous room: A or B\n"
    source = store._install_prepared_syllabus(domain_id, 0, raw_bytes=b"source", prepared_document=prepared(text), role="authoritative",
                                    display_name="course.pdf", occurred_at="2026-08-19T00:00:00Z")
    before = tree(directory)
    bad = [located(text, "Graphs", predicate="coverage.topic", label="Topic", semantic_roles=["planning_topic"], value_type="text", status="specified", value="Graphs")]
    bad[0]["locators"][0]["quote"] = "Grapes"
    with pytest.raises(ValidationError, match="quote"):
        store.propose_syllabus(domain_id, 1, prepared_event_id=source["source_event_id"], occurred_at="2026-08-19T00:01:00Z",
                               producer={"trust": "external_unverified", "name": "test", "version": "1"}, proposals=bad)
    assert tree(directory) == before
    ambiguous = [located(text, "A or B", predicate="class.room", label="Room", semantic_roles=["logistics"], value_type="text", status="ambiguous", value=None,
                         ambiguity={"reason": "two rooms listed", "unresolved_dimensions": ["room"], "candidates": []})]
    proposal = store.propose_syllabus(domain_id, 1, prepared_event_id=source["source_event_id"], occurred_at="2026-08-19T00:01:00Z",
                                      producer={"trust": "external_unverified", "name": "test", "version": "1"}, proposals=ambiguous)
    before_decision = tree(directory)
    with pytest.raises(ValidationError, match="ambiguous proposals cannot be accepted"):
        store.decide_syllabus(domain_id, 2, proposal_event_id=proposal["proposal_event_id"], occurred_at="2026-08-19T00:02:00Z",
                              accepted_proposal_ids=proposal["proposal_ids"], deferred_proposal_ids=[], rejected_proposal_ids=[], corrections=[])
    assert tree(directory) == before_decision
    with pytest.raises(ValidationError, match="completely and exactly"):
        store.decide_syllabus(domain_id, 2, proposal_event_id=proposal["proposal_event_id"], occurred_at="2026-08-19T00:02:00Z",
                              accepted_proposal_ids=[], deferred_proposal_ids=[], rejected_proposal_ids=[], corrections=[])
    assert tree(directory) == before_decision
    store.decide_syllabus(domain_id, 2, proposal_event_id=proposal["proposal_event_id"], occurred_at="2026-08-19T00:02:00Z",
                          accepted_proposal_ids=[], deferred_proposal_ids=[], rejected_proposal_ids=[],
                          corrections=[{"target_proposal_id": proposal["proposal_ids"][0], "predicate": "class.room", "semantic_roles": ["logistics"],
                                        "value_type": "text", "status": "specified", "value": "A", "rationale": "Learner confirmed room A", "origin": "learner_correction"}])
    effective = store.context(domain_id)["state"]["grounding"]["effective_assertions"][0]
    assert effective["origin"] == "learner_correction" and effective["citations"] == []
    assert effective["document_context"][0]["quote"] == "A or B"


def test_planning_topic_correction_requires_text_and_is_write_free(tmp_path: Path) -> None:
    store, domain_id, directory = initialized(tmp_path)
    text = "Topic: Trees\n"
    source = store._install_prepared_syllabus(
        domain_id, 0, raw_bytes=b"planning-topic", prepared_document=prepared(text),
        role="authoritative", display_name="course.pdf", occurred_at="2026-08-19T00:00:00Z",
    )
    proposal = store.propose_syllabus(
        domain_id, 1, prepared_event_id=source["source_event_id"], occurred_at="2026-08-19T00:01:00Z",
        producer={"trust": "external_unverified", "name": "test", "version": "1"},
        proposals=[located(text, "Trees", predicate="coverage.topic", label="Topic", semantic_roles=["planning_topic"],
                           value_type="text", status="specified", value="Trees")],
    )
    before = tree(directory)
    with pytest.raises(ValidationError, match="planning_topic corrections must use text values"):
        store.decide_syllabus(
            domain_id, 2, proposal_event_id=proposal["proposal_event_id"], occurred_at="2026-08-19T00:02:00Z",
            accepted_proposal_ids=[], deferred_proposal_ids=[], rejected_proposal_ids=[],
            corrections=[{"target_proposal_id": proposal["proposal_ids"][0], "predicate": "coverage.topic",
                          "semantic_roles": ["planning_topic"], "value_type": "integer", "status": "specified",
                          "value": 7, "rationale": "Adversarial non-text topic", "origin": "learner_correction"}],
        )
    assert tree(directory) == before


def test_generic_and_legacy_correction_ids_cannot_collide_with_source_assertions() -> None:
    proposals = [
        {"proposal_id": "source-a", "predicate": "coverage.topic", "semantic_roles": ["planning_topic"],
         "value_type": "text", "status": "specified", "value": "A", "locators": []},
        {"proposal_id": "source-b", "predicate": "coverage.topic", "semantic_roles": ["planning_topic"],
         "value_type": "text", "status": "specified", "value": "B", "locators": []},
    ]
    generic = {
        "accepted_proposal_ids": [], "deferred_proposal_ids": ["source-b"], "rejected_proposal_ids": [],
        "corrections": [{"correction_id": "source-b", "target_proposal_id": "source-a"}],
        "decision_set_sha256": "unused",
    }
    legacy = {
        "accepted_assertion_ids": [], "deferred_assertion_ids": ["source-b"],
        "corrections": [{"correction_assertion_id": "source-b", "target_assertion_id": "source-a"}],
        "approval_set_sha256": "unused",
    }
    for decision, is_legacy in ((generic, False), (legacy, True)):
        with pytest.raises(ValidationError, match="must not collide with proposal/source assertion IDs"):
            _decision_view(decision, {}, proposals, legacy=is_legacy)


def test_authoritative_update_pending_and_nonforking_lineage(tmp_path: Path) -> None:
    store, domain_id, directory = initialized(tmp_path)
    first, _, first_decision = lifecycle(store, domain_id)
    second = store._install_prepared_syllabus(domain_id, 3, raw_bytes=b"generic-source-v2", prepared_document=prepared("Topic: Graphs\n"), role="authoritative",
                                    display_name="update.pdf", occurred_at="2026-08-19T00:03:00Z", supersedes_source_version_id=first["source_version_id"])
    grounding = store.context(domain_id)["state"]["grounding"]
    assert second["grounding_status"] == "approved_update_pending"
    assert grounding["status"] == "approved_update_pending"
    assert grounding["active_decision"]["event_id"] == first_decision["decision_event_id"]
    before = tree(directory)
    with pytest.raises(ValidationError, match="forks/skips"):
        store._install_prepared_syllabus(domain_id, 4, raw_bytes=b"fork", prepared_document=prepared("Topic: Fork\n"), role="authoritative",
                               display_name="fork.pdf", occurred_at="2026-08-19T00:04:00Z", supersedes_source_version_id=first["source_version_id"])
    assert tree(directory) == before
    assert second["source_event_id"] in directory.joinpath("events.jsonl").read_text()


def test_supplement_visible_but_authority_planning_and_mastery_neutral(tmp_path: Path) -> None:
    store, domain_id, _ = initialized(tmp_path)
    _, _, decision = lifecycle(store, domain_id)
    before = store.context(domain_id)["state"]
    supplement, proposal, supplement_decision = lifecycle(store, domain_id, role="supplement", raw=b"supplement", base_minute=1)
    assert supplement["grounding_status"] == "approved"
    assert supplement_decision["grounding_status"] == "approved"
    after = store.context(domain_id)["state"]
    assert after["grounding"]["active_decision"]["event_id"] == decision["decision_event_id"]
    assert after["grounding"]["planning_topics"] == before["grounding"]["planning_topics"]
    assert after["grounding"]["supplements"][0]["source_version_id"] == supplement["source_version_id"]
    mastery_keys = ["stage", "generation", "subjects", "calibration", "archived_exams"]
    assert canonical_json({key: before[key] for key in mastery_keys}) == canonical_json({key: after[key] for key in mastery_keys})


def test_missing_corrupt_and_orphaned_cas_are_integrity_failures(tmp_path: Path) -> None:
    store, domain_id, directory = initialized(tmp_path)
    source, _, _ = lifecycle(store, domain_id)
    raw_path = directory / "sources" / "sha256" / source["source_sha256"]
    original = raw_path.read_bytes()
    raw_path.unlink()
    with pytest.raises(ValidationError, match="missing canonical"):
        store.rebuild(domain_id)
    raw_path.write_bytes(b"corrupt")
    with pytest.raises(ValidationError, match="corrupt"):
        store.context(domain_id)
    raw_path.write_bytes(original)
    orphan = directory / "sources" / "sha256" / ("f" * 64)
    orphan.write_bytes(b"orphan")
    with pytest.raises(ValidationError, match="orphaned canonical"):
        store.validate(domain_id)
    raw_path.unlink()
    with pytest.raises(ValidationError) as combined:
        store.validate(domain_id)
    assert "missing canonical" in str(combined.value) and "orphaned canonical" in str(combined.value)


def test_chronology_admin_session_and_payload_bounds_are_enforced(tmp_path: Path) -> None:
    store, domain_id, directory = initialized(tmp_path)
    text = "Topic: Trees\n"
    source = store._install_prepared_syllabus(domain_id, 0, raw_bytes=b"chronology", prepared_document=prepared(text), role="authoritative",
                                    display_name="generic.pdf", occurred_at="2026-08-19T00:10:00Z")
    proposal_input = [located(text, "Trees", predicate="coverage.topic", label="Topic", semantic_roles=["planning_topic"], value_type="text", status="specified", value="Trees")]
    before = tree(directory)
    with pytest.raises(ValidationError, match="cannot precede"):
        store.propose_syllabus(domain_id, 1, prepared_event_id=source["source_event_id"], occurred_at="2026-08-19T00:09:00Z",
                               producer={"trust": "external_unverified", "name": "test", "version": "1"}, proposals=proposal_input)
    assert tree(directory) == before
    oversized = deepcopy(proposal_input); oversized[0]["note"] = "x" * 4097
    with pytest.raises(ValidationError, match="4096"):
        store.propose_syllabus(domain_id, 1, prepared_event_id=source["source_event_id"], occurred_at="2026-08-19T00:11:00Z",
                               producer={"trust": "external_unverified", "name": "test", "version": "1"}, proposals=oversized)
    proposal = store.propose_syllabus(domain_id, 1, prepared_event_id=source["source_event_id"], occurred_at="2026-08-19T00:11:00Z",
                                      producer={"trust": "external_unverified", "name": "test", "version": "1"}, proposals=proposal_input)
    with pytest.raises(ValidationError, match="cannot precede"):
        store.decide_syllabus(domain_id, 2, proposal_event_id=proposal["proposal_event_id"], occurred_at="2026-08-19T00:10:30Z",
                              accepted_proposal_ids=proposal["proposal_ids"], deferred_proposal_ids=[], rejected_proposal_ids=[], corrections=[])
    admin_session = json.loads(directory.joinpath("events.jsonl").read_text().splitlines()[0])["session_id"]
    reset = {"schema_version": 1, "event_id": "bad-admin-reuse", "session_id": admin_session, "occurred_at": "2026-08-19T00:12:00Z", "kind": "domain_reset"}
    with pytest.raises(ValidationError, match="administrative session"):
        store.commit(domain_id, 2, {"events": [reset]})


def test_legacy_unresolved_correction_remains_uncitable() -> None:
    fixture = Path(__file__).parent / "fixtures" / "syllabus" / "legacy-v1" / "events.jsonl"
    source, approval, _ = [json.loads(line) for line in fixture.read_text().splitlines()]
    approval = deepcopy(approval)
    approval["accepted_assertion_ids"] = []
    approval["corrections"] = [{"correction_assertion_id": "legacy-unresolved-correction", "target_assertion_id": "legacy-topic-graphs",
                                "field": "coverage.topic", "status": "unresolved", "normalized_value": {"possibilities": ["Graphs", "Trees"]},
                                "rationale": "historical ambiguity", "origin": "learner_correction"}]
    approval["approval_set_sha256"] = syllabus_approval_set_sha256(approval)
    timeline = reduce_grounding_timeline([source, approval])
    view = timeline.current_view()
    assert view is not None
    assert view["effective_assertions"] == []
    assert view["unresolved_assertions"][0]["status"] == "ambiguous"
    with pytest.raises(ValidationError):
        timeline.resolve_assertion(approval["event_id"], "legacy-unresolved-correction")


def test_rejected_proposals_are_visible_but_not_citable(tmp_path: Path) -> None:
    store, domain_id, directory = initialized(tmp_path)
    text = "Topic: Optional Lab\n"
    source = store._install_prepared_syllabus(domain_id, 0, raw_bytes=b"reject", prepared_document=prepared(text), role="authoritative",
                                    display_name="generic.pdf", occurred_at="2026-08-19T00:00:00Z")
    proposal = store.propose_syllabus(domain_id, 1, prepared_event_id=source["source_event_id"], occurred_at="2026-08-19T00:01:00Z",
                                      producer={"trust": "external_unverified", "name": "test", "version": "1"},
                                      proposals=[located(text, "Optional Lab", predicate="coverage.topic", label="Optional lab", semantic_roles=["planning_topic"], value_type="text", status="specified", value="Optional Lab")])
    store.decide_syllabus(domain_id, 2, proposal_event_id=proposal["proposal_event_id"], occurred_at="2026-08-19T00:02:00Z",
                          accepted_proposal_ids=[], deferred_proposal_ids=[], rejected_proposal_ids=proposal["proposal_ids"], corrections=[])
    state = store.context(domain_id)["state"]
    assert state["grounding"]["planning_topics"] == []
    receipt = directory / "syllabus" / f"{source['source_version_id']}.md"
    rendered = receipt.read_text()
    assert "## Rejected" in rendered and proposal["proposal_ids"][0] in rendered and "Rejected:" in rendered


def test_legacy_v1_fixture_replays_without_pdf_manifest_or_cas(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "syllabus" / "legacy-v1"
    profile = json.loads(fixture.joinpath("profile.yaml").read_text())
    directory = tmp_path / "domains" / profile["domain_id"]
    (directory / "sessions").mkdir(parents=True)
    shutil.copyfile(fixture / "profile.yaml", directory / "profile.yaml")
    shutil.copyfile(fixture / "events.jsonl", directory / "events.jsonl")
    store = LocalStore(tmp_path)
    state = store.context(profile["domain_id"])["state"]
    assert state["grounding"]["status"] == "approved"
    assert state["grounding"]["active_decision"]["reference_field"] == "approval_event_id"
    assert state["grounding"]["planning_topics"][0]["label"] == "Graphs"
    source_event_id = json.loads(fixture.joinpath("events.jsonl").read_text().splitlines()[0])["event_id"]
    content = store.syllabus_content(profile["domain_id"], source_event_id)
    assert content["storage"] == "legacy_text_only"
    assert content["raw_bytes"] is None and content["prepared_bytes"] is None
    historical = json.loads(fixture.joinpath("events.jsonl").read_text().splitlines()[2])
    continued = deepcopy(historical)
    continued.update({
        "event_id": "legacy-assessment-continued",
        "session_id": "legacy-learning-session-continued",
        "occurred_at": "2025-01-01T00:03:00Z",
        "task_id": "legacy-task-continued",
    })
    continued["grounding"] = {
        "decision_event_id": "historical-approval-1",
        "assertion_ids": ["legacy-topic-graphs"],
    }
    result = store.commit(profile["domain_id"], profile["revision"], {"events": [continued]})
    assert result["status"] == "committed"
    store.rebuild(profile["domain_id"])
    assert not directory.joinpath("sources").exists()
    assert not directory.joinpath("prepared").exists()
    assert store.validate(profile["domain_id"])["status"] == "valid"
