"""Active Dunk prompt/config contracts for the local-first Item 2 architecture."""

from __future__ import annotations

import json
from pathlib import Path

DUNK = Path(__file__).resolve().parents[2]
SKILLS = DUNK / "skills"
DLN_REFS = SKILLS / "dln" / "references"

PHASE_SKILLS = {
    "dln-dot": {
        "stage": "Acquire/Discriminate",
        "store_stage": "acquire",
        "operations": ("acquire", "discriminate"),
    },
    "dln-linear": {
        "stage": "Relate/Abstract",
        "store_stage": "relate",
        "operations": ("relate", "abstract"),
    },
    "dln-network": {
        "stage": "Predict/Revise/Compress",
        "store_stage": "revise",
        "operations": ("predict",),
    },
}

SHARED_REFERENCES = (
    "local-store-schema.md",
    "local-persistence-protocol.md",
    "syllabus-grounding-protocol.md",
    "evidence-protocol.md",
    "session-receipt-format.md",
)

LEGACY_FILES = (
    DLN_REFS / "init-template.md",
    DLN_REFS / "merge-payload-schema.md",
    DLN_REFS / "merge-protocol.md",
    DUNK / "scripts" / "ks-merge.py",
    DUNK / "scripts" / "tests" / "test_ks_merge.py",
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def active_markdown() -> list[Path]:
    paths = [SKILLS / "dln" / "SKILL.md", DUNK / "agents" / "dln-syllabus.md"]
    for skill in PHASE_SKILLS:
        paths.extend(
            (
                SKILLS / skill / "SKILL.md",
                SKILLS / skill / "evaluations" / "trigger-tests.md",
            )
        )
        paths.extend((SKILLS / skill / "references").glob("*.md"))
    paths.extend(
        (
            SKILLS / "dln" / "evaluations" / "trigger-tests.md",
            SKILLS / "dln-compress" / "SKILL.md",
            SKILLS / "dln-compress" / "evaluations" / "trigger-tests.md",
            DLN_REFS / "local-store-schema.md",
            DLN_REFS / "local-persistence-protocol.md",
            DLN_REFS / "syllabus-grounding-protocol.md",
            DLN_REFS / "evidence-protocol.md",
            DLN_REFS / "session-receipt-format.md",
            DLN_REFS / "sync-protocol.md",
        )
    )
    return sorted(set(paths))


def test_active_remote_sync_path_is_removed() -> None:
    assert not (DUNK / "agents" / "dln-sync.md").exists()
    assert not (DUNK / "hooks" / "hooks.json").exists()
    assert not (DUNK / "scripts" / "validate-ks-markers.sh").exists()

    config = json.loads(read(DUNK / ".mcp.json"))
    assert all(name.casefold() != "notion" for name in config["mcpServers"])

    for path in active_markdown():
        text = read(path).casefold()
        assert "dln-sync" not in text, path
        assert "mcp__plugin_dunk_notion" not in text, path
        assert "notion-write" not in text, path


def test_phase_skills_link_shared_local_contracts_and_operations() -> None:
    for skill, contract in PHASE_SKILLS.items():
        text = read(SKILLS / skill / "SKILL.md")
        for reference in SHARED_REFERENCES:
            assert reference in text, (skill, reference)
        assert contract["stage"] in text
        assert f'state.stage == "{contract["store_stage"]}"' in text
        for operation in contract["operations"]:
            assert operation in text
        assert "stable" in text.casefold()
        assert "retry" in text.casefold() or "retries" in text.casefold()
        assert "session_completed" in text
        assert "Session Receipt" in text
        assert "state.grounding" in text
        assert "decision_event_id" in text
        assert "assertion_ids" in text
        assert "supplemental" in text
        assert "unresolved" in text
        assert "approved_update_pending" in text
        assert "pending-source" in text
        assert "merge-protocol.md" not in text
        assert "merge-payload-schema.md" not in text


def test_orchestrator_uses_context_routing_local_cli_and_receipt() -> None:
    text = read(SKILLS / "dln" / "SKILL.md")
    assert "dln-store.py" in text
    assert "Route solely from `state.stage`" in text
    for stage in ("acquire", "relate", "revise"):
        assert f"`{stage}`" in text
    assert "domain_reset" in text
    assert "profile_patch" in text
    assert "session_completed" in text
    assert "sole canonical summary" in text
    assert "spacing was not measured" in text
    assert "stable across retries" in text
    for runtime_token in ("uv run", "--project", "--python 3.10.20", "--frozen"):
        assert runtime_token in text
    assert 'python3 "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py"' in text
    assert "UV_PROJECT_ENVIRONMENT" in text
    for token in (
        "prepare-syllabus",
        "syllabus-content",
        "propose-syllabus",
        "decide-syllabus",
        "state.grounding",
        "transient attachment",
        "do not patch `profile.syllabus`",
        "ungrounded curriculum",
        "both new and existing domains",
        "approved_update_pending",
        "pending-source proposals",
    ):
        assert token in text


def test_syllabus_agent_is_return_only_and_has_no_remote_write_tool() -> None:
    text = read(DUNK / "agents" / "dln-syllabus.md")
    frontmatter = text.split("---", 2)[1]
    assert "Notion" not in frontmatter
    assert "notion-" not in frontmatter.casefold()
    assert "tools: []" in frontmatter
    assert "profile_patch" in text
    assert '"research_availability"' in text
    assert '"grounding_status": "ungrounded"' in text
    assert "never document-derived" in text
    assert "do not accept attachments" in text.casefold()
    assert "parent owns all persistence" in text.casefold()


def test_internal_compressor_cannot_create_learning_claims_or_artifacts() -> None:
    text = read(SKILLS / "dln-compress" / "SKILL.md").casefold()
    for phrase in (
        "do not create events",
        "do not infer or decide proficiency",
        "do not alter, summarize, or compress the learner's pedagogical model",
        "do not produce a learner-facing summary or artifact",
        "do not drop, rename, or flatten `state.grounding`",
        "do not present legacy ungrounded topics as citable",
    ):
        assert phrase in text
    assert "dln-store context" in text
    for token in ("planning_topics", "assertion_ids", "citable", "approved_update_pending"):
        assert token in text


def test_shared_contracts_cover_storage_evidence_and_receipt_boundaries() -> None:
    for reference in SHARED_REFERENCES:
        assert (DLN_REFS / reference).is_file()

    schema = read(DLN_REFS / "local-store-schema.md")
    for token in (
        "profile.yaml",
        "events.jsonl",
        "state.json",
        "dashboard.md",
        "sessions/<session-id>.md",
        "expected",
        "independent",
        "supported",
    ):
        assert token in schema

    persistence = read(DLN_REFS / "local-persistence-protocol.md")
    for runtime_token in ("uv run", "--project", "--python 3.10.20", "--frozen"):
        assert runtime_token in persistence
    for token in (
        "--expected-revision",
        "Exit `3`",
        "Retry once",
        "doctor --recover",
        "prepare-syllabus",
        "syllabus-content",
        "propose-syllabus",
        "decide-syllabus",
    ):
        assert token.casefold() in persistence.casefold()

    storage = read(DUNK / "LOCAL_STORAGE.md")
    for runtime_token in ("uv sync", "uv run", "--project", "--python 3.10.20", "--frozen", "pypdf.__version__ == '6.14.2'"):
        assert runtime_token in storage
    for token in (
        "Obsidian vault", ".DS_Store", ".locks/", "doctor --domain-id",
        "import-legacy-ks", "--root <path>", "rebuild` once per existing domain",
    ):
        assert token in storage
    for document in (persistence, storage):
        assert "UV_PROJECT_ENVIRONMENT" in document
        assert 'python3 "$STORE" validate' in document
        assert "CLAUDE_PLUGIN_ROOT}/scripts/.venv" not in document
        for command in document.replace("\\\n", " ").splitlines():
            if "uv run" not in command:
                continue
            assert "prepare-syllabus" in command or "import pypdf" in command, command

    grounding = read(DLN_REFS / "syllabus-grounding-protocol.md")
    for token in (
        "prepare-syllabus",
        "syllabus-content",
        "propose-syllabus",
        "decide-syllabus",
        "decision_required",
        "decision_event_id",
        "ambiguous",
        "supplement",
        "never learner evidence",
    ):
        assert token in grounding
    assert "st5201x" not in grounding.casefold()

    evidence = read(DLN_REFS / "evidence-protocol.md")
    for operation in ("acquire", "discriminate", "relate", "abstract", "predict"):
        assert f"`{operation}`" in evidence
    assert "dialogue length" in evidence.casefold()
    assert "novelty: novel" in evidence

    receipt = read(DLN_REFS / "session-receipt-format.md")
    for heading in (
        "Course Grounding",
        "Independent Evidence",
        "Supported Performance",
        "Prediction Error and Model Revision",
        "Delayed Retrieval",
        "Calibration",
        "Next Action and Review",
    ):
        assert heading in receipt


def test_legacy_ks_stack_remains_present_and_labeled() -> None:
    for path in LEGACY_FILES:
        assert path.is_file(), path
        assert "legacy" in read(path).casefold(), path


def test_trigger_evaluations_use_current_stage_names() -> None:
    orchestrator_eval = read(SKILLS / "dln" / "evaluations" / "trigger-tests.md")
    assert "Acquire/Discriminate" in orchestrator_eval
    assert "generated Session Receipt" in orchestrator_eval

    for skill, contract in PHASE_SKILLS.items():
        text = read(SKILLS / skill / "evaluations" / "trigger-tests.md")
        assert contract["stage"] in text
        assert "generated session receipt" in text.casefold()
        assert "approved course grounding" in text.casefold()
        assert "assertion" in text.casefold()
