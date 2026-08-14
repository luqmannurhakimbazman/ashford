"""Tests for Docker profile migration script."""

from migrate_docker import (
    convert_legacy_profile,
    extract_factor_prose,
    extract_sections,
    infer_concept_status,
    parse_bullet_chains,
    parse_bullet_concepts,
    parse_html_table,
    parse_weakness_bullets,
    rows_to_pipe_table,
)


def test_rows_to_pipe_table_basic():
    rows = [
        {"Name": "Alice", "Score": "10"},
        {"Name": "Bob", "Score": "20"},
    ]
    columns = ["Name", "Score"]
    result = rows_to_pipe_table(rows, columns)
    expected = "| Name | Score |\n|------|------|\n| Alice | 10 |\n| Bob | 20 |"
    assert result == expected


def test_rows_to_pipe_table_empty_rows():
    rows = []
    columns = ["Name", "Score"]
    result = rows_to_pipe_table(rows, columns)
    expected = "| Name | Score |\n|------|------|"
    assert result == expected


def test_rows_to_pipe_table_missing_key():
    rows = [{"Name": "Alice"}]
    columns = ["Name", "Score"]
    result = rows_to_pipe_table(rows, columns)
    expected = "| Name | Score |\n|------|------|\n| Alice |  |"
    assert result == expected


# --- Task 2: parse_html_table ---


def test_parse_html_table_factors():
    html = (
        '<table header-row="true">\n'
        "<tr>\n<td>Factor</td>\n<td>Status</td>\n<td>Evidence</td>\n<td>Last Tested</td>\n</tr>\n"
        "<tr>\n<td>Existence ≠ Readiness</td>\n<td>mastered</td>\n"
        "<td>Gate pass (S3).</td>\n<td>2026-03-13</td>\n</tr>\n"
        "</table>"
    )
    result = parse_html_table(html)
    assert len(result) == 1
    assert result[0]["Factor"] == "Existence ≠ Readiness"
    assert result[0]["Status"] == "mastered"
    assert result[0]["Evidence"] == "Gate pass (S3)."
    assert result[0]["Last Tested"] == "2026-03-13"


def test_parse_html_table_multiple_rows():
    html = (
        '<table header-row="true">\n'
        "<tr>\n<td>A</td>\n<td>B</td>\n</tr>\n"
        "<tr>\n<td>1</td>\n<td>2</td>\n</tr>\n"
        "<tr>\n<td>3</td>\n<td>4</td>\n</tr>\n"
        "</table>"
    )
    result = parse_html_table(html)
    assert len(result) == 2
    assert result[0] == {"A": "1", "B": "2"}
    assert result[1] == {"A": "3", "B": "4"}


# --- Task 3: parse_bullet_concepts and parse_bullet_chains ---


def test_parse_bullet_concepts():
    section = (
        "- **Image**: Read-only template built from a Dockerfile.\n"
        "- **Container**: Running instance of an image.\n"
        "- **CMD vs ENTRYPOINT**: CMD = default command, overridable.\n"
    )
    result = parse_bullet_concepts(section)
    assert len(result) == 3
    assert result[0]["Concept"] == "Image"
    assert result[1]["Concept"] == "Container"
    assert result[2]["Concept"] == "CMD vs ENTRYPOINT"


def test_parse_bullet_chains():
    section = (
        "- **Layer cache invalidation**: Code change → COPY layer invalidated\n"
        "- **Service networking**: Compose creates default network\n"
    )
    result = parse_bullet_chains(section)
    assert len(result) == 2
    assert result[0]["Chain"] == "Layer cache invalidation"
    assert result[1]["Chain"] == "Service networking"


# --- Task 4: parse_weakness_bullets ---


def test_parse_weakness_bullets():
    section = (
        "- Container run flags syntax (--rm, -it distinction) — needs one more rep\n"
        "- Multi-stage build syntax (alpine vs slim, missing pip install) — needs one more rep\n"
        "- Network Isolation — matching network membership to actual communication requirements\n"
        "- Network Isolation — matching network membership to actual communication requirements\n"
    )
    result = parse_weakness_bullets(section)
    # Deduplicates the duplicate Network Isolation entry
    assert len(result) == 3
    assert result[0]["Item"] == "Container run flags syntax (--rm, -it distinction)"
    assert result[0]["Priority"] == "1"
    assert result[0]["Severity"] == "low"
    assert result[2]["Item"] == "Network Isolation"
    assert result[2]["Priority"] == "3"


# --- Task 5: infer_concept_status ---


def test_infer_status_mastered():
    weakness_items = ["Container run flags syntax"]
    open_questions_text = "Hands-on syntax practice needed"
    assert infer_concept_status("Image", weakness_items, open_questions_text) == "mastered"


def test_infer_status_partial_weakness():
    weakness_items = ["Container run flags syntax"]
    open_questions_text = ""
    # "Container Run Flags" concept should match "Container run flags syntax" weakness
    assert (
        infer_concept_status("Container Run Flags", weakness_items, open_questions_text)
        == "partial"
    )


def test_infer_status_partial_open_question():
    weakness_items = []
    open_questions_text = "Container commands and multi-stage build syntax need repetition"
    assert (
        infer_concept_status("Multi-Stage Builds", weakness_items, open_questions_text) == "partial"
    )


# --- Task 6: extract_sections ---


def test_extract_sections():
    body = (
        "# Knowledge State\n"
        "## Syllabus\nGoal: test\n- [x] Topic A\n"
        "## Concepts\n- **Foo**: bar\n"
        "## Chains\n- **Chain1**: desc\n"
        "## Session 1 — 2026-03-10 (Dot Phase)\nSession log here\n"
        "## Session 2 — 2026-03-11\nMore log\n"
    )
    sections, session_logs = extract_sections(body)
    assert "Syllabus" in sections
    assert "Goal: test" in sections["Syllabus"]
    assert "Concepts" in sections
    assert "Chains" in sections
    assert "## Session 1" in session_logs
    assert "## Session 2" in session_logs


def test_extract_sections_separates_session_logs():
    body = (
        "# Knowledge State\n"
        "## Concepts\n- **Foo**: bar\n"
        "## Load Profile\n### Session History\nsome data\n"
        "## Session 1 — 2026-03-10\nLog\n"
    )
    sections, session_logs = extract_sections(body)
    # "### Session History" should NOT be treated as a session log boundary
    assert "Load Profile" in sections
    assert "### Session History" in sections["Load Profile"]
    assert "## Session 1" in session_logs


# --- Task 7: extract_factor_prose ---


def test_extract_factor_prose():
    section = (
        "- **Existence ≠ Readiness**: Docker checks verify existence...\n"
        "- **Read-Run-Diagnose-Fix Loop**: For any config-driven system...\n"
        '<table header-row="true">\n'
        "<tr>\n<td>Factor</td>\n<td>Status</td>\n</tr>\n"
        "<tr>\n<td>Existence ≠ Readiness</td>\n<td>mastered</td>\n</tr>\n"
        "</table>"
    )
    prose, table_html = extract_factor_prose(section)
    assert "- **Existence ≠ Readiness**:" in prose
    assert "<table" in table_html
    assert "<table" not in prose


# --- Task 8: convert_legacy_profile ---


def test_convert_legacy_profile_structure():
    """Test that output has correct structure: markers, H1, sections in order, session logs."""
    body = (
        "# Knowledge State\n"
        "## Syllabus\nGoal: test\n- [x] Topic A\n- [ ] Topic B\n"
        "## Concepts\n- **Foo**: a concept\n"
        "## Chains\n- **Chain1**: a chain\n"
        "## Factors\n- **F1**: a factor\n"
        '<table header-row="true">\n'
        "<tr>\n<td>Factor</td>\n<td>Status</td>\n<td>Evidence</td>\n<td>Last Tested</td>\n</tr>\n"
        "<tr>\n<td>F1</td>\n<td>mastered</td>\n<td>Gate pass</td>\n<td>2026-03-13</td>\n</tr>\n"
        "</table>\n"
        "## Compressed Model\n"
        "## Interleave Pool\n(empty)\n"
        "## Calibration Log\nSome notes\n"
        "### Concept-Level Confidence\n"
        '<table header-row="true">\n'
        "<tr>\n<td>Factor</td>\n<td>Self-Rating</td>\n<td>Actual Performance</td>\n"
        "<td>Gap</td>\n<td>Date</td>\n</tr>\n"
        "<tr>\n<td>F1</td>\n<td>4</td>\n<td>mastered</td>\n"
        "<td>well-calibrated</td>\n<td>2026-03-13</td>\n</tr>\n"
        "</table>\n"
        "### Gate Predictions\n"
        "### Calibration Trend\n"
        "## Load Profile\n### Baseline\n- batch size: 2\n### Session History\n"
        "## Open Questions\n- Some question\n"
        "## Weakness Queue\n- Foo syntax — needs one more rep\n"
        "## Engagement Signals\n- Momentum: positive\n"
        "## Session 1 — 2026-03-10 (Dot Phase)\n### Progress\nDid stuff\n"
    )
    result = convert_legacy_profile(body)

    # Structural checks
    assert result.startswith("<!-- KS:start -->\n# Knowledge State")
    assert "<!-- KS:end -->" in result
    assert result.index("<!-- KS:start -->") < result.index("<!-- KS:end -->")

    # Session logs after KS:end marker
    ks_end_pos = result.index("<!-- KS:end -->")
    after_marker = result[ks_end_pos:]
    assert "## Session 1" in after_marker

    # Init-template section order within KS block
    ks_block = result[:ks_end_pos]
    sections_in_order = [
        "## Syllabus",
        "## Concepts",
        "## Chains",
        "## Factors",
        "## Compressed Model",
        "## Interleave Pool",
        "## Calibration Log",
        "## Load Profile",
        "## Open Questions",
        "## Weakness Queue",
        "## Engagement Signals",
    ]
    positions = [ks_block.index(s) for s in sections_in_order]
    assert positions == sorted(positions), f"Sections out of order: {positions}"

    # Pipe-delimited tables present (not HTML tables)
    assert "<table" not in ks_block
    assert "| Concept |" in ks_block
    assert "| Chain |" in ks_block
    assert "| Factor |" in ks_block
    assert "| Priority |" in ks_block

    # Factor prose preserved
    assert "- **F1**: a factor" in ks_block

    # Syllabus preserved verbatim
    assert "Goal: test" in ks_block
    assert "- [x] Topic A" in ks_block
    assert "- [ ] Topic B" in ks_block


def test_convert_calibration_log_gate_predictions():
    """Test that Gate Predictions text entry is parsed into a table row."""
    body = (
        "# Knowledge State\n"
        "## Syllabus\nGoal: test\n"
        "## Concepts\n"
        "## Chains\n"
        "## Factors\n"
        "## Compressed Model\n"
        "## Interleave Pool\n"
        "## Calibration Log\n"
        "- Gate Predictions: Phase Gate Linear→Network \\| Predicted: 3/5 overall \\| "
        "Actual: Pass (all criteria met) \\| Date: 2026-03-13\n"
        "### Concept-Level Confidence\n"
        "### Gate Predictions\n"
        "### Calibration Trend\n"
        "## Load Profile\n### Baseline\n- batch size: 2\n"
        "## Open Questions\n"
        "## Weakness Queue\n"
        "## Engagement Signals\n- Momentum: neutral\n"
    )
    result = convert_legacy_profile(body)
    assert "| Linear→Network |" in result
    assert "| 3/5 overall |" in result
    assert "| Pass (all criteria met) |" in result


def test_convert_calibration_log_no_subsections():
    """Test fallback when Calibration Log has no ### subsections."""
    body = (
        "# Knowledge State\n"
        "## Syllabus\nGoal: test\n"
        "## Concepts\n"
        "## Chains\n"
        "## Factors\n"
        "## Compressed Model\n"
        "## Interleave Pool\n"
        "## Calibration Log\nJust some text notes\n"
        "## Load Profile\n### Baseline\n- batch size: 2\n"
        "## Open Questions\n"
        "## Weakness Queue\n"
        "## Engagement Signals\n- Momentum: neutral\n"
    )
    result = convert_legacy_profile(body)
    assert "### Concept-Level Confidence" in result
    assert "### Gate Predictions" in result
    assert "### Calibration Trend" in result
