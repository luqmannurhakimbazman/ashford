# Changelog

All notable changes to this plugin will be documented in this file.

## [2.2.0] - 2026-08-19

### Added

- Added generic local and explicit-consent HTTPS acquisition for PDF/HTML with stable failure codes, injected offline security tests, and retained content-addressed source/prepared data outside the plugin.
- Added bounded stdlib HTML extraction and an isolated, fixed-argument PDF worker pinned exactly to `pypdf==6.14.2`, with page/text/time/output/resource limits and golden extraction fixtures.
- Added the generic `prepare-syllabus`, `syllabus-content`, `propose-syllabus`, and `decide-syllabus` CLI lifecycle.

### Changed

- Replaced fixture-specific runtime instructions with portable proposal/decision contracts. New grounding citations use `decision_event_id`; historical `approval_event_id` remains replay-compatible.
- Authoritative updates now preserve the prior active decision until a complete superseding decision. Supplements remain visible but cannot affect authority, planning citations, or mastery.
- Canonical backups now include bounded original bytes and normalized prepared text. Rebuild never reacquires or re-extracts them.
- `state.json` now carries a `grounding` object, `dashboard.md` gains a Course Grounding section, and each canonical syllabus source event generates a Syllabus Intake Receipt. Canonical `profile.yaml` and `events.jsonl` still load without migration, but projections written by 2.0/2.1 are stale: run `rebuild` once per existing domain after upgrading, before `validate`.
- Only `prepare-syllabus --media-type application/pdf` requires the frozen `pypdf` environment; every other command stays stdlib-only, and `uv` must be pointed outside `${CLAUDE_PLUGIN_ROOT}` with `UV_PROJECT_ENVIRONMENT`.

## [2.1.0] - 2026-08-19

Superseded by 2.2.0 before publication; no 2.1.0 release was distributed. Its digest-bound ST5201X adapter no longer exists in any shipped version, and the entries below remain only as the development record for the invariants 2.2.0 generalized.

### Added

- Added a digest-bound `st5201x-2026-v1` intake adapter for the exact supplied ST5201X syllabus, immutable extraction/assertion and learner approval history, bounded approved grounding, deterministic Syllabus Intake Receipts, and stable grounding citations in teaching receipts.
- Added truthful unavailable, unreadable, media-type, size, and digest-mismatch degradation plus replay/rebuild/validation coverage.

### Changed

- Approved syllabus coverage now derives the flat planning projection; legacy `profile.syllabus` remains readable but explicitly ungrounded and non-citable. A `profile_patch.syllabus` edit is rejected while an approved grounding is active instead of being silently discarded by the projection.
- Syllabus administrative sessions are now rejected symmetrically: neither intake nor approval may claim a session id already used by an earlier event.
- Syllabus assertions, approval, coverage, and citations remain separate from mastery evidence. Raw PDFs are not retained, and broader PDF/OCR support remains out of scope.
- Existing 2.0 canonical pairs load without migration. Domains containing new syllabus events are forward-only because Dunk 2.0 rejects their event kinds; back up the matched canonical pair before intake if downgrade may be required.

## [2.0.0] - 2026-08-18

### Changed

- Made `profile.yaml` and append-only `events.jsonl` the canonical local learning record; `state.json`, `dashboard.md`, and Session Receipts are deterministic projections.
- Reframed the teaching flow as evidence-driven acquire/discriminate, relate/abstract, and predict/revise/compress operations.
- Removed Notion authentication, bidirectional synchronization, and the marker-validation hook from the active path. Generated Markdown is directly readable in Obsidian without a plugin or MCP server.
- Added optimistic revisions, idempotent event commits, single-writer locking, crash recovery, deterministic rebuilds, and actionable validation diagnostics.
- Added `import-legacy-ks` for a non-destructive, one-time import of manually exported legacy Knowledge State Markdown. Imported claims remain unverified and evidence-ineligible.
- Restricted delayed-retrieval satisfaction to independent passing retrievals; supported or non-passing retrievals remain measured evidence and no longer promote a subject beyond `needs-retrieval`, and both the dashboard and the Session Receipt label a non-independent retrieval as `(supported)`.

## [1.0.0] - 2026-08-14

### Added

- Initial Dot-Linear-Network learning system.
- Notion synchronization agents and marker-validation hook.
- Knowledge-state merge tooling and tests.
