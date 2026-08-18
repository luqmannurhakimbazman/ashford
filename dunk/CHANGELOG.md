# Changelog

All notable changes to this plugin will be documented in this file.

## [2.0.0] - 2026-08-18

### Changed

- Made `profile.yaml` and append-only `events.jsonl` the canonical local learning record; `state.json`, `dashboard.md`, and Session Receipts are deterministic projections.
- Reframed the teaching flow as evidence-driven acquire/discriminate, relate/abstract, and predict/revise/compress operations.
- Removed Notion authentication, bidirectional synchronization, and the marker-validation hook from the active path. Generated Markdown is directly readable in Obsidian without a plugin or MCP server.
- Added optimistic revisions, idempotent event commits, single-writer locking, crash recovery, deterministic rebuilds, and actionable validation diagnostics.
- Added `import-legacy-ks` for a non-destructive, one-time import of manually exported legacy Knowledge State Markdown. Imported claims remain unverified and evidence-ineligible.
- Restricted delayed-retrieval satisfaction to independent passing retrievals; supported or non-passing retrievals remain measured evidence and no longer promote a subject beyond `needs-retrieval`.

## [1.0.0] - 2026-08-14

### Added

- Initial Dot-Linear-Network learning system.
- Notion synchronization agents and marker-validation hook.
- Knowledge-state merge tooling and tests.
