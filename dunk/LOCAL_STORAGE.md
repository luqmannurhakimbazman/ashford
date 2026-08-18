# Dunk local storage

Dunk is local-first. The filesystem is authoritative, and Obsidian is an optional viewer for ordinary Markdown. No Obsidian plugin, MCP server, database, or Notion account is required.

## Choose a vault root

Dunk resolves its root in this order:

1. `--root <path>` on the current command.
2. `DLN_VAULT_ROOT`.
3. `${CLAUDE_PLUGIN_DATA}/dln-vault` when `CLAUDE_PLUGIN_DATA` is present in the invoking Bash environment.

Strict plugin validation confirms the plugin package, not runtime environment-variable injection. If the CLI reports that the root is unconfigured, set `DLN_VAULT_ROOT` explicitly. Do not put learner data under `${CLAUDE_PLUGIN_ROOT}` because plugin upgrades replace that directory.

```bash
export DLN_VAULT_ROOT="/path/to/dln-vault"
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" list
```

When running from this repository, use `python3 dunk/scripts/dln-store.py` instead.

## Canonical layout

Each learning domain has this layout:

```text
<root>/
├── .locks/
└── domains/
    └── <domain-id>/
        ├── profile.yaml
        ├── events.jsonl
        ├── state.json
        ├── dashboard.md
        └── sessions/
            └── <session-id>.md
```

- `profile.yaml` is user-owned configuration plus store-owned identity/revision fields. It is a JSON-compatible YAML subset so the stdlib-only CLI can validate it without a YAML dependency.
- `events.jsonl` is the append-only, immutable learning and assessment history. Never edit, reorder, or truncate it.
- `state.json` is a deterministic cache.
- `dashboard.md` is the generated longitudinal Obsidian view.
- `sessions/<session-id>.md` is the sole canonical learner-facing Session Receipt for a completed session.

You choose `domain` during initialization; it is then immutable because it determines the directory identity. You may edit `goal`, `syllabus`, `annotations`, `review_preferences`, and `exam` in `profile.yaml`. Do not edit `domain`, `schema_version`, `domain_id`, or `revision`; initialize a new domain to rename it.

## Common operations

```bash
STORE="${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py"

python3 "$STORE" init --domain "Probability" --goal "Calibrate Bayesian predictions"
python3 "$STORE" list
python3 "$STORE" context --domain-id probability-<id>
python3 "$STORE" validate --domain-id probability-<id>
python3 "$STORE" rebuild --domain-id probability-<id>
```

Tutoring skills commit validated event requests with the revision returned by `context`:

```bash
python3 "$STORE" commit \
  --domain-id probability-<id> \
  --expected-revision 0 \
  --request /path/to/request.json
```

A stale revision is rejected without changing valid files. Replaying byte-equivalent events with the same event IDs is an idempotent no-op. Reusing an event ID with different content is an integrity error.

## Obsidian

Open `<root>` or `<root>/domains` directly as an Obsidian vault. Open a domain's `dashboard.md` for the longitudinal view and follow its `sessions/...` links to completed-session receipts. The source files remain portable Markdown, JSON, and JSONL even if Obsidian is never installed.

Obsidian or filesystem sync is external to Dunk. It is not a writer lock or conflict-resolution authority. Avoid concurrent writers, and resolve any external sync conflict before the next commit.

## Backup, validation, and recovery

Back up `profile.yaml` and `events.jsonl`; every other domain artifact is reproducible. Preserve the directory structure and file bytes. A backup is only complete when both canonical files are from the same committed revision.

Use `validate` to check canonical sources and projection drift. It exits nonzero when a generated projection is modified, missing, or an unexpected receipt is present. Run `rebuild` for modified or missing projections; remove an unexpected receipt only after confirming it is absent from canonical events. Leading-dot editor or operating-system metadata under `sessions/`, such as `.DS_Store` or `.obsidian/`, is ignored. If a process was interrupted, inspect before changing files:

```bash
python3 "$STORE" doctor --domain-id probability-<id>
python3 "$STORE" doctor --domain-id probability-<id> --recover
python3 "$STORE" validate --domain-id probability-<id>
```

The store uses a single-writer lock, staged files, backups, a transaction journal, fsync, and atomic replacement. Reads do not return mixed revisions: recover the interrupted transaction or stop. Use `--break-stale-lock` only when `doctor` identifies stale metadata and no writer process is active.

## Import a legacy Knowledge State

Dunk does not read from or write to Notion. First export the legacy marker-delimited Knowledge State Markdown to a local file, then run:

```bash
python3 "$STORE" import-legacy-ks \
  --domain "Imported domain" \
  --input /path/to/exported-knowledge-state.md
```

The import is non-destructive and offline. It creates one deterministic event keyed by the source SHA-256. Imported syllabus, concepts, chains, factors, model text, and questions are displayed as unverified prior claims; they do not satisfy evidence gates, mastery, delayed retrieval, or calibration. Reimporting the identical source is a no-op, while a different snapshot is rejected for a non-empty domain. Existing remote pages are never modified.
