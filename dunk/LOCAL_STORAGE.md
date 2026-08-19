# Dunk local storage

Dunk is local-first. The filesystem is authoritative and Obsidian is an optional viewer for ordinary Markdown. No Obsidian plugin, MCP server, database, or Notion account is required. Every learning domain lives outside `${CLAUDE_PLUGIN_ROOT}`; the plugin installation is code only, so never put learner data or syllabus content inside it.

## Choose a vault root

Dunk resolves its root in this order:

1. `--root <path>` on the current command.
2. `DLN_VAULT_ROOT`.
3. `${CLAUDE_PLUGIN_DATA}/dln-vault` when `CLAUDE_PLUGIN_DATA` is present in the invoking Bash environment.

Strict plugin validation confirms the plugin package, not runtime environment-variable injection. If the CLI reports that the root is unconfigured, set `DLN_VAULT_ROOT` explicitly; Dunk never chooses an implicit home directory. Plugin upgrades replace `${CLAUDE_PLUGIN_ROOT}`, so nothing persistent may be written there.

```bash
export DLN_VAULT_ROOT="/path/to/private/dln-vault"
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" list
```

When running from this repository, use `python3 dunk/scripts/dln-store.py` instead.

## Canonical domain data

```text
<root>/
├── .locks/
└── domains/
    └── <domain-id>/
        ├── profile.yaml
        ├── events.jsonl
        ├── sources/sha256/<source-digest>
        ├── prepared/sha256/<prepared-digest>.json
        ├── state.json
        ├── dashboard.md
        ├── syllabus/
        │   └── <source-version-id>.md
        └── sessions/
            └── <session-id>.md
```

- `profile.yaml`: user configuration plus store-owned identity/revision fields. It is a JSON-compatible YAML subset so the stdlib-only CLI can validate it without a YAML dependency.
- `events.jsonl`: append-only learning and reserved syllabus lifecycle events. Never edit, reorder, or truncate it.
- `sources/sha256/<source-digest>`: canonical bounded original PDF/HTML bytes.
- `prepared/sha256/<prepared-digest>.json`: canonical normalized extracted text and unit hashes.
- `state.json`, `dashboard.md`, `sessions/*.md`, and `syllabus/*.md`: deterministic projections.

You choose `domain` during initialization; it is then immutable because it determines the directory identity. You may edit `goal`, `annotations`, `review_preferences`, and `exam` in `profile.yaml`. Flat `profile.syllabus` remains editable only as a legacy ungrounded curriculum; learner-decided document-backed topics are derived from syllabus events. Do not edit `domain`, `schema_version`, `domain_id`, or `revision`; initialize a new domain to rename it.

Back up `profile.yaml`, `events.jsonl`, `sources/`, and `prepared/` together. Raw source bytes and normalized prepared text are private canonical backup material. Missing or corrupt CAS is an integrity failure; `rebuild` regenerates derived files only and never downloads, extracts, or invents canonical content. Hashes prove internal consistency, not protection from a malicious filesystem owner who rewrites all canonical artifacts consistently.

## Locked runtime

Every command except PDF preparation runs on stdlib `python3`. Portable PDF preparation has one exact direct runtime dependency, `pypdf==6.14.2`, and runs in the locked Python 3.10 environment. From an Ashford checkout, install and verify it:

```bash
uv sync --project dunk/scripts --python 3.10.20 --frozen
uv run --project dunk/scripts --python 3.10.20 --frozen python -c \
  "import pypdf; assert pypdf.__version__ == '6.14.2'"
```

`uv` places a project environment in the project directory unless `UV_PROJECT_ENVIRONMENT` says otherwise, so an installed plugin must direct it outside the replaceable install cache:

```bash
export UV_PROJECT_ENVIRONMENT="$DLN_VAULT_ROOT/.uv/pypdf-6.14.2"
```

HTML preparation uses bounded stdlib `html.parser`. There is no OCR, alternate PDF engine, browser rendering, crawling, JavaScript, linked-resource fetching, ambient proxy/auth/cookie use, compression, or automatic redirect following. PDF preparation requires exactly `pypdf==6.14.2`: a missing dependency reports `extractor_unavailable`, while a different version reports `extractor_version_mismatch`.

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

## Generic syllabus lifecycle

```bash
PROJECT="${CLAUDE_PLUGIN_ROOT}/scripts"
STORE="${PROJECT}/dln-store.py"

uv run --project "$PROJECT" --python 3.10.20 --frozen python "$STORE" prepare-syllabus --domain-id generic-<id> --expected-revision 0 \
  --file /path/to/syllabus.pdf --media-type application/pdf \
  --role authoritative --display-name syllabus.pdf \
  --occurred-at 2026-08-19T00:00:00Z

python3 "$STORE" syllabus-content --domain-id generic-<id> \
  --source-event-id syllabus-prepared-<sha256>

python3 "$STORE" propose-syllabus --domain-id generic-<id> \
  --expected-revision 1 --request "$PROPOSAL_FILE"

python3 "$STORE" decide-syllabus --domain-id generic-<id> \
  --expected-revision 2 --request "$DECISION_FILE"
```

Only `prepare-syllabus --media-type application/pdf` needs the locked environment; HTML preparation and every other command are stdlib-only. For HTTPS, use an explicit query-free URL and explicit consent:

```bash
uv run --project "$PROJECT" --python 3.10.20 --frozen python "$STORE" prepare-syllabus ... --url https://example.edu/syllabus.pdf \
  --network-consent
```

Redirects remain disabled unless `--allow-redirects` is also explicitly supplied. Every redirect is revalidated and the chain is capped at three. HTTPS accepts port 443 only and rejects unsafe URL forms, mixed/non-global DNS, rebinding/peer mismatch, oversized headers/bodies, compression, timeouts, and MIME/magic disagreement with stable machine-readable codes. Tests use injected resolvers/transports and never public network.

`prepare-syllabus` acquires and extracts before the item-1 atomic commit boundary. Any acquisition/extraction rejection leaves the domain byte-for-byte unchanged. `propose-syllabus` seals externally unverified proposals; `decide-syllabus` records the learner's complete accept/correct/defer/reject partition. Supplements are visible but never authoritative. A previous decision remains active as `approved_update_pending` until its explicitly superseding authoritative source receives a complete decision.

## Obsidian

Open `<root>` or `<root>/domains` directly as an Obsidian vault. Open a domain's `dashboard.md` for the longitudinal view and follow its `sessions/...` and `syllabus/...` links to generated receipts. The source files remain portable Markdown, JSON, and JSONL even if Obsidian is never installed.

Obsidian or filesystem sync is external to Dunk. It is not a writer lock or conflict-resolution authority. Avoid concurrent writers, and resolve any external sync conflict before the next commit.

## Validation and recovery

Use `validate` to check canonical sources, canonical content integrity, and projection drift. It exits nonzero when a generated projection is modified or missing, when an unexpected receipt is present, or when a referenced source/prepared object is missing, corrupt, or orphaned. Run `rebuild` for modified or missing projections; remove an unexpected receipt only after confirming it is absent from canonical events. Leading-dot editor or operating-system metadata under `sessions/` and `syllabus/`, such as `.DS_Store` or `.obsidian/`, is ignored. If a process was interrupted, inspect before changing files:

```bash
python3 "$STORE" doctor --domain-id probability-<id>
python3 "$STORE" doctor --domain-id probability-<id> --recover
python3 "$STORE" validate --domain-id probability-<id>
```

Never run `rebuild` to erase a validation failure; it reconstructs derived files from valid canonical sources only. The store uses a single-writer lock, staged files, backups, a transaction journal, fsync, and atomic replacement. Reads do not return mixed revisions: recover the interrupted transaction or stop. Use `--break-stale-lock` only when `doctor` identifies stale metadata and no writer process is active.

## Privacy, sync, migration, and rollback

Treat syllabus originals, extracted text, decisions, and learning evidence as private. Sync canonical files only through a trusted private channel and preserve permissions, filenames, bytes, and append order. Never merge ledgers line-by-line or sync a live `.dln-transaction`.

Dunk 2.2.0 adds a `grounding` object to `state.json` and a Course Grounding section to `dashboard.md`, and it generates one Syllabus Intake Receipt per canonical syllabus source event. Canonical `profile.yaml` and `events.jsonl` from Dunk 2.0/2.1 load without migration, but their older generated projections are stale: run `rebuild` once per existing domain after upgrading, before `validate`.

New lifecycle events and store-invoked provenance are forward-only for older Dunk versions. Rollback restores a matched pre-upgrade backup of the complete domain; never truncate or rewrite a live ledger. Legacy text-only sources remain replayable without raw-byte CAS and retain historical `approval_event_id` citations. New learning writes use `decision_event_id`.

## Import a legacy Knowledge State

Dunk does not read from or write to Notion. First export the legacy marker-delimited Knowledge State Markdown to a local file, then run:

```bash
python3 "$STORE" import-legacy-ks \
  --domain "Imported domain" \
  --input /path/to/exported-knowledge-state.md
```

The import is non-destructive and offline. It creates one deterministic event keyed by the source SHA-256. Imported syllabus, concepts, chains, factors, model text, and questions are displayed as unverified prior claims; they do not satisfy evidence gates, mastery, delayed retrieval, or calibration. Reimporting the identical source is a no-op, while a different snapshot is rejected for a non-empty domain. Existing remote pages are never modified.
