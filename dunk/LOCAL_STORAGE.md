# Dunk local storage

Dunk keeps each learning domain outside `${CLAUDE_PLUGIN_ROOT}` beneath `DLN_VAULT_ROOT` (or `CLAUDE_PLUGIN_DATA`). The plugin installation is code only; do not put learner data or syllabus content inside it.

## Canonical domain data

Each `domains/<domain-id>/` contains:

- `profile.yaml`: user configuration plus store-owned identity/revision fields.
- `events.jsonl`: append-only learning and reserved syllabus lifecycle events.
- `sources/sha256/<source-digest>`: canonical bounded original PDF/HTML bytes.
- `prepared/sha256/<prepared-digest>.json`: canonical normalized extracted text and unit hashes.
- `state.json`, `dashboard.md`, `sessions/*.md`, and `syllabus/*.md`: deterministic projections.

Back up `profile.yaml`, `events.jsonl`, `sources/`, and `prepared/` together. Raw source bytes and normalized prepared text are private canonical backup material. Missing or corrupt CAS is an integrity failure; `rebuild` regenerates derived files only and never downloads, extracts, or invents canonical content. Hashes prove internal consistency, not protection from a malicious filesystem owner who rewrites all canonical artifacts consistently.

## Locked runtime

Portable PDF preparation has one exact direct runtime dependency: `pypdf==6.14.2`. From an Ashford checkout, install and verify the locked Python 3.10 environment:

```bash
uv sync --project dunk/scripts --python 3.10.20 --frozen
uv run --project dunk/scripts --python 3.10.20 --frozen python -c \
  "import pypdf; assert pypdf.__version__ == '6.14.2'"
```

HTML preparation uses bounded stdlib `html.parser`. There is no OCR, alternate PDF engine, browser rendering, crawling, JavaScript, linked-resource fetching, ambient proxy/auth/cookie use, compression, or automatic redirect following. PDF preparation requires exactly `pypdf==6.14.2`: a missing dependency reports `extractor_unavailable`, while a different version reports `extractor_version_mismatch`.

## Generic lifecycle

```bash
export DLN_VAULT_ROOT='/path/to/private/dln-vault'
PROJECT="${CLAUDE_PLUGIN_ROOT}/scripts"
STORE="${PROJECT}/dln-store.py"

uv run --project "$PROJECT" --python 3.10.20 --frozen python "$STORE" prepare-syllabus --domain-id generic-<id> --expected-revision 0 \
  --file /path/to/syllabus.pdf --media-type application/pdf \
  --role authoritative --display-name syllabus.pdf \
  --occurred-at 2026-08-19T00:00:00Z

uv run --project "$PROJECT" --python 3.10.20 --frozen python "$STORE" syllabus-content --domain-id generic-<id> \
  --source-event-id syllabus-prepared-<sha256>

uv run --project "$PROJECT" --python 3.10.20 --frozen python "$STORE" propose-syllabus --domain-id generic-<id> \
  --expected-revision 1 --request "$PROPOSAL_FILE"

uv run --project "$PROJECT" --python 3.10.20 --frozen python "$STORE" decide-syllabus --domain-id generic-<id> \
  --expected-revision 2 --request "$DECISION_FILE"
```

For HTTPS, use an explicit query-free URL and explicit consent:

```bash
uv run --project "$PROJECT" --python 3.10.20 --frozen python "$STORE" prepare-syllabus ... --url https://example.edu/syllabus.pdf \
  --network-consent
```

Redirects remain disabled unless `--allow-redirects` is also explicitly supplied. Every redirect is revalidated and the chain is capped at three. HTTPS accepts port 443 only and rejects unsafe URL forms, mixed/non-global DNS, rebinding/peer mismatch, oversized headers/bodies, compression, timeouts, and MIME/magic disagreement with stable machine-readable codes. Tests use injected resolvers/transports and never public network.

`prepare-syllabus` acquires and extracts before the item-1 atomic commit boundary. Any acquisition/extraction rejection leaves the domain byte-for-byte unchanged. `propose-syllabus` seals externally unverified proposals; `decide-syllabus` records the learner's complete accept/correct/defer/reject partition. Supplements are visible but never authoritative. A previous decision remains active as `approved_update_pending` until its explicitly superseding authoritative source receives a complete decision.

## Privacy, sync, migration, and rollback

Treat syllabus originals, extracted text, decisions, and learning evidence as private. Sync canonical files only through a trusted private channel and preserve permissions, filenames, bytes, and append order. Never merge ledgers line-by-line or sync a live `.dln-transaction`.

New lifecycle events and store-invoked provenance are forward-only for older Dunk versions. Rollback restores a matched pre-upgrade backup of the complete domain; never truncate or rewrite a live ledger. Legacy text-only sources remain replayable without raw-byte CAS and retain historical `approval_event_id` citations. New learning writes use `decision_event_id`.
