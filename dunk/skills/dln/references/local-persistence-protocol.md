# Local Persistence Protocol

The parent orchestrator is the only authority allowed to invoke `dln-store.py`. Phase skills and `dln-syllabus` return structured requests only.

## Commands

```bash
uv run --project "${CLAUDE_PLUGIN_ROOT}/scripts" --python 3.10.20 --frozen python "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" init --domain "$DOMAIN" --goal "$GOAL"
uv run --project "${CLAUDE_PLUGIN_ROOT}/scripts" --python 3.10.20 --frozen python "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" list
uv run --project "${CLAUDE_PLUGIN_ROOT}/scripts" --python 3.10.20 --frozen python "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" context --domain-id "$DOMAIN_ID"
uv run --project "${CLAUDE_PLUGIN_ROOT}/scripts" --python 3.10.20 --frozen python "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" commit --domain-id "$DOMAIN_ID" --expected-revision "$REVISION" --request "$REQUEST_FILE"
uv run --project "${CLAUDE_PLUGIN_ROOT}/scripts" --python 3.10.20 --frozen python "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" prepare-syllabus --domain-id "$DOMAIN_ID" --expected-revision "$REVISION" --file "$DOCUMENT" --media-type application/pdf --role authoritative --display-name syllabus.pdf --occurred-at "$TIMESTAMP"
uv run --project "${CLAUDE_PLUGIN_ROOT}/scripts" --python 3.10.20 --frozen python "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" syllabus-content --domain-id "$DOMAIN_ID" --source-event-id "$SOURCE_EVENT_ID"
uv run --project "${CLAUDE_PLUGIN_ROOT}/scripts" --python 3.10.20 --frozen python "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" propose-syllabus --domain-id "$DOMAIN_ID" --expected-revision "$REVISION" --request "$PROPOSAL_FILE"
uv run --project "${CLAUDE_PLUGIN_ROOT}/scripts" --python 3.10.20 --frozen python "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" decide-syllabus --domain-id "$DOMAIN_ID" --expected-revision "$REVISION" --request "$DECISION_FILE"
uv run --project "${CLAUDE_PLUGIN_ROOT}/scripts" --python 3.10.20 --frozen python "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" validate --domain-id "$DOMAIN_ID"
uv run --project "${CLAUDE_PLUGIN_ROOT}/scripts" --python 3.10.20 --frozen python "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" doctor --domain-id "$DOMAIN_ID" --recover
uv run --project "${CLAUDE_PLUGIN_ROOT}/scripts" --python 3.10.20 --frozen python "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" rebuild --domain-id "$DOMAIN_ID"
```

PDF preparation must run in the frozen environment containing exactly `pypdf==6.14.2`. HTTPS replaces `--file` with an explicit `--url` and requires `--network-consent`; redirects additionally require `--allow-redirects` and are capped/revalidated.

## Optimistic revision and recovery

Every mutation carries `--expected-revision`. Exit `3` means stale revision: reload `context`, verify the semantic request is unchanged, update the revision, and Retry once. Never blindly retry acquisition or a changed learner decision. Exit `5` means an interrupted transaction; run `doctor --recover`, reload context, and continue only from verified state.

Reserved syllabus kinds cannot enter generic `commit`. `prepare-syllabus`, `propose-syllabus`, and `decide-syllabus` share the same item-1 candidate projection, lock, journal, CAS, ledger, profile, and receipt transaction. Acquisition/extraction occurs before that boundary; stable failures leave the domain byte-identical.

## Portable intake rules

- Local sources use no-follow descriptor checks, regular-file checks, and a 16 MiB limit.
- HTTPS is explicit only: port 443, no userinfo/query/fragment/proxy/cookies/auth/compression/automatic redirects, all-address DNS validation, direct validated-address connection with hostname TLS/SNI, and peer match.
- PDF uses the pinned worker, at most 500 pages and 8 MiB normalized NFC/LF text. HTML uses bounded stdlib parsing and fetches no subresources.
- Run `syllabus-content` after preparation, obtain externally unverified proposals, then collect a complete learner decision.
- `proposal_required`, `decision_required`, `approved`, and `approved_update_pending` are routing statuses. The prior active decision remains authoritative during an update.
- Supplements are visible but never authoritative. Decisions never count as learning evidence.

On success, parse returned IDs/revision and reload context. On validation or intake error, preserve the structured request privately and report the stable `code`; do not fabricate events or receipts.
