# Local Persistence Protocol (Active)

All active Dunk writes go through `${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py`. Never edit `events.jsonl`, `state.json`, `dashboard.md`, or a receipt directly. Never fall back to dialogue, notes, or generated Markdown as state.

Read `@local-store-schema.md` before constructing a request and `@evidence-protocol.md` before deciding what belongs in `events`.

## Commands

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" list
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" init --domain "$DOMAIN" --goal "$GOAL"
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" context --domain-id "$DOMAIN_ID"
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" commit --domain-id "$DOMAIN_ID" --expected-revision "$REVISION" --request "$REQUEST_FILE"
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" ingest-syllabus --domain-id "$DOMAIN_ID" --expected-revision "$REVISION" --document "$DOCUMENT" --original-filename "$FILENAME" --media-type application/pdf --adapter st5201x-2026-v1 --occurred-at "$TIMESTAMP"
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" approve-syllabus --domain-id "$DOMAIN_ID" --expected-revision "$REVISION" --request "$APPROVAL_FILE"
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" validate --domain-id "$DOMAIN_ID"
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" doctor --domain-id "$DOMAIN_ID" --recover
```

The root is provided by `DLN_VAULT_ROOT` or `CLAUDE_PLUGIN_DATA`; use `--root` only when the user explicitly supplied it.

## Session flow

1. Run `context` and retain its `profile`, `state`, and `state.revision`.
2. Choose the teaching operation from `state.stage`, due retrieval, current subject evidence, syllabus, and goal. Do not route from prose memory.
3. Teach or assess. Content delivery alone produces no event.
4. At a meaningful assessment boundary, create one request containing only observed structured evidence and any supported profile patch.
5. Write the request to a private temporary directory with permissions restricted by the process umask. Install a cleanup trap and never print the request contents unless debugging with the user's consent.
6. Run `commit --expected-revision <retained revision>`.
7. On `committed`, replace the retained revision with the returned revision. On `noop`, retain the returned revision.
8. Before ending, atomically commit any remaining evidence followed by one `session_completed` event. Then read `sessions/<session-id>.md` and present that generated Session Receipt verbatim or with only a short link/path introduction. Do not create a competing session summary.

Example private request handling:

```bash
DLN_TMP=$(mktemp -d "${TMPDIR:-/tmp}/dln-commit.XXXXXXXXXX")
chmod 700 "$DLN_TMP"
trap 'rm -rf "$DLN_TMP"' EXIT HUP INT TERM
REQUEST_FILE="$DLN_TMP/request.json"
# Write the already-constructed JSON request to REQUEST_FILE, then commit it.
```

## Stable identity

Create each `event_id` exactly once from stable session/task context before the first commit attempt. Keep the complete event body and ID unchanged across retries. Do not use a random new ID after an ambiguous result; replaying the original body is how the store proves idempotency.

Use one session ID for the entire live session. Never reuse an ID after `session_completed` succeeds.

## Syllabus command rules

`ingest-syllabus` requires readable bytes and accepts only the exact digest-bound ST5201X adapter described in `syllabus-grounding-protocol.md`. Unavailable, unreadable, wrong-media-type, wrong-size, or digest-mismatched input must fail without changing the revision. Exact source replay returns `noop`.

After intake, reload `context`, present the generated Syllabus Intake Receipt, and collect a complete learner decision before `approve-syllabus`. Keep approval JSON private like a commit request. Exact approval replay is a no-op; conflicting ID reuse fails. Generic `commit` cannot create either reserved syllabus event kind.

## Stale revision retry

Exit `3` means the expected revision is stale:

1. Run `context` again.
2. Confirm the pending assessment still describes the learner response and that every cited prior event remains valid.
3. Retry once with the new revision and exactly the same event IDs and bodies.
4. If it is stale again, keep the pending request in the current conversation, clearly state that persistence stopped, and make no further writes for that boundary. Do not claim it was saved.

A profile patch or complete syllabus approval request may be re-created against the new context only if it has the same user-approved meaning and source assertion set. Intake and event bodies must not be rewritten to fit new state.

## Exit handling

| Exit | Meaning | Required behavior |
|---|---|---|
| `0` | committed, initialized, valid, rebuilt, or idempotent no-op | Parse stdout JSON and continue from its revision/status. |
| `1` | OS/runtime failure | Stop persistent writes; report the diagnostic. |
| `2` | schema, reference, path, corruption, or integrity error | Stop. Correct an uncommitted construction error only; never patch canonical files. |
| `3` | stale revision | Follow the single retry protocol above. |
| `4` | writer lock unavailable/stale diagnostics | Stop writes; use `doctor` for diagnostics. Do not break a lock without explicit user approval. |
| `5` | interrupted transaction needs recovery | Stop reads/writes and run `doctor --recover`; resume only after a successful `context`. |

Never run `rebuild` as a way to erase validation failures. `rebuild` reconstructs derived files from valid canonical sources only.

## Reset, syllabus, and exam metadata

- Reset is a revision-checked `domain_reset` event. It preserves historic events and receipts.
- Goal, review preferences, annotations, and current exam configuration are `profile_patch` fields.
- Flat `profile.syllabus` is editable only while the domain is ungrounded or awaiting approval; once an approval is active the store rejects that patch instead of silently ignoring it. Grounded source values use dedicated intake/approval commands and superseding approval snapshots.
- Closing an exam cycle is an `exam_cycle_closed` event; it does not delete earlier evidence.
- A legacy import uses `import-legacy-ks` on a manually exported block and never contacts a remote service.
