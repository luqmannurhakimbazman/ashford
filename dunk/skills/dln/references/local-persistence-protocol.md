# Local Persistence Protocol (Active)

All active Dunk writes go through `${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py`. Never edit `events.jsonl`, `state.json`, `dashboard.md`, or a receipt directly. Never fall back to dialogue, notes, or generated Markdown as state.

Read `@local-store-schema.md` before constructing a request and `@evidence-protocol.md` before deciding what belongs in `events`.

## Commands

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" list
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" init --domain "$DOMAIN" --goal "$GOAL"
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" context --domain-id "$DOMAIN_ID"
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" commit --domain-id "$DOMAIN_ID" --expected-revision "$REVISION" --request "$REQUEST_FILE"
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

## Stale revision retry

Exit `3` means the expected revision is stale:

1. Run `context` again.
2. Confirm the pending assessment still describes the learner response and that every cited prior event remains valid.
3. Retry once with the new revision and exactly the same event IDs and bodies.
4. If it is stale again, keep the pending request in the current conversation, clearly state that persistence stopped, and make no further writes for that boundary. Do not claim it was saved.

A profile patch may be re-created against the new profile only if it has the same user-approved meaning. Events themselves must not be rewritten to fit new state.

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
- Goal, syllabus, review preferences, annotations, and current exam configuration are `profile_patch` fields.
- Closing an exam cycle is an `exam_cycle_closed` event; it does not delete earlier evidence.
- A legacy import uses `import-legacy-ks` on a manually exported block and never contacts a remote service.
