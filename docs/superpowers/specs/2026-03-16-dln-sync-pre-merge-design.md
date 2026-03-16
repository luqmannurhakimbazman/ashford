# Move KS merge out of dln-sync subagent

**Date:** 2026-03-16
**Status:** Approved
**Scope:** dln-sync agent, phase skills (dln-dot, dln-linear, dln-network), shared references

## Problem

The dln-sync subagent needs Bash (to run ks-merge.py) and Write (to create temp files) permissions, but Claude Code's permission model does not allow subagents to elevate permissions beyond the parent session. Setting `permissionMode: dontAsk` or `bypassPermissions` in agent frontmatter has no effect — the parent session's permission mode always takes precedence.

Result: KS block replacements (compressed model, factors table, weakness queue rewrites) are partial or missing. Column property updates and small Notion appends get through because they use MCP tools, which are unaffected.

## Solution

Move the normalize + merge steps out of dln-sync and into the parent conversation, where Bash/Write permissions are approved interactively. dln-sync becomes a pure Notion I/O agent — fetch, replace, verify, compress.

## New dispatch flow

Per sync boundary, the phase skill runs a two-dispatch sequence:

```
Phase Skill (parent)              dln-sync (subagent)
─────────────────────             ──────────────────────
1. dispatch fetch ───────────────► fetch page, extract KS block
                  ◄─────────────── return raw KS block
2. construct JSON payload
3. write temp files (Write tool)
4. run ks-merge.py (Bash)
5. capture merged KS
6. dispatch replace ─────────────► fetch page, replace KS,
                                   append progress notes,
                                   verify, compress
                    ◄─────────────── return re-anchor payload
7. clean up temp files
```

## New dln-sync action set

| Action | Input | dln-sync does | Returns |
|--------|-------|---------------|---------|
| `fetch` | `page_id` | Fetches page, extracts KS block (between markers) | Raw KS block |
| `replace` | `page_id`, `merged_ks`, `progress_notes`, `session_number`, `phase` | Fetches page, replaces KS with merged_ks, appends progress notes, verifies | Compressed re-anchor payload |
| `replace-end` | Same as `replace` + `column_updates` | Same as `replace` + updates column properties | Compressed re-anchor payload |
| `plan-write` | `page_id`, `merged_ks` (optional), `plan_content`, `session_number`, `phase` | Fetches page, replaces KS if merged_ks provided, appends session plan, verifies | Compressed re-anchor payload |

- `sync` action is removed (replaced by `fetch` + parent merge + `replace`)
- `session-end` action is removed (replaced by `fetch` + parent merge + `replace-end`)
- `plan-write` stays but now receives pre-merged KS instead of prose

## New shared reference: merge-protocol.md

Location: `dunk/skills/dln/references/merge-protocol.md`

Contains (extracted from dln-sync):
- Normalizer Schema (JSON format for merge payloads)
- Normalizer examples
- Normalization rules (which fields need existing KS for full-rewrite vs. append-only)
- Parent-side merge sequence (the 10-step procedure)

All three phase skills reference this file at every sync boundary instead of sending prose to dln-sync. The phase skills construct the JSON payload; the shared reference tells them how.

## Phase skill changes

Every dispatch instruction in dln-dot, dln-linear, dln-network changes from:

> "dispatch dln-sync with action sync and prose payload"

To:

> "follow the merge protocol in `@references/merge-protocol.md`"

The phase skills still describe *what* changed (mastery updates, weakness queue, progress notes). The merge-protocol tells them *how* to package, merge, and dispatch.

## Removals from dln-sync

**Sections removed:**
- Normalizer Schema (~70 lines)
- Normalizer Examples (~40 lines)
- Step 2a: NORMALIZE
- Step 2b: MERGE
- Merging Rules section (reference only — logic lives in ks-merge.py)
- Golden Rule #4 (no inline scripts — no longer relevant)

**Tools removed:** `Bash`, `Write`, `Read`

**Permission mode:** `bypassPermissions` reverts to `dontAsk` (only MCP tools needed)

## Unchanged in dln-sync

- Golden Rules 1-3 (fetch before write, target by session number, fail fast)
- MARKER RULE
- Step 3: REPLACE logic
- Step 4: VERIFY logic
- Session log append logic (progress notes, plan content)
- Compression (dln-compress skill)
- Error handling for Notion failures
- Database Reference
- PreToolUse hook for KS marker validation

## Other removals

- `dunk/scripts/block-inline-scripts.sh` — hook script no longer needed
- Bash matcher entry in `dunk/hooks/hooks.json` — no longer needed

## Risks

**Double dispatch latency:** Each sync boundary now requires two agent dispatches (fetch + replace) instead of one. Adds ~5-10 seconds per boundary. Acceptable given boundaries are natural teaching pauses.

**Stale KS between fetch and replace:** Another process could modify the page between the fetch dispatch and the replace dispatch. Mitigated by dln-sync's existing verify step — it re-fetches and retries on mismatch.

**Phase skill context growth:** The JSON payload construction and merge execution happen in the parent conversation, adding to context. Mitigated by the merge-protocol reference keeping instructions out of the skill body.

## Files to modify

| File | Change |
|------|--------|
| `dunk/agents/dln-sync.md` | Remove normalize/merge sections, tools, update actions |
| `dunk/skills/dln/references/merge-protocol.md` | **New** — extracted normalizer schema + parent merge sequence |
| `dunk/skills/dln-dot/SKILL.md` | Replace dispatch instructions with merge-protocol reference |
| `dunk/skills/dln-linear/SKILL.md` | Replace dispatch instructions with merge-protocol reference |
| `dunk/skills/dln-network/SKILL.md` | Replace dispatch instructions with merge-protocol reference |
| `dunk/skills/dln/references/sync-protocol.md` | Minor updates to reference new action names |
| `dunk/hooks/hooks.json` | Remove Bash matcher entry |
| `dunk/scripts/block-inline-scripts.sh` | **Delete** |
