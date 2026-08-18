# DLN Sync Pre-Merge Implementation Plan

> **Superseded by Dunk 2.0.0.** Historical record of the removed `dln-sync`/Notion remote path; the local-first filesystem store in `dunk/skills/dln/references/local-persistence-protocol.md` is now authoritative.

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move KS normalize + merge steps out of the dln-sync subagent into the parent conversation so Bash/Write permissions are no longer needed inside the subagent sandbox.

**Architecture:** Two-dispatch sequence per sync boundary — parent dispatches `fetch` to get raw KS, runs ks-merge.py locally, then dispatches `replace` with pre-merged result. dln-sync becomes a pure Notion I/O agent.

**Tech Stack:** Markdown (Claude Code plugin skills/agents), Python (ks-merge.py — unchanged), Notion MCP

**Spec:** `docs/superpowers/specs/2026-03-16-dln-sync-pre-merge-design.md`

---

## Chunk 1: Infrastructure (merge-protocol, hooks, cleanup)

### Task 1: Create merge-protocol.md shared reference

**Files:**
- Create: `dunk/skills/dln/references/merge-protocol.md`

This is the central reference all three phase skills will use. It contains the normalizer schema, examples, and the parent-side merge sequence.

- [ ] **Step 1: Create merge-protocol.md**

Extract the Normalizer Schema (dln-sync.md lines 63-131), Normalizer Examples (lines 132-194), and normalization rules into a new shared reference. Add the parent-side merge sequence.

```markdown
# Merge Protocol

This reference is shared across all DLN phase skills (dln-dot, dln-linear, dln-network). It defines how to construct a merge payload, run the merge script, and dispatch dln-sync for Notion writes.

---

## Why This Exists

The dln-sync subagent cannot run Bash or Write tools (Claude Code subagents cannot elevate permissions beyond the parent session). The merge script (`ks-merge.py`) runs in the parent conversation instead.

---

## Parent-Side Merge Sequence

At every sync boundary, follow this sequence:

### 1. Dispatch `fetch`

Dispatch the `dln-sync` agent with:
- **action**: `fetch`
- **page_id**: the Notion page ID for this domain's profile

The agent returns the raw KS block (everything between `<!-- KS:start -->` and `<!-- KS:end -->` markers).

### 2. Construct JSON payload

Using the KS block from step 1 and the teaching boundary outcomes, construct a JSON object conforming to the Normalizer Schema below. Only include fields that have updates.

**Key rule:** Full-rewrite fields (`weakness_queue`, `section_rewrites`) require looking up existing values in the fetched KS block to produce COMPLETE replacement content. Append fields (`mastery_updates`, `section_appends`) only need the boundary outcomes — the script handles lookup and merge.

### 3. Write temp files

Use the **Write tool** to create two files:
- `/tmp/ks-merge-payload-<page_id_8chars>.json` — the JSON payload
- `/tmp/ks-merge-ks-<page_id_8chars>.md` — the raw KS block from step 1

Call both Write operations in parallel since they are independent.

### 4. Run ks-merge.py

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/ks-merge.py" /tmp/ks-merge-payload-<page_id_8chars>.json /tmp/ks-merge-ks-<page_id_8chars>.md
```

- **Exit 0:** stdout is the merged KS block. Proceed to step 5.
- **Exit 1:** Hard fail. Read stderr for error message. Log the error and temp file paths. Queue the writes for the next boundary. Skip to step 6 (dispatch `replace` without merged KS — dln-sync will handle the failure status). Do NOT attempt manual merge.

### 5. Dispatch `replace` (or `replace-end` / `plan-write`)

Dispatch the `dln-sync` agent with:
- **action**: `replace` (mid-session sync), `replace-end` (session end), or `plan-write` (session start)
- **page_id**: the Notion page ID
- **merged_ks**: the stdout from ks-merge.py (step 4)
- **progress_notes**: the progress notes to append to the session log
- **session_number**: current session number
- **phase**: current phase (Dot/Linear/Network)
- **column_updates** (only for `replace-end`): Phase, Session Count, Last Session, Next Review, Review Interval
- **plan_content** (only for `plan-write`): the session plan markdown
- **queued_writes** (if any): previously failed writes to include

The agent returns the compressed re-anchor payload.

### 6. Clean up temp files

On successful return from dln-sync:
```bash
rm -f /tmp/ks-merge-payload-<page_id_8chars>.json /tmp/ks-merge-ks-<page_id_8chars>.md
```

On failure (dln-sync returns `Status.Write: failed`): do NOT clean up — temp files persist for manual inspection.

---

## plan-write Without KS Updates

On the very first session (empty KS), skip steps 1-4 and dispatch `plan-write` directly with no `merged_ks` field. The agent will write the session plan without touching the KS block.

---

## Merge Failure Handling

If ks-merge.py fails (exit 1):
1. Log the error message (stderr) and temp file paths in-conversation.
2. Queue the intended writes for the next boundary.
3. Dispatch `replace` anyway with progress notes only (no `merged_ks`) — session log appends are independent of the KS merge.
4. If 3+ consecutive merge failures: announce to the learner that persistence is temporarily offline. Continue with in-conversation checkpoints only.

---

## Normalizer Schema

JSON format for merge payloads. All fields are optional — only include fields that have updates from the current boundary.

```json
{
  "mastery_updates": [
    {
      "table": "concepts | chains | factors",
      "name": "row identifier — exact match against Concept/Chain/Factor column",
      "status": "not-mastered | partial | mastered",
      "evidence": "string to append to Evidence column (comma-separated)",
      "last_tested": "YYYY-MM-DD",
      "syllabus_topic": "optional — concepts table only"
    }
  ],
  "weakness_queue": [
    {
      "priority": 1,
      "item": "string",
      "type": "concept | chain | factor",
      "phase": "Dot | Linear | Network",
      "severity": "string",
      "source": "string",
      "added": "YYYY-MM-DD"
    }
  ],
  "syllabus_updates": [
    {
      "topic": "string — must match existing syllabus line",
      "status": "checked | unchecked"
    }
  ],
  "section_rewrites": {
    "compressed_model": "COMPLETE replacement text for ## Compressed Model",
    "open_questions": "COMPLETE replacement text for ## Open Questions",
    "interleave_pool": "COMPLETE replacement text for ## Interleave Pool"
  },
  "subsection_rewrites": {
    "calibration_trend": "COMPLETE replacement text for ### Calibration Trend"
  },
  "section_appends": {
    "calibration_concept": "pipe-delimited table row (no header) to append to ### Concept-Level Confidence",
    "calibration_gate": "pipe-delimited table row (no header) to append to ### Gate Predictions",
    "load_session_history": "pipe-delimited table row (no header) to append to ### Session History"
  },
  "load_baseline": {
    "working_batch_size": "number — observed working batch size",
    "hint_tolerance": "string — e.g. 'low (needs <=1 hint per concept)'",
    "recovery_pattern": "string — e.g. 'responds well to different analogies'"
  },
  "engagement": {
    "momentum": "positive | neutral | negative",
    "consecutive_struggles": 0,
    "last_celebration": "string",
    "notes": "string"
  }
}
```

**Rules:**
- `section_rewrites` targets `##`-level headers. Values must be COMPLETE replacement content — not deltas.
- `subsection_rewrites` targets `###`-level headers. Same complete-replacement semantics.
- `weakness_queue` is a full rewrite — the script replaces the entire queue.
- `syllabus_updates` toggles checkboxes in the `## Syllabus` section.
- `mastery_updates` never delete rows. Existing rows are upserted; new rows are appended.

---

## Normalizer Examples

**Example 1: Dot phase sync boundary**

Teaching boundary outcomes:
- Concept "Put-Call Parity" — comprehension check passed. Learner identified C - P = S - PV(K).
- Chain "Option Pricing → Put-Call Parity → Synthetic Positions" — built, traced correctly.
- Weakness queue: remove Put-Call Parity, keep Greeks intuition (priority 1).
- Syllabus topic "Options Basics" — all concepts now mastered.

JSON payload:
```json
{
  "mastery_updates": [
    {"table": "concepts", "name": "Put-Call Parity", "status": "mastered", "evidence": "Comprehension check pass (S4)", "last_tested": "2026-03-16", "syllabus_topic": "Options Basics"},
    {"table": "chains", "name": "Option Pricing → Put-Call Parity → Synthetic Positions", "status": "partial", "evidence": "Chain trace pass (S4)", "last_tested": "2026-03-16"}
  ],
  "weakness_queue": [
    {"priority": 1, "item": "Greeks intuition", "type": "concept", "phase": "Dot", "severity": "not-mastered", "source": "S3 gap", "added": "2026-03-15"}
  ],
  "syllabus_updates": [
    {"topic": "Options Basics", "status": "checked"}
  ]
}
```

**Example 2: Network phase sync boundary**

Teaching boundary outcomes:
- Stress-test: dividend case broke the model. Missing dividend adjustment.
- Contraction: model revised 45 → 32 words, same coverage.
- Engagement: momentum positive, struggles 0.

JSON payload:
```json
{
  "section_rewrites": {
    "compressed_model": "Options pricing is arbitrage-enforced replication. Any derivative payoff decomposable into underlying + bonds has a unique price. Put-call parity is the base case; Greeks measure sensitivity to replication inputs.",
    "open_questions": "- How does continuous dividend yield change the replication argument?"
  },
  "engagement": {
    "momentum": "positive",
    "consecutive_struggles": 0
  }
}
```

- [ ] **Step 2: Verify the file was created correctly**

Run: Read `dunk/skills/dln/references/merge-protocol.md` and confirm all sections are present: Parent-Side Merge Sequence (6 steps), plan-write Without KS Updates, Merge Failure Handling, Normalizer Schema, Normalizer Examples (2 examples).

- [ ] **Step 3: Commit**

```bash
git add dunk/skills/dln/references/merge-protocol.md
git commit -m "feat(dln): add merge-protocol shared reference for parent-side KS merge"
```

---

### Task 2: Remove Bash hook and block-inline-scripts.sh

**Files:**
- Modify: `dunk/hooks/hooks.json` (lines 13-21)
- Delete: `dunk/scripts/block-inline-scripts.sh`

- [ ] **Step 1: Remove Bash matcher from hooks.json**

Edit `dunk/hooks/hooks.json` to remove the Bash matcher entry (lines 12-21), keeping only the notion-update-page matcher. Result:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "mcp__plugin_Notion_notion__notion-update-page",
        "hooks": [
          {
            "type": "command",
            "command": "bash ${CLAUDE_PLUGIN_ROOT}/scripts/validate-ks-markers.sh"
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 2: Delete block-inline-scripts.sh**

```bash
rm dunk/scripts/block-inline-scripts.sh
```

- [ ] **Step 3: Commit**

```bash
git add dunk/hooks/hooks.json
git rm dunk/scripts/block-inline-scripts.sh
git commit -m "chore(dln): remove Bash hook and block-inline-scripts (no longer needed)"
```

---

### Task 3: Update merge-payload-schema.md framing

**Files:**
- Modify: `dunk/references/merge-payload-schema.md` (line 1-3)

- [ ] **Step 1: Update the opening description**

Change line 3 from:
```
JSON contract between the dln-sync normalizer and `ks-merge.py`. The normalizer produces this; the script consumes it. All fields are optional — a dispatch only includes what changed.
```

To:
```
JSON contract between the phase skill normalizer (running in the parent conversation) and `ks-merge.py`. The phase skill produces this per the merge protocol (`dln/references/merge-protocol.md`); the script consumes it. All fields are optional — a dispatch only includes what changed.
```

- [ ] **Step 2: Update the syllabus_updates rule**

Change line 68 from:
```
- **`syllabus_updates`** is consolidated into the merge payload by the normalizer. In the pre-script dln-sync, this was a standalone input field — now it's part of the merge JSON.
```

To:
```
- **`syllabus_updates`** is part of the merge JSON payload. The phase skill includes it when concept mastery changes affect syllabus topic coverage.
```

- [ ] **Step 3: Commit**

```bash
git add dunk/references/merge-payload-schema.md
git commit -m "docs(dln): update merge-payload-schema ownership from dln-sync to phase skills"
```

---

## Chunk 2: Rewrite dln-sync agent

### Task 4: Rewrite dln-sync agent

**Files:**
- Modify: `dunk/agents/dln-sync.md`

This is the largest single change. The agent loses normalize/merge sections and gains the new action set.

- [ ] **Step 1: Update frontmatter (lines 1-25)**

Replace the current frontmatter with:

```yaml
---
name: dln-sync
description: >
  Internal agent — dispatched programmatically by DLN phase skills (dln-dot,
  dln-linear, dln-network) at teaching boundaries. Not user-facing.

  Handles all Notion MCP calls for the DLN running ledger: replacing
  Knowledge State blocks, appending progress notes, reading back page
  content, and updating column properties. Compresses raw read-back into
  a re-anchor payload using the preloaded dln-compress skill format.
model: sonnet
color: blue
tools:
  - mcp__plugin_Notion_notion__notion-fetch
  - mcp__plugin_Notion_notion__notion-update-page
  - mcp__plugin_Notion_notion__notion-search
  - mcp__plugin_Notion_notion__notion-query-database-view
skills:
  - dln-compress
permissionMode: dontAsk
maxTurns: 15
---
```

Changes: removed `Bash`, `Write`, `Read` from tools. Changed `bypassPermissions` to `dontAsk`. Updated description to say "replacing Knowledge State blocks" instead of "updating Knowledge State sections".

- [ ] **Step 2: Remove Golden Rule #4 (line 41)**

Delete:
```
4. **No inline scripts via Bash.** NEVER pass multi-line Python/Ruby/etc. code through `Bash` (e.g., `cat ... | python3 -c "..."`). Multi-line Bash commands with `#`-prefixed lines trigger a security block that cannot be bypassed. Instead: use the **Read** tool to read files, the **Write** tool to write files, and only use Bash for single-line commands like `python3 script.py arg1 arg2` or `rm -f path`.
```

- [ ] **Step 3: Replace Input Format section (lines 45-61)**

Replace the current Input Format with the new action set:

```markdown
## Input Format

You will receive a payload from the teaching skill with these fields:

- **action**: `fetch` | `replace` | `replace-end` | `plan-write`
- **page_id**: The Notion page ID for this domain's profile

### For `fetch` action:
No additional fields. Fetch the page and return the raw KS block.

### For `replace` action:
- **merged_ks**: The pre-merged KS block (output of ks-merge.py, run by the parent)
- **progress_notes**: Progress notes to append to the current session log
- **session_number**: Current session number
- **phase**: Current phase (Dot/Linear/Network)
- **queued_writes**: (optional) Previously failed progress note appends to retry

### For `replace-end` action:
Same as `replace`, plus:
- **column_updates**: Column property updates (Phase, Session Count, Last Session, Next Review, Review Interval)

### For `plan-write` action:
- **merged_ks**: (optional) The pre-merged KS block. Omitted on first session when KS is empty.
- **plan_content**: The session plan markdown to append after the KS block
- **session_number**: Current session number
- **phase**: Current phase (Dot/Linear/Network)
```

- [ ] **Step 4: Remove Normalizer Schema, Examples, and Rules (lines 63-194)**

Delete everything from `## Normalizer Schema` through the end of `**Example 2: Network phase sync dispatch**` (the closing ` ``` ` of the JSON block on line 194). This content now lives in `merge-protocol.md`.

- [ ] **Step 5: Rewrite Execution Protocol (lines 196-317)**

Replace the entire Execution Protocol section with:

```markdown
## Execution Protocol

### For `fetch` action:

1. Call `notion-fetch` with the page_id.
2. Extract the **KS block**: everything from `\<!-- KS:start --\>` through `\<!-- KS:end --\>` (inclusive).
3. **Fallback** — if markers are not found:
   a. Find `# Knowledge State` as the start boundary.
   b. Find the first `## Session` header (or end of content) as the end boundary.
   c. Extract that range as the KS block.
   d. Log a warning: `"Markers not found — migrating page to marker format."`
4. Return the raw KS block as your entire response. No compression, no re-anchor format.

### For `replace` action:

Let SNAPSHOT = the most recent `notion-fetch` result available at each step.

1. **FETCH:** Call `notion-fetch` with the page_id. Save the full page content as SNAPSHOT. Also save everything after `\<!-- KS:end --\>` as the **session logs snapshot**.
2. **REPLACE:** If `merged_ks` is provided, replace the KS block:
   ```json
   {
     "page_id": "...",
     "command": "update_content",
     "content_updates": [{
       "old_str": "<entire KS block from SNAPSHOT — exact bytes from notion-fetch>",
       "new_str": "<merged_ks from dispatch payload>"
     }]
   }
   ```
3. **APPEND:** Append progress notes to the current session's Progress section using a **separate** `update_content` call:
   - Re-fetch the page if the REPLACE step ran (content positions may have shifted). Update SNAPSHOT.
   - In SNAPSHOT, find `## Session {session_number}`.
   - Construct `old_str` from the session header and its existing content (enough trailing lines to guarantee uniqueness).
   - Construct `new_str` as that same content + the new progress notes appended.
   - If `## Session {session_number}` is not found, set `Status.Write` to `failed` for this append and include it in `failed_writes`.
   - If there are `queued_writes`, include them in this append.
4. **VERIFY:** Re-fetch the page. If the REPLACE step was skipped (no `merged_ks`), skip KS verification (steps 4a-4b) and verify only the progress note append (step 4c). Otherwise, confirm:
   a. Both `\<!-- KS:start --\>` and `\<!-- KS:end --\>` are present.
   b. The merged KS content is reflected (spot-check: new rows, updated queue, etc.).
   c. Content after `\<!-- KS:end --\>` includes the appended progress notes.
   d. On verify failure: re-run steps 1-3 with freshly fetched content. Check whether deltas are already present to avoid double-applying. If retry fails: set `Status.Write` to `failed` and include `failed_writes`.
5. **COMPRESS:** Compress the verified page content into the re-anchor format (dln-compress skill). Return the re-anchor payload.

### For `replace-end` action:

1. Run the `replace` action steps above.
2. Update column properties via `update_properties` command: Phase, Session Count, Last Session, Next Review, Review Interval.
3. Return the compressed re-anchor payload.

### For `plan-write` action:

Let SNAPSHOT = the most recent `notion-fetch` result available at each step.

1. **FETCH:** Call `notion-fetch` with the page_id. Save as SNAPSHOT.
2. **REPLACE KS (conditional):** If `merged_ks` is provided, replace the KS block (same as `replace` step 2). Re-fetch and update SNAPSHOT.
3. **APPEND PLAN:** Append the session plan section **after** `\<!-- KS:end --\>`:
   - If no session logs exist yet, use `old_str` = `\<!-- KS:end --\>` and `new_str` = `<!-- KS:end -->\n\n<plan_content>`.
   - If session logs exist, find the last `## Session` header in SNAPSHOT. Use `old_str` = that header and its trailing content (enough lines for uniqueness), and `new_str` = that same content + the new plan appended.
   - If not found in SNAPSHOT, re-fetch once. If still not found, set `Status.Write` to `failed`.
4. **VERIFY:** Re-fetch and confirm plan content is present.
5. **COMPRESS:** Return the compressed re-anchor payload.
```

- [ ] **Step 6: Remove Merging Rules section (lines 319-347)**

Delete the entire `## Merging Rules` section. This content is reference documentation for ks-merge.py and already exists in `dunk/references/merge-payload-schema.md`.

- [ ] **Step 7: Update Error Handling section (lines 353-384)**

Replace the current Error Handling with:

```markdown
## Error Handling

If the REPLACE step fails:
- Run the verify-and-retry logic described in step 4.
- If retry fails, set `Status.Write` to `failed` and include `failed_writes`.
- Still attempt the progress note append and compression with whatever data is available.
- Continue with session log appends even if KS replacement failed — session logs are independent.

If a session log append fails:
- Include it in `failed_writes`.
- The KS replacement is already done — do not roll it back.

If a marker format violation is blocked by the PreToolUse hook:
- You will receive a denial message explaining which marker rule was violated.
- Fix the markers in your `old_str` or `new_str` per the MARKER RULE above and retry.
- Remember: `old_str` uses escaped form from `notion-fetch`, `new_str` uses unescaped form.
```

- [ ] **Step 8: Verify the rewritten agent**

Read the full file and confirm:
- Frontmatter has no `Bash`, `Write`, or `Read` tools
- `permissionMode: dontAsk`
- No references to "normalize", "normalizer", "temp file", "ks-merge.py", or "python3"
- Actions are `fetch`, `replace`, `replace-end`, `plan-write`
- MARKER RULE is preserved
- Compression section is preserved
- Database Reference is preserved

- [ ] **Step 9: Commit**

```bash
git add dunk/agents/dln-sync.md
git commit -m "refactor(dln): rewrite dln-sync as pure Notion I/O agent

Remove normalize/merge steps (now run by parent via merge-protocol).
Remove Bash, Write, Read tools. New actions: fetch, replace,
replace-end, plan-write."
```

---

## Chunk 3: Update phase skills and shared references

### Task 5: Update dln-dot sync loop and write-back

**Files:**
- Modify: `dunk/skills/dln-dot/SKILL.md` (lines 63-67, 126-158, 403-417)

- [ ] **Step 1: Update Session Plan Write dispatch (lines 63-83)**

Change lines 65-66 from:
```
Before any teaching begins, **dispatch the `dln-sync` agent** with action `plan-write`. Include `session_number: <Session Count + 1>` in the dispatch payload, along with the following plan content:
```

To:
```
Before any teaching begins, write the session plan to Notion. If the Knowledge State has existing content (Session Count > 0), follow the full merge protocol in `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/merge-protocol.md` with action `plan-write`. If this is the first session (Session Count = 0, empty KS), dispatch `dln-sync` directly with action `plan-write` (no `merged_ks` needed). Include `session_number: <Session Count + 1>` and the following plan content:
```

- [ ] **Step 2: Update Sync Loop dispatch (lines 126-158)**

Replace lines 126-158 with:

```markdown
### Sync Loop (runs at every teaching boundary)

After each of the following boundaries, **run the merge protocol** in `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/merge-protocol.md` with action `replace`:
- After each concept batch (2-3 concepts) + comprehension check
- After each chain explain-back
- After each worked example
- Before and after the phase gate

**Boundary outcomes** — gather these before running the merge protocol:
- Progress notes to append (append-only, never edit existing blocks):
```
- Concept [X] — delivered, comprehension check: [pass/partial/fail]. [Brief note on learner's response.]
- Concept [Y] — delivered, comprehension check: [pass/partial/fail]. [Brief note.]
- Chain [X→Y] — built. Learner traced [correctly on first attempt / needed N hints].
```
- Knowledge State updates: newly confirmed concepts for `## Concepts`, newly built chains for `## Chains`
- Weakness Queue rebuild: [full updated queue reflecting mastery changes this boundary]
- Syllabus updates: if any concepts changed to `mastered` this boundary, check whether all concepts sharing that `Syllabus Topic` are now mastered. If so, include `syllabus_updates` in the JSON payload.
- Any queued writes from previous failed syncs

**On agent return** — follow the learner-generated checkpoint, plan adjustment, calibration-driven adjustment, and Notion failure handling protocols in `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/sync-protocol.md`.
```

- [ ] **Step 3: Update Notion Write-Back section (lines 403-417)**

Replace lines 403-417 with:

```markdown
## Notion Write-Back

Most write-back happens continuously via the merge protocol during the sync loop. At session end, run the merge protocol one final time with action `replace-end`. Include `session_number: <Session Count + 1>`, along with:

| Target | Field | Action |
|--------|-------|--------|
| Column property | Last Session | Set to today's date |
| Column property | Session Count | Increment by 1 |
| Column property | Phase | Set to **Linear** if phase gate passed; keep **Dot** otherwise |
| Column property | Next Review | Set to computed date (see orchestrator interval rules) |
| Column property | Review Interval | Set to computed interval (see orchestrator interval rules) |
| Page body | Knowledge State | Verify and patch any gaps |
| Page body | Current session Progress | Append final status and exit ritual response |

Database IDs are handled by the `dln-sync` agent — phase skills do not need them.
```

- [ ] **Step 4: Update retrieval warm-up dispatch (line 119)**

Change line 119 from:
```
7. **Dispatch `dln-sync`** with action `sync` after the retrieval warm-up completes. Include retrieval results in the progress notes:
```

To:
```
7. **Run the merge protocol** (`@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/merge-protocol.md`) with action `replace` after the retrieval warm-up completes. Include retrieval results in the progress notes:
```

- [ ] **Step 5: Commit**

```bash
git add dunk/skills/dln-dot/SKILL.md
git commit -m "refactor(dln): update dln-dot to use merge-protocol for sync dispatches"
```

---

### Task 6: Update dln-linear sync loop and write-back

**Files:**
- Modify: `dunk/skills/dln-linear/SKILL.md` (lines 40-42, 103, 105-124, 391-405)

- [ ] **Step 1: Update Session Plan Write dispatch (lines 40-42)**

Change line 42 from:
```
Before any teaching begins, **dispatch the `dln-sync` agent** with action `plan-write`. Include `session_number: <Session Count + 1>` in the dispatch payload, along with the following plan content:
```

To:
```
Before any teaching begins, write the session plan to Notion. Follow the merge protocol in `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/merge-protocol.md` with action `plan-write`. Include `session_number: <Session Count + 1>` and the following plan content:
```

- [ ] **Step 2: Update retrieval warm-up dispatch (line 103)**

Change line 103 from:
```
8. **Dispatch `dln-sync`** with retrieval results in progress notes.
```

To:
```
8. **Run the merge protocol** (`@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/merge-protocol.md`) with action `replace`. Include retrieval results in progress notes.
```

- [ ] **Step 3: Update Sync Loop dispatch (lines 105-124)**

Replace lines 105-124 with the same pattern as dln-dot Task 5 Step 2, but using Linear-specific boundaries and progress note templates:

```markdown
### Sync Loop (runs at every teaching boundary)

After each of the following boundaries, **run the merge protocol** in `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/merge-protocol.md` with action `replace`:
- After each cross-pollination comparison
- After each factor hypothesis + precision rating
- After each upgrade operator round
- Before and after the phase gate

**Boundary outcomes** — gather these before running the merge protocol:
- `session_number`: current session number (Session Count + 1)
- Progress notes to append (append-only):
```
- Cross-pollination [Chain A vs Chain B] — learner identified [shared structure / missed it]. Precision: [vague/structural/transferable].
- Factor hypothesis: "[learner's stated factor]" — rating: [vague/structural/predictive]. [Notes on precision pushback.]
- Upgrade operator: converted [Dot question] → [Linear question]. [Success/needed guidance.]
```
- Knowledge State updates: newly confirmed factors for `## Factors`, updated chains
- Weakness Queue rebuild: [full updated queue reflecting mastery changes]
- Any queued writes from previous failed syncs

**On agent return** — follow the learner-generated checkpoint, plan adjustment, calibration-driven adjustment, and Notion failure handling protocols in `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/sync-protocol.md`.
```

- [ ] **Step 4: Update Notion Write-Back section (lines 391-405)**

Replace with the same pattern as dln-dot Task 5 Step 3, using Linear-specific column updates:

```markdown
## 5. Notion Write-Back

Most write-back happens continuously via the merge protocol during the sync loop. At session end, run the merge protocol one final time with action `replace-end`. Include `session_number: <Session Count + 1>`, along with:

| Target | Field | Action |
|--------|-------|--------|
| Column property | Last Session | Set to today's date |
| Column property | Session Count | Increment by 1 |
| Column property | Phase | Set to **Network** if phase gate passed |
| Column property | Next Review | Set to computed date (see orchestrator interval rules) |
| Column property | Review Interval | Set to computed interval (see orchestrator interval rules) |
| Page body | Knowledge State | Verify Factors and Open Questions are complete |
| Page body | Current session Progress | Append final status and exit ritual response |

Database IDs are handled by the `dln-sync` agent — phase skills do not need them.
```

- [ ] **Step 5: Commit**

```bash
git add dunk/skills/dln-linear/SKILL.md
git commit -m "refactor(dln): update dln-linear to use merge-protocol for sync dispatches"
```

---

### Task 7: Update dln-network sync loop and write-back

**Files:**
- Modify: `dunk/skills/dln-network/SKILL.md` (lines 24-26, 56-76, 300-315)

- [ ] **Step 1: Update Session Plan Write dispatch (lines 24-26)**

Change line 26 from:
```
Before asking for the learner's model, **dispatch the `dln-sync` agent** with action `plan-write`. Include `session_number: <Session Count + 1>` in the dispatch payload, along with the following plan content:
```

To:
```
Before asking for the learner's model, write the session plan to Notion. Follow the merge protocol in `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/merge-protocol.md` with action `plan-write`. Include `session_number: <Session Count + 1>` and the following plan content:
```

- [ ] **Step 2: Update Sync Loop dispatch (lines 56-76)**

Replace with:

```markdown
### Sync Loop (runs at every teaching boundary)

After each of the following boundaries, **run the merge protocol** in `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/merge-protocol.md` with action `replace`:
- After state model capture (Step 1)
- After each stress-test round (Step 2)
- After each contraction attempt (Step 4)
- After the transfer test (Step 5)

**Boundary outcomes** — gather these before running the merge protocol:
- `session_number`: current session number (Session Count + 1)
- Progress notes to append (append-only):
```
- State model captured: "[verbatim model]"
- Stress-test [N]: [edge case presented] → model [held/broke]. [What was missing.]
- Contraction [N]: model revised — [word count before] → [word count after]. Coverage: [broader/same/narrower].
- Transfer test: [adjacent domain] → model [transferred successfully / broke at X].
```
- Knowledge State updates: replace `## Compressed Model` with latest revision, append new factors to `## Factors`, update `## Open Questions` with remaining gaps
- Weakness Queue rebuild: [full updated queue reflecting factor mastery changes]
- Any queued writes from previous failed syncs

**On agent return** — follow the learner-generated checkpoint, plan adjustment, calibration-driven adjustment, and Notion failure handling protocols in `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/sync-protocol.md`. For Network phase, plan adjustments use this format:

```
### Plan Adjustment — [reason]
- Adding stress-tests: [new edge cases to probe]
- Shifting transfer domain: [original] → [new target]
```

- [ ] **Step 3: Update Notion Write-Back section (lines 300-315)**

Replace with:

```markdown
## Notion Write-Back

Most write-back happens continuously via the merge protocol during the sync loop. At session end, run the merge protocol one final time with action `replace-end`. Include `session_number: <Session Count + 1>`, along with:

| Target | Field | Action |
|--------|-------|--------|
| Column property | Last Session | Set to today's date |
| Column property | Session Count | Increment by 1 |
| Column property | Next Review | Set to computed date (see orchestrator interval rules) |
| Column property | Review Interval | Set to computed interval (see orchestrator interval rules) |
| Page body | Knowledge State | Verify Compressed Model, Factors, and Open Questions reflect final state |
| Page body | Current session Progress | Append exit ritual summary (starting model, what broke, revised model, open questions) |

No Phase column update — Network is the terminal phase.

Database IDs are handled by the `dln-sync` agent — phase skills do not need them.
```

- [ ] **Step 4: Commit**

```bash
git add dunk/skills/dln-network/SKILL.md
git commit -m "refactor(dln): update dln-network to use merge-protocol for sync dispatches"
```

---

### Task 8: Update sync-protocol.md

**Files:**
- Modify: `dunk/skills/dln/references/sync-protocol.md` (lines 27, 63-79, 84-87)

- [ ] **Step 1: Update Plan Adjustment dispatch reference (line 27)**

Change line 27 from:
```
If the re-anchor payload reveals drift from the original plan, include a **plan adjustment** in the next `dln-sync` dispatch to append:
```

To:
```
If the re-anchor payload reveals drift from the original plan, include a **plan adjustment** in the next merge protocol run to append:
```

- [ ] **Step 2: Remove Syllabus Updates section (lines 61-79)**

Delete the entire `## Syllabus Updates` section (lines 61-79). Syllabus checkbox toggling is now handled by `ks-merge.py` via the `syllabus_updates` field in the JSON payload. The merge-protocol.md normalizer schema documents this.

- [ ] **Step 3: Rewrite Notion Failure Handling (lines 82-87)**

Replace lines 82-87 with:

```markdown
## Notion Failure Handling

The merge protocol involves two dispatches: `fetch` then `replace`. Failures can occur at either step.

**If `fetch` fails** (dln-sync returns error or empty KS):
1. Log the failure in-conversation.
2. Skip the merge — you cannot merge without a KS block.
3. Dispatch `replace` with progress notes only (no `merged_ks`). Session log appends are independent.

**If `replace` returns with `Status.Write: failed`:**
1. Log the intended update in-conversation as a visible checkpoint.
2. Queue the failed writes — include them in the next merge protocol run (as `queued_writes`).
3. If 3+ consecutive dispatches return failure, announce to the learner that persistence is temporarily offline. Continue with in-conversation checkpoints only. Attempt a single bulk write-back at session end.
```

- [ ] **Step 4: Commit**

```bash
git add dunk/skills/dln/references/sync-protocol.md
git commit -m "refactor(dln): update sync-protocol for two-dispatch merge model"
```

---

### Task 9: Final verification

- [ ] **Step 1: Grep for stale references**

Search for any remaining references to the old action names or patterns that should have been updated:

```bash
# Should return 0 results for these patterns in skills/ and agents/:
grep -r "action.*sync" dunk/skills/ dunk/agents/ --include="*.md" | grep -v "merge-protocol" | grep -v "sync-protocol" | grep -v "sync loop" | grep -v "sync boundary"
grep -r "action.*session-end" dunk/skills/ dunk/agents/ --include="*.md"
grep -r "dispatch.*dln-sync.*sync" dunk/skills/ --include="*.md"
```

Expected: no results matching old dispatch patterns. References to "sync loop" and "sync boundary" are fine (they describe the teaching concept, not the action name).

- [ ] **Step 2: Verify dln-sync has no Bash/Write/Read references**

```bash
grep -E "Bash|Write tool|Read tool|python3|ks-merge|temp file|normalize" dunk/agents/dln-sync.md
```

Expected: no matches.

- [ ] **Step 3: Verify merge-protocol.md is referenced by all three phase skills**

```bash
grep -r "merge-protocol" dunk/skills/ --include="*.md"
```

Expected: matches in dln-dot/SKILL.md, dln-linear/SKILL.md, dln-network/SKILL.md.

- [ ] **Step 4: Commit any fixups if needed, then final commit**

```bash
git add -A dunk/
git status
# If clean, no commit needed. If fixups were made:
git commit -m "fix(dln): address stale references from pre-merge refactor"
```
