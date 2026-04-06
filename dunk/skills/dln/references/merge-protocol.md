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
- **old_ks**: the raw KS block from step 1's `fetch` dispatch (the agent uses this as `old_str` for KS replacement, avoiding a redundant full-page fetch)
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
