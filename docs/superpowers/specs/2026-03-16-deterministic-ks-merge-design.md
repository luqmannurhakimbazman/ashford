# Deterministic KS Merge Script

**Date:** 2026-03-16
**Status:** Draft
**Scope:** dunk plugin — dln-sync agent MERGE step, new merge script, payload schema
**Depends on:** Atomic KS Sync (2026-03-16-atomic-ks-sync-design.md) — must be implemented first

---

## Problem

The atomic KS sync protocol (already implemented) fixes the "multiple matches" bug by replacing the entire KS block as one unit. But the MERGE step — where deltas from the phase skill are applied to the fetched KS snapshot — is still done by Sonnet via inline string manipulation. The agent parses pipe-delimited markdown tables, matches rows by name, appends evidence strings, toggles checkboxes, and reassembles the block. This is mechanical work that an LLM does unreliably:

- Row matching in markdown tables is error-prone under context pressure
- Evidence appending can duplicate entries on retry
- The double-apply protection logic (checking whether deltas are already present) requires conditional string matching that Sonnet gets wrong sometimes
- Each sync boundary burns ~3-5 Sonnet turns on string manipulation that a script does perfectly

9 of 11 dln-sync responsibilities are deterministic. Only compression genuinely needs an LLM. The merge step is the largest block of mechanical work still done by the LLM.

## Solution

Replace the LLM-driven merge with a two-step process:

1. **Normalizer** (LLM, one turn) — dln-sync reads the prose dispatch payload from the phase skill and produces a structured JSON file conforming to a strict schema.
2. **Merge script** (deterministic, Python) — takes the JSON payload + raw KS block, outputs the merged KS block to stdout.

This is **Option A** from the design exploration: phase skills keep sending prose-style payloads, the LLM interprets once (normalization), and the mechanical merge is deterministic. Phase skills are unchanged.

### Why Option A over full formalization (Option B)

Option B (phase skills send typed JSON directly) eliminates the normalizer entirely but requires rewriting dispatch sections in all three phase skills. Option A is lower risk — no phase skill changes, and if the normalizer proves reliable, we can evolve to B later by teaching phase skills to produce JSON directly, removing the normalizer step.

---

## Architecture

```
Phase skill (LLM)
  → prose dispatch payload (unchanged)
    → dln-sync FETCH (Notion MCP)
      → raw KS block (between markers)
    → dln-sync NORMALIZE (LLM, 1 turn)
      → /tmp/ks-merge-payload-<page_id>-<timestamp>.json
      → /tmp/ks-merge-ks-<page_id>-<timestamp>.json
    → ks-merge.py (deterministic)
      → merged KS block (stdout)
    → dln-sync REPLACE (Notion MCP)
    → dln-sync VERIFY (Notion MCP)
    → dln-sync COMPRESS (LLM)
      → re-anchor payload
```

### Turn budget impact

Current dln-sync MERGE step: ~3-5 Sonnet turns (string manipulation).
New MERGE step: 1 Sonnet turn (normalize) + 1 Bash call (script). Net savings: 2-4 turns per sync boundary.

---

## Merge Payload Schema

The JSON contract that the dln-sync normalizer must produce. All fields are optional — a dispatch only includes what changed. The script ignores absent fields.

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
    "compressed_model": "full replacement text for ## Compressed Model",
    "open_questions": "full replacement text for ## Open Questions",
    "interleave_pool": "full replacement text for ## Interleave Pool",
    "calibration_trend": "full replacement text for ### Calibration Trend"
  },
  "section_appends": {
    "calibration_concept": "pipe-delimited table row (no header) to append to ### Concept-Level Confidence",
    "calibration_gate": "pipe-delimited table row (no header) to append to ### Gate Predictions",
    "load_session_history": "pipe-delimited table row (no header) to append to ### Session History"
  },
  "engagement": {
    "momentum": "positive | neutral | negative",
    "consecutive_struggles": 0,
    "last_celebration": "string",
    "notes": "string"
  }
}
```

This schema is documented in `dunk/references/merge-payload-schema.md` and included verbatim in the dln-sync normalizer prompt so Sonnet knows exactly what structure to produce.

---

## Merge Script Specification

### File: `dunk/scripts/ks-merge.py`

**Runtime:** Python 3.8+ stdlib only. No pip dependencies.

### Interface

```
python3 ks-merge.py <payload_path> <ks_block_path>
```

- **Arg 1:** Path to JSON payload file (schema above)
- **Arg 2:** Path to file containing raw KS block (markdown between markers, inclusive of `<!-- KS:start -->` and `<!-- KS:end -->`)
- **stdout:** Merged KS block with unescaped markers
- **stderr:** Warnings (non-fatal) or error message (fatal)
- **Exit 0:** Success — stdout has the merged block
- **Exit 1:** Failure — stdout empty or contains original KS block unchanged

### Operations (in order)

1. **Parse JSON payload** — exit 1 if malformed JSON or unexpected structure.
2. **Parse KS block into sections** — split on `## ` headers. Preserve section order.
3. **mastery_updates** — For each entry:
   - Identify target table by `table` field: `concepts` → `## Concepts`, `chains` → `## Chains`, `factors` → `## Factors`.
   - Find existing row by matching the name column (first data column after `|`). Match is exact, case-sensitive.
   - If found: update `Status` if provided, append `evidence` to `Evidence` column (comma-separated, space after comma), set `Last Tested` to `last_tested` value.
   - If not found: append new row with all provided fields. For concepts table, include `Syllabus Topic` column if `syllabus_topic` is provided.
   - Never delete rows.
4. **weakness_queue** — Find `## Weakness Queue` section. Replace everything between the header line and the next `##` header (or end of KS block) with the pipe-delimited table header row + new data rows from the payload. This is a full rewrite, not a merge.
5. **syllabus_updates** — Find lines matching `- [ ] <topic>` or `- [x] <topic>` within the `## Syllabus` section. Toggle to `- [x]` if status is `checked`, `- [ ]` if `unchecked`. If topic not found, warn to stderr, skip.
6. **section_rewrites** — For each key:
   - Map key to header: `compressed_model` → `## Compressed Model`, `open_questions` → `## Open Questions`, `interleave_pool` → `## Interleave Pool`, `calibration_trend` → `### Calibration Trend`.
   - Replace content between the header and the next same-or-higher-level header with the provided text.
7. **section_appends** — For each key:
   - Map key to header: `calibration_concept` → `### Concept-Level Confidence`, `calibration_gate` → `### Gate Predictions`, `load_session_history` → `### Session History`.
   - Find the last `|...|` line in that section's table. Append the new row after it.
8. **engagement** — Find each key-value line in `## Engagement Signals`:
   - `momentum` → `- Momentum: <value>`
   - `consecutive_struggles` → `- Consecutive struggles: <value>`
   - `last_celebration` → `- Last celebration: <value>`
   - `notes` → `- Notes: <value>`
   - Replace only the value portion after the colon.
9. **Reassemble** all sections in original order, wrap in `<!-- KS:start -->` and `<!-- KS:end -->` (unescaped), output to stdout.

### Invariants

- Section order is preserved — the script never reorders sections.
- Unrecognized sections pass through untouched — the script only modifies sections it has operations for.
- Empty/absent payload fields are skipped, not errored.
- The script never creates new sections — it only modifies existing ones. If a target section doesn't exist in the KS block, warn to stderr and skip that operation.

---

## dln-sync Agent Changes

### Modified Step 2: MERGE

Replace the current MERGE step (which instructs the agent to apply deltas manually) with:

#### Step 2a: NORMALIZE

Read the prose dispatch payload from the phase skill. Produce a JSON object conforming to the merge payload schema (included verbatim in the agent prompt). Write it to:

```
/tmp/ks-merge-payload-<page_id>-<timestamp>.json
```

Also write the raw KS block from Step 1 (FETCH) to:

```
/tmp/ks-merge-ks-<page_id>-<timestamp>.json
```

Use the page_id (first 8 chars) and Unix timestamp for uniqueness.

#### Step 2b: MERGE

Call the merge script:

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/ks-merge.py" /tmp/ks-merge-payload-<page_id>-<timestamp>.json /tmp/ks-merge-ks-<page_id>-<timestamp>.json
```

If exit 0: use stdout as the merged KS block for Step 3 (REPLACE).
If exit 1: hard fail — set `Status.Write` to `failed`, include stderr message and temp file paths in the re-anchor payload, queue writes for next boundary. Skip to compression.

### Merging Rules section

Keep the existing Merging Rules section in dln-sync as documentation. It serves as the source of truth for what the script implements. The agent no longer executes these rules manually — but they remain as reference for anyone reading the agent definition or maintaining the script.

### Normalizer prompt addition

Add the merge payload schema verbatim to the dln-sync agent definition, inside a new section between Input Format and Execution Protocol:

```markdown
## Normalizer Schema

When producing the merge payload JSON, use exactly this structure. All fields are optional — only include fields that have updates from the dispatch.

<schema verbatim from merge-payload-schema.md>
```

---

## Failure Handling and Diagnostics

### On merge script failure (exit 1):

1. **Temp files persist** — `/tmp/ks-merge-payload-*` and `/tmp/ks-merge-ks-*` are NOT cleaned up. They remain for manual inspection.
2. **dln-sync reads stderr** — captures the script's error message.
3. **Re-anchor payload includes diagnostics:**
   ```
   ### Status
   - Write: failed
   - Failed writes: [list of intended updates from the dispatch]
   - Merge error: "<stderr message from ks-merge.py>"
   - Debug artifacts:
     - Payload: /tmp/ks-merge-payload-<page_id>-<timestamp>.json
     - KS block: /tmp/ks-merge-ks-<page_id>-<timestamp>.json
   ```
4. **Writes queued** for next boundary per existing Notion Failure Handling protocol in sync-protocol.md.
5. **Session log appends still attempted** — they are independent of the KS merge and use a separate `update_content` call.

### On merge script success (exit 0):

Temp files are cleaned up by dln-sync after VERIFY (Step 4) passes. If VERIFY fails and the retry also fails, temp files persist for diagnostics.

### Diagnosing failures between sessions

The three failure points and how to attribute them:

1. **Phase skill sent bad prose** — the JSON payload will be missing fields or have wrong values. The prose was ambiguous or incomplete. Fix: improve the phase skill's dispatch instructions.
2. **Normalizer produced bad JSON** — the JSON payload doesn't match the schema, or it misinterpreted the prose. The prose was clear but the LLM got it wrong. Fix: improve the normalizer prompt in dln-sync, add examples.
3. **Merge script has a bug** — the JSON is valid, the KS block is valid, but the script crashed or produced wrong output. Fix: fix the Python code.

To diagnose: read the payload JSON first. If it looks right, the script has a bug. If it looks wrong, compare it to the original prose dispatch (visible in the dln-sync agent's conversation transcript) to determine whether the prose or the normalizer was at fault.

---

## Files Changed

| File | Change |
|------|--------|
| `dunk/scripts/ks-merge.py` | **New** — deterministic merge script |
| `dunk/references/merge-payload-schema.md` | **New** — JSON contract for normalizer output |
| `dunk/agents/dln-sync.md` | **Modified** — MERGE step becomes normalize + script call; schema added to prompt; failure handling updated |

## Files NOT Changed

| File | Why |
|------|-----|
| Phase skills (dln-dot, dln-linear, dln-network) | They keep sending prose. Option A means no phase skill changes. |
| `dln-compress` | Compression format unchanged. |
| `dunk/hooks/hooks.json` | PreToolUse hook unchanged — it validates markers, not merge payloads. |
| `dunk/skills/dln/references/init-template.md` | Template unchanged. |
| `dunk/skills/dln/SKILL.md` | Orchestrator unchanged. |

## Testing

1. **Unit test: valid payload + populated KS block** — run the script locally with a realistic JSON payload and a KS block copied from init-template with some populated rows. Verify: rows upserted, queue replaced, checkboxes toggled, engagement updated, markers present, section order preserved.
2. **Unit test: empty payload** — all fields absent. Verify: output equals input KS block unchanged.
3. **Unit test: new rows** — mastery_updates with names not in the table. Verify: rows appended, existing rows untouched.
4. **Unit test: section_rewrites** — replace compressed model and open questions. Verify: old content gone, new content in place, adjacent sections untouched.
5. **Unit test: malformed JSON** — invalid JSON input. Verify: exit 1, stderr has error message, stdout empty.
6. **Unit test: missing section** — mastery_updates targeting `## Factors` when KS block has no Factors section. Verify: warning to stderr, other operations still complete, exit 0.
7. **Integration test: full sync boundary** — run a Dot session with 2+ sync boundaries. Verify dln-sync normalizes, calls script, writes successfully, and re-anchor payload has correct data.

## Alternatives Considered

### Full formalization (Option B) — phase skills send typed JSON directly

Eliminates the normalizer step entirely. Phase skills construct the structured payload themselves and dln-sync passes it straight to the script.

**Why deferred:** Requires rewriting dispatch sections in all three phase skills. Higher risk of breaking teaching flow if payload construction instructions interfere with pedagogical prompts. Option A can evolve into Option B later by teaching phase skills to produce JSON directly, then removing the normalizer. The merge script and schema are identical in both options — only the input source changes.

### Debug agent on failure

A subagent that spins up when the merge script fails, reads the temp files, fixes the JSON, and retries.

**Why rejected:** The three failure points (bad prose, bad normalization, script bug) each require different fixes. A debug agent trying to fix JSON is just a second normalizer — same problem, different LLM call. For script bugs, no LLM can help. The temp files + stderr provide enough information for manual diagnosis between sessions. Adding an LLM retry step moves away from the goal of making the merge path deterministic.

### LLM fallback on script failure

If the script fails, dln-sync falls back to doing the merge manually (the current behavior).

**Why rejected:** Maintaining two merge paths (script + LLM fallback) doubles the surface area for bugs. The Merging Rules stay as documentation for the script's behavior, but the agent should not execute them. Hard fail + queue writes is simpler and forces failures to be diagnosed and fixed rather than silently worked around.
