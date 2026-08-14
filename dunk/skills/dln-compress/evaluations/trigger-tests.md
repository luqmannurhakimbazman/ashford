# DLN Compress Skill — Trigger Tests

## SHOULD Trigger

### T1: Agent preload
**Context:** The `dln-sync` agent preloads `dln-compress` while returning a Notion page read-back.
**Expected:** Applies the exact re-anchor compression format without adding teaching content.

### T2: Knowledge-state compression
**Context:** A preloaded dln-sync task asks to compress Concepts, Chains, Factors, weaknesses, and engagement signals.
**Expected:** Produces the documented `## Re-anchor` structure concisely.

## SHOULD NOT Trigger

### T3: User asks to compress prose
**Input:** "Compress this essay to 200 words."
**Expected:** Does not activate `dln-compress`; this is not a DLN sync read-back.

### T4: User names the internal skill
**Input:** "Run dln-compress for me."
**Expected:** Does not expose or activate the internal skill directly; routes through `dln` if the user wants DLN learning.

### T5: Teaching request
**Input:** "Teach me the basics of immunology."
**Expected:** Activates `dln` or `dln-dot`, not `dln-compress`.
