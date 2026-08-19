# DLN Compress Skill — Trigger Tests

## SHOULD Trigger

### T1: Optional internal preload
**Context:** A phase caller explicitly supplies validated `dln-store context` output and needs a smaller machine-readable subset.
**Expected:** Returns a compact state object while preserving revision, stage, event IDs, independent/supported separation, and `not-measured` statuses.

### T2: Missing field
**Context:** The supplied state lacks revision or stage.
**Expected:** Returns a missing-field diagnostic and asks the caller to reload local context; it does not fill gaps from dialogue.

## SHOULD NOT Trigger

### T3: User prose compression
**Input:** "Compress this essay to 200 words."
**Expected:** Does not activate `dln-compress`.

### T4: Direct name
**Input:** "Run dln-compress for me."
**Expected:** Does not expose the internal formatter; routes through `dln` only if structured learning is intended.

### T5: Learner model compression
**Input:** "Help me revise and compress my model after testing its predictions."
**Expected:** Routes to `dln-network`; internal context formatting never changes a learner model.

### T6: Raw dialogue
**Context:** A caller supplies session dialogue and asks for evidence/mastery extraction.
**Expected:** Refuses; dialogue and notes are not evidence.

## CONTRACT

### T7: No side effects
**Context:** Valid projected state is compacted.
**Expected:** Creates no event, profile patch, persistence write, receipt, dashboard, or learner-facing artifact; makes no readiness decision; preserves `active_decision`/`decision_event_id`; and never invokes prepare/propose/decide or promotes supplements/pending sources.
