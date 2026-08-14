# DLN Linear Skill — Trigger Tests

## SHOULD Trigger

### T1: Orchestrator route
**Context:** The DLN profile reports `Phase = Linear`.
**Expected:** Activates `dln-linear` and starts factor discovery across known chains.

### T2: Explicit phase request
**Input:** "Run a Linear session on options pricing."
**Expected:** Activates `dln-linear`.

### T3: Shared structure
**Input:** "What do my inflation, rates, and FX chains have in common?"
**Expected:** Activates `dln-linear` and elicits a domain-independent factor.

### T4: Pattern discovery
**Input:** "Help me find transferable patterns across the procedures I already know."
**Expected:** Activates `dln-linear`.

## SHOULD NOT Trigger

### T5: No foundations
**Input:** "I know nothing about immunology; teach me from scratch."
**Expected:** Routes to `dln-dot`, not `dln-linear`.

### T6: Compression and edge cases
**Input:** "Try to break my mental model and help me compress it."
**Expected:** Routes to `dln-network`, not `dln-linear`.

### T7: Unrelated comparison
**Input:** "Compare these two laptops."
**Expected:** Does not activate `dln-linear`.
