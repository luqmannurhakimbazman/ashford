# DLN Linear Skill — Trigger Tests

## SHOULD Trigger

### T1: Orchestrator route
**Context:** Local `state.stage` is `relate`.
**Expected:** Activates `dln-linear` under the learner-facing Relate/Abstract name using bounded local context.

### T2: Historical trigger
**Input:** "Run a Linear session on options pricing."
**Expected:** Keeps the backward-compatible skill ID while describing the operation as Relate/Abstract.

### T3: Shared structure
**Input:** "What do my inflation, rates, and FX chains have in common?"
**Expected:** Elicits a relation and abstraction before teaching the target factor.

### T4: Transfer
**Input:** "Test whether this principle works on an unfamiliar kind of system."
**Context:** The local domain is in `relate`.
**Expected:** Uses an unsignaled task and marks `novelty: novel` only when the structure is genuinely new.

## SHOULD NOT Trigger

### T5: No foundation
**Input:** "I know nothing about immunology; teach me from scratch."
**Expected:** Routes to `dln-dot`, not `dln-linear`.

### T6: Model revision
**Input:** "Try to break my current model and help me rewrite it."
**Expected:** Routes to `dln-network`, not `dln-linear`.

### T7: Unrelated comparison
**Input:** "Compare these two laptops."
**Expected:** Does not activate `dln-linear`.

## CONTRACT

### T8: Assistance
**Context:** The tutor names the shared factor before the learner answers.
**Expected:** Any resulting assessment is supported, not independent.

### T9: Gate
**Context:** Relate-to-Revise gate is attempted.
**Expected:** Requires committed independent relation/abstraction evidence and a genuinely novel transfer, then cites those IDs in `stage_transition`.

### T10: Completion
**Context:** Session ends.
**Expected:** Presents the generated Session Receipt and no competing summary.
