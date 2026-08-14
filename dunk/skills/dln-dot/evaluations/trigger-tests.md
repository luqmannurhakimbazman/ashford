# DLN Dot Skill — Trigger Tests

## SHOULD Trigger

### T1: Orchestrator route
**Context:** The DLN profile reports `Phase = Dot`.
**Expected:** Activates `dln-dot` and begins the foundational session flow.

### T2: Explicit zero knowledge
**Input:** "I know nothing about derivatives. Start from zero."
**Expected:** Activates `dln-dot` and uses high-delivery foundational teaching.

### T3: Basics request
**Input:** "Teach me the basics of distributed systems from the ground up."
**Expected:** Activates `dln-dot` after the domain is established.

### T4: Foundational gap
**Input:** "I can't explain any of the core concepts in bond pricing yet."
**Expected:** Activates `dln-dot` when used within the DLN learning workflow.

## SHOULD NOT Trigger

### T5: Cross-chain abstraction
**Input:** "Help me find the shared principle across these three causal chains."
**Expected:** Routes to `dln-linear`, not `dln-dot`.

### T6: Model stress test
**Input:** "Stress-test my compressed model with counterexamples."
**Expected:** Routes to `dln-network`, not `dln-dot`.

### T7: Simple factual answer
**Input:** "Define duration in one sentence."
**Expected:** Does not activate a full Dot session unless structured DLN learning is requested.
