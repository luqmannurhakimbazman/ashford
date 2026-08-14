# DLN Network Skill — Trigger Tests

## SHOULD Trigger

### T1: Orchestrator route
**Context:** The DLN profile reports `Phase = Network`.
**Expected:** Activates `dln-network` and begins a distributed revision cycle.

### T2: Explicit phase request
**Input:** "Run a Network session on market microstructure."
**Expected:** Activates `dln-network`.

### T3: Stress test
**Input:** "Stress-test my model of monetary policy with edge cases and counterexamples."
**Expected:** Activates `dln-network`.

### T4: Compression request
**Input:** "Help me compress my understanding of compiler optimization without losing coverage."
**Expected:** Activates `dln-network` when the learner already has factors and a working model.

### T5: Transfer test
**Input:** "Test whether my model transfers to a completely different domain."
**Expected:** Activates `dln-network` within an established DLN domain.

## SHOULD NOT Trigger

### T6: Beginner request
**Input:** "Teach me accounting from zero."
**Expected:** Routes to `dln-dot`, not `dln-network`.

### T7: Factor discovery
**Input:** "Help me discover what these chains share."
**Expected:** Routes to `dln-linear`, not `dln-network`.

### T8: Software network troubleshooting
**Input:** "Why can't my laptop connect to the Wi-Fi network?"
**Expected:** Does not activate `dln-network`; “network” is being used in an unrelated sense.
