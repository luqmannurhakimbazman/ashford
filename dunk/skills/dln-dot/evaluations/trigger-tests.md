# DLN Dot Skill — Trigger Tests

## SHOULD Trigger

### T1: Orchestrator route
**Context:** Local `state.stage` is `acquire`.
**Expected:** Activates `dln-dot` under the learner-facing Acquire/Discriminate name and uses bounded local context.

### T2: Explicit zero knowledge
**Input:** "I know nothing about derivatives. Start from zero."
**Expected:** Routes through `dln` setup and then `dln-dot` for foundation teaching plus distinct structured checks.

### T3: Discrimination request
**Input:** "Help me stop confusing duration and convexity."
**Context:** The local domain is in `acquire`.
**Expected:** Uses a predeclared discrimination task and records assistance honestly.

### T4: Delayed review
**Context:** A prior same-subject assessment is due after a positive delay.
**Expected:** Asks unaided retrieval before cues and links the new assessment to the prior event.

## SHOULD NOT Trigger

### T5: Relational abstraction
**Input:** "Find the shared structure across these causal chains."
**Expected:** Routes to `dln-linear`, not `dln-dot`.

### T6: Model stress test
**Input:** "Make my model predict edge cases, then help me revise it."
**Expected:** Routes to `dln-network`, not `dln-dot`.

### T7: One-sentence fact
**Input:** "Define duration in one sentence."
**Expected:** Does not start a persistent stage session unless structured DLN learning is requested.

## CONTRACT

### T8: Teaching is not evidence
**Context:** The tutor explains a concept and gives an example, but performs no distinct check.
**Expected:** Creates no assessment event.

### T9: Gate
**Context:** Acquire-to-Relate gate is attempted.
**Expected:** Commits independent acquire/discriminate assessments first, then cites them in `stage_transition`; supported work and coverage cannot pass.

### T10: Completion
**Context:** Session ends.
**Expected:** Commits terminal completion and presents the generated Session Receipt rather than an improvised recap.

### T11: Approved course grounding
**Context:** `state.grounding` is approved and a citable planning topic selects the task.
**Expected:** Uses the active decision and backing settled assertion IDs on assessment/completion events via `decision_event_id`, keeps unresolved ambiguity unresolved, labels outside material supplemental, and never treats syllabus coverage as evidence. It does not invoke prepare/propose/decide or promote supplements/pending sources.
