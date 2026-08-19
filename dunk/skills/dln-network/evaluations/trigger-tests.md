# DLN Network Skill — Trigger Tests

## SHOULD Trigger

### T1: Orchestrator route
**Context:** Local `state.stage` is `revise`.
**Expected:** Activates `dln-network` under the learner-facing Predict/Revise/Compress name using bounded local context.

### T2: Historical trigger
**Input:** "Run a Network session on market microstructure."
**Expected:** Preserves the backward-compatible skill ID while using pre-outcome predictions and cited revisions.

### T3: Stress test
**Input:** "Stress-test my model of monetary policy with edge cases."
**Expected:** Elicits the model's prediction before revealing the result.

### T4: Learner model compression
**Input:** "Help me make my compiler model shorter without losing predictive coverage."
**Context:** The local domain is in `revise` and has a current model.
**Expected:** Uses prediction/retest evidence and records learner-authored `model_revision`; it does not invoke internal context compression as pedagogy.

### T5: Transfer
**Input:** "Test whether my model transfers to a completely different domain."
**Expected:** Uses an unsignaled prediction and marks novelty honestly.

## SHOULD NOT Trigger

### T6: Beginner
**Input:** "Teach me accounting from zero."
**Expected:** Routes to `dln-dot`.

### T7: Factor discovery
**Input:** "Help me discover what these chains share."
**Expected:** Routes to `dln-linear`.

### T8: Software network issue
**Input:** "Why can't my laptop connect to Wi-Fi?"
**Expected:** Does not activate `dln-network`.

## CONTRACT

### T9: Revision citation
**Context:** The learner updates a model after a counterexample.
**Expected:** Commits the prediction assessment first; non-initial `model_revision` cites its event ID and never treats word count alone as evidence.

### T10: Baseline
**Context:** No current model exists.
**Expected:** May capture an `initial_model: true` revision with no trigger IDs, but does not call it correctness evidence.

### T11: Completion
**Context:** Session ends.
**Expected:** Generated Session Receipt is the sole canonical summary.

### T12: Approved course grounding
**Context:** An approved syllabus planning topic informs a prediction task.
**Expected:** Cites the active decision via `decision_event_id` and backing settled assertion IDs, discloses unresolved claims and supplemental material, never treats source decisions or coverage as prediction evidence, and does not invoke prepare/propose/decide.
