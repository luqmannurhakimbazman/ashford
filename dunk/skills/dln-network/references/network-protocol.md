# Predict/Revise/Compress Protocol

Detailed protocol for the backward-compatible `dln-network` skill ID. Shared local schema, persistence, evidence, and Session Receipt rules are authoritative.

## Objective

Turn an explicit learner model into pre-outcome predictions, expose error or survival, and record learner-authored revisions that cite those predictions. Compression is valuable only when predictive coverage is retained or improved.

## Revision cycle

### 1. Baseline

If no projected current model exists, ask the learner to state one before correction. Persist it as an initial `model_revision` with:

- `initial_model: true`;
- `triggering_prediction_event_ids: []`;
- learner model, decision, and rationale.

This is a baseline, not assessment evidence.

### 2. Predict

Present a fully specified case and ask for the model's prediction before revealing the result. Use `predict-model-v1`:

- `pass`: predicted outcome and material mechanism/boundary are correct;
- `partial`: direction is right but mechanism/boundary is materially incomplete;
- `fail`: wrong direction, wrong applicable principle, or no testable prediction.

The event uses `operation: predict`. If the tutor reveals the relevant factor, narrows the case, or gives a hint, mark it supported. Do not alter the prediction text/body after seeing the result.

### 3. Diagnose error

Ask:

- Which assumption produced the mismatch?
- Is the boundary wrong, missing, or overgeneralized?
- Does this require revising, expanding, or separating cases?
- What future observation would falsify the new claim?

This dialogue is preparation, not another evidence event.

### 4. Revise

The learner states the updated model and rationale. Persist a non-initial `model_revision` only after its triggering `predict` assessment exists. Cite all relevant prediction IDs and the prior model event when available.

Decision meanings:

- `exploit`: keep the model after a survived test;
- `revise`: change a claim or boundary;
- `expand`: add necessary scope;
- `fallback-independent`: stop forcing cases into one model.

Optional word counts measure length only. They do not prove quality or readiness.

### 5. Retest

Use a new case that probes the revision. Prefer a variant that isolates the changed boundary, then a genuinely novel unsignaled transfer case. Repeat until the learner chooses a defensible model or the session should end.

## Stress-test families

- **Boundary case:** near the point where the model changes prediction.
- **Counterexample:** satisfies apparent premises but contradicts the claim.
- **Assumption removal:** removes one hidden condition.
- **Interaction:** two principles apply and compete.
- **Adjacent-domain transfer:** surface differs while deep structure may persist.

Do not tell the learner which family/factor applies before an independent prediction.

## Retrieval and calibration

For a due review, elicit the model/application before showing `state.current_model`. A linked assessment with positive delay is measured retrieval. Confidence must be collected before the answer and paired with a numeric score.

## Compression rubric: `model-compression-v1`

Evaluate only after retesting:

- **Predictive coverage:** handles the tested cases and names boundaries.
- **Parsimony:** removes redundancy without exception stacking.
- **Falsifiability:** makes testable predictions.
- **Transfer:** applies to a genuinely new case when claimed.

A shorter model that loses coverage is not improved compression. A longer model may be a necessary expansion; record it honestly.

## Earlier-stage transition

When independent predictions repeatedly fail for the same missing structural relation or foundation, a `stage_transition` may return from `revise` to `relate` or `acquire`. It must cite the committed independent prediction assessment IDs and use a stable rubric such as `revise-to-relate-gap-v1` or `revise-to-acquire-gap-v1`. Reload context after committing it.

## Visuals

Model maps and before/after diagrams are teaching aids. They may help the learner revise but never count as evidence. The learner's pre-outcome prediction is the measurement event.

## Session end

Commit remaining prediction/model/transition events followed by terminal `session_completed`. The generated Session Receipt shows predictions, revisions, independent/supported separation, retrieval/calibration, and next action. Do not replace it with a handcrafted distributed-revision recap.
