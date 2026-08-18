# Structured Evidence Protocol (Active)

Dunk is a **DLN-inspired tutoring adaptation**. The stage names and operations below organize tutoring behavior; they do not claim to implement or validate Alia Wu's computational model.

## What counts

An `assessment` event records a learner attempt against a predeclared task and rubric. The prompt, expected criteria, assistance, novelty, and confidence (when used) must be fixed before the learner sees outcome feedback.

Evidence is not:

- content delivered by the tutor;
- dialogue length, note production, enthusiasm, or agreement;
- a plan, explanation, or worked example copied back by the learner;
- tool/model latency;
- a confidence rating collected after the answer;
- an imported legacy mastery label;
- a syllabus assertion, topic, schedule, policy, approval, correction, citation, or coverage/completion claim.

Syllabus grounding establishes what the course source says, not what the learner knows. A `grounding` reference on an assessment or completion event records provenance only and must not change subjects, stage gates, retrieval, transfer, or calibration.

After teaching, ask a distinct check before recording evidence. If the tutor hints, prompts, supplies steps, or exposes the answer, record `evidence_mode: supported` with accurate assistance. Never upgrade supported work to independent.

## Operations and stages

| Store stage | User-facing stage | Operations | Measurement focus |
|---|---|---|---|
| `acquire` | Acquire/Discriminate | `acquire`, `discriminate` | recall, recognition boundaries, application, misconception separation |
| `relate` | Relate/Abstract | `relate`, `abstract` | structural comparison, causal linkage, abstraction, novel transfer |
| `revise` | Predict/Revise/Compress | `predict` plus `model_revision` | pre-outcome prediction, prediction error, explicit model change |

`dln-dot`, `dln-linear`, and `dln-network` remain implementation skill IDs only.

## Independent versus supported

Independent evidence requires no hints and the exact assistance object:

```json
{"level":"none","hint_count":0}
```

Any prompt that narrows the answer, reminder of a key step, partial solution, or worked demonstration makes the attempt supported. Supported attempts are useful for instruction but cannot justify a stage transition or overwrite the latest independent outcome in projected state.

## Retrieval and spacing

A retrieval event is an assessment linked to a prior assessment of the same stable subject through:

```json
{"prior_event_id":"...","scheduled_date":"YYYY-MM-DD","observed_delay_days":7}
```

Only a retrieval on a later UTC calendar date in the current reset generation counts. `observed_delay_days` must equal the calendar-date difference from the cited assessment, and `scheduled_date` must be after the prior assessment and no later than the attempt. A scheduled review that was not attempted is not evidence. Only an independent passing retrieval satisfies the delayed-retrieval gate; supported or non-passing retrievals are recorded as measured evidence but never promote a subject's projected status. If no valid delayed retrieval occurred, say **spacing was not measured** rather than inferring retention.

Prefer independent free recall or application before cues. If cues are later supplied, record a separate supported assessment rather than altering the independent attempt.

## Transfer

Transfer is visible only when an assessment sets `novelty: novel`. Use `variant` when surface details change but the same familiar structure is signaled; use `repeat` for a substantially repeated task. A novel task must not reveal which prior method applies.

## Calibration

Collect `confidence_before` between 0 and 1 before the outcome is known. Store it only with numeric `score` and `max_score`. Do not translate post-answer confidence into a pre-answer value. Discuss calibration as the gap between prediction and observed normalized score, not as a personality trait.

## Prediction and revision

A prediction is an independent or supported `assessment` with `operation: predict`, made before the outcome or counterexample is revealed. A later `model_revision` cites the prediction event IDs that triggered the decision and preserves the learner's resulting model, decision, and rationale.

Capturing the first model uses `initial_model: true`; it is a baseline, not proof of correctness. Model compression is pedagogical only when the learner revises a model in the Predict/Revise/Compress stage. Internal context shortening by `dln-compress` is not a learner model revision.

## Stage gates

A transition requires one or more already-committed independent assessments from the current reset generation, cited by `assessment_event_ids`.

- Acquire → Relate: passing `acquire`/`discriminate` evidence across the intended foundation; syllabus coverage alone is insufficient.
- Relate → Revise: passing `relate`/`abstract` evidence including at least one novel comparison or transfer task.
- Revise → Acquire/Relate: partial or failed `predict` evidence showing the model needs foundational repair; explain the decision.

A gate decision must name a versioned/stable `rubric_id`. Never transition based only on supported performance, self-rating, session count, coverage percentage, or imported claims.

## Response time

Omit `response_time_ms` unless the learner's response was explicitly timed with a reliable clock. Never infer it from tool-call duration, streaming delay, or the tutor's perception.
