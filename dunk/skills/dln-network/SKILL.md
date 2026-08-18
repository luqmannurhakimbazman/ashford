---
name: dln-network
description: >
  Use when the dln orchestrator routes store stage `revise`, or when a learner
  explicitly requests the historical Network phase, model stress-testing,
  counterexamples, or learner-model compression after foundations and relations
  exist. The user-facing operation is Predict/Revise/Compress and records
  pre-outcome predictions and cited model revisions in the local event store.
---

# DLN Network — Predict/Revise/Compress

`dln-network` is the backward-compatible implementation ID for the **Predict/Revise/Compress** stage. It is a DLN-inspired tutoring operation, not a claim to implement or validate a computational DLN model.

## Read first

- `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/local-store-schema.md`
- `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/local-persistence-protocol.md`
- `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/syllabus-grounding-protocol.md`
- `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/evidence-protocol.md`
- `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/session-receipt-format.md`
- `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/sync-protocol.md`
- `@references/network-protocol.md`

Do not use legacy KS merge references in an active session.

## Input contract

Receive domain/domain ID, bounded `context` output including `state.grounding`, retained revision, stable session ID, command/exam intent, and review-due flag. Require `state.stage == "revise"`. Use `state.current_model` as the prior model when present; do not reconstruct one from dialogue or generated Markdown. When grounding status is `approved` or `approved_update_pending`, select course tasks only from the prior active approval's `planning_topics`, exclude pending-source assertions, retain backing approved assertion IDs, and add the active `approval_event_id` plus used settled `assertion_ids` to relevant `assessment` and `session_completed` events. Deferred Week 7–13 alignment remains unresolved; non-syllabus textbook/web/model material is supplemental. Legacy `profile.syllabus` is ungrounded and non-citable.

## Teaching stance

Elicit more than you deliver. The learner states a model, predicts before outcomes, sees counterevidence, and decides whether to exploit, revise, expand, or fall back. A model that changes without a cited prior prediction is not a measured revision.

Compression means reducing the learner's model while retaining useful predictive coverage. It is distinct from `dln-compress`, an optional internal context formatter that must never modify a learner model or decide readiness.

## Session loop

1. **Orient.** Name the current model/evidence and one pressure point. Keep plans in conversation.
2. **Retrieve when due.** Ask the learner to reconstruct or apply the model before displaying it; record linked delayed retrieval only when schema-valid.
3. **Capture baseline.** If no current model exists, ask for one before correction and commit a `model_revision` with `initial_model: true`, empty trigger IDs, and a baseline rationale. This captures a model; it is not correctness evidence.
4. **Pre-outcome prediction.** Present a task/counterexample setup and ask what the model predicts before revealing the outcome. Collect confidence beforehand only if scoring numerically.
5. **Assess.** Record an `assessment` with `operation: predict`, honest novelty, assistance, and outcome.
6. **Expose error.** Compare prediction with the result and ask what assumption or boundary failed.
7. **Revise/compress.** Ask the learner to state the new model and decision. A non-initial `model_revision` must cite the committed `predict` assessment event ID(s).
8. **Retest.** Use a new variant/novel prediction to test the revision; do not claim success from word-count reduction alone.
9. **Commit boundaries.** The parent performs revision-checked local CLI commits and retains only returned revisions.

Tutor explanations, stress-test descriptions, diagrams, word counts, syllabus assertions, approval, citations, and coverage are not evidence. Only observed assessments and schema-valid learner model revisions enter the learning record.

## Prediction construction

Use stable IDs/bodies across retries. A prediction assessment includes:

- `operation: predict`;
- a predeclared task/context/rubric;
- stable subject representing the model claim or target;
- novelty (`repeat`, `variant`, or genuinely unsignaled `novel`);
- honest independent/supported assistance;
- pass/partial/fail outcome after the result is known;
- optional numeric score/confidence and explicit timing only when measured.

Do not rewrite the prediction event after seeing the outcome. The outcome evaluates the original prediction against the predeclared rubric.

## Model revision construction

A revision includes:

- `triggering_prediction_event_ids` citing prior committed `predict` assessments;
- optional `prior_model_revision_event_id` citing the earlier model;
- learner's current `model` text;
- `decision: exploit | revise | expand | fallback-independent`;
- explicit `rationale` naming the prediction error/evidence;
- optional measured word counts.

`exploit` means retaining a model because the prediction survived this test; it still cites the prediction. `revise` changes a claim/boundary; `expand` adds necessary scope; `fallback-independent` abandons the unified model for cases that need separate handling.

An initial baseline alone cannot establish readiness, transfer, retrieval, or calibration.

## Stress-test selection

Prefer tests that:

- target a stated model boundary or hidden assumption;
- contrast near cases that the model should distinguish;
- require an unsignaled prediction in a new setting;
- retest a prior revision without repeating the same surface task.

Do not reveal the intended failure mode before the prediction. If guidance narrows it, mark the assessment supported.

## Regression to earlier work

There is no automatic promotion beyond `revise`. If committed independent prediction evidence shows that the failure is a missing relation or foundation, a stage transition may return to `relate` or `acquire` with cited assessment IDs and a stable rubric. Reload context after the transition. A model becoming longer, learner frustration, or one counterexample alone does not justify regression.

## Exam-aware behavior

Use the exam format and priority to select prediction tasks. Do not call a familiar past-paper item novel or infer response time from tool latency. Time pressure never changes assistance classification or the need to cite prediction evidence for revisions.

## Completion

Atomically commit remaining predictions/revisions/transitions followed by one `session_completed` event citing all learner-visible same-session evidence. Present only the generated Session Receipt. If persistence fails, report that the session is not durably closed and follow recovery rules without prose fallback.
