---
name: dln-linear
description: >
  Use when the dln orchestrator routes store stage `relate`, or when a learner
  explicitly requests the historical Linear phase, shared-pattern discovery, or
  cross-structure comparison within an established domain. The user-facing
  operation is Relate/Abstract and records structured comparison, abstraction,
  novel-transfer, retrieval, and calibration evidence in the local event store.
---

# DLN Linear — Relate/Abstract

`dln-linear` is the backward-compatible implementation ID for the **Relate/Abstract** stage. Dunk is a DLN-inspired tutoring adaptation, not a validation of a computational DLN model.

## Read first

- `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/local-store-schema.md`
- `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/local-persistence-protocol.md`
- `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/syllabus-grounding-protocol.md`
- `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/evidence-protocol.md`
- `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/session-receipt-format.md`
- `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/sync-protocol.md`
- `@references/linear-protocol.md`

Do not use legacy KS merge references in an active session.

## Input contract

Receive domain/domain ID, bounded `context` output, retained revision, stable session ID, command/exam intent, and review-due flag. Require `state.stage == "relate"`. Do not infer the stage or prior evidence from conversation, dashboard, or receipts.

Use the goal, projected subjects/current model, and bounded `state.grounding` only to select tasks. When grounding status is `approved` or `approved_update_pending`, choose course work only from the prior active approval's `planning_topics`, exclude pending-source assertions, retain the backing approved assertion IDs, and add the active `approval_event_id` plus used settled `assertion_ids` to relevant `assessment` and `session_completed` events. Describe deferred Week 7–13 alignment as unresolved and label textbook/web/model additions as supplemental. Legacy `profile.syllabus` is ungrounded and non-citable. A polished explanation or syllabus assertion is not learning evidence unless learner performance is recorded by an assessment event.

## Teaching stance

Balance elicitation and targeted teaching. The learner should compare known subjects, articulate shared structure, test it on variants, and abstract only after the comparison. Do not state the target abstraction before an assessment intended to measure discovery or transfer.

## Session loop

1. **Orient from state.** Name current independent evidence, supported-only work, due retrieval, and the structural target. Keep the plan in conversation.
2. **Retrieve when due.** Run an independent linked retrieval before cues or comparisons.
3. **Surface structures.** Choose two or more measured subjects/chains with a meaningful structural relation.
4. **Relate.** Ask the learner to compare causal roles, constraints, invariants, and failure points.
5. **Abstract.** Ask for the minimal domain-independent principle and its boundary conditions.
6. **Test transfer.** Present an unsignaled new case. Use `novelty: novel` only if the structure and applicable method are genuinely not disclosed.
7. **Score a predeclared rubric.** Classify assistance and collect confidence before the response only when paired with a numeric score.
8. **Commit structured evidence.** Use only `operation: relate` or `operation: abstract` for assessments in this stage. The parent performs revision-checked CLI calls.
9. **Adjust.** On failure, diagnose whether the gap is foundational, relational, or abstraction precision; then teach and run a separate supported check.

Content delivery, tutor-created diagrams, and conversational hypotheses are not evidence. Commit only completed assessments, valid transitions, approved profile patches, and terminal completion.

## Assessment construction

Every event uses stable IDs/bodies across retries and includes honest subject, task, context, rubric, novelty, assistance, mode, and outcome fields.

Typical tasks:

- explain how two measured chains share a mechanism (`relate`);
- identify a decisive structural difference (`relate`);
- state an invariant and boundary conditions without domain-specific surface terms (`abstract`);
- choose or apply an abstraction on an unsignaled novel case (`abstract`, `novelty: novel`);
- delayed retrieval of a prior structural subject linked to its assessment.

If the tutor names the factor/principle or narrows the relevant chains, the result is supported. Supported success guides teaching but cannot justify the gate.

## Foundational gaps

If a relation task fails because a prerequisite subject is independently partial/fail or supported-only, do not manufacture an abstraction. Either:

- remediate inside the session and later schedule an independent check; or
- propose a cited evidence-based transition back to `acquire` when the gap is broad enough to require Acquire/Discriminate work.

A newly added syllabus topic is planning context, not automatic demotion.

## Gate to Predict/Revise/Compress

Propose `relate → revise` only after already-committed independent evidence demonstrates:

- accurate relation of multiple known structures;
- a precise abstraction with boundary conditions;
- at least one genuinely novel, unsignaled transfer task;
- explanation of why the abstraction predicts the observed result.

Use `relate-to-revise-v1` from `@references/linear-protocol.md`. Commit the assessment evidence first, then a `stage_transition` with cited IDs, `from: relate`, and `to: revise`. Reload context after success.

Do not gate on coverage, number of factors, session count, supported comparisons, confidence, or imported claims.

## Exam-aware behavior

Use exam configuration to choose task form and priority, not to alter the rubric or evidence mode. Repeated past-paper questions are `repeat`/`variant` unless they demand an unsignaled new structural mapping. Timing is recorded only when explicitly measured.

## Completion

Atomically commit remaining evidence followed by `session_completed`, citing all learner-visible same-session assessment/transition IDs. On success, present the generated Session Receipt as the only summary. On failure, report that the session is not durably closed and follow the persistence protocol without prose fallback.
