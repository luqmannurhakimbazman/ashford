---
name: dln-dot
description: >
  Use when the dln orchestrator routes store stage `acquire`, or when a learner
  explicitly asks to start a domain from zero, learn foundations, or run the
  historical Dot phase. The user-facing operation is Acquire/Discriminate:
  teach foundations, then collect structured recall, distinction, application,
  and delayed-retrieval evidence through the local event store.
---

# DLN Dot — Acquire/Discriminate

`dln-dot` is the backward-compatible implementation ID for the **Acquire/Discriminate** stage. It is part of a DLN-inspired tutoring adaptation, not a claim to implement or validate a computational DLN model.

## Read first

- `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/local-store-schema.md`
- `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/local-persistence-protocol.md`
- `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/syllabus-grounding-protocol.md`
- `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/evidence-protocol.md`
- `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/session-receipt-format.md`
- `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/sync-protocol.md`
- `@references/dot-protocol.md`

Do not use legacy KS merge references in an active session.

## Input contract

Receive from the orchestrator:

- domain and `domain_id`;
- bounded output of `dln-store context` (`profile` and `state`);
- retained expected revision;
- one stable session ID;
- command/exam intent and whether delayed review is due.

Require `state.stage == "acquire"`. If context is missing or stale, ask the parent to reload it. Do not reconstruct state from prior dialogue, receipts, or dashboard Markdown.

## Teaching stance

Teach more than you ask when the foundation is new, but separate instruction from measurement. Growth-oriented feedback must name effort or strategy without turning praise, engagement, or note production into evidence.

Use syllabus and state only for planning:

- prioritize subjects with no independent evidence or independent partial/fail;
- distinguish supported-only performance from independent performance;
- run a due delayed retrieval before new teaching;
- when `state.grounding.status` is `approved` or `approved_update_pending`, select course work only from the prior active decision's `state.grounding.planning_topics`, retain its backing assertion IDs, and exclude every pending-source assertion;
- describe every deferred or ambiguous document claim as unresolved;
- label textbook, web, or model additions absent from the approved bundle as supplemental;
- use legacy `profile.syllabus` only as ungrounded, non-citable coverage planning, never as proof of learning.

## Session loop

1. **Orient.** State the goal, current measured evidence, and one to three targets. Plans stay in conversation and are not persisted.
2. **Retrieve when due.** Ask an unprompted same-subject task before cues. Link a valid positive-delay retrieval event to the prior assessment.
3. **Acquire.** Explain one small concept or demonstrate one procedure. Delivery creates no event.
4. **Discriminate.** Contrast the concept with a plausible near-miss or misconception.
5. **Assess.** Predeclare the task/rubric, collect confidence beforehand only when numeric scoring will follow, then wait for the response.
6. **Classify assistance honestly.** No cues is independent; any narrowing prompt, hint, step, or worked material is supported.
7. **Commit the boundary.** Construct standardized `assessment` events using only `operation: acquire` or `operation: discriminate`, then ask the parent to commit them with the retained revision.
8. **Update revision.** Continue only from parsed `committed`/`noop` output. On stale or recovery errors, follow the local persistence protocol exactly.
9. **Feedback/remediation.** Give targeted correction. A supported re-check is a new event; never overwrite the independent attempt.

Do not commit at content-delivery boundaries. Commit only observed assessments, stage transitions, model events (normally none in this stage), profile patches approved by the learner, and terminal completion.

## Assessment construction

Each assessment includes:

- a stable event ID retained across retry;
- stable task/context/rubric IDs;
- subject `{id,label,type}`;
- `operation: acquire | discriminate`;
- honest `novelty`, `evidence_mode`, `outcome`, and `assistance`;
- score/confidence/retrieval/response time only when actually measured;
- when approved syllabus assertions informed the task or teaching, `grounding` with the active `decision_event_id` and used settled `assertion_ids`; include the same grounding on `session_completed`.

Typical tasks:

- unaided free recall or explanation (`acquire`);
- choose and justify between close alternatives (`discriminate`);
- apply the concept to a variant without being told the method (`discriminate`);
- delayed same-subject retrieval linked to an earlier assessment.

An explain-back immediately after seeing the same wording is usually `repeat`, not transfer. A prompted correction is supported even if correct.

## Gate to Relate/Abstract

Propose `acquire → relate` only when already-committed independent evidence demonstrates:

- accurate recall of the intended foundation;
- discrimination from important near-misses;
- application on at least a variant task;
- enough subjects for the learner's goal, not merely syllabus checkboxes.

Use the rubric in `@references/dot-protocol.md`. Commit a `stage_transition` with `from: acquire`, `to: relate`, the stable gate rubric ID, and cited independent assessment IDs. Reload context after success. Supported work, self-confidence, session count, content coverage, and legacy claims cannot satisfy the gate.

If the gate fails, keep stage `acquire`, explain the exact evidence gap, and choose the next task from that gap.

## Exam-aware behavior

Exam configuration changes task selection and surface format only. Match the declared exam format when useful, mark a repeated past question honestly, and never weaken independent-evidence requirements because time is short. Record response time only with an explicit reliable timer.

## Intake boundary

This phase consumes only active decided grounding. It must not prepare, propose, decide, fetch, extract, or promote supplements/pending authoritative sources. Source decisions are separate from learner evidence and mastery.

## Completion

Before ending:

1. Gather every same-session assessment/transition ID that should appear in the learner record.
2. Choose the next action from observed gaps and a review date from the review plan; use `null` when none is scheduled.
3. Ask the parent to atomically commit remaining events followed by `session_completed`.
4. After success, present the generated Session Receipt. Do not author a second summary.

If completion fails, say the session is not durably closed. Keep the structured request for recovery and do not fabricate a receipt or fall back to prose-as-state.
