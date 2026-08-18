# Local Session Checkpoint Protocol (Active)

> Historical filename retained for compatibility. This file no longer describes remote synchronization. Authoritative storage rules are in `@local-persistence-protocol.md`.

## Learner-generated checkpoint

After a meaningful boundary, ask the learner for one compact restatement before giving corrective feedback. The checkpoint is instructional context only until evaluated against a stated task/rubric.

A checkpoint may become an `assessment` event only when:

- the task and rubric were fixed before the response;
- assistance and novelty are recorded honestly;
- the response produced an observable `pass`, `partial`, or `fail` outcome;
- the event follows `@evidence-protocol.md`.

Never persist the learner's prose as an unstructured progress note or treat note length as evidence.

## Plan adjustment

Plans are conversational and disposable. Adjust the live plan when evidence shows prerequisite gaps, excessive support, a failed novel transfer, or a prediction error. Do not persist plan prose. Persist only the assessment/model event that justified the change.

Tell the learner what changed and why, for example:

> "That independent comparison exposed a gap in X, so we'll test one simpler contrast before returning to Y."

## Calibration feedback

When `confidence_before`, `score`, and `max_score` were recorded, discuss the observed gap after feedback. Do not persist a qualitative label such as "overconfident" as if it were a measurement. If confidence was collected after answering, it is reflection only and must not enter the calibration fields.

## Persistence checkpoint

At each selected boundary:

1. Build the event(s) from the observed outcome.
2. Commit through the local CLI using the retained revision.
3. Update the retained revision only from parsed success output.
4. Continue teaching only after distinguishing `committed`, `noop`, and failure.

Follow stale-revision and recovery handling exactly as defined in `@local-persistence-protocol.md`. On any unresolved failure, retain pending structured events in the current conversation, say they are unsaved, and do not create prose-as-state.

## Session end

Commit remaining evidence and one terminal `session_completed` event atomically. Present the generated receipt defined by `@session-receipt-format.md`; it replaces hand-written session logs and summaries.
