# Session Receipt Format (Active)

A completed session has exactly one canonical learner-facing artifact:

```text
<root>/domains/<domain-id>/sessions/<session-id>.md
```

The local store generates this receipt only after accepting a terminal `session_completed` event. The tutor never hand-writes, edits, or substitutes for it.

## Required sections

Receipt schema version 1 contains:

1. **Independent Evidence** — independently attempted assessments, with subject, operation, outcome, and novelty.
2. **Supported Performance** — prompted/worked assessments, with assistance shown explicitly.
3. **Prediction Error and Model Revision** — recorded prediction outcomes and any cited learner model revisions.
4. **Delayed Retrieval** — only linked retrieval attempts with a positive observed delay, marked `(supported)` when the attempt was not independent; otherwise says it was not due or not measured.
5. **Calibration** — only pre-answer confidence paired with a numeric result; otherwise says it was not measured.
6. **Course Grounding** — the cited decision (or legacy approval), its source version and SHA-256, and each cited settled assertion with its media-neutral `unit_id`, character span, and quote; learner corrections retain the target document citation. If no settled assertion was cited, says so explicitly.
7. **Next Action and Review** — the exact next action and nullable review date committed in `session_completed`.

The receipt is generated from `session_completed.evidence_event_ids` plus the optional `grounding` references pinned on `assessment` and `session_completed`, which cite the active `decision_event_id` (or a legacy `approval_event_id`). Grounding is provenance, never evidence. Before closing the session, ensure the evidence list includes every same-session assessment, model revision, and stage transition the learner should see. It must not cite plan text, dialogue, profile patches, or events from another session.

## Completion sequence

1. Finish the final assessment and feedback.
2. Choose `next_action` from actual evidence gaps and `next_review_date` from the review plan. If no review is scheduled, use `null`.
3. Commit remaining evidence events followed by `session_completed` in one request. The completion event may cite earlier events in that same request because events are reduced in order.
4. Confirm the commit succeeded.
5. Open/read the generated receipt path from `state.completed_sessions` or `sessions/<session-id>.md`.
6. Present the receipt, optionally preceded by one sentence naming its path. Do not add another prose recap that could disagree with it.

If the completion commit fails, no receipt exists. Say the session is not durably closed and follow the persistence recovery protocol; never fabricate a receipt.

## Dashboard boundary

`dashboard.md` is the generated longitudinal view: current stage, subject evidence, retrieval/transfer/calibration status, current model, course grounding status with its active source and Syllabus Intake Receipt link, completed-session links, and next action. It is not a per-session receipt and is not canonical measurement input.

- Use the dashboard to orient the learner across sessions.
- Use the receipt to close and share one completed session.
- Use `state.json`/`context` for machine routing.
- Use `profile.yaml` and `events.jsonl` as the only canonical sources.

Generated Markdown may be opened directly in Obsidian; no Obsidian extension, plugin, or MCP is part of the persistence contract.
