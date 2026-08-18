---
name: dln-compress
description: >
  Internal optional formatter for bounded context derived from local state.json.
  Never user-triggered. It may compact already-projected machine state for a phase
  skill, but cannot read raw dialogue as evidence, write persistence, decide
  mastery/readiness, revise a learner model, or create learner-facing artifacts.
---

# DLN Compress — Internal State Context

This skill is an optional internal formatter. `dln-store context` already returns the authoritative bounded `profile` and `state`; prefer that output directly when it fits.

## Input

Accept only validated `context` output or its `state` object. Never accept raw dialogue, arbitrary notes, generated dashboard/receipt text, or legacy KS Markdown as evidence input.

## Output

Return a compact machine-readable object with only fields needed for immediate routing:

```json
{
  "domain_id": "...",
  "revision": 0,
  "stage": "acquire|relate|revise",
  "goal": "...",
  "due_review": {"date": null, "action": null},
  "subjects": [
    {
      "id": "...",
      "status": "...",
      "independent": null,
      "supported": null,
      "retrieval": {"status": "not-measured"},
      "transfer": {"count": 0}
    }
  ],
  "current_model": null,
  "calibration": {"status": "not-measured"},
  "grounding": {
    "status": "ungrounded|approval_required|approved|approved_update_pending",
    "active_approval": null,
    "planning_topics": [{"label": "...", "assertion_ids": [], "citable": false}],
    "unresolved_assertions": [],
    "pending_sources": []
  },
  "next_action": null
}
```

Preserve event IDs and measurement status exactly when present. Omit verbose labels only if stable subject IDs remain sufficient for the immediate task.

Carry `state.grounding` through under the same field names the phase skills read. Keep `status`, `active_approval.event_id`, every `planning_topics` entry with its `label`, `assertion_ids`, and `citable` flag, the `assertion_id` of each unresolved assertion, and each pending source's `receipt` path. Drop only the verbose `effective_assertions` citation bodies; a phase skill that needs a quote reloads `dln-store context`.

## Prohibitions

- Do not write `profile.yaml`, `events.jsonl`, `state.json`, dashboard, or receipts.
- Do not create events or profile patches.
- Do not infer or decide proficiency, mastery, stage transitions, readiness, retrieval, transfer, or calibration.
- Do not turn delivered content, dialogue, plans, summaries, or self-ratings into evidence.
- Do not alter, summarize, or compress the learner's pedagogical model. Learner model revision/compression belongs only to `dln-network` and requires cited prediction events.
- Do not produce a learner-facing summary or artifact.
- Do not hide independent/supported separation or convert `not-measured` into a qualitative claim.
- Do not drop, rename, or flatten `state.grounding`, and do not separate a planning topic from its backing `assertion_ids` or its `citable` flag.
- Do not present legacy ungrounded topics as citable, resolve a deferred assertion, or promote a pending-source assertion into the active approval.

If required state is missing, return an explicit missing-field diagnostic and ask the caller to reload `dln-store context`. Never fill gaps from conversation memory.
