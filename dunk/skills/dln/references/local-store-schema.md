# Local Store Schema (Active)

This is the authoritative serialized contract for Dunk's local-first store. The implementation is `scripts/dln-store.py` and `scripts/dln_store/`.

## Canonical layout

The vault root resolves in this order: `--root`, `DLN_VAULT_ROOT`, then `${CLAUDE_PLUGIN_DATA}/dln-vault`. If none is available, stop and ask the user to set `DLN_VAULT_ROOT`; never invent a location.

```text
<root>/
  domains/<domain-id>/
    profile.yaml
    events.jsonl
    state.json
    dashboard.md
    sessions/<session-id>.md
```

`profile.yaml` and `events.jsonl` are authoritative. `state.json`, `dashboard.md`, and receipts are generated projections. Do not treat edits to generated files as learning evidence.

## Ownership

`profile.yaml` uses the JSON-compatible YAML subset accepted by the stdlib parser. The learner chooses `domain` at initialization; it is then immutable because it determines the directory identity. User-editable fields are `goal`, `syllabus`, `annotations`, `review_preferences`, and `exam`. Store-owned fields are `schema_version`, `domain_id`, and `revision`; never patch them. To rename a domain, initialize a new domain instead of renaming its directory or profile.

`events.jsonl` is append-only. Each line is one canonical JSON event. Event IDs and session IDs are portable identifiers: letters, digits, dot, underscore, or hyphen, at most 128 characters. Timestamps are UTC RFC 3339 strings ending in `Z`.

## Commit request

```json
{"events":[],"profile_patch":{}}
```

A request must contain at least one of `events` or `profile_patch`. The current profile revision is passed separately as CLI `--expected-revision`; the store rejects a stale value without changing canonical files. A successful non-noop commit increments the profile revision once, regardless of event count. Exact duplicate event replay is a no-op; the same event ID with different content is an integrity error.

## Event variants

Every event requires `schema_version: 1`, `event_id`, `session_id`, `occurred_at`, and `kind`.

### `assessment`

Required fields:

- `operation`: `acquire | discriminate | relate | abstract | predict`
- `task_id`, `context_id`, `rubric_id`
- `subject`: `{id, label, type}`; a stable ID may not change label or type
- `novelty`: `repeat | variant | novel`
- `evidence_mode`: `independent | supported`
- `outcome`: `pass | partial | fail`
- `assistance`: `{level: none | prompt | worked, hint_count}`

Optional paired fields: `score` and `max_score`; `confidence_before` requires both and ranges from 0 to 1. `retrieval` is `{prior_event_id, scheduled_date, observed_delay_days}` and must cite an assessment of the same subject in the current reset generation. The attempt must occur on a later UTC date, `observed_delay_days` must equal the UTC calendar-date difference, and the scheduled date must fall after the prior assessment and no later than the retrieval attempt. `response_time_ms` is allowed only when explicitly timed. Independent evidence requires `assistance: {"level":"none","hint_count":0}`.

### `model_revision`

Required: `triggering_prediction_event_ids`, `model`, `decision`, and `rationale`. `decision` is `exploit | revise | expand | fallback-independent`. A non-initial revision cites at least one prior `predict` assessment. An initial captured model uses `initial_model: true` and no triggers. Optional fields are `prior_model_revision_event_id`, `word_count_before`, and `word_count_after`.

### `stage_transition`

Required: `from`, `to`, `rubric_id`, non-empty `assessment_event_ids`, and `decision`. Stages are `acquire | relate | revise`. Every cited assessment must already exist, be independent, and belong to the current reset generation. Acquire → Relate requires passing `acquire`/`discriminate` evidence. Relate → Revise requires passing `relate`/`abstract` evidence including a novel task. Revise → Acquire/Relate requires partial or failed `predict` evidence.

### `session_completed`

Required: `next_action`, nullable `next_review_date`, `evidence_event_ids`, and `receipt_schema_version: 1`. Cited evidence must already occur in the same session and be an assessment, model revision, or stage transition. This event is terminal: no later event may use that session ID.

### Other events

- `domain_reset`: optional `reason`; starts a fresh projected generation without deleting history.
- `exam_cycle_closed`: `archived_exam` plus optional `self_reported_outcome`.
- `legacy_snapshot_imported`: import-only, source-hashed, and always `evidence_eligible: false`.

## Projected state

`state.json` exposes the current `revision`, `stage`, `subjects`, `current_model`, calibration aggregate, completed receipt index, review date/action, profile fields, source hashes, reset generation, exam archive, and legacy claims.

Supported performance never overwrites the latest independent result. Transfer increments only for `novel` assessments. Delayed retrieval becomes measured only after a timestamp-consistent, positive calendar-day delay in the current generation, and each subject's `retrieval` block records `count`, the `latest` attempt including its `evidence_mode`, and `satisfied_by`. Only an independent passing retrieval sets `satisfied_by`; supported or non-passing attempts stay measured evidence and leave a passing subject at `needs-retrieval`. Calibration includes only paired pre-answer confidence and score. Imported claims never satisfy gates.
