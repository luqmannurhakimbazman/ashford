# Local Store Schema (Active)

This is the authoritative serialized contract for Dunk's local-first store. The implementation is `scripts/dln-store.py` and `scripts/dln_store/`.

## Canonical layout

The vault root resolves in this order: `--root`, `DLN_VAULT_ROOT`, then `${CLAUDE_PLUGIN_DATA}/dln-vault`. If none is available, stop and ask the user to set `DLN_VAULT_ROOT`; never invent a location.

```text
<root>/
  domains/<domain-id>/
    profile.yaml
    events.jsonl
    sources/sha256/<source-digest>
    prepared/sha256/<prepared-digest>.json
    state.json
    dashboard.md
    syllabus/<source-version-id>.md
    sessions/<session-id>.md
```

Canonical data is `profile.yaml`, append-only `events.jsonl`, retained original source bytes under `sources/sha256/`, and normalized prepared documents under `prepared/sha256/`. Syllabus source, proposal, decision, correction, and supersession history lives append-only in `events.jsonl`. `state.json`, `dashboard.md`, Syllabus Intake Receipts, and Session Receipts are generated projections. Do not treat edits to generated files as learning evidence.

The profile/event pair and referenced content-addressed objects must validate independently before reduction. Every mutation carries an expected revision; the store uses one lock and one journaled transaction. `validate` reports missing, corrupt, and orphaned canonical content. `rebuild` regenerates projections only and never downloads, extracts, or backfills canonical content.

## Ownership

`profile.yaml` uses the JSON-compatible YAML subset accepted by the stdlib parser. The learner chooses `domain` at initialization; it is then immutable because it determines the directory identity. User-editable fields are `goal`, `syllabus`, `annotations`, `review_preferences`, and `exam`. The store rejects a `syllabus` patch while grounding status is `approved` or `approved_update_pending`. Store-owned fields are `schema_version`, `domain_id`, and `revision`; never patch them. To rename a domain, initialize a new domain instead of renaming its directory or profile.

`events.jsonl` is append-only. Each line is one canonical JSON event. Event IDs and session IDs are portable identifiers: letters, digits, dot, underscore, or hyphen, at most 128 characters, starting with a letter or digit. Timestamps are UTC RFC 3339 strings ending in `Z`.

## Commit request

```json
{"events":[],"profile_patch":{}}
```

A request must contain at least one of `events` or `profile_patch`, and no other keys. The current profile revision is passed separately as CLI `--expected-revision`; the store rejects a stale value without changing canonical files. A successful non-noop commit increments the profile revision once, regardless of event count. Exact duplicate event replay is a no-op; the same event ID with different content is an integrity error.

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

Optional paired fields: `score` and `max_score`; `confidence_before` requires both and ranges from 0 to 1. Optional `grounding` carries exactly one authority key plus `assertion_ids`. New `commit` writes must use `{decision_event_id, assertion_ids}`; `approval_event_id` is accepted only when replaying existing legacy ledgers and is rejected in a commit request. The authority must be the decision active before the event, and `assertion_ids` must be non-empty, unique, and settled and effective under it. `retrieval` is `{prior_event_id, scheduled_date, observed_delay_days}` and must cite an assessment of the same subject in the current reset generation. The attempt must occur on a later UTC date, `observed_delay_days` must equal the UTC calendar-date difference, and the scheduled date must fall after the prior assessment and no later than the retrieval attempt. `response_time_ms` is allowed only when explicitly timed. Independent evidence requires `assistance: {"level":"none","hint_count":0}`.

### `model_revision`

Required: `triggering_prediction_event_ids`, `model`, `decision`, and `rationale`. `decision` is `exploit | revise | expand | fallback-independent`. A non-initial revision cites at least one prior `predict` assessment. An initial captured model uses `initial_model: true` and no triggers. Optional fields are `prior_model_revision_event_id`, `word_count_before`, and `word_count_after`.

### `stage_transition`

Required: `from`, `to`, `rubric_id`, non-empty `assessment_event_ids`, and `decision`. Stages are `acquire | relate | revise`. Every cited assessment must already exist, be independent, and belong to the current reset generation. Acquire → Relate requires passing `acquire`/`discriminate` evidence. Relate → Revise requires passing `relate`/`abstract` evidence including a novel task. Revise → Acquire/Relate requires partial or failed `predict` evidence.

### `session_completed`

Required: `next_action`, nullable `next_review_date`, `evidence_event_ids`, and `receipt_schema_version: 1`. Optional `grounding` uses the same active-authority contract as `assessment`. Cited evidence must already occur in the same session and be an assessment, model revision, or stage transition. This event is terminal: no later event may use that session ID.

### Other events

- `domain_reset`: optional `reason`; starts a fresh projected generation without deleting history.
- `exam_cycle_closed`: `archived_exam` plus optional `self_reported_outcome`.
- `legacy_snapshot_imported`: import-only, source-hashed, and always `evidence_eligible: false`.

Supported and independent evidence remain distinct. A syllabus proposal or decision is never an assessment and cannot change mastery.

## Reserved syllabus events

Generic `commit` rejects all reserved kinds. Dedicated operations create:

- `syllabus_source_prepared`: role, source byte identity/CAS, prepared-document identity/CAS, bounded acquisition/extraction provenance, and authoritative predecessor.
- `syllabus_assertions_proposed`: immutable bounded proposal set pinned to prepared content, with typed values, semantic roles, and exact `{unit_id,start_char,end_char,quote}` locators. Producer metadata is `external_unverified`.
- `syllabus_decision_recorded`: learner-authored complete accept/correct/defer/reject partition pinned to the proposal digest. Ambiguous proposals cannot be accepted.

One non-forking authoritative source/decision lineage exists per domain. Later authority explicitly supersedes the latest source and decision. `approved_update_pending` preserves the prior active decision. Supplements are labeled non-authoritative and cannot drive active grounding, planning topics, citable assertions, or mastery.

New grounding references use `decision_event_id` and non-empty `assertion_ids`. `approval_event_id` is accepted only for legacy replay; legacy `syllabus_source_ingested` and `syllabus_approval_recorded` events remain text-only and require no CAS backfill.

## Prepared document

Prepared schema version 1 permits `application/pdf` page units or one `text/html` document unit. Text is normalized NFC with LF endings, exact SHA-256 per unit, no more than 500 units, and no more than 8 MiB total UTF-8 text. Original source is capped at 16 MiB. Locators must equal exact string slices.

## Projected state

`state.json` exposes the current `revision`, `stage`, `subjects`, `current_model`, calibration aggregate, completed receipt index, review date/action, profile fields, source hashes, reset generation, exam archive, legacy claims, derived `syllabus`, and bounded `grounding`.

`state.grounding` is the bounded grounding bundle: `status`, `active_source`, `active_decision`, `pending_sources`, `supplements`, `planning_topics`, `effective_assertions`, `unresolved_assertions`, and `legacy_fallback`. Status values are `ungrounded`, `proposal_required`, `decision_required`, `approved`, and `approved_update_pending`. `active_approval` and `pending_authoritative_sources` are retained compatibility aliases. Decided coverage assertions derive `state.syllabus`; otherwise flat `profile.syllabus` remains a legacy ungrounded, non-citable fallback and every planning topic reports `citable: false`. Full prepared text and authority history stay outside the bounded phase bundle.

Supported performance never overwrites the latest independent result. Transfer increments only for `novel` assessments. Delayed retrieval becomes measured only after a timestamp-consistent, positive calendar-day delay in the current generation, and each subject's `retrieval` block records `count`, the `latest` attempt including its `evidence_mode`, and `satisfied_by`. Only an independent passing retrieval sets `satisfied_by`; supported or non-passing attempts stay measured evidence and leave a passing subject at `needs-retrieval`. Calibration includes only paired pre-answer confidence and score. Imported claims never satisfy gates.

Canonical content belongs outside `${CLAUDE_PLUGIN_ROOT}` and must be backed up privately with the ledger/profile pair. Hashes prove consistency, not resistance to a malicious filesystem owner.
