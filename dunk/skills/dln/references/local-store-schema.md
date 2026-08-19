# Local Store Schema

Each external domain directory contains canonical `profile.yaml`, append-only `events.jsonl`, `sources/sha256/<digest>`, and `prepared/sha256/<digest>.json`. Derived `state.json`, `dashboard.md`, `sessions/<session-id>.md`, and syllabus receipts are deterministic and replaceable.

The profile/event pair and referenced CAS must validate independently before reduction. Every mutation carries an expected revision; the store uses one lock and journaled transaction. `validate` reports missing, corrupt, and orphaned canonical content. `rebuild` never downloads, extracts, or backfills content.

## Learning evidence

Ordinary kinds remain `assessment`, `model_revision`, `stage_transition`, `session_completed`, `domain_reset`, `exam_cycle_closed`, and `legacy_snapshot_imported`. supported and independent evidence remain distinct. A source proposal/decision is never an assessment and cannot change mastery.

## Reserved syllabus events

Generic `commit` rejects all reserved kinds. Dedicated operations create:

- `syllabus_source_prepared`: role, source byte identity/CAS, prepared-document identity/CAS, bounded acquisition/extraction provenance, and authoritative predecessor.
- `syllabus_assertions_proposed`: immutable bounded proposal set pinned to prepared content, with typed values, semantic roles, and exact `{unit_id,start_char,end_char,quote}` locators. Producer metadata is `external_unverified`.
- `syllabus_decision_recorded`: learner-authored complete accept/correct/defer/reject partition pinned to the proposal digest. Ambiguous proposals cannot be accepted.

One non-forking authoritative source/decision lineage exists per domain. Later authority explicitly supersedes the latest source and decision. `approved_update_pending` preserves the prior active decision. Supplements are labeled non-authoritative and cannot drive active grounding, planning topics, eligible citations, or mastery.

New grounding references use `decision_event_id` and non-empty `assertion_ids`. `approval_event_id` is accepted only for legacy replay; legacy `syllabus_source_ingested` and `syllabus_approval_recorded` events remain text-only and require no CAS backfill.

## Prepared document

Prepared schema version 1 permits `application/pdf` page units or one `text/html` document unit. Text is normalized NFC with LF endings, exact SHA-256 per unit, no more than 500 units, and no more than 8 MiB total UTF-8 text. Original source is capped at 16 MiB. Locators must equal exact string slices.

## State grounding

`state.grounding` exposes status, active source/proposal/decision, pending authoritative update, visible supplements, planning topics, and eligible citations. Status values include `ungrounded`, `proposal_required`, `decision_required`, `approved`, and `approved_update_pending`. Existing historical projections may surface legacy compatibility labels, but new writes use the three event kinds above.

Canonical content belongs outside `${CLAUDE_PLUGIN_ROOT}` and must be backed up privately with the ledger/profile pair. Hashes prove consistency, not resistance to a malicious filesystem owner.
