# Syllabus Grounding Protocol (Active)

This protocol applies only to the digest-bound `st5201x-2026-v1` adapter. It is not a general PDF, OCR, or document-extraction interface.

## Registration boundary

A supplied syllabus can become authoritative only when the runtime exposes a readable file containing the actual bytes. A transient attachment preview, pasted summary, model recollection, web result, or inaccessible path is not a byte channel. If the file is unavailable or unreadable, stop registration, state the failure truthfully, and offer either a retry or a separate ungrounded generated curriculum.

The adapter accepts only:

- adapter `st5201x-2026-v1`;
- media type `application/pdf`;
- byte size `45185`;
- SHA-256 `53909df562e2658ab3e1327eb8c33120fa12b37489178dc87bb4d632e4f15376`.

Any other adapter, media type, size, or digest is rejected without a canonical write. Raw PDF bytes are not retained. Re-verification requires the learner to resupply the document; canonical extracted page text, located assertions, provenance, and digest remain replayable.

## Intake and approval lifecycle

1. Initialize or load the domain and retain the current revision.
2. Run `ingest-syllabus` with the readable document path, original filename, media type, adapter, timestamp, and expected revision.
3. On success, reload `context` and present the generated `syllabus/<source-version-id>.md` Syllabus Intake Receipt. Status is `approval_required`; intake alone is not authority.
4. Collect a complete learner decision over every source assertion: accepted, corrected, or deferred. Unresolved assertions such as `st5201x.schedule.weeks_7_13_alignment` cannot be represented as settled facts.
5. Run `approve-syllabus` with a private JSON request and the retained revision, then reload `context` before selecting or teaching course work.
6. A later complete approval must set `supersedes_approval_event_id`. A later source must declare `supersedes_source_version_id`; an unapproved newer source does not replace the active approved source.

Both commands use the stale-revision, recovery, and idempotent-replay rules in `local-persistence-protocol.md`. Never inject `syllabus_source_ingested` or `syllabus_approval_recorded` through generic `commit`.

## Canonical identity and history

A source version is identified by `sha256-<full-digest>`. The source ingestion event preserves source identity, digest, original filename, extraction tool/version, canonical page text, located assertions, and assertion-set digest. A syllabus approval event is an immutable complete snapshot containing:

- `source_version_id` and `source_assertion_set_sha256`;
- learner actor;
- `accepted_assertion_ids`;
- `deferred_assertion_ids`;
- learner `corrections` with stable correction assertion IDs and target document assertion IDs;
- nullable `supersedes_approval_event_id`.

A correction changes the effective tutoring value but never erases the document-derived value or citation. History remains append-only.

## Bounded grounding bundle

`context` exposes the bounded bundle at `state.grounding`:

- `status`: `ungrounded`, `approval_required`, `approved`, or `approved_update_pending`;
- active source and approval summaries;
- compact effective assertions and citations;
- unresolved assertions;
- pending source summaries and receipt paths;
- `planning_topics`, each with backing assertion IDs and `citable`.

Do not pass raw PDF bytes, canonical page text, full event history, or prior chat to a phase skill. When approved grounding exists, select course tasks from `state.grounding.planning_topics`; `state.syllabus` is the derived flat projection. When only `profile.syllabus` exists, it is a legacy ungrounded planning fallback and its topics are not citable; once an approval is active the store rejects a `profile_patch.syllabus` edit.

Course-specific logistics, policies, dates, requirements, and topic selection must use effective approved assertions. Deferred/unresolved assertions must be described as unresolved. Textbook, web, or model additions that are absent from the approved assertions must be labeled supplemental rather than syllabus-derived.

## Learning-event references

An `assessment` or `session_completed` may include:

```json
{"grounding":{"approval_event_id":"approval-id","assertion_ids":["stable-assertion-id"]}}
```

Use the approval active before the learning event and include every approved effective assertion actually used for course-specific task selection or teaching. IDs must be non-empty, unique, settled, and valid under that approval. Historical Session Receipts resolve through the cited approval, so later corrections do not rewrite earlier provenance.

Grounding is source metadata only. Syllabus assertions, coverage, approval, citations, corrections, and completion of syllabus topics are never assessments, mastery evidence, delayed retrieval, transfer, calibration, or stage-gate support. Only learner performance recorded under `evidence-protocol.md` affects learning state.
