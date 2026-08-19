# Investigation: Dunk Authoritative, Versioned Syllabus Ingestion

> **Historical investigation record.** Its fixture-bound design was generalized before release and shipped in Dunk 2.2.0, so the proposal-time names below differ from the shipped contract and the `syllabus_source_ingested`/`syllabus_approval_recorded` events it describes are now read-only legacy v1 kinds. `dunk/skills/dln/references/syllabus-grounding-protocol.md` is authoritative.

## Summary
Dunk conflates authoritative course grounding with a mutable planning list. The smallest complete issue-3 slice is a fixture-bound, PDF-aware intake command plus replayable intake/approval events, reducer-derived grounding/topics, deterministic receipts, and stable non-mastery grounding references for later teaching; generalized extraction and raw-PDF storage remain P2.

## Symptoms
- Dunk has a local event store and rendered learner views, but issue #3 requires an authoritative, versioned syllabus ingestion path grounded in the supplied 2026 PDF.
- The smallest coherent change must meet the issue definition of done without optional P2 breadth or production implementation during this investigation.

## Background / Prior Research
- [GitHub issue #3](https://github.com/luqmannurhakimbazman/ashford/issues/3) defines the problem as a missing first-class course-source model, not merely a missing PDF parser. Definition of done: ingest ST5201X through a declared interface; propose cited structured assertions while leaving Weeks 7–13 unresolved; durably version approval/corrections; rebuild the approved grounding from disk; and use it in later teaching without original chat context, while keeping learner mastery separate.
- The issue recommends a P0/P1 split: P0 covers truthful source status, explicit document input, durable source identity, ambiguity disclosure, and an approval receipt; P1 completes the structured course model, downstream citations/routing, versioned updates, and dashboard/session integration. P2 extractor breadth, richer review UI, and extra adversarial fixtures are excluded here.
- The supplied ST5201X syllabus fixture independently verifies as 45,185 bytes, one PDF page, SHA-256 `53909df562e2658ab3e1327eb8c33120fa12b37489178dc87bb4d632e4f15376`. PDFKit extraction recovers the expected ST5201X facts but flattens Week 7–13 labels, topics, `TBA`, and Homework 3–5 into non-alignable runs; exact row mappings cannot be asserted from extracted text.
- Git archaeology: `7ad8347` introduced the authoritative append-only local store; `43b0200` moved syllabus research to a write-free child returning JSON while the parent performs learner-approved revision-checked persistence; `f11caed` established `profile.yaml` and `events.jsonl` as canonical and rendered files as projections. Earlier syllabus commits culminated in extraction into `dunk/` at `f4ee204`.
- Historical implications: extend the existing store and parent approval boundary; preserve the flat topic list only as a derived compatibility projection; use source-digest idempotence/provenance patterns from legacy import; and avoid remote sync, broad teaching redesign, or P2 extractor breadth.

## Investigator Findings

### Verdict

The context-builder proposal is the right minimum architecture **only if “intake + approval events” are self-contained canonical records and downstream use is explicitly referenceable**. A PDF parser alone or a richer `profile_patch.syllabus` does not meet the issue. The smallest coherent slice is:

- a declared PDF-aware intake boundary that computes the digest from the supplied bytes and persists one immutable proposal event;
- a separate learner approval/correction event that is sufficient, together with the proposal event, to rebuild the effective grounding without the PDF or chat;
- a projector-derived current grounding, compatibility topic list, dashboard/intake receipt, and stable grounding references in later teaching/Session Receipts.

This reaches P0/P1 for the ST5201X fixture without P2 extractor breadth, generic retrieval infrastructure, a raw-PDF blob store, remote sync, or any redesign of Dunk’s learning-evidence model.

### Current extension points and invariants

- The event registry is closed at `dunk/scripts/dln_store/schema.py:14-23`, and `validate_event` strictly dispatches per-kind shapes at `schema.py:330-475`; the two syllabus event variants belong there. Keep global `SCHEMA_VERSION = 1` and use a syllabus/grounding schema version instead of forcing a migration.
- The current syllabus is only a mutable unique `list[str]`: initialized at `schema.py:212-224`, validated at `schema.py:227-249`, and overwritten by an allowed profile patch at `schema.py:277-297`. A patch increments the profile revision but records no source or approval history (`dunk/scripts/dln_store/store.py:659-675`).
- Cross-event integrity belongs in the reducer. `project_state` validates each event, rejects duplicate IDs/case-colliding or reused terminal session IDs, resolves only prior events, and indexes after reduction (`dunk/scripts/dln_store/projector.py:57-98,305-365`). It currently copies `profile.syllabus` unchanged into state (`projector.py:395-418`). Add a syllabus lineage/current-approval reducer alongside—not inside—the subject/mastery reducer. `domain_reset` currently resets learning state at `projector.py:335-343`; it must not reset course grounding.
- Existing storage already supplies the required durability boundary. `commit` checks the optimistic revision, deduplicates byte-equivalent event IDs, validates the complete candidate projection before I/O, preserves the event-log prefix, and publishes sources plus projections together (`dunk/scripts/dln_store/store.py:617-706`). The transaction stages, backs up, hashes, fsyncs, journals, and installs files (`store.py:416-509`); recovery discards an uninstalled preparation or rolls forward/restores a verified installation (`store.py:356-413`). Embedding source metadata, extracted page text, assertions, and approval decisions in `events.jsonl` reuses this boundary.
- Do **not** add the raw PDF as a third canonical file in the minimum slice. That would expand `DomainPaths`, source preconditions, recovery, backup, and validation. The issue permits raw extracted text by page/span instead of a durable content-addressed blob. The intake command must still read the original bytes and compute the stored SHA-256 itself; a caller-supplied path or digest alone is insufficient.
- `_projection_targets` already centralizes rebuildable state/dashboard/receipt output (`dunk/scripts/dln_store/store.py:528-545`). Add deterministic `syllabi/<source-id>-v<version>.md` intake receipts here and render approved source/version plus unresolved items in the dashboard (`dunk/scripts/dln_store/render.py:31-155`). `validate` already byte-checks all expected targets (`store.py:747-796`); extend orphan checks to the new receipt directory.
- A later Session Receipt cannot currently prove grounding use: it renders only evidence events cited by `session_completed.evidence_event_ids` (`dunk/scripts/dln_store/render.py:158-300`). Add optional non-mastery `grounding_refs` to an assessment (or an equivalent terminal session metadata field), validate them against the then-current approved assertion set in the reducer, and render a distinct “Course Grounding Used” section. These refs must never affect stage, subjects, retrieval, transfer, calibration, or outcomes.

### Minimum canonical model

1. `syllabus_source_ingested`: common event fields plus a stable logical source ID and monotonic source version; original filename, `application/pdf`, byte size, page count, store-computed SHA-256; ingestion/extraction timestamps; extractor name/version/status/diagnostics; raw extracted text by page/span; a canonical assertion-set hash; structured proposed assertions with stable IDs, typed normalized values, document/learner/external/inference origin, page/span/verbatim locator, and confirmation status; derived planning topics linked to assertion IDs; explicit ambiguity records. The ST5201X Week 7–13 record must retain the raw support and unresolved fields without asserting row mappings.
2. `syllabus_grounding_approved`: a prior intake-event/source-version reference, matching assertion-set hash, disjoint accepted/corrected/deferred IDs, complete learner correction values and attribution, actor/timestamp, grounding version, and prior-approval/supersession reference. The two events together must be a complete replayable approval record—not a pointer to chat or the transient PDF. The reducer overlays corrections without deleting the source-derived value, rejects unknown/cross-source refs and forks/skipped versions, and makes only the latest valid approval current. A pending newer source must not silently displace the prior approved source.
3. Projected state exposes only the current approved grounding plus compact version/status history. `state.syllabus` is derived from approved planning topics when grounding exists and falls back to `profile.syllabus` for legacy domains. `profile.syllabus` remains readable but is labeled legacy/ungrounded and is not co-written during source approval, avoiding two canonical copies of the same planning projection.

Use stable administrative session IDs for intake/approval events because every current event requires `session_id` and completed teaching session IDs are terminal (`projector.py:86-97`). Same-digest ingestion under a second event ID must be rejected or returned as an idempotent no-op; current exact-ID deduplication alone (`store.py:633-657`) does not prevent duplicate source registrations.

### Contract changes needed for fresh-session grounding

- The current `dln-syllabus` agent explicitly accepts only `{domain, goal}`, has no file/PDF read channel, produces only flat topics, and returns no citations or ambiguities (`dunk/agents/dln-syllabus.md:3-15,20-70`). Preserve its write-free/parent-persistence boundary, but add an authoritative-document proposal mode. The orchestrator must resolve the supplied document through a declared runtime channel; the child returns a strict, source-located proposal and never persists it. The parent calls a specialized intake CLI that rereads the PDF bytes, computes/verifies identity, and commits the proposal, then shows the generated intake receipt and separately commits learner approval/corrections.
- Initialization currently persists only `profile_patch.goal/syllabus` (`dunk/skills/dln/SKILL.md:69-81`). Resume already reloads and passes bounded disk-derived context (`dunk/skills/dln/SKILL.md:83-98`), so it is the correct no-chat-context seam: add current approved grounding and require course-specific claims to cite stable approved refs, unresolved claims to remain unresolved, and web/textbook material to remain supplemental.
- Dot explicitly plans from `profile.syllabus` (`dunk/skills/dln-dot/SKILL.md:26-47`); Linear uses profile syllabus/goal (`dunk/skills/dln-linear/SKILL.md:26-30`); Network has no grounding rule (`dunk/skills/dln-network/SKILL.md:26-32`). Link all three to one shared `syllabus-grounding-protocol.md` instead of duplicating logic or redesigning their pedagogy.
- `dln-compress` currently omits even syllabus from its allowed compact output (`dunk/skills/dln-compress/SKILL.md:12-45`). It must preserve the selected approved source/version, assertion IDs, locators, and ambiguity status byte-for-byte, or be forbidden on source-grounded routes; otherwise provenance can disappear between `context` and a phase.
- Keep grounding strictly separate from learning evidence. Syllabus events and grounding refs must not qualify as assessments, change stages, or count as mastery; the existing evidence gates and phase operations require no redesign.

### Smallest implementation decomposition

1. **Canonical intake/approval and projections.** Add the two event schemas and reducer invariants; a specialized PDF intake CLI that computes byte identity while accepting the child’s structured proposal; derived current grounding/topic compatibility; deterministic intake receipt, dashboard section, Session Receipt grounding section; schema/persistence/rendering documentation. Reuse the existing generic commit, transaction, recovery, and rebuild machinery rather than adding a blob store.
2. **Runtime contracts and downstream use.** Extend the write-free syllabus agent and orchestrator for explicit document proposal → generated review receipt → revision-checked approval; add one shared grounding protocol; update Dot/Linear/Network and `dln-compress` to consume/carry approved refs, disclose unresolved facts, label supplements, and keep grounding non-evidentiary.
3. **Fixture acceptance and evidence.** Add the supplied PDF (or an approved redistributable/redacted equivalent), source/approval/reducer/render/recovery tests, contract/evaluation cases for a fresh later session, and a reproducible `docs/evidence/issue-3/` capture.

### Exact acceptance checks

1. **Fixture identity/intake:** before any approval, run the declared intake interface against `syllabus2026.pdf`. Assert `45185` bytes, one page, `application/pdf`, and SHA-256 `53909df562e2658ab3e1327eb8c33120fa12b37489178dc87bb4d632e4f15376`; extractor name/version/status and diagnostics are present. Missing, unreadable, wrong-media, or extraction-unavailable input returns a truthful error/degraded status and does not claim approved grounding.
2. **Structured proposal/non-invention:** assert cited page-1 records for ST5201X, title, Semester 1 2026/2027, Zhang Yao, Thursday/Friday 7–10 PM at LT34, tutorials from Week 3, the John Rice third-edition textbook, Homework 50%, final exam 50%, stated submission/lateness/exam/materials/calculator policies, broad topics, homework milestones, and final-exam date `TBA`/not specified. Assert Week 7–13 row alignment is unresolved, raw support is retained, and no exact week/topic/homework mapping exists unless a learner correction explicitly supplies it. Every planning topic cites at least one assertion.
3. **Approval/version integrity:** commit approval only after intake and assert accepted/corrected/deferred partitions, actor/time, source version, assertion-set hash, and grounding version. Unknown/future intake refs, digest/hash mismatch, overlapping decision sets, missing correction provenance, skipped/forked approval versions, and approval of a degraded intake fail without changing the tree. A correction creates version 2 referencing version 1 while preserving the original value. Identical PDF/event replay at the refreshed revision is `noop`; same digest under another ID does not create a source version; changed PDF bytes create a new pending source version without replacing current approval.
4. **Atomicity/recovery:** stale revision, conflicting same ID, and reversed/malformed references leave all files byte-identical. Inject existing transaction failpoints during intake and approval. Caught failures restore the prior coherent tree; a subprocess crash blocks `context` until `doctor --recover`, then rolls forward or restores a state that is never half-approved. These extend the proven patterns at `dunk/scripts/tests/test_local_store_recovery.py:61-85,321-361`.
5. **Restart/rebuild/render:** instantiate a fresh store/process and run `context`; remove derived state/dashboard/intake/session receipts, run `rebuild` twice, and assert byte-identical current grounding, source/assertion IDs, flat topics, ambiguity, dashboard, and receipts from only `profile.yaml` + `events.jsonl`, with canonical bytes unchanged. Extend the byte-snapshot/rebuild checks at `dunk/scripts/tests/test_local_projections.py:54-70,456-480`.
6. **Downstream proof:** in a fresh chat/process with no PDF reread or original intake dialogue, load `context`, route an Acquire session, select an approved planning topic, emit a course-specific statement with its approved assertion/page locator, persist stable grounding refs, and assert the generated Session Receipt names the source/version/topic/assertions. A missing/unapproved grounding path says “ungrounded” and does not present generic web/model content as syllabus fact; an unresolved Week 7–13 query remains unresolved.
7. **Mastery separation/regression:** compare stage, generation, subjects, current model, retrieval, transfer, and calibration before/after intake, approval, correction, re-ingestion, and `domain_reset`; syllabus operations change none of them, and reset retains grounding. Run focused new tests plus the complete existing `pytest dunk/scripts/tests` suite, strict plugin validation, and deterministic fixture rebuild checks from `.github/workflows/validate.yml:53-74`.
8. **Contract surface:** static/evaluation tests assert the syllabus child remains return-only/no persistence tools; dln and all three phases reference the shared grounding protocol; the compressor preserves or rejects grounding explicitly; course-specific claims require approved refs; supplemental origins remain labeled; and syllabus coverage/ref use is never described as mastery.

### Evidence and screenshot plan

Create `docs/evidence/issue-3/` with a README and deterministic reproduce script, the exact input hash/size receipt, redacted request/response JSON, and four captures:

1. **Terminal intake/approval/rebuild:** fixture hash; intake revision/status; approval version; fresh-process `context`; delete/rebuild/validate success with unchanged canonical hashes.
2. **Syllabus Intake Receipt in Obsidian:** exact source fingerprint/extractor, cited ST5201X facts and policies, derived coverage, explicit Week 7–13 unresolved warning, then approved version/correction attribution.
3. **Dashboard after restart:** current approved source/version and unresolved count beside an empty/unchanged learner-evidence state, visibly demonstrating grounding ≠ mastery.
4. **Fresh later teaching + Session Receipt:** a course-specific response cites approved assertion/page IDs without the PDF/chat, and the generated receipt’s “Course Grounding Used” section names the same source/version/topic/assertions.

### Risks and eliminated alternatives

- **Runtime document resolution:** the current child has no PDF read tool. The implementation must name and test one resolvable attachment/path channel while ensuring the store hashes bytes itself. Do not claim grounding when that channel is unavailable.
- **Extractor/locator stability:** page/span locators are versioned with the source and extractor. Do not compare coordinates across extractor versions as if identical; retain verbatim page text for review.
- **Context size/provenance loss:** expose only current approved grounding plus compact history and route relevant refs; ensure `dln-compress` cannot strip them. A generic indexing/retrieval subsystem is P2.
- **Fixture distribution:** the supplied PDF contains staff/contact information. Confirm redistribution permission or prepare an accepted redacted fixture; still retain a live exact-hash acceptance run for the supplied file.
- **Raw PDF retention:** copying source blobs into each domain is intentionally eliminated for this slice because event-embedded extracted pages satisfy replay of approved grounding. Add a content-addressed blob store only if byte re-opening, not just identity/audit/rebuild, becomes a requirement.
- **Profile expansion:** a nested mutable profile syllabus, approval encoded only as a profile revision, or co-writing profile topics and event grounding creates ambiguous authority and is eliminated. Keep legacy profile lists as an explicitly ungrounded fallback.
- **Pedagogical redesign:** new assessment operations/stages, mastery effects, remote sync, broad PDF-format support, richer review UI, and automatic resolution of Week 7–13 are P2 or out of scope.

## Investigation Log

### Phase 1 — Issue, history, and fixture
**Hypothesis:** Issue #3 is a parser gap that can be fixed by extracting the supplied PDF into `profile.syllabus`.
**Findings:** The issue requires source identity, structured located assertions, durable approval/correction history, downstream use, and mastery separation. The supplied PDF matches the expected one-page/45,185-byte digest and its extracted Week 7–13 runs cannot be reliably aligned. Store history already establishes event sourcing and parent-owned approval.
**Evidence:** GitHub issue #3; SHA-256 `53909df562e2658ab3e1327eb8c33120fa12b37489178dc87bb4d632e4f15376`; commits `7ad8347`, `43b0200`, `f11caed`.
**Conclusion:** Eliminated. This is a course-authority contract gap, not just extraction.

### Phase 2 — Store, projection, and rendering
**Hypothesis:** New canonical files or a parallel syllabus store are required.
**Findings:** Existing append-only commit/recovery/rebuild already validates a complete candidate projection and atomically publishes sources plus derived artifacts (`store.py:617-706`). Event-embedded extracted page text, assertions, and decisions are sufficient canonical replay inputs; state currently copies only `profile.syllabus` (`projector.py:395-418`), and receipts currently render only cited learning evidence (`render.py:158-300`).
**Conclusion:** A sidecar store and raw-PDF blob are unnecessary for the minimum slice. Extend the event reducer and projection targets.

### Phase 3 — Runtime and downstream contracts
**Hypothesis:** Intake persistence alone satisfies fresh-session grounding.
**Findings:** The syllabus child accepts only `{domain, goal}` and returns uncited flat topics (`dunk/agents/dln-syllabus.md:22-70`); initialization writes only a profile patch (`dunk/skills/dln/SKILL.md:69-81`); and `dln-compress` omits syllabus grounding entirely (`dunk/skills/dln-compress/SKILL.md:20-45`).
**Conclusion:** Eliminated. A shared bounded grounding contract and stable receipt references are definition-of-done requirements, not optional breadth.

## Root Cause
Dunk has two distinct concepts but only one representation: authoritative course claims and mutable tutoring-plan topics are both collapsed into `profile.syllabus: list[str]` (`schema.py:212-249,277-297`). Profile revision records concurrency, not source or grounding lineage; the event registry has no syllabus intake/approval kinds (`schema.py:14-23`); projection and rendering therefore cannot reconstruct provenance, ambiguity, approval, or later use. The agent/orchestrator contracts reinforce the conflation by generating and committing only flat research topics.

## Recommendations
1. **Canonical intake/approval and projections:** add fixture-bound PDF intake that hashes bytes itself, plus replayable `syllabus_source_ingested` and `syllabus_grounding_approved` events, reducer lineage/ambiguity rules, derived legacy-compatible topics, deterministic intake/dashboard/session rendering, and focused schema/store/recovery tests. Keep extracted page text in events; do not add a raw-PDF store for this slice.
2. **Runtime contracts and downstream use:** preserve the write-free child/parent persistence boundary, add authoritative-document proposal and explicit approval flow, introduce one shared grounding protocol for Dot/Linear/Network, make `dln-compress` preserve grounding, and validate stable non-mastery grounding refs on later assessments/receipts.
3. **Fixture acceptance and evidence:** add an approved redistributable/redacted fixture or exact-hash local acceptance path, deterministic expected proposals/receipts, fresh-process and rebuild tests, negative Week 7–13 assertions, and `docs/evidence/issue-3/` reproduction captures.

## Preventive Measures
- Keep source/grounding schema versions independent from store schema and profile revision, and validate monotonic non-forking lineage in the reducer.
- Require every course-specific planning topic and later factual use to reference approved source assertions; label web/textbook/model content separately and preserve unresolved status through compression.
- Add negative non-invention tests, digest/idempotence tests, restart/rebuild snapshots, transaction failpoint coverage, and invariants proving syllabus operations never change learner-evidence state.
- Keep unsupported documents truthful: no approved grounding, no silent research fallback, and no claim that a caller-supplied path or digest proves byte identity.
