# Investigation: Generalizable Syllabus Intake

## Summary
Issue #3 should keep its privileged store, atomic transaction, replay, approval-lineage, and mastery-separation design, but replace the production ST5201X digest/manifest constructor with a media-neutral three-event pipeline: an atomic store-prepared source, a structurally sealed but semantically untrusted proposal, and a complete learner decision. Local and HTTPS acquisition must be bounded and store-owned; HTML extraction can be stdlib-native, while PDF support must be an explicitly available, fully provenanced adapter with truthful failure. ST5201X becomes only an adversarial non-invention fixture.

## Symptoms
- `st5201x_syllabus.py` and its bundled manifest encode source- and layout-specific knowledge in the production path.
- The CLI/store can replay approved assertions, but intake trust currently depends on a fixture-specific parser and event shapes that may admit caller-forged acquisition or extraction claims if generalized naively.
- Plugin execution must remain truthful under a zero-runtime-dependency/local-first model, including hosts without PDF libraries or unrestricted network access.
- HTTPS resolution introduces bounded-download, redirect, DNS/SSRF, media-sniffing, hashing, and provenance requirements absent from local fixture intake.

## Initial Hypotheses
- Acquisition, extraction, assertion proposal, and approval should be separate trust stages with independent provenance and hashes.
- Canonical replay should persist store-produced acquisition/extraction records and learner decisions, not trust arbitrary caller-composed source events.
- A stdlib core can safely support bounded local bytes and conservative HTTPS fetching, while PDF extraction likely needs an explicitly discovered system adapter or optional helper rather than a hidden runtime dependency.
- ST5201X should remain a negative/non-invention acceptance fixture, not a parser dispatch key or bundled production manifest.

## Background / Prior Research
- Git archaeology confirms fixture binding was deliberate, not accidental: `2459625` introduced the digest-bound ST5201X adapter/manifest, `7339047` tightened grounding invariants, and `6aa80c2`/`4e9e3d3` clarified reset/bundle behavior. The earlier `7ad8347` event-store and `43b0200` write-free-child/parent-persistence boundaries remain the architectural base.
- The current package declares Python `>=3.10` and `dependencies = []`. Plugin-installed content is replaceable/read-only, Python dependencies are not host-installed automatically, and canonical learner state must remain in explicit external storage rather than the plugin cache. This rules out silently requiring a PDF library or writing fetched artifacts under `${CLAUDE_PLUGIN_ROOT}`.
- HTML can be extracted with `html.parser` under a versioned, bounded normalization policy, but it is not browser-equivalent visible text and must suppress scripts/styles/templates and avoid subresource fetching. [Python `html.parser`](https://docs.python.org/3/library/html.parser.html)
- Viable PDF engines are environment adapters, not interchangeable truth: optional `pypdf` is portable/pure Python/BSD-3-Clause; `pdftotext` offers a killable Poppler subprocess when installed; macOS PDFKit is platform-specific; PyMuPDF is fast but large/native and AGPL/commercial-license sensitive. Each attempt needs exact engine/build/options/platform provenance and stable failure states such as `extractor_unavailable`, `encrypted`, `timeout`, `resource_limit`, `parse_error`, and `no_text`. [pypdf extraction guidance](https://github.com/py-pdf/pypdf/blob/main/docs/user/extract-text.md), [PyMuPDF installation](https://pymupdf.readthedocs.io/en/latest/installation.html), [Poppler `pdftotext`](https://manpages.debian.org/unstable/poppler-utils/pdftotext.1.en.html)
- A safe online-source resolver must accept one explicit HTTPS URL, reject credentials/fragments/ambiguous authorities, disable ambient proxies and automatic redirects, revalidate every redirect, reject any resolved non-global address, pin/verify the connected peer while retaining the hostname for TLS, impose connect/read/total and wire/decoded/extraction bounds, request identity encoding when possible, require conservative media agreement, and hash streamed decoded source bytes. URL/timestamp metadata is not source identity; the validated content hash is. Application policy should be backed by egress controls where available. [OWASP SSRF prevention](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html), [Python `urllib.request`](https://docs.python.org/3/library/urllib.request.html), [WHATWG MIME sniffing](https://mimesniff.spec.whatwg.org/)
- Reproducible provenance should include acquisition-policy version; request/redirect/final URLs; response/media/encoding decisions; resolved and connected addresses; byte counts and source hash; extractor ID/version/build/options; normalization-policy version; raw/normalized output hashes; warnings/status; and proposal/parser versions. Deterministic replay must consume persisted trusted records and approved assertions, not re-fetch mutable URLs.

## Investigator Findings

### 1. Current flow and fixture-specific seams

The current happy path is one fixture adapter followed by one learner decision:

1. `ingest-syllabus` accepts a local path plus caller-declared filename, media type, adapter, timestamp, and optional predecessor (`dunk/scripts/dln_store/cli.py:45-65`, `107-119`).
2. `LocalStore.ingest_syllabus()` calls `st5201x_syllabus.build_ingestion_event()` before entering the common commit path, then permits exactly the reserved source kind (`dunk/scripts/dln_store/store.py:834-872`).
3. The adapter opens a non-symlink regular file by descriptor, hashes it, and accepts only one 45,185-byte digest (`dunk/scripts/dln_store/st5201x_syllabus.py:21-23`, `49-98`). It then copies a checked-in manifest into one compound source/extraction/pages/assertions event (`st5201x_syllabus.py:103-136`, `139-181`).
4. `approve-syllabus` turns a complete caller request into a reserved approval event and commits it through the same transaction path (`dunk/scripts/dln_store/schema.py:1041-1084`; `dunk/scripts/dln_store/store.py:874-892`).
5. Every candidate commit reduces the complete timeline and renders state/dashboard/session/syllabus projections before installing the canonical pair and projections together (`dunk/scripts/dln_store/store.py:540-560`, `751-815`). Rebuild follows the same projection path (`store.py:894-909`).

The fixture coupling is broader than the filename:

- Production constants fix the source ID and adapter ID, while `SYLLABUS_FIELDS` includes the ST5201X-specific `schedule.weeks_7_13_alignment` (`dunk/scripts/dln_store/schema.py:14-57`).
- The generic-looking value validator admits `unresolved` only for that exact field, requires exactly seven Week 7–13 labels and two exact unresolved dimensions, and requires no alternatives (`schema.py:388-447`). It also fixes policy categories and caps homework numbers at five (`schema.py:469-500`).
- Source validation requires the hard-coded source ID, PDF media, the ST adapter/version/method, page-shaped text units, and rejects every exact schedule row from week 7 onward (`schema.py:792-903`). These are acceptance-fixture rules in the canonical schema, not adapter-local policy.
- The manifest hard-codes assertion IDs, normalized values, PDFKit/macOS provenance, a one-page flattened text snapshot, and the exact ambiguity caused by this PDF's Week 7–13 columns (`dunk/scripts/dln_store/data/st5201x-2026-v1.json:912-989`).
- Receipt grouping is reasonably reusable, but it assumes page text and says that original **PDF** bytes are not stored even though a future source may be HTML (`dunk/scripts/dln_store/render.py:96-151`, `244-263`).
- The complete test module imports the ST adapter directly and its helpers bake in the fixture path, digest, adapter, ST assertion IDs, and ST domain (`dunk/scripts/tests/test_syllabus_grounding.py:1-33`, `64-149`). Thus the tests strongly establish the adversarial fixture's behavior but do not establish a generic parser boundary.
- Runtime instructions also explicitly route every supplied document through `--adapter st5201x-2026-v1` and describe other bytes as unsupported (`dunk/skills/dln/SKILL.md:78-90`; `dunk/skills/dln/references/syllabus-grounding-protocol.md:1-29`; `dunk/LOCAL_STORAGE.md:52-68`, `100-104`).

One current local-file technique should survive: descriptor-based `O_NOFOLLOW` plus `fstat` prevents a path swap from turning the opened object into a symlink or non-regular file (`st5201x_syllabus.py:49-75`). One technique should not: after crossing the expected size, the loop clears retained chunks but continues reading and hashing to EOF (`st5201x_syllabus.py:76-89`). That is bounded memory, not bounded I/O or time. A general source reader must stop immediately at its configured byte ceiling.

### 2. What the current event authority does—and does not—prove

The normal API has a useful authority boundary. `RESERVED_SYLLABUS_EVENT_KINDS` contains source and approval kinds (`schema.py:14-28`); `_commit_request()` rejects either kind unless a dedicated store method explicitly allows it (`store.py:659-680`); and tests prove generic `commit` cannot inject even a schema-valid source or approval event without changing the tree (`dunk/scripts/tests/test_syllabus_grounding.py:458-545`). The source event is constructed after the adapter reads the descriptor, while the approval builder adds the event kind, schema version, canonical ordering, and approval hash (`schema.py:1041-1084`). Generalization must keep this rule: no public request may carry a ready-made acquisition, extraction, or prepared-source event body.

The boundary is application-level, not cryptographic authentication of the JSONL ledger:

- `parse_events_bytes()` accepts any event satisfying `validate_event()` (`schema.py:1086-1143`). A digest-shaped event ID, matching text hashes, and matching assertion-set hash establish internal consistency only; they cannot prove that the claimed raw bytes were ever observed or that the named extractor ran.
- The tests themselves demonstrate this distinction. A deep copy of the ST source event is assigned an arbitrary new source digest, event/session/version IDs, filename, and predecessor; unchanged extraction, pages, and assertions are then accepted by `project_state()` as a pending update (`test_syllabus_grounding.py:725-738`). This is appropriate for testing timeline behavior but is direct counter-evidence to treating schema validation as source attestation.
- The documented threat model says the filesystem is authoritative and `events.jsonl` must never be edited (`dunk/LOCAL_STORAGE.md:1-18`, `40-46`). Without retained bytes, a keyed MAC/signature, or an external immutable root, a local user able to rewrite both canonical files can forge a self-consistent history. Hashes detect accidental inconsistency, not an adversarial file owner.

Therefore the truthful guarantee should be scoped precisely: **the supported CLI/store write path does not trust caller-composed source/extraction events; replay trusts the canonical ledger under the existing local-filesystem-owner threat model.** If product requires detecting a malicious local vault editor, that is a separate signing/key-management design and cannot be achieved by adding more unkeyed hashes.

Approval integrity is substantially stronger at the relational level. Reduction requires an approval to reference the latest known source and its exact assertion-set hash, occur no earlier than ingestion, completely and disjointly accept/defer/correct every assertion, preserve a correction's target field, avoid no-op corrections, and form a linear supersession chain (`dunk/scripts/dln_store/grounding.py:306-423`). Grounded learning events must cite the approval active before them and only settled effective assertions (`grounding.py:214-266`). Those invariants should be retained, generalized from `source assertion` to `proposal`, and enforced while projecting the candidate transaction exactly as today.

### 3. Minimal truthful canonical shape

Four implementation stages are real—acquire, extract/normalize, propose, decide—but four separate canonical events are not the minimal truthful ledger. The best fit is **three canonical events, with the first event compound**:

1. **`syllabus_source_prepared` (store-owned compound event).** One dedicated method performs acquisition and extraction/normalization, then atomically records nested `source`, `acquisition`, `extraction`, and ordered `text_units`, plus independent hashes and policy versions. For PDF, units are pages; for HTML they may be one document or stable normalized blocks. The event has no semantic assertions.
2. **`syllabus_assertions_proposed` (non-authoritative proposal event).** A dedicated method accepts only proposals pinned to the prepared-event ID/hash. The store supplies kind/schema/event identity, validates bounded JSON and exact located evidence against canonical text units, generates stable proposal IDs and the proposal-set hash, and records producer provenance at the level it can prove. A host/model-supplied proposal must be labeled `external_unverified` (an optional model/tool label is merely declared metadata); a parser actually invoked by the store may be labeled `store_invoked` with exact adapter/version/config. Evidence-location validation proves that a quote exists, not that the proposed interpretation follows from it.
3. **`syllabus_decision_recorded` (complete learner snapshot).** This is the generalized current approval event, pinned to the prepared event and proposal-set hash. Every proposal receives exactly one disposition: accept, correct, or defer. It remains non-mastery metadata and forms a linear supersession chain.

This shape is preferable to the alternatives:

- **Four canonical events (`source_acquired`, `content_extracted`, `assertions_proposed`, `decision`) are eliminated for v1.** Acquisition and extraction happen in one CLI invocation over transient bytes. Since raw bytes are deliberately not retained (`dunk/LOCAL_STORAGE.md:88-104`), committing acquisition before successful extraction would leave a source that cannot be deterministically retried after a URL changes or a local file disappears. It also adds a partial state and foreign-key/revision transitions without improving crash safety.
- **One giant prepared-source event including assertions is eliminated as the generic interface.** It matches the current fixture snapshot but either forces a weak generic parser into the trusted store or invites a caller to forge parser/extractor provenance. Separating proposals makes the trust boundary explicit and permits asynchronous model-assisted proposal generation without making it source authority.
- **Keeping the current two-event shape is acceptable only if every supported parser is store-invoked.** That is too restrictive for general syllabus semantics under a stdlib-only runtime; extraction can be adapter-driven, but robust cross-institution semantic proposal generation cannot truthfully be promised by fixed regexes.

The existing store already supplies the required atomicity. It constructs all candidate canonical/projection bytes before `_install_transaction()` (`store.py:751-815`); recovery discards a merely prepared transaction, rolls forward a validated installing transaction, or restores backups (`store.py:365-422`). Each of the three lifecycle transitions can therefore be one ordinary atomic commit. Multiple stage events in one commit would not make replay more deterministic.

Recommended identities are distinct:

- `source_version_id = sha256-<raw-body-digest>` identifies exact acquired body bytes, independent of URL or filename.
- `prepared_document_sha256` hashes canonical normalized text units plus extraction/normalization policy identity; it detects a different representation of the same bytes.
- `proposal_set_sha256` hashes the ordered, validated semantic proposals.
- Each later event pins all relevant predecessor IDs and hashes. Replay never re-fetches, re-runs an extractor, or invokes a proposer; it reduces persisted prepared text, proposals, and decisions.

A re-extraction of identical raw bytes that produces a different prepared hash should not silently overwrite or deduplicate. Require an explicit `reprepare`/supersession action or reject it in v1; otherwise extractor upgrades can rewrite evidence locations beneath approvals.

### 4. Adapter boundaries under the zero-dependency/plugin model

The package contract is Python 3.10+ with `dependencies = []` (`dunk/scripts/pyproject.toml:1-8`), and canonical learner data belongs under the external vault, never the replaceable `${CLAUDE_PLUGIN_ROOT}` (`dunk/LOCAL_STORAGE.md:5-20`). A truthful boundary is therefore:

- **Acquirer (stdlib core, store invoked):** `local-file` and explicit single-URL `https` implementations stream into a bounded private temporary file or memory buffer while hashing. The temporary artifact is removed before return and never lives in the plugin directory. Caller metadata such as display filename or initial URL is not identity.
- **Extractor/normalizer (capability adapter, store invoked):** HTML has a stdlib adapter based on `html.parser` with a versioned normalization policy, no scripts/styles/templates, no subresource retrieval, bounded nesting/text/output, and explicit charset handling. PDF has no honest stdlib extractor. Discover adapters explicitly: a fixed-argument, no-shell `pdftotext` subprocess when installed; optional `pypdf` only when importable; a platform PDFKit helper only on supported macOS. Record exact engine/library version, executable identity/build if available, fixed options, platform, timeout, output limits, normalization policy, raw and normalized output hashes, warnings, and status.
- **Proposer (possibly external):** store-invoked deterministic rules may cover obvious headings and fields, but are not a general syllabus parser. The orchestration/model may submit proposals against a prepared document; the event must say the producer was external and unverified. Learner decision, not parser branding or confidence, creates authority.

PDF engines are not interchangeable. In-process libraries expose the long-lived CLI to parser CPU/memory faults; a `pdftotext` child is easier to time out and kill but still parses hostile input and is not a sandbox. Use fixed argv, no shell, a minimal environment, restrictive temporary directories, input/output/time limits, and the host's sandbox/resource controls where available. `extractor_unavailable`, `encrypted`, `timeout`, `resource_limit`, `parse_error`, and `no_text` must be stable non-success outcomes. No adapter availability means no canonical prepared event, not a guessed extraction or an ST manifest fallback.

For HTTPS, use a dedicated conservative transport rather than default `urlopen` behavior:

- accept one explicit `https` URL; reject userinfo, fragments, malformed/ambiguous authorities, and unsupported ports;
- disable ambient proxies, cookies, authentication, automatic redirects, and subresource fetching;
- resolve and validate every hop, reject any non-global or mixed global/non-global address set, connect to a selected validated address while preserving the hostname for TLS/SNI/certificate verification, and verify the connected peer; repeat for every bounded redirect;
- impose connect/read/total timeouts, header count/size, redirect count, raw/body/extracted-text limits, reject oversized `Content-Length`, and stop streaming immediately when the actual body exceeds the cap;
- request identity encoding and, for the minimal implementation, reject non-identity `Content-Encoding` rather than risk decompression bombs; require conservative agreement between declared media and content (`%PDF-` for PDF, declared HTML plus bounded sniffing for HTML);
- hash the exact response body bytes after transfer framing and before text decoding/normalization; persist request URL, redirect chain, final URL, validated/connected addresses, status/media/encoding decisions, byte counts, timestamps, acquisition-policy version, and body digest.

This reduces SSRF risk but does not turn application validation into a network sandbox. Deployment egress controls remain the stronger boundary. Online intake should be explicit/opt-in, with local files continuing to work fully offline.

### 5. Generic proposal, evidence, ambiguity, and decision invariants

Replace fixture-shaped `page_number` evidence with a media-neutral locator:

```json
{"unit_id":"page:1","start_char":10,"end_char":24,"quote":"exact normalized slice"}
```

A prepared document contains ordered unique units with `unit_id`, `kind` (`page`, `html_block`, or `document`), optional label, NFC/LF UTF-8 text, and `text_sha256`. Evidence must reference an existing unit, remain within configured bounds, and equal the exact slice. This preserves the strong current quote check (`schema.py:528-625`) for both PDF and HTML.

Proposal IDs and event IDs should be store-generated from canonical content (including locator occurrence where necessary), not trusted caller IDs. A proposal contains a versioned field/predicate, bounded typed JSON value, status, located evidence, optional note, ambiguity, and producer reference. Use a versioned core vocabulary for fields the planner understands (especially coverage topics), plus a bounded namespaced extension form that renders but is never silently used by planning. Remove ST-specific cardinalities and categories from the core schema.

Status and confidence must not be conflated:

- `specified`: one proposed value; evidence is required.
- `ambiguous`: no settled value; require a reason and one or more candidates or explicitly unknown dimensions, each linked to evidence where possible.
- `not_specified`: use only when the text explicitly says TBA/unspecified/not applicable and cite that text. Do not manufacture assertions from mere absence; omit them instead.

An ambiguous proposal cannot become citable merely because its ID appears in `accepted`. Require defer or a learner correction that settles it. A correction must target one proposal and preserve the original proposal/evidence as historical document context. The corrected value's authority is `learner_correction`; the original quote must not be presented as if it semantically supports the corrected value. The current reducer preserves target citations on corrected effective assertions (`grounding.py:44-65`), so rendering and grounding should distinguish `document_context` from evidence for the learner-supplied correction.

Retain the current invariants: complete/disjoint dispositions, exact proposal-set pinning, latest-source/latest-proposal approval, monotonic timestamps, linear source and decision supersession, immutable historical resolution, and mastery neutrality (`grounding.py:214-266`, `306-423`; `projector.py:357-360`, `404-430`).

### 6. Precise acceptance fixtures beyond ST5201X

1. **`generic-two-page-syllabus.pdf` (local text-layer PDF).** A committed deterministic two-page PDF with no `ST5201X` token: course `CS101`, two instructors, assessment weights, a page-2 schedule, one explicit `Exam date: TBA`, and one repeated header. Pin raw digest and expected units/proposals for each supported production extractor. In zero-dependency CI, run the same store boundary with an injected deterministic test extractor; separately assert that a host with no PDF adapter returns `extractor_unavailable` and leaves the canonical tree byte-identical. This proves PDF adapter discovery, multi-page evidence, generic field IDs, explicit-not-specified handling, and truthful degradation.
2. **`generic-ambiguous-columns.pdf` (non-ST adversarial PDF).** A two-column schedule whose extracted order cannot determine whether two assignments belong to Week 4 or Week 5. Expected result: topics and assignment labels may be proposed, but the mapping is one generic `ambiguous` proposal with located evidence/candidates; no exact schedule rows are invented. This ensures ST5201X is not the only non-invention test and that ambiguity is not keyed by its digest or course code.
3. **`generic-syllabus.html` via a scripted HTTPS transport.** HTML contains entities, headings, a grading table, duplicated/conflicting room values, `<script>`, `<style>`, `<template>`, comments, a relative image, and a link. The scripted resolver/transport records a valid global address and one redirect. Expected: hidden content is excluded, no linked resource is requested, conflict remains ambiguous, hashes/redirect/final URL are pinned, and offline rebuild succeeds after the transport is removed.
4. **Local/HTTPS byte-equivalence pair.** Serve the exact bytes of fixture 1 as `application/pdf` over the scripted HTTPS transport and ingest the same bytes locally in a fresh equivalent domain. Expected raw source digest and normalized document hash are equal; acquisition provenance differs; exact replay is a no-op and never re-downloads during context/rebuild.
5. **HTTPS rejection matrix (scripted, no public network):** credentials, `http`, fragment, redirect to loopback/private/link-local/`169.254.169.254`/IPv6 ULA, mixed DNS answers, peer-IP mismatch/rebinding, redirect loop/overflow, header overflow, lying `Content-Length`, streamed body overflow, non-identity compression, timeout, PDF MIME without magic, and HTML/PDF media mismatch. Every case must return a stable error and preserve the complete domain tree.
6. **Proposal/decision tampering and replay fixture.** Against fixtures 1 and 3, submit an out-of-range quote, mismatched quote, proposal pinned to another prepared hash, forged ready-made prepared event through generic `commit`, incomplete/overlapping dispositions, acceptance of ambiguity, and a correction with a mismatched field. All fail without writes. A valid accept/correct/defer snapshot commits; with network, extractor, and proposer subsequently unavailable, `context`, `validate`, and `rebuild` reproduce byte-identical state and receipts from canonical events.

Keep ST5201X as an additional adversarial fixture: run it through the generic PDF path, require its Week 7–13 alignment to remain ambiguous, and prohibit any digest/course-code/filename dispatch to a bundled assertion manifest.

### 7. Product choices that cannot be decided by code evidence

1. **Local tamper threat model:** Is preventing caller-forged events limited to the supported CLI/store API (consistent with today's local-authoritative filesystem), or must Dunk detect a malicious editor of `events.jsonl`? The latter requires retained source bytes and revalidation, or signing/MAC key management/external trust; unkeyed event hashes cannot provide it.
2. **Online intake default and SSRF policy:** Should HTTPS intake be disabled by default/require an explicit flag, and are redirects permitted at all? A no-redirect default is safer; permitting redirects needs the full per-hop policy above.
3. **PDF support promise:** Is `pdftotext` the preferred optional production adapter, should optional `pypdf`/PDFKit also be supported, or is “extractor unavailable until the host installs one supported adapter” acceptable? Exact golden text and support burden depend on this choice.
4. **Source retention:** Continue `extracted_text_only`, or optionally retain source bytes in a content-addressed vault for later audit/re-extraction? The former is more private/minimal but cannot independently re-prove extraction after intake.
5. **Learner-added assertions:** May a decision add facts absent from proposals? The minimal design allows only corrections to proposed fields. If additions are required, represent them as `learner_supplied` and never as document-derived unless a later proposal supplies located evidence.
6. **Lineage cardinality:** Is there exactly one syllabus lineage per learning domain, as current projection assumes, or can one domain have multiple simultaneously active syllabus-like sources? Multiple active lineages materially changes approval selection and grounding and should not be slipped into this revision.

## Investigation Log

### Phase 1 — External constraints and archaeology
**Hypothesis:** A general parser can replace the fixture adapter without changing the existing trust boundary.  
**Findings:** The fixture was intentionally introduced by `2459625` because the host does not install PDF tooling and the package declares no runtime dependencies. HTML has a bounded stdlib path; PDF requires an optional capability. Safe HTTPS behavior cannot use ambient `urllib` defaults.  
**Evidence:** `dunk/scripts/pyproject.toml:1-8`; `dunk/LOCAL_STORAGE.md:5-20`; the primary-source links in Background / Prior Research.  
**Conclusion:** Partly confirmed. Preserve the privileged boundary, but introduce explicit acquisition/extraction capabilities and stable unavailable/error states.

### Phase 2 — Fixture and schema seams
**Hypothesis:** Only `st5201x_syllabus.py` and its manifest are fixture-specific.  
**Findings:** ST5201X identifiers, field vocabulary, Week 7–13 ambiguity, PDF-only units, renderer wording, runtime docs, and tests are also encoded in production contracts.  
**Evidence:** `dunk/scripts/dln_store/schema.py:14-57,388-500,792-903`; `dunk/scripts/dln_store/render.py:96-151,244-263`; `dunk/scripts/tests/test_syllabus_grounding.py:1-33,64-149`; `dunk/skills/dln/SKILL.md:78-90`.  
**Conclusion:** Confirmed cross-cutting fixture coupling. Legacy validators remain readable, but none of these rules should drive new ingestion.

### Phase 3 — Authority and replay
**Hypothesis:** More hashes can prevent caller-forged source events.  
**Findings:** Dedicated store methods and reserved kinds protect the supported write API (`store.py:657-680`; tests at `test_syllabus_grounding.py:458-545`). Hashes validate internal relationships, but a vault owner can rewrite a self-consistent JSONL history; `test_syllabus_grounding.py:725-738` demonstrates schema-valid synthetic source replacement. Existing projection and recovery already make each accepted transition atomic (`store.py:751-815`, `365-422`).  
**Conclusion:** Reject cryptographic-attestation language. The supported store path is trusted; the canonical local files remain authoritative under the existing filesystem-owner threat model.

### Phase 4 — Event granularity
**Hypothesis:** Acquisition and extraction need separate canonical events.  
**Findings:** Without retained bytes, a committed acquisition followed by extraction failure is a durable half-intake that may be impossible to retry. The four logical stages do not require four ledger records.  
**Conclusion:** Use three records: compound successful preparation, proposal, decision. Failed acquisition/extraction returns bounded diagnostics and makes no domain mutation. A separate operational audit log, if later required, is not course grounding.

## Root Cause
The issue-3 branch correctly made the store—not a child agent or generic JSON request—the authority for syllabus events, but the only privileged constructor equates “trusted intake” with “this exact ST5201X digest unlocks this bundled manifest.” The coupling appears in the adapter (`st5201x_syllabus.py:143-182`), production schema (`schema.py:31-57,388-500,792-903`), runtime instructions, render wording, and acceptance helpers. This makes the current claim truthful but non-portable.

Generalizing by merely accepting caller-built source events would break the strongest existing invariant: generic `commit` cannot create reserved syllabus events (`store.py:657-680`). Generalizing by putting an LLM or arbitrary parser inside that trusted constructor would overstate what the store can prove. The store can attest that it acquired exact bytes under a named policy, invoked a named extractor, computed hashes, and verified locator slices. It cannot attest that an external proposer identity is authentic or that a located quote semantically entails a proposed fact. Only explicit learner decisions make proposals usable course grounding.

## Recommended Architecture
### Canonical lifecycle

1. **`syllabus_source_prepared` — privileged and atomic.** A dedicated store method accepts either one local regular file or one explicit HTTPS URL, performs bounded acquisition, identifies exact accepted bytes by SHA-256, invokes one extraction/normalization adapter, and commits only on successful reviewable text production. The event contains nested acquisition and extraction provenance, ordered normalized text units, `source_sha256`, and a `prepared_document_sha256`. It contains no semantic assertions. Generic `commit` rejects this kind.
2. **`syllabus_assertions_proposed` — sealed, not authoritative.** A dedicated method accepts bounded proposal data pinned to the prepared event/hash. The store generates IDs/order/hash, validates structural types and exact evidence slices against canonical units, and records producer provenance honestly: `store_invoked` only for code it actually invoked; otherwise `external_unverified` with optional declared model/tool metadata. The store attests locator integrity, not semantic entailment. Generic `commit` rejects ready-made proposal events.
3. **`syllabus_decision_recorded` — learner authority.** A complete, disjoint snapshot pins the prepared document and proposal-set hashes and assigns every proposal exactly one disposition: `accepted`, `corrected`, `deferred`, or `rejected`. Ambiguous proposals cannot be accepted; they must be corrected, deferred, or rejected. Corrections preserve the original proposal/evidence, record learner attribution/rationale, and distinguish `learner_correction` from document support. Decisions form a monotonic, non-forking supersession chain.

Retain `syllabus_source_ingested` and `syllabus_approval_recorded` as read-only legacy v1 event variants so existing ledgers that already contain them rebuild unchanged. Stop emitting them. Move the ST adapter and assertion manifest out of the production dispatch path and into test/evidence fixtures; a small legacy validator may retain their old schema without exposing a live ingestion adapter.

### Acquisition boundary

- **Local:** preserve descriptor-based `lstat`/regular-file checks, `O_NOFOLLOW` where available, and `fstat`; stream and stop immediately at the byte cap; do not persist absolute paths. Persist display filename, byte count/hash, detected media, timestamp, and policy version.
- **HTTPS:** opt-in, one explicit `https` URL only, no credentials/fragments/ambiguous authority, no subresources, cookies, auth, ambient proxies, or automatic redirects. Resolve each hop; reject any mixed or non-global IPv4/IPv6 result; connect to a pinned validated address while preserving hostname for TLS/SNI and certificate validation; verify the peer; and repeat policy on every permitted redirect. Enforce header, redirect, connect/read/total-time, wire/body, and extracted-output bounds. Request and initially require identity encoding. Require conservative declared/sniffed media agreement and hash accepted response bytes while streaming. Deployment egress controls remain stronger than application checks.
- **Proposed v1 limits:** 16 MiB source, 8 MiB normalized text, 500 units/pages, 5 s connect, 10 s read inactivity, 30 s total. Default to no redirects; if product enables them, cap at three and fully revalidate every hop. Serialize the policy/version so these numbers are replayable provenance.
- Failed acquisition or extraction returns a stable machine-readable error and leaves `profile.yaml`, `events.jsonl`, and projections byte-identical. Do not canonically record an unreviewable half-intake.

### Extraction/parser strategy

| Strategy | Portability/security | Recommendation |
|---|---|---|
| Stdlib `html.parser` | Available everywhere; no scripts/subresources; not browser-equivalent and needs explicit charset/block/whitespace rules | Ship as the built-in HTML adapter with versioned normalization and strict bounds. |
| Poppler `pdftotext` | Mature and killable out-of-process; not universally installed; native parser is not a sandbox | Strong initial PDF adapter if discovered/configured; fixed argv, no shell, private temp dir, minimal environment, timeout/output limits, exact executable/version/options provenance. |
| Optional `pypdf` | Cross-platform, pure Python, BSD-3-Clause; in-process CPU/memory exposure and no OCR | Supported optional adapter only when importable, preferably through an isolated helper; pin/version provenance. |
| macOS PDFKit helper | Zero Python package on macOS; OS/framework-dependent output | Optional platform adapter, never the portable baseline. |
| PyMuPDF | Fast/rich but large/native and AGPL/commercial-license sensitive | Do not make the default; require legal review and explicit capability configuration. |

Adapter status is one of `success`, `extractor_unavailable`, `encrypted`, `timeout`, `resource_limit`, `parse_error`, `no_text`, or `unsupported`. Record adapter contract/engine/build/options/platform, policy versions, timestamps, warnings/diagnostic digests, raw output hash, normalization hash, and ordered text-unit hashes. Automatic discovery order, if offered, must itself be versioned and recorded; explicit `--extractor` is easier to reproduce. Deterministic replay means reducing persisted text/proposals/decisions without rerunning the extractor—not promising identical future extraction across versions.

### Generic proposal contract

- **Text unit:** `{unit_id, kind: page|html_block|document, label?, text, text_sha256}` using NFC UTF-8 and LF endings.
- **Evidence:** `{unit_id, start_char, end_char, quote}`; unit exists, interval is bounded, and quote equals the exact canonical slice. Multiple locators are allowed.
- **Proposal:** store-generated ID; versioned/namespaced predicate; presentation label/category; semantic roles such as `planning_topic`; structural value type (`text`, `integer`, `decimal`, `boolean`, `date`, `time`, `percentage`, `list`, `object`, `unknown`); canonical JSON value; status; located evidence; note; ambiguity references; producer reference.
- **Status:** `specified`; `explicitly_unknown` only when the source itself says TBA/unknown and is cited; or `ambiguous`. Mere absence creates no assertion. Confidence is optional declared metadata, never authority.
- **Ambiguity:** reason, unresolved dimensions, located support, and optional candidate interpretations. Candidates are non-citable until learner correction. Production code contains no Week 7–13 special case.
- The planner consumes explicit roles rather than magic predicates such as `coverage.topic`. Unknown namespaced predicates render safely but do not silently drive planning.

### Replay, retention, and trust wording

Persist bounded normalized extraction, proposals, and decisions in canonical events; retain `extracted_text_only` as the default and do not add a blob store in this revision. `context`, `validate`, `rebuild`, dashboards, and receipts never access the network, reopen files, invoke extractors, or call proposers. The hash chain is:

`source bytes → prepared document → proposal set → learner decision → effective grounding`.

Later assessments keep pinning the active decision plus settled assertion IDs and never affect mastery through grounding alone. A re-extraction of identical raw bytes that changes the prepared hash is a new explicit preparation/supersession, never an in-place rewrite. Receipts must say which claims are document-proposed, learner-corrected, deferred/rejected, and unresolved.

## Implementation Decomposition
1. **General canonical model and migration.** Add generic prepared/proposal/decision schemas, reserved dedicated store methods, hash/lineage/reducer invariants, role-based bounded grounding, generic receipts, and read-only legacy v1 dispatch. Remove ST5201X from production intake/ontology while preserving old-ledger replay. Extend atomicity, recovery, idempotence, tamper-at-API, rebuild, and mastery-neutrality tests.
2. **Bounded source and extractor adapters.** Add local acquisition, opt-in conservative HTTPS transport with injected test transport/resolver, stdlib HTML normalization, and a versioned PDF adapter interface with at least one explicitly chosen optional engine plus truthful unavailable/error behavior. Keep `dependencies = []`; do not use MCP, crawl links, retain source blobs, or add OCR.
3. **Runtime proposal/approval contracts and acceptance.** Add a return-only document proposal mode/agent, store sealing and complete learner decision flow, update the shared grounding/compression/phase protocols, and replace fixture-specific acceptance with the portable matrix below. Regenerate issue evidence only after the generic fixtures pass.

## Acceptance Fixtures
Beyond ST5201X, add these exact fixtures; CI uses injected resolvers/transports/extractors and never depends on public network:

1. **`generic-two-page-syllabus.pdf`** — deterministic local text-layer PDF with course `CS101`, two instructors, assessment weights, a page-2 schedule, `Exam date: TBA`, and a repeated header. Pin bytes/hash and expected units/proposals for the chosen acceptance extractor. Verify multi-page locators, explicit-unknown semantics, a no-engine `extractor_unavailable` result, no writes on failure, and no course-code/digest dispatch.
2. **`generic-ambiguous-columns.pdf`** — non-ST two-column schedule where extraction cannot assign two milestones to Week 4 versus Week 5. Require a generic ambiguity with located candidates/unresolved dimensions and prohibit invented schedule rows.
3. **`generic-syllabus.html` over scripted HTTPS** — entities, headings, grading table, conflicting room values, script/style/template/comments, relative image, and link; one allowed redirect with fixed public DNS/peer metadata. Require hidden content suppression, zero subresource requests, conflict ambiguity, exact acquisition/text hashes, and offline rebuild after transport removal.
4. **Local/HTTPS byte-equivalence pair** — ingest the exact fixture-1 PDF bytes through local and scripted HTTPS paths in equivalent domains. Raw and normalized hashes match; acquisition provenance differs; replay/context/rebuild performs no download or extraction.
5. **HTTPS rejection matrix** — credentials, `http`, fragment, forbidden/mixed DNS addresses, loopback/private/link-local/metadata/IPv6 ULA redirects, peer mismatch/rebinding, redirect loop/overflow, headers/body/declared-length overflow, non-identity encoding, timeout, MIME/magic disagreement. Every case yields a stable code and byte-identical canonical tree.
6. **Proposal/decision/replay tamper matrix** — invalid interval/quote, cross-document locator/hash, generic-commit injection of every reserved kind, incomplete/overlapping disposition, accepting ambiguity, mismatched correction predicate, and changed HTTPS body. Failures make no writes; a valid accept/correct/defer/reject decision rebuilds byte-identically with network, extractor, proposer, and original source unavailable.

Keep **ST5201X** as an additional adversarial acceptance fixture through the same generic PDF path. It must preserve the Week 7–13 uncertainty, fresh-session citations, and mastery separation; production code may not reference its digest, course code, filename, assertion IDs, or bundled manifest.

## Product Questions
1. **Threat model:** Is the requirement limited to preventing forged provenance through supported CLI/store APIs, consistent with the current local-filesystem-owner model? Detecting a malicious editor of `events.jsonl` requires a separate source-retention or MAC/signing/key-management design.
2. **PDF support promise:** Which adapter is supported first—`pdftotext`, optional `pypdf`, PDFKit, or more than one—and is truthful `extractor_unavailable` acceptable on hosts with none? This determines golden extraction fixtures and support burden.
3. **Online policy/privacy:** Is HTTPS disabled unless explicitly enabled; are any redirects allowed; and may query-bearing/signed URLs, exact redirect URLs, and resolved/peer IPs be persisted? Recommended v1 is opt-in, no redirects by default, port 443, and rejection of query-bearing URLs until a redaction/retention policy exists.
4. **Source retention:** Is `extracted_text_only` sufficient, or must raw source bytes be content-addressed for independent audit/re-extraction? The recommendation is extracted text only; raw blobs materially expand privacy, recovery, and storage scope.
5. **Cardinality:** Is there exactly one active syllabus lineage per learning domain? Supporting multiple simultaneously authoritative sources introduces conflict/merge semantics and should be a separate feature.

## Resolved decisions (item 2 implementation)

The later living plan superseded the preliminary zero-dependency/no-source-retention recommendation above. Item 2 selected the newest validated Python-3.10-compatible stable release, exact direct pin `pypdf==6.14.2`, with plain-mode golden extraction through a fixed-argument resource-bounded worker. Original bytes and normalized prepared text are retained as canonical private CAS beneath each external domain store; rebuild never fetches or extracts. The production lifecycle is generic `prepare-syllabus` → `syllabus-content` → `propose-syllabus` → `decide-syllabus`. HTTPS requires explicit network consent, uses query-free port-443 URLs, has redirects disabled unless separately consented, and revalidates at most three hops. Tests inject resolver/transport and never contact public network. Exactly one authoritative lineage exists; supplements remain non-authoritative.

## Preventive Measures
- Keep every provenance-bearing event kind reserved and construct it only inside dedicated store methods; test generic-commit injection for each new kind.
- Version acquisition, extraction, normalization, proposal, and decision policies independently; store all predecessor hashes and reject silent re-extraction changes.
- Test negative non-invention on both ST5201X and an unrelated ambiguous-layout PDF so uncertainty handling cannot become fixture dispatch.
- Keep network/extractor/proposer code out of replay paths and assert this with failing/injected dependencies during `context`, `validate`, and repeated `rebuild`.
- Preserve original proposal/evidence through corrections, label external producer metadata unverified, and state explicitly that hash consistency is not local-ledger authentication.
- Maintain resource, SSRF, media, prompt-injection, markdown-literal, and mastery-neutrality regression suites; no live network, OCR, crawling, rich UI, or source blob store enters this revision implicitly.
