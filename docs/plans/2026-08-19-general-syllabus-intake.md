# Issue #3 Revision Plan: Portable Authoritative Syllabus Intake

**Status:** Items 1–2 and item-3 evidence/release preparation complete; commit/no-mistakes/PR work remains with the outer orchestrator  
**Branch baseline:** `issue-3-authoritative-syllabus` at post-no-mistakes commit `4e9e3d3`  
**Grounding:** `docs/investigations/general-syllabus-intake-2026-08-19.md`, RepoPrompt Context Builder plan, and Oracle review

## Settled scope

Replace the production ST5201X digest/manifest path with a generic **prepare → propose → decide** lifecycle while preserving the current reserved-event authority boundary, candidate replay, journaled atomic install/recovery, deterministic offline rebuild, complete learner decision, and mastery separation.

- Ship one exact direct `pypdf==<validated-version>` runtime pin for portable text-layer PDF extraction; no OCR or alternate PDF engine in this revision. Use stdlib `html.parser` for HTML.
- Retain bounded original bytes and normalized extracted text as content-addressed canonical data under each external domain store, never `${CLAUDE_PLUGIN_ROOT}`.
- Permit exactly one authoritative syllabus lineage per domain. Each new authoritative version explicitly supersedes the latest version; supplements remain labeled non-authoritative and can never drive active grounding, planning topics, or learning citations.
- Support local files and explicit opt-in HTTPS for PDF and HTML. HTTPS must be bounded and safe against SSRF, redirects, DNS rebinding, and connected-peer mismatch. CI must use injected transports/resolvers, never public URLs.
- Keep the local-filesystem-owner threat model: hashes prove internal consistency, not protection from a malicious owner rewriting every canonical artifact.
- ST5201X remains only an adversarial fixture. Production code/data/ontology must not dispatch on its name, digest, course code, assertion IDs, or Week 7–13 layout.

## 1. Canonical lifecycle, compatibility, and atomic content store

**Depends on:** current `4e9e3d3` transaction/replay design.  
**Status:** Complete — independently verified 121 Dunk tests and no ST-specific production dispatch.

### Work

- In `dunk/scripts/dln_store/schema.py`, replace fixture-specific production validation with three reserved kinds:
  - `syllabus_source_prepared`: role, raw-byte identity, CAS references, acquisition/extraction provenance, prepared-text digest, and authoritative predecessor when applicable.
  - `syllabus_assertions_proposed`: immutable bounded proposal set pinned to the prepared source, portable typed fields/semantic roles, and exact media-neutral locators `{unit_id,start_char,end_char,quote}`. External producer metadata is explicitly unverified.
  - `syllabus_decision_recorded`: complete accept/correct/defer/reject partition pinned to the proposal-set digest. Ambiguity cannot be accepted; corrections remain learner-authored and preserve document context.
- In `dunk/scripts/dln_store/store.py`, add external-domain CAS paths such as `sources/sha256/<source-digest>` and `prepared/sha256/<prepared-digest>.json`. Extend the existing transaction target/journal/recovery path so new CAS bytes, `events.jsonl`, `profile.yaml`, projections, and receipts install or recover as one unit. Verify non-symlinked paths, exact hashes, and referenced content before reduction; `validate` reports missing, corrupt, and orphaned content, while `rebuild` must never refetch, re-extract, or backfill it.
- Add dedicated `prepare_syllabus`, `propose_syllabus`, `decide_syllabus`, and read-only verified `syllabus_content` store operations. Generic `commit` must continue rejecting every old and new reserved kind.
- In `dunk/scripts/dln_store/legacy.py` and `grounding.py`, normalize existing `syllabus_source_ingested` events to authoritative prepared sources plus their embedded proposal sets, and `syllabus_approval_recorded` events to learner decisions. Preserve historical IDs and `approval_event_id` citations; new writes use only the three new kinds and `decision_event_id`. Do not rewrite ledgers or invent raw-byte CAS backfills for legacy text-only events.
- Update `grounding.py`, `projector.py`, and `render.py` for proposal/decision phases, page or HTML-document locators, visible supplements, and one non-forking authoritative source/decision chain. A prior authoritative decision stays active as `approved_update_pending` until the explicitly superseding version receives a complete decision. Mastery reducers remain unchanged.
- Delete production `dunk/scripts/dln_store/st5201x_syllabus.py` and `dunk/scripts/dln_store/data/st5201x-2026-v1.json`; move only the data needed for legacy replay assertions under `dunk/scripts/tests/fixtures/syllabus/st5201x/`.
- Rewrite `test_syllabus_grounding.py`; update `test_local_store_recovery.py`, `test_local_store.py`, `test_local_projections.py`, and affected `fixtures/local_store/expected-*` snapshots.

### Done criteria

- A committed legacy-v1 ledger fixture rebuilds and resolves historical citations without its source PDF, production ST manifest, or CAS backfill.
- Prepared/proposed/decided transitions are atomic and idempotent; prepared/installing failpoints prove CAS, canonical pair, and projections roll back or roll forward together.
- Reserved-kind injection, invalid/mismatched locators, incomplete/overlapping decisions, ambiguity acceptance, missing/corrupt CAS, and authoritative forks/skipped predecessors fail without domain writes.
- Supplements render as non-authoritative but cannot alter active status/source/decision, planning topics, eligible citations, or the byte-equivalent mastery view.
- `dunk/scripts/dln_store/` contains no ST5201X identifier, digest dispatch, bundled manifest, or fixture-specific ontology rule.

## 2. Portable bounded acquisition/extraction, dependency, CLI, and contracts

**Depends on:** Item 1 schemas, CAS transaction inputs, and lifecycle APIs.  
**Status:** Complete — pinned `pypdf==6.14.2`; 68 targeted acquisition/security/contract tests and 169 full Dunk tests pass under the frozen Python 3.10.20 runtime.

### Work

- Add `dunk/scripts/dln_store/acquisition.py`, `extraction.py`, and `pdf_worker.py`:
  - Local acquisition preserves descriptor-based no-follow/regular-file checks and stops immediately above a 16 MiB source limit.
  - PDF extraction invokes the pinned `pypdf` in a fixed-argument child process with no shell, private temporary files, minimal environment, timeout/output/resource limits, at most 500 pages, and at most 8 MiB normalized NFC/LF UTF-8 text.
  - HTML extraction uses a bounded stdlib parser, excludes scripts/styles/templates/comments, fetches no subresources, and emits one normalized document unit.
  - Stable failures include unsafe URL/DNS/peer/redirect, source/header/body limits, timeout, media mismatch, encrypted/parse/resource/no-text errors; every pre-commit failure removes temporary data and preserves the domain tree byte-for-byte.
- HTTPS is available only through an explicit URL plus network-consent flag. Disable ambient proxy/cookies/auth/compression/automatic redirects; accept HTTPS port 443 only and reject userinfo, fragments, queries, and ambiguous authorities. Resolve and validate every hop, reject empty/mixed/non-global/private/loopback/link-local/metadata/reserved/multicast/unspecified/IPv6-ULA answers, connect to a selected validated address while retaining hostname TLS/SNI verification, and require the connected peer to match. Redirects are off by default; explicit redirects are capped at three and fully revalidated. Bound connect/read/total time, header count/bytes, declared and streamed body size, identity encoding, and declared/sniffed PDF/HTML agreement.
- Replace fixture commands in `dunk/scripts/dln_store/cli.py` with `prepare-syllabus`, `propose-syllabus`, `decide-syllabus`, and `syllabus-content`; enforce option/predecessor rules and return stable machine-readable error codes.
- In `dunk/scripts/pyproject.toml`, replace the zero-runtime-dependency contract with one exact direct `pypdf==<validated-version>` pin selected during implementation after Python 3.10 and golden-extraction validation. Regenerate `dunk/scripts/uv.lock`, review the resolved graph, and prove the existing CI `uv sync/run --frozen --python 3.10.20` path installs and imports that exact version. Change `.github/workflows/validate.yml` only if additional explicit frozen/security test coverage is needed.
- Add `test_syllabus_acquisition.py` and deterministic fixtures under `tests/fixtures/syllabus/generic/`: a non-ST two-page text-layer PDF, a non-ST ambiguous-column PDF, adversarial HTML, pinned expected `pypdf` extraction, and proposal/decision/supplement requests. Keep `st5201x/syllabus2026.pdf` unchanged as an extra adversarial input processed through the same extractor.
- Update `test_plugin_contracts.py`, `dunk/LOCAL_STORAGE.md`, `dunk/CHANGELOG.md`, `dunk/agents/dln-syllabus.md`, `dunk/skills/dln/**`, and the dln-compress/dot/linear/network contracts and trigger tests. Add a resolved-decisions note to the investigation.

### Done criteria

- Local PDF/HTML and scripted HTTPS PDF/HTML complete prepare → content read → propose → decide with no filename/course/digest dispatch; byte-equivalent local/HTTPS inputs share source/prepared hashes but retain different safe acquisition provenance.
- Scripted tests cover forbidden URL forms, mixed/non-global DNS, redirect revalidation/overflow, rebinding or peer mismatch, time/header/body/encoding bounds, and MIME/magic disagreement. No test contacts the public network, and every rejection has a stable code plus byte-identical domain state.
- Both the generic ambiguous PDF and ST5201X remain ambiguous through the same pinned extractor; no layout meaning is invented.
- After successful commit, deleting the input and disabling network/extractor/proposer dependencies does not change `syllabus-content`, `context`, `validate`, repeated `rebuild`, or receipts.
- A clean Python 3.10 frozen install imports the exact direct `pypdf` pin and passes all tests; CAS content stays outside the plugin installation and is documented as canonical backup/private data.

## 3. Evidence, fresh no-mistakes gate, and existing PR update

**Depends on:** Items 1–2 complete with final serialized formats and lockfile.  
**Status:** Evidence and release preparation complete; commit/no-mistakes/PR work intentionally left to the outer orchestrator.

### Work

- Rewrite `docs/evidence/issue-3/README.md` and `reproduce.sh` around a generic PDF prepare/propose/decide/grounded-session flow, CAS verification, scripted HTTPS rejection coverage, legacy replay, offline rebuild, the frozen dependency environment, and strict plugin validation.
- Replace `approval-request.json` with `proposal-request.json` and `decision-request.json`; update `grounded-session-request.json` to cite `decision_event_id`. Generalize `render-evidence.swift`, remove the three fixture-branded ST5201X screenshots, add portable intake/decision/session screenshots, and regenerate `terminal-validation.png` from real final outputs.
- Commit implementation, fixtures, documentation, lockfile, and regenerated evidence on the existing feature branch. With a clean worktree, inspect `no-mistakes axi`, then start a **fresh** `no-mistakes axi run --intent "<full issue #3 revision intent and settled choices>"` rather than relying on the gate that produced `4e9e3d3`.
- While AXI owns an active run, inspect its findings and advance only with the displayed `no-mistakes axi respond` commands; do not edit, commit, abort, rerun, or hand-push mid-run. Escalate `ask-user` findings. After a terminal failure, follow `branch_sync`, commit fixes only after custody is returned, and start a fresh full run. Stop at `checks-passed` or `passed`.
- Let the successful pipeline push `issue-3-authoritative-syllabus` and update the existing issue-3 PR; do not open a parallel PR. Ensure the PR records legacy/rollback behavior, the exact dependency pin, evidence links, final validated commit, and CI result.

### Done criteria

- `reproduce.sh` succeeds from a clean checkout without public syllabus-network access; it verifies CAS hashes, removes the original input, and proves deterministic offline rebuild and final validation.
- Evidence contains no personal/temp paths, query secrets, or ST-branded production claim; ST5201X appears only as an adversarial test result.
- The final committed branch is clean before AXI, a fresh run reaches `checks-passed`/`passed`, and the existing PR—not a new PR—points at the validated head with green CI and updated migration/evidence notes.

## Migration and rollback notes

- New event kinds are forward-only for older Dunk versions. Rollback restores a matched pre-upgrade domain backup; never truncate or rewrite a live ledger.
- Legacy sources are labeled text-only: they retain inline extraction/assertions and historical approval citations, but the system must not claim retained original bytes or reproducible re-extraction for them.
- New source/prepared CAS is canonical backup material. Missing canonical content is an integrity failure; rebuild regenerates derived files only and must never reconstruct, download, or re-extract it.
- Any future `pypdf` upgrade requires an explicit direct-pin/lock/golden-fixture review and creates no silent reinterpretation of already prepared sources.
