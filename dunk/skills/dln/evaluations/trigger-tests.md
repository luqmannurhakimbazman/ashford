# DLN Skill — Trigger Tests

## SHOULD Trigger

### T1: Direct command
**Input:** "dln"
**Expected:** Activates `dln`, runs the local domain list/setup flow, and asks for a domain when needed.

### T2: Cold start
**Input:** "Teach me options pricing from zero."
**Expected:** Activates `dln`, initializes/loads a local domain, obtains an approved syllabus profile patch, and routes store stage `acquire` to `dln-dot` as Acquire/Discriminate.

### T3: Resume
**Input:** "Continue learning compiler design where I left off."
**Expected:** Activates `dln`, loads `dln-store context`, and routes solely from `state.stage`; it does not reconstruct progress from dialogue or generated Markdown.

### T4: List
**Input:** "dln list"
**Expected:** Uses the local CLI and reports local domains/revisions without requiring remote authentication.

### T5: Reset
**Input:** "dln reset fixed income"
**Expected:** Confirms first, then commits a `domain_reset` event without deleting event history or receipts.

### T6: Exam/mock
**Input:** "Run a DLN mock for fixed income."
**Expected:** Uses learner-supplied or approved exam-profile tasks as structured assessments and closes with a generated Session Receipt; if none exist, explains that automatic mock generation is deferred.

### T7: Review due
**Context:** `state.next_review_date` is due and a prior same-subject assessment exists.
**Expected:** Runs an unaided linked retrieval before cues; only positive observed delay counts as measured spacing.

## SHOULD NOT Trigger

### T8: One-off fact
**Input:** "What is the capital of Portugal?"
**Expected:** Does not activate `dln`.

### T9: Document summary
**Input:** "Summarize this report in five bullets."
**Expected:** Does not activate `dln` unless structured learning is also requested.

### T10: Generic reminder
**Input:** "Remind me to study tomorrow."
**Expected:** Does not activate `dln`.

## CONTRACT

### T11: Completion
**Context:** A persistent DLN session is ending.
**Expected:** Commits remaining evidence plus terminal `session_completed`, then presents the generated receipt as the sole canonical summary.

### T12: Persistence failure
**Context:** Local commit returns schema, recovery, or repeated stale-revision failure.
**Expected:** Stops persistent writes, reports unsaved structured events, and never falls back to prose-as-state.

### T13: Supplied ST5201X syllabus
**Input:** "Teach me ST5201X from this syllabus." The exact supported PDF is available at a readable runtime path.
**Expected:** Initializes/loads the domain, runs `ingest-syllabus`, presents the generated Syllabus Intake Receipt, collects accepted/corrected/deferred assertions, runs `approve-syllabus`, reloads `context`, and does not patch `profile.syllabus`.

### T14: Attachment without byte channel
**Input:** "Use the attached syllabus." The host exposes only transient attachment content and no readable path/bytes.
**Expected:** Says registration is unavailable, makes no grounding claim, and offers either retrying with a readable file or a separately labeled ungrounded generated curriculum.

### T15: Digest mismatch
**Context:** A readable PDF does not match the exact `st5201x-2026-v1` size/digest.
**Expected:** Reports truthful intake failure, leaves the revision unchanged, and never substitutes research/model output as document-derived.

### T16: Approved ambiguity
**Context:** ST5201X intake is approved with `st5201x.schedule.weeks_7_13_alignment` deferred.
**Expected:** Passes bounded `state.grounding`, keeps Weeks 7–13 alignment unresolved, selects tasks from citable planning topics, and records used stable assertion IDs without treating coverage as evidence.

### T17: Existing-domain registration
**Input:** "Add this ST5201X syllabus to my existing statistics domain." A readable exact fixture is available.
**Expected:** Loads `context`, ingests through the dedicated interface at the retained revision, presents the pending receipt, records the complete approval, reloads context, and only then routes teaching.

### T18: Pending update
**Context:** An existing domain has `state.grounding.status: approved_update_pending`.
**Expected:** Presents the pending receipt for approval while keeping only the prior `active_source`/`active_approval` authoritative; pending assertions are not cited or taught as settled.
