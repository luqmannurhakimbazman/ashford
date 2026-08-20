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

### T13: Generic local PDF syllabus
**Input:** "Teach me from this syllabus." A readable text-layer PDF path is available.
**Expected:** Runs `prepare-syllabus`, reads verified `syllabus-content`, obtains bounded proposals, runs `propose-syllabus`, collects a complete learner decision, runs `decide-syllabus`, reloads context, and does not patch `profile.syllabus`.

### T14: Attachment without byte channel
**Input:** "Use the attached syllabus." The host exposes only a transient preview and no readable byte channel.
**Expected:** Makes no grounding claim and offers a readable local path, explicit HTTPS consent, or a separately labeled ungrounded curriculum.

### T15: HTTPS consent and redirects
**Input:** "Use https://example.edu/syllabus.html." No network consent has been given.
**Expected:** Requests explicit network consent; it requests separate redirect consent before `--allow-redirects` and never uses ambient proxy/auth or a query-bearing URL.

### T16: Ambiguous generic layout
**Context:** Prepared text contains an ambiguous two-column week/milestone layout.
**Expected:** Proposes it as `ambiguous`, refuses acceptance until corrected/deferred/rejected, and never invents layout meaning.

### T17: Existing-domain HTML registration
**Input:** "Add this HTML syllabus to my existing domain." A readable path is available.
**Expected:** Loads `context`, completes prepare/content/propose/decide at retained revisions, ignores scripts/styles/templates/subresources, reloads context, and only then routes teaching.

### T18: Pending authoritative update
**Context:** `state.grounding.status` is `approved_update_pending`.
**Expected:** Presents the pending proposal/decision work while keeping only prior `active_source`/`active_decision` authoritative; pending proposals and supplements are not cited or taught as settled.
