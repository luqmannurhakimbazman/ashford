---
name: dln
description: >
  This skill should be used when the user wants to learn a new domain from scratch
  using structured cognitive phases, or when they say "dln", "dln list",
  "dln reset [domain]", "learn [domain]", "teach me [domain] from zero",
  "cold-start [domain]", "start learning [domain]", "continue learning [domain]",
  "resume [domain]", "pick up [domain]", "review [domain]",
  "dln exam [domain] by [date]", "dln exam [domain] status",
  "dln mock [domain]", "dln cram [N]d [domain]", "exam mode", "exam prep",
  or reference the Dot-Linear-Network framework. It orchestrates three phase
  skills (dln-dot, dln-linear, dln-network) based on the learner's current phase
  stored in a Notion database, routing them to the appropriate learning protocol
  for their level of understanding. Also handles exam mode: metadata capture,
  Blueprint creation, priority scoring, time-horizon presets, and exam lifecycle
  management.
---

# DLN Learn — Domain-Agnostic Learning Orchestrator

A meta-learning skill that accelerates cold-start learning in any new domain using the Dot-Linear-Network (DLN) cognitive topology framework. This skill routes learners to the appropriate phase skill based on their tracked progress.

---

## 1. The DLN Framework

Learning follows three cognitive phases, each requiring a different teaching strategy:

| Phase | Mental State | Teaching Ratio | Goal |
|-------|-------------|----------------|------|
| **Dot** | Isolated facts, no connections | 70% delivery / 30% elicitation | Build concept nodes and basic causal chains |
| **Linear** | Procedural chains, domain-specific | 50% delivery / 50% elicitation | Discover shared factors across chains, build transferable understanding |
| **Network** | Compressed model, cross-domain links | 20% delivery / 80% elicitation | Stress-test, refine, and compress the learner's mental model |

The framework is domain-agnostic — it works for options pricing, compiler design, immunology, or any domain where the learner starts from zero.

---

## 2. Notion Database

All learner state is persisted in the **DLN Profiles** database in Notion (under Maekar). Database IDs are owned by the `dln-sync` agent — the orchestrator and phase skills reference the database by name only.

### Schema

#### Column Properties (queryable metadata)

| Property | Type | Purpose |
|----------|------|---------|
| Domain | title | The learning domain (e.g., "Options Pricing") |
| Phase | select | Current phase: Dot, Linear, or Network |
| Last Session | date | Timestamp of most recent session |
| Session Count | number | Total sessions in this domain (authoritative source) |
| Next Review | date | Computed review date — when this domain should next be reviewed |
| Review Interval | number | Current spacing interval in days (starts at 1, expands on successful review) |
| Exam Mode | checkbox | Whether exam mode is active for this domain |
| Exam Date | date | Target exam date (only set when Exam Mode is true) |

#### Page Body (learning content)

All learning content lives in the domain page body, not column properties. The page body has two sections:

**Knowledge State** — Persistent header updated at every teaching boundary. Contains:
- `## Concepts` — Concept nodes learned (Dot phase output)
- `## Chains` — Procedural sequences built (Dot phase output)
- `## Factors` — Shared structures discovered (Linear phase output)
- `## Compressed Model` — Latest model statement (Network phase output)
- `## Interleave Pool` — Concepts and factors eligible for interleaving (introduced in a prior session and passed initial comprehension check). Maintained by phase skills to enable cross-topic practice.
- `## Calibration Log` — Per-concept confidence ratings, gate predictions vs actual outcomes, and calibration trend over time. Used by phase skills to detect overconfidence/underconfidence and adjust teaching intensity.
- `## Load Profile` — Baseline cognitive load observations (working batch size, hint tolerance, recovery pattern) and per-session load history. Used by Dot phase for dynamic batch sizing and by all phases for load-aware pacing.
- `## Exam Metadata` — Exam configuration (date, format, duration, marks, AI policy, target score). Only populated when exam mode is active.
- `## Exam Blueprint` — Priority-scored topic map, high-yield queue, and past paper analysis. Only populated when exam mode is active.
- `## Exam Metrics` — Per-topic accuracy, time/question, retention deltas, aggregate readiness. Updated by phase skills during exam mode sessions.
- `## Question Bank` — Collected exam-style questions with metadata for mock generation.
- `## Mock History` — Results from mock exam sessions.
- `## Error Taxonomy` — Classified error patterns with frequency and remediation tracking.
- `## Past Exams` — Archive of completed exam cycles with self-reported results.
- `## Open Questions` — Unresolved gaps
- `## Weakness Queue` — Priority-ranked queue of items the learner has not mastered. Rewritten (not appended) at each teaching boundary. Derived from mastery table statuses. Used by phase skills to drive session planning.
- `## Engagement Signals` — Lightweight motivational state (momentum, consecutive struggles, last celebration, notes). Updated by phase skills at teaching boundaries.

Each of Concepts, Chains, and Factors uses a mastery tracking table with columns:
- **Status:** `not-mastered` | `partial` | `mastered`
- **Evidence:** Compact append-only log of assessment events (e.g., "Recall pass (S2), chain trace fail (S3)"), each tagged with session number.
- **Last Tested:** Date of most recent assessment event.

Mastery status is updated by phase skills at every teaching boundary via the `dln-sync` agent. The orchestrator does not interpret mastery data — it passes the extracted Knowledge State block to the phase skill, which reads and acts on the tables.

**Session Logs** — Dated sections appended below Knowledge State by each phase skill. Contains session plan, progress notes, and plan adjustments. Old session logs are kept for audit but are NOT read back during mid-session syncs.

#### Page Body Initialization Template

When creating a new domain profile, write the skeleton from `@references/init-template.md` to the page body.

---

## 3. Orchestrator Flow

### Step 1: Parse Domain

Extract the domain name and command type from the user's message. Examples:
- "dln options pricing" → standard session, domain = `Options Pricing`
- "learn compiler design from scratch" → standard session, domain = `Compiler Design`
- "dln list" → list command
- "dln reset options pricing" → reset command
- "dln exam options pricing by June 15" → exam command, domain = `Options Pricing`, date = `June 15`
- "dln exam options pricing status" → exam status command, domain = `Options Pricing`
- "dln mock options pricing" → mock exam command, domain = `Options Pricing`
- "dln cram 5d options pricing" → cram command, domain = `Options Pricing`, days = 5

If no domain is specified, ask: *"What domain would you like to learn? Give me a topic and I'll set up your learning path."*

### Step 2: Handle Special Commands

**`list`** — Query the DLN Profiles database and display all domains with their current phase, session count, last session date, and review status in a table. For each domain, compute review status:

- **Overdue** — Today is past Next Review date. Show how many days overdue in red: "⚠ 5 days overdue"
- **Due today** — Next Review is today. Show: "Due today"
- **Upcoming** — Next Review is in the future. Show: "In [N] days"
- **No data** — Next Review is empty (legacy profile). Show: "Not scheduled"

Sort the table with overdue domains first (most overdue at top), then due today, then upcoming.

For domains with Exam Mode = true, add an Exam column showing the exam date and days remaining:

Example output:

| Domain | Phase | Sessions | Last Session | Coverage | Review Status | Exam |
|--------|-------|----------|-------------|----------|---------------|------|
| Options Pricing | Linear | 7 | 2026-03-05 | 14/16 (88%) | ⚠ 4 days overdue | Jun 15 (71d) |
| Compiler Design | Dot | 3 | 2026-03-10 | 5/12 (42%) | Due today | — |
| Immunology | Network | 12 | 2026-03-09 | 20/20 (100%) | In 5 days | — |

If no syllabus exists for a domain, show "No syllabus" in the Coverage column.

**`reset [domain]`** — Find the matching row. Confirm with the user before executing. Then:
1. Replace the page body with the initialization template from `@references/init-template.md` (clearing all Knowledge State and session logs)
2. Set Phase back to Dot
3. Reset Session Count to 0
4. Clear Last Session

### Step 2a: Handle Exam Commands

Exam commands are processed after standard special commands. If the parsed command is an exam command, handle it here instead of proceeding to Step 3.

#### `dln exam [domain] by [date]`

1. Parse domain and exam date from the command.
2. Query or create profile (Step 3 logic).
3. Check if the profile already has exam metadata populated (non-empty `exam_date` in `## Exam Metadata`):
   - **If yes:** Ask the learner: *"You already have an exam set up for [domain] on [existing_date]. Do you want to (a) start a new exam cycle (archives the old one) or (b) update specific fields?"*
     - If (a): Archive current exam to `## Past Exams` (see Section 7: Exit Logic), then proceed to metadata capture.
     - If (b): Ask which fields to update and apply changes directly.
   - **If no:** Proceed to metadata capture.
4. Run **Exam Metadata Capture** (below).
5. Build **Exam Blueprint** via P0 Manual Weighting Mode (see Section 7).
6. Compute `time_horizon_preset` from days remaining (see Section 7: Time-Horizon Presets).
7. Set column properties: `Exam Mode = true`, `Exam Date = [parsed date]`.
8. Route to phase skill with exam context (Step 4).

##### Exam Metadata Capture

Prompt the learner with a structured intake form:

> "Setting up exam mode for **[domain]**. I need a few details:
>
> 1. **Exam date:** [already captured: DATE]
> 2. **Exam format:** (e.g., multiple choice, short answer, essay, problem sets, mixed)
> 3. **Duration:** (e.g., 2 hours)
> 4. **Total marks:** (e.g., 100)
> 5. **AI policy:** exactly one of `closed-book` | `open-notes` | `open-ai`
> 6. **Target score:** (e.g., 70/100, or 'pass', or 'as high as possible')
> 7. **Artifacts available?** (past papers, mark schemes, lecture slides, etc.)
>
> Upload or paste any artifacts now, or we can add them later."

Wait for the learner's response. Validate each field:

**AI Policy validation:** Must be exactly `closed-book`, `open-notes`, or `open-ai`. If the learner provides free text (e.g., "no notes allowed"), re-prompt: *"AI policy must be one of: `closed-book`, `open-notes`, or `open-ai`. Which applies?"*

**Target Score normalization:**

| Raw input pattern | target_score_numeric |
|---|---|
| `N/M` (e.g., `70/100`) | `N * (total_marks / M)` |
| `N%` or bare `N` (0-100) | `N/100 * total_marks` |
| Grade letter (`A+`/`A`/`A-`/`B+`/`B`/`B-`/`C+`/`C`/`C-`/`D`/`F`) | Mapping: A+=0.95, A=0.90, A-=0.85, B+=0.80, B=0.75, B-=0.70, C+=0.65, C=0.60, C-=0.55, D=0.50, F=0.40. Multiply by total_marks. |
| `pass` / `passing` | `0.50 * total_marks`. Confirm: *"I'm treating 'pass' as 50% — is that right?"* |
| `as high as possible` / `max` / `top mark` | `0.95 * total_marks`. Set `aspirational_target = true`. |
| Unparseable | Re-prompt: *"I couldn't parse that target. Please use a format like '70/100', '85%', 'B+', 'pass', or 'as high as possible'."* No partial write. |

Store both `target_score_raw` (the learner's verbatim input, for display) and `target_score_numeric` (normalized number in [0, total_marks], for computation).

After all fields are validated, write the metadata to `## Exam Metadata` in the KS block via a merge payload dispatched to `dln-sync`.

#### `dln exam [domain] status`

Show a simplified exam dashboard. Query the profile and compute:

1. **Days remaining:** `exam_date - today`
2. **Time horizon:** Current preset (Long/Medium/Short/Critical)
3. **Coverage:** Syllabus topics covered vs. total
4. **Top priority topics:** Top 5 topics from Exam Blueprint sorted by Priority Score (descending)
5. **Sessions completed:** Session Count value
6. **Estimated readiness:** From `## Exam Metrics` aggregate Readiness field (if populated), otherwise "Not enough data"

Format as:

> **Exam Dashboard: [domain]**
>
> | | |
> |---|---|
> | Exam date | [date] |
> | Days remaining | [N] days ([preset] horizon) |
> | Target | [target_score_raw] ([target_score_numeric]/[total_marks]) |
> | Format | [exam_format], [duration] |
> | AI policy | [ai_policy] |
> | Sessions | [session_count] |
> | Coverage | [N]/[M] topics ([%]) |
> | Readiness | [readiness] |
>
> **Top Priority Topics:**
> 1. [Topic] — Priority: [score], Current: [no-ai score]
> 2. ...
> 3. ...
> 4. ...
> 5. ...

After displaying, ask: *"Want to start a session, or adjust something?"*

#### `dln mock [domain]`

Mock exams are not yet implemented (Phase 4). Respond:

> "Mock exams will be available in a future update. For now, use regular DLN sessions with exam mode active — I'll prioritize exam-relevant topics and adapt to your exam format."

#### `dln cram [N]d [domain]`

Cram mode overrides the time horizon calculation:

1. Parse N (number of days) from the command.
2. Query or create profile (Step 3 logic).
3. Override `time_horizon_preset` based on N using the Time-Horizon Presets table (Section 7), regardless of the actual `exam_date`.
4. If the profile has no exam metadata, tell the user: *"Cram mode works best with exam mode active. Set up an exam first with `dln exam [domain] by [date]`, or I'll use cram pacing without exam-specific priorities."*
5. Route to phase skill with the overridden preset.

### Step 3: Query or Create Profile

Use the Notion MCP to query the DLN Profiles database for a row matching the domain name.

**If found:** Read the current Phase, Session Count, and page body content. Then validate the page body structure (see Step 3 validation below).

**If not found:** Create a new row with:
- Domain = parsed domain name
- Phase = Dot
- Session Count = 0

Then write the page body initialization template (see Schema section above) to the new page.

Set Next Review = tomorrow's date and Review Interval = 1 for new domains.

Tell the user: *"New domain detected. Starting you in the Dot phase — we'll build your foundational concepts first."*

#### Step 3 Validation: Page Body Structure Check

After loading an existing profile, check whether the page body contains these four core Knowledge State headers: `## Concepts`, `## Chains`, `## Factors`, `## Compressed Model`.

These four are sufficient because they are the structural headers that phase skills actively read mastery tables from. The remaining headers (`## Interleave Pool`, `## Calibration Log`, `## Load Profile`, `## Open Questions`, `## Weakness Queue`, `## Engagement Signals`, `## Syllabus`) are auxiliary — phase skills create them on first write if absent.

**If all four core headers are present:** The profile is valid. Proceed to exam validation (if applicable), then Step 3a.

**If any core header is missing:** The profile exists but predates the current DLN template (or was created outside DLN). Auto-initialize:

1. Read the current page body content.
2. Write the initialization template from `@references/init-template.md` to the page body, appending the original content under a `## Prior Notes` header at the bottom.
3. Backfill column properties, each only if that property is currently empty/missing:
   - Phase → Dot (only if empty)
   - Session Count → 0 (only if empty)
   - Next Review → tomorrow's date (only if empty)
   - Review Interval → 1 (only if empty)
4. Tell the user: *"Upgraded your [domain] profile to the current DLN format. Your previous session content is preserved under Prior Notes."*
5. Proceed to Step 3a as normal.

#### Step 3 Exam Validation

When the domain has `Exam Mode = true` (column property), the validator additionally requires `## Exam Metadata` and `## Exam Blueprint` headers in the page body.

**If both are present:** Exam profile is valid. Proceed.

**If either is missing:** The profile was created before exam mode was added or was corrupted. Run exam profile migration:

1. Dispatch `dln-sync` with action `fetch` to read the current KS block.
2. Identify which exam sections are missing (`## Exam Metadata`, `## Exam Blueprint`, `## Exam Metrics`, `## Question Bank`, `## Mock History`, `## Error Taxonomy`, `## Past Exams`).
3. For each missing section, append its empty skeleton (from `@references/init-template.md`) at the canonical position within the KS block.
4. Dispatch `dln-sync` with action `replace-ks-only` to write the migrated KS.
5. Tell the learner: *"Upgraded your [domain] profile with exam-prep sections. Your existing progress is preserved."*

#### KS Boundary Markers

The Knowledge State block is wrapped in `<!-- KS:start -->` / `<!-- KS:end -->` HTML comment markers. These markers are managed by the `dln-sync` agent — the orchestrator does not add, remove, or check for them. If a profile is missing markers (pre-marker profiles), `dln-sync` will add them automatically on its first sync operation.

### Step 3a: Review Check

After loading the profile, compute the review status by comparing today's date against the Next Review column value.

**If the domain is overdue or due today:**

1. Inform the learner:

> "It's been [N] days since your last session on [domain]. Your Next Review was [date] — you're [N days] overdue. Before we continue with new material, let's do a quick retrieval warm-up to see what's stuck."

2. Route to the phase-appropriate **Review Protocol** (below) BEFORE routing to the phase skill for new teaching.

3. After the review protocol completes, route to the phase skill as normal (Step 4).

**If the domain is not due for review:** Proceed directly to Step 4 (no review needed).

**If the domain has never been reviewed (Next Review is empty):** This is a legacy profile. Set Next Review = today and Review Interval = 1, then proceed to Step 4. The review system activates from the next session onward.

#### Review Protocol

The review protocol runs inside the orchestrator, before the phase skill is invoked. It takes 3-8 minutes depending on phase.

**Dot Phase Review:**
- Ask the learner to list all concepts they remember from this domain (unprompted, no hints).
- Ask them to trace one causal chain from memory.
- Compare their recall against the Concepts and Chains in Knowledge State.
- Score: count recalled vs. total. Note which concepts were forgotten.
- If recall < 50%: warn the learner that significant decay has occurred. Recommend spending this session on reinforcement rather than new material. Pass `review_results` to the phase skill so it can prioritize re-teaching forgotten concepts.
- If recall >= 50%: acknowledge what they remembered, note gaps, and proceed to new material. Pass `review_results` to the phase skill for priority adjustment.

**Linear Phase Review:**
- Ask the learner to name the factors they've discovered so far (unprompted).
- Ask them to pick one factor and explain which chains it connects and why.
- Compare against the Factors section in Knowledge State.
- Score: count recalled factors vs. total, check if explanations are structural (not just names).
- Same threshold logic as Dot: < 50% triggers reinforcement recommendation.

**Network Phase Review:**
- Ask the learner to state their compressed model from memory (no looking at notes).
- Compare against the Compressed Model in Knowledge State.
- Score qualitatively: did they capture the core principles? What was lost?
- For Network phase, there is no "reinforcement" redirect — instead, the forgotten elements become the first stress-test targets in the session.

#### Interval Computation

After each session completes (regardless of whether a review protocol ran), compute the next review interval using these rules:

**Base intervals by phase:**
- Dot phase: intervals expand as 1 → 2 → 4 → 7 → 14 → 30 days
- Linear phase: intervals expand as 1 → 3 → 7 → 14 → 30 days
- Network phase: intervals expand as 2 → 7 → 14 → 30 → 60 days

**Adjustment rules:**
- If the review protocol ran and recall was >= 70%: advance to the next interval in the sequence. Set Review Interval to the new value.
- If the review protocol ran and recall was 50-69%: repeat the current interval (no advancement). Keep Review Interval the same.
- If the review protocol ran and recall was < 50%: reset the interval to the first value in the sequence for the current phase.
- If no review protocol ran (domain was not overdue): advance to the next interval in the sequence.
- If the learner's phase changed this session (phase gate passed): reset to the first interval of the NEW phase.

**Exam mode interval cap:** If `Exam Mode = true` and `days_remaining < review_interval`, cap the interval at `max(1, days_remaining / 2)`. This ensures reviews happen before the exam, not after.

**Set Next Review = today + Review Interval (in days).**

Pass the computed Next Review and Review Interval values to the `dln-sync` agent in the `replace-end` dispatch as column_updates.

### Step 3b: Syllabus Check

After loading the profile and running any review check, inspect the `## Syllabus` section in the page body.

**If the `## Syllabus` section is empty or contains only the placeholder goal:**

1. Tell the user: "No syllabus exists for this domain yet. Let me research and generate one based on your learning goal."
2. Spawn the `dln-syllabus` subagent via the **Agent tool**, passing in the prompt:
   - Domain name
   - The user's original goal prompt (from when they first created this domain, or ask them now)
   - Page ID for writing the syllabus to Notion
3. The subagent runs in its own context window — it does all web search, context7 lookups, and domain research there. Only the final topic list returns to the main session. This keeps the teaching context clean.
4. When the subagent returns, present the topic list to the user for review:

> "Here's the syllabus I've generated for **[domain]** based on your goal:
>
> - Topic A
> - Topic B
> - Topic C
> - ...
>
> **[N] topics total.** Add, remove, or edit anything before we start. You can also edit this anytime in Notion."

5. Apply the user's edits. If they request changes, update the `## Syllabus` section in Notion directly from the orchestrator (this is cheap — just text manipulation, no research needed).
6. If the subagent fails or the user declines, proceed without a syllabus. The orchestrator skips coverage reporting. The syllabus can be generated in a future session.

**If the `## Syllabus` section has topics:**

Compute and display coverage stats before routing to the phase skill:

> **Syllabus Progress:**
> - Coverage: [N]/[M] topics covered ([percentage]%)
> - Mastery: [X] mastered, [Y] partial, [Z] not-mastered
> - Uncovered: [list of unchecked topics with no concepts yet]

A topic is *covered* when at least one concept in `## Concepts` has a matching `Syllabus Topic` column value. A topic is *checked off* when all related concepts are `mastered`.

Pass the syllabus content to the phase skill alongside the page body (it's already in the page body, so no additional passing is needed).

### Step 3c: Exam Exit Check

After loading the profile, if `Exam Mode = true`, check whether the exam date has passed:

**If `today >= exam_date + 1 day`:** The exam is over. Run the exit flow:

1. Archive current exam metadata to `## Past Exams` by appending a row to the Past Exams table with: Exam Date, Format, Total Marks, Target (raw), Target (numeric), Mock Count (from Mock History), Best Mock Score, Self-Reported Result (empty — will be filled below), Archived date.
2. Reset all exam-specific sections to their empty skeletons (from `@references/init-template.md`): `## Exam Metadata`, `## Exam Blueprint`, `## Exam Metrics`, `## Question Bank`, `## Mock History`, `## Error Taxonomy`.
3. Set column properties: `Exam Mode = false`, clear `Exam Date`.
4. Dispatch `dln-sync` with action `replace-ks-only` to persist the changes.
5. **Preserve all learning progress** — `## Concepts`, `## Chains`, `## Factors`, `## Compressed Model`, and all other non-exam sections remain untouched.
6. Ask the learner: *"Your [domain] exam was [date]. How did it go? What score did you get?"*
7. Store the learner's self-reported result in the `Self-Reported Result` column of the newly archived Past Exams row (via merge payload to `dln-sync`).
8. Tell the learner: *"Exam mode deactivated for [domain]. Your learning progress is preserved — you can continue regular DLN sessions or set up a new exam anytime."*

**If the exam has not passed:** Proceed normally.

### Step 4: Load Context and Route

Read the **full page body** of the domain's Notion page, then **extract only the Knowledge State block** to pass to the phase skill. This prevents old session logs from consuming context tokens.

**Extraction rule:** Find the `<!-- KS:start -->` and `<!-- KS:end -->` boundary markers. Extract everything between them (inclusive of markers). Discard everything after `<!-- KS:end -->` — that's session logs from prior sessions.

If the markers are missing (pre-marker profile), pass the full page body as-is and let `dln-sync` add markers on its first sync.

The extracted KS block includes the `## Syllabus` section. Phase skills read it directly — no separate syllabus parameter is needed.

#### Exam-Aware Routing

When the domain has `Exam Mode = true`, before routing to the phase skill, compute and attach exam context:

1. Compute `days_remaining = exam_date - today`.
2. Look up `time_horizon_preset` from the Time-Horizon Presets table (Section 7). If a `dln cram` override is active, use that instead.
3. Read exam fields from `## Exam Metadata`: `exam_format`, `ai_policy`, `target_score_raw`, `target_score_numeric`.
4. Check re-prioritization cadence: if `sessions_since_reprioritization >= 3`, run re-prioritization (see Section 7) before routing.

The exam context is passed to the phase skill alongside the standard context (Step 5).

| Phase | Route To |
|-------|----------|
| Dot | `dln-dot` skill |
| Linear | `dln-linear` skill |
| Network | `dln-network` skill |

### Step 5: Invoke Phase Skill

Pass the following context to the phase skill:
1. **Domain name**
2. **Knowledge State block** (extracted KS block from Step 4, not the full page body)
3. **Session count** (from Session Count column property — authoritative source)
4. **Page reference** (so the phase skill can write back to the page body)

**When exam mode is active, additionally pass:**
5. **exam_mode:** `true`
6. **days_remaining:** computed days until exam
7. **time_horizon_preset:** one of `Long`, `Medium`, `Short`, `Critical`
8. **exam_format:** the exam format string
9. **ai_policy:** one of `closed-book`, `open-notes`, `open-ai`
10. **target_score_raw:** learner's verbatim target (display)
11. **target_score_numeric:** normalized target in [0, total_marks] (computation)

Use the Skill tool to invoke the appropriate phase skill (`dln-dot`, `dln-linear`, or `dln-network`).

After the phase skill completes, no additional write-back is needed — the phase skill handles all Notion persistence.

---

## 4. Phase Transition Rules

Phase transitions are handled by the phase skills themselves:

- **Dot → Linear:** When the learner passes the Dot phase gate (can name concepts, explain causal chains, trace through a scenario). The `dln-dot` skill updates Phase to Linear.
- **Linear → Network:** When the learner passes the Linear phase gate (can name shared factors, predict unseen problems, identify minimal principle set). The `dln-linear` skill updates Phase to Network.
- **Network** is terminal — no further phase transitions. The skill tracks revision count and compression quality.

---

## 5. Error Handling

- **Notion unavailable:** Tell the user Notion is unreachable and offer to run the session without persistence (phase skill still works, just no state saved).
- **Multiple domain matches:** If the query returns multiple rows, show them and ask the user to clarify.
- **Phase skill not found:** This shouldn't happen in normal operation. If it does, report the error and suggest the user check their plugin installation.
- **Notion fails mid-session:** Phase skills delegate all Notion I/O to the `dln-sync` agent. If `dln-sync` returns with a failure status, the phase skill logs the intended update in-conversation, queues failed writes for the next dispatch, and falls back to in-conversation-only tracking if 3+ consecutive dispatches fail. See phase skill instructions for details.
- **Exam metadata validation failure:** If exam metadata fields fail validation (e.g., invalid AI policy, unparseable target score), re-prompt for the specific field. Never write partial exam metadata — all required fields must be valid before persisting.

---

## 6. Motivational Architecture

All DLN phase skills embed motivational design into their teaching. This is not a separate system — it is woven into every interaction. The principles:

1. **Growth mindset framing:** Attribute success to effort and strategy, not innate ability. "You worked through that" not "You're smart." Attribute struggle to the difficulty of the material or the need for a different approach, never to the learner's capacity.

2. **Visible progress:** At every checkpoint, tell the learner where they are relative to where they started. Use concrete counts: "You've mastered 8 of 12 concepts" or "Your model is 40% more compressed than last session." Progress must be tangible, not just "you're doing well."

3. **Frustration detection and response:** Monitor for frustration signals (see phase skill instructions). When detected, intervene immediately — do not push through. Simplify, provide a quick win, then rebuild momentum.

4. **Desirable difficulty calibration:** Learning should feel challenging but achievable. If the learner breezes through everything, increase difficulty. If they hit a wall, reduce scope and provide scaffolding. The target emotional state is "stretched but not overwhelmed."

5. **Celebration at milestones:** Phase transitions, mastery achievements, and session count thresholds are celebrated explicitly. Not with empty praise — with specific acknowledgment of what the learner accomplished and what it means.

The `## Engagement Signals` section in the Knowledge State persists motivational context between sessions so the next session can calibrate its tone appropriately.

**Momentum time-decay rule:** If 7+ days have elapsed since the last session, reset Momentum to `neutral` regardless of its stored value. A `fragile` state from a bad session should not persist indefinitely — after a week, the learner has had enough distance that opening with fragile calibration feels mismatched. The phase skill reads Last Session from the profile and applies this rule before using the stored Momentum value.

---

## 7. Exam Mode

Exam mode transforms DLN from open-ended learning into targeted exam preparation. The orchestrator owns all exam lifecycle logic — phase skills receive exam context and adapt their teaching accordingly (but the orchestrator controls metadata, Blueprint, priorities, and exit).

### Time-Horizon Strategy Presets

The time horizon determines the balance between deep understanding and drill-heavy practice. It is computed automatically from `days_remaining = exam_date - today`.

| Horizon | Days | Understanding | Drills | Transfer |
|---------|------|---------------|--------|----------|
| **Long** | > 28 | 50% | 30% | 20% |
| **Medium** | 14-28 | 35% | 45% | 20% |
| **Short** | 7-13 | 20% | 60% | 20% |
| **Critical** | < 7 | 10% | 70% | 20% |

Selection is automatic based on `days_remaining`. The `dln cram` command can override this — e.g., `dln cram 5d options pricing` forces the Critical preset regardless of actual exam date.

Phase skills use these percentages to allocate session time between concept teaching (Understanding), practice problems (Drills), and cross-topic application (Transfer).

### Exam Blueprint Creation (P0 Manual Weighting Mode)

After metadata capture, the orchestrator builds an Exam Blueprint by collecting topic weights from the learner. The Blueprint lives in `## Exam Blueprint > ### Topic Map`.

**Offer the learner three flows:**

> "Now let's set up your exam Blueprint — this tells me which topics to prioritize. Pick a mode:
>
> 1. **Quick weighting** — I'll walk through each syllabus topic and ask you to rate it (takes 2-3 min)
> 2. **Bulk paste** — paste a pre-filled table with your weights
> 3. **Equal defaults** — all topics get equal priority (you can refine later)"

#### Flow 1: Inline Quick Weighting

For each topic in the syllabus, ask:

> **[Topic name]:**
> - Marks weight? (`high` / `medium` / `low`) — how many marks does this typically carry?
> - Exam frequency? (`every exam` / `most exams` / `some exams`) — how often does this appear?

Map responses:
- Marks weight: high = 0.15, medium = 0.08, low = 0.03
- Exam frequency: every exam = 0.9, most exams = 0.5, some exams = 0.2

Set defaults for auto-computed fields:
- `TransferLeverage = 1` (refined in Phase 2+)
- `HoursToReachFloor = 1.0` (refined in Phase 2+)
- `CurrentNoAIScore = 0` (updated by phase skills after assessments)

#### Flow 2: Bulk Paste

Accept a pasted table in this format:

```
| Topic | Marks Weight | Exam Frequency |
|-------|-------------|----------------|
| Topic A | high | every exam |
| Topic B | low | some exams |
```

Parse and map values using the same conversion as Flow 1.

#### Flow 3: Equal Defaults

For N syllabus topics:
- `MarksWeight = 1/N` for all topics
- `ExamFrequency = 0.5` for all topics
- `TransferLeverage = 1` for all topics
- `HoursToReachFloor = 1.0` for all topics
- `CurrentNoAIScore = 0` for all topics

#### Priority Score Computation

After collecting weights, compute the Priority Score for each topic:

```
Priority = (MarksWeight * ExamFrequency * (1 - CurrentNoAIScore) * TransferLeverage) / HoursToReachFloor
```

Sort topics by Priority Score descending. The top topics populate `### High-Yield Queue`.

Write the Blueprint to Notion via merge payload dispatched to `dln-sync`, populating `exam_blueprint.topic_map` with all computed values.

### Re-Prioritization Cadence

Every 3 sessions (tracked via `sessions_since_reprioritization` in `## Exam Metadata`), the orchestrator re-computes Priority Scores:

1. Read current `CurrentNoAIScore` values from `## Exam Metrics` (updated by phase skills).
2. Re-compute Priority for each topic using the formula above.
3. Re-sort and update `### High-Yield Queue`.
4. Display the updated top 5 topics to the learner:

> **Priority re-check (every 3 sessions):**
>
> Your top 5 topics to focus on:
> 1. [Topic] — Priority: [score] (was: [old_score])
> 2. ...
>
> Any topics you want to override?

5. Reset `sessions_since_reprioritization = 0` in the merge payload.
6. Dispatch `dln-sync` with the updated Blueprint.

### Exam Mode Exit Logic

Exam exit is triggered automatically when `today >= exam_date + 1 day` (checked in Step 3c). The full exit flow:

1. **Archive** — Copy current exam metadata to `## Past Exams` table row.
2. **Reset** — Clear all exam-specific sections to empty skeletons.
3. **Update columns** — Set `Exam Mode = false`, clear `Exam Date`.
4. **Persist** — Dispatch `dln-sync` with `replace-ks-only` action.
5. **Preserve progress** — All learning sections (`## Concepts`, `## Chains`, `## Factors`, `## Compressed Model`, mastery tables, etc.) remain intact.
6. **Debrief** — Ask the learner how the exam went and record their self-reported score.

See Step 3c for the detailed implementation.
