# Learner Profile Specification

All paths below are scoped to this plugin via `${CLAUDE_PLUGIN_DATA}`. The profile hooks migrate legacy files from `~/.local/share/claude/` once, without overwriting files already present in plugin data.

Two persistent files store cross-session learning state. Claude reads the profile every session; the ledger is read only on demand.

---

## File 1: `${CLAUDE_PLUGIN_DATA}/markets-teacher-profile.md` (Working Memory)

Read at every session start (loaded by SessionStart hook). Capped sections. Three required sections:

### Section: About Me

Learner-written. Contains goals, target firms (e.g., Goldman Sachs S&T, Citadel, Jane Street), asset class focus (e.g., rates, FX, equities, credit, commodities, derivatives), preferred level (introductory/intermediate/advanced), and timeline to interview. Claude reads but **never overwrites without asking**. Treat user edits as authoritative.

### Section: Known Weaknesses

Max 20 active entries + max 10 resolved entries (in a `### Resolved` subsection).

Each entry format:

```
- **[short label]** — [description naming specific concept, instrument, or market relationship].
  First observed: [ISO timestamp]. Last tested: [ISO timestamp]. Last failed: [ISO timestamp].
  Last clean streak start: [ISO timestamp]. Sessions since last failure: [N].
  Status: [new | recurring | improving | resolved (short-term) | resolved (long-term)]
```

#### Specificity Rule

Every weakness description **must** name a specific concept, instrument, or market relationship. Not "struggles with rates" but "confuses modified duration with Macaulay duration" or "misprices an FX forward by ignoring the interest rate differential" or "cannot derive the put-call parity relationship from no-arbitrage first principles". If a gap cannot be made specific yet, reference the session timestamp where it was observed.

#### Status Lifecycle

| Status | Meaning | Transition |
|--------|---------|------------|
| `new` | Observed once, watching | → `recurring` if seen again within 3 sessions |
| `recurring` | Observed 2+ times, actively probe | → `improving` after 2 consecutive clean sessions |
| `improving` | Streak building, monitor | → `resolved (short-term)` after 3+ consecutive clean sessions |
| `resolved (short-term)` | Clean streak of 3+, stays on active list | → `resolved (long-term)` if clean sessions span 4+ weeks (verified against ledger). Retest after 2+ week gap |
| `resolved (long-term)` | Clean streak spanning 4+ weeks, moved to Resolved subsection | Stays in Resolved. Can return to `recurring` if observed again |

**Short-term resolution**: 3+ consecutive clean sessions (no time window required). The entry stays on the active list for retesting.

**Long-term promotion**: Verify against ledger that the `Last clean streak start` spans 4+ weeks. Only then move to the Resolved subsection.

**Retest protocol**: At session start, the SessionStart hook flags `resolved (short-term)` entries that have not been tested in 2+ weeks. Claude offers these as optional retest questions.

### Section: Session History

Reverse-chronological, capped at 20 entries. Semi-structured format with a pipe-delimited metadata line (bash-parseable) followed by free-form observations:

```
### [ISO timestamp] | [topic-name] | [asset-class] | [mode] | [verdict_label]
Gaps: [semicolon-separated tags, or "none"]
Review: [ISO date]
---
[2-4 sentences of natural language observations]
```

Example:

```
### 2026-02-19T14:30 | duration-convexity | rates | learning | solved_with_minor_hints
Gaps: concept:modified_vs_macaulay_duration; concept:convexity_adjustment_sign
Review: 2026-02-22
---
Correctly derived the DV01 from first principles. Confused modified and Macaulay duration for the third time — needs a dedicated probe next session.
Convexity sign intuition (prices rise more than they fall for a given yield move) was understood once the Taylor expansion was written out.
```

The first line is trivially `cut -d'|'`-able for scripts. Observations go below the delimiter for human readability.

#### Verdict Labels

**Learning Mode:**

| Label | Meaning |
|-------|---------|
| `solved_independently` | Reached correct answer or derivation without hints |
| `solved_with_minor_hints` | Needed Tier 1-2 hints only |
| `solved_with_significant_scaffolding` | Needed Tier 3 hints or major conceptual guidance |
| `did_not_reach_understanding` | Did not produce a correct or coherent answer |

**Recall Mode:**

| Label | Meaning |
|-------|---------|
| `strong_pass` | Recalled fluently with no conceptual gaps |
| `pass` | Recalled correctly with minor hesitation |
| `borderline` | Recalled partially; some gaps remain |
| `needs_work` | Could not recall or recalled incorrectly |

**Mock Interview Mode:**

| Label | Meaning |
|-------|---------|
| `strong_hire` | Exceptional across all sections; would hire immediately |
| `hire` | Clear pass; demonstrated competence and market intuition |
| `on_the_fence` | Mixed performance; some strong areas, some significant gaps |
| `no_hire` | Did not demonstrate sufficient knowledge or market reasoning |

---

## File 2: `${CLAUDE_PLUGIN_DATA}/markets-teacher-ledger.md` (Long-Term Record)

Append-only markdown table. Never edited, never capped, never read at session start.

```markdown
# Session Ledger

| Timestamp | Session ID | Topic | Asset Class | Mode | Verdict | Gaps | Review Due |
|-----------|------------|-------|-------------|------|---------|------|------------|
| 2026-02-19T14:30 | abc123 | duration-convexity | rates | recall | pass | concept:modified_vs_macaulay_duration | 2026-02-26 |
```

**When to read the ledger:**
- Profile Review Mode ("how am I doing overall?")
- Long-term retention checks (verifying 4+ week clean spans before promoting a weakness to `resolved (long-term)`)
- Coverage analysis (identifying asset classes or topics untouched for 4+ weeks)

**Ledger is the source of truth.** If profile and ledger conflict, ledger wins. Write ledger row first, then profile entry.

---

## Rules for Claude

1. **Never fabricate history.** Only reference sessions that exist in the profile log or ledger.
2. **Specificity requirement.** Each weakness must name a specific concept, instrument, or market relationship — not a vague domain ("struggles with options") but a precise gap ("cannot explain why a long gamma position benefits from realized volatility exceeding implied volatility").
3. **Track trajectory with timestamps.** Update `Last tested`, `Last failed`, `Last clean streak start`, and `Sessions since last failure` on every relevant session.
4. **Respect user edits as authoritative.** Do not re-add removed entries. Do not overwrite About Me without asking.
5. **Dual-write order.** Write ledger row first (structured source of truth), then profile entry (elaboration). Verdict and gap tags must be identical between the two.
6. **Short-term → long-term promotion** only after verifying a 4+ week clean span against the ledger.
7. **Use profile silently to calibrate.** Do not dump contents at session start. Reference specific observations when relevant ("I notice you have confused modified and Macaulay duration before — let's make sure we probe that distinction today").

---

## ISO Timestamp Format

All timestamps use: `YYYY-MM-DDTHH:MM` (e.g., `2026-02-19T14:30`). No seconds, no timezone. Source: `Session Timestamp` from `=== SESSION METADATA ===` (injected at session start), or from `${CLAUDE_PLUGIN_DATA}/markets-session-state.md` (after compaction), or via `date +%Y-%m-%dT%H:%M` bash fallback if neither is available.

## Session ID

Extracted from hook input JSON (`session_id` field). Used in ledger rows for debugging. If unavailable, use `manual`.

---

## Update Protocol

### Timestamp Rule

Use a single Session Timestamp for all writes in a session — ledger row, profile session header, and all weakness field updates (`Last tested`, `Last failed`, `Last clean streak start`, `First observed`). Source precedence:

1. `Session Timestamp` from `=== SESSION METADATA ===` (injected at session start)
2. `Session Timestamp` from `${CLAUDE_PLUGIN_DATA}/markets-session-state.md` (after compaction)
3. **Fallback** (neither available — hook failure or manual invocation): run `date +%Y-%m-%dT%H:%M` via Bash tool to get the current time

### Update Protocol — Learning Mode

After generating study notes or a Socratic explanation, update the persistent learner profile. **Write ledger first (source of truth), then profile (elaboration).**

1. **Append row to Ledger** (`${CLAUDE_PLUGIN_DATA}/markets-teacher-ledger.md`):
   - Session Timestamp (per timestamp rule above), session_id (from `=== SESSION METADATA ===` or `markets-session-state.md`, else `manual`), topic name, asset class, mode (`learning`), verdict label (`solved_independently` / `solved_with_minor_hints` / `solved_with_significant_scaffolding` / `did_not_reach_understanding`), gaps (semicolon-separated tags, or `none`), review due date.

2. **Append to Session History** in profile (`${CLAUDE_PLUGIN_DATA}/markets-teacher-profile.md`), newest first. Enforce 20-entry cap by removing the oldest entry if needed. Use the semi-structured format:
   ```
   ### [ISO timestamp] | [topic-name] | [asset-class] | learning | [verdict_label]
   Gaps: [semicolon-separated tags, or "none"]
   Review: [ISO date]
   ---
   [2-4 sentences of natural language observations]
   ```
   Verdict and gap tags **must match** the ledger row exactly.

3. **Update Known Weaknesses**:
   - **Gap observed again** → update `Last tested`, `Last failed`, reset `Last clean streak start` to empty, set `Sessions since last failure` to 0. Status → `recurring` if was `new`, stays `recurring` if already was. All timestamps use Session Timestamp.
   - **Gap NOT observed when expected** → update `Last tested`, increment `Sessions since last failure`. If streak just started, set `Last clean streak start` to Session Timestamp. After 3+ consecutive clean sessions: mark `resolved (short-term)`. For long-term: check that `Last clean streak start` spans 4+ weeks (verify against ledger).
   - **New gap** → add with status `new`, `First observed: [Session Timestamp]`. Description **must** name a specific concept, instrument, or market relationship (not "struggles with options" but "cannot explain why gamma increases as an option approaches expiry at-the-money").
   - Enforce 20-entry active cap. If full, promote the most-resolved entry or ask the learner which to archive.

4. **Confirm** briefly — do not dump the full profile. On first session (if `[FIRST SESSION]` tag was present), show the About Me draft populated from session observations and ask the learner to correct/confirm.

### Update Protocol — Recall Mode

After the recall debrief, update the persistent learner profile. **Write ledger first, then profile.**

1. **Append row to Ledger** (`${CLAUDE_PLUGIN_DATA}/markets-teacher-ledger.md`):
   - Session Timestamp (per timestamp rule above), session_id, topic, asset class, mode (`recall`), verdict (`strong_pass` / `pass` / `borderline` / `needs_work`), gaps (semicolon-separated tags, or `none`), review due date.
   - **Review interval by verdict:** Strong Pass = previous interval x2 (minimum 7d), Pass = previous interval x1.5 (minimum 5d), Borderline = 2d, Needs Work = 1d. If no previous interval exists, use the minimums.

2. **Append to Session History** in profile (newest first, enforce 20-entry cap by removing oldest if needed):
   - Use the semi-structured format: `### [timestamp] | [topic] | [asset-class] | recall | [verdict]` followed by `Gaps:`, `Review:`, `---`, and 2-4 observation sentences.
   - Verdict and gap tags **must match** the ledger row.
   - Note trajectory vs. previous sessions on the same topic (consult ledger for older entries if needed).

3. **Update Known Weaknesses** (same rules as Learning Mode above):
   - Gap observed again → update `Last tested`, `Last failed`, reset `Last clean streak start`, status to `recurring`. All timestamps use Session Timestamp.
   - Gap NOT observed → update `Last tested`, increment `Sessions since last failure`, manage streak/resolution. All timestamps use Session Timestamp.
   - New gap → add with status `new`, `First observed: [Session Timestamp]`, must name a specific concept, instrument, or market relationship.
   - Enforce 20-entry active cap.

4. **Confirm** briefly. On first session, show About Me draft and ask learner to correct.

### Update Protocol — Mock Interview Mode

After delivering section-by-section feedback and an M6 overall verdict, update the persistent learner profile. **Write ledger first, then profile.**

1. **Append row to Ledger** (`${CLAUDE_PLUGIN_DATA}/markets-teacher-ledger.md`):
   - Session Timestamp (per timestamp rule above), session_id, topic (use the interview theme or firm type, e.g., `rates-s&t-interview`), asset class (primary asset class covered in the mock), mode (`mock_interview`), verdict mapped from M6 scoring (`strong_hire` / `hire` / `on_the_fence` / `no_hire`), gaps derived from section-by-section feedback (semicolon-separated tags naming the specific concepts where the candidate was penalized), review due date.
   - **Review interval by verdict:** Strong Hire = 14d, Hire = 7d, On the Fence = 3d, No Hire = 1d.

2. **Append to Session History** in profile (newest first, enforce 20-entry cap by removing oldest if needed):
   - Use the semi-structured format: `### [timestamp] | [topic] | [asset-class] | mock_interview | [verdict]` followed by `Gaps:`, `Review:`, `---`, and 2-4 sentences summarizing strong sections, penalized sections, and the most critical gap to address before the next session.
   - Verdict and gap tags **must match** the ledger row.

3. **Update Known Weaknesses** using gaps extracted from the section-by-section feedback:
   - Each penalized concept from the mock becomes a gap tag. Apply the same new/recurring/improving/resolved rules.
   - Sections where the candidate performed well and where a pre-existing weakness was probed without surfacing count as clean sessions for that weakness.
   - Enforce 20-entry active cap.

4. **Confirm** briefly — name the overall verdict, the top strength, and the single most critical weakness to work on next. Do not dump the full profile.
