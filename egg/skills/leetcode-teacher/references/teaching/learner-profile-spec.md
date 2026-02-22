# Learner Profile Specification

Two persistent files store cross-session learning state. Claude reads the profile every session; the ledger is read only on demand.

---

## File 1: `~/.claude/leetcode-teacher-profile.md` (Working Memory)

Read at every session start (loaded by SessionStart hook). Capped sections. Three required sections:

### Section: About Me

Learner-written. Contains goals, timeline, preferred language, self-assessed level. Claude reads but **never overwrites without asking**. Treat user edits as authoritative.

### Section: Known Weaknesses

Max 20 active entries + max 10 resolved entries (in a `### Resolved` subsection).

Each entry format:

```
- **[short label]** — [description naming specific input class or code pattern].
  First observed: [ISO timestamp]. Last tested: [ISO timestamp]. Last failed: [ISO timestamp].
  Last clean streak start: [ISO timestamp]. Sessions since last failure: [N].
  Status: [new | recurring | improving | resolved (short-term) | resolved (long-term)]
```

#### Specificity Rule

Every weakness description **must** name a specific input class or code pattern. Not "struggles with edge cases" but "misses empty input check on array problems" or "forgets to handle negative indices in sliding window". If a gap can't be made specific yet, reference the session timestamp where it was observed.

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

**Retest protocol**: At session start, the SessionStart hook flags `resolved (short-term)` entries that haven't been tested in 2+ weeks. Claude offers these as optional retest problems.

### Section: Session History

Reverse-chronological, capped at 20 entries. Semi-structured format with a pipe-delimited metadata line (bash-parseable) followed by free-form observations:

```
### [ISO timestamp] | [problem-name] | [mode] | [verdict_label]
Gaps: [semicolon-separated tags, or "none"]
Review: [ISO date]
---
[2-4 sentences of natural language observations]
```

Example:

```
### 2026-02-19T14:30 | two-sum | learning | solved_with_minor_hints
Gaps: edge:empty_input; edge:duplicate_values
Review: 2026-02-22
---
Quickly identified hash map approach. Forgot empty input check (3rd occurrence).
Dict comprehension syntax noticeably improved since Feb 14.
```

The first line is trivially `cut -d'|'`-able for scripts. Observations go below the delimiter for human readability.

#### Verdict Labels

| Label | Meaning |
|-------|---------|
| `solved_independently` | Reached correct solution without hints |
| `solved_with_minor_hints` | Needed Tier 1-2 hints only |
| `solved_with_significant_scaffolding` | Needed Tier 3 hints or major guidance |
| `did_not_reach_solution` | Did not produce a working solution |

For Recall Mode, use the R7 verdict directly: `strong_pass`, `pass`, `borderline`, `needs_work`.

---

## File 2: `~/.claude/leetcode-teacher-ledger.md` (Long-Term Record)

Append-only markdown table. Never edited, never capped, never read at session start.

```markdown
# Session Ledger

| Timestamp | Session ID | Problem | Pattern | Mode | Verdict | Gaps | Review Due |
|-----------|------------|---------|---------|------|---------|------|------------|
| 2026-02-19T14:30 | abc123 | two-sum | hash_map | recall | pass | edge:empty_input | 2026-02-26 |
```

**When to read the ledger:**
- Profile Review Mode ("how am I doing?")
- Long-term retention checks (verifying 4+ week clean spans)
- Gap detection (pattern untouched for 4+ weeks)

**Ledger is the source of truth.** If profile and ledger conflict, ledger wins. Write ledger row first, then profile entry.

---

## Rules for Claude

1. **Never fabricate history.** Only reference sessions that exist in the log/ledger.
2. **Specificity requirement.** Each weakness must name a specific input class or code pattern.
3. **Track trajectory with timestamps.** Update `Last tested`, `Last failed`, `Last clean streak start`, `Sessions since last failure` on every relevant session.
4. **Respect user edits as authoritative.** Don't re-add removed entries. Don't overwrite About Me without asking.
5. **Dual-write order.** Write ledger row first (structured source of truth), then profile entry (elaboration). Verdict and gap tags must be consistent between the two.
6. **Short-term → long-term promotion** only after verifying 4+ week span against ledger.
7. **Use profile silently to calibrate.** Don't dump contents. Reference specific observations when relevant ("I notice you've struggled with empty input checks before — let's make sure we cover that").

---

## ISO Timestamp Format

All timestamps use: `YYYY-MM-DDTHH:MM` (e.g., `2026-02-19T14:30`). No seconds, no timezone. Source: `Session Timestamp` from `=== SESSION METADATA ===` (injected at session start), or from `~/.claude/leetcode-session-state.md` (after compaction), or via `date +%Y-%m-%dT%H:%M` bash fallback if neither is available.

## Session ID

Extracted from hook input JSON (`session_id` field). Used in ledger rows for debugging. If unavailable, use `manual`.

---

## Update Protocol

### Timestamp Rule

Use a single Session Timestamp for all writes in a session — ledger row, profile session header, and all weakness field updates (`Last tested`, `Last failed`, `Last clean streak start`, `First observed`). Source precedence:
1. `Session Timestamp` from `=== SESSION METADATA ===` (injected at session start)
2. `Session Timestamp` from `~/.claude/leetcode-session-state.md` (after compaction)
3. **Fallback** (neither available — hook failure or manual invocation): run `date +%Y-%m-%dT%H:%M` via Bash tool to get the current time

### Update Protocol — Learning Mode

After generating study notes, update the persistent learner profile. **Write ledger first (source of truth), then profile (elaboration).**

1. **Append row to Ledger** (`~/.claude/leetcode-teacher-ledger.md`):
   - Session Timestamp (per timestamp rule above), session_id (from `=== SESSION METADATA ===` or `leetcode-session-state.md`, else `manual`), problem name, pattern, mode (`learning`), verdict label (`solved_independently` / `solved_with_minor_hints` / `solved_with_significant_scaffolding` / `did_not_reach_solution`), gaps (semicolon-separated tags, or `none`), review due date.

2. **Append to Session History** in profile (`~/.claude/leetcode-teacher-profile.md`), newest first. Enforce 20-entry cap by removing the oldest entry if needed. Use the semi-structured format:
   ```
   ### [ISO timestamp] | [problem-name] | learning | [verdict_label]
   Gaps: [semicolon-separated tags, or "none"]
   Review: [ISO date]
   ---
   [2-4 sentences of natural language observations]
   ```
   Verdict and gap tags **must match** the ledger row exactly.

3. **Update Known Weaknesses**:
   - **Gap observed again** → update `Last tested`, `Last failed`, reset `Last clean streak start` to empty, set `Sessions since last failure` to 0. Status → `recurring` if was `new`, stays `recurring` if already was. All timestamps use Session Timestamp.
   - **Gap NOT observed when expected** → update `Last tested`, increment `Sessions since last failure`. If streak just started, set `Last clean streak start` to Session Timestamp. After 3+ consecutive clean sessions: mark `resolved (short-term)`. For long-term: check that `Last clean streak start` spans 4+ weeks (verify against ledger).
   - **New gap** → add with status `new`, `First observed: [Session Timestamp]`. Description **must** name a specific input class or code pattern (not "struggles with edge cases" but "misses empty input check on array problems").
   - Enforce 20-entry active cap. If full, promote the most-resolved entry or ask the learner which to archive.

4. **Confirm** briefly — don't dump the full profile. On first session (if `[FIRST SESSION]` tag was present), show the About Me draft populated from session observations and ask the learner to correct/confirm.

### Update Protocol — Recall Mode

After the R7 debrief, update the persistent learner profile. **Write ledger first, then profile.**

1. **Append row to Ledger** (`~/.claude/leetcode-teacher-ledger.md`):
   - Session Timestamp (per timestamp rule above), session_id (from `=== SESSION METADATA ===` or `leetcode-session-state.md`, else `manual`), problem, pattern, mode (`recall`), verdict from R7 (`strong_pass` / `pass` / `borderline` / `needs_work`), gaps (semicolon-separated tags), review due date.
   - **Review interval from R7 verdict:** Strong Pass = previous interval x2 (minimum 7d), Pass = previous interval x1.5 (minimum 5d), Borderline = 2d, Needs Work = 1d. If no previous interval exists, use the minimums.

2. **Append to Session History** in profile (newest first, enforce 20-entry cap by removing oldest if needed):
   - Use the semi-structured format: `### [timestamp] | [problem] | recall | [verdict]` followed by `Gaps:`, `Review:`, `---`, and 2-4 observation sentences.
   - Verdict and gap tags **must match** the ledger row.
   - Note trajectory vs. previous sessions (consult ledger for older entries if needed).

3. **Update Known Weaknesses** (same rules as Learning Mode above):
   - Gap observed again → update `Last tested`, `Last failed`, reset `Last clean streak start`, status to `recurring`. All timestamps use Session Timestamp.
   - Gap NOT observed → update `Last tested`, increment `Sessions since last failure`, manage streak/resolution. All timestamps use Session Timestamp.
   - New gap → add with status `new`, `First observed: [Session Timestamp]`, must name specific input class or code pattern.
   - Enforce 20-entry active cap.

4. **Confirm** briefly. On first session, show About Me draft and ask learner to correct.
