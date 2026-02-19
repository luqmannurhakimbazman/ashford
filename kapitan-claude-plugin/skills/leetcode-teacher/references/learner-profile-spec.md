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
