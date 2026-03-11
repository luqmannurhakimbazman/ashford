---
name: dln
description: >
  This skill should be used when the user wants to learn a new domain from scratch
  using structured cognitive phases, or when they say "dln", "dln list",
  "dln reset [domain]", "learn [domain]", "teach me [domain] from zero",
  "cold-start [domain]", "start learning [domain]", "continue learning [domain]",
  "resume [domain]", "pick up [domain]", "review [domain]", or reference the
  Dot-Linear-Network framework. It orchestrates three phase skills (dln-dot,
  dln-linear, dln-network) based on the learner's current phase stored in a Notion
  database, routing them to the appropriate learning protocol for their level of
  understanding.
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
- `## Open Questions` — Unresolved gaps
- `## Weakness Queue` — Priority-ranked queue of items the learner has not mastered. Rewritten (not appended) at each teaching boundary. Derived from mastery table statuses. Used by phase skills to drive session planning.
- `## Engagement Signals` — Lightweight motivational state (momentum, consecutive struggles, last celebration, notes). Updated by phase skills at teaching boundaries.

Each of Concepts, Chains, and Factors uses a mastery tracking table with columns:
- **Status:** `not-mastered` | `partial` | `mastered`
- **Evidence:** Compact append-only log of assessment events (e.g., "Recall pass (S2), chain trace fail (S3)"), each tagged with session number.
- **Last Tested:** Date of most recent assessment event.

Mastery status is updated by phase skills at every teaching boundary via the `dln-sync` agent. The orchestrator does not interpret mastery data — it passes the full page body to the phase skill, which reads and acts on the tables.

**Session Logs** — Dated sections appended below Knowledge State by each phase skill. Contains session plan, progress notes, and plan adjustments. Old session logs are kept for audit but are NOT read back during mid-session syncs.

#### Page Body Initialization Template

When creating a new domain profile, write the skeleton from `@references/init-template.md` to the page body.

---

## 3. Orchestrator Flow

### Step 1: Parse Domain

Extract the domain name from the user's message. Examples:
- "dln options pricing" → `Options Pricing`
- "learn compiler design from scratch" → `Compiler Design`
- "dln list" → list command
- "dln reset options pricing" → reset command

If no domain is specified, ask: *"What domain would you like to learn? Give me a topic and I'll set up your learning path."*

### Step 2: Handle Special Commands

**`list`** — Query the DLN Profiles database and display all domains with their current phase, session count, last session date, and review status in a table. For each domain, compute review status:

- **Overdue** — Today is past Next Review date. Show how many days overdue in red: "⚠ 5 days overdue"
- **Due today** — Next Review is today. Show: "Due today"
- **Upcoming** — Next Review is in the future. Show: "In [N] days"
- **No data** — Next Review is empty (legacy profile). Show: "Not scheduled"

Sort the table with overdue domains first (most overdue at top), then due today, then upcoming.

Example output:

| Domain | Phase | Sessions | Last Session | Review Status |
|--------|-------|----------|-------------|---------------|
| Options Pricing | Linear | 7 | 2026-03-05 | ⚠ 4 days overdue |
| Compiler Design | Dot | 3 | 2026-03-10 | Due today |
| Immunology | Network | 12 | 2026-03-09 | In 5 days |

**`reset [domain]`** — Find the matching row. Confirm with the user before executing. Then:
1. Replace the page body with the initialization template from `@references/init-template.md` (clearing all Knowledge State and session logs)
2. Set Phase back to Dot
3. Reset Session Count to 0
4. Clear Last Session

### Step 3: Query or Create Profile

Use the Notion MCP to query the DLN Profiles database for a row matching the domain name.

**If found:** Read the current Phase, Session Count, and page body content.

**If not found:** Create a new row with:
- Domain = parsed domain name
- Phase = Dot
- Session Count = 0

Then write the page body initialization template (see Schema section above) to the new page.

Set Next Review = tomorrow's date and Review Interval = 1 for new domains.

Tell the user: *"New domain detected. Starting you in the Dot phase — we'll build your foundational concepts first."*

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

**Set Next Review = today + Review Interval (in days).**

Pass the computed Next Review and Review Interval values to the `dln-sync` agent in the `session-end` dispatch as column_updates.

### Step 4: Load Context and Route

Read the **full page body** of the domain's Notion page. Pass it to the phase skill along with the Phase and Session Count from column properties.

The phase skill ignores sections irrelevant to its phase — no phase-specific filtering at the orchestrator level.

| Phase | Route To |
|-------|----------|
| Dot | `dln-dot` skill |
| Linear | `dln-linear` skill |
| Network | `dln-network` skill |

### Step 5: Invoke Phase Skill

Pass the following context to the phase skill:
1. **Domain name**
2. **Page body content** (full page body from Step 4)
3. **Session count** (from Session Count column property — authoritative source)
4. **Page reference** (so the phase skill can write back to the page body)

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
