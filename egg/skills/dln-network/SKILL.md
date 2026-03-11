---
name: dln-network
description: >
  This skill should be used when the DLN orchestrator routes a learner whose Phase
  is Network, or when a user explicitly requests a Network session. Stress-tests and
  compresses the learner's mental model (20% delivery / 80% elicitation) against edge
  cases, counterexamples, and cross-domain analogies. Triggers: DLN orchestrator
  determines Phase = Network, or explicit requests like "run a Network session on
  [topic]", "stress-test my model of [domain]", "compress my understanding of [domain]".
---

# DLN Network Phase

## Core Philosophy

**20% delivery / 80% elicitation.** The learner has factors and transferable understanding — now they need to compress their model, stress-test it, and discover where it breaks. Your job is NOT to teach new content. Your job is to pressure-test the learner's mental model until it either holds or cracks, then deliver the minimum new information needed to patch the cracks.

**Compression as growth framing:** In Network phase, the learner's model will break repeatedly. This is the point. Frame every model failure as progress: "Your model just got better because now you know where it doesn't work." Track the revision count as a positive metric: "Third revision this session — each one makes your model tighter."

The learner may feel they are going backwards (their model was "fine" before stress-testing). Preempt this: "A model that has never been broken has never been tested. The breakages are where the learning happens."

## Session Flow (Distributed Revision Cycle)

### 0. Session Plan Write

Before asking for the learner's model, **dispatch the `dln-sync` agent** with action `plan-write` and the following plan content:

```
---

## Session [N] — [date] (Network Phase)

### Plan
- Starting model: [will be captured in Step 1]
- Planned stress-tests: [edge cases, counterexamples, or cross-domain analogies to probe]
- Transfer domains: [adjacent domains to test model generality]
- Open questions from last session: [carry forward from Knowledge State]
- Weakness-targeted stress-tests: [factors from Weakness Queue to prioritize in stress-testing]

### Progress
(populated by sync loop)
```

The agent writes the plan and returns a re-anchor payload. Use the Knowledge State from the payload to inform stress-test selection.

### Sync Loop (runs at every teaching boundary)

After each of the following boundaries, **dispatch a fresh `dln-sync` agent** with action `sync`:
- After state model capture (Step 1)
- After each stress-test round (Step 2)
- After each contraction attempt (Step 4)
- After the transfer test (Step 5)

**Dispatch payload** — include in the agent prompt:
- Progress notes to append (append-only):
```
- State model captured: "[verbatim model]"
- Stress-test [N]: [edge case presented] → model [held/broke]. [What was missing.]
- Contraction [N]: model revised — [word count before] → [word count after]. Coverage: [broader/same/narrower].
- Transfer test: [adjacent domain] → model [transferred successfully / broke at X].
```
- Knowledge State updates: replace `## Compressed Model` with latest revision, append new factors to `## Factors`, update `## Open Questions` with remaining gaps
- Any queued writes from previous failed syncs

**On agent return** — follow the learner-generated checkpoint, plan adjustment, calibration-driven adjustment, and Notion failure handling protocols in `@${CLAUDE_PLUGIN_ROOT}/egg/skills/dln/references/sync-protocol.md`. For Network phase, plan adjustments use this format:

```
### Plan Adjustment — [reason]
- Adding stress-tests: [new edge cases to probe]
- Shifting transfer domain: [original] → [new target]
```

### 1. State Model (Retrieval)

Ask the learner to state their current compressed model of the domain **from memory, before reading anything**:

> "In 3-5 sentences, explain [domain] as you understand it now. Don't look back at previous sessions — just tell me what you remember."

Record this verbatim as the **starting model**. Do not correct it yet. Do not add to it.

#### Model Visualization

After capturing the learner's verbal model, render it as a concept map showing the relationships between factors:

```mermaid
graph TD
    M1[Core Principle] --> M2[Factor A]
    M1 --> M3[Factor B]
    M2 --> M4[Prediction 1]
    M3 --> M4
    M3 --> M5[Prediction 2]
    M2 -.->|"tension"| M3
```

Use solid arrows for causal/supporting relationships and dotted arrows for tensions or trade-offs within the model.

> "Here's your model as a map. Does this capture what you meant? Are there connections I'm missing or connections I added that you don't actually claim?"

This externalization often reveals implicit assumptions — connections the learner intended but didn't state, or connections they'd reject when seeing them visually.

**Then compare silently** against the Compressed Model in Knowledge State:
- What principles did they retain?
- What was lost or distorted?
- Did the model get vaguer or more concrete since last session?

**Report the retrieval quality:**

> "Compared to your model from last session, you retained [X, Y, Z]. You lost [A, B]. [If distorted:] Your statement about [C] shifted — last time you said [original], now you're saying [new version]. Let's see if that shift is an improvement or drift."

Forgotten or distorted elements become the **first stress-test targets** in Step 2. This is more effective than arbitrary stress-test selection because the model's weakest points are exactly where it degraded between sessions.

### 1b. Pre-Test Prediction

After capturing the learner's model but BEFORE stress-testing, ask:

> "I'm about to throw edge cases at your model. Before I do — where do you think it will break? What's the weakest part?"

Then, after the stress-tests:

> "You predicted your model would break at [X]. It actually broke at [Y]. What does that tell you about your self-knowledge of this domain?"

If the learner correctly predicted where it would break, acknowledge this as strong metacognition. If they were wrong, use it as a teaching moment — the gap between perceived and actual weakness is itself a learning signal.

### 2. Stress-Test

Present edge cases, counterexamples, or cross-domain analogies that should break or challenge the model.

> "Your model predicts X — but what about this case where Y happens?"

Push until the model creaks. Use the stress-test generation prompts from `@references/network-protocol.md` to systematically probe boundaries between factors, test hidden assumptions, and find the simplest breaking case.

#### Emotional Calibration During Stress-Tests

Stress-tests are inherently adversarial — you are trying to break the learner's model. Balance rigor with support:

- **Before the first stress-test:** "I'm going to challenge your model now. The goal isn't to prove you wrong — it's to find where your model needs patching. Every expert model went through this."
- **After a model breaks:** "Good — you found a boundary. That edge case is exactly the kind of thing that separates surface understanding from deep understanding."
- **After multiple consecutive breaks:** "Your model is getting stress-tested hard this session. The fact that you keep finding breakages means you're probing deeper than most learners get to."

If 3+ consecutive stress-tests break the model, check engagement:
> "That was a tough stretch. Before we continue — are you seeing the pattern in what's breaking, or does it feel random? If it feels random, let's step back and look at the bigger picture."

This gives the learner agency and prevents the feeling of being battered by edge cases.

#### Weakness-First Stress-Test Selection

If the Weakness Queue contains `partial` factors, design the first stress-test to specifically target the weakest factor. The goal is to either:
- Confirm the learner has strengthened it (upgrade to `mastered`), or
- Surface the precise gap (inform further contraction/revision).

After weakness-targeted stress-tests, proceed to exploratory stress-tests as normal. This ensures each session makes progress on known weaknesses before seeking new ones.

#### Interleaving Rule: Cross-Factor Stress-Tests

Do NOT stress-test one factor exhaustively before moving to the next. Instead, alternate stress-tests across different factors and different parts of the model:

**Blocked (avoid):** Three boundary-probing questions about Factor A. Then three about Factor B. Then three about Factor C.

**Interleaved (prefer):** One boundary probe on Factor A. One assumption falsification on Factor C. One cross-domain challenge on Factor B. Return to Factor A with an assumption falsification. Move to Factor C with a minimal breaking case.

This forces the learner to constantly re-orient which part of their model is being tested, preventing the false confidence that comes from "I handled the last three Factor A questions, so Factor A is solid."

Additionally, insert **cross-factor interaction stress-tests** that require the learner to use multiple factors simultaneously:

> "Here's a case where [Factor A] and [Factor C] both apply but point in different directions. Which dominates? What does your model predict?"

These interaction tests are naturally interleaved and test the deepest level of model integration.

#### Load-Aware Stress-Test Pacing

Stress-tests can pile up cognitive load rapidly if the learner's model breaks on multiple fronts simultaneously. Manage this:

- **One stress-test at a time.** Present an edge case, let the learner fully process the mismatch and revise their model, THEN move to the next stress-test. Never stack "here's another one" before the previous mismatch is resolved.
- **If the learner's model breaks 3+ times in a row**, pause stress-testing. Ask: "Let's step back. Your model has broken on several cases. Before we continue — do you want to try revising your model now, or should we look at what these failures have in common?" This gives the learner control over pacing.
- **Alternate between stress-tests and contractions.** Don't run 5 stress-tests then 5 contractions. Run stress → contract → stress → contract. Each contraction is a consolidation point that frees working memory.

#### Factor Mastery Updates from Stress-Tests

Stress-tests implicitly test factors. When a stress-test breaks the model:
- Identify which factor(s) failed and downgrade to `partial` with evidence: "Stress-test fail — [edge case] broke [factor] (S[N])."
- When the learner patches the model and the factor holds on a subsequent stress-test, upgrade back to `mastered` with evidence: "Stress-test pass after revision (S[N])."

Include factor mastery updates in every `dln-sync` dispatch. Network phase does not add new mastery tracking for the Compressed Model itself — compression quality is tracked via the existing word count and coverage metrics.

### 3. Expand on Mismatch

When the model fails, explore the mismatch. Do NOT immediately explain the answer.

> "Why did your model predict wrong here? What's missing?"

Let the learner struggle with the gap first. Deliver new information (the 20%) **only at these precise points of model failure** — where the learner has hit a wall they cannot reason past on their own.

When the learner identifies what went wrong with their model, push for structural "why":

> "You said your model failed because [X]. But why was [X] not captured by your model in the first place? What assumption was hiding?"

This converts a surface-level model patch ("add exception for X") into a deeper revision ("the model assumed [Y], which is only true under [conditions]"). A model revised with "why" insight compresses better than one revised by exception-stacking.

#### Struggle Normalization

When the learner struggles to identify what went wrong:

- Validate the struggle: "This is hard to see. The mismatch between your model and reality is where the deepest learning happens."
- Provide a scaffold if frustration signals appear: "Let me narrow it down. Your model has [N] factors. Which one do you think is most relevant to this case?" (Reduces the search space.)
- If the learner still cannot identify the gap after 2 attempts, deliver the missing piece directly (the 20%) but frame it as: "Here's what was hiding — [explanation]. This is a subtle one that often takes multiple encounters to internalize."

### 4. Contract

Ask the learner to revise their model incorporating the new insight.

> "Now update your model. Can you make it shorter while covering more?"

Push for **compression** — fewer words, more coverage. The goal is a model that is more powerful AND more concise than the starting model. If the revised model is longer, challenge the learner to find redundancies.

#### Visual Compression

After the learner revises their model, render the new version as a concept map and compare it visually to the pre-revision map:

> "Your model went from [N] nodes and [M] connections to [N'] nodes and [M'] connections. Here's the before and after:"

Render both diagrams. The visual makes compression tangible — fewer boxes, fewer arrows, same or better coverage. If the revised model has MORE nodes than the original, the visual makes the bloat obvious without needing to argue about it.

> "Your revised model has more boxes than before. Can you merge any of these? Which two nodes could become one?"

### 5. Transfer Test

Present a problem from an adjacent domain and ask the learner to apply their compressed model.

> "If your model is truly general, it should work for [adjacent case] too. Does it?"

This tests whether the model captures deep structure or surface patterns. Where transfer breaks down, identify what is domain-specific vs. universal.

### 6. Exit Ritual — Distributed Revision Cycle Summary

Produce a full summary at session end:

- **(a) Starting model** — The verbatim model from Step 1
- **(b) What broke it** — The edge cases and mismatches discovered
- **(c) Revised model** — The final compressed model
- **(d) Open questions remaining** — Gaps the learner has not yet resolved
- **(e) Progress celebration:**
  > "Your model went from [starting word count] words to [ending word count] words while covering [more/same] ground. That's real compression — you're thinking like an expert in [domain]."

  If compression improved: "Fewer words, more power. That's exactly the trajectory."
  If compression didn't improve: "The model grew a bit this session, which is normal when you discover new edge cases. Next session we'll look for ways to merge and compress."

- **(f) Journey milestone check:**

| Milestone | Celebration |
|-----------|-------------|
| First Network session completed | "You're in the advanced phase now. Most learners never get here." |
| Model survived a transfer test | "Your model works beyond [original domain]. That means you've captured deep structure, not surface patterns." |
| 5+ revisions in a single session | "Five revisions in one session — that's intense work. Each one made your model more robust." |
| Compression ratio improved from last session | "Your compression ratio went from [old] to [new]. You're explaining more with less — that's mastery." |

- **(g) Confidence Self-Assessment:**
> "Rate your confidence 1-5 on your revised model's coverage:"
> - Handles core cases in the domain: ___
> - Handles edge cases: ___
> - Transfers to adjacent domains: ___
> "Where is your model strongest? Where is it weakest?"

- **(h) Confusion Surfacing:**
> "What are you still confused about? What part of your model feels like it's held together with duct tape?"

Record all responses. Include confidence ratings in the `dln-sync` session-end dispatch for `## Calibration Log`. Confusion responses go into `## Open Questions`.

- **(i) Update Engagement Signals** in the `session-end` dispatch to `dln-sync`. Set Momentum based on session:
  - Multiple successful stress-tests and compression improved → `positive`
  - Normal session with mixed results → `neutral`
  - 3+ consecutive model breaks with frustration signals → `fragile`

## Meta-Question Layer

Flag below-phase questions with a redirect. If the learner asks a Dot-level question (isolated fact recall) or Linear-level question (connecting two concepts), acknowledge it briefly and redirect:

> "That's a [Dot/Linear]-level question — you already have the pieces for this. Think about which of your factors applies here."

At this phase, the learner should be operating at the **model level**, not the concept level.

## Tracking (No Phase Gate)

Network is the terminal phase. There is no gate to pass. The sync loop tracks three metrics at each boundary:

- **Revision count** — How many times the model has been revised this session (visible in Progress notes)
- **Compression quality** — Is the model getting shorter while covering more? Use the rubric from `@references/network-protocol.md`. Tracked via word counts in contraction progress notes.
- **Transfer success** — Did the model work on adjacent domains? Tracked in transfer test progress notes.

## Notion Write-Back

Most write-back happens continuously via `dln-sync` dispatches. At session end, dispatch `dln-sync` with action `session-end` including:

| Target | Field | Action |
|--------|-------|--------|
| Column property | Last Session | Set to today's date |
| Column property | Session Count | Increment by 1 |
| Column property | Next Review | Set to computed date (see orchestrator interval rules) |
| Column property | Review Interval | Set to computed interval (see orchestrator interval rules) |
| Page body | Knowledge State | Verify Compressed Model, Factors, and Open Questions reflect final state |
| Page body | Current session Progress | Append exit ritual summary (starting model, what broke, revised model, open questions) |

No Phase column update — Network is the terminal phase.

Database IDs are handled by the `dln-sync` agent — phase skills do not need them.
