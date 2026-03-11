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

**On agent return** — use the re-anchor payload to prompt a **learner-generated checkpoint**. Do NOT state the summary yourself — ask the learner to produce it:

> "Quick checkpoint — before we move on, summarize where we are. What have we covered so far today, and what's the key takeaway?"

Wait for their response. Compare it against the re-anchor payload. If they miss something significant, prompt:

> "You covered the main points. One thing you didn't mention — [missed item]. Can you connect that to what you just said?"

If they nail it, confirm briefly and move on:

> "Exactly right. Let's continue."

The learner generating the summary is a retrieval event that strengthens retention. The teacher stating the summary is re-study — dramatically less effective.

#### Plan Adjustment

If stress-tests reveal unexpected weaknesses, include a **plan adjustment** in the next `dln-sync` dispatch:

```
### Plan Adjustment — [reason]
- Adding stress-tests: [new edge cases to probe]
- Shifting transfer domain: [original] → [new target]
```

#### Calibration-Driven Adjustment

When the Calibration Log shows a pattern across 2+ sessions, adjust teaching strategy:

**Overconfident learner** (mean calibration gap > +1.0):
- Increase stress-testing intensity — present harder edge cases earlier.
- Before accepting a comprehension check as "pass," ask one additional probe: "Are you sure? Walk me through your reasoning one more time."
- In the phase gate, use the harder end of the scenario spectrum.
- Never tell the learner they are overconfident. Instead, increase the difficulty until their confidence matches their ability.

**Underconfident learner** (mean calibration gap < -1.0):
- Add more reinforcement — revisit successful chains and name the learner's wins explicitly.
- After comprehension checks, say: "You got that right. That's a solid understanding."
- In worked examples, let the learner lead more — they often know more than they believe.
- Surface the pattern explicitly: "I notice you rate yourself lower than your actual performance. Your understanding is stronger than you think."

**Well-calibrated learner** (mean gap between -1.0 and +1.0):
- Proceed normally. Note in the sync payload that calibration is good.
- Periodically validate: "Your self-assessments have been accurate — that metacognitive skill will serve you well."

#### Notion Failure Handling

If `dln-sync` returns with `Status.Write: failed`:
1. Log the intended update in-conversation as a visible checkpoint.
2. Queue the failed writes — include them in the next `dln-sync` dispatch payload. (This queue exists only in conversation context.)
3. If 3+ consecutive dispatches return failure, announce to the learner that persistence is temporarily offline. Continue with in-conversation checkpoints only. Attempt a single bulk write-back via `dln-sync` at session end.

### 1. State Model (Retrieval)

Ask the learner to state their current compressed model of the domain **from memory, before reading anything**:

> "In 3-5 sentences, explain [domain] as you understand it now. Don't look back at previous sessions — just tell me what you remember."

Record this verbatim as the **starting model**. Do not correct it yet. Do not add to it.

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

### 4. Contract

Ask the learner to revise their model incorporating the new insight.

> "Now update your model. Can you make it shorter while covering more?"

Push for **compression** — fewer words, more coverage. The goal is a model that is more powerful AND more concise than the starting model. If the revised model is longer, challenge the learner to find redundancies.

### 5. Transfer Test

Present a problem from an adjacent domain and ask the learner to apply their compressed model.

> "If your model is truly general, it should work for [adjacent case] too. Does it?"

This tests whether the model captures deep structure or surface patterns. Where transfer breaks down, identify what is domain-specific vs. universal.

### 6. Exit Ritual — Distributed Revision Cycle Summary

At the end of every session, run this close:

**Part 1 — Self-Summary:**
Summarize the session: starting model, what broke, revised model, remaining open questions.

**Part 2 — Confidence Self-Assessment:**
> "Rate your confidence 1-5 on your revised model's coverage:"
> - Handles core cases in the domain: ___
> - Handles edge cases: ___
> - Transfers to adjacent domains: ___
> "Where is your model strongest? Where is it weakest?"

**Part 3 — Confusion Surfacing:**
> "What are you still confused about? What part of your model feels like it's held together with duct tape?"

Record all responses. Include confidence ratings in the `dln-sync` session-end dispatch for `## Calibration Log`. Confusion responses go into `## Open Questions`.

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
