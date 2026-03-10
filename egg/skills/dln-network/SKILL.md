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

**On agent return** — use the re-anchor payload to deliver a **visible checkpoint**:

> "Quick checkpoint: your model has been revised [N] times this session. Current compression: [word count]. It [held/broke] on [last stress-test]. Next: [what's coming]."

#### Plan Adjustment

If stress-tests reveal unexpected weaknesses, include a **plan adjustment** in the next `dln-sync` dispatch:

```
### Plan Adjustment — [reason]
- Adding stress-tests: [new edge cases to probe]
- Shifting transfer domain: [original] → [new target]
```

#### Notion Failure Handling

If `dln-sync` returns with `Status.Write: failed`:
1. Log the intended update in-conversation as a visible checkpoint.
2. Queue the failed writes — include them in the next `dln-sync` dispatch payload. (This queue exists only in conversation context.)
3. If 3+ consecutive dispatches return failure, announce to the learner that persistence is temporarily offline. Continue with in-conversation checkpoints only. Attempt a single bulk write-back via `dln-sync` at session end.

### 1. State Model

Ask the learner to state their current compressed model of the domain.

> "In 3-5 sentences, explain [domain] as you understand it now."

Record this verbatim as the **starting model**. Do not correct it yet. Do not add to it. Just capture it.

### 2. Stress-Test

Present edge cases, counterexamples, or cross-domain analogies that should break or challenge the model.

> "Your model predicts X — but what about this case where Y happens?"

Push until the model creaks. Use the stress-test generation prompts from `@references/network-protocol.md` to systematically probe boundaries between factors, test hidden assumptions, and find the simplest breaking case.

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

Produce a full summary at session end:

- **(a) Starting model** — The verbatim model from Step 1
- **(b) What broke it** — The edge cases and mismatches discovered
- **(c) Revised model** — The final compressed model
- **(d) Open questions remaining** — Gaps the learner has not yet resolved

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
| Page body | Knowledge State | Verify Compressed Model, Factors, and Open Questions reflect final state |
| Page body | Current session Progress | Append exit ritual summary (starting model, what broke, revised model, open questions) |

No Phase column update — Network is the terminal phase.

Database IDs are handled by the `dln-sync` agent — phase skills do not need them.
