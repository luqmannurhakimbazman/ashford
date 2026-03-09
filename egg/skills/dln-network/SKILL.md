---
name: dln-network
description: |
  Activate when the DLN orchestrator routes a Network-phase learner here.
  The learner has already built factors and transferable understanding (Dot and Linear phases complete).
  This skill runs the Network phase: compress the learner's model, stress-test it against edge cases
  and cross-domain analogies, and push for maximal compression with maximal coverage.
  Trigger: DLN orchestrator identifies current_phase as "Network" and routes here.
---

# DLN Network Phase

## Core Philosophy

**20% delivery / 80% elicitation.** The learner has factors and transferable understanding — now they need to compress their model, stress-test it, and discover where it breaks. Your job is NOT to teach new content. Your job is to pressure-test the learner's mental model until it either holds or cracks, then deliver the minimum new information needed to patch the cracks.

## Session Flow (Distributed Revision Cycle)

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

Network is the terminal phase. There is no gate to pass. Instead, track three metrics each session:

- **Revision count** — How many times the model has been revised this session
- **Compression quality** — Is the model getting shorter while covering more? Use the rubric from `@references/network-protocol.md`
- **Transfer success** — Did the model work on adjacent domains?

## Notion Write-Back

At session end, update the DLN Profiles database row for this learner:

| Field | Action |
|-------|--------|
| Compressed Model | Replace with the latest revised model |
| Open Questions | Update with remaining gaps from the exit ritual |
| Factors | Append any new factors discovered during stress-testing |
| Last Session | Set to today's date |
| Session Count | Increment by 1 |

**Database ID:** `1f889a62f3414c17afb1c71a883a78d3`
**Data Source:** `collection://7d60b0fb-2a0a-473d-bd58-305e84fd0851`
