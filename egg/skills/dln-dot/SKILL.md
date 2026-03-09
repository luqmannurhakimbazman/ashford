---
name: dln-dot
description: >
  Dot-phase teaching skill for the DLN (Dot-Linear-Network) learning system.
  Activated when the DLN orchestrator routes a learner in the Dot phase to this skill.
  The Dot phase is for learners who know almost nothing about a domain — they need
  foundational concepts introduced, connected into causal chains, and validated
  through worked examples before advancing to Linear phase.
  Trigger: DLN orchestrator determines Phase = Dot for a domain and routes here.
---

# DLN Dot Phase — Foundational Concept Teaching

## Core Philosophy

**70% delivery / 30% elicitation.** The learner knows almost nothing — teach more than you ask, but always check comprehension. Never assume prior knowledge. Build from the ground up.

## Session Flow

### 1. Orientation

- State the domain clearly: "Today we're working on [domain]."
- Acknowledge what the learner already knows by reading their Concepts and Chains fields from the DLN Profiles database. If empty, say so honestly: "This is a fresh start — no prior concepts recorded."
- Preview today's learning goals: 3-5 concepts you plan to cover and the chain(s) you'll build from them.

### 2. Concept Delivery

Teach in batches of **2-3 concepts**. For each concept, deliver:

1. **Plain-language definition** — No jargon unless you define it inline.
2. **Concrete analogy** — Something from everyday life that maps to the concept.
3. **Why it matters** — One sentence on why this concept is important in the domain.

After each batch, run a **comprehension check** before moving on. Use questions from `@references/dot-protocol.md` comprehension check templates. Do not proceed to the next batch until the learner demonstrates understanding of the current one.

### 3. Chain Building

Connect the delivered concepts into **causal or procedural sequences**. A chain answers: "If X happens, what follows? Why?"

- Present the chain explicitly first (teaching mode).
- Then ask the learner to **explain the chain back** in their own words.
- Use chain-building prompts from `@references/dot-protocol.md`.

Example: "We covered inflation, interest rates, and bond prices. Now: if inflation rises, what happens to interest rates? And then what happens to bond prices? Walk me through it."

### 4. Worked Example

Walk through a **concrete scenario** in the domain that exercises the chain:

1. Set up the scenario with specific details.
2. Ask the learner to identify which concepts apply.
3. Trace through step by step together — the learner leads, you guide.
4. Highlight where the chain applies in practice.

Use the worked example scaffolding structure from `@references/dot-protocol.md`.

### 5. Phase Gate

Test whether the learner is ready to advance to Linear phase. The learner must demonstrate:

- **(a)** Name the core concepts without prompting (target: 5+ concepts).
- **(b)** Explain at least one causal chain clearly and correctly.
- **(c)** Trace through a **new** scenario (not the worked example) with minimal help (≤2 hints).

If they pass all three criteria, update their Phase to **Linear** in Notion.

If they fail, identify which criterion was missed, reinforce that area, and keep Phase at Dot. Note what needs revisiting in the next session.

See the full rubric in `@references/dot-protocol.md`.

## Exit Ritual

At the end of every session, ask:

> "What did you learn today? What connects to what?"

Capture their response as a comprehension signal. This self-summary reinforces retention and gives you data on what stuck.

## Meta-Question Layer

- **Below-phase questions** (already covered material): Redirect gently. "We covered that earlier — can you recall what we said about [concept]?" Use it as a retrieval practice opportunity.
- **Above-phase questions** (Linear/Network level): Acknowledge the curiosity. "Great question — that's something we'll get to once the foundations are solid." Park it in Open Questions for later.

## Notion Write-Back

At session end, update the learner's row in the **DLN Profiles** database:

| Field | Action |
|-------|--------|
| Concepts | Append new concepts learned this session |
| Chains | Append new chains built this session |
| Open Questions | Update with any parked above-phase questions |
| Last Session | Set to today's date |
| Session Count | Increment by 1 |
| Phase | Set to **Linear** if phase gate passed; keep **Dot** otherwise |

### Database Reference

- **Database ID:** `1f889a62f3414c17afb1c71a883a78d3`
- **Data Source:** `collection://7d60b0fb-2a0a-473d-bd58-305e84fd0851`
