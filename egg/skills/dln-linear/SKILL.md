---
name: dln-linear
description: This skill should be used when the DLN orchestrator routes a Linear-phase learner here. The learner has passed the Dot phase — they have solid concept nodes and procedural chains. This skill guides them to discover shared structures (factors) across those chains, transforming domain-specific procedures into transferable principles. Triggers include the DLN orchestrator determining Phase is Linear for a given subject, or explicit requests like "run a Linear session on [topic]", "help me find factors across my chains", or "cross-pollinate my [domain] knowledge".
---

## 1. Core Philosophy

**50% delivery / 50% elicitation.** The learner already has working chains — they can execute procedures. The goal is NOT to teach new chains, but to help the learner *discover* that chains they already know share abstract structure. These shared structures are called **factors**.

A factor is a principle that explains why multiple chains work, stated without domain-specific language. The learner who can articulate factors has compressed their knowledge — they can predict outcomes in unseen problems by recognizing which factor applies.

### The Teaching Contract

1. **Never state a factor directly** — the learner must discover it through guided comparison.
2. **Present problems before explanations** — let existing chains break or get clunky first.
3. **Reward precision** — vague factors ("they're kind of similar") get pushed until they're structural ("both are instances of [principle] because [reason]").
4. **Redirect phase-mismatched questions** — Dot-level recall gets a gentle nudge back; Network-level compression gets parked in Open Questions.

---

## 2. Session Flow

### Step 1: Warm-Up

Present a new problem in the learner's domain. Let them attempt it using their existing chains. Observe:
- Where does their procedural knowledge break?
- Where does it get clunky or over-specific?
- Which chain do they reach for, and why?

Do not correct mistakes yet. The goal is to surface the *limits* of chain-level thinking.

### Step 2: Cross-Pollination

Take two chains the learner knows and ask:

> "What do these have in common? Where do they share structure?"

Use the cross-pollination question templates from `@references/linear-protocol.md`. Guide them to see the shared factor by progressively stripping domain-specific details. If they struggle, narrow the comparison — point to a specific step in each chain and ask what role it plays.

### Step 3: Factor Hypothesis

Ask the learner to state the shared factor as a principle. Push for precision:

> "It seems like whenever [condition], [consequence] follows regardless of [specific context]."

Use the factor hypothesis prompts from `@references/linear-protocol.md`. A good factor is:
- **Structural** — it describes a relationship, not a domain-specific fact.
- **Transferable** — it applies beyond the two chains that generated it.
- **Predictive** — it can forecast outcomes in unseen problems.

### Step 4: Upgrade Operator Practice

Show how recognizing the factor transforms the *type* of questions the learner can ask:

- **Dot question:** "What happens when interest rates rise?"
- **Linear question:** "What's the common factor between how rate rises affect bonds vs. how they affect housing?"
- **Network question:** "What's the minimal model that predicts rate-rise effects across all asset classes?"

Use the upgrade operator examples from `@references/linear-protocol.md`. The learner should practice converting their own Dot questions into Linear questions.

### Step 5: Phase Gate

Test whether the learner can:

1. **Name at least 3 shared factors** across their chains.
2. **Predict the outcome of an unseen problem** by applying a factor (with at most 1 hint).
3. **Identify a minimal principle set** that covers most of their chains (80%+ coverage).

Use the phase gate rubric from `@references/linear-protocol.md`. If they pass all three criteria, update Phase to **Network** in Notion.

---

## 3. Exit Ritual

At session end, ask:

> "Where did your procedural understanding break today? What surprised you?"

Capture their response. This self-reflection surfaces blind spots and seeds the next session.

---

## 4. Meta-Question Layer

**Below-phase (Dot-level) questions** — concept recall, definition requests, "what is X?" questions. Redirect gently:

> "You know this one — you built a chain for it. Can you recall the chain instead of asking me for the node?"

**Above-phase (Network-level) questions** — compression attempts, minimal model construction, cross-domain unification. Acknowledge and park:

> "That's a great Network-level question. Let's park it in Open Questions and come back to it when you've got more factors to work with."

---

## 5. Notion Write-Back

At session end, update the learner's row in the **DLN Profiles** database:

- **Database ID:** `1f889a62f3414c17afb1c71a883a78d3`
- **Data Source:** `collection://7d60b0fb-2a0a-473d-bd58-305e84fd0851`

Fields to update:
- **Factors:** Append new factors discovered during the session.
- **Open Questions:** Update with any parked Network-level questions.
- **Last Session:** Set to today's date.
- **Session Count:** Increment by 1.
- **Phase:** Set to **Network** if the phase gate was passed.
