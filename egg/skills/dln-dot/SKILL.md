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
- Read the Knowledge State from the page body. Acknowledge what the learner already knows from the Concepts and Chains sections. If empty, say so honestly: "This is a fresh start — no prior concepts recorded."
- Preview today's learning goals: 3-5 concepts you plan to cover and the chain(s) you'll build from them.

### 1a. Session Plan Write

Before any teaching begins, **dispatch the `dln-sync` agent** with action `plan-write` and the following plan content:

```
---

## Session [N] — [date] (Dot Phase)

### Plan
- Concepts: [list the 3-5 concepts you plan to cover]
- Target chains: [the causal chains you'll build from these concepts]
- Comprehension checks: [specific questions you'll ask after each batch]
- Priorities: [anything to reinforce from previous sessions, based on Knowledge State review]

### Progress
(populated by sync loop)
```

The session number is derived from the current Session Count column property + 1 (Session Count is incremented at session end, so the plan header uses `Session Count + 1`). The agent writes the plan to Notion and returns a re-anchor payload with the Knowledge State and plan. Teach from the returned payload.

### Sync Loop (runs at every teaching boundary)

After each of the following boundaries, **dispatch a fresh `dln-sync` agent** with action `sync`:
- After each concept batch (2-3 concepts) + comprehension check
- After each chain explain-back
- After each worked example
- Before and after the phase gate

**Dispatch payload** — include in the agent prompt:
- Progress notes to append (append-only, never edit existing blocks):
```
- Concept [X] — delivered, comprehension check: [pass/partial/fail]. [Brief note on learner's response.]
- Concept [Y] — delivered, comprehension check: [pass/partial/fail]. [Brief note.]
- Chain [X→Y] — built. Learner traced [correctly on first attempt / needed N hints].
```
- Knowledge State updates: newly confirmed concepts for `## Concepts`, newly built chains for `## Chains`
- Any queued writes from previous failed syncs

**On agent return** — use the re-anchor payload to deliver a **visible checkpoint** to the learner:

> "Quick checkpoint: we've covered [X] and [Y], and you showed solid understanding of [Z]. Next up is [W], which connects to what we just built."

This doubles as retrieval practice for the learner.

#### Plan Adjustment

If the re-anchor payload reveals drift from the original plan, include a **plan adjustment** in the next `dln-sync` dispatch to append:

```
### Plan Adjustment — [reason]
- Reordering: [what changed and why]
- Deferred: [what's pushed to next session]
```

Tell the learner what changed and why: "I'm adjusting our plan — we'll spend more time on [X] before moving to [Y]."

#### Notion Failure Handling

If `dln-sync` returns with `Status.Write: failed`:
1. Log the intended update in-conversation as a visible checkpoint.
2. Queue the failed writes — include them in the next `dln-sync` dispatch payload. (This queue exists only in conversation context.)
3. If 3+ consecutive dispatches return failure, announce to the learner that persistence is temporarily offline. Continue with in-conversation checkpoints only. Attempt a single bulk write-back via `dln-sync` at session end.

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

Most write-back happens continuously via `dln-sync` dispatches during the sync loop. At session end, dispatch `dln-sync` one final time with action `session-end` including:

| Target | Field | Action |
|--------|-------|--------|
| Column property | Last Session | Set to today's date |
| Column property | Session Count | Increment by 1 |
| Column property | Phase | Set to **Linear** if phase gate passed; keep **Dot** otherwise |
| Page body | Knowledge State | Verify and patch any gaps |
| Page body | Current session Progress | Append final status and exit ritual response |

Database IDs are handled by the `dln-sync` agent — phase skills do not need them.
