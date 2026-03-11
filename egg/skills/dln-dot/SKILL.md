---
name: dln-dot
description: >
  This skill should be used when the DLN orchestrator routes a learner whose Phase
  is Dot, or when a user wants to learn a domain from scratch with no prior knowledge.
  Covers foundational concept delivery (70% teaching / 30% elicitation), causal chain
  building, worked examples, and phase gate assessment. Triggers: DLN orchestrator
  determines Phase = Dot, or user says "I know nothing about [domain]", "start from
  zero", "teach me the basics of [domain]".
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

#### Mastery Status Updates

After each comprehension check, update the mastery status of the concepts in the batch:

| Check Outcome | Status Update |
|---------------|---------------|
| Pass — learner paraphrases correctly, gives own example | `mastered` |
| Partial — correct direction but imprecise, or needed one clarifying question | `partial` |
| Fail — circular definition, cannot paraphrase, confuses concepts | `not-mastered` |

Include mastery updates in the `dln-sync` dispatch payload:

```
- Knowledge State updates:
  - Concept [X]: status → mastered. Evidence: "Recall pass — paraphrased correctly (S[N])."
  - Concept [Y]: status → partial. Evidence: "Recall partial — correct direction, confused mechanism (S[N])."
```

If a concept was previously `partial` and the learner demonstrates understanding in a later check (chain explain-back, worked example, or retrieval practice), upgrade to `mastered` and append evidence.

If a concept was `mastered` in a prior session but the learner fails to recall it in the current session's warm-up or chain-building, downgrade to `partial` and append evidence. This prevents false mastery from decaying recall.

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

#### Chain Mastery Updates

After each chain explain-back, update the chain's mastery status:

| Explain-Back Outcome | Status Update |
|----------------------|---------------|
| Correct direction, correct mechanism, complete — first attempt | `mastered` |
| Correct direction but missing mechanism or intermediate step, OR needed 1 correction | `partial` |
| Wrong direction, major gaps, or needed full re-teaching | `not-mastered` |

Include chain mastery in the `dln-sync` dispatch. A chain cannot be `mastered` unless ALL its constituent concepts are `mastered` or `partial`. If a concept downgrades, any chain containing it downgrades to at most `partial`.

### 4. Worked Example

Walk through a **concrete scenario** in the domain that exercises the chain:

1. Set up the scenario with specific details.
2. Ask the learner to identify which concepts apply.
3. Trace through step by step together — the learner leads, you guide.
4. Highlight where the chain applies in practice.

Use the worked example scaffolding structure from `@references/dot-protocol.md`.

### 5. Phase Gate

#### Pre-Gate Mastery Check

Before running the phase gate assessment, review the mastery table from the latest re-anchor payload. The learner must meet these **prerequisites** before the gate is attempted:

- **All core concepts** must be `mastered` or `partial` (no `not-mastered` items).
- **At least 2 chains** must be `mastered`.

If prerequisites are not met, do NOT run the phase gate. Instead:
1. Identify the `not-mastered` and `partial` items.
2. Run **targeted remediation** — re-teach the weakest item using a different analogy, then re-check comprehension.
3. Update mastery status via `dln-sync` after remediation.
4. If the learner reaches prerequisites within the same session, proceed to the gate. Otherwise, end the session with a clear note on what needs work next time.

Tell the learner: "Before we test your readiness to advance, let's make sure your foundations are solid. I noticed [concept/chain] needs some reinforcement — let's work on that."

#### Gate Assessment

Test whether the learner is ready to advance to Linear phase. The learner must demonstrate:

- **(a)** Name the core concepts without prompting (target: 5+ concepts).
- **(b)** Explain at least 2 causal chains clearly and correctly.
- **(c)** Trace through a **new** scenario (not the worked example) with minimal help (≤2 hints).

#### Gate-Driven Mastery Updates

The phase gate itself generates mastery updates:
- **Criterion (a):** Each concept the learner names unprompted gets evidence "Gate recall pass (S[N])." Concepts they miss get evidence "Gate recall miss (S[N])" — downgrade to `partial` if currently `mastered`.
- **Criterion (b):** Chains explained correctly get evidence "Gate chain pass (S[N])." Chains with errors get "Gate chain fail — [specific issue] (S[N])."
- **Criterion (c):** All concepts and chains exercised in the novel scenario get evidence "Gate scenario [pass/partial/fail] (S[N])."

Dispatch `dln-sync` with all gate mastery updates before announcing the result.

#### Pass Criteria (modified)

The learner passes only if:
1. All three criteria (a), (b), (c) are met, AND
2. Zero concepts remain at `not-mastered` status after gate updates, AND
3. At least 80% of concepts are at `mastered` status.

If criteria 1 is met but 2 or 3 is not, this is a **conditional near-pass**. Tell the learner: "You demonstrated strong understanding overall. A couple of concepts need one more round of reinforcement before we move on." Keep Phase at Dot. Next session should prioritize the `partial` items, then re-attempt the gate.

If they pass all criteria, update their Phase to **Linear** in Notion.

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
| Column property | Next Review | Set to computed date (see orchestrator interval rules) |
| Column property | Review Interval | Set to computed interval (see orchestrator interval rules) |
| Page body | Knowledge State | Verify and patch any gaps |
| Page body | Current session Progress | Append final status and exit ritual response |

Database IDs are handled by the `dln-sync` agent — phase skills do not need them.
