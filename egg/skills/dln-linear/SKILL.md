---
name: dln-linear
description: >
  This skill should be used when the DLN orchestrator routes a learner whose Phase
  is Linear, or when a user explicitly requests a Linear session. Guides factor
  discovery (50% delivery / 50% elicitation) — finding shared structures across
  procedural chains and transforming them into transferable principles. Triggers:
  DLN orchestrator determines Phase = Linear, or explicit requests like "run a
  Linear session on [topic]", "help me find factors across my chains", "cross-pollinate
  my [domain] knowledge".
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

### Step 0: Session Plan Write

Before any teaching begins, **dispatch the `dln-sync` agent** with action `plan-write` and the following plan content:

```
---

## Session [N] — [date] (Linear Phase)

### Plan
- Chains to compare: [which chains from Knowledge State will be juxtaposed]
- Target factors: [hypothesized shared structures to discover]
- Upgrade operator goals: [Dot→Linear question upgrades to practice]
- Priorities: [reinforcement needs from previous sessions]

### Progress
(populated by sync loop)
```

The agent writes the plan and returns a re-anchor payload. Teach from the returned payload.

### Sync Loop (runs at every teaching boundary)

After each of the following boundaries, **dispatch a fresh `dln-sync` agent** with action `sync`:
- After each cross-pollination comparison
- After each factor hypothesis + precision rating
- After each upgrade operator round
- Before and after the phase gate

**Dispatch payload** — include in the agent prompt:
- Progress notes to append (append-only):
```
- Cross-pollination [Chain A vs Chain B] — learner identified [shared structure / missed it]. Precision: [vague/structural/transferable].
- Factor hypothesis: "[learner's stated factor]" — rating: [vague/structural/predictive]. [Notes on precision pushback.]
- Upgrade operator: converted [Dot question] → [Linear question]. [Success/needed guidance.]
```
- Knowledge State updates: confirmed factors for `## Factors`, parked Network-level questions for `## Open Questions`
- Any queued writes from previous failed syncs

**On agent return** — use the re-anchor payload to deliver a **visible checkpoint**:

> "Quick checkpoint: we've discovered [N] factors so far — [factor names]. Next we'll compare [Chain X] with [Chain Y] to look for more shared structure."

#### Plan Adjustment

If the re-anchor payload reveals drift, include a **plan adjustment** in the next `dln-sync` dispatch:

```
### Plan Adjustment — [reason]
- Reordering: [what changed and why]
- Deferred: [what's pushed to next session]
```

#### Notion Failure Handling

If `dln-sync` returns with `Status.Write: failed`:
1. Log the intended update in-conversation as a visible checkpoint.
2. Queue the failed writes — include them in the next `dln-sync` dispatch payload. (This queue exists only in conversation context.)
3. If 3+ consecutive dispatches return failure, announce to the learner that persistence is temporarily offline. Continue with in-conversation checkpoints only. Attempt a single bulk write-back via `dln-sync` at session end.

### Step 1: Warm-Up

Present a new problem in the learner's domain. Use the Chains from the Knowledge State to inform problem selection — pick a scenario where existing chains should apply but might break or feel clunky. Let them attempt it using their existing chains. Observe:
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

#### Factor Mastery Updates

After each factor hypothesis is stated and validated, update the factor's mastery status:

| Hypothesis Quality | Status |
|-------------------|--------|
| Structural + transferable + predictive (passes all three precision checks) | `mastered` |
| Structural but domain-locked, OR transferable but vague, OR predictive on known cases but untested on novel ones | `partial` |
| Vague ("they're kind of similar"), domain-specific, or non-predictive | `not-mastered` |

Include factor mastery updates in the `dln-sync` dispatch payload:

```
- Knowledge State updates:
  - Factor "[factor statement]": status → partial. Evidence: "Hypothesis structural but domain-locked (S[N])."
```

When a `partial` factor gets refined to structural + transferable in a later round, upgrade to `mastered` and append evidence. When a previously `mastered` factor fails on an unseen problem, downgrade to `partial`.

### Step 4: Upgrade Operator Practice

Show how recognizing the factor transforms the *type* of questions the learner can ask:

- **Dot question:** "What happens when interest rates rise?"
- **Linear question:** "What's the common factor between how rate rises affect bonds vs. how they affect housing?"
- **Network question:** "What's the minimal model that predicts rate-rise effects across all asset classes?"

Use the upgrade operator examples from `@references/linear-protocol.md`. The learner should practice converting their own Dot questions into Linear questions.

### Step 5: Phase Gate

#### Pre-Gate Mastery Check

Before running the phase gate, review the factor mastery table from the latest re-anchor payload:

- **All confirmed factors** must be `mastered` or `partial` (no `not-mastered` items).
- **At least 2 factors** must be `mastered`.

If prerequisites are not met, run targeted factor refinement — take the weakest factor and re-run cross-pollination with a new chain pair that exercises it. Update mastery via `dln-sync`.

Tell the learner: "Before we test your readiness for the next level, let's sharpen a couple of your factors."

#### Gate Assessment

Test whether the learner can:

1. **Name at least 3 shared factors** across their chains.
2. **Predict the outcome of an unseen problem** by applying a factor (with at most 1 hint).
3. **Identify a minimal principle set** that covers most of their chains (80%+ coverage).

#### Gate-Driven Mastery Updates

- **Criterion 1:** Each factor named gets evidence "Gate articulation pass (S[N])." Factors the learner cannot name: evidence "Gate articulation miss (S[N])" — downgrade to `partial`.
- **Criterion 2:** Factor used for prediction: "Gate prediction [pass/fail] — [note] (S[N])."
- **Criterion 3:** Factors in the minimal set: "Gate coverage included (S[N])." Factors excluded: "Gate coverage excluded — [reason] (S[N])."

Dispatch `dln-sync` with gate mastery updates before announcing the result.

#### Pass Criteria (modified)

The learner passes only if:
1. All three gate criteria are met, AND
2. Zero factors remain at `not-mastered`, AND
3. At least 2 factors are at `mastered` status.

Use the full rubric from `@references/linear-protocol.md`. If they pass, update Phase to **Network**.

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

Most write-back happens continuously via `dln-sync` dispatches. At session end, dispatch `dln-sync` with action `session-end` including:

| Target | Field | Action |
|--------|-------|--------|
| Column property | Last Session | Set to today's date |
| Column property | Session Count | Increment by 1 |
| Column property | Phase | Set to **Network** if phase gate passed |
| Column property | Next Review | Set to computed date (see orchestrator interval rules) |
| Column property | Review Interval | Set to computed interval (see orchestrator interval rules) |
| Page body | Knowledge State | Verify Factors and Open Questions are complete |
| Page body | Current session Progress | Append final status and exit ritual response |

Database IDs are handled by the `dln-sync` agent — phase skills do not need them.
