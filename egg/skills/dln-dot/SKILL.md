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

#### Weakness-Driven Priority Setting

Before previewing today's goals, check the **Weakness Queue** from the page body:

- **If the queue is non-empty:** The top 1-2 items from the queue become the session's primary targets. Preview today's plan as: "Last time, [item] gave you trouble — we're going to nail that first today, then move on to new material."
- **If the queue is empty:** Plan normally — preview 3-5 new concepts and target chains.

The session plan must allocate time as follows:
- **Queue non-empty:** First third of the session is remediation of queued items. Remaining two-thirds is new content.
- **Queue empty:** Full session is new content.

This ensures weakness remediation always happens before new content delivery, preventing knowledge gaps from compounding.

### 1a. Session Plan Write

Before any teaching begins, **dispatch the `dln-sync` agent** with action `plan-write` and the following plan content:

```
---

## Session [N] — [date] (Dot Phase)

### Plan
- Weakness remediation: [top 1-2 items from Weakness Queue, or "none — queue empty"]
- Remediation strategy: [for each item: different analogy / break into sub-concepts / micro-example + re-check]
- New concepts: [list the 2-5 new concepts, adjusted for time spent on remediation]
- Target chains: [the causal chains you'll build]
- Comprehension checks: [specific questions]

### Progress
(populated by sync loop)
```

The session number is derived from the current Session Count column property + 1 (Session Count is incremented at session end, so the plan header uses `Session Count + 1`). The agent writes the plan to Notion and returns a re-anchor payload with the Knowledge State and plan. Teach from the returned payload.

### 1b. Retrieval Warm-Up

**Skip this step on the very first session** (Session Count = 0, Knowledge State is empty). For all subsequent sessions, run this BEFORE any new concept delivery.

**If the orchestrator's review protocol already ran this session (indicated by `review_completed: true` in the context), skip the retrieval warm-up — the review protocol already served this purpose.**

The purpose is not assessment — it is a learning event. Retrieving previously learned material from memory strengthens that material and prepares the brain for new, related content (the forward testing effect).

#### Protocol

1. **Concept free recall** — Ask the learner to list every concept they remember from previous sessions, without looking at notes or prompts:

> "Before we start today's new material, I want you to recall everything you can from our previous sessions. List all the key concepts you remember about [domain]. Take your time — there's no penalty for forgetting."

2. **Wait for their response.** Do not prompt or hint. Let silences sit. The effort of retrieval is the learning event.

3. **Chain recall** — Pick one chain from the Knowledge State (preferably one that connects to today's planned material) and ask the learner to trace it:

> "Good. Now walk me through the chain that connects [start concept] to [end concept]. What's the causal sequence?"

4. **Score silently.** Compare their recall against the Knowledge State:
   - Count concepts recalled vs. total in Knowledge State
   - Note which concepts were forgotten (these become reinforcement priorities)
   - Rate chain recall: accurate (all links correct), partial (direction right but gaps), or failed (major errors or can't attempt)

5. **Respond with targeted feedback** — but do NOT re-teach yet:

> "You recalled [N] of [M] concepts. You missed [list]. Your chain from [X] to [Y] was [accurate/partial/incomplete]. Let's keep those gaps in mind — we'll reinforce them as we go today."

6. **Adjust session plan** — If recall < 50%, or if chain recall failed:
   - Move the first batch of new concepts to the end of the session
   - Spend the first teaching segment re-delivering forgotten concepts using a *different* analogy than the original (re-reading the same explanation does not help — the new analogy forces deeper processing)
   - Include a plan adjustment in the next `dln-sync` dispatch

7. **Dispatch `dln-sync`** with action `sync` after the retrieval warm-up completes. Include retrieval results in the progress notes:

```
- Retrieval warm-up: [N/M] concepts recalled. Forgotten: [list]. Chain [X→Y]: [accurate/partial/failed]. Retrieval score: [N%].
- Session adjustment: [none / reinforcing X, Y before new material / re-teaching Z with new analogy]
```

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
- Weakness Queue rebuild: [full updated queue reflecting mastery changes this boundary]
- Any queued writes from previous failed syncs

**On agent return** — use the re-anchor payload to prompt a **learner-generated checkpoint**. Do NOT state the summary yourself — ask the learner to produce it:

> "Quick checkpoint — before we move on, summarize where we are. What have we covered so far today, and what's the key takeaway?"

Wait for their response. Compare it against the re-anchor payload. If they miss something significant, prompt:

> "You covered the main points. One thing you didn't mention — [missed item]. Can you connect that to what you just said?"

If they nail it, confirm briefly and move on:

> "Exactly right. Let's continue."

The learner generating the summary is a retrieval event that strengthens retention. The teacher stating the summary is re-study — dramatically less effective.

#### Plan Adjustment

If the re-anchor payload reveals drift from the original plan, include a **plan adjustment** in the next `dln-sync` dispatch to append:

```
### Plan Adjustment — [reason]
- Reordering: [what changed and why]
- Deferred: [what's pushed to next session]
```

Tell the learner what changed and why: "I'm adjusting our plan — we'll spend more time on [X] before moving to [Y]."

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

Teach in batches of **1-4 concepts**, dynamically sized based on concept complexity and learner performance. The default starting batch size is **2**. Adjust using the rules below.

#### Estimating Concept Complexity

Before each batch, classify each concept's **element interactivity** — how many elements the learner must hold in working memory simultaneously to understand the concept:

| Complexity | Element Interactivity | Examples | Default Batch Size |
|------------|----------------------|----------|-------------------|
| **Low** | 1-2 elements, can be understood in isolation | stock, bond, dividend, variable, function | 3-4 concepts per batch |
| **Medium** | 3-4 elements, requires relating to other concepts | interest rate (relates to inflation, bond price, lending), recursion (relates to base case, call stack, return) | 2 concepts per batch |
| **High** | 5+ elements, requires simultaneous manipulation of multiple interacting pieces | options delta (requires: underlying price, strike, time, volatility, and their interactions), backpropagation (requires: loss, chain rule, gradients, weights, layers) | 1 concept per batch |

When a batch contains concepts of mixed complexity, size the batch to the MOST complex concept in it. A batch with one High concept is a single-concept batch, even if the other planned concepts are Low.

#### Load Monitoring Signals

Watch for these **overload indicators** during and after each batch:

| Signal | Severity | What It Means |
|--------|----------|---------------|
| Failed comprehension check (first attempt) | Moderate | May be a gap, not necessarily overload. Re-teach with different analogy. |
| Failed comprehension check (second attempt after re-teach) | High | Likely overloaded. The concept or batch is too complex. |
| Learner asks to repeat the explanation | Moderate | Working memory full. Slow down. |
| Learner confuses current concept with a previous one | High | Too many similar items in working memory. Separate them. |
| Learner gives circular or verbatim-repeat answers | High | Not processing — just echoing. Overloaded. |
| Learner stops attempting and says "I don't know" | High | Shut down. Immediate intervention needed. |
| Learner answers quickly and correctly | Low (positive) | Capacity available. Can increase batch size. |
| Learner asks "above-phase" questions | Low (positive) | Engaged and ahead. Can increase batch size. |

#### Overload Response Protocol

When 2+ High-severity signals appear in a single batch:

1. **Stop the current batch immediately.** Do not push through.
2. **Acknowledge without blame:** "Let's slow down — I threw too much at you at once."
3. **Reduce batch size by 1** for the remainder of the session (minimum: 1 concept per batch).
4. **Add a worked micro-example** for the concept that caused overload — a 2-3 sentence scenario that exercises just that one concept in isolation.
5. **Re-attempt the comprehension check** after the micro-example.
6. **Log the adjustment** in the next `dln-sync` dispatch as a plan adjustment.

When the learner shows positive signals across 2+ consecutive batches, increase batch size by 1 (maximum: 4).

#### Interaction with Worked Examples

When batch size is reduced to 1 due to overload:
- Add a worked example after EVERY concept, not just after chains are built.
- Use the "faded example" progression: first example is fully worked, second has one step for the learner to fill in, third has two steps for the learner.
- This increases germane load (productive processing) while reducing extraneous load (unnecessary complexity).

When the learner is performing well (batch size 3-4):
- Worked examples can be more complex and combine multiple concepts.
- The learner should lead the walkthrough with minimal guidance.
- Skip the faded progression — go straight to learner-led examples.

For each concept, deliver:

1. **Plain-language definition** — No jargon unless you define it inline.
2. **Concrete analogy** — Something from everyday life that maps to the concept.
3. **Why it matters** — One sentence on why this concept is important in the domain.

After each batch, run a **comprehension check** before moving on. Use questions from `@references/dot-protocol.md` comprehension check templates. Do not proceed to the next batch until the learner demonstrates understanding of the current one.

#### Elaborative Interrogation

After a learner passes a comprehension check on a concept, follow up with a "why" question — but ONLY if the concept is their second or later exposure to related material. Do NOT use elaborative interrogation on the very first concept in a domain or on concepts with no connection to anything the learner already knows.

**The rule:** A learner needs at least one anchor concept before "why" questions become productive. Without background knowledge, "why" questions produce frustrated guessing, not meaningful integration.

**Timing within a batch:**
- Concept 1 of a batch: Recall and relationship questions only. No "why" yet.
- Concept 2+ of a batch: After the comprehension check passes, add one "why" question that connects the new concept to a previously learned one.

**Example sequence:**
1. Deliver "inflation" (first concept). Comprehension check: "In your own words, what is inflation?" (Recall only.)
2. Deliver "interest rates" (second concept). Comprehension check: "How do interest rates relate to inflation?" Then elaborative: **"Why do central banks raise interest rates when inflation rises? What's the mechanism?"**

The "why" question forces the learner to generate a causal explanation that integrates the new concept with prior knowledge. This is different from asking "what happens" (which is a chain question) — it asks "why does this mechanism exist in the first place?"

**Evaluating "why" answers:**

| Quality | Description | Response |
|---------|-------------|----------|
| **Generative** | Learner produces a causal mechanism that connects concepts, even if imperfect | Accept and refine: "That's the right direction. Let me sharpen one detail..." |
| **Circular** | "It happens because it does" or restates the definition | Push deeper: "You've told me WHAT happens. I want to know WHY the mechanism works that way." |
| **Speculative but wrong** | Learner invents a plausible but incorrect mechanism | Valuable attempt. Correct gently: "Interesting hypothesis — the actual mechanism is [X]. But your instinct to look for a cause was exactly right." |
| **Blank** | "I don't know why" | The concept may be too new for elaboration. Back off: "That's okay — we'll come back to the 'why' after you've seen more of the picture." |

#### Interleaving Rule: Block-Then-Interleave

**First exposure to a concept = blocked delivery.** Teach the new batch of 2-3 concepts as a coherent unit with its comprehension check, exactly as described above. Do not interleave during initial teaching — blocking is superior for initial acquisition.

**Comprehension checks on review concepts = interleaved.** After the comprehension check on the new batch, insert 1-2 questions about concepts from PREVIOUS sessions, drawn from the Interleave Pool in Knowledge State. These questions should be mixed unpredictably with the new material — do NOT group them as a separate "review" block.

Implementation:
1. After each new batch comprehension check, select 1-2 concepts from the Interleave Pool.
2. Choose concepts that are DISSIMILAR to the current batch — the point of interleaving is discrimination, not similarity reinforcement.
3. Ask a question that requires the learner to identify WHICH concept applies, not just recall a definition. Use the Application Questions or Relationship Questions from the comprehension check question bank.
4. If the learner confuses an old concept with a new one, that's a productive error. Clarify the distinction — this discrimination learning is exactly what interleaving produces.

Example (economics domain, new batch = GDP + Trade Balance):

> "Quick question from earlier — if the central bank raises interest rates, what happens to bond prices? And is that the same mechanism as what affects GDP, or different?"

The shift between topics forces the learner to identify which conceptual framework applies, building discrimination ability that blocked practice cannot produce.

**Update the Interleave Pool** via `dln-sync` at each sync boundary: add newly taught concepts that passed their comprehension check. Concepts that failed comprehension checks are NOT added to the pool — they need more blocked practice first.

### 2a. Interleaved Practice Round (sessions 3+)

**Skip this step if Session Count < 3** — there aren't enough prior concepts to meaningfully interleave.

After all new concept batches for this session have been delivered and checked, run one round of interleaved practice before chain building. This round mixes old and new concepts together.

#### Protocol

1. Prepare 4-6 questions that draw from at least 3 different concept batches (including today's new batch and 2+ previous batches from the Interleave Pool).

2. Present the questions in a deliberately jumbled order — alternate between old and new concepts, different chains, and different sub-topics. The sequence should feel slightly disorienting. **This is intentional.** Interleaved practice feels harder and less fluent than blocked practice, but produces 43% better delayed retention (Rohrer & Taylor, 2007).

3. For each question, the learner must first identify WHICH concept or chain applies before answering. This two-step process (identify → apply) is the key mechanism.

4. After the round, briefly acknowledge that this felt harder:

> "If that felt harder than the batched practice, that's expected. The research says that harder practice now means better retention later. You're building discrimination — the ability to tell concepts apart, not just recall them one at a time."

5. Log the results in the sync dispatch:
```
- Interleaved practice round: [N/M] correct. Confusions: [concept A ↔ concept B]. Discrimination improving/needs work on [specific distinction].
```

### 3. Chain Building

Connect the delivered concepts into **causal or procedural sequences**. A chain answers: "If X happens, what follows? Why?"

- Present the chain explicitly first (teaching mode).
- Then ask the learner to **explain the chain back** in their own words.
- Use chain-building prompts from `@references/dot-protocol.md`.

Example: "We covered inflation, interest rates, and bond prices. Now: if inflation rises, what happens to interest rates? And then what happens to bond prices? Walk me through it."

#### Chain Visualization

After presenting a chain verbally, render it as a Mermaid flowchart:

```mermaid
graph LR
    A[Inflation rises] -->|"central bank responds"| B[Interest rates rise]
    B -->|"fixed coupons less attractive"| C[Bond prices fall]
```

Label the edges with the **mechanism** (the "why" of each link), not just the direction. This forces both teacher and learner to articulate the causal mechanism, not just the sequence.

After the learner explains the chain back, ask them to describe what their version of the diagram would look like:

> "If you were drawing this chain, what boxes would you have, and what labels would go on the arrows?"

Compare their verbal diagram to yours. Discrepancies reveal misunderstandings that verbal-only explanations might miss — a learner who puts the arrows in the wrong direction has a different causal model than they expressed in words.

#### Chain Mastery Updates

After each chain explain-back, update the chain's mastery status:

| Explain-Back Outcome | Status Update |
|----------------------|---------------|
| Correct direction, correct mechanism, complete — first attempt | `mastered` |
| Correct direction but missing mechanism or intermediate step, OR needed 1 correction | `partial` |
| Wrong direction, major gaps, or needed full re-teaching | `not-mastered` |

Include chain mastery in the `dln-sync` dispatch. A chain cannot be `mastered` unless ALL its constituent concepts are `mastered` or `partial`. If a concept downgrades, any chain containing it downgrades to at most `partial`.

### 3a. Remediation Block

If the Weakness Queue was non-empty at session start, the remediation block runs BEFORE new concept delivery (or interleaved with the first concept batch if the weak item connects to new material).

#### Remediation Protocol

For each queued item, in priority order:

1. **Re-activate:** Ask the learner to recall what they know about the item. Do not re-teach yet. This surfaces the current state of their understanding.
2. **Diagnose:** Compare their recall to the mastery rubric. Identify the specific gap:
   - Cannot recall at all → full re-teach with new analogy
   - Recalls partially but wrong mechanism → targeted correction
   - Recalls but cannot apply → application-focused practice
3. **Intervene:** Use the recovery action matched to the diagnosis. Always use a DIFFERENT approach than what was used when the item was first taught (different analogy, different example, different angle of explanation).
4. **Re-check:** Run a comprehension check immediately after intervention. Score using the mastery rubric.
5. **Update:** Include the mastery update in the next `dln-sync` dispatch. If the item reaches `mastered`, it exits the Weakness Queue. If it improves to `partial`, reduce its severity. If it stays `not-mastered`, escalate its severity and keep it at the top of the queue.

#### Remediation Limits

- Spend at most **2 remediation attempts** per item per session. If the item is still `not-mastered` after 2 attempts, note this in progress and move on. The item stays in the queue for next session with severity `high`.
- If the queue has 3+ items, remediate only the top 2 per session. The rest carry forward.

Tell the learner: "We'll spend a few minutes reinforcing [item] before we dive into new material."

### 4. Worked Example

Walk through a **concrete scenario** in the domain that exercises the chain:

1. Set up the scenario with specific details.
2. Ask the learner to identify which concepts apply.
3. Trace through step by step together — the learner leads, you guide.
4. Highlight where the chain applies in practice.

Use the worked example scaffolding structure from `@references/dot-protocol.md`.

#### Scenario Trace Diagram

After completing the worked example walkthrough, render a diagram that traces the scenario through the chain:

```mermaid
graph TD
    S["Scenario: Fed announces 0.5% rate hike"] --> A[Interest rates ↑]
    A --> B[Bond prices ↓]
    A --> C[Mortgage rates ↑]
    C --> D[Housing demand ↓]

    style S fill:#f9f,stroke:#333
```

Use a distinct style for the scenario trigger node to visually separate "what happened" from "what the chain predicts." This gives the learner a spatial map of how the abstract chain applies to the concrete case.

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

### 5a. Pre-Gate Confidence Check

Before running the phase gate, ask the learner to predict their own performance:

> "Before I test you, I want you to predict how you'll do. Rate your confidence 1-5 on each:
> - Naming core concepts without help: ___
> - Explaining causal chains clearly: ___
> - Tracing through a brand new scenario: ___
>
> And overall: do you think you'll pass the gate? (1 = definitely not, 5 = definitely yes)"

Record these predictions verbatim. Do NOT react to them or adjust the gate based on them. The predictions must be captured BEFORE the gate begins — no revising mid-test.

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

### 5b. Post-Gate Calibration Feedback

After the phase gate (pass or fail), surface the calibration data:

> "You predicted [X/5] on concept recall — you actually [passed easily / struggled with 2].
> You predicted [Y/5] on chain explanation — you [nailed both / got one chain wrong].
> You predicted [Z/5] on scenario tracing — you [needed 0 hints / needed 3 hints].
> Overall you predicted [W/5] and the result was [pass/fail]."

Name the direction of miscalibration explicitly:

- **Overconfident** (predicted higher than actual): "You overestimated your readiness on [area]. This is normal and common — the fix is more practice on exactly the things you feel most confident about."
- **Underconfident** (predicted lower than actual): "You underestimated yourself on [area]. You know more than you think — trust your chains."
- **Well-calibrated** (within 1 point): "Your self-assessment was accurate — that's a valuable skill in itself."

Include the calibration data in the next `dln-sync` dispatch for the `## Calibration Log` section.

## Exit Ritual

At the end of every session, run this three-part close:

**Part 1 — Self-Summary (existing):**
> "What did you learn today? What connects to what?"

**Part 2 — Confidence Self-Assessment (new):**
> "Rate your confidence 1-5 on each concept we covered today:"
> [List each concept from the session]
> "Which concept are you MOST confident about? Which are you LEAST confident about?"

**Part 3 — Confusion Surfacing (new):**
> "What are you still confused about? What felt shaky or incomplete?"

Record all responses. Include the per-concept confidence ratings in the `dln-sync` session-end dispatch for the `## Calibration Log`. The confusion responses go into `## Open Questions` if they identify genuine gaps.

Do NOT reassure the learner that "everything is fine" if they express confusion. Validate the confusion: "That's a real gap — we'll address it next session." Then note it in the sync payload.

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
