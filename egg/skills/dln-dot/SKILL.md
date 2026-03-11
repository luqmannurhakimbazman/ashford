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

**Growth mindset threading:** Throughout all Dot interactions, attribute learning outcomes to effort and strategy. Say "You worked through that carefully" not "You're a natural." When the learner struggles, say "This concept takes time to click — let's try a different angle" not "Let me make it easier." Normalize struggle as part of learning: "The fact that this is hard means you're learning something genuinely new."

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

#### Progress Visibility

Check the Engagement Signals from the page body. Calibrate your opening tone:

- **Momentum = positive:** Open with energy. Reference their recent wins: "Last session you nailed [concept] — let's build on that momentum."
- **Momentum = neutral:** Standard opening. Preview the session plan.
- **Momentum = fragile:** Open gently. Acknowledge the difficulty explicitly: "Last session was a tough one. That's normal — some of the hardest concepts are the ones just before a breakthrough. Today we're going to start with something you're already strong on."

If this is session 1, set momentum to `neutral` and skip this calibration.

Also provide a concrete progress count if the learner has prior sessions:
> "Quick status: you've [mastered/partially learned] [X] of [Y] concepts so far, and built [Z] chains. Here's where we're headed today."

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

**On agent return** — follow the learner-generated checkpoint, plan adjustment, calibration-driven adjustment, and Notion failure handling protocols in `@/Users/luqman/Desktop/projects/my-cc-plugin/ashford/egg/skills/dln/references/sync-protocol.md`.

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

#### Load Monitoring and Overload Response

Watch for overload signals: failed checks after re-teach, learner confusion between concepts, circular answers, "I don't know" shutdowns (all High severity); failed first-attempt checks, requests to repeat (Moderate). Positive signals: quick correct answers, above-phase questions.

When 2+ High-severity signals appear in a single batch, follow the overload response and faded example protocols in `@references/dot-protocol.md` (section 10). Key actions: stop the batch, acknowledge without blame, reduce batch size by 1, add a worked micro-example, re-attempt the check, and log the adjustment.

When the learner shows positive signals across 2+ consecutive batches, increase batch size by 1 (maximum: 4).

For each concept, deliver:

1. **Plain-language definition** — No jargon unless you define it inline.
2. **Concrete analogy** — Something from everyday life that maps to the concept.
3. **Why it matters** — One sentence on why this concept is important in the domain.

After each batch, run a **comprehension check** before moving on. Use questions from `@references/dot-protocol.md` comprehension check templates. Do not proceed to the next batch until the learner demonstrates understanding of the current one.

#### Effort Attribution on Comprehension Checks

- **On pass:** "You explained that clearly — the way you connected it to [analogy/prior concept] shows you're building real understanding." (Process praise, not person praise.)
- **On partial:** "You're on the right track — you got the core idea. Let me sharpen one thing..." (Validate effort, then correct.)
- **On fail:** "That's a tricky one. The fact that you attempted it is what matters — let me come at it from a different angle." (Normalize, reframe, re-teach.)

#### Elaborative Interrogation

After a learner passes a comprehension check, follow up with a "why" question — but ONLY on concept 2+ of a batch (never the first concept in a domain or batch). The learner needs at least one anchor concept before "why" questions become productive.

Use the elaborative interrogation question templates and evaluation rubric from `@references/dot-protocol.md` (section 2).

#### Interleaving Rule: Block-Then-Interleave

**First exposure = blocked delivery.** Teach the new batch as a coherent unit. Do not interleave during initial teaching.

**Review = interleaved.** After each new batch comprehension check, insert 1-2 questions about PREVIOUS concepts from the Interleave Pool. Choose DISSIMILAR concepts. Ask identification questions ("which concept applies?"), not just recall. Use the interleaved comprehension check templates from `@references/dot-protocol.md` (section 9).

If the learner confuses old and new concepts, that's a productive error — clarify the distinction.

**Update the Interleave Pool** via `dln-sync` at each sync boundary: add concepts that passed comprehension checks. Failed concepts stay out until they pass blocked practice.

### 2a. Interleaved Practice Round (sessions 3+)

**Skip if Session Count < 3.** After all new concept batches are delivered, run one round mixing old and new concepts. Prepare 4-6 questions from 3+ concept batches in jumbled order. The learner must identify WHICH concept applies before answering. Use the sequence design rules and templates from `@references/dot-protocol.md` (section 9).

After the round, acknowledge it felt harder: "The research says harder practice now means better retention later." Log results in the sync dispatch.

### 3. Chain Building

Connect the delivered concepts into **causal or procedural sequences**. A chain answers: "If X happens, what follows? Why?"

- Present the chain explicitly first (teaching mode).
- Then ask the learner to **explain the chain back** in their own words.
- Use chain-building prompts from `@references/dot-protocol.md`.

Example: "We covered inflation, interest rates, and bond prices. Now: if inflation rises, what happens to interest rates? And then what happens to bond prices? Walk me through it."

#### Chain Visualization

After presenting a chain verbally, render it as a Mermaid flowchart. Label edges with the **mechanism** (the "why"), not just the direction. After the learner explains back, ask them to describe their version of the diagram verbally. Compare for discrepancies — wrong arrow direction reveals a different causal model. Use the visual templates from `@references/dot-protocol.md` (section 11).

#### Chain Mastery Updates

After each chain explain-back, update the chain's mastery status:

| Explain-Back Outcome | Status Update |
|----------------------|---------------|
| Correct direction, correct mechanism, complete — first attempt | `mastered` |
| Correct direction but missing mechanism or intermediate step, OR needed 1 correction | `partial` |
| Wrong direction, major gaps, or needed full re-teaching | `not-mastered` |

Include chain mastery in the `dln-sync` dispatch. A chain cannot be `mastered` unless ALL its constituent concepts are `mastered` or `partial`. If a concept downgrades, any chain containing it downgrades to at most `partial`.

### 3a. Remediation Block

If the Weakness Queue was non-empty at session start, the remediation block runs BEFORE new concept delivery (or interleaved with the first concept batch if the weak item connects to new material). Follow the remediation protocol and frustration detection/response protocol in `@references/dot-protocol.md` (sections 7 and 12).

Tell the learner: "We'll spend a few minutes reinforcing [item] before we dive into new material."

### 4. Worked Example

Walk through a **concrete scenario** in the domain that exercises the chain:

1. Set up the scenario with specific details.
2. Ask the learner to identify which concepts apply.
3. Trace through step by step together — the learner leads, you guide.
4. Highlight where the chain applies in practice.

Use the worked example scaffolding structure from `@references/dot-protocol.md`.

#### Scenario Trace Diagram

After completing the worked example, render a diagram tracing the scenario through the chain. Use a distinct style for the trigger node to separate "what happened" from "what the chain predicts." See the worked example trace template in `@references/dot-protocol.md` (section 11).

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

At the end of every session:

**1. Self-summary (retrieval practice):**
> "What did you learn today? What connects to what?"

Capture their response as a comprehension signal.

**2. Progress celebration:**
Provide concrete progress metrics:
> "Today you mastered [N] new concepts and built [M] new chains. You now have [total] concepts in your foundation — that's [percentage] of what we'll need for the Linear phase."

**3. Milestone celebrations** (when applicable):

| Milestone | Celebration |
|-----------|-------------|
| First session completed | "You've taken the hardest step — starting. Everything from here builds on what you did today." |
| 5 concepts mastered | "Five concepts mastered. You're building a real knowledge base now." |
| First chain mastered | "Your first chain — that means you're not just learning facts, you're seeing how they connect." |
| Phase gate passed | "You've graduated from Dot phase. That means you have a solid foundation — you're ready to start seeing deeper patterns." |
| Phase gate failed (but improved from last attempt) | "You're closer than last time — [specific improvement]. One more session on [specific area] and you'll be there." |

**4. Forward look:**
> "Next session, we'll [preview]. You've got the pieces — we're going to put them together."

**5. Confidence Self-Assessment:** Ask the learner to rate 1-5 on each concept covered. Which is most/least confident? Use the calibration templates from `@references/dot-protocol.md` (section 8).

**6. Confusion Surfacing:** "What are you still confused about?" Validate confusion honestly — do NOT reassure. Record responses for `## Calibration Log` and `## Open Questions` in the session-end sync dispatch.

**7. Update Engagement Signals:**
Set Momentum based on session outcome:
- Session ended with mastery gains and no frustration → `positive`
- Normal session with mixed results → `neutral`
- Session ended early due to frustration, or 3+ consecutive struggles occurred → `fragile`

Include in the `session-end` dispatch to `dln-sync`.

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
