# Linear Phase Protocol

Detailed teaching templates and rubrics for the DLN Linear phase skill.

---

## 1. Cross-Pollination Question Templates

Use these to guide the learner toward discovering shared structure across chains they already know.

### Primary Templates

- "What do [chain A] and [chain B] have in common?"
- "If you squint, [process X] looks a lot like [process Y] — why?"
- "Strip away the domain-specific details from [chain] — what's the abstract structure?"

### Narrowing Templates (when the learner struggles)

- "Look at step [N] in [chain A] and step [M] in [chain B]. What role does each play?"
- "Both chains have a point where [observation]. Why does that keep showing up?"
- "If I swapped the domain vocabulary between these two chains, would the logic still hold?"

### Deepening Templates (when the learner sees surface similarity)

- "You said they're both about [surface feature]. Go deeper — *why* does that feature appear in both?"
- "That's the what. I want the why. What structural property makes [surface feature] inevitable here?"
- "If [surface feature] disappeared, would the chains still be related? What else connects them?"

---

## 2. Factor Hypothesis Prompts

Use these to push the learner from vague similarity toward precise, structural factor statements.

### Elicitation Prompts

- "Can you state that as a general principle?"
- "Would this factor still hold if we changed [variable]?"
- "What's the minimal statement that captures what you just noticed?"

### Precision Prompts

- "Your factor mentions [domain term]. Can you restate it without that term?"
- "If someone from a completely different field heard your factor, would they understand it?"
- "You said 'whenever [X], [Y] happens.' Are there exceptions? If so, what's the boundary condition?"

### Validation Prompts

- "Name a chain where this factor does NOT apply. Why not?"
- "Can you predict what would happen in [unseen scenario] using this factor alone?"
- "If this factor is true, what else must be true that you haven't checked yet?"

---

## 3. Upgrade Operator Examples

Show how recognizing factors transforms the type of questions the learner can ask. Each example shows the Dot, Linear, and Network versions of a question.

### Example 1: Interest Rates

- **Dot:** "What happens when interest rates rise?"
- **Linear:** "What's the common factor between how rate rises affect bonds vs. how they affect housing?"
- **Network:** "What's the minimal model that predicts rate-rise effects across all asset classes?"

### Example 2: Sorting Algorithms

- **Dot:** "How does quicksort work?"
- **Linear:** "What do quicksort and mergesort share in terms of how they decompose the problem?"
- **Network:** "What's the minimal set of decomposition strategies that covers all comparison-based sorting?"

### Example 3: Machine Learning

- **Dot:** "What is regularization?"
- **Linear:** "What's the common factor between L2 regularization in linear models and dropout in neural networks?"
- **Network:** "What's the minimal principle set that explains why all forms of regularization improve generalization?"

### Example 4: Biology

- **Dot:** "What is homeostasis?"
- **Linear:** "What do thermoregulation and blood glucose regulation share structurally?"
- **Network:** "What's the minimal feedback model that predicts homeostatic behavior across all physiological systems?"

### How to Use

After presenting an example, ask the learner to:
1. Take one of their own Dot questions and upgrade it to Linear.
2. Explain what factor they had to recognize to make the upgrade.
3. Speculate on what the Network version might look like (park it in Open Questions).

---

## 4. Phase Gate Rubric

### Pass Criteria

All three must be met:

1. **Factor Articulation** — The learner can name at least 3 shared factors across their chains. Each factor must be:
   - Structural (describes a relationship, not a domain fact)
   - Stated without domain-specific vocabulary
   - Applicable to 2+ chains

2. **Unseen Problem Prediction** — Present a problem the learner has not encountered. They must predict the outcome by identifying which factor applies. Allow at most 1 hint. The prediction must be:
   - Correct in direction (even if not precise in magnitude)
   - Justified by explicit reference to a factor
   - Not just pattern-matching to a similar chain they memorized

3. **Minimal Principle Set** — The learner can articulate a set of factors that covers 80%+ of their known chains. The set must be:
   - Smaller than the number of chains (actual compression, not relabeling)
   - Non-overlapping (each factor covers distinct ground)
   - The learner can map each chain to its covering factor(s)

### Fail Criteria

Any of these indicate the learner is not ready for Network phase:

- **Vague factors** — "They're kind of similar" or "both involve change" without structural specificity.
- **Domain-locked factors** — Factors that only make sense within one domain and can't transfer.
- **Chain relabeling** — The "minimal principle set" is just their chains reworded, not compressed.
- **Transfer failure** — Cannot apply factors to unseen problems, or needs 2+ hints.
- **Coverage gaps** — Principle set covers less than 80% of known chains with no explanation for the gaps.

### Scoring

- **3/3 pass criteria met, 0 fail criteria triggered** — Pass. Update Phase to Network.
- **2/3 pass criteria met** — Near pass. Note which criterion failed, assign targeted practice, revisit next session.
- **1/3 or 0/3 pass criteria met** — Not ready. Continue Linear phase. Identify whether the issue is factor discovery (more cross-pollination needed) or factor precision (more hypothesis refinement needed).
