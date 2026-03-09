# Network Phase Protocol

Detailed teaching templates and rubrics for the DLN Network phase.

---

## 1. Distributed Revision Cycle Templates

### Eliciting the Model

- "In 3-5 sentences, explain [domain] as you understand it now. Don't look anything up — just tell me your current model."
- "If you had to teach [domain] to someone in 60 seconds, what would you say?"
- "What are the core principles that govern how [domain] works?"
- "Draw the causal chain: what causes what in [domain]?"

### Stress-Testing the Model

- "Your model says [X]. What happens when [edge case]? Does your model still hold?"
- "I'm going to give you three cases. For each one, tell me what your model predicts BEFORE I tell you the answer."
- "Here's a real-world example that experts find surprising. What does your model predict?"
- "What's the weakest part of your model? Where are you least confident?"

### Exploring Mismatches

- "Your model predicted [X] but the answer is [Y]. Before I explain why — what do you think went wrong?"
- "Which of your factors failed to account for this case?"
- "Is this a missing factor, or did you have the right factors but connect them incorrectly?"
- "If you had to add exactly one sentence to your model to cover this case, what would it be?"

### Pushing for Compression

- "Your revised model is [N] sentences. Can you cut it to [N-2] without losing coverage?"
- "Which two of your sentences overlap the most? Can you merge them?"
- "What's the single most important sentence in your model? If you could only keep one, which would it be?"
- "A good model is like a good equation — every term does work. Which of your sentences isn't doing enough work?"

---

## 2. Stress-Test Generation Prompts

Use these systematically to probe the learner's model:

### Boundary Probing

- "What happens at the boundary between [factor A] and [factor B]?"
- "At what point does [factor A] stop mattering and [factor B] take over?"
- "Is there a case where [factor A] and [factor B] point in opposite directions? What wins?"

### Assumption Falsification

- "Your model assumes [assumption] — what if that's false?"
- "In what context would [assumption] break down?"
- "Name one thing your model takes for granted. Now imagine a world where that's not true."

### Cross-Domain Challenge

- "In [adjacent domain], this works differently — can your model explain why?"
- "A practitioner in [other field] uses a completely different framework. Where does their model beat yours?"
- "If your model is truly general, it should apply to [distant analogy]. Does it?"

### Minimal Breaking Case

- "What's the simplest case that breaks your model?"
- "Can you construct the smallest possible example where your model gives the wrong answer?"
- "If I wanted to prove your model wrong, where would I look?"

---

## 3. Factor Discovery Question Bank

Use when stress-tests reveal new factors the learner hadn't identified:

### Relating New Factors to Existing Ones

- "You just found a new factor — how does it relate to your existing factors?"
- "Where does this new factor sit in your causal chain? Does it come before or after your existing factors?"
- "Which of your existing factors does this new one interact with most?"

### Testing for Subsumption

- "Does this factor subsume any of your previous factors?"
- "Is this really a new factor, or is it a special case of [existing factor]?"
- "If you zoom out, are [factor X] and this new factor both expressions of something deeper?"

### Merging and Deepening

- "Can you merge [factor X] and [factor Y] into a single deeper principle?"
- "What's the common thread between these factors? Name the underlying mechanism."
- "If you had to explain both [factor X] and [factor Y] with a single rule, what would it be?"

---

## 4. Structural Hypothesis Testing Prompts

Push the learner to make testable predictions from their model:

### Forward Prediction

- "If your model is correct, what else must be true?"
- "Your model implies [consequence]. Is that actually the case?"
- "Make a prediction from your model that we can test right now."

### Falsification

- "What would falsify your current model?"
- "If I showed you [specific evidence], would that break your model or could your model explain it?"
- "What's the strongest argument against your model?"

### Sufficiency Testing

- "Your model has [N] factors. Are they sufficient to explain [domain], or is something still missing?"
- "If I gave you a new case you've never seen, could your model handle it? Let's try."
- "What class of cases is your model weakest on?"

---

## 5. Compression Quality Rubric

Rate the learner's compressed model each session:

### High Compression Quality

- Model is **5 sentences or fewer**
- Covers **90%+ of known cases** (including edge cases encountered in session)
- Every sentence does meaningful work (no redundancy)
- Model makes correct predictions on transfer tests

### Medium Compression Quality

- Model is **5-10 sentences**, OR
- Covers **70-90% of known cases**
- Some redundancy or overlap between sentences
- Model partially succeeds on transfer tests

### Low Compression Quality

- Model is **verbose (>10 sentences)**, OR
- Covers **less than 70% of known cases**
- Significant redundancy or vague language
- Model fails on transfer tests

### Tracking Across Sessions

Track the **compression ratio** over time:

```
Compression Ratio = (cases covered) / (sentences in model)
```

A rising compression ratio means the learner is building a more powerful, more concise model. This is the primary indicator of Network-phase progress.

---

## 6. Transfer Test Templates

### Applying to Adjacent Domains

- "Apply your model to [adjacent domain]. What does it predict?"
- "A beginner in [adjacent domain] asks you to explain [concept]. Using only your model, what would you say?"
- "Your model was built for [original domain]. Does it work for [adjacent domain] without modification?"

### Diagnosing Transfer Failure

- "Where does the transfer break down? What's domain-specific vs. universal?"
- "Your model worked for [original case] but not [transfer case]. What's different?"
- "Is the failure because of a missing factor, or because [domain] operates on different principles entirely?"

### Cross-Practitioner Testing

- "A practitioner in [other field] would say [X] — does your model agree or disagree?"
- "In [other field], the standard explanation is [Y]. Is that compatible with your model, or does one of them have to be wrong?"
- "If you and a [other field] expert both looked at [shared phenomenon], would you explain it the same way?"
