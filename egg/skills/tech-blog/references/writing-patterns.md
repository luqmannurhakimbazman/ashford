# Writing Patterns for Technical Blog Posts

Detailed structure templates and techniques for different blog post types.

## Table of Contents

- [Gundersen Writing Style (Default)](#gundersen-writing-style-default)
- [Deep Dive Pattern](#deep-dive-pattern)
- [Explainer Pattern](#explainer-pattern)
- [Tutorial Pattern](#tutorial-pattern)
- [Project Showcase Pattern](#project-showcase-pattern)
- [Opening Hook Techniques](#opening-hook-techniques)
- [Transition Techniques](#transition-techniques)
- [Conclusion Patterns](#conclusion-patterns)
- [Storytelling in Technical Content](#storytelling-in-technical-content)

---

## Gundersen Writing Style (Default)

The default writing style for this skill, based on Gregory Gundersen's technical blog. These conventions apply unless the user specifies a different style.

### Progressive Explanation Pattern

Follow the **concrete → abstract → concrete** cycle:

1. Start with a concrete example or motivation
2. Build intuitive understanding (no jargon, no math yet)
3. Formalize with notation and equations
4. Return to the concrete example using the formal tools

At least **20-30% of content** should be intuition-building before any formalization. The reader should understand the *idea* before seeing the *math*.

### Voice

- **First-person singular** is the natural voice: "I want to explain...", "My goal here is...", "In my experience..."
- **"We"** for walking the reader through derivations together: "Let's derive...", "If we substitute..."
- **No "we"** as false modesty when expressing personal opinions — use "I think..." instead
- **No passive voice** for key statements: "The gradient points uphill" not "It is noted that the gradient is directed uphill"
- **Warm but direct** — not chatty, not stiff. Academic but accessible.

### Opening Paragraph Formula

Every post opens with:

1. **Context** (1-2 sentences) — What area are we in? What does the reader already know?
2. **Gap or problem** (optional, 1 sentence) — What's missing, confusing, or hard to find?
3. **Purpose statement** — "The goal of this post is to..." (explicit, always present)
4. **Roadmap** (brief) — What the post covers and in what order

Never open with a definition, equation, vague statement ("X is important in many fields..."), or a rhetorical question without immediate payoff.

### Subtitle Format

The post subtitle (rendered as `<h4>` below the title) follows a strict pattern:

1. Starts with **"I"**
2. Contains an **action verb** (derive, explain, explore, prove, work through, discuss, show, review, implement)
3. States the **specific focus** of the post
4. Is **exactly one sentence** with a period

Examples:
- "I derive the evidence lower bound (ELBO) from scratch using Jensen's inequality."
- "I explain how Gaussian processes work by building intuition from weight-space to function-space views."

### Table of Contents

Required for posts estimated at **> 1500 words**. Place after the opening paragraph:

> "This is a long post. Please feel free to jump around as desired:"

Followed by a numbered list with anchor links to each major section.

### Section Headers

Headers are **descriptive and action-oriented**. Never use generic labels:

| Avoid | Use Instead |
|-------|-------------|
| Introduction | Setting up the problem |
| Background | What we need from probability theory |
| Methods | Deriving the objective function |
| Results | Visualizing the learned representations |
| Conclusion | Summary and further reading |

### "Naming Things" Technique

When introducing notation, use this explicit pattern:

> "First, let's name things. Let $X$ denote our observed data, $Z$ the latent variables we want to infer, and $\theta$ our model parameters."

Rules:
- Every symbol defined **before or at first use**
- Connect symbols to previously built intuition ("$Z$ represents the hidden structure we discussed above")
- Group related notation together

### Intuition-to-Formalism Transition

Use explicit transition phrases when shifting from intuition to math:

- "Now that we have a geometrical intuition for [topic], let's formalize these ideas."
- "With this picture in mind, we can be more precise."
- "Let's make this concrete with notation."

Acknowledge the shift in difficulty when it happens.

### Signposting

Guide the reader at every structural transition:

- **Purpose statements** at section starts: "The goal of this section is to..."
- **Progress markers** at transitions: "Now that we understand X..."
- **Previews** of what's next: "In the next section, we'll..."
- **Summaries** after complex derivations: "To summarize..."
- **Preemptive question answering**: "You might ask why...", "At this point, we may be wondering..."

### "And That's It!" Closure

After completing a complex explanation, provide explicit closure:

- "And that's it! That's the evidence lower bound."
- "And that's the key insight — X is just Y viewed differently."

This signals to the reader that the hard part is done.

### Equation Conventions

- **Numbered** with `\tag{N}` for equations referenced later in the text
- **Derived step-by-step** — never jump more than one logical step
- **Non-obvious steps marked** — "Step * holds because..." or "where we used [identity/theorem]"
- **Plain-language sentence** after each non-trivial equation explaining what it *means*

### Intellectual Honesty

- "In my experience..." — own your perspective
- "I think..." — distinguish opinion from fact
- "My understanding is..." — signal uncertainty
- Acknowledge when something is beyond the post's scope
- Point to better resources: "For a rigorous treatment, see [reference]"

### Technical Diction

When the audience is mathematically fluent (quants, ML practitioners, statisticians, traders), reach for the precise technical term over the plain-English approximation. Words like *orthogonal*, *ergodic*, *convexity*, *idempotent*, *stationary*, *Lipschitz*, *manifold*, *concentration*, *measure-zero*, *almost surely*, *Pareto*, *positive-definite*, *non-stationary*, *heavy-tailed*, and *information-theoretic* compress dense concepts into a single load-bearing word — and signal that you think in terms of structure (geometry, dynamics, second-order effects) rather than narrative.

The rule: use the technical word when it's the most precise word available, not as decoration.

- **Earn the word.** Only reach for the term if (a) you can define it on demand and (b) it is *literally* true under the rigorous meaning. "These two effects are orthogonal" must mean genuinely independent, not merely "different." "The payoff is convex in volatility" must hold for the second derivative, not vibe. If you can't pass a quick examination on the term, replace it with plain English.
- **Don't downshift it.** Don't paraphrase "convexity" as "curved upward" or "ergodic" as "the average works out the same" when writing for a fluent audience. The technical word *is* the plain word for them, and the paraphrase loses information.
- **Gloss on first use.** Even fluent readers vary. The first time a load-bearing term appears, give a one-clause gloss in the prose: "the process is *ergodic* — its time average equals its ensemble average — so we can substitute one for the other." After that, use the term freely without re-defining.
- **Avoid jargon-as-flex.** If a plain word adds equal precision, use the plain word. "These two refactors are orthogonal" is fine and earns its keep; "the codebase has an orthogonal feel" is bullshit and should be cut. The test: would removing the technical word *lose information* or *just lose vocabulary*?

Reaching for the rigorous word when it fits — and only when it fits — is a texture choice that distinguishes serious quantitative writing from generic explainer prose. It is the lexical counterpart to anti-condescension: both are about respecting the reader's intelligence.

### Anti-Condescension

Never use "obviously", "trivially", "simple algebra shows", "the reader should know", or "it's easy to see." Use instead: "We can see that...", "A quick calculation shows...", "Recall that...", "Notice that..."

### Cross-References

Link to your own previous posts when relevant: "If you're unfamiliar with X, see my post on [topic]." Provide enough inline context that the post stands alone.

### Fine-Grained Sub-Styles

The conventions above define the baseline Gundersen style. For more precise style matching, four sub-styles codify distinct patterns observed across his blog:

- **Cataloging Explainer** — General form + specific instances/properties (e.g., exponential family distributions)
- **Interpretive Reframer** — Lesser-known interpretations of familiar concepts (e.g., variance as optimization)
- **First-Principles Narrative** — "Why" questions about algorithms and techniques (e.g., why backprop goes backward)
- **Comprehensive Treatment** — Exhaustive multi-perspective coverage of foundational topics (e.g., linear regression, PCA)

Select based on the reader's goal. For detailed system instructions, structural templates, few-shot excerpts, and anti-patterns for each sub-style, see `references/gundersen-style-cards.md`.

---

## Deep Dive Pattern

For thorough exploration of a single concept or technique. Target: 2000-4000 words.

### Structure (Gundersen Default)

```
1. Title + Subtitle (subtitle as purpose statement starting with "I")
2. Opening paragraph (context → gap → "The goal of this post is..." → roadmap)
3. Table of contents (if > 1500 words)
4. Background / setup (what the reader needs, intuition first)
5. Core explanation (3-5 sections, progressive concrete → abstract → concrete)
   a. Start with high-level intuition and concrete examples
   b. "Let's name things" — introduce notation
   c. Formalize with derivations and equations
   d. Return to concrete examples with the formal tools
   e. Explicit closure at end of complex sections ("And that's it!")
6. Practical implications (how this affects real work)
7. Edge cases and limitations
8. Conclusion (restate insight, connect to purpose, further reading)
```

### Alternative Structure (Hook-Based)

For posts that benefit from a more dramatic opening:

```
1. Hook (counterintuitive claim or surprising fact)
2. Why this matters (1-2 paragraphs establishing stakes)
3. Background context (what the reader needs to know first)
4. Core explanation (the meat — 3-5 sections)
   a. Start with high-level intuition
   b. Progressively add detail and formalism
   c. Include worked examples at each level
5. Practical implications (how this affects real work)
6. Edge cases and limitations
7. Conclusion (takeaway + further reading)
```

### Example Outline: "Why Batch Normalization Actually Works"

```markdown
# Why Batch Normalization Actually Works
## (It's Not About Internal Covariate Shift)

Hook: The original BN paper's explanation was wrong. Here's what's really going on.

### The Standard Story (And Why It's Incomplete)
- Original ICS explanation from Ioffe & Szegedy
- Why experiments don't support this narrative

### What Batch Norm Actually Does
- Loss landscape smoothing (Santurkar et al.)
- Walk through gradient flow with and without BN
- Worked example: 3-layer network gradient magnitudes

### The Smoothing Effect Visualized
- Diagram: loss landscape with/without BN
- Code: computing loss surface for a small network

### Implications for Practice
- When BN helps most (deep networks, aggressive learning rates)
- When it hurts (small batches, recurrent architectures)
- Alternatives: LayerNorm, GroupNorm, and when to use each

### Conclusion
- BN works by smoothing optimization, not fixing ICS
- Further reading: Santurkar et al. 2018, GroupNorm paper
```

---

## Explainer Pattern

For making complex topics accessible. Target: 1200-2500 words.

### Structure

```
1. Hook (relatable analogy or direct question)
2. The simple version (explain like the reader is smart but unfamiliar)
3. Building complexity (2-3 layers of increasing detail)
   a. Layer 1: Core idea with analogy
   b. Layer 2: How it works mechanically
   c. Layer 3: Why it works (theory/math)
4. Common misconceptions (address 1-2)
5. Where to go next
```

### Key Technique: The Zoom Pattern

Start zoomed out, progressively zoom in:

```
Paragraph 1: "PCA finds the directions of maximum variance in your data."
Paragraph 2: "Concretely, it computes eigenvectors of the covariance matrix."
Paragraph 3: "Here's what that eigendecomposition looks like for a 2D dataset..."
Code block: Worked example with actual numbers
```

Each paragraph adds one level of detail. The reader can stop at any depth and still have a useful understanding.

### Analogy Guidelines

Effective analogies for quantitative topics:

| Topic Domain | Analogy Source | Example |
|-------------|---------------|---------|
| Optimization | Physical systems | "Gradient descent is a ball rolling downhill on a bumpy surface" |
| Statistics | Sampling | "Bootstrapping is like asking the same 100 people slightly different questions" |
| Linear algebra | Geometry | "A matrix is a machine that stretches and rotates space" |
| Finance | Risk | "Options pricing is insurance math applied to stocks" |

Rules for analogies:
- Introduce the analogy, then immediately map it to the technical concept
- Acknowledge where the analogy breaks down
- Never let the analogy replace the actual explanation — it supplements it

---

## Tutorial Pattern

For step-by-step building. Target: 1500-3000 words.

### Structure

```
1. What we're building (screenshot, demo, or result preview)
2. Prerequisites (tools, knowledge, environment setup)
3. Step-by-step implementation
   a. Each step: what, why, code, verify
   b. Show intermediate results at each stage
4. The complete result
5. Extensions and next steps
```

### Step Format

Each step follows this micro-pattern:

```markdown
### Step N: [Action verb] the [thing]

Brief explanation of *why* this step matters.

\`\`\`python
# code for this step
result = do_thing(input)
print(result)  # Expected: some_output
\`\`\`

Verify: [How to confirm this step worked correctly]
```

### Tutorial Anti-Patterns

| Anti-Pattern | Problem | Fix |
|-------------|---------|-----|
| Wall of code at the end | Reader lost without context | Show code incrementally per step |
| No verification steps | Reader doesn't know if it's working | Add expected output after each code block |
| Skipping "why" | Reader copies without understanding | One sentence of reasoning per step |
| Assuming environment | "Just install X" | Explicit prerequisites section |

---

## Project Showcase Pattern

For walking through a build or project. Target: 1500-2500 words.

### Structure

```
1. The problem / motivation
2. Solution overview (architecture diagram)
3. Key design decisions (2-3, with reasoning)
4. Implementation highlights (interesting parts only)
5. Results / demo
6. What I'd do differently
7. Links (repo, demo, related work)
```

Focus on the interesting decisions, not every line of code. Link to the repo for complete source.

---

## Opening Hook Techniques

### The Counterintuitive Claim

> "Adding more data to your training set can make your model worse."

Works when: The post explains a non-obvious phenomenon. Requires delivering on the promise — explain *why* the counterintuitive thing is true.

### The Concrete Scenario

> "It's 3 AM. Your model is in production. Accuracy has dropped 12% in the last hour, and you have no idea why."

Works when: The post solves a practical problem. Creates urgency and relevance.

### The Direct Question

> "How do you price something that doesn't exist yet?"

Works when: The post answers a fundamental question. Question should be genuinely interesting, not rhetorical filler.

### The Surprising Number

> "GPT-3 has 175 billion parameters. You can understand how it works with just 12."

Works when: The post demystifies something that seems complex. The contrast creates curiosity.

### The Historical Anchor

> "In 1958, Frank Rosenblatt promised that the perceptron would eventually be 'able to walk, talk, see, write, reproduce itself, and be conscious of its existence.' He was early."

Works when: The post connects to a longer historical arc. The contrast between past and present creates interest.

### Hooks to Avoid

- "In this blog post, we will explore..." — Buries the lead
- "X has gained significant attention in recent years..." — Generic, says nothing
- "As we all know, Y is important..." — Assumes knowledge, adds no value
- Starting with a dictionary definition — Overused, signals weak opening

---

## Transition Techniques

Transitions between sections prevent the post from reading as disconnected chunks.

### Bridge Sentences

End a section by pointing to what comes next:

> "Understanding the loss landscape explains *why* batch norm helps. But it doesn't tell us *when* to use it. For that, we need to look at the training dynamics."

### Question Transitions

Pose the question the next section answers:

> "This works perfectly for i.i.d. data. But what happens when your data has temporal dependencies?"

### Callback Transitions

Reference an earlier point:

> "Remember the 3x3 matrix from our earlier example? Here's what happens when we apply it 1000 times."

### Contrast Transitions

Set up a tension between what was just explained and what's coming:

> "The theory predicts convergence in O(1/t) steps. In practice, the story is messier."

### End-of-Section Signposting

Combine a summary of what was accomplished with a preview of what's next. This is the default transition style for the Gundersen approach:

> "Now that we've derived the ELBO and seen why it's a lower bound on the log-evidence, we can ask the practical question: how do we actually optimize it? In the next section, we'll introduce the reparameterization trick."

Pattern: "Now that we [what was accomplished], [what's next]."

---

## Conclusion Patterns

### The Takeaway Restatement

Restate the core insight in a single sentence, then expand:

> "Batch normalization works by smoothing the loss landscape, not by fixing internal covariate shift. This means [practical implication]."

### The "So What" Conclusion

Connect the post's content to the reader's work:

> "Next time you see training instability, check your normalization choice before tuning hyperparameters. The right normalization can save hours of debugging."

### The Forward Look

Point to open questions or related topics:

> "We've covered how attention works in transformers. The deeper question — *why* attention enables in-context learning — is still an active research area. I'll explore that in a future post."

### The Resource List

End with actionable next steps:

> **Want to go deeper?**
> - Paper: [link] — The original proof
> - Code: [link] — My implementation
> - Book: [title] Chapter 4 — Thorough treatment

---

## Storytelling in Technical Content

### The Discovery Narrative

Frame the post as a journey of understanding:

```
1. I encountered this problem
2. The obvious solution didn't work
3. Here's what I discovered
4. Here's why it works
5. Here's what you should do
```

Works especially well for debugging stories, performance optimization, and "why does X behave this way" posts.

### The Historical Narrative

Frame around how the field evolved:

```
1. People used to think X
2. Then Y was discovered
3. This changed everything because Z
4. Here's the modern understanding
```

Works for explainers of well-established concepts with interesting origin stories.

### The Comparison Narrative

Build understanding through contrast:

```
1. Approach A works like this
2. Approach B works like this
3. Here's when each is better and why
4. Here's the surprising connection between them
```

Works for posts comparing algorithms, frameworks, or techniques.

### Narrative Anti-Patterns

- **Burying the insight** — Don't save the key takeaway for the end. State it early, then explain.
- **Artificial suspense** — Technical readers want the answer first, details second.
- **Over-personalizing** — First person ("I") is natural and encouraged for purpose statements, opinions, and guidance. The anti-pattern is when the *author's journey* overshadows the *concept being explained*. Keep the focus on the ideas; use "I" to express purpose ("My goal is..."), experience ("In my experience..."), and honest opinions ("I think...").
