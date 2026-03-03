---
name: Technical Blog Writer
description: This skill should be used when the user asks to "write a blog post", "draft a blog post", "create a technical blog", "write a deep dive", "write an explainer", "blog about", "write a tutorial post", "turn this into a blog post", or wants to create technical content for a personal blog or static site. Default platform is Jekyll (Gundersen-style) with KaTeX math, BibTeX citations via jekyll-scholar, and custom figure HTML. Covers deep dives, explainers, tutorials, and project showcases on ML, statistics, computer science, finance, math, and quantitative topics. Generates Markdown with SEO frontmatter, code examples, and diagram suggestions.
---

# Technical Blog Writer

Produce publication-ready technical blog posts as Markdown files. The default platform is a Gundersen-style Jekyll blog with KaTeX math, jekyll-scholar citations, and custom HTML figures. For other platforms (Hugo, Astro, Next.js), consult `references/seo-frontmatter.md` for platform-specific frontmatter and conventions.

## Core Philosophy: Collaborative Blog Writing

**Blog writing is collaborative, but Claude should be proactive in delivering drafts.**

The typical workflow starts with a topic, a codebase, a paper, or a vague idea. Claude's role is to:

1. **Understand the source material** by exploring repos, papers, or notes
2. **Deliver a complete first draft** when confident about the angle
3. **Search for citations** using web search and APIs to support claims
4. **Refine through feedback cycles** when the author provides input
5. **Ask for clarification** only when genuinely uncertain about key decisions

**Key Principle**: Be proactive. If the topic and angle are clear, deliver a full draft. Don't block waiting for feedback on every section — authors are busy. Produce something concrete they can react to, then iterate based on their response.

---

### Balancing Proactivity and Collaboration

**Default: Be proactive. Deliver drafts, then iterate.**

| Confidence Level | Action |
|-----------------|--------|
| **High** (clear topic, obvious angle) | Write full draft, deliver, iterate on feedback |
| **Medium** (some ambiguity) | Write draft with flagged uncertainties, continue |
| **Low** (major unknowns) | Ask 1-2 targeted questions, then draft |

**Draft first, ask with the draft** (not before):

| Section | Draft Autonomously | Flag With Draft |
|---------|-------------------|-----------------|
| Outline | Yes | "Framed as deep dive — adjust if you prefer tutorial" |
| Opening | Yes | "Emphasized problem X — correct if wrong angle" |
| Body sections | Yes | "Included sections A, B, C — reorder if needed" |
| Code examples | Yes | "Used Python — switch to R/Julia if preferred" |
| Citations | Yes | "Cited papers X, Y, Z — add any I missed" |

**Only block for input when:**
- Target audience is unclear (experts vs. beginners changes everything)
- Topic is too broad to pick a single angle
- Platform is unclear (Jekyll vs. Hugo vs. other)
- Explicit request to review before continuing

---

## CRITICAL: Never Hallucinate Citations

**This is the most important rule in blog writing with AI assistance.**

### The Problem
AI-generated citations have a **~40% error rate**. Hallucinated references — papers that don't exist, wrong authors, incorrect years, fabricated DOIs — are a serious credibility problem. Blog posts persist indefinitely with no retraction process, and readers propagate errors by citing your post in their own work.

### The Rule
**NEVER generate BibTeX entries from memory. ALWAYS fetch programmatically.**

| Action | Correct | Wrong |
|--------|---------|-------|
| Adding a citation | Search API → verify → fetch BibTeX | Write BibTeX from memory |
| Uncertain about a paper | Mark as `PLACEHOLDER` | Guess the reference |
| Can't find exact paper | Note: "placeholder — verify" | Invent similar-sounding paper |

### When You Can't Verify a Citation

Use an explicit placeholder pattern so the author knows to check:

```liquid
{% cite PLACEHOLDER_author2024_verify %}
<!-- TODO: Could not verify this citation exists. Please confirm before publishing. -->
```

**Always tell the author**: "I've marked [X] citations as placeholders that need verification. I could not confirm these references exist."

For the complete citation verification workflow, see `references/citation-workflow.md`.

---

## CRITICAL: Jekyll Post Generation Rules

**These rules prevent the most common rendering failures. Every rule was learned from real broken builds.**

| # | Rule | Why |
|---|------|-----|
| 1 | Wrap entire post body in `{% katexmm %}...{% endkatexmm %}` | KaTeX won't render `$...$` or `$$...$$` without it |
| 2 | Do NOT include `{% bibliography --cited %}` in posts | The layout renders the bibliography automatically — including it creates duplicates |
| 3 | Image paths use `/image/{post-slug}/` | Other paths (e.g., `/assets/`, relative paths) break on the live site |
| 4 | Number figures by narrative order | First figure *referenced* in text = Figure 1, regardless of creation order |
| 5 | No CSS classes in post HTML | The site stylesheet handles all styling — custom classes in posts cause conflicts |
| 6 | Never modify site infrastructure | Only touch `_posts/`, `_bibliography/`, `image/` — never `_layouts/`, `_includes/`, `css/`, `_config.yml` |

### KaTeX Wrapping (Required)

The entire post body (everything after the YAML frontmatter) must be wrapped in `{% katexmm %}...{% endkatexmm %}`. This enables `$...$` for inline math and `$$...$$` for display math throughout the post.

```liquid
---
title: "Post Title"
layout: default
date: 2024-03-15
---
{% katexmm %}

Your entire post content goes here. Inline math like $x^2$ and display math:

$$\mathcal{L}(\theta) = \sum_{i=1}^{N} \log p(x_i \mid \theta) \tag{1}$$

All math works because the whole post is inside the katexmm block.

{% endkatexmm %}
```

**Do NOT use individual `{% katex %}...{% endkatex %}` blocks** — they are legacy/alternative syntax. The `katexmm` wrapper is simpler and prevents missed math blocks. MathJax is legacy; KaTeX is the active renderer.

### Bibliography: Layout Renders It Automatically

The site's `default.html` layout already includes the bibliography rendering logic. Adding `{% bibliography --cited %}` to a post body causes a **duplicate bibliography** at the bottom of the page.

**Correct:** Just use `{% cite key %}` inline. The bibliography appears automatically.

**Wrong:** Adding `{% bibliography --cited %}` at the end of the post.

### Figure Numbering: Narrative Order

Number figures by the order they are **first referenced** in the text, not by the order they were created or appear in a source document.

| First reference in text | Figure number |
|------------------------|---------------|
| "Consider [this diagram], which shows..." | Figure 1 |
| "As shown in [this plot]..." | Figure 2 |
| "[This table] compares..." | Figure 3 |

The `<div class='caption'>` must appear immediately after the `<img>` tag, inside the same `<div class='figure'>` container.

### Boundary Rule: Never Modify Site Infrastructure

Blog post generation should **only** create or modify files in:
- `_posts/` — the post Markdown file
- `_bibliography/` — BibTeX entries for citations
- `image/` — figures and diagrams

**Never** suggest changes to `_layouts/`, `_includes/`, `css/`, `_config.yml`, or any other site infrastructure files. If something seems broken in the rendering, the post content is wrong — not the site.

---

## Workflow 0: Starting from Source Material

When the user provides a codebase, paper, library, or project as source material, start here before the Core Workflow.

```
Source Material Workflow:
- [ ] Step 1: Explore the source material
- [ ] Step 2: Identify the blog angle
- [ ] Step 3: Confirm angle and audience with user
- [ ] Step 4: Find citations in source material
- [ ] Step 5: Search for additional references
- [ ] Step 6: Proceed to Core Workflow Step 2 (outline)
```

### Step 1: Explore the Source Material

Understand what you're working with:

```bash
# For a codebase
ls -la
find . -name "*.py" -o -name "*.rs" -o -name "*.ts" | head -20
find . -name "README*" -o -name "*.md" | head -10

# For a paper or PDF
# Read the abstract, introduction, and conclusion first

# For a library
# Read the README, check examples/, look at core API
```

Look for:
- `README.md` — Project overview and key claims
- `examples/`, `notebooks/` — Demonstrations of the concept
- `tests/` — Edge cases and expected behavior
- Existing `.bib` files or citation references
- Any draft documents or notes

### Step 2: Identify the Blog Angle

Ask: "What would Gundersen write about this?" The best Gundersen-style posts:
- Take **one concept** and explain it thoroughly
- Derive from first principles, not just describe
- Connect theory to code
- Assume an intelligent reader who wants to understand, not just use

Candidate angles:
- **The derivation** — "Deriving X from scratch" (deep dive)
- **The intuition** — "What X really means" (explainer)
- **The implementation** — "Building X step by step" (tutorial)
- **The connection** — "How X relates to Y" (deep dive)

### Step 3: Confirm Angle and Audience

Present your proposed framing:

> "Based on the source material, I propose writing a [deep dive/explainer/tutorial] on [specific angle]. The target reader would be [audience level]. The key takeaway: [one sentence]. Should I proceed with this framing?"

**Then proceed with the draft.** Don't wait for confirmation unless the angle is genuinely ambiguous.

### Step 4: Find Citations in Source Material

Check for papers already referenced:

```bash
# Find existing citations in codebase
grep -r "arxiv\|doi\|cite\|reference" --include="*.md" --include="*.bib" --include="*.py" --include="*.rs"
find . -name "*.bib"
```

These are high-signal starting points — the author has already deemed them relevant.

### Step 5: Search for Additional References

Use Exa MCP (if available) or web search to find supporting references:

```
Search queries to try:
- "[main concept] tutorial" or "[main concept] explained"
- "[concept] original paper"
- "[concept] textbook reference"
- Author names from existing citations
```

Then verify and retrieve BibTeX using the workflow in `references/citation-workflow.md`.

### Step 6: Skip to Core Workflow Step 2

With the angle confirmed and citations gathered, skip Step 1 (Clarify Scope) and proceed directly to **Step 2: Research and Outline** in the Core Workflow below.

---

## Core Workflow

Follow this sequence for every blog post:

```
Blog Post Workflow:
1. Clarify scope and audience
2. Research and outline
3. Generate frontmatter
4. Write the draft
5. Add code examples, figures, and citations
6. Review and polish
```

### Step 1: Clarify Scope and Audience

Before writing, establish:

- **Topic boundary** — What specific aspect to cover (not "transformers" but "how multi-head attention computes contextual embeddings")
- **Target reader** — Assumed background level (beginner, intermediate, advanced practitioner)
- **Key takeaway** — The one thing the reader should understand after reading
- **Estimated length** — Short (800-1200 words), Medium (1500-2500 words), Long (3000+ words). If > 1500 words, plan a table of contents.

If the user provides a codebase, paper, or project as source material, explore it first to identify the most interesting angle for a blog post.

### Step 2: Research and Outline

Build a structured outline before drafting:

1. **Opening paragraph** — Context (1-2 sentences) → gap/problem (optional) → purpose statement ("The goal of this post is to...") → roadmap (brief)
2. **Table of contents** — Plan one if estimated length > 1500 words
3. **Core sections** — 3-5 sections building understanding progressively (concrete → abstract → concrete)
4. **Practical application** — Code, examples, or worked problems
5. **Conclusion** — Key takeaway restated, further reading

Present the outline for user approval before writing the full draft. Adjust based on feedback.

For detailed structure patterns (deep dive, tutorial, explainer formats), consult `references/writing-patterns.md`.

### Step 3: Generate Frontmatter

Generate YAML frontmatter for the user's blog platform. The default Jekyll format:

```yaml
---
title: "Deriving the Evidence Lower Bound"
subtitle: "I derive the evidence lower bound (ELBO) from scratch using Jensen's inequality."
layout: default
date: 2024-03-15
keywords: variational-inference, elbo, bayesian-inference, jensens-inequality
published: true
---
```

Key fields:

- **`title`** — Clear, specific, SEO-friendly (50-60 characters ideal)
- **`subtitle`** — Starts with "I" + action verb (derive, explain, explore, prove, work through) + specific focus. Exactly one sentence with a period. Doubles as meta description.
- **`layout`** — `default` (single custom layout)
- **`date`** — `YYYY-MM-DD`
- **`keywords`** — Comma-separated string (not an array), 4-6 keywords
- **`published`** — `true` to publish, `false` to hide from build

For other platforms, consult `references/seo-frontmatter.md`.

### Step 4: Write the Draft

#### Opening Paragraph

Every post opens with context → purpose → roadmap. Include an explicit purpose statement: "The goal of this post is to..." Avoid generic openings like "In this post, we will explore..." or "X has become increasingly popular..." For specific patterns, consult `references/writing-patterns.md`.

#### Body Sections

For each section:

- **Lead with the intuition** before formulas or code
- **Use concrete examples** — Numbers, scenarios, visualizations over abstract descriptions
- **Build progressively** — Each section should depend on the previous one
- **Include transition sentences** between sections connecting ideas
- **Signpost at transitions** — Use purpose statements, progress markers, previews, and summaries (see `references/writing-patterns.md`)

#### Writing Philosophy

- **Intuition before formalism** — At least 20-30% of content should be intuition-building before formalization
- **"Naming things"** — When introducing notation, use "First, let's name things." Define every symbol before or at first use, connected to previously built intuition
- **Progressive explanation** — Follow the concrete → abstract → concrete cycle
- **Subtitle as summary** — The subtitle captures the entire post's purpose in one sentence
- **Cross-reference own posts** — Link to previous posts for background: "If you're unfamiliar with X, see my post on [topic]"
- **Acknowledge analogy limitations** — When using analogies, state where they break down
- **Explicit closure** — After complex explanations, provide closure: "And that's it! That's the [concept]."

#### Math and Formulas

Wrap the entire post body in `{% katexmm %}...{% endkatexmm %}` (see [CRITICAL: Jekyll Post Generation Rules](#critical-jekyll-post-generation-rules)). This enables `$...$` for inline math and `$$...$$` for display math throughout the post. Do NOT use individual `{% katex %}` blocks.

- Introduce notation before using it ("First, let's name things...")
- Provide intuitive explanation alongside formal definitions
- Use inline math (`$x$`) for variables in prose, display math (`$$...$$`) for key equations
- **Number equations** with `\tag{N}` for equations referenced later in the text
- **Derive step-by-step** — never jump more than one logical step
- **Justify non-obvious steps** — "Step * holds because..." or "where we used [identity/theorem]"
- Add a "what this means" sentence after every non-trivial equation

#### Writing Style

- **Active voice** — "The gradient descent algorithm updates weights" not "Weights are updated by..."
- **Short paragraphs** — 3-5 sentences maximum for screen readability
- **Concrete over abstract** — "A 3x3 matrix" not "a matrix of arbitrary dimensions"
- **No filler** — Cut "basically", "essentially", "it's worth noting that", "it should be mentioned"
- **No LLM anti-patterns** — See the banned patterns list below
- **First person naturally** — Use "I" for purpose, experience, opinions, and guidance. Use "we" for walking through derivations together.
- **Anti-condescension** — Never use "obviously", "trivially", "simple algebra shows", "the reader should know." Use instead: "We can see that...", "A quick calculation shows...", "Recall that..."
- **Signposting** — Purpose statements at section starts, progress markers at transitions, summaries after complex sections

#### LLM Anti-Patterns (Banned)

These patterns are common in AI-generated prose and must never appear in blog output:

| Pattern | Example | Fix |
|---------|---------|-----|
| Em dashes for tone or filler | "This changes everything — or does it?" | Use em dashes only when they add structural clarity (e.g., parenthetical asides). Default to commas, periods, or semicolons. |
| Rhetorical questions for drama | "The twist?" "Do you know what I realized?" "What if I told you..." | State the point directly. |
| Formulaic intensifiers | "It wasn't just X, it was Y" "The real issue? Something else entirely." | Write the actual claim without the theatrical setup. |

### Step 5: Code Examples, Figures, and Citations

#### Code Examples

Include code when it reinforces understanding:

- **Minimal and focused** — Show only the relevant concept, strip boilerplate
- **Imports included** — Show exactly what to install/import
- **Annotated** — Add comments explaining non-obvious lines, connect to theory ("This implements Equation N")
- **Runnable** — Reader should be able to copy-paste and execute
- **Language-appropriate** — Python for ML/data, TypeScript/Rust for systems, pseudocode for algorithms

Structure code blocks as:

```markdown
Brief explanation of what this code does:

\`\`\`python
# Clear, annotated code here
\`\`\`

Explanation of the output or key insight from running this.
```

#### Figures

Use the HTML figure convention for the default Jekyll blog. For other platforms, use standard Markdown images.

```html
<div class='figure'>
    <img src='/image/[topic-slug]/[filename].png'
         style='width: 60%; min-width: 250px;' />
    <div class='caption'>
        <span class='caption-label'>Figure N.</span> Caption text that
        fully describes what the figure shows.
    </div>
</div>
```

Figure rules:
- **Number by narrative order** — First figure *referenced* in the text = Figure 1 (see [CRITICAL: Jekyll Post Generation Rules](#critical-jekyll-post-generation-rules))
- **Caption immediately after image** — `<div class='caption'>` goes right after `<img>`, inside the same `<div class='figure'>`
- **Caption is self-contained** — Reader understands the figure from caption alone
- **Reference in text** — "Consider Figure 3, which shows..."
- **Consistent design** — Same colors, fonts (Arial), and line weights across all figures
- **Progressive complexity** — Simple visuals before complex ones

For detailed figure conventions and Mermaid patterns, consult `references/diagrams.md`.

#### Citations

For the default Jekyll blog using jekyll-scholar:

- Cite inline with `{% cite key %}` — renders as (Author, Year)
- Store references in `_bibliography/references.bib`
- Do **NOT** add `{% bibliography --cited %}` in the post — the layout renders it automatically (see [CRITICAL: Jekyll Post Generation Rules](#critical-jekyll-post-generation-rules))
- Use consistent citation keys: `author_year_firstword` (e.g., `vaswani_2017_attention`)

**When to cite:**
- Claims about prior work, theoretical results, or experimental findings
- Definitions or theorems attributed to specific sources
- Datasets, benchmarks, or tools you reference
- When saying "it has been shown that..." — cite the source or rephrase

**Verification requirement:** Every citation must be verified programmatically. Never write BibTeX from memory.

Condensed verification workflow:
1. **Search** — Use Exa MCP or Semantic Scholar to find the paper
2. **Verify** — Confirm it exists in 2+ sources (Semantic Scholar + CrossRef/arXiv)
3. **Retrieve** — Get BibTeX via DOI content negotiation
4. **Validate** — Confirm the specific claim appears in the source
5. **Add** — Append verified entry to `_bibliography/references.bib`

If any step fails, use the placeholder pattern:

```liquid
{% cite PLACEHOLDER_author2024_verify %}
<!-- TODO: Could not verify — author must confirm before publishing -->
```

For the complete workflow with Python code and API details, see `references/citation-workflow.md`. For Jekyll setup, see `references/jekyll-setup.md`.

### Step 6: Review and Polish

Before delivering the final draft, verify:

- [ ] **Purpose statement present** — Opening includes "The goal of this post is..."
- [ ] **Subtitle format correct** — Starts with "I" + action verb + specific focus, one sentence
- [ ] **Intuition before formalism** — At least 20-30% intuition-building before math/formal content
- [ ] **Signposting at transitions** — Purpose, progress, preview, and summary markers throughout
- [ ] **Worked examples** — Each major concept has a worked example with concrete numbers
- [ ] **Anti-condescension check** — No "obviously", "trivially", "simple algebra shows"
- [ ] **Progressive structure** — Each section builds on the last (concrete → abstract → concrete)
- [ ] **Code is runnable** — Examples include imports, comments, and work if copy-pasted
- [ ] **Math is introduced** — No undefined notation, "naming things" technique used
- [ ] **Figures numbered and referenced** — Every figure has a caption and is referenced in text
- [ ] **Frontmatter is complete** — Title, subtitle, layout, date, keywords, published
- [ ] **No filler language** — Every sentence adds value
- [ ] **No LLM anti-patterns** — No gratuitous em dashes, no rhetorical questions for drama, no "It wasn't just X, it was Y" formulaic structures
- [ ] **Conclusion connects to purpose** — Restates insight, further reading included
- [ ] **Table of contents** — Present if post > 1500 words
- [ ] **Length matches scope** — Not padded, not rushed
- [ ] **KaTeX wrapping** — Entire post body wrapped in `{% katexmm %}...{% endkatexmm %}`
- [ ] **No bibliography tag** — Post does NOT contain `{% bibliography --cited %}`
- [ ] **Image paths** — All images use `/image/{post-slug}/` absolute paths
- [ ] **Figure numbering** — Figures numbered by narrative order (first referenced = Figure 1)
- [ ] **No CSS classes** — Post HTML does not include custom CSS classes
- [ ] **No infrastructure modifications** — Only `_posts/`, `_bibliography/`, `image/` touched

For a detailed scoring rubric (10 criteria, 1-5 scale), consult `references/style-rubric.md`.

## Adapting by Post Type

| Type | Focus | Length | Code | Diagrams |
|------|-------|--------|------|----------|
| **Deep dive** | Thorough exploration of one concept | 2000-4000 words | Supporting | Architecture, flow |
| **Explainer** | Make complex topic accessible | 1200-2500 words | Illustrative | Concept maps |
| **Tutorial** | Step-by-step building | 1500-3000 words | Central | Setup, flow |
| **Project showcase** | Walk through a build | 1500-2500 words | Central | Architecture |

For detailed patterns per post type, see `references/writing-patterns.md`.

## Quantitative Content Guidelines

When writing about ML, statistics, finance, or math:

- **Ground formulas in intuition** — Explain what an equation "is doing" before showing it
- **Use worked examples** — Walk through a calculation with actual numbers
- **Connect to practice** — Show how theory maps to code or real-world application
- **Cite sources** — Use `{% cite key %}` for jekyll-scholar, or link to papers and textbooks
- **Be precise about assumptions** — State when results require specific conditions (i.i.d., stationarity, etc.)

## Writing Philosophy: The Gundersen Approach

This skill synthesizes writing philosophy from researchers and writers who excel at clear technical exposition:

| Source | Key Contribution to This Skill |
|--------|-------------------------------|
| **Gregory Gundersen** | Primary exemplar. Subtitle format, "naming things" technique, purpose statements, explicit closure, intuition-first derivations, cross-referencing |
| **Gopen & Swan** | 7 principles of reader expectations — topic/stress positions, old-before-new, subject-verb proximity |
| **Andrej Karpathy** | Single contribution focus, clear framing, connecting theory to practice |
| **Zachary Lipton** | Word choice, eliminating hedging, cutting intensifiers |

### Distinctive Gundersen Conventions (With Attribution)

- **Subtitle as summary** — One sentence starting with "I" + action verb. Captures the entire post's purpose. *Source: Consistent pattern across Gundersen's posts.*
- **"First, let's name things."** — Introduce all notation in one place before using it, connected to intuition built earlier. *Source: Gundersen's derivation posts.*
- **Purpose statement** — "The goal of this post is to..." in the opening paragraph. *Source: Gundersen's consistent opening structure.*
- **Explicit closure** — "And that's it! That's the [concept]." after complex explanations. *Source: Gundersen's deep dives.*
- **Intuition before formalism** — At least 20-30% intuition-building before any formal content. *Source: Gundersen + Karpathy's teaching philosophy.*
- **Cross-reference own posts** — "If you're unfamiliar with X, see my post on [topic]." *Source: Gundersen's blog interconnections.*
- **Anti-condescension** — Never "obviously" or "trivially." Always "We can see that..." or "Recall that..." *Source: Lipton's word choice heuristics, adapted for blogs.*

### Fine-Grained Style Selection

The baseline conventions above apply to all Gundersen-style posts. For more precise style matching, select one of four sub-styles based on the reader's goal:

| Sub-Style | Reader Goal | Typical Topics |
|-----------|-------------|----------------|
| **Cataloging Explainer** | Understand a general pattern and see instances | Distribution families, model classes, property sheets |
| **Interpretive Reframer** | See a familiar concept from a new angle | Loss function motivations, dual characterizations |
| **First-Principles Narrative** | Understand *why* something works | Algorithm design choices, "why" questions |
| **Comprehensive Treatment** | Get a complete multi-perspective reference | Foundational models (OLS, PCA), canonical decompositions |

For detailed system instructions, structural templates, few-shot excerpts, and anti-patterns for each sub-style, see `references/gundersen-style-cards.md`.

For exemplar posts demonstrating each convention, see `references/sources.md`. For detailed writing patterns, see `references/writing-patterns.md`. For the scoring rubric, see `references/style-rubric.md`.

---

## What Blog Readers Actually Read

Understanding reader behavior helps prioritize effort:

| Content Element | % Readers Who See It | Implication |
|----------------|---------------------|-------------|
| **Title** | 100% | Must be specific and compelling |
| **Opening paragraph** | 80-90% | Front-load purpose and hook |
| **Table of contents** | 70-80% (if present) | Readers scan for sections they care about |
| **Figures and diagrams** | 60-70% | Examined before surrounding text |
| **Code examples** | 50-60% (technical readers) | Readers copy-paste and test |
| **Math/derivations** | 30-50% | Only engaged readers follow derivations |
| **Conclusion** | 40-60% | Many skip to the end after scanning |

### Time Allocation

Spend approximately **equal time** on each of:
1. **Title + subtitle + opening paragraph** — This is where most readers decide to keep reading or leave
2. **Figures + code examples** — These are what readers actually engage with
3. **Everything else** — Body text, derivations, transitions, conclusion

**If your title and opening don't hook the reader, your brilliant derivation in Section 3 will never be read.**

---

## Common Issues and Solutions

**Issue: Generic opening**
- *Symptom*: First sentence could apply to any post on the topic ("X has become increasingly popular...")
- *Fix*: Delete the first sentence. Start with the purpose: "The goal of this post is to..."

**Issue: No clear purpose**
- *Symptom*: Reader finishes and thinks "what was the point?"
- *Fix*: Add an explicit purpose statement in the opening paragraph and ensure the conclusion connects back to it

**Issue: Too much math too early**
- *Symptom*: Equations appear before the reader understands why they matter
- *Fix*: Apply the 20-30% intuition rule. Explain what an equation "does" before showing it. Use the "naming things" technique.

**Issue: Code examples don't run**
- *Symptom*: Reader copy-pastes code and gets ImportError or NameError
- *Fix*: Include all imports at the top. Test-run the example yourself. Add version notes if library APIs change frequently.

**Issue: Unverified citations**
- *Symptom*: BibTeX entries written from memory, plausible but possibly fabricated
- *Fix*: Follow `references/citation-workflow.md`. Search → Verify → Retrieve → Validate → Add. Mark anything unverified as `PLACEHOLDER`.

**Issue: Post covers too much ground**
- *Symptom*: 4000+ words, reader loses the thread, no single takeaway
- *Fix*: Split into a series. Each post should have exactly one key takeaway. Link between posts with cross-references.

**Issue: No signposting**
- *Symptom*: Reader loses track of where they are in the argument
- *Fix*: Add purpose statements at section starts ("In this section, we show..."), progress markers ("Now that we have X, we can..."), and previews ("Next, we'll see how..."). See `references/writing-patterns.md`.

**Issue: Wrong subtitle format**
- *Symptom*: Subtitle is a fragment, a question, or doesn't start with "I"
- *Fix*: Rewrite to: "I [verb] [topic] [using/by/from] [method/approach]." One sentence, period at the end.

**Issue: Duplicate bibliography at page bottom**
- *Symptom*: Two identical bibliographies appear at the bottom of the rendered page
- *Fix*: Remove `{% bibliography --cited %}` from the post body. The `default.html` layout renders the bibliography automatically. See [CRITICAL: Jekyll Post Generation Rules](#critical-jekyll-post-generation-rules).

**Issue: Math not rendering (KaTeX)**
- *Symptom*: Raw `$...$` or `$$...$$` appears as plain text instead of rendered math
- *Fix*: Wrap the entire post body in `{% katexmm %}...{% endkatexmm %}` (after the YAML frontmatter, before any content). See [CRITICAL: Jekyll Post Generation Rules](#critical-jekyll-post-generation-rules).

---

## Additional Resources

### Reference Files

For detailed guidance beyond this core workflow:

- **`references/writing-patterns.md`** — Gundersen writing style defaults. Blog structure templates for deep dives, explainers, and tutorials. Opening paragraph formula, transition techniques, signposting patterns, conclusion formulas.
- **`references/seo-frontmatter.md`** — Jekyll frontmatter (default) and other platforms (Hugo, Astro, Next.js). SEO title formulas, meta description templates, tag strategies.
- **`references/diagrams.md`** — HTML figure convention (Jekyll default). Mermaid diagram syntax for common patterns. When to use diagrams vs. prose. Figure design rules and caption conventions.
- **`references/jekyll-setup.md`** — Jekyll configuration reference. Plugins (jekyll-katex, jekyll-scholar), CSS design conventions, file organization, design philosophy.
- **`references/style-rubric.md`** — Draft evaluation rubric with 10-criteria scoring. Quick reference templates for openings, transitions, conclusions, and subtitles.
- **`references/citation-workflow.md`** — Citation verification workflow for blogs. API-based 5-step verification, Python implementation, BibTeX entry formats, hallucination prevention rules, jekyll-scholar troubleshooting.
- **`references/gundersen-style-cards.md`** — Four sub-style reference cards (Cataloging Explainer, Interpretive Reframer, First-Principles Narrative, Comprehensive Treatment). Selection heuristics, structural templates, few-shot examples, and anti-patterns for fine-grained Gundersen style guidance.
- **`references/sources.md`** — Source bibliography with Gundersen exemplar posts. Writing advice sources, tools and APIs, Jekyll ecosystem references.
