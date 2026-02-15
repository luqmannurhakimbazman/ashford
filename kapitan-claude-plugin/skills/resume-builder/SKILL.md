---
name: resume-builder
description: This skill should be used when the user wants to tailor a resume for a specific job description. Trigger phrases include "tailor resume", "tailor my resume", "optimize resume for JD", "build resume for", "target job description", "customize resume for", "adapt resume to job", "resume for this role", "refactor resume", "update resume for", "match resume to JD", "resume for this position", or when a user pastes a job description alongside their resume. It performs keyword extraction, gap analysis, and produces a tailored LaTeX resume with detailed analysis notes.
---

# Resume Builder

Tailor the master resume (`hojicha/resume.tex`) for a specific job description. Output goes to `hojicha/<company>-<role>-resume/` containing `notes.md` (analysis) and `resume.tex` (tailored resume).

## Critical Rules

1. **NEVER fabricate experiences, skills, or achievements.** Only use content from the master resume. You may rephrase, reorder, and emphasize — never invent.
2. **Preserve the `fed-res.cls` document class.** Do not modify `\documentclass[letterpaper,12pt]{fed-res}` or add packages. Copy `hojicha/fed-res.cls` into the output directory. See `references/latex-commands.md` for available commands.
3. **Maintain ATS compatibility.** No graphics, tables outside the cls structure, or custom fonts. The cls already sets `\pdfgentounicode=1`.
4. **Keep to one page — less is more.** The resume must fit a single letter-sized page. Highlight only your strongest 2-3 bullets per role — cut average achievements. Fewer strong bullets beat many mediocre ones. If a bullet doesn't directly support the target JD, consider removing it.
5. **Use XYZ bullet format.** Every experience bullet should follow "Accomplished [X] as measured by [Y], by doing [Z]." See `references/xyz-formula.md`.
6. **Strategic uncommenting.** The master resume contains commented-out sections (LSE, SUSS, Arcane, KlimaDAO, SuperAI, Ripple, CFA). Uncomment entries that are relevant to the target role.
7. **Strategic commenting.** Comment out entries that are irrelevant or weaken the application for the target role.

---

## Workflow

### Step 1: Parse Inputs

Read the master resume and the job description provided by the user.

```
Required:
- Job description (pasted text, file, or URL — if a URL is provided, use a web-fetching tool to retrieve the JD content)
- Master resume: hojicha/resume.tex

Optional (ask if not provided):
- Company name (for output directory naming)
- Role title (for output directory naming)
- Any special instructions (e.g., "emphasize ML experience")

Contact info note: If the role is location-sensitive or requires phone screening, ensure the resume header includes a phone number and city/country alongside email, LinkedIn, and GitHub.
```

Derive `<company>` and `<role>` from the JD for the output directory name. Use lowercase, hyphenated slugs (e.g., `kronos-research-ml-researcher-resume`).

### Step 2: Keyword Analysis

Extract keywords and requirements from the JD. Categorize them:

| Category | Examples |
|----------|---------|
| Hard skills | Python, PyTorch, distributed training |
| Soft skills | Leadership, cross-functional collaboration |
| Domain knowledge | NLP, reinforcement learning, quantitative finance |
| Tools/platforms | AWS, Docker, Kubernetes |
| Qualifications | BSc in CS, 3+ years experience |

See `references/ats-keywords.md` for extraction strategies and ATS mechanics.

### Step 3: Professional Summary

Generate a role-specific professional summary to place at the top of the resume:

1. **Headline**: Write a role-specific headline (<10 words) to use as the `\section{}` title. Do NOT use generic titles like "Professional Summary" — use a descriptive noun phrase (e.g., "ML Engineer & Quantitative Researcher").
2. **Summary paragraph**: Write a <50-word summary starting with a job role noun. Use action words and active voice. Highlight the candidate's top 3-5 selling points that match the JD.
3. **LaTeX**: Add a `\section{<headline>}` with a single paragraph before the Education section.

### Step 4: Section Ordering

Reorder resume sections based on the candidate's experience level relative to the target role:

| Experience Level | Recommended Order |
|------------------|-------------------|
| <3 years / recent grad | Summary → Education → Experience → Projects → Skills |
| 3+ years | Summary → Experience → Education → Projects → Skills |

The master resume uses Education → Experience → Projects → Skills. Adjust the order to put the strongest sections first for the target role.

### Step 5: Gap Analysis

Map each JD requirement to existing resume content. Identify:

- **Strong matches**: Resume already demonstrates this clearly
- **Reframeable**: Experience exists but needs rephrasing to highlight relevance
- **Gaps**: No matching experience (document honestly in notes.md — do NOT fabricate)

### Step 6: Avenues to Strengthen Application

For each gap identified in Step 5, generate an actionable mitigation entry. This section helps the candidate understand what they can do *before or alongside* applying to improve their chances.

For each gap, provide:

- **Severity**:
  - 🔴 High — hard requirement explicitly listed in the JD; candidate has no evidence of it
  - 🟡 Medium — preferred/advantageous qualification; candidate lacks it but it's not a dealbreaker
  - 🟢 Low — nice-to-have, or the gap is easily bridged by reframing existing experience

- **Mitigation Strategy**: A specific, actionable recommendation — not generic advice. Examples of good mitigations:
  - "Build a small MCP server in Python that wraps a public API — add it to your Projects section"
  - "In your cover letter, explicitly bridge your equity factor strategy experience to orderbook data"
  - "Complete the fast.ai Transformer course and implement a small attention-based model"
  - "Contribute a PR to an open-source Jax project to demonstrate working proficiency"

- **Why it works**: Brief explanation of what signal it sends to the hiring manager

Rules:
1. Prioritize 🔴 High gaps — these need the most attention and should appear first
2. Include at least one concrete project idea that could be added to the resume's Projects section if completed
3. Be honest about gaps that cannot be easily closed (e.g., "Do not list C++ if you don't know it — acknowledge it as a growth area in your cover letter instead")
4. **Never recommend fabricating experience** — this is a strict rule across the entire skill
5. For 🟢 Low gaps, it's acceptable to say "No action needed" if existing experience covers it sufficiently

### Step 7: XYZ Bullet Optimization

Rewrite experience bullets using the XYZ formula, incorporating target keywords naturally. See `references/xyz-formula.md` for methodology and examples.

Priority order for keyword placement (optimized for human readers — recruiters read top-down):
1. Professional summary / first bullet of most relevant role
2. Most recent experience section
3. Projects/Leadership section
4. Skills section (highest ATS hit rate — see `references/ats-keywords.md` for ATS-specific priority)

### Step 8: Strategic Uncommenting & Commenting

Review commented-out sections in the master resume. Uncomment entries that strengthen the application:

| Commented Section | Uncomment When Targeting |
|-------------------|--------------------------|
| LSE Summer School | Quantitative finance, computational methods |
| SUSS Linear Algebra | Math-heavy roles, ML theory positions |
| Arcane Group | Growth/BD roles, crypto/web3 |
| KlimaDAO | Climate tech, ESG, web3 research |
| SuperAI Hackathon | Regulatory tech, AWS, agentic AI |
| Ripple Hackathon | Blockchain, DeFi, full-stack web3 |
| CFA Challenge | Finance, ESG, investment research |

Similarly, comment out entries that are irrelevant or that weaken the narrative for the target role.

When including projects, ensure project names link to GitHub repos where possible using `\href{https://github.com/...}{\textbf{Project Name}}`.

### Step 9: Skills Reordering

Reorder skills categories and items within each category to front-load the most relevant ones. The first items in each line are what ATS and recruiters see first.

### Step 10: Output Generation

Create the output directory and files:

```
hojicha/<company>-<role>-resume/
  notes.md      # Analysis and tailoring decisions
  resume.tex    # Tailored resume
```

**notes.md structure:**

```markdown
# Resume Tailoring: <Company> — <Role>

## JD Summary
<Brief summary of the role and key requirements>

## Keyword Analysis
<Table of extracted keywords by category>

## Gap Analysis
| Requirement | Status | Resume Evidence |
|-------------|--------|-----------------|
| ... | Strong Match / Reframed / Gap | ... |

## Avenues to Strengthen Application

| Gap | Severity | Mitigation Strategy |
|-----|----------|---------------------|
| **<Skill/Requirement>** | 🔴 High — <why it's critical> | <Specific, actionable recommendation> |
| **<Skill/Requirement>** | 🟡 Medium — <why it matters> | <Specific, actionable recommendation> |
| **<Skill/Requirement>** | 🟢 Low — <why it's minor> | <Recommendation or "No action needed"> |
| ... | ... | ... |

## Changes Made
- <List of specific changes: reworded bullets, uncommenting, reordering>

## Sections Commented Out
- <Entries removed and why>

## Sections Uncommented
- <Entries added and why>
```

**resume.tex**: Copy the master resume structure exactly, applying all modifications. Include `\documentclass[letterpaper,12pt]{fed-res}` and all original formatting. Reference `references/latex-commands.md` for the cls command reference.

**Plain text verification**: After generating `resume.tex`, verify that all meaningful content is conveyed through text, not through visual layout alone. Check that: (1) no critical information relies solely on bold/italic/positioning to convey meaning, (2) all special characters render as readable text when LaTeX formatting is stripped, and (3) acronyms are expanded at least once so ATS can match both forms.

---

## Quick Reference

### Output Directory Convention

```
hojicha/<company>-<role>-resume/
```

Examples:
- `hojicha/kronos-research-ml-researcher-resume/`
- `hojicha/grab-data-engineer-resume/`
- `hojicha/stripe-backend-engineer-resume/`

### XYZ Formula

```
Accomplished [X] as measured by [Y], by doing [Z]
```

See `references/xyz-formula.md` for full methodology.

### fed-res.cls Commands

| Command | Usage |
|---------|-------|
| `\resumeSubheading{Org}{Loc}{Title}{Date}` | Experience/education entry |
| `\resumeItem{text}` | Bulleted item |
| `\resumeSubHeadingListStart/End` | Wrap subheading groups |
| `\resumeItemListStart/End` | Wrap bullet lists |
| `\resumeProjectHeading{Title \| Tech}{Date}` | Project entry |

See `references/latex-commands.md` for the full reference.
