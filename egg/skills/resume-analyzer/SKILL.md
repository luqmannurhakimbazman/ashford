---
name: resume-analyzer
description: This skill should be used when the user wants to analyze a job description against their resume, extract keywords, identify gaps, or prepare tailoring notes. Trigger phrases include "analyze JD", "analyze this job description", "extract keywords from JD", "gap analysis for", "what does this role need", "compare my resume to this JD", "tailor resume", "optimize resume for JD", "build resume for", "target job description", "customize resume for", "resume for this role", "refactor resume", "update resume for", "match resume to JD", or when a user pastes a job description alongside their resume. It produces a notes.md analysis file that resume-tailor uses to generate the final resume.
---

# Resume Analyzer

Analyze a job description against the user's master resume and candidate context. Output: `<application-dir>/notes.md`.

## Resume Workspace

Resolve the inputs before analysis:

1. Use a resume workspace directory supplied by the user.
2. Otherwise, default `<resume-root>` to `${CLAUDE_PROJECT_DIR}/resumes`.
3. If `${CLAUDE_PROJECT_DIR}` is unavailable or the required files are absent, ask the user for the resume workspace or individual file paths. Never assume a machine-specific directory.

By convention, the master resume is `<resume-root>/resume.tex`, candidate context is `<resume-root>/candidate-context.md`, and `<application-dir>` is `<resume-root>/<company>-<role>-resume/`. Explicit user-provided paths override these conventions.

## Critical Rules

1. **NEVER fabricate experiences, skills, or achievements.** Only use content from the master resume and `candidate-context.md`. You may rephrase and emphasize -- never invent.
2. **Never generate generic content -- ask instead.** If a bullet or gap description would be vague, ask a probing question to get real specifics.

## Workflow

### Step 1: Parse Inputs

Read the master resume, candidate context, and the job description.

```
Required:
- Job description (pasted text, file, or URL -- fetch URL content if needed)
- Master resume: user-provided path or `<resume-root>/resume.tex`
- Candidate context: user-provided path or `<resume-root>/candidate-context.md`
Optional (ask if not provided):
- Company name
- Role title
- Special instructions (e.g., "emphasize ML experience")
```

Derive `<company>` and `<role>` from the JD. Use lowercase, hyphenated slugs (e.g., `kronos-research-ml-researcher-resume`).

### Step 2: Discovery Interview (Conditional)

Cross-reference JD requirements against candidate materials. If there are 2+ areas where the JD demands depth that current materials don't address:

1. Read `references/candidate-discovery.md` for probing techniques
2. Ask targeted questions one at a time -- wait for each response
3. Maximum two follow-ups per topic
4. Append new discoveries to the resolved candidate context file

Skip if `candidate-context.md` already covers the JD requirements. Note in the output if skipped and why.

### Step 3: Keyword Extraction

Extract and categorize keywords from the JD:

| Category | Examples |
|----------|---------|
| Hard skills | Python, PyTorch, distributed training |
| Soft skills | Leadership, cross-functional collaboration |
| Domain knowledge | NLP, reinforcement learning, quantitative finance |
| Tools/platforms | AWS, Docker, Kubernetes |
| Qualifications | BSc in CS, 3+ years experience |

Separate into Required vs Preferred. Count frequency -- terms repeated across the JD are high-priority. Include both acronyms and full forms (e.g., "Natural Language Processing (NLP)").

### Step 4: Gap Analysis

Map each JD requirement to existing resume content:

- **Strong match**: Resume already demonstrates this
- **Reframeable**: Experience exists but needs rephrasing
- **Gap**: No matching experience -- before marking, ask the candidate if they have unlisted experience. Persist any discoveries to `candidate-context.md`

### Step 5: Avenues to Strengthen Application

For each gap, provide:

- **Severity**: High (hard requirement, no evidence), Medium (preferred qualification, missing), Low (nice-to-have, easily bridged)
- **Mitigation strategy**: Specific, actionable recommendation (not generic advice)
- **Why it works**: What signal it sends to the hiring manager

Prioritize High gaps first. Include at least one concrete project idea. Never recommend fabricating experience.

### Step 6: Output

Create `<application-dir>/notes.md` containing:

- JD summary
- Keyword analysis table
- Gap analysis table (requirement, status, resume evidence)
- Avenues to strengthen application
- Recommended section ordering (Education-first for <3yr experience, Experience-first for 3+yr)
- Recommended sections to uncomment/comment from master resume

After completing the analysis, tell the user: "Analysis complete. Run the `resume-tailor` skill to generate the tailored resume.tex from these notes."
