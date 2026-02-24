---
name: behavioral-interview-prepper
description: This skill should be used when the user wants to prepare for behavioral interviews, generate a behavioral answer bank, practice STAR or SOAR format answers, prep a resume walkthrough narrative, or generate questions to ask their interviewer. Trigger phrases include "prep behavioral", "behavioral interview prep", "prep me for interview at", "practice behavioral questions", "generate behavioral answers", "behavioral prep for", "interview stories for", "STAR method answers", "SOAR answers", "prep my stories", "answer bank for interview", "resume walkthrough", "walk me through your resume prep", "questions to ask my interviewer", or when a user has completed a resume-builder run and asks for interview preparation. It chains off resume-builder output (notes.md, resume.tex, candidate-context.md) to produce a tailored question-and-answer bank.
---

# Behavioral Interview Prepper

Generate a tailored behavioral interview answer bank from resume-builder output. Output goes to the same `hojicha/<company>-<role>-resume/` directory as `behavioral-prep.md`.

## Critical Rules

1. **NEVER fabricate experiences.** Only use content from the resume and `candidate-context.md`. Rephrase and reframe — never invent.
2. **Chain from resume-builder output.** Read existing `notes.md` and `resume.tex` from the output directory. Do not re-parse the JD from scratch.
3. **Honest gap handling.** When the candidate lacks an experience for a question **and has confirmed this after probing (see Rule 5 and Step 4 inline probing)**, provide a deflection strategy — not a made-up story.
4. **Story reuse limit.** No single experience may be used for more than 5 questions. Uses 4-5 require a distinct reframing angle (different trait cluster emphasis). See `references/story-mapping.md`.
5. **Never generate generic or hypothetical answers — ask instead.** If you would produce a vague behavioral answer, stock STAR response, or hypothetical framework where a real experience should be, STOP and ask the candidate a probing question. See `references/candidate-discovery.md` for anti-generic detection and probing techniques. Every answer must be grounded in real experience. Hypothetical frameworks are only acceptable when the candidate explicitly confirms they have no related experience.

---

## Workflow

### Step 1: Parse Inputs

Read from the existing resume-builder output directory:

```
Required:
- hojicha/<company>-<role>-resume/notes.md (JD summary, keyword analysis, gap analysis)
- hojicha/<company>-<role>-resume/resume.tex (tailored resume bullets)
- hojicha/candidate-context.md (supplementary experiences beyond the resume)
```

Derive the company name and role from the directory name or the JD summary in `notes.md`.

**If required files do not exist:** Prompt the user to run the `resume-builder` skill first with the target JD. This skill requires resume-builder output — it does not accept raw JD/resume input directly.

### Step 2: Extract Behavioral Signals

Scan the JD keywords and culture indicators from `notes.md`. Map each behavioral keyword to a trait cluster (e.g., "fast-paced" → Adaptability, "cross-functional" → Collaboration). Weight clusters by frequency. See `references/behavioral-signals.md` for the full taxonomy.

### Step 2.5: Story Discovery

Before predicting questions, probe the candidate for stories that their current materials don't cover — especially for high-priority trait clusters.

1. **Identify thin clusters.** Review the weighted trait clusters from Step 2. Cross-reference against available stories in `resume.tex` and `candidate-context.md`. Flag any cluster with fewer than 2 distinct stories.
2. **Probe for real experiences.** For each under-covered cluster, ask a targeted question from the matching category in `references/candidate-discovery.md`. Example: "Your materials don't have a strong resilience story. Can you think of a time something went wrong — at work, in a project, or even personally — where you had to figure it out under pressure? What happened?"
3. **Ask one at a time.** Wait for the candidate's response before moving to the next cluster. Follow up on vague answers per the probing technique in `references/candidate-discovery.md`.
4. **Persist discoveries.** Append all new stories to `hojicha/candidate-context.md` using the persistence format in `references/candidate-discovery.md`.
5. **Proceed with enriched story bank.** Continue to Step 3 with the updated materials.

**Skip conditions:** If all high-priority trait clusters have 2+ distinct stories in the existing materials, you may skip this step.

### Step 3: Predict Questions

Select 10-15 questions from the master bank based on the top-weighted trait clusters and the role type. Prioritize primary clusters, then fill with secondary clusters. See `references/question-bank.md` for the full question bank.

### Step 4: Map Experiences to Questions

For each predicted question, find the best-fit experience from the tailored resume bullets and `candidate-context.md`. Reframe the experience to match the question's trait cluster. Enforce the 5-question reuse limit (uses 4-5 require distinct reframing angles). See `references/story-mapping.md` for mapping methodology.

**Inline probing:** Before flagging a question as a gap (requiring deflection), ask the candidate first: "Before I flag this as a gap — do you have any experience with [specific situation from the question]? Even from outside work — personal projects, school, volunteer work, or life experiences?" Follow probing technique from `references/candidate-discovery.md`. Only use gap handling strategies from `references/story-mapping.md` after the candidate confirms they don't have a relevant experience. Persist any new stories to `candidate-context.md`.

### Step 5: Generate Answers

Write a structured answer for each question using the appropriate format based on question type. Answers should be detailed enough to serve as speaking notes but not scripted word-for-word. See `references/answer-formats.md` for templates.

For the "Tell me about yourself" / resume walkthrough answer, also reference `references/resume-walkthrough.md` for the 90-second narrative arc framework, positive transition framing, and anti-patterns.

**Inline probing:** When drafting a STAR answer and the Action or Result feels thin or generic (per anti-generic heuristics in `references/candidate-discovery.md`), pause and ask: "For this story about [X], can you tell me more about what you specifically did? What was the hardest part? What would you do differently looking back?" Use the candidate's response to add authentic detail to the answer.

### Step 6: Finance Layer (Conditional)

If the role is in finance, trading, or quant, load the finance-specific behavioral layer. Detect by scanning `notes.md` for keywords listed in the "When to Load" section of `references/finance-behavioral.md`. Add domain-specific questions and adjust answer framing for finance culture.

### Step 7: Generate Questions for Interviewer

Generate 5-8 tailored questions the candidate should ask their interviewer, based on JD signals, company context, and role type. Organize by audience:

- **For Technical/Hiring Manager interviews:** Questions about current projects, team challenges, technical stack decisions, and what success looks like in the role. Lead with "tell me what you're working on now" — if interesting, this can fill 5-10 minutes of natural conversation.
- **For Team Member/Culture interviews:** Questions about team dynamics, collaboration patterns, and day-to-day experience.
- **Role-specific questions:** Questions that demonstrate you've read the JD carefully and thought deeply about the role.
- **Avoid asking (save for recruiter):** Timing/logistics for next rounds, compensation, benefits, travel. These should be directed to the recruiting coordinator, not the interviewer.

**Anti-patterns:** Don't ask about things you should already know from 3 months of networking (staffing model, industry mix, training program). Show you've done your homework by asking questions that go deeper.

**Candidate-driven questions:** After generating the initial question list, ask the candidate: "What genuinely interests you about this company or role? What would you actually want to know if you were sitting across from the interviewer?" Use their real curiosity to replace or refine stock questions. The best interviewer questions come from authentic interest, not from a template.

### Step 8: Output

Write `behavioral-prep.md` in the same output directory:

```
hojicha/<company>-<role>-resume/
  notes.md            # Already exists (from resume-builder)
  resume.tex          # Already exists (from resume-builder)
  cover-letter.md     # May exist (from resume-builder)
  behavioral-prep.md  # Generated by this skill
```

---

## Output Structure

The generated `behavioral-prep.md` should follow this structure:

```markdown
# Behavioral Interview Prep: <Company> — <Role>

## Behavioral Signals Extracted

| Signal | JD Evidence | Trait Cluster |
|--------|-------------|---------------|
| ... | ... | ... |

## Your Story Bank

| # | Story | Source | Trait Clusters | Questions Assigned |
|---|-------|--------|----------------|-------------------|
| 1 | ... | resume / candidate-context | ... | Q2, Q5 |
| ... | ... | ... | ... | ... |

## Predicted Questions & Answers

### [Trait Cluster Name]

#### Q1: "[Question text]"
**Format:** [STAR / SOAR / Thesis / Framework-First / Claim+Proof / Present-Past-Future]
**Draw from:** [Story #N reference]

> [Structured answer using the selected format]

#### Q2: "[Question text]"
...

### Motivation & Fit

#### QN: "Why this company?"
**Format:** Thesis (3 reasons)
**Draw from:** JD research + candidate values

> [Structured answer]

#### QN+1: "Tell me about yourself"
**Format:** Present → Past → Future
**Draw from:** [Story reference]

> [Structured answer]

### Finance Behavioral (if applicable)

#### QN+2: "[Finance-specific question]"
...

## Gap Awareness

| Gap | Deflection Strategy |
|-----|-------------------|
| No direct experience with X | Pivot to adjacent experience Y, emphasize transferable skill Z |
| ... | ... |

## Questions to Ask Your Interviewer

### For Technical / Hiring Manager Interviews
1. [Tailored question based on JD signal or company context]
2. [Question about current team challenges or projects]
3. [Question demonstrating deep JD reading]

### For Team Member / Culture Interviews
1. [Question about team dynamics]
2. [Question about day-to-day experience]

### Avoid Asking (save for recruiter)
- Timeline and logistics for next rounds
- Compensation and benefits details
- Basic company info available on the website
```

---

## Quick Reference

### Question Type → Answer Format

| Question Type | Answer Format | Reference |
|---|---|---|
| "Tell me about a time..." | STAR with metrics | `references/answer-formats.md` |
| "Why this company/role?" | Thesis (3 reasons) | `references/answer-formats.md` |
| "What would you do if..." | Framework-first | `references/answer-formats.md` |
| "Strengths/weaknesses" | Claim + proof + growth | `references/answer-formats.md` |
| "Tell me about yourself" | Present → Past → Future | `references/answer-formats.md` |
| "Tell me about a time..." (adversity) | SOAR with obstacle emphasis | `references/answer-formats.md` |
| "Walk me through your resume" | 90-second narrative arc | `references/resume-walkthrough.md` |

### Output Directory Convention

```
hojicha/<company>-<role>-resume/behavioral-prep.md
```

Examples:
- `hojicha/kronos-research-ml-researcher-resume/behavioral-prep.md`
- `hojicha/grab-data-engineer-resume/behavioral-prep.md`
- `hojicha/stripe-backend-engineer-resume/behavioral-prep.md`

### Trait Cluster Summary

See `references/behavioral-signals.md` for the full taxonomy mapping JD keywords to trait clusters (Leadership, Collaboration, Adaptability, Problem-Solving, Communication, Drive, Technical Depth).

### Resume Walkthrough

See `references/resume-walkthrough.md` for the 90-second narrative arc framework with positive transition framing, anti-patterns, and worked examples.
