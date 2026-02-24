# Design: behavioral-interview-prepper Skill

## Overview

A skill that generates a tailored behavioral interview answer bank by chaining off resume-builder output. All company-specificity is derived from the JD at runtime — no pre-built company profiles.

## Inputs & Pipeline

**Input:** Resume-builder output directory (`hojicha/<company>-<role>-resume/`)
- `notes.md` — JD summary, keyword analysis, gap analysis
- `resume.tex` — tailored resume with the candidate's strongest bullets for this role
- `candidate-context.md` — supplementary experiences not on the resume

**Trigger:** "prep behavioral", "behavioral interview prep", "prep me for interview at X", or similar, after completing a resume-builder run.

**Output:** `behavioral-prep.md` saved in the same output directory:
```
hojicha/<company>-<role>-resume/
  notes.md
  resume.tex
  cover-letter.md (if generated)
  behavioral-prep.md   <- new
```

## Workflow

1. **Parse resume-builder output** — Read `notes.md`, `resume.tex`, and `candidate-context.md`.
2. **Extract behavioral signals** — Scan JD for culture/behavioral keywords. Categorize into trait clusters (leadership, collaboration, resilience, initiative, etc.). Ref: `behavioral-signals.md`.
3. **Predict likely questions** — Based on signals + role type (tech vs finance), select 10-15 questions from master bank. Weight toward traits the JD emphasizes most. Ref: `question-bank.md`.
4. **Map experiences to questions** — For each predicted question, find best-fit experience from resume bullets and candidate-context. One experience can serve multiple questions (reframed), but no story used more than 2-3 times. Ref: `story-mapping.md`.
5. **Generate structured answers** — Use the appropriate format per question type. Ref: `answer-formats.md`.
   - Behavioral ("Tell me about a time...") -> STAR with metrics
   - Motivational ("Why this company/role?") -> Thesis format (3 reasons with evidence)
   - Hypothetical ("What would you do if...") -> Framework-first, then example
   - Self-assessment ("Strengths/weaknesses") -> Claim + proof point + growth angle
6. **Finance behavioral layer (conditional)** — If role is at a trading firm, quant fund, or finance company, add trading-specific behavioral questions. Ref: `finance-behavioral.md`.
7. **Output generation** — Write `behavioral-prep.md` organized by category.

## Output Structure (`behavioral-prep.md`)

```markdown
# Behavioral Interview Prep: <Company> — <Role>

## Behavioral Signals Extracted
| Signal | JD Evidence | Trait Cluster |
|--------|-------------|---------------|

## Your Story Bank
| Story | Source | Questions It Serves |
|-------|--------|---------------------|

## Predicted Questions & Answers

### Leadership & Ownership
#### Q1: "Tell me about a time you took ownership..."
**Format:** STAR
**Draw from:** [specific resume bullet]
> **S:** ...  **T:** ...  **A:** ...  **R:** ... (with metric)

### Collaboration & Communication
...

### Motivation & Fit
#### Q5: "Why <Company>?"
**Format:** Thesis (3 reasons)
...

### [Finance Behavioral — if applicable]
...

## Gap Awareness
Experiences you DON'T have that might come up, with honest deflection strategies.
```

## References

| File | Purpose | Loaded When |
|------|---------|-------------|
| `behavioral-signals.md` | Taxonomy of behavioral keywords -> trait clusters. How to extract signals from JD language. | Always (Step 2) |
| `question-bank.md` | Master list of behavioral questions by trait cluster. Tech + finance categories. | Always (Step 3) |
| `answer-formats.md` | Templates for each answer type: STAR w/ metrics, thesis, framework-first, claim+proof. | Always (Step 5) |
| `finance-behavioral.md` | Trading/quant-specific behavioral questions and answer guidance. | Only for finance roles (Step 6) |
| `story-mapping.md` | Guide for mapping experiences to questions, reframing, avoiding repetition, handling gaps. | Always (Step 4) |

## SKILL.md Scope

**In SKILL.md (~150-200 lines):**
- Frontmatter with name + description
- Critical rules (never fabricate, only use real experiences)
- 7-step workflow with pointers to references
- Output directory convention
- Quick reference table for answer formats

**In references:** All detailed content (question banks, signal taxonomies, format templates, finance layer).

## Critical Rules

1. **NEVER fabricate experiences.** Only use content from the resume and candidate-context.md.
2. **Chain from resume-builder output.** Read the existing notes.md and resume.tex, don't re-parse the JD from scratch.
3. **Honest gap handling.** When the candidate lacks an experience, provide a deflection strategy, not a made-up story.

---

## Improvement: WSO Behavioral Crash Course Integration (2026-02-24)

Based on a WallStreetOasis post on behavioral/fit interview best practices (20+ interviews, MBA consulting internships). 5 targeted changes to the existing skill.

### Change 1: New `references/resume-walkthrough.md`

Dedicated reference for crafting the 90-second resume narrative / "tell me about yourself" answer.

**Contents:**
- 3-act framework: Where you started → Key transitions → Why you're here now
- Timing: 90 seconds total, ~20s per transition, ~15s for the landing
- Core rules: every transition must have a positive motivating reason ("I wanted to deepen my exposure to X" not "I was bored"). Every move should seem intentional and self-initiated.
- Anti-pattern list: too long, too detailed, negative framing ("left because of bad management"), no narrative arc, "wanted to try something new"
- Fill-in-the-blank template with 2-3 worked examples (tech→tech, tech→finance, career changer)
- Integration: Step 5 references this when generating "Tell me about yourself" answer

### Change 2: Add SOAR format to `references/answer-formats.md`

6th format alongside existing 5:
- **SOAR (Situation-Obstacle-Action-Result):** Emphasizes the obstacle explicitly before the action
- Timing: Situation 10-15s (~25-40 words), Obstacle 10-15s (~25-40 words), Action 60-75s (~150-190 words), Result 15-30s (~40-75 words)
- Total: 1.5-2 minutes spoken / ~250-350 words written
- When to use over STAR: When the story's challenge is its strongest element — front-loads "why this was hard"
- Before/after example following existing pattern
- Update format selection guide to include SOAR

### Change 3: New workflow step — "Questions for Interviewer"

Insert as Step 7 (current Step 7 becomes Step 8):
- Generate 5-8 tailored questions per company/role in behavioral-prep.md
- Categories: Project/day-to-day, Team/culture, Role-specific, Growth
- Audience guidance: flag which questions are for interviewers vs. recruiters
- Anti-patterns: don't ask things you should know from networking, don't ask salary/benefits in early rounds
- Output section:

```markdown
## Questions to Ask Your Interviewer

### For Technical/Hiring Manager Interviews
1. [Tailored question based on JD signal]

### For Team Member / Culture Interviews
1. ...

### Avoid Asking (save for recruiter)
- Timeline/logistics questions
- Compensation/benefits questions
```

### Change 4: Bump story reuse cap in `references/story-mapping.md`

- Reuse limit: 3 → 5 questions per story
- Uses 4 and 5 require a distinct reframing angle (different trait cluster emphasis)
- Add "Reframe Angle" column to story bank table for uses beyond 3
- Philosophy note: "6-8 core stories should cover your full question set. Slight modifications in framing — not new stories — are how experienced candidates scale."

### Change 5: SKILL.md workflow integration

- Reference `references/resume-walkthrough.md` in Step 5 for "Tell me about yourself"
- Insert new Step 7 (Questions for Interviewer) before output step
- Update output structure template to include "Questions to Ask" section
- Update Quick Reference table to include SOAR format
- Update story reuse cap reference from 3 to 5

### Files Touched

| # | Change | Files |
|---|--------|-------|
| 1 | Resume walkthrough reference | New: `references/resume-walkthrough.md` |
| 2 | SOAR format | Edit: `references/answer-formats.md` |
| 3 | Questions-to-ask generation | Edit: `SKILL.md` |
| 4 | Story reuse cap bump | Edit: `references/story-mapping.md` |
| 5 | Workflow integration | Edit: `SKILL.md` |
