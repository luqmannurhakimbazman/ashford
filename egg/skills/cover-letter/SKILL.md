---
name: cover-letter
description: This skill should be used when the user wants to write a cover letter for a job application. Trigger phrases include "write cover letter", "cover letter for", "draft cover letter", "write a cover letter for this role", or when a user asks for a cover letter after resume tailoring. It reads the notes.md from a prior resume-analyzer run and produces a tailored cover letter.
---

# Cover Letter

Generate a tailored cover letter. Reads `<application-dir>/notes.md` (produced by `resume-analyzer`) for keyword and gap analysis.

**Prerequisite:** `notes.md` should exist for best results. Resolve `<application-dir>` from a user-provided directory, or default it to `${CLAUDE_PROJECT_DIR}/resumes/<company>-<role>-resume/`. If `${CLAUDE_PROJECT_DIR}` is unavailable, ask the user where to write the file. Explicit user-provided paths override the convention. If `notes.md` is missing, ask for a JD and do lightweight keyword extraction inline.

## Critical Rules

1. **NEVER fabricate experiences.** Only reference real experiences from the master resume and `candidate-context.md`.
2. **Every sentence must pass the "So what?" test.** If it doesn't add value, cut it.

## When to Write a Cover Letter

| Candidate Profile | Value |
|-------------------|-------|
| Entry-level / student | High -- differentiates from identical CVs |
| Career changer | High -- explains the pivot |
| Experienced (5+ years) | Low -- write only if explicitly requested |
| Referral / networking intro | Medium -- reinforces the warm introduction |

## Four-Paragraph Structure

### Paragraph 1: Hook & Introduction
State who you are, the exact role, and a personal connection to the firm. Answer: "Why this specific firm?"

### Paragraph 2: Selling Your Experience
Map top 2-3 experiences directly to JD requirements. Use JD keywords. Lead with accomplishments, quantify impact.

### Paragraph 3: Why This Firm
Demonstrate genuine knowledge. Contrast with peers. Name-drop employees met, cite recent deals/initiatives/blog posts. This paragraph is the hardest to write well and easiest to spot when faked.

### Paragraph 4: Professional Close
Restate interest, mention availability, thank the reader. 2-3 sentences.

## LLM Anti-Patterns (Banned)

| Pattern | Fix |
|---------|-----|
| Em dashes for tone or filler | Use only for structural clarity. Default to commas, periods, semicolons. |
| Rhetorical questions for drama ("The twist?") | State the point directly. |
| Formulaic intensifiers ("It wasn't just X, it was Y") | Write the actual claim without theatrical setup. |

## Constraints

- Length: 1 page max (~600 words standard, ~300 for short-form personal statements)
- File: Save as `<application-dir>/cover-letter.md`
- Don't repeat resume bullet-for-bullet -- add context the resume can't convey
- Don't copy-paste company statistics from their About page
