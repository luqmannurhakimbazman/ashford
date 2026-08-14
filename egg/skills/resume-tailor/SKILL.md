---
name: resume-tailor
description: This skill should be used when the user has already run resume-analyzer and wants to generate the tailored resume.tex. Trigger phrases include "generate resume", "write the resume", "create resume.tex", "tailor the resume now", "build the resume from notes", or when the user asks to proceed after a resume analysis session. It reads the notes.md produced by resume-analyzer and generates a tailored LaTeX resume.
---

# Resume Tailor

Generate a tailored `resume.tex` from analysis notes. Reads `<application-dir>/notes.md` (produced by `resume-analyzer`) and the user's master resume.

**Prerequisite:** `notes.md` must exist in the output directory. If missing, tell the user to run `resume-analyzer` first.

## Resume Workspace

Resolve the inputs before tailoring:

1. Use a resume workspace or application directory supplied by the user.
2. Otherwise, default `<resume-root>` to `${CLAUDE_PROJECT_DIR}/resumes` and resolve `<application-dir>` as `<resume-root>/<company>-<role>-resume/`.
3. If `${CLAUDE_PROJECT_DIR}` is unavailable or required inputs are absent, ask the user for the application directory and individual file paths. Never assume a machine-specific directory.

By convention, the master resume is `<resume-root>/resume.tex`, candidate context is `<resume-root>/candidate-context.md`, and the document class is `<resume-root>/fed-res.cls`. Explicit user-provided paths override these conventions.

## Critical Rules

1. **NEVER fabricate experiences, skills, or achievements.** Only use content from the master resume, `candidate-context.md`, and discoveries persisted during analysis.
2. **Preserve the `fed-res.cls` document class.** Do not modify `\documentclass[letterpaper,12pt]{fed-res}` or add packages. Copy the resolved `fed-res.cls` into the output directory.
3. **Maintain ATS compatibility.** No graphics, tables outside the cls structure, or custom fonts.
4. **Keep to one page.** Highlight only your strongest 2-3 bullets per role. Fewer strong bullets beat many mediocre ones.
5. **Never generate generic content -- ask instead.** If a bullet would be vague, ask for real specifics.

## Workflow

### Step 1: Read Inputs

```
Required:
- notes.md from the output directory (keyword analysis, gap analysis, recommendations)
- Master resume: user-provided path or `<resume-root>/resume.tex`
- Candidate context: user-provided path or `<resume-root>/candidate-context.md`
- Document class: user-provided path or `<resume-root>/fed-res.cls`
```

### Step 2: Professional Summary (Optional)

Generate a role-specific summary if space permits. Omit if the resume already speaks for itself.

- Headline: <10 words, descriptive noun phrase (NOT "Professional Summary")
- Summary: <50 words starting with a job role noun, top 3-5 selling points matching JD
- Don't claim to "specialize" in too many things
- Don't state your objective unless you're a career changer

### Step 3: Section Ordering

| Experience Level | Order |
|------------------|-------|
| <3 years / recent grad | Summary, Education, Experience, Projects, Skills |
| 3+ years | Summary, Experience, Education, Projects, Skills |

Use the recommendation from `notes.md` if present.

### Step 4: XYZ Bullet Optimization

Rewrite experience bullets using the XYZ formula:

```
Accomplished [X] as measured by [Y], by doing [Z]
```

- Lead with the accomplishment, not the task
- Quantify whenever possible (numbers, percentages, scale, scope)
- Weave target keywords into the Z component naturally
- One bullet = one accomplishment
- Strong action verbs: Architected, Engineered, Developed, Built, Led, Delivered, Automated, Optimized, Reduced

Keyword placement priority:
1. Professional summary / first bullet of most relevant role
2. Most recent experience section
3. Projects section
4. Skills section (highest ATS hit rate -- ensure keywords appear here AND in contextual bullets)

### Step 5: Strategic Uncommenting & Commenting

Review commented-out sections in the master resume. Uncomment entries that strengthen the application per `notes.md` recommendations. Comment out entries that weaken the narrative. Link project names to GitHub repos where possible.

### Step 6: Skills Reordering

Reorder skills categories and items to front-load the most relevant ones. First items in each line are what ATS and recruiters see first.

### Step 7: Output

Create `<application-dir>/resume.tex`:

- Copy master resume structure exactly, applying all modifications
- Include `\documentclass[letterpaper,12pt]{fed-res}`
- Copy the resolved `fed-res.cls` into the output directory

**Verify:** All meaningful content is text (not layout-dependent), special characters render when LaTeX is stripped, acronyms expanded at least once.

Append a "Changes Made" section to `notes.md` listing all modifications.

## fed-res.cls Commands

| Command | Usage |
|---------|-------|
| `\resumeSubheading{Org}{Loc}{Title}{Date}` | Experience/education entry |
| `\resumeSubheadingShort{Title}{Date}` | Entry without location |
| `\resumeSubSubheading{Title}{Date}` | Sub-position at same org |
| `\resumeProjectHeading{Title \| Tech}{Date}` | Project entry |
| `\resumeItem{text}` | Bulleted item |
| `\resumeSubItem{text}` | Compact bullet variant |
| `\resumeSubHeadingListStart/End` | Wrap subheading groups |
| `\resumeItemListStart/End` | Wrap bullet lists |

**Skills section pattern:**

```latex
\section{Skills and Interests}
    \begin{itemize}[leftmargin=0.15in, label={}]
        {\item{
            \textbf{Category}{: item1, item2, item3} \\
            \textbf{Category}{: item1, item2, item3}
        }}
    \end{itemize}
```

**Header pattern:**

```latex
\begin{center}
    \textbf{\Huge \scshape Name} \\ \vspace{1pt}
    \underline{email} \ $|$ \ \underline{linkedin} \ $|$ \
    \underline{github} \\ \vspace{1pt}
\end{center}
```

**Commenting:** Prefix every line with `%` to hide a section. Remove `%` to restore.

Do NOT add packages to `resume.tex` -- all formatting is handled by `fed-res.cls`.
