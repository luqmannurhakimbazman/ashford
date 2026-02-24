# Behavioral Interview Prepper — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a skill that chains off resume-builder output to generate a tailored behavioral interview answer bank.

**Architecture:** Skill reads `hojicha/<company>-<role>-resume/` (notes.md, resume.tex, candidate-context.md) and produces `behavioral-prep.md` in the same directory. Five reference files provide the frameworks; SKILL.md orchestrates the 7-step workflow. All company-specificity derived from JD at runtime.

**Tech Stack:** Claude Code skill (Markdown), no scripts or assets needed.

**Design doc:** `docs/plans/2026-02-24-behavioral-interview-prepper-design.md`

---

### Task 1: Create skill directory structure

**Files:**
- Create: `egg/skills/behavioral-interview-prepper/SKILL.md` (placeholder)
- Create: `egg/skills/behavioral-interview-prepper/references/` (empty dir)

**Step 1: Create directory and placeholder SKILL.md**

```bash
mkdir -p egg/skills/behavioral-interview-prepper/references
```

Create `egg/skills/behavioral-interview-prepper/SKILL.md` with minimal frontmatter:

```yaml
---
name: behavioral-interview-prepper
description: PLACEHOLDER — will be written in Task 7
---

# Behavioral Interview Prepper

TODO: Implement after references are complete.
```

**Step 2: Commit**

```bash
git add egg/skills/behavioral-interview-prepper/
git commit -m "chore(skill): scaffold behavioral-interview-prepper directory"
```

---

### Task 2: Write `references/behavioral-signals.md`

**Files:**
- Create: `egg/skills/behavioral-interview-prepper/references/behavioral-signals.md`

**Purpose:** Taxonomy of JD behavioral keywords → trait clusters. Used in Step 2 of the workflow.

**Content to include:**

1. **Signal extraction process** — How to scan a JD for behavioral/culture keywords (look for adjectives describing ideal candidate, team descriptions, company values sections, "you will" / "you are" phrases)

2. **Trait cluster taxonomy** — Map keywords to clusters:

| Trait Cluster | JD Signal Keywords |
|---|---|
| Leadership & Ownership | "ownership", "autonomy", "drive results", "end-to-end", "self-starter", "take initiative" |
| Collaboration | "cross-functional", "stakeholders", "team player", "partner with", "influence without authority" |
| Resilience & Ambiguity | "fast-paced", "ambiguity", "evolving priorities", "comfortable with uncertainty", "scrappy" |
| Technical Rigor | "first principles", "high bar", "scalable", "production-grade", "code quality" |
| Innovation & Curiosity | "creative problem solving", "think big", "experiment", "continuous learning", "cutting edge" |
| Communication | "communicate complex ideas", "executive presence", "written communication", "present to leadership" |
| Customer/Impact Focus | "customer obsession", "user-centric", "business impact", "data-driven decisions" |
| Growth Mindset | "feedback", "learn from mistakes", "mentorship", "coaching", "develop others" |

3. **FAANG-specific signal mapping** — Common company value → trait cluster mappings:
   - Amazon Leadership Principles → which clusters they map to
   - Google "Googliness" → collaboration + curiosity
   - Meta "Move Fast" → resilience + ownership
   - Apple "Secrecy + Craft" → technical rigor + communication

4. **Weighted extraction** — Count frequency of signals across the JD. Clusters with 3+ signals are primary; 1-2 signals are secondary. Primary clusters get 2-3 questions each, secondary get 1.

**Style:** Match `ats-keywords.md` — tables, numbered processes, practical examples. ~100-150 lines.

**Step 1: Write the file**

Write the complete reference file following the content outline above.

**Step 2: Commit**

```bash
git add egg/skills/behavioral-interview-prepper/references/behavioral-signals.md
git commit -m "feat(skill): add behavioral signals reference for interview prepper"
```

---

### Task 3: Write `references/question-bank.md`

**Files:**
- Create: `egg/skills/behavioral-interview-prepper/references/question-bank.md`

**Purpose:** Master list of behavioral questions organized by trait cluster. Used in Step 3 of the workflow.

**Content to include:**

1. **Question categories** — One section per trait cluster from behavioral-signals.md. Each section has 5-8 questions.

2. **Question metadata** — Each question tagged with:
   - Format type: behavioral / motivational / hypothetical / self-assessment
   - Difficulty: standard / probing (follow-up that interviewers use to dig deeper)
   - Common at: FAANG / quant / AI labs / general

3. **Categories and sample questions:**

   **Leadership & Ownership**
   - "Tell me about a time you took ownership of something outside your job description." [behavioral, standard, FAANG]
   - "Describe a project where you had to make a decision without your manager's input." [behavioral, standard, general]
   - Follow-up: "What would you do differently?" [probing]

   **Collaboration & Communication**
   - "Tell me about a time you had to convince someone who disagreed with you." [behavioral, standard, FAANG]
   - "Describe working with a difficult teammate." [behavioral, standard, general]

   **Resilience & Ambiguity**
   - "Tell me about a time you failed." [behavioral, standard, general]
   - "Describe a situation where requirements changed mid-project." [behavioral, standard, FAANG]

   **Technical Rigor**
   - "Walk me through a technically complex problem you solved." [behavioral, standard, AI labs]
   - "Tell me about a time you had to balance speed with quality." [behavioral, standard, FAANG]

   **Motivation & Fit**
   - "Why this company?" [motivational, standard, general]
   - "Why this role specifically?" [motivational, standard, general]
   - "Where do you see yourself in 5 years?" [motivational, standard, general]
   - "Tell me about yourself." [motivational, standard, general]

   **Self-Assessment**
   - "What's your greatest strength?" [self-assessment, standard, general]
   - "What's an area you're working to improve?" [self-assessment, standard, general]

   **Hypothetical/Situational**
   - "What would you do if you disagreed with your tech lead's design decision?" [hypothetical, standard, FAANG]
   - "How would you prioritize if you had three urgent tasks?" [hypothetical, standard, general]

4. **Finance-specific note** — "For trading/quant roles, also load `finance-behavioral.md` for additional questions."

**Style:** Clean tables or grouped lists. Tag each question clearly. ~150-200 lines.

**Step 1: Write the file**

Write the complete reference file following the content outline above.

**Step 2: Commit**

```bash
git add egg/skills/behavioral-interview-prepper/references/question-bank.md
git commit -m "feat(skill): add behavioral question bank reference"
```

---

### Task 4: Write `references/answer-formats.md`

**Files:**
- Create: `egg/skills/behavioral-interview-prepper/references/answer-formats.md`

**Purpose:** Templates for each answer type with examples. Used in Step 5 of the workflow.

**Content to include:**

1. **STAR with Metrics** (for behavioral questions)

   Template:
   ```
   **S (Situation):** [1-2 sentences setting the scene — company, team, context]
   **T (Task):** [What was your specific responsibility or challenge?]
   **A (Action):** [What did YOU do? Be specific — tools, decisions, steps. This is the longest section.]
   **R (Result):** [Quantifiable outcome. Use the same XYZ metrics from resume bullets.]
   ```

   Rules:
   - Action section should be 50-60% of the answer
   - Result MUST include a metric (mirrors the Y in XYZ bullets from resume)
   - Use "I" not "we" — interviewers want YOUR contribution
   - Target 90-120 seconds spoken delivery (~200-250 words)

   Example (before/after like xyz-formula.md):
   - Weak: "We built a data pipeline and it worked well."
   - Strong: Full STAR with metrics from a real resume bullet.

2. **Thesis Format** (for motivational questions — "Why X?")

   Template:
   ```
   [Opening hook — 1 sentence connecting your background to the company]

   Reason 1: [Specific company attribute + how it connects to your experience]
   Reason 2: [Specific role attribute + what excites you]
   Reason 3: [Growth/mission alignment + what you'd contribute]
   ```

   Rules:
   - Research the company — generic answers fail
   - Each reason must reference something specific (product, mission, recent news, team)
   - Connect reasons to YOUR experience, not just admiration
   - Target 60-90 seconds (~150-200 words)

3. **Framework-First** (for hypothetical questions)

   Template:
   ```
   [State your framework/approach — "I'd approach this in three steps..."]

   Step 1: [Action + reasoning]
   Step 2: [Action + reasoning]
   Step 3: [Action + reasoning]

   [Optional: "I've done something similar when..." — bridge to a real example]
   ```

   Rules:
   - Show structured thinking, not just an answer
   - Bridge to a real experience when possible (upgrades hypothetical to evidence)
   - Acknowledge trade-offs ("This depends on X — if A, I'd do B; if C, I'd do D")

4. **Claim + Proof + Growth** (for self-assessment questions)

   Template:
   ```
   Claim: [State the strength/weakness clearly]
   Proof: [Specific example demonstrating it]
   Growth: [How you're leveraging/improving it]
   ```

   Rules:
   - For weaknesses: choose a REAL weakness, not a humble brag
   - Growth angle must show concrete action (course, habit, system), not vague intent
   - For strengths: the proof must be a specific story, not a personality trait

5. **"Tell Me About Yourself"** (special case)

   Template:
   ```
   [Present]: What you're doing now — role, focus area, 1 key achievement
   [Past]: How you got here — relevant background, 1-2 transitions
   [Future]: Why this role — what draws you to the company/position
   ```

   Rules:
   - 60-90 seconds max
   - Tailor to the JD — emphasize the parts of your background most relevant to THIS role
   - End with a hook that invites follow-up

**Style:** Match `xyz-formula.md` — templates with rules, before/after examples. ~150-200 lines.

**Step 1: Write the file**

Write the complete reference file following the content outline above.

**Step 2: Commit**

```bash
git add egg/skills/behavioral-interview-prepper/references/answer-formats.md
git commit -m "feat(skill): add answer format templates reference"
```

---

### Task 5: Write `references/story-mapping.md`

**Files:**
- Create: `egg/skills/behavioral-interview-prepper/references/story-mapping.md`

**Purpose:** Guide for mapping resume experiences to behavioral questions. Used in Step 4 of the workflow.

**Content to include:**

1. **Story extraction process**
   - Read each resume bullet from `resume.tex` as a potential story source
   - Read `candidate-context.md` for supplementary experiences not on the resume
   - For each experience, identify which trait clusters it demonstrates

2. **Story-to-question mapping rules**
   - Each story can serve 2-3 questions max (reframed with different emphasis)
   - No story should appear more than 3 times in the answer bank
   - Prioritize stories from the most relevant/recent roles
   - If a story maps to multiple clusters, assign it to the cluster with fewest stories

3. **Reframing technique**
   - Same experience, different STAR emphasis:
     - Leadership angle: emphasize the decision you made and why
     - Collaboration angle: emphasize the stakeholders and how you aligned them
     - Technical angle: emphasize the approach and trade-offs
   - Example: "Built ML pipeline" can answer ownership, technical rigor, or ambiguity depending on which STAR elements you emphasize

4. **Gap handling**
   - When no experience maps to a predicted question, three strategies:
     1. **Adjacent experience**: Find a related but not exact experience. Acknowledge the gap: "I haven't done X specifically, but in a similar situation at Y..."
     2. **Transferable skill**: Frame a different domain experience as evidence of the underlying trait
     3. **Honest deflection**: "I haven't encountered that situation professionally, but here's how I'd approach it..." — then use framework-first format
   - Map gaps to the gap analysis from resume-builder's `notes.md` — if resume-builder flagged a gap, the behavioral prep should have a deflection ready

5. **Story bank table format**

   ```markdown
   | # | Story (short name) | Source | Trait Clusters | Questions Assigned |
   |---|---|---|---|---|
   | S1 | ML pipeline at Company X | resume bullet #3 | Technical Rigor, Ownership | Q1, Q4, Q7 |
   | S2 | Cross-team API migration | candidate-context | Collaboration, Resilience | Q2, Q8 |
   ```

**Style:** Process-oriented with clear rules and a template. ~80-120 lines.

**Step 1: Write the file**

Write the complete reference file following the content outline above.

**Step 2: Commit**

```bash
git add egg/skills/behavioral-interview-prepper/references/story-mapping.md
git commit -m "feat(skill): add story mapping reference for interview prepper"
```

---

### Task 6: Write `references/finance-behavioral.md`

**Files:**
- Create: `egg/skills/behavioral-interview-prepper/references/finance-behavioral.md`

**Purpose:** Trading/quant-specific behavioral questions and answer guidance. Loaded conditionally in Step 6.

**Content to include:**

1. **When to load this file** — If the JD mentions: trading, quant, hedge fund, prop shop, asset management, S&T, rates, FX, equities, commodities, market making, alpha, PnL, risk.

2. **Finance-specific behavioral questions** (not covered in the general question bank):

   **Why Trading / Why Finance?**
   - "Why trading and not software engineering?"
   - "Why this firm and not [competitor]?"
   - "What's the difference between a trader's mindset and an engineer's mindset?"

   **Risk & Decision-Making**
   - "Describe a time you took a calculated risk."
   - "How do you make decisions with incomplete information?"
   - "Tell me about a time you were wrong and had to change course quickly."

   **Market Awareness**
   - "What's the most interesting thing happening in markets right now?"
   - "If I gave you $1M to invest today, what would you do?"
   - "What trade are you following right now?"

   **Quantitative Thinking**
   - "Walk me through how you'd evaluate a trading strategy."
   - "How would you estimate [market size / probability / fair value]?"

3. **Answer guidance for finance behavioral**
   - Show genuine market interest — reference specific instruments, not vague sectors
   - Demonstrate risk-reward thinking — quantify trade-offs
   - Admit uncertainty — "I don't know but here's how I'd think about it" is strong
   - Connect engineering skills to trading value — "My ML background helps with signal extraction from noisy data"

4. **Trader mindset signals table** (adapted from global-markets-teacher's `interview-fit-behavioral.md` but focused on answer generation rather than teaching):

   | Signal | What They Want to Hear | Red Flag to Avoid |
   |---|---|---|
   | Quick decision-making | Comfort making calls with incomplete info | "I'd need more data first" |
   | Intellectual curiosity | Genuine market interest with specifics | Generic "I like finance" |
   | Admitting when wrong | "I was wrong because X, learned Y" | Never wrong, blames others |

**Style:** Match the other reference files. ~100-130 lines.

**Step 1: Write the file**

Write the complete reference file following the content outline above.

**Step 2: Commit**

```bash
git add egg/skills/behavioral-interview-prepper/references/finance-behavioral.md
git commit -m "feat(skill): add finance behavioral reference for interview prepper"
```

---

### Task 7: Write `SKILL.md`

**Files:**
- Modify: `egg/skills/behavioral-interview-prepper/SKILL.md` (replace placeholder)

**Purpose:** Main skill file with frontmatter, critical rules, and 7-step workflow.

**Content to include:**

1. **Frontmatter:**

```yaml
---
name: behavioral-interview-prepper
description: This skill should be used when the user wants to prepare for behavioral interviews after tailoring a resume. Trigger phrases include "prep behavioral", "behavioral interview prep", "prep me for interview at", "practice behavioral questions", "generate behavioral answers", "behavioral prep for", "interview stories for", or when a user has completed a resume-builder run and asks for interview preparation. It chains off resume-builder output to generate a tailored answer bank with structured responses mapped to the candidate's real experiences.
---
```

2. **Header:**
```markdown
# Behavioral Interview Prepper

Generate a tailored behavioral interview answer bank from resume-builder output. Output goes to the same `hojicha/<company>-<role>-resume/` directory as `behavioral-prep.md`.
```

3. **Critical Rules:**
   1. NEVER fabricate experiences — only use content from the resume and candidate-context.md
   2. Chain from resume-builder output — read existing notes.md and resume.tex
   3. Honest gap handling — provide deflection strategies, not made-up stories
   4. Story reuse limit — no single experience used for more than 3 questions

4. **Workflow (Steps 1-7)** — Concise version of the design doc workflow, with `See references/<file>.md` pointers at each step. Keep each step to 3-5 lines in SKILL.md.

5. **Output structure** — The `behavioral-prep.md` template from the design doc.

6. **Quick Reference table:**

| Question Type | Answer Format | Reference |
|---|---|---|
| "Tell me about a time..." | STAR with metrics | `references/answer-formats.md` |
| "Why this company/role?" | Thesis (3 reasons) | `references/answer-formats.md` |
| "What would you do if..." | Framework-first | `references/answer-formats.md` |
| "Strengths/weaknesses" | Claim + proof + growth | `references/answer-formats.md` |
| "Tell me about yourself" | Present → Past → Future | `references/answer-formats.md` |

**Target:** ~150-200 lines. All detailed content lives in references.

**Step 1: Write the complete SKILL.md**

Replace the placeholder with the full skill file following the content outline above.

**Step 2: Commit**

```bash
git add egg/skills/behavioral-interview-prepper/SKILL.md
git commit -m "feat(skill): write behavioral-interview-prepper SKILL.md"
```

---

### Task 8: Integration test — run skill against real resume-builder output

**Files:**
- Read: An existing `hojicha/<company>-<role>-resume/` directory (pick one that exists)
- Create: `behavioral-prep.md` in that directory (test output)

**Step 1: Verify a resume-builder output directory exists**

```bash
ls hojicha/*/notes.md
```

Pick one. If none exist, create a minimal test fixture with a sample notes.md and resume.tex.

**Step 2: Run the skill manually**

Trigger the skill with: "prep behavioral for [company] [role]" pointing at the existing resume-builder output. Verify:
- [ ] Skill reads notes.md, resume.tex, candidate-context.md
- [ ] Behavioral signals extracted from JD
- [ ] Questions selected and weighted by signal frequency
- [ ] Stories mapped from resume bullets (no fabrication)
- [ ] Answers use correct format per question type
- [ ] Finance layer loads only for finance roles
- [ ] Gap awareness section maps to resume-builder gap analysis
- [ ] Output saved as behavioral-prep.md in correct directory

**Step 3: Review output quality**

Check that:
- Every answer traces back to a real resume bullet or candidate-context entry
- No story used more than 3 times
- STAR answers have metrics
- "Why company?" answer references specific company details from JD
- Gap deflections are honest, not fabricated

**Step 4: Commit test output (if keeping as example)**

```bash
git add hojicha/<company>-<role>-resume/behavioral-prep.md
git commit -m "test(skill): validate behavioral-interview-prepper output"
```

---

### Task 9: Final commit — all references and SKILL.md together

**Step 1: Verify all files exist**

```bash
ls -la egg/skills/behavioral-interview-prepper/
ls -la egg/skills/behavioral-interview-prepper/references/
```

Expected:
```
SKILL.md
references/
  behavioral-signals.md
  question-bank.md
  answer-formats.md
  story-mapping.md
  finance-behavioral.md
```

**Step 2: Verify SKILL.md frontmatter is valid**

Check that name and description fields are present and description starts with "This skill should be used when".

**Step 3: Final commit if any cleanup was needed**

```bash
git add egg/skills/behavioral-interview-prepper/
git commit -m "feat(skill): complete behavioral-interview-prepper skill with all references"
```
