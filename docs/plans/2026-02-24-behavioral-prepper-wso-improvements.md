# Behavioral Interview Prepper — WSO Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enhance the behavioral-interview-prepper skill with insights from a WallStreetOasis behavioral crash course post: add resume walkthrough reference, SOAR answer format, dynamic questions-for-interviewer generation, and raise story reuse cap.

**Architecture:** 4 targeted edits to existing skill files + 1 new reference file. No structural changes to the workflow — just extending it with a new step and enriching existing references.

**Tech Stack:** Claude Code skill (Markdown), no scripts or assets needed.

**Design doc:** `docs/plans/2026-02-24-behavioral-interview-prepper-design.md` (WSO section at bottom)

---

### Task 1: Create `references/resume-walkthrough.md`

**Files:**
- Create: `egg/skills/behavioral-interview-prepper/references/resume-walkthrough.md`

**Step 1: Write the file**

Create `egg/skills/behavioral-interview-prepper/references/resume-walkthrough.md` with:

```markdown
# Resume Walkthrough

## Overview

Guide for crafting the 90-second resume narrative — the "tell me about yourself" or "walk me through your resume" opener. This is the first impression and sets the tone for the entire interview. A dialed-in walkthrough makes the interviewer predisposed to like you before the first behavioral question.

## The 3-Act Framework

Structure the walkthrough as a narrative arc with positive, deliberate transitions:

### Act 1: Where You Started (15-20 seconds)
- Anchor in your earliest relevant professional context
- One sentence on what drew you to that field
- Example: "I studied computer science at [University] because I was fascinated by how systems process information at scale."

### Act 2: Key Transitions (40-50 seconds, ~20s per transition)
- Cover 2-3 career moves, each with a **positive motivating reason**
- Every transition must sound intentional and self-initiated
- Pattern: "I wanted to [grow/deepen/expand] my [skill/exposure/impact], so I [sought/pursued/moved to]..."

### Act 3: Why You're Here Now (15-20 seconds)
- Land on the current role/interview target
- Connect your trajectory to THIS specific opportunity
- End with a hook that invites a follow-up question

**Total: 90 seconds / ~225 words**

## Transition Framing Rules

Every career move must have a positive, forward-looking reason. The interviewer should think "this person is deliberate about their career."

### Strong Transition Phrases
- "I wanted to deepen my exposure to X, so I pursued..."
- "After developing strong skills in A, I sought a role where I could apply them to B..."
- "I was drawn to [Company] because they offered the chance to work on C at scale..."
- "I realized I was most energized by D, which led me to..."

### Never Say (Anti-Patterns)
| Weak Framing | Why It Fails | Strong Alternative |
|---|---|---|
| "I was bored" | Signals you get bored easily | "I'd mastered the core challenges and wanted to grow into X" |
| "Wanted to try something new" | No direction, sounds aimless | "I was drawn to Y because of my experience with Z" |
| "Left because of bad management" | Negative, blames others | "I sought a team culture that aligned with my collaborative working style" |
| "The company was struggling" | Sounds like you flee problems | "I was looking to apply my skills in a higher-growth environment" |
| "The money was better" | Signals mercenary mindset | "The role offered both a technical challenge and career growth" |
| "I got laid off" | Even if true, reframe it | "When the team was restructured, I took the opportunity to pursue X" |

## Worked Examples

### Example 1: Tech → Tech (Backend → ML)

> I started as a backend engineer at [Startup] building real-time data pipelines — I was drawn to the challenge of processing millions of events reliably. After two years, I realized the most impactful part of my work was the data modeling layer, which led me to pursue machine learning more formally. I moved to [Company] where I could combine my systems engineering background with ML — building production inference pipelines that serve 50M daily predictions. That intersection of systems and ML is exactly what excites me about this role: you're building the infrastructure that makes ML work at scale, and my background sits right at that intersection.

### Example 2: Tech → Finance (SWE → Quant)

> I spent four years building low-latency trading systems at [Tech Company], where I developed a deep appreciation for how milliseconds of optimization translate to real P&L impact. That experience sparked my interest in the quantitative side — not just building the systems, but understanding the strategies they execute. I completed a master's in financial engineering to formalize that interest, focusing on statistical arbitrage and market microstructure. Now I'm looking to bring both skill sets together: my engineering ability to build production-grade systems and my quantitative training to develop and deploy trading strategies. Your firm's focus on systematic trading with a strong technology culture is exactly that combination.

### Example 3: Career Changer (Non-Tech → Tech)

> I started my career in management consulting at [Firm], where I spent three years solving operational problems for Fortune 500 clients. I loved the analytical rigor but found myself consistently gravitating toward the technical implementation side — I kept volunteering for the data analysis workstreams and eventually taught myself Python to automate our reporting. That hands-on experience convinced me to make the switch to engineering full-time. I completed [Bootcamp/Degree] and joined [Company] as a data engineer, where my consulting background in stakeholder management and structured problem-solving turned out to be a major advantage in a cross-functional team. This role appeals to me because it combines technical depth with the kind of business impact I valued in consulting.

## Common Mistakes

1. **Too long (>2 minutes):** The walkthrough is a warm-up, not your life story. If you're past 90 seconds, cut details.
2. **Too chronological:** Don't narrate every job. Pick the 2-3 transitions that build toward THIS role.
3. **No narrative arc:** Random facts about each job ≠ a story. Each transition should build on the previous one.
4. **Missing the landing:** The walkthrough must end by connecting to the role you're interviewing for. Don't trail off.
5. **Reciting the resume:** The interviewer has your resume. The walkthrough adds the WHY behind the WHAT.
6. **Negative framing:** One negative transition poisons the whole narrative. Always move toward, never away.
```

**Step 2: Commit**

```bash
git add egg/skills/behavioral-interview-prepper/references/resume-walkthrough.md
git commit -m "feat(skill): add resume walkthrough reference with 90-second narrative framework"
```

---

### Task 2: Add SOAR format to `references/answer-formats.md`

**Files:**
- Modify: `egg/skills/behavioral-interview-prepper/references/answer-formats.md:183-184` (after Present-Past-Future section, before Format Selection Guide)

**Step 1: Add SOAR section**

Insert after line 184 (the closing of the Present-Past-Future strong example) and before line 186 (`## Format Selection Guide`):

```markdown

---

## 6. SOAR (Situation-Obstacle-Action-Result)

**Use for:** Behavioral questions where the **challenge is the strongest element** of the story — "Tell me about a time you overcame...", "Describe a difficult situation...", "Tell me about a time you failed..."

SOAR is an alternative to STAR that explicitly names the obstacle before the action, creating stronger narrative tension. Use SOAR when the "why this was hard" matters more than the task assignment.

### Template

```
S (Situation): [1-2 sentences — company, team, context. Set the scene fast.] (10-15 seconds / 25-40 words)
O (Obstacle): [What made this hard? The specific challenge, constraint, or conflict.] (10-15 seconds / 25-40 words)
A (Action): [What YOU did to overcome the obstacle. Decisions, trade-offs, steps.] (60-75 seconds / 150-190 words)
R (Result): [Quantifiable outcome + what you learned.] (15-30 seconds / 40-75 words)
```

**Total: 1.5-2 minutes spoken / 250-350 words written**

### Rules

1. **Obstacle ≠ Task** — In STAR, the Task is your assignment. In SOAR, the Obstacle is what made the assignment hard. "I was asked to migrate the pipeline" is a task. "The migration had no runbook, the legacy system had no documentation, and the team had never done real-time inference" is an obstacle.
2. **Action = 50-60% of the answer** — same as STAR, but now the action explicitly addresses the obstacle
3. **Result includes a learning** — SOAR stories often end with "and I learned X" because the obstacle-driven narrative naturally leads to reflection
4. **Use when STAR feels flat** — if your STAR answer's "Task" section is the most interesting part, switch to SOAR
5. **Target 1.5-2 minutes** — slightly longer than STAR because the obstacle setup adds context

### When to Use SOAR vs STAR

| Scenario | Use |
|---|---|
| Clear assignment with strong results | STAR |
| Challenge/failure/adversity is the hook | SOAR |
| "Tell me about a time you failed" | SOAR |
| "Describe a difficult situation" | SOAR |
| "Tell me about a time you led a team" | STAR (unless the leadership challenge was extreme) |
| "How did you handle ambiguity?" | SOAR |

### Before / After

#### Weak (obstacle buried in the action)

> At my company, I was asked to migrate our ML pipeline from batch to real-time. I had to figure out the architecture, coordinate with three teams, and deal with the fact that there was no documentation. I chose a streaming approach, wrote the docs myself, and got it working. Latency went from 6 hours to 200ms.

Problems: the obstacle is mixed into the action, no narrative tension, feels like a flat list of things that happened.

#### Strong (SOAR with explicit obstacle)

> **S:** At [Company], our ML pipeline ran batch predictions every 6 hours, meaning user recommendations were always stale by the time they were served. **O:** The migration to real-time had three blockers: the legacy system had zero documentation, no one on the team had built streaming infrastructure before, and the platform team flagged reliability concerns that could block the project entirely. **A:** I tackled each blocker sequentially. First, I spent a week reverse-engineering the legacy system and producing the first-ever architecture doc, which unblocked the team to start designing in parallel. Second, I ran a 2-week spike benchmarking three streaming architectures — Kafka Streams, Flink, and a custom solution — and presented a risk/reward comparison to leadership that secured buy-in for Kafka Streams. Third, I built a comprehensive rollback plan with automated health checks that addressed every concern the platform team had raised, converting them from blockers to advocates. I ran weekly cross-team syncs to keep all three teams aligned. **R:** Launched on schedule with zero rollback needed. Latency dropped from 6 hours to 200ms, recommendation click-through improved 15%, and the rollback plan template I created became the standard for all future migrations. I learned that the biggest risk in a migration isn't the technology — it's the organizational alignment.
```

**Step 2: Update Format Selection Guide**

At line 192, add a SOAR row to the table:

```markdown
| Behavioral (adversity/failure) | SOAR | "Overcome", "Difficult", "Failed", "Struggled", "Dealt with" |
```

**Step 3: Commit**

```bash
git add egg/skills/behavioral-interview-prepper/references/answer-formats.md
git commit -m "feat(skill): add SOAR answer format with timing targets and STAR comparison"
```

---

### Task 3: Bump story reuse cap in `references/story-mapping.md`

**Files:**
- Modify: `egg/skills/behavioral-interview-prepper/references/story-mapping.md:24-25` (reuse cap rules)
- Modify: `egg/skills/behavioral-interview-prepper/references/story-mapping.md:79-85` (story bank table)

**Step 1: Update reuse rules**

Replace lines 24-25:

```
1. **Reuse cap**: Each story can serve 2-3 questions max, reframed with different emphasis per question.
2. **Hard ceiling**: No story should appear more than 3 times in the final answer bank.
```

With:

```
1. **Reuse cap**: Each story can serve up to 5 questions, reframed with different emphasis per question.
2. **Hard ceiling**: No story should appear more than 5 times in the final answer bank.
3. **Reframe requirement for uses 4-5**: Uses beyond the 3rd must shift to a **different trait cluster emphasis** than uses 1-3. This prevents the interviewer from hearing the same story with superficial rewording. The philosophy: 6-8 core stories should cover your full question set. Slight modifications in framing — not new stories — are how experienced candidates scale.
```

**Step 2: Update story bank table format**

Replace lines 79-85:

```markdown
| # | Story | Source | Trait Clusters | Questions Assigned | Gap? |
|---|---|---|---|---|---|
| S1 | ML pipeline migration at Company X | resume bullet #3 | Technical Rigor, Leadership & Ownership | Q1, Q4, Q7 | — |
| S2 | Cross-team API redesign | resume bullet #7 | Collaboration, Communication | Q2, Q5 | — |
| S3 | Onboarding mentorship program | candidate-context.md | Growth Mindset | Q8 | — |
| S4 | — | — | Resilience & Ambiguity | Q3 | GAP: adjacent experience only |
```

With:

```markdown
| # | Story | Source | Trait Clusters | Questions Assigned | Reframe Angle (uses 4-5) | Gap? |
|---|---|---|---|---|---|---|
| S1 | ML pipeline migration at Company X | resume bullet #3 | Technical Rigor, Leadership & Ownership, Resilience | Q1, Q4, Q7, Q11, Q13 | Q11: Collaboration (cross-team coordination), Q13: Resilience (no-documentation obstacle) | — |
| S2 | Cross-team API redesign | resume bullet #7 | Collaboration, Communication | Q2, Q5 | — | — |
| S3 | Onboarding mentorship program | candidate-context.md | Growth Mindset | Q8 | — | — |
| S4 | — | — | Resilience & Ambiguity | Q3 | — | GAP: adjacent experience only |
```

**Step 3: Commit**

```bash
git add egg/skills/behavioral-interview-prepper/references/story-mapping.md
git commit -m "feat(skill): raise story reuse cap to 5 with reframe requirement"
```

---

### Task 4: Add Questions-for-Interviewer step and update SKILL.md

**Files:**
- Modify: `egg/skills/behavioral-interview-prepper/SKILL.md:52-55` (insert new Step 7)
- Modify: `egg/skills/behavioral-interview-prepper/SKILL.md:57-66` (update output step numbering)
- Modify: `egg/skills/behavioral-interview-prepper/SKILL.md:72-128` (update output template)
- Modify: `egg/skills/behavioral-interview-prepper/SKILL.md:132-155` (update quick reference)

**Step 1: Insert new Step 7 in workflow**

After the current Step 6 (Finance Layer) at line 55, insert:

```markdown
### Step 7: Generate Questions for Interviewer

Generate 5-8 tailored questions the candidate should ask their interviewer, based on JD signals, company context, and role type. Organize by audience:

- **For Technical/Hiring Manager interviews:** Questions about current projects, team challenges, technical stack decisions, and what success looks like in the role. Lead with "tell me what you're working on now" — if interesting, this can fill 5-10 minutes of natural conversation.
- **For Team Member/Culture interviews:** Questions about team dynamics, collaboration patterns, and day-to-day experience.
- **Role-specific questions:** Questions that demonstrate you've read the JD carefully and thought deeply about the role.
- **Avoid asking (save for recruiter):** Timing/logistics for next rounds, compensation, benefits, travel. These should be directed to the recruiting coordinator, not the interviewer.

**Anti-patterns:** Don't ask about things you should already know from 3 months of networking (staffing model, industry mix, training program). Show you've done your homework by asking questions that go deeper.
```

**Step 2: Renumber current Step 7 to Step 8**

Current Step 7 (Output) becomes Step 8. Update the heading.

**Step 3: Add resume-walkthrough reference to Step 5**

At the end of Step 5 (Generate Answers), add:

```markdown
For the "Tell me about yourself" / resume walkthrough answer, also reference `references/resume-walkthrough.md` for the 90-second narrative arc framework, positive transition framing, and anti-patterns.
```

**Step 4: Update output template**

Add this section to the `behavioral-prep.md` template, after the Gap Awareness section and before the closing:

```markdown
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

**Step 5: Update Quick Reference table**

Add SOAR row and resume walkthrough reference:

```markdown
| "Tell me about a time..." (adversity) | SOAR with obstacle emphasis | `references/answer-formats.md` |
| "Walk me through your resume" | 90-second narrative arc | `references/resume-walkthrough.md` |
```

**Step 6: Update story reuse cap reference**

Change the Critical Rule #4 from:

```
4. **Story reuse limit.** No single experience may be used for more than 3 questions. Spread stories across the answer bank.
```

To:

```
4. **Story reuse limit.** No single experience may be used for more than 5 questions. Uses 4-5 require a distinct reframing angle (different trait cluster emphasis). See `references/story-mapping.md`.
```

**Step 7: Update Trait Cluster Summary reference list**

Add to the bottom of the file:

```markdown
### Resume Walkthrough

See `references/resume-walkthrough.md` for the 90-second narrative arc framework with positive transition framing, anti-patterns, and worked examples.
```

**Step 8: Commit**

```bash
git add egg/skills/behavioral-interview-prepper/SKILL.md
git commit -m "feat(skill): add questions-for-interviewer step and SOAR/walkthrough references to SKILL.md"
```

---

### Task 5: Update trigger tests

**Files:**
- Modify: `egg/skills/behavioral-interview-prepper/evaluations/trigger-tests.md`

**Step 1: Add trigger test cases for new features**

Add test cases that verify:
- "help me prep questions to ask my interviewer at Google" → triggers skill
- "walk me through your resume prep" → triggers skill
- "practice my resume walkthrough" → triggers skill

**Step 2: Commit**

```bash
git add egg/skills/behavioral-interview-prepper/evaluations/trigger-tests.md
git commit -m "test(skill): add trigger tests for walkthrough and interviewer questions"
```
