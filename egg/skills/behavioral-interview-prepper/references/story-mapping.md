# Story Mapping

## Overview
Guide for mapping resume experiences to predicted behavioral questions. After questions are predicted (Step 4), use this reference to find the best experience for each question, reframe stories for multiple clusters, and identify gaps.

## Story Extraction Process

Extract every potential story from the candidate's materials.

1. **Resume bullets** — Read each bullet from `resume.tex` as a discrete story source. Each bullet = one potential STAR answer.
2. **Candidate context** — Read `candidate-context.md` for supplementary experiences not on the resume (side projects, volunteer work, academic research, earlier roles).
3. **Trait tagging** — For each extracted experience, identify which trait clusters (from `behavioral-signals.md`) it demonstrates. Most experiences map to 1-2 primary clusters and 1 secondary.

Tag format per story:
- What happened (the situation/task)
- What the candidate did (the action)
- What changed (the result)
- Which clusters it naturally serves

## Story-to-Question Mapping Rules

Apply these constraints when assigning stories to predicted questions:

1. **Reuse cap**: Each story can serve 2-3 questions max, reframed with different emphasis per question.
2. **Hard ceiling**: No story should appear more than 3 times in the final answer bank.
3. **Recency bias**: Prioritize stories from the most recent and most relevant roles. Stories older than 5 years should only fill gaps.
4. **Role relevance**: Stories from roles similar to the target role outrank stories from unrelated roles, even if the unrelated story is more recent.
5. **Cluster balancing**: If a story maps to multiple clusters, assign it to the cluster with the fewest stories first. This prevents cluster starvation.
6. **Uniqueness spread**: Across all primary clusters, aim for at least 2 distinct stories each. Avoid a single "hero story" carrying the entire bank.

## Reframing Technique

The same experience can answer different behavioral questions by shifting STAR emphasis.

### Angle Shifts

| Angle | Emphasis in STAR | Lead With |
|---|---|---|
| Leadership | The decision you made and why | "I decided to..." / "I took ownership of..." |
| Collaboration | The stakeholders and how you aligned them | "I brought together..." / "I partnered with..." |
| Technical | The approach, trade-offs, and methodology | "I evaluated X vs Y and chose..." |
| Resilience | The challenge, ambiguity, or setback and how you navigated it | "The situation was unclear because..." / "When X broke..." |

### Reframing Example

**Raw experience:** Led migration of ML pipeline from batch to real-time inference, coordinating with platform and product teams, reducing latency from 6 hours to 200ms.

**Leadership angle (Q: "Tell me about a time you drove a major technical decision"):**
- S/T: Our batch pipeline created a 6-hour delay for user recommendations.
- A: I proposed the migration to real-time inference, built the business case showing revenue impact, and got VP approval to prioritize it.
- R: Latency dropped from 6 hours to 200ms; recommendation click-through improved 15%.

**Collaboration angle (Q: "Describe a time you worked cross-functionally"):**
- S/T: The migration required changes across ML, platform infrastructure, and product teams.
- A: I set up a shared design doc, ran weekly syncs with all three teams, and negotiated SLA tradeoffs with the platform team to unblock the rollout.
- R: Shipped on schedule with zero cross-team escalations; became the template for future cross-team projects.

**Resilience angle (Q: "Tell me about a time you dealt with ambiguity"):**
- S/T: No one had done real-time ML inference at our scale; there was no playbook and the platform team flagged reliability concerns.
- A: I ran a 2-week spike to benchmark three architectures, presented risk/reward tradeoffs to leadership, and built a rollback plan to de-risk the launch.
- R: Launched successfully; the rollback plan was never needed but earned trust from the platform team for future migrations.

## Gap Handling

When no experience maps to a predicted question, use these strategies in priority order:

1. **Adjacent experience** — A related but not exact match. Frame it explicitly: "I haven't done X specifically, but in a similar situation at Y, I [action] which required the same [trait]."
2. **Transferable skill** — Different domain, same underlying trait. A story about debugging a production outage can demonstrate the same resilience as navigating org ambiguity.
3. **Honest deflection** — When no experience is close enough: "I haven't encountered that professionally, but here's how I'd approach it based on [framework/principle]." Use a framework-first format: state the principle, then walk through how you'd apply it.
4. **Gap-to-development bridge** — If `notes.md` from resume-builder identified this as a resume gap, reference it directly. The candidate can proactively acknowledge the gap and pivot to a learning narrative.

Flag every gap in the story bank table so the candidate knows which answers need extra rehearsal.

## Story Bank Table Format

Output the final mapping as a table. This is the deliverable for the story-mapping step.

```markdown
| # | Story | Source | Trait Clusters | Questions Assigned | Gap? |
|---|---|---|---|---|---|
| S1 | ML pipeline migration at Company X | resume bullet #3 | Technical Rigor, Leadership & Ownership | Q1, Q4, Q7 | — |
| S2 | Cross-team API redesign | resume bullet #7 | Collaboration, Communication | Q2, Q5 | — |
| S3 | Onboarding mentorship program | candidate-context.md | Growth Mindset | Q8 | — |
| S4 | — | — | Resilience & Ambiguity | Q3 | GAP: adjacent experience only |
```

Rules for the table:
- Number stories sequentially (S1, S2, ...).
- **Source** must reference the exact resume bullet number or `candidate-context.md`.
- **Trait Clusters** lists the clusters the story serves across all assigned questions.
- **Questions Assigned** references question numbers from the predicted question list.
- **Gap?** is blank if the story is a direct match. If it required gap handling, note which strategy was used.
- Sort by question coverage descending (stories covering the most questions first).
