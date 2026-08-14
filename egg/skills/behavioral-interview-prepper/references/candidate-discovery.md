# Candidate Discovery

## Overview

This reference defines how to probe the candidate for real experiences instead of generating generic content. Both `resume-analyzer` and `behavioral-interview-prepper` reference this file. When in doubt, ASK — the only way to produce authentic output is to understand the candidate's actual life experiences, decision-making, and values.

## Core Principle

**If you are unsure about the candidate's experience, motivations, or reasoning — ask.** Ask liberally. The candidate's psychological makeup, moral outlook, and genuine life experiences are what differentiate a strong resume/behavioral answer from a generic one. No amount of rephrasing can substitute for real context. Generic content is a failure state.

---

## 1. Question Taxonomy

Four probing categories. Use the category that matches the context gap you've identified.

### Values & Motivation
**Purpose:** Understand why the candidate makes choices — what they optimize for, what they'd walk away from, what drives them.

Example probes:
- "What made you leave [X] for [Y]? What were you optimizing for?"
- "What would make you turn down a high-paying offer?"
- "Why did you pick [this project/technology/company] over alternatives?"
- "What part of this role genuinely excites you vs. what you're tolerating?"

### Decision Reasoning
**Purpose:** Surface how the candidate thinks under uncertainty — their frameworks, trade-off analysis, and judgment calls.

Example probes:
- "Walk me through a decision where you had incomplete information. What framework did you use?"
- "What trade-offs did you weigh when choosing [approach X] over [approach Y]?"
- "When you had to prioritize between [competing demands], how did you decide?"
- "What's a decision you made that others disagreed with? What was your reasoning?"

### Conflict & Resilience
**Purpose:** Reveal real friction — interpersonal disagreements, organizational blockers, failures that weren't the candidate's fault — and how they navigated it.

Example probes:
- "Describe a disagreement with a teammate or manager. What did you actually say?"
- "When something went wrong that wasn't your fault, what did you do?"
- "Tell me about a time you had to push back on a decision from above. What happened?"
- "What's the most frustrating work situation you've been in? How did you handle it?"

### Failure & Growth
**Purpose:** Uncover genuine self-awareness — not humble-brags, but real failures and what the candidate actually learned.

Example probes:
- "What's something you got wrong? Not a humble-brag — what actually went badly and what did you learn?"
- "What skill or area are you genuinely weak in? How do you compensate?"
- "Tell me about a project that didn't ship or a goal you missed. What happened?"
- "If you could redo one professional decision, what would it be and why?"

---

## 2. Anti-Generic Detection

### Red-Flag Phrases

If you find yourself writing any of these (or close variants), STOP and ask a probing question instead:

- "Collaborated with cross-functional teams"
- "Drove results in a fast-paced environment"
- "Leveraged strong communication skills"
- "Demonstrated leadership in..."
- "Passionate about..."
- "Proven track record of..."
- "Strong problem-solving skills"
- "Detail-oriented and results-driven"

### Structural Red Flags

- **Hollow Action:** A bullet where the Action could apply to literally anyone in the same role. Ask: "What specifically did YOU do that someone else in your position might not have?"
- **Missing Specifics in STAR:** A Situation that lacks names, numbers, dates, or concrete context. Ask: "Can you ground this — what team, what timeframe, what was at stake?"
- **Copy-Paste "Why This Company?":** An answer that could be sent to a competitor with only the company name changed. Ask: "What genuinely interests you about this specific company? What would you actually want to know from someone who works there?"
- **Hypothetical Masquerading as Experience:** Using "I would..." when the question asks "Tell me about a time..." Ask: "Do you have a real example of this? Even a small one — from school, a side project, or personal life?"

---

## 3. Probing Technique

### Rules
1. **One question at a time.** Don't overwhelm the candidate with a list.
2. **Socratic, not leading.** Ask "What did you do?" not "Did you show leadership?"
3. **Push past abstractions.** When the candidate says "I collaborated with the team," follow up: "What did that look like concretely? Who did you talk to and what did you say?"
4. **Ask for the uncomfortable specifics.** "What was the hardest part?" / "What would you do differently?" / "What did you actually say in that conversation?"
5. **Respect boundaries.** If the candidate says they'd rather not discuss something, move on gracefully. Note the boundary for future reference.
6. **JD-informed probing.** Frame questions around the specific role's requirements, not generic behavioral categories. The goal is to surface experiences that match THIS job.
7. **Follow up on vague answers.** If the candidate's response is still generic after the first probe, ask once more with a more specific angle. If still vague after two attempts, move on — the candidate may not have a relevant experience.

### Question Framing

**Good:** "The JD emphasizes navigating regulatory ambiguity. Can you think of a time you had to make a call without clear rules or precedent?"

**Bad:** "Tell me about a time you showed adaptability."

The difference: Good probes give the candidate a concrete scenario to anchor their memory. Bad probes are the same generic prompts they've seen on every interview prep site.

---

## 4. Persistence Protocol

### When to Persist
After each discovery exchange where the candidate provides substantive new information, append to the resolved candidate context file (a user-provided path or `<resume-root>/candidate-context.md`).

### Format

```
## Discovery: <YYYY-MM-DD> — <skill-name> — <Company> <Role>

### <Category>
- **Q:** <Question asked>
  **A:** <Candidate's response, cleaned up for clarity but preserving their voice and specifics>
```

Only include categories where the candidate provided new information. Omit empty categories.

### Rules
1. **Append-only.** Never overwrite or modify existing content in `candidate-context.md`.
2. **Preserve voice.** Clean up grammar and structure, but keep the candidate's specific details, phrasing, and personality.
3. **Tag source.** Every discovery section is tagged with the skill name and target role so future runs know which context came from where.
4. **Flag resume candidates.** If the candidate describes an experience that could become a resume bullet, add a comment: `<!-- RESUME-CANDIDATE: Brief description of the experience and which role/section it could strengthen -->`
5. **No duplication.** Before appending, scan existing `candidate-context.md` for the same experience. If it's already there (even phrased differently), skip or note it as a reinforcement rather than duplicating.
