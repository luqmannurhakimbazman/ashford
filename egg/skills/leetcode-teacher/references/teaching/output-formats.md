# Output Format Templates

Full templates for study notes generated at the end of Learning Mode and Recall Mode sessions.

---

## Learning Mode Output

```markdown
# [Problem Name]

**Source:** [URL or description]
**Difficulty:** [Easy/Medium/Hard]
**Pattern:** [Pattern name]
**Date:** [Today's date]
**Mode:** Learning

## 1. Layman Intuition
[Real-world analogy — 2-3 sentences]

## 2. Brute Force
[Approach description]
[Code with comments]
- **Time:** O(...)
- **Space:** O(...)
- **Why not good enough:** [explanation]

## 3. Optimal Solution
[Key insight — 1-2 sentences]
[Step-by-step algorithm]
[Code with comments]
- **Time:** O(...)
- **Space:** O(...)

## 4. Alternatives
[1-2 alternative approaches with trade-offs]

## 5. Summary
| Approach | Time | Space | Notes |
|----------|------|-------|-------|
| ... | ... | ... | ... |

**Key Takeaway:** [One sentence]
**Related Problems:** [2-3 problems]

## 6. Interview Tips
- [How to present this in an interview]
- [Common follow-ups]
- [Edge cases to mention]

## Reflection Questions
1. What was the key insight?
2. What pattern does this belong to?
3. What similar problems use this pattern?
```

For ML implementations, add these additional sections:

```markdown
## Mathematical Foundation
[Key equations with term explanations]

## Numerical Walkthrough
[Step-by-step with small tensors]

## Implementation Gotchas
[Common mistakes and how to avoid them]
```

---

## Recall Mode Output

````markdown
## Recall — [YYYY-MM-DD]

**Mode:** Recall — [Full Mock Interview / Edge Cases + Complexity / Variation Challenge]
**Verdict:** [Strong Pass / Pass / Borderline / Needs Work] — [one-sentence summary]

## 1. Reconstruction
- **Approach identified:** [correct/partial/incorrect] — [summary of what they described]
- **Conceptual explanation quality:** [prose assessment of how clearly they articulated the approach]
- **Code quality:** [correct/minor bugs/major bugs/not attempted]

**User's submitted code:**
```python
[the raw code the user produced during reconstruction]
```

**Corrections & guidance:**
- [Bug/issue #1] → [fix and brief explanation]
- [Bug/issue #2] → [fix and brief explanation]
- [... additional as needed]

## 2. Edge Cases
| Edge Case | Result | Notes |
|-----------|--------|-------|
| Empty input | [caught/missed] | [reasoning — e.g., "Caught — correctly traced to `len(stack) == 0`"] |
| Single element | [caught/missed] | [reasoning] |
| [problem-specific] | [caught/missed] | [reasoning] |

## 3. Complexity Analysis
| Metric | User's Answer | Correct Answer | Result |
|--------|--------------|----------------|--------|
| Time | [their answer] | [correct] | [correct/incorrect] |
| Space | [their answer] | [correct] | [correct/incorrect] |
| Justification | [their reasoning] | — | [solid/weak/missing] |

## 4. Pattern Classification
- **Pattern identified:** [correct/incorrect/partial]
- **Related problems named:** [list] ([correct count]/[total asked])

## 5. Variation Response
- **Variation posed:** [description]
- **Adaptation:** [fluent/struggled/failed]
- **Summary:** [what they did]

## 6. Gaps to Review
| Gap | Details | Priority |
|-----|---------|----------|
| [specific gap] | [correct answer, explanation, or approach note] | [high/medium/low] |

## 7. Recommended Review Schedule
- **Next review:** [date based on spaced repetition]
- **Focus areas:** [specific topics to revisit]

## 8. Mode Transitions
<!-- Include only if a downshift/upshift occurred during the session. Omit this section entirely otherwise. -->
- **Transition:** [Downshift / Upshift] at [which step, e.g., R2]
- **Concept gap:** [what triggered it]
- **Resolution:** [what was taught or confirmed]
- **Resumed:** [yes — at which step / no — switched to full learning]

## Reflection Questions
1. Which part of the solution was hardest to recall, and why?
2. What would you do differently if this were a real interview?
3. What's the one concept you should review before your next session?

## Reference Solution
<!-- Include only when Verdict is Borderline/Needs Work, or the user requests it. Omit otherwise. -->
```python
[clean, commented reference solution]
```
- **Time:** O(...)
- **Space:** O(...)
````

---

## Filename Convention

All sessions for a problem live in one file: `[problem-name].md` (e.g., `valid-parentheses.md`).

- **Learning Mode:** Creates the file with the initial learning notes.
- **Recall Mode:** Appends a `## Recall — [YYYY-MM-DD]` section to the existing `[problem-name].md` file. If no learning notes file exists yet, create `[problem-name].md` with the recall section as the first content.
