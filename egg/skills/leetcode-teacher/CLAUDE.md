# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A Socratic teaching skill for LeetCode/ML problems with 69 reference files across 11 subdirectories. The skill uses Make It Stick learning science (retrieval practice, desirable difficulties, interleaving) and never gives answers directly.

## Architecture

```
SKILL.md                          ← Master instructions (philosophy, 8-step workflow, recall mode)
references/
  frameworks/
    reference-routing.md          ← DISPATCH TABLE: technique → reference file (single source of truth)
    algorithm-frameworks.md       ← Enumeration principle, binary tree centrality, recursion
    problem-patterns.md           ← Pattern family catalog
    socratic-questions.md         ← Question banks by teaching stage
    learning-principles.md        ← Make It Stick framework
  teaching/
    learner-profile-spec.md       ← Persistent profile format (ledger + profile files)
    recall-workflow.md            ← R1-R7 recall protocol
    recall-drills.md              ← Recall question banks
    output-formats.md             ← Study note templates
    practice-strategy.md          ← Post-problem reflection
  algorithms/                     ← 24 files: paradigms (DP, greedy, backtracking, etc.)
  techniques/                     ← 6 files: DS-specific patterns (array, string, linked-list, etc.)
  data-structures/                ← 3 files: fundamentals + advanced (DSU, Fenwick, segment tree)
  graphs/                         ← 10 files: traversal, shortest paths, flow, matching, etc.
  math/                           ← 7 files: number theory, combinatorics, geometry, probability
  numeric/                        ← 4 files: bit manipulation, numerical search
  problems/                       ← 2 files: classic interview problems, brain teasers
  ml/                             ← 2 files: ML implementations + special Socratic handling
  libraries/                      ← 1 file: pandas fundamentals
evaluations/
  trigger-tests.md                ← 31 tests (activation, routing, reference pre-loading, integrity)
```

**Runtime flow:** Step 2B of SKILL.md loads `reference-routing.md` → resolves techniques to reference files → loaded references inform Socratic prompts in Steps 3-7.

## Critical File: reference-routing.md

`references/frameworks/reference-routing.md` is the dispatch table mapping every technique to its reference file. **Any new reference file must have a corresponding row here**, or the skill won't find it during Step 2B.

## Adding a New Reference File

1. Create in the appropriate subdirectory (algorithm paradigm → `algorithms/`, DS pattern → `techniques/`, etc.)
2. Follow this structure:
   - `# Title` → definition/problem statement → `## Concept` sections with Socratic prompts → code templates with *why* comments → `## See Also` cross-references
3. **Add a row to `reference-routing.md`** — this is the step people forget
4. Verify no broken links:
   ```bash
   grep -roh 'references/[a-zA-Z_/-]*\.md' egg/skills/leetcode-teacher/ \
     | sort -u | while read ref; do
     [ -f "egg/skills/leetcode-teacher/$ref" ] || echo "BROKEN: $ref"
   done
   ```

## Reference File Conventions

- **One technique per file** (atomic design) — don't merge unrelated topics
- **Lowercase hyphenated names:** `sliding-window.md`, `binary-search-framework.md`
- **All paths use subdirectory format:** `references/algorithms/sliding-window.md`, never bare `references/sliding-window.md`
- **Socratic prompts inline:** e.g., *"What are you enumerating? Where is the redundancy?"*
- **Code templates explain why, not what:** comments describe purpose, not syntax
- **Cross-references:** every file should have a `## See Also` section pointing to related files

## Evaluations

`evaluations/trigger-tests.md` has three test types:

- **MANUAL** — requires live Claude Code session (activation tests, mode routing, reference pre-loading, full flow)
- **AUTO** — bash commands checking no stale bare paths, no broken reference links, no empty files

The AUTO integrity tests are the ones to run after adding/renaming reference files.

## Three Modes, Three Personas

- **Learning Mode** (default): 6-section structure, progressive hints (Tier 1 → 2 → 3), brute force before optimal
- **Recall Mode**: Interviewer persona, neutral acknowledgments, R1-R7 protocol, verdict assignment (Strong Pass / Pass / Borderline / Needs Work)
- **Aha Mode**: Solution provider, delivers optimal solution immediately. No Socratic scaffolding, no tracking, no transitions. Triggered by "aha mode" keyword.

Mode transitions: upshift to recall when learner shows mastery, downshift to learning for knowledge gaps. Aha Mode has no transitions — it delivers and ends.

## Persistent State (Cross-Session)

The skill writes two files to `~/.local/share/claude/` via the `leetcode-profile-sync` agent (dispatched at skill activation, resumed at write-back):
- `leetcode-teacher-ledger.md` — append-only session log (source of truth)
- `leetcode-teacher-profile.md` — working memory with Known Weaknesses (max 10, status lifecycle: new → recurring → improving → resolved)

Weakness specificity rule: not "struggles with edge cases" but "misses empty input check on array problems."
