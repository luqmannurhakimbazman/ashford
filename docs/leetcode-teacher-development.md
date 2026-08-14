# LeetCode Teacher Development Guide

This repository document describes how to maintain the `egg/skills/leetcode-teacher/` plugin skill. It is intentionally outside the plugin directory: `CLAUDE.md` files shipped inside plugins are not loaded as plugin context. Runtime instructions belong in `SKILL.md`, agents, hooks, or referenced skill files.

## Architecture

The skill contains 69 reference files across 11 reference subdirectories.

```text
egg/skills/leetcode-teacher/
├── SKILL.md
├── evaluations/trigger-tests.md
└── references/
    ├── frameworks/       # routing, algorithm frameworks, patterns, Socratic prompts
    ├── teaching/         # profile schema, recall workflow, drills, output formats
    ├── algorithms/       # algorithm paradigms and dynamic-programming families
    ├── techniques/       # data-structure-specific techniques
    ├── data-structures/  # core and advanced structures
    ├── graphs/           # traversal, paths, flow, matching, trees
    ├── math/             # number theory, combinatorics, geometry, probability
    ├── numeric/          # bit and numerical methods
    ├── problems/         # classic interview problems and brain teasers
    ├── ml/               # ML implementations and special handling
    └── libraries/        # pandas fundamentals
```

At runtime, `SKILL.md` loads `references/frameworks/reference-routing.md`, which maps techniques to the reference files used by the teaching workflow.

## Adding a Reference

1. Create a lowercase, hyphenated Markdown file in the appropriate `references/` subdirectory.
2. Keep the file focused on one technique. Include concepts, Socratic prompts, templates whose comments explain why, and related references where useful.
3. Add or update the route in `references/frameworks/reference-routing.md`.
4. Update `evaluations/trigger-tests.md` when activation, routing, or integrity expectations change.
5. Run the validation below from the repository root.

```bash
grep -rho 'references/[a-zA-Z_/-]*\.md' egg/skills/leetcode-teacher/ \
  | sort -u \
  | while read -r ref; do
      [ -f "egg/skills/leetcode-teacher/$ref" ] || echo "BROKEN: $ref"
    done
```

## Teaching Modes

- **Learning:** structured Socratic teaching with progressive hints and a brute-force-to-optimal path.
- **Recall:** interviewer persona using the R1-R7 recall protocol and verdict assignment.
- **Aha:** immediate optimal solution without Socratic scaffolding or tracking.

## Persistent State

The `leetcode-profile-sync` agent reads and writes:

- `${CLAUDE_PLUGIN_DATA}/leetcode-teacher-profile.md`
- `${CLAUDE_PLUGIN_DATA}/leetcode-teacher-ledger.md`

The plugin's `SessionStart` hook migrates the corresponding files and session state once from the legacy `~/.local/share/claude/` directory when no file already exists at the plugin-scoped destination. Do not add new home-directory state paths or write into `${CLAUDE_PLUGIN_ROOT}`, which points at the replaceable install cache.
