---
name: ralph
description: >
  This skill should be used when the user wants to set up an autonomous coding loop,
  plan a Ralph loop, prepare for headless Claude execution, create a spec and implementation
  plan for autonomous coding, or run an autonomous development workflow.
  Trigger phrases include "ralph", "autonomous loop", "coding loop", "ralph prep",
  "set up ralph", "headless loop", "autonomous coding", "ralph plan".
---

# Ralph — Autonomous Coding Loop

An autonomous coding loop where Claude executes one task per iteration with a fresh context window. The spec and implementation plan are the single source of truth — not conversation history. Context rot is structurally impossible.

## Core Principle

Each loop iteration is a clean `claude -p` invocation. The loop script dynamically assembles each iteration's prompt from ONLY the spec sections and codebase files that task declares it needs. Context waste is zero.

## Planning Phase Workflow

You are in the planning phase. Your job is to help the user go from an idea to a complete spec and implementation plan, ready for autonomous execution.

### Step 0: Load Planning Framework

Load the planning questions framework:
@references/planning-questions.md

### Step 1: Brain-Dump Capture

Ask questions from the framework ONE AT A TIME. Skip questions the user already answered. Use multiple choice when possible. Goal: understand what they're building, current codebase state, what "done" looks like, constraints.

### Step 2: Bidirectional Questioning

After you understand the project, share your assumptions as a numbered list. Explicitly invite the user to correct them AND to ask you questions back. This surfaces hidden assumptions — the #1 source of bugs in autonomous execution.

Say: "Now I'll share what I'm assuming. Tell me where I'm wrong, and then ask ME any questions about what I'll be building."

### Step 3: Generate Spec

Load the spec template:
@references/spec-template.md

Generate `docs/ralph/SPEC.md` following the template. Present it to the user section by section. The user MUST review and approve every section. They own this file.

Rules:
- Section names are lowercase kebab-style (used as identifiers in task frontmatter)
- Each section under 50 lines
- `goal` and `constraints` are almost always needed
- Domain sections are project-specific — use what fits

Create the directory structure:
```bash
mkdir -p docs/ralph/tasks docs/ralph/logs
```

Write the spec file to `docs/ralph/SPEC.md`.

### Step 4: Generate Plan

Load the plan template:
@references/plan-template.md

Generate `docs/ralph/PLAN.md` (lightweight checkbox index) and individual task files in `docs/ralph/tasks/NN-kebab-title.md`.

For each task file, include YAML frontmatter with:
- `spec-sections`: Which SPEC.md sections this task needs (array of section heading names)
- `codebase-files`: Which existing project files Claude needs to see (array of paths)

Present the plan to the user. They must approve the task list, ordering, and per-task context declarations.

### Step 5: Context-Aware Validation

For each task, estimate context load:
- Sum the line counts of declared spec sections + declared codebase files + task body + iteration protocol (~60 lines)
- Flag any task estimated to exceed ~1500 lines of combined input
- Suggest decomposition for oversized tasks

### Step 6: Handoff

Tell the user:

> "Your spec and plan are ready in `docs/ralph/`. To launch the autonomous loop, run `/ralph start`."

Do NOT launch the loop yourself. The `/ralph start` command handles that.

## Important Constraints

- **Human owns the spec.** Draft it, present it, let them edit. Never auto-finalize.
- **One question at a time.** Don't overwhelm during brain-dump capture.
- **Lean artifacts.** Every byte in the spec and plan is read every iteration. Bloat degrades performance.
- **Tasks must be atomic.** Each task completable in one context window. When in doubt, split.
- **Filesystem-first state.** All state persists through files. Nothing relies on conversation history.
