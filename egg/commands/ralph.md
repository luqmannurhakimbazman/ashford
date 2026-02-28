---
description: Plan and launch an autonomous Ralph coding loop
argument-hint: "[start]"
allowed-tools: ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "Skill"]
---

# Ralph Command

## Behavior

### No arguments: `/ralph`

Invoke the `ralph` skill to start the planning workflow. This guides the user through brain-dump → spec → plan generation.

Use the Skill tool to invoke the ralph skill:
```
Skill: ralph
```

### With `start` argument: `/ralph start`

Launch the autonomous loop. Follow these steps exactly:

#### 1. Validate artifacts exist

Check that all required files are present:
- `docs/ralph/SPEC.md`
- `docs/ralph/PLAN.md`
- `docs/ralph/tasks/` directory with at least one `.md` file

If any are missing, tell the user: "Ralph artifacts not found. Run `/ralph` first to create your spec and plan."

#### 2. Copy iteration protocol

Copy the iteration protocol into the project so the loop script can access it:

```bash
cp "${CLAUDE_PLUGIN_ROOT}/skills/ralph/references/iteration-protocol.md" docs/ralph/iteration-protocol.md
```

#### 3. Copy and prepare loop script

```bash
cp "${CLAUDE_PLUGIN_ROOT}/scripts/ralph-loop-template.sh" ralph-loop.sh
chmod +x ralph-loop.sh
```

#### 4. Show pre-launch summary

Display:
- Number of tasks (count unchecked items in PLAN.md)
- Number of task files in docs/ralph/tasks/
- Estimated iterations

Ask the user to confirm before launching.

#### 5. Launch the loop

```bash
nohup ./ralph-loop.sh > docs/ralph/logs/ralph-loop.log 2>&1 &
echo "Ralph loop started (PID: $!)"
echo "Monitor: tail -f docs/ralph/logs/ralph-loop.log"
```

Tell the user:
- The PID to kill it if needed
- The tail command to watch progress
- That they can check `docs/ralph/PLAN.md` for task status at any time
