# Ralph Skill — Trigger Tests

## MANUAL: Activation Tests

### T1: Direct trigger
**Input:** "set up ralph"
**Expected:** Skill activates, begins brain-dump capture questions

### T2: Synonym trigger
**Input:** "I want to run an autonomous coding loop"
**Expected:** Skill activates, begins brain-dump capture questions

### T3: Prep trigger
**Input:** "ralph prep"
**Expected:** Skill activates, begins brain-dump capture questions

### T4: Command trigger — no args
**Input:** `/ralph`
**Expected:** Invokes ralph skill, begins planning workflow

### T5: Command trigger — start (no artifacts)
**Input:** `/ralph start` (without docs/ralph/ existing)
**Expected:** Error message telling user to run `/ralph` first

### T6: Command trigger — start (with artifacts)
**Input:** `/ralph start` (with valid docs/ralph/SPEC.md, PLAN.md, tasks/)
**Expected:** Copies iteration-protocol.md and loop script, shows summary, asks for confirmation

## MANUAL: Planning Flow Tests

### T7: Brain-dump one-at-a-time
**Expected:** Skill asks questions one at a time, not all at once

### T8: Bidirectional questioning
**Expected:** After brain-dump, skill shares assumptions and invites user to ask questions back

### T9: Spec generation
**Expected:** Produces docs/ralph/SPEC.md with sectioned ## headings, presents for review

### T10: Plan generation with frontmatter
**Expected:** Each task file in docs/ralph/tasks/ has YAML frontmatter with spec-sections and codebase-files arrays

### T11: Context validation
**Expected:** Skill flags any task with estimated context load exceeding threshold

## AUTO: File Structure Tests

```bash
#!/bin/bash
PASS=0; FAIL=0

# Test: SKILL.md exists
[[ -f egg/skills/ralph/SKILL.md ]] && PASS=$((PASS + 1)) || { echo "FAIL: SKILL.md missing"; FAIL=$((FAIL + 1)); }

# Test: iteration-protocol.md exists
[[ -f egg/skills/ralph/references/iteration-protocol.md ]] && PASS=$((PASS + 1)) || { echo "FAIL: iteration-protocol.md missing"; FAIL=$((FAIL + 1)); }

# Test: spec-template.md exists
[[ -f egg/skills/ralph/references/spec-template.md ]] && PASS=$((PASS + 1)) || { echo "FAIL: spec-template.md missing"; FAIL=$((FAIL + 1)); }

# Test: plan-template.md exists
[[ -f egg/skills/ralph/references/plan-template.md ]] && PASS=$((PASS + 1)) || { echo "FAIL: plan-template.md missing"; FAIL=$((FAIL + 1)); }

# Test: planning-questions.md exists
[[ -f egg/skills/ralph/references/planning-questions.md ]] && PASS=$((PASS + 1)) || { echo "FAIL: planning-questions.md missing"; FAIL=$((FAIL + 1)); }

# Test: ralph command exists
[[ -f egg/commands/ralph.md ]] && PASS=$((PASS + 1)) || { echo "FAIL: ralph.md command missing"; FAIL=$((FAIL + 1)); }

# Test: loop template exists and is executable
[[ -x egg/scripts/ralph-loop-template.sh ]] && PASS=$((PASS + 1)) || { echo "FAIL: ralph-loop-template.sh missing or not executable"; FAIL=$((FAIL + 1)); }

# Test: SKILL.md has correct frontmatter
grep -q "^name: ralph" egg/skills/ralph/SKILL.md && PASS=$((PASS + 1)) || { echo "FAIL: SKILL.md missing name frontmatter"; FAIL=$((FAIL + 1)); }

# Test: SKILL.md references all reference files
for ref in iteration-protocol spec-template plan-template planning-questions; do
    grep -q "$ref" egg/skills/ralph/SKILL.md && PASS=$((PASS + 1)) || { echo "FAIL: SKILL.md doesn't reference $ref"; FAIL=$((FAIL + 1)); }
done

# Test: Command has description frontmatter
grep -q "^description:" egg/commands/ralph.md && PASS=$((PASS + 1)) || { echo "FAIL: ralph.md missing description"; FAIL=$((FAIL + 1)); }

# Test: Loop script has extract_section function
grep -q "extract_section" egg/scripts/ralph-loop-template.sh && PASS=$((PASS + 1)) || { echo "FAIL: loop script missing extract_section"; FAIL=$((FAIL + 1)); }

# Test: Loop script has parse_frontmatter_array function
grep -q "parse_frontmatter_array" egg/scripts/ralph-loop-template.sh && PASS=$((PASS + 1)) || { echo "FAIL: loop script missing parse_frontmatter_array"; FAIL=$((FAIL + 1)); }

echo ""
echo "Results: $PASS passed, $FAIL failed"
```
