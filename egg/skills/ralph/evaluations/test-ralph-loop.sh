#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
LOOP_TEMPLATE="$REPO_ROOT/egg/scripts/ralph-loop-template.sh"
FIXTURE_ROOT=$(mktemp -d)
trap 'rm -rf "$FIXTURE_ROOT"' EXIT

PASS=0
FAIL=0

pass() {
    PASS=$((PASS + 1))
    printf 'PASS: %s\n' "$1"
}

fail() {
    FAIL=$((FAIL + 1))
    printf 'FAIL: %s\n' "$1" >&2
}

setup_case() {
    local name="$1"
    local directory="$FIXTURE_ROOT/$name"

    mkdir -p "$directory/docs/ralph/tasks" "$directory/docs/ralph/logs" "$directory/bin"
    cp "$LOOP_TEMPLATE" "$directory/ralph-loop.sh"
    chmod +x "$directory/ralph-loop.sh"
    cat > "$directory/docs/ralph/SPEC.md" <<'EOF'
# Spec

## goal
Preserve input exactly.
EOF
    cat > "$directory/docs/ralph/iteration-protocol.md" <<'EOF'
# Protocol
Complete only this task.
EOF
    cat > "$directory/bin/claude" <<'EOF'
#!/usr/bin/env bash
cat > "${PROMPT_CAPTURE:?}"
printf 'invoked\n' >> "${CALL_LOG:?}"
if [[ "${CLAUDE_STUB_COMPLETE:-0}" == "1" ]]; then
    sed '0,/^- \[ \] /s//- [x] /' docs/ralph/PLAN.md > docs/ralph/PLAN.md.tmp
    mv docs/ralph/PLAN.md.tmp docs/ralph/PLAN.md
fi
exit "${CLAUDE_STUB_EXIT:-0}"
EOF
    chmod +x "$directory/bin/claude"
    printf '%s\n' "$directory"
}

write_task() {
    local directory="$1"
    local number="$2"
    local codebase_files="${3:-}"
    cat > "$directory/docs/ralph/tasks/${number}-fixture.md" <<EOF
---
spec-sections: [goal]
codebase-files: [$codebase_files]
---

Implement fixture task $number.
EOF
}

run_loop() {
    local directory="$1"
    shift
    (
        cd "$directory"
        PATH="$directory/bin:$PATH" \
        PROMPT_CAPTURE="$directory/prompt.txt" \
        CALL_LOG="$directory/calls.log" \
        "$@" ./ralph-loop.sh
    )
}

# Completed plans must reach the completion branch and exit zero.
case_dir=$(setup_case completion)
printf '%s\n' '- [x] 1 Done' > "$case_dir/docs/ralph/PLAN.md"
write_task "$case_dir" 1
if output=$(run_loop "$case_dir" 2>&1) && grep -q 'Ralph Loop Complete' <<< "$output"; then
    pass "completed plan exits zero and reports completion"
else
    fail "completed plan exits zero and reports completion"
fi

# Digits in task text must not override the leading task number.
case_dir=$(setup_case extraction)
printf '%s\n' '- [ ] 7 Add OAuth2 support' > "$case_dir/docs/ralph/PLAN.md"
write_task "$case_dir" 7
if output=$(run_loop "$case_dir" env CLAUDE_STUB_COMPLETE=1 2>&1) \
    && grep -q 'Task 7' <<< "$output" \
    && [[ $(wc -l < "$case_dir/calls.log" | tr -d ' ') == "1" ]]; then
    pass "task extraction uses the leading plan number"
else
    fail "task extraction uses the leading plan number"
fi

# An unchanged task must be blocked instead of invoked indefinitely.
case_dir=$(setup_case repetition)
printf '%s\n' '- [ ] 3 Retry-prone task' > "$case_dir/docs/ralph/PLAN.md"
write_task "$case_dir" 3
if output=$(run_loop "$case_dir" 2>&1) \
    && grep -q 'remained unchecked' <<< "$output" \
    && grep -q '^- \[BLOCKED: repeated without completion\] 3' "$case_dir/docs/ralph/PLAN.md" \
    && [[ $(wc -l < "$case_dir/calls.log" | tr -d ' ') == "1" ]]; then
    pass "repeated task is blocked after one unchanged iteration"
else
    fail "repeated task is blocked after one unchanged iteration"
fi

# Blocking task 1 must not also block task 10.
case_dir=$(setup_case exact_blocking)
printf '%s\n' '- [ ] 1 Missing fixture' '- [ ] 10 Later task' > "$case_dir/docs/ralph/PLAN.md"
write_task "$case_dir" 10
if output=$(run_loop "$case_dir" 2>&1) \
    && grep -q '^- \[BLOCKED: task file missing\] 1 Missing fixture$' "$case_dir/docs/ralph/PLAN.md" \
    && grep -q '^- \[BLOCKED: repeated without completion\] 10 Later task$' "$case_dir/docs/ralph/PLAN.md" \
    && [[ $(wc -l < "$case_dir/calls.log" | tr -d ' ') == "1" ]]; then
    pass "blocking task 1 does not block task 10"
else
    fail "blocking task 1 does not block task 10"
fi

# The iteration cap must stop while unchecked work remains.
case_dir=$(setup_case max_iterations)
printf '%s\n' '- [ ] 1 First task' '- [ ] 2 Second task' > "$case_dir/docs/ralph/PLAN.md"
write_task "$case_dir" 1
write_task "$case_dir" 2
set +e
output=$(run_loop "$case_dir" env MAX_ITERATIONS=1 CLAUDE_STUB_COMPLETE=1 2>&1)
status=$?
set -e
if [[ $status -ne 0 ]] && grep -q 'Reached MAX_ITERATIONS=1' <<< "$output"; then
    pass "MAX_ITERATIONS stops additional invocations"
else
    fail "MAX_ITERATIONS stops additional invocations"
fi

# A claude failure must survive the tee pipeline.
case_dir=$(setup_case exit_status)
printf '%s\n' '- [ ] 4 Failing task' > "$case_dir/docs/ralph/PLAN.md"
write_task "$case_dir" 4
set +e
output=$(run_loop "$case_dir" env CLAUDE_STUB_EXIT=7 2>&1)
status=$?
set -e
if [[ $status -eq 7 ]] && grep -q 'claude -p failed.*exit 7' <<< "$output"; then
    pass "claude exit status propagates through tee"
else
    fail "claude exit status propagates through tee"
fi

# Backslash escapes in injected files must reach claude unchanged.
case_dir=$(setup_case prompt_bytes)
printf '%s\n' '- [ ] 5 Preserve prompt bytes' > "$case_dir/docs/ralph/PLAN.md"
printf '%s\n' 'literal = "\t"' 'zero = "\0"' > "$case_dir/sample.py"
write_task "$case_dir" 5 sample.py
if run_loop "$case_dir" env CLAUDE_STUB_COMPLETE=1 >/dev/null 2>&1 \
    && grep -Fq 'literal = "\t"' "$case_dir/prompt.txt" \
    && grep -Fq 'zero = "\0"' "$case_dir/prompt.txt"; then
    pass "prompt emission preserves literal backslash escapes"
else
    fail "prompt emission preserves literal backslash escapes"
fi

printf '\nResults: %d passed, %d failed\n' "$PASS" "$FAIL"
[[ "$FAIL" -eq 0 ]]
