#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
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

write_profile() {
  local path="$1"
  cat > "$path" <<'EOF'
## About Me

Fixture profile.

## Known Weaknesses

## Session History

EOF
}

run_loader() {
  local script="$1"
  local home="$2"
  local data_dir="$3"
  local session_id="$4"

  printf '{"session_id":"%s"}\n' "$session_id" \
    | HOME="$home" CLAUDE_PLUGIN_DATA="$data_dir" bash "$script" >/dev/null
}

test_migration() {
  local kind="$1"
  local profile="$2"
  local ledger="$3"
  local state="$4"
  local script="$REPO_ROOT/egg/scripts/${kind}-profile-load.sh"
  local root="$FIXTURE_ROOT/migration-$kind"
  local home="$root/home"
  local data_dir="$root/data"
  local legacy_dir="$home/.local/share/claude"

  mkdir -p "$legacy_dir"
  write_profile "$legacy_dir/$profile"
  printf 'legacy ledger\n' > "$legacy_dir/$ledger"
  printf 'legacy state\n' > "$legacy_dir/$state"

  run_loader "$script" "$home" "$data_dir" "migration-$kind"

  if [ -f "$data_dir/$profile" ] \
    && [ -f "$data_dir/$ledger" ] \
    && [ -f "$data_dir/$state" ] \
    && [ ! -e "$legacy_dir/$profile" ] \
    && [ ! -e "$legacy_dir/$ledger" ] \
    && [ ! -e "$legacy_dir/$state" ]; then
    pass "$kind legacy files migrate to plugin data"
  else
    fail "$kind legacy files migrate to plugin data"
  fi
}

test_non_overwrite() {
  local kind="$1"
  local profile="$2"
  local script="$REPO_ROOT/egg/scripts/${kind}-profile-load.sh"
  local root="$FIXTURE_ROOT/non-overwrite-$kind"
  local home="$root/home"
  local data_dir="$root/data"
  local legacy_dir="$home/.local/share/claude"

  mkdir -p "$legacy_dir" "$data_dir"
  printf 'legacy source\n' > "$legacy_dir/$profile"
  write_profile "$data_dir/$profile"
  printf 'destination marker\n' >> "$data_dir/$profile"

  run_loader "$script" "$home" "$data_dir" "non-overwrite-$kind"

  if grep -q '^destination marker$' "$data_dir/$profile" \
    && ! grep -q '^legacy source$' "$data_dir/$profile" \
    && grep -q '^legacy source$' "$legacy_dir/$profile"; then
    pass "$kind migration does not overwrite an existing destination"
  else
    fail "$kind migration does not overwrite an existing destination"
  fi
}

test_migration_failure() {
  local kind="$1"
  local profile="$2"
  local script="$REPO_ROOT/egg/scripts/${kind}-profile-load.sh"
  local root="$FIXTURE_ROOT/migration-failure-$kind"
  local home="$root/home"
  local legacy_dir="$home/.local/share/claude"
  local data_path="$root/data-is-a-file"

  mkdir -p "$legacy_dir"
  printf 'legacy source\n' > "$legacy_dir/$profile"
  printf 'not a directory\n' > "$data_path"

  run_loader "$script" "$home" "$data_path" "migration-failure-$kind"

  if grep -q '^legacy source$' "$legacy_dir/$profile"; then
    pass "$kind migration failure preserves the legacy source"
  else
    fail "$kind migration failure preserves the legacy source"
  fi
}

write_transcript() {
  local path="$1"
  local reference_path="$2"

  cat > "$path" <<EOF
{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Read","input":{"file_path":"$reference_path"}}]}}
{"type":"user","message":{"content":"first plain-string prompt"}}
{"type":"user","message":{"content":[{"type":"tool_result","content":"carrier only"}]}}
{"type":"user","message":{"content":[{"type":"text","text":"second human prompt"}]}}
EOF
}

run_stop_check() {
  local script="$1"
  local home="$2"
  local data_dir="$3"
  local session_id="$4"
  local transcript="$5"

  printf '{"session_id":"%s","transcript_path":"%s","stop_hook_active":false}\n' \
    "$session_id" "$transcript" \
    | HOME="$home" CLAUDE_PLUGIN_DATA="$data_dir" bash "$script"
}

test_transcript_and_guard() {
  local kind="$1"
  local reference_path="$2"
  local guard_prefix="$3"
  local script="$REPO_ROOT/egg/scripts/${kind}-profile-check.sh"
  local root="$FIXTURE_ROOT/stop-$kind"
  local home="$root/home"
  local data_dir="$root/data"
  local transcript="$root/transcript.jsonl"
  local first_status second_status

  mkdir -p "$home" "$data_dir"
  write_transcript "$transcript" "$reference_path"

  set +e
  run_stop_check "$script" "$home" "$data_dir" "fixture-session" "$transcript" >/dev/null 2>&1
  first_status=$?
  run_stop_check "$script" "$home" "$data_dir" "fixture-session" "$transcript" >/dev/null 2>&1
  second_status=$?
  set -e

  if [ "$first_status" -eq 2 ] \
    && [ "$second_status" -eq 0 ] \
    && [ -d "$data_dir/.stop-hook-guards/${guard_prefix}-fixture-session" ]; then
    pass "$kind transcript parsing counts humans, ignores tool-result carriers, and blocks once"
  else
    fail "$kind transcript parsing counts humans, ignores tool-result carriers, and blocks once"
  fi
}

test_bounded_stale_pruning() {
  local root="$FIXTURE_ROOT/stale-pruning"
  local home="$root/home"
  local data_dir="$root/data"
  local guard_root="$data_dir/.stop-hook-guards"
  local stale_count
  local i

  mkdir -p "$home" "$guard_root/learner-current-session" "$guard_root/learner-recent"
  write_profile "$data_dir/leetcode-teacher-profile.md"

  i=1
  while [ "$i" -le 105 ]; do
    mkdir "$guard_root/learner-stale-$i"
    touch -t 202001010000 "$guard_root/learner-stale-$i"
    i=$((i + 1))
  done

  run_loader "$REPO_ROOT/egg/scripts/learner-profile-load.sh" \
    "$home" "$data_dir" "current-session"

  stale_count=$(find "$guard_root" -type d -name 'learner-stale-*' | wc -l | tr -d ' ')
  if [ "$stale_count" -eq 5 ] \
    && [ -d "$guard_root/learner-recent" ] \
    && [ ! -e "$guard_root/learner-current-session" ]; then
    pass "SessionStart prunes at most 100 stale guards and resets only the current guard"
  else
    fail "SessionStart prunes at most 100 stale guards and resets only the current guard"
  fi
}

test_migration learner leetcode-teacher-profile.md leetcode-teacher-ledger.md leetcode-session-state.md
test_migration markets markets-teacher-profile.md markets-teacher-ledger.md markets-session-state.md

test_non_overwrite learner leetcode-teacher-profile.md
test_non_overwrite markets markets-teacher-profile.md

test_migration_failure learner leetcode-teacher-profile.md
test_migration_failure markets markets-teacher-profile.md

test_transcript_and_guard learner \
  /plugin/skills/leetcode-teacher/references/learning-principles.md learner
test_transcript_and_guard markets \
  /plugin/skills/global-markets-teacher/references/learning-principles.md markets

test_bounded_stale_pruning

printf '\nResults: %d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
