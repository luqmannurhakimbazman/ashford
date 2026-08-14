#!/bin/bash
# Stop command hook: Verify learner profile and ledger were written during
# global-markets-teacher sessions. Exit 2 to block stop if write-back is missing.

INPUT=$(cat)

[ -n "${CLAUDE_PLUGIN_DATA:-}" ] || exit 0
DATA_DIR="$CLAUDE_PLUGIN_DATA"

# Current Stop input semantics set this after a Stop hook has continued the turn.
STOP_HOOK_ACTIVE=$(printf '%s' "$INPUT" | jq -r '.stop_hook_active // false' 2>/dev/null)
if [ "$STOP_HOOK_ACTIVE" = "true" ]; then
  exit 0
fi

SESSION_ID=$(printf '%s' "$INPUT" | jq -r '.session_id // empty' 2>/dev/null)
TRANSCRIPT=$(printf '%s' "$INPUT" | jq -r '.transcript_path // empty' 2>/dev/null)
if [ -z "$TRANSCRIPT" ] || [ ! -f "$TRANSCRIPT" ]; then
  exit 0
fi

# Independent defense-in-depth guard. Atomic mkdir permits at most one block for
# this check and session even if stop_hook_active is absent or malformed.
block_once() {
  local session_key guard_root guard_path
  session_key=$(printf '%s' "$SESSION_ID" | tr -cd '[:alnum:]_-')
  [ -n "$session_key" ] || session_key="unknown"
  guard_root="$DATA_DIR/.stop-hook-guards"
  guard_path="$guard_root/markets-$session_key"
  mkdir -p "$guard_root" 2>/dev/null || return 1
  mkdir "$guard_path" 2>/dev/null
}

# Detect actual teaching — reference file reads only happen during active teaching.
REFS_READ=$(jq -r '
  select(.type == "assistant")
  | (.message.content // [])
  | if type == "array" then .[] else empty end
  | select(.type == "tool_use" and .name == "Read")
  | (.input.file_path // .input.path // "")
  | select(contains("global-markets-teacher/references/"))
  | 1
' "$TRANSCRIPT" 2>/dev/null | wc -l | tr -d ' ')
if [ "${REFS_READ:-0}" -eq 0 ]; then
  exit 0
fi

USER_TURNS=$(jq -r '
  select(.type == "user")
  | select(
      (.message.content // [])
      | if type == "array" then any(.type != "tool_result") else true end
    )
  | 1
' "$TRANSCRIPT" 2>/dev/null | wc -l | tr -d ' ')
USER_TURNS=${USER_TURNS:-0}
if [ "$USER_TURNS" -lt 2 ]; then
  exit 0
fi

PROFILE_WRITTEN=$(jq -r '
  select(.type == "assistant")
  | (.message.content // [])
  | if type == "array" then .[] else empty end
  | select(.type == "tool_use" and (.name == "Write" or .name == "Edit" or .name == "MultiEdit"))
  | (.input.file_path // .input.path // "")
  | select(contains("markets-teacher-profile"))
  | 1
' "$TRANSCRIPT" 2>/dev/null | wc -l | tr -d ' ')
PROFILE_WRITTEN=${PROFILE_WRITTEN:-0}
LEDGER_WRITTEN=$(jq -r '
  select(.type == "assistant")
  | (.message.content // [])
  | if type == "array" then .[] else empty end
  | select(.type == "tool_use" and (.name == "Write" or .name == "Edit" or .name == "MultiEdit"))
  | (.input.file_path // .input.path // "")
  | select(contains("markets-teacher-ledger"))
  | 1
' "$TRANSCRIPT" 2>/dev/null | wc -l | tr -d ' ')
LEDGER_WRITTEN=${LEDGER_WRITTEN:-0}

if [ "$PROFILE_WRITTEN" -gt 0 ] && [ "$LEDGER_WRITTEN" -gt 0 ]; then
  exit 0
elif ! block_once; then
  exit 0
elif [ "$PROFILE_WRITTEN" -gt 0 ]; then
  echo "Profile was updated but ledger was not. Append the missing ledger row for this session." >&2
  exit 2
elif [ "$LEDGER_WRITTEN" -gt 0 ]; then
  echo "Ledger was updated but profile was not. Update the profile session history and known weaknesses." >&2
  exit 2
else
  echo "This global-markets-teacher session ended without updating the learner profile or ledger. Complete Step 8B/R7B/M6B: write the ledger row first (source of truth), then update the profile. Both files are under ${CLAUDE_PLUGIN_DATA}." >&2
  exit 2
fi
