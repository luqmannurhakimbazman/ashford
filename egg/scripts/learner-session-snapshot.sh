#!/bin/bash
# PreCompact hook: Write a minimal procedural reminder before context compaction.
# No transcript parsing, no semantic extraction — just a procedural nudge.

[ -n "${CLAUDE_PLUGIN_DATA:-}" ] || exit 0
DATA_DIR="$CLAUDE_PLUGIN_DATA"
LEGACY_SCRATCHPAD="$HOME/.local/share/claude/leetcode-session-state.md"
SCRATCHPAD="$DATA_DIR/leetcode-session-state.md"
INPUT=$(cat)

# One-time migration without overwriting state already present in plugin data.
if [ -f "$LEGACY_SCRATCHPAD" ] && [ ! -e "$SCRATCHPAD" ]; then
  mkdir -p "$DATA_DIR" 2>/dev/null || exit 0
  mv -n "$LEGACY_SCRATCHPAD" "$SCRATCHPAD" 2>/dev/null || true
fi
TRANSCRIPT=$(echo "$INPUT" | jq -r '.transcript_path // empty' 2>/dev/null)

# Only write if this looks like a leetcode session
if [ -n "$TRANSCRIPT" ] && [ -f "$TRANSCRIPT" ] && grep -q "leetcode-teacher" "$TRANSCRIPT" 2>/dev/null; then
  mkdir -p "$DATA_DIR" 2>/dev/null || exit 0
  SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // "unknown"' 2>/dev/null)
  if [ -z "$SESSION_ID" ] || [ "$SESSION_ID" = "null" ]; then
    SESSION_ID="unknown"
  fi

  EXISTING_TS=""
  if [ -f "$SCRATCHPAD" ]; then
    EXISTING_TS=$(grep -o 'Session Timestamp: [0-9T:-]*' "$SCRATCHPAD" | sed 's/Session Timestamp: //')
  fi
  SESSION_TS="${EXISTING_TS:-$(date +%Y-%m-%dT%H:%M)}"

  cat > "$SCRATCHPAD" << EOF
# LeetCode Session In Progress (saved before compaction)
- You are in a leetcode-teacher session. Read ${CLAUDE_PLUGIN_DATA}/leetcode-teacher-profile.md for context.
- Session ID: ${SESSION_ID}
- Session Timestamp: ${SESSION_TS}
- Write-back required at session end: Step 8B (learning) or R7B (recall).
- Write ledger row first, then profile entry.
- The leetcode-profile-sync agent was dispatched at session start, but its agent ID is not preserved across compaction. Fall back to direct file writes per references/teaching/learner-profile-spec.md.
EOF
fi

exit 0
