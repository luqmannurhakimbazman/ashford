#!/usr/bin/env bash
# validate-ks-markers.sh
# PreToolUse hook for mcp__plugin_Notion_notion__notion-update-page
# Validates KS marker escaping in update_content payloads.
#
# Rules:
#   old_str must use ESCAPED markers:   \<!-- KS:start --\>  /  \<!-- KS:end --\>
#   new_str must use UNESCAPED markers: <!-- KS:start -->    /  <!-- KS:end -->
#
# Exit 0 = allow, Exit 2 = block (with JSON on stdout, reason on stderr)

# Fail open if jq is not available
if ! command -v jq &>/dev/null; then
  exit 0
fi

INPUT=$(cat)

# Extract content_updates array. If absent or null, this is not an update_content
# call (e.g., update_properties, replace_content) — pass through.
CONTENT_UPDATES=$(echo "$INPUT" | jq -r '.tool_input.content_updates // empty')
if [[ -z "$CONTENT_UPDATES" ]]; then
  exit 0
fi

NUM_UPDATES=$(echo "$INPUT" | jq '.tool_input.content_updates | length')
if [[ "$NUM_UPDATES" -eq 0 ]]; then
  exit 0
fi

# Check each update object
for i in $(seq 0 $((NUM_UPDATES - 1))); do
  OLD_STR=$(echo "$INPUT" | jq -r ".tool_input.content_updates[$i].old_str // empty")
  NEW_STR=$(echo "$INPUT" | jq -r ".tool_input.content_updates[$i].new_str // empty")

  # If neither string contains any KS marker text, this is a non-KS update — skip
  COMBINED="$OLD_STR$NEW_STR"
  if [[ "$COMBINED" != *"KS:start"* && "$COMBINED" != *"KS:end"* ]]; then
    continue
  fi

  # Check new_str for ESCAPED markers (wrong — should be unescaped)
  # Escaped form has literal backslash before < and >
  if [[ "$NEW_STR" == *'\<!-- KS:start --\>'* || "$NEW_STR" == *'\<!-- KS:end --\>'* ]]; then
    echo "BLOCKED: new_str contains escaped KS markers. Use unescaped form: <!-- KS:start --> / <!-- KS:end -->. Notion will escape them on read-back automatically." >&2
    jq -n '{
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: "deny",
        permissionDecisionReason: "new_str contains escaped KS markers. Use unescaped: <!-- KS:start --> / <!-- KS:end -->"
      }
    }'
    exit 2
  fi

  # Check old_str for UNESCAPED markers (wrong — should be escaped)
  # Unescaped = <!-- KS:start --> NOT preceded by backslash
  # We check: old_str contains <!-- KS:start --> but NOT \<!-- KS:start --\>
  if [[ "$OLD_STR" == *"<!-- KS:start -->"* || "$OLD_STR" == *"<!-- KS:end -->"* ]]; then
    # Verify it's truly unescaped (not the escaped form which contains the unescaped as substring)
    # The escaped form \<!-- KS:start --\> when stored in a bash variable does contain
    # the substring <!-- KS:start --> (minus the backslashes). We need to check that the backslashes
    # are NOT present.
    if [[ "$OLD_STR" != *'\<!-- KS:start --\>'* && "$OLD_STR" == *"<!-- KS:start -->"* ]] || \
       [[ "$OLD_STR" != *'\<!-- KS:end --\>'* && "$OLD_STR" == *"<!-- KS:end -->"* ]]; then
      echo "BLOCKED: old_str contains unescaped KS markers. Use the escaped form from notion-fetch output: \\<!-- KS:start --\\> / \\<!-- KS:end --\\>" >&2
      jq -n '{
        hookSpecificOutput: {
          hookEventName: "PreToolUse",
          permissionDecision: "deny",
          permissionDecisionReason: "old_str contains unescaped KS markers. Use escaped form from notion-fetch: \\<!-- KS:start --\\> / \\<!-- KS:end --\\>"
        }
      }'
      exit 2
    fi
  fi
done

# All checks passed
exit 0
