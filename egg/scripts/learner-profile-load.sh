#!/bin/bash
# SessionStart hook: Load, validate, and sync-heal the learner profile.
# Stdout is injected into Claude's context at session start.

# Claude Code provides a plugin-scoped persistent directory that survives updates.
[ -n "${CLAUDE_PLUGIN_DATA:-}" ] || exit 0
DATA_DIR="$CLAUDE_PLUGIN_DATA"
LEGACY_DATA_DIR="$HOME/.local/share/claude"
PROFILE="$DATA_DIR/leetcode-teacher-profile.md"
LEDGER="$DATA_DIR/leetcode-teacher-ledger.md"

# One-time migration. mv is atomic within the user data filesystem, and -n
# prevents a legacy file from overwriting state already created at the new path.
migrate_legacy_file() {
  local filename="$1"
  [ -f "$LEGACY_DATA_DIR/$filename" ] || return 0
  [ ! -e "$DATA_DIR/$filename" ] || return 0
  mkdir -p "$DATA_DIR" 2>/dev/null || return 0
  mv -n "$LEGACY_DATA_DIR/$filename" "$DATA_DIR/$filename" 2>/dev/null || true
}

migrate_legacy_file "leetcode-teacher-profile.md"
migrate_legacy_file "leetcode-teacher-ledger.md"
migrate_legacy_file "leetcode-session-state.md"

# Extract session_id from hook input JSON (stdin)
INPUT=$(cat)
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // "unknown"' 2>/dev/null)
if [ -z "$SESSION_ID" ] || [ "$SESSION_ID" = "null" ]; then
  SESSION_ID="unknown"
fi

# Bound cleanup work while removing empty Stop guards left by old sessions.
GUARD_ROOT="$DATA_DIR/.stop-hook-guards"
if [ -d "$GUARD_ROOT" ]; then
  find "$GUARD_ROOT" ! -path "$GUARD_ROOT" -type d -mtime +7 -print 2>/dev/null \
    | head -n 100 \
    | while IFS= read -r guard_dir; do
        rmdir "$guard_dir" 2>/dev/null || true
      done
fi

# SessionStart must not create persistent files merely because the plugin is installed.
[ -f "$PROFILE" ] || exit 0

# Reset the independent Stop guard for this session.
SESSION_KEY=$(printf '%s' "$SESSION_ID" | tr -cd '[:alnum:]_-')
[ -n "$SESSION_KEY" ] || SESSION_KEY="unknown"
rmdir "$GUARD_ROOT/learner-$SESSION_KEY" 2>/dev/null || true

# --- Profile exists: validate structure ---
REPAIRS=""

if ! grep -q "^## About Me" "$PROFILE" 2>/dev/null; then
  # Prepend About Me section
  TMPFILE=$(mktemp)
  echo '## About Me' > "$TMPFILE"
  echo '' >> "$TMPFILE"
  echo '<!-- Write your goals, timeline, preferred language, and self-assessed level here. -->' >> "$TMPFILE"
  echo '' >> "$TMPFILE"
  cat "$PROFILE" >> "$TMPFILE"
  mv "$TMPFILE" "$PROFILE"
  REPAIRS="${REPAIRS}[REPAIRED] Restored missing section: About Me. Let the learner know if they didn't intend to remove it.\n"
fi

if ! grep -q "^## Known Weaknesses" "$PROFILE" 2>/dev/null; then
  echo '' >> "$PROFILE"
  echo '## Known Weaknesses' >> "$PROFILE"
  echo '' >> "$PROFILE"
  echo '### Resolved' >> "$PROFILE"
  echo '' >> "$PROFILE"
  REPAIRS="${REPAIRS}[REPAIRED] Restored missing section: Known Weaknesses. Let the learner know if they didn't intend to remove it.\n"
fi

if ! grep -q "^## Session History" "$PROFILE" 2>/dev/null; then
  echo '' >> "$PROFILE"
  echo '## Session History' >> "$PROFILE"
  echo '' >> "$PROFILE"
  REPAIRS="${REPAIRS}[REPAIRED] Restored missing section: Session History. Let the learner know if they didn't intend to remove it.\n"
fi

sanitize_cell() {
  printf '%s' "$1" | tr '\r\n' '  ' | sed 's/|/\\|/g'
}

append_ledger_row() {
  printf '| %s | %s | %s | %s | %s | %s | %s | %s |\n' \
    "$(sanitize_cell "$1")" "$(sanitize_cell "$2")" \
    "$(sanitize_cell "$3")" "$(sanitize_cell "$4")" \
    "$(sanitize_cell "$5")" "$(sanitize_cell "$6")" \
    "$(sanitize_cell "$7")" "$(sanitize_cell "$8")" >> "$LEDGER"
}

# --- Self-heal sync: check profile vs ledger consistency ---
# Get last session history entry timestamp from profile
LAST_PROFILE_TS=$(grep -E '^### [0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2} \|' "$PROFILE" 2>/dev/null | head -1 | sed 's/^### //' | cut -d'|' -f1 | xargs)

# Get latest ledger row timestamp (robust to out-of-order rows)
LAST_LEDGER_TS=$(grep -E '^\|' "$LEDGER" 2>/dev/null \
  | sed 's/^| *//' \
  | cut -d'|' -f1 \
  | tr -d ' ' \
  | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}$' \
  | sort -r \
  | head -1)

if [ -n "$LAST_PROFILE_TS" ] && [ -n "$LAST_LEDGER_TS" ]; then
  # Check if profile has a newer entry than ledger
  if [[ "$LAST_PROFILE_TS" > "$LAST_LEDGER_TS" ]]; then
    # Profile has entry not in ledger — extract fields and append
    ENTRY_LINE=$(grep -E "^### ${LAST_PROFILE_TS} \|" "$PROFILE" 2>/dev/null | head -1)
    P_PROBLEM=$(echo "$ENTRY_LINE" | cut -d'|' -f2 | xargs)
    P_MODE=$(echo "$ENTRY_LINE" | cut -d'|' -f3 | xargs)
    P_VERDICT=$(echo "$ENTRY_LINE" | cut -d'|' -f4 | xargs)

    # Get gaps and review from lines following the entry
    ENTRY_LINE_NUM=$(grep -n "^### ${LAST_PROFILE_TS} |" "$PROFILE" | head -1 | cut -d: -f1)
    if [ -n "$ENTRY_LINE_NUM" ]; then
      GAPS_LINE_NUM=$((ENTRY_LINE_NUM + 1))
      REVIEW_LINE_NUM=$((ENTRY_LINE_NUM + 2))
      P_GAPS=$(sed -n "${GAPS_LINE_NUM}p" "$PROFILE" | sed 's/^Gaps: //')
      P_REVIEW=$(sed -n "${REVIEW_LINE_NUM}p" "$PROFILE" | sed 's/^Review: //')
    fi

    [ -z "$P_GAPS" ] && P_GAPS="none"
    [ -z "$P_REVIEW" ] && P_REVIEW="—"

    append_ledger_row "$LAST_PROFILE_TS" "sync-heal" "$P_PROBLEM" "unknown" "$P_MODE" "$P_VERDICT" "$P_GAPS" "$P_REVIEW"
    REPAIRS="${REPAIRS}[SYNC] Appended missing ledger row for ${LAST_PROFILE_TS} session (pattern=unknown).\n"
  fi
elif [ -n "$LAST_PROFILE_TS" ] && [ -z "$LAST_LEDGER_TS" ]; then
  # Profile has entries but ledger is empty (header only)
  ENTRY_LINE=$(grep -E "^### ${LAST_PROFILE_TS} \|" "$PROFILE" 2>/dev/null | head -1)
  P_PROBLEM=$(echo "$ENTRY_LINE" | cut -d'|' -f2 | xargs)
  P_MODE=$(echo "$ENTRY_LINE" | cut -d'|' -f3 | xargs)
  P_VERDICT=$(echo "$ENTRY_LINE" | cut -d'|' -f4 | xargs)

  ENTRY_LINE_NUM=$(grep -n "^### ${LAST_PROFILE_TS} |" "$PROFILE" | head -1 | cut -d: -f1)
  if [ -n "$ENTRY_LINE_NUM" ]; then
    GAPS_LINE_NUM=$((ENTRY_LINE_NUM + 1))
    REVIEW_LINE_NUM=$((ENTRY_LINE_NUM + 2))
    P_GAPS=$(sed -n "${GAPS_LINE_NUM}p" "$PROFILE" | sed 's/^Gaps: //')
    P_REVIEW=$(sed -n "${REVIEW_LINE_NUM}p" "$PROFILE" | sed 's/^Review: //')
  fi

  [ -z "$P_GAPS" ] && P_GAPS="none"
  [ -z "$P_REVIEW" ] && P_REVIEW="—"

  append_ledger_row "$LAST_PROFILE_TS" "sync-heal" "$P_PROBLEM" "unknown" "$P_MODE" "$P_VERDICT" "$P_GAPS" "$P_REVIEW"
  REPAIRS="${REPAIRS}[SYNC] Appended missing ledger row for ${LAST_PROFILE_TS} session (pattern=unknown).\n"
fi

# --- Output session metadata ---
echo "=== SESSION METADATA ==="
echo "Session ID: ${SESSION_ID}"
echo "Session Timestamp: $(date +%Y-%m-%dT%H:%M)"
echo "=== END SESSION METADATA ==="

# --- Output full profile ---
echo "=== LEARNER PROFILE (loaded automatically at session start) ==="
cat "$PROFILE"
echo ""
echo "=== END LEARNER PROFILE ==="

# --- Output repairs if any ---
if [ -n "$REPAIRS" ]; then
  echo ""
  echo -e "$REPAIRS"
fi

# --- Retest suggestions for resolved (short-term) weaknesses ---
RETEST_OUTPUT=""
if command -v python3 &>/dev/null; then
  RETEST_OUTPUT=$(python3 - "$PROFILE" <<'PY' 2>/dev/null
from datetime import datetime
import re
import sys

profile_path = sys.argv[1]

try:
    with open(profile_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
except OSError:
    sys.exit(0)

known_weaknesses = []
in_known_weaknesses = False
for line in lines:
    stripped = line.rstrip("\n")
    if stripped.startswith("## Known Weaknesses"):
        in_known_weaknesses = True
        continue
    if in_known_weaknesses and stripped.startswith("## Session History"):
        break
    if in_known_weaknesses:
        known_weaknesses.append(stripped)

label_re = re.compile(r"^- \*\*(.+?)\*\*")
last_tested_re = re.compile(r"Last tested:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2})")
today = datetime.now()
results = []

i = 0
while i < len(known_weaknesses):
    current = known_weaknesses[i].strip()
    label_match = label_re.match(current)
    if not label_match:
        i += 1
        continue

    label = label_match.group(1).strip()
    block_lines = [current]
    i += 1

    while i < len(known_weaknesses):
        peek = known_weaknesses[i].strip()
        if label_re.match(peek) or peek.startswith("## ") or peek.startswith("### "):
            break
        block_lines.append(peek)
        i += 1

    block_text = " ".join(block_lines)
    if "resolved (short-term)" not in block_text:
        continue

    tested_match = last_tested_re.search(block_text)
    if not tested_match:
        continue

    last_tested = tested_match.group(1)
    try:
        tested_dt = datetime.strptime(last_tested, "%Y-%m-%dT%H:%M")
    except ValueError:
        continue

    days_ago = (today - tested_dt).days
    if days_ago >= 14:
        weeks_ago = days_ago // 7
        results.append(
            f"- {label} (last tested {last_tested}, {weeks_ago} weeks ago)"
        )

if results:
    print("\n".join(results))
PY
)
fi

if [ -n "$RETEST_OUTPUT" ]; then
  echo ""
  echo "=== RETEST SUGGESTIONS ==="
  printf '%s\n' "$RETEST_OUTPUT"
  echo "=== END RETEST SUGGESTIONS ==="
fi

exit 0
