#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EVIDENCE_DIR="$REPO_ROOT/docs/evidence/issue-3"
EVIDENCE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/dunk-issue3-evidence.XXXXXX")"
trap 'rm -rf "$EVIDENCE_ROOT"' EXIT

cd "$REPO_ROOT"
VAULT="$EVIDENCE_ROOT/vault"
STORE="dunk/scripts/dln-store.py"
FIXTURE="dunk/scripts/tests/fixtures/syllabus/st5201x/syllabus2026.pdf"
APPROVAL="$EVIDENCE_DIR/approval-request.json"
SESSION_REQUEST="$EVIDENCE_DIR/grounded-session-request.json"
LOG="$EVIDENCE_ROOT/terminal-validation.txt"
CONTEXT_JSON="$EVIDENCE_ROOT/context.json"
SOURCE_VERSION="sha256-53909df562e2658ab3e1327eb8c33120fa12b37489178dc87bb4d632e4f15376"
RENDER_DATE="${DUNK_EVIDENCE_RENDER_DATE:-2026-08-19}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$EVIDENCE_ROOT/uv-cache}"

INIT="$(python3 "$STORE" init --root "$VAULT" \
  --domain "ST5201X" --goal "Learn the approved ST5201X syllabus")"
DOMAIN_ID="$(printf '%s' "$INIT" | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["domain_id"])')"
DOMAIN="$VAULT/domains/$DOMAIN_ID"

python3 "$STORE" ingest-syllabus --root "$VAULT" --domain-id "$DOMAIN_ID" \
  --expected-revision 0 --document "$FIXTURE" \
  --original-filename syllabus2026.pdf --media-type application/pdf \
  --adapter st5201x-2026-v1 --occurred-at 2026-08-19T00:00:00Z >/dev/null
cp "$DOMAIN/syllabus/$SOURCE_VERSION.md" "$EVIDENCE_ROOT/intake-preapproval.md"

python3 "$STORE" approve-syllabus --root "$VAULT" --domain-id "$DOMAIN_ID" \
  --expected-revision 1 --request "$APPROVAL" >/dev/null
python3 "$STORE" commit --root "$VAULT" --domain-id "$DOMAIN_ID" \
  --expected-revision 2 --request "$SESSION_REQUEST" >/dev/null
python3 "$STORE" context --root "$VAULT" --domain-id "$DOMAIN_ID" >"$CONTEXT_JSON"
python3 "$STORE" validate --root "$VAULT" --domain-id "$DOMAIN_ID" >/dev/null

normalize() {
  sed -e "s#${REPO_ROOT}#<repo>#g" -e "s#${EVIDENCE_ROOT}#<tmp>#g"
}

{
  echo '$ uv run --project dunk/scripts --python 3.10 --frozen pytest -q dunk/scripts/tests'
  uv run --project dunk/scripts --python 3.10 --frozen pytest -q dunk/scripts/tests
  echo
  echo '$ claude plugin validate ./dunk --strict'
  claude plugin validate ./dunk --strict 2>&1 | normalize
  echo
  echo '$ dln-store context  # fresh process, bounded approved grounding'
  python3 - "$CONTEXT_JSON" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
grounding = value["state"]["grounding"]
print("status=" + grounding["status"])
print("approval=" + grounding["active_approval"]["event_id"])
print("source=" + grounding["active_source"]["source_version_id"])
print("unresolved=" + ",".join(item["assertion_id"] for item in grounding["unresolved_assertions"]))
PY
  echo
  echo '$ dln-store ingest-syllabus wrong-digest.pdf  # truthful degradation'
  python3 - "$FIXTURE" "$EVIDENCE_ROOT/wrong-digest.pdf" <<'PY'
import sys
from pathlib import Path
source = bytearray(Path(sys.argv[1]).read_bytes())
source[-1] ^= 1
Path(sys.argv[2]).write_bytes(source)
PY
  REVISION_BEFORE="$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' \
    "$DOMAIN/profile.yaml")"
  set +e
  FAILURE="$(python3 "$STORE" ingest-syllabus --root "$VAULT" \
    --domain-id "$DOMAIN_ID" --expected-revision "$REVISION_BEFORE" \
    --document "$EVIDENCE_ROOT/wrong-digest.pdf" \
    --original-filename wrong-digest.pdf --media-type application/pdf \
    --adapter st5201x-2026-v1 --occurred-at 2026-08-19T00:30:00Z 2>&1)"
  FAILURE_STATUS=$?
  set -e
  REVISION_AFTER="$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' \
    "$DOMAIN/profile.yaml")"
  printf '%s\n' "$FAILURE" | normalize
  printf 'exit=%s revision_before=%s revision_after=%s\n' \
    "$FAILURE_STATUS" "$REVISION_BEFORE" "$REVISION_AFTER"
  [[ "$FAILURE_STATUS" -eq 2 && "$REVISION_BEFORE" == "$REVISION_AFTER" ]]
  [[ "$FAILURE" == *'SyllabusDigestMismatchError'* ]]
  [[ "$(wc -c <"$EVIDENCE_ROOT/wrong-digest.pdf")" -eq 45185 ]]
  echo 'digest_mismatch_at_expected_size=true'
  echo 'degradation_preserved=true'
  echo
  echo '$ sha256 generated artifacts; remove; dln-store rebuild; sha256 again'
  (cd "$DOMAIN" && shasum -a 256 \
    state.json dashboard.md "syllabus/$SOURCE_VERSION.md" \
    sessions/st5201x-grounded-session-001.md) >"$EVIDENCE_ROOT/hashes-before.txt"
  rm -f "$DOMAIN/state.json" "$DOMAIN/dashboard.md" \
    "$DOMAIN/syllabus/$SOURCE_VERSION.md" \
    "$DOMAIN/sessions/st5201x-grounded-session-001.md"
  python3 "$STORE" rebuild --root "$VAULT" --domain-id "$DOMAIN_ID" | normalize
  (cd "$DOMAIN" && shasum -a 256 \
    state.json dashboard.md "syllabus/$SOURCE_VERSION.md" \
    sessions/st5201x-grounded-session-001.md) >"$EVIDENCE_ROOT/hashes-after.txt"
  cat "$EVIDENCE_ROOT/hashes-after.txt"
  diff -q "$EVIDENCE_ROOT/hashes-before.txt" "$EVIDENCE_ROOT/hashes-after.txt" >/dev/null
  echo 'hashes_unchanged=true'
  echo
  echo '$ dln-store validate  # fresh process after rebuild'
  python3 "$STORE" validate --root "$VAULT" --domain-id "$DOMAIN_ID"
} >"$LOG" 2>&1

SWIFT_CACHE="$EVIDENCE_ROOT/swift-cache"
mkdir -p "$SWIFT_CACHE"

if grep -E '/Users/|/private/|dunk-issue3-evidence\.' \
  "$EVIDENCE_ROOT/intake-preapproval.md" \
  "$DOMAIN/dashboard.md" \
  "$DOMAIN/sessions/st5201x-grounded-session-001.md" \
  "$LOG" >/dev/null; then
  echo 'personal or temporary path leaked into rendered evidence text' >&2
  exit 1
fi

SWIFT_MODULECACHE_PATH="$SWIFT_CACHE" \
CLANG_MODULE_CACHE_PATH="$SWIFT_CACHE" \
  swift "$EVIDENCE_DIR/render-evidence.swift" \
    "$EVIDENCE_ROOT/intake-preapproval.md" \
    "$DOMAIN/dashboard.md" \
    "$DOMAIN/sessions/st5201x-grounded-session-001.md" \
    "$LOG" "$EVIDENCE_DIR" "$RENDER_DATE" "$DOMAIN_ID"

printf 'Evidence regenerated for disposable domain %s\n' "$DOMAIN_ID"
cat "$EVIDENCE_ROOT/hashes-after.txt"
