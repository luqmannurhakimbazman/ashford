#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EVIDENCE_DIR="$REPO_ROOT/docs/evidence/issue-1"
EVIDENCE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/dunk-issue1-evidence.XXXXXX")"
trap 'rm -rf "$EVIDENCE_ROOT"' EXIT

cd "$REPO_ROOT"
VAULT="$EVIDENCE_ROOT/vault"
STORE="dunk/scripts/dln-store.py"
REQUEST="$EVIDENCE_DIR/session-request.json"
LOG="$EVIDENCE_ROOT/terminal-validation.txt"
RENDER_DATE="${DUNK_EVIDENCE_RENDER_DATE:-$(date -u +%Y-%m-%d)}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$EVIDENCE_ROOT/uv-cache}"

INIT="$(python3 "$STORE" init --root "$VAULT" \
  --domain "Bayesian Forecasting" --goal "Calibrate predictions with evidence")"
DOMAIN_ID="$(printf '%s' "$INIT" | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["domain_id"])')"
DOMAIN="$VAULT/domains/$DOMAIN_ID"
python3 "$STORE" commit --root "$VAULT" --domain-id "$DOMAIN_ID" \
  --expected-revision 0 --request "$REQUEST" >/dev/null
python3 "$STORE" context --root "$VAULT" --domain-id "$DOMAIN_ID" >/dev/null
python3 "$STORE" validate --root "$VAULT" --domain-id "$DOMAIN_ID" >/dev/null

{
  echo '$ uv run --project dunk/scripts --python 3.10 --frozen pytest -q dunk/scripts/tests'
  uv run --project dunk/scripts --python 3.10 --frozen pytest -q dunk/scripts/tests
  echo
  echo '$ uv run --project dunk/scripts --python 3.10 --frozen pytest -q dunk/scripts/tests/test_ks_merge.py'
  uv run --project dunk/scripts --python 3.10 --frozen pytest -q \
    dunk/scripts/tests/test_ks_merge.py
  echo
  echo '$ claude plugin validate ./dunk --strict'
  claude plugin validate ./dunk --strict 2>&1 | \
    sed -E 's#Validating plugin manifest: .*/dunk/#Validating plugin manifest: ./dunk/#'
  echo
  echo '$ dln-store commit --expected-revision 1 --request session-request.json  # duplicate replay'
  python3 "$STORE" commit --root "$VAULT" --domain-id "$DOMAIN_ID" \
    --expected-revision 1 --request "$REQUEST"
  echo
  echo '$ dln-store commit --expected-revision 0 --request session-request.json  # stale write'
  REVISION_BEFORE="$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' \
    "$DOMAIN/profile.yaml")"
  set +e
  STALE_OUTPUT="$(python3 "$STORE" commit --root "$VAULT" --domain-id "$DOMAIN_ID" \
    --expected-revision 0 --request "$REQUEST" 2>&1)"
  STALE_STATUS=$?
  set -e
  REVISION_AFTER="$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' \
    "$DOMAIN/profile.yaml")"
  printf '%s\n' "$STALE_OUTPUT"
  printf 'exit=%s revision_before=%s revision_after=%s\n' \
    "$STALE_STATUS" "$REVISION_BEFORE" "$REVISION_AFTER"
  [[ "$STALE_STATUS" -eq 3 && "$REVISION_BEFORE" == "$REVISION_AFTER" ]]
  echo
  echo '$ sha256 state.json dashboard.md sessions/session-demo-001.md; dln-store rebuild; sha256 again'
  (cd "$DOMAIN" && shasum -a 256 \
    state.json dashboard.md sessions/session-demo-001.md) \
    >"$EVIDENCE_ROOT/hashes-before.txt"
  python3 "$STORE" rebuild --root "$VAULT" --domain-id "$DOMAIN_ID"
  (cd "$DOMAIN" && shasum -a 256 \
    state.json dashboard.md sessions/session-demo-001.md) \
    >"$EVIDENCE_ROOT/hashes-after.txt"
  cat "$EVIDENCE_ROOT/hashes-after.txt"
  diff -q "$EVIDENCE_ROOT/hashes-before.txt" \
    "$EVIDENCE_ROOT/hashes-after.txt" >/dev/null
  echo 'hashes_unchanged=true'
} >"$LOG" 2>&1

mkdir -p /tmp/dunk-swift-cache
SWIFT_MODULECACHE_PATH=/tmp/dunk-swift-cache \
CLANG_MODULE_CACHE_PATH=/tmp/dunk-swift-cache \
  swift "$EVIDENCE_DIR/render-evidence.swift" \
    "$DOMAIN/dashboard.md" \
    "$DOMAIN/sessions/session-demo-001.md" \
    "$LOG" \
    "$EVIDENCE_DIR" \
    "$RENDER_DATE"

printf 'Evidence regenerated from disposable vault %s\n' "$DOMAIN_ID"
cat "$EVIDENCE_ROOT/hashes-after.txt"
