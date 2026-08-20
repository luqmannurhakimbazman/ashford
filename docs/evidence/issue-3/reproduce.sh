#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EVIDENCE_DIR="$REPO_ROOT/docs/evidence/issue-3"
EVIDENCE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/dunk-issue3-evidence.XXXXXX")"
trap 'rm -rf "$EVIDENCE_ROOT"' EXIT

cd "$REPO_ROOT"
VAULT="$EVIDENCE_ROOT/vault"
STORE="dunk/scripts/dln-store.py"
FIXTURE="dunk/scripts/tests/fixtures/syllabus/generic/two-page-syllabus.pdf"
GOLDEN="dunk/scripts/tests/fixtures/syllabus/generic/expected-two-page-extraction.json"
LEGACY_FIXTURE="dunk/scripts/tests/fixtures/syllabus/legacy-v1"
INPUT="$EVIDENCE_ROOT/generic-two-page-syllabus.pdf"
LOG="$EVIDENCE_ROOT/terminal-validation.txt"
RAW_LOG="$EVIDENCE_ROOT/full-validation.txt"
CONTEXT_JSON="$EVIDENCE_ROOT/context.json"
EXPECTED_SOURCE_SHA="e481c068230a7cb006d783c33b74720f036928317bcb21ecaa9c8392f0084839"
RENDER_DATE="${DUNK_EVIDENCE_RENDER_DATE:-2026-08-19}"
export UV_CACHE_DIR="$EVIDENCE_ROOT/uv-cache"
export UV_PROJECT_ENVIRONMENT="$EVIDENCE_ROOT/venv"

run_store() {
  uv run --project dunk/scripts --python 3.10.20 --frozen python "$STORE" "$@"
}

json_field() {
  uv run --project dunk/scripts --python 3.10.20 --frozen python -c \
    'import json,sys; value=json.load(sys.stdin); print(value[sys.argv[1]])' "$1"
}

hydrate() {
  local source="$1" destination="$2" key="$3" value="$4"
  uv run --project dunk/scripts --python 3.10.20 --frozen python - "$source" "$destination" "$key" "$value" <<'PY'
import json
import sys

source, destination, key, value = sys.argv[1:]
payload = json.load(open(source, encoding="utf-8"))

def replace(item):
    if isinstance(item, dict):
        return {name: replace(value_) for name, value_ in item.items()}
    if isinstance(item, list):
        return [replace(value_) for value_ in item]
    return value if item == key else item

with open(destination, "w", encoding="utf-8") as handle:
    json.dump(replace(payload), handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

normalize() {
  sed -e "s#${REPO_ROOT}#<repo>#g" -e "s#${EVIDENCE_ROOT}#<tmp>#g"
}

for command in uv jq shellcheck claude swift; do
  command -v "$command" >/dev/null || { echo "missing required command: $command" >&2; exit 1; }
done

uv sync --project dunk/scripts --python 3.10.20 --frozen >"$RAW_LOG" 2>&1
RUNTIME="$(uv run --project dunk/scripts --python 3.10.20 --frozen python -c \
  'import platform,pypdf; assert platform.python_version()=="3.10.20"; assert pypdf.__version__=="6.14.2"; print("python="+platform.python_version()+" pypdf="+pypdf.__version__)')"

cp "$FIXTURE" "$INPUT"
[[ "$(shasum -a 256 "$INPUT" | awk '{print $1}')" == "$EXPECTED_SOURCE_SHA" ]]

INIT="$(run_store init --root "$VAULT" --domain "Portable Distributed Systems" \
  --goal "Learn from a learner-decided authoritative syllabus")"
DOMAIN_ID="$(printf '%s' "$INIT" | json_field domain_id)"
DOMAIN="$VAULT/domains/$DOMAIN_ID"

PREPARE="$(run_store prepare-syllabus --root "$VAULT" --domain-id "$DOMAIN_ID" \
  --expected-revision 0 --file "$INPUT" --media-type application/pdf \
  --role authoritative --display-name generic-two-page-syllabus.pdf \
  --occurred-at 2026-08-19T00:00:00Z)"
PREPARED_EVENT_ID="$(printf '%s' "$PREPARE" | json_field source_event_id)"
SOURCE_VERSION_ID="$(printf '%s' "$PREPARE" | json_field source_version_id)"
SOURCE_SHA="$(printf '%s' "$PREPARE" | json_field source_sha256)"
PREPARED_SHA="$(printf '%s' "$PREPARE" | json_field prepared_document_sha256)"
[[ "$SOURCE_SHA" == "$EXPECTED_SOURCE_SHA" ]]

CONTENT="$(run_store syllabus-content --root "$VAULT" --domain-id "$DOMAIN_ID" \
  --source-event-id "$PREPARED_EVENT_ID")"
printf '%s' "$CONTENT" >"$EVIDENCE_ROOT/content.json"
uv run --project dunk/scripts --python 3.10.20 --frozen python - \
  "$EVIDENCE_ROOT/content.json" "$GOLDEN" "$EXPECTED_SOURCE_SHA" <<'PY'
import json
import sys

content = json.load(open(sys.argv[1], encoding="utf-8"))
golden = json.load(open(sys.argv[2], encoding="utf-8"))["prepared_document"]
assert content["storage"] == "cas"
assert content["raw"] == {
    "available": True,
    "byte_count": content["source"]["byte_size"],
    "sha256": sys.argv[3],
}
assert content["prepared_document"] == golden
page = next(unit for unit in golden["units"] if unit["unit_id"] == "page:2")
assert page["text"][38:56] == "Week 1 Foundations"
PY

RAW_CAS="$DOMAIN/sources/sha256/$SOURCE_SHA"
PREPARED_CAS="$DOMAIN/prepared/sha256/$PREPARED_SHA.json"
cmp "$INPUT" "$RAW_CAS"
[[ "$(shasum -a 256 "$RAW_CAS" | awk '{print $1}')" == "$SOURCE_SHA" ]]
[[ "$(shasum -a 256 "$PREPARED_CAS" | awk '{print $1}')" == "$PREPARED_SHA" ]]
cp "$DOMAIN/syllabus/$SOURCE_VERSION_ID.md" "$EVIDENCE_ROOT/intake-prepared.md"
rm "$INPUT"
[[ ! -e "$INPUT" ]]

hydrate "$EVIDENCE_DIR/proposal-request.json" "$EVIDENCE_ROOT/proposal-request.json" \
  PREPARED_EVENT_ID "$PREPARED_EVENT_ID"
PROPOSE="$(run_store propose-syllabus --root "$VAULT" --domain-id "$DOMAIN_ID" \
  --expected-revision 1 --request "$EVIDENCE_ROOT/proposal-request.json")"
PROPOSAL_EVENT_ID="$(printf '%s' "$PROPOSE" | json_field proposal_event_id)"
PROPOSAL_ID="$(printf '%s' "$PROPOSE" | uv run --project dunk/scripts --python 3.10.20 --frozen python -c \
  'import json,sys; value=json.load(sys.stdin); assert len(value["proposal_ids"])==1; print(value["proposal_ids"][0])')"

hydrate "$EVIDENCE_DIR/decision-request.json" "$EVIDENCE_ROOT/decision-request-1.json" \
  PROPOSAL_EVENT_ID "$PROPOSAL_EVENT_ID"
hydrate "$EVIDENCE_ROOT/decision-request-1.json" "$EVIDENCE_ROOT/decision-request.json" \
  PROPOSAL_ID "$PROPOSAL_ID"
DECIDE="$(run_store decide-syllabus --root "$VAULT" --domain-id "$DOMAIN_ID" \
  --expected-revision 2 --request "$EVIDENCE_ROOT/decision-request.json")"
DECISION_EVENT_ID="$(printf '%s' "$DECIDE" | json_field decision_event_id)"
cp "$DOMAIN/dashboard.md" "$EVIDENCE_ROOT/decision-dashboard.md"

hydrate "$EVIDENCE_DIR/grounded-session-request.json" "$EVIDENCE_ROOT/session-request-1.json" \
  DECISION_EVENT_ID "$DECISION_EVENT_ID"
hydrate "$EVIDENCE_ROOT/session-request-1.json" "$EVIDENCE_ROOT/session-request.json" \
  PROPOSAL_ID "$PROPOSAL_ID"
run_store commit --root "$VAULT" --domain-id "$DOMAIN_ID" --expected-revision 3 \
  --request "$EVIDENCE_ROOT/session-request.json" >/dev/null
run_store context --root "$VAULT" --domain-id "$DOMAIN_ID" >"$CONTEXT_JSON"
uv run --project dunk/scripts --python 3.10.20 --frozen python - \
  "$CONTEXT_JSON" "$DECISION_EVENT_ID" "$PROPOSAL_ID" <<'PY'
import json
import sys

grounding = json.load(open(sys.argv[1], encoding="utf-8"))["state"]["grounding"]
assert grounding["status"] == "approved"
assert grounding["active_decision"]["event_id"] == sys.argv[2]
assert grounding["planning_topics"] == [{
    "assertion_ids": [sys.argv[3]],
    "citable": True,
    "label": "Week 1 Foundations",
}]
PY
SESSION_RECEIPT="$DOMAIN/sessions/portable-grounded-session-001.md"
[[ -f "$SESSION_RECEIPT" ]]

MANIFEST_PATHS=(
  profile.yaml events.jsonl state.json dashboard.md
  "sources/sha256/$SOURCE_SHA" "prepared/sha256/$PREPARED_SHA.json"
  "syllabus/$SOURCE_VERSION_ID.md" sessions/portable-grounded-session-001.md
)
(cd "$DOMAIN" && shasum -a 256 "${MANIFEST_PATHS[@]}") >"$EVIDENCE_ROOT/hashes-before.txt"
rm "$DOMAIN/state.json" "$DOMAIN/dashboard.md" \
  "$DOMAIN/syllabus/$SOURCE_VERSION_ID.md" "$SESSION_RECEIPT"

PYTHONPATH=dunk/scripts uv run --project dunk/scripts --python 3.10.20 --frozen python - \
  "$VAULT" "$DOMAIN_ID" "$PREPARED_EVENT_ID" <<'PY'
import socket
import subprocess
import sys
from pathlib import Path

import dln_store.extraction as extraction
from dln_store.store import LocalStore

def forbidden(*args, **kwargs):
    raise AssertionError("offline rebuild attempted an external dependency")

socket.socket = forbidden
socket.getaddrinfo = forbidden
subprocess.run = forbidden
subprocess.Popen = forbidden
extraction.acquire_local = forbidden
extraction.acquire_https = forbidden
extraction.extract_pdf = forbidden
extraction.extract_html = forbidden
store = LocalStore(Path(sys.argv[1]))
assert store.syllabus_content(sys.argv[2], sys.argv[3])["storage"] == "cas"
store.context(sys.argv[2])
store.rebuild(sys.argv[2])
store.validate(sys.argv[2])
PY
run_store rebuild --root "$VAULT" --domain-id "$DOMAIN_ID" >/dev/null
(cd "$DOMAIN" && shasum -a 256 "${MANIFEST_PATHS[@]}") >"$EVIDENCE_ROOT/hashes-after.txt"
diff -u "$EVIDENCE_ROOT/hashes-before.txt" "$EVIDENCE_ROOT/hashes-after.txt" >/dev/null
run_store validate --root "$VAULT" --domain-id "$DOMAIN_ID" >/dev/null

LEGACY_DOMAIN_ID="legacy-generic-course-3009531d"
LEGACY_DOMAIN="$VAULT/domains/$LEGACY_DOMAIN_ID"
mkdir -p "$LEGACY_DOMAIN"
cp "$LEGACY_FIXTURE/profile.yaml" "$LEGACY_FIXTURE/events.jsonl" "$LEGACY_DOMAIN/"
run_store rebuild --root "$VAULT" --domain-id "$LEGACY_DOMAIN_ID" >/dev/null
LEGACY_CONTENT="$(run_store syllabus-content --root "$VAULT" --domain-id "$LEGACY_DOMAIN_ID" \
  --source-event-id syllabus-source-166912c8793f4ae981ee997c3065cadedc44390a4df43144206faa56db2fe427)"
LEGACY_CONTEXT="$(run_store context --root "$VAULT" --domain-id "$LEGACY_DOMAIN_ID")"
printf '%s' "$LEGACY_CONTENT" | uv run --project dunk/scripts --python 3.10.20 --frozen python -c \
  'import json,sys; value=json.load(sys.stdin); assert value["storage"]=="legacy_text_only" and value["raw"]=={"available":False}'
printf '%s' "$LEGACY_CONTEXT" | uv run --project dunk/scripts --python 3.10.20 --frozen python -c \
  'import json,sys; value=json.load(sys.stdin); text=json.dumps(value); assert "approval_event_id" in text and "historical-approval-1" in text'
[[ ! -e "$LEGACY_DOMAIN/sources" && ! -e "$LEGACY_DOMAIN/prepared" ]]
run_store validate --root "$VAULT" --domain-id "$LEGACY_DOMAIN_ID" >/dev/null

{
  echo '$ uv sync --project dunk/scripts --python 3.10.20 --frozen'
  echo 'frozen_sync=true'
  echo
  echo '$ verify exact frozen extractor environment'
  echo "$RUNTIME"
  echo
  echo '$ generic PDF prepare -> content -> propose -> decide -> grounded session'
  echo "source_sha256=$SOURCE_SHA"
  echo "prepared_sha256=$PREPARED_SHA"
  echo 'cas_raw_bytes_retained=true'
  echo 'original_input_deleted=true'
  echo 'grounding_status=approved'
  echo
  echo '$ offline content/context/rebuild/validate with network and extractor blocked'
  echo 'offline_dependencies_unused=true'
  echo 'repeated_rebuild_hashes_unchanged=true'
  echo
  echo '$ legacy-v1 replay without original bytes or CAS backfill'
  echo 'legacy_storage=legacy_text_only'
  echo 'historical_approval_citation_resolved=true'
  echo 'legacy_cas_backfill=false'
  echo
  echo '$ scripted HTTPS safety matrix (injected resolver/transport; no public network)'
  uv run --project dunk/scripts --python 3.10.20 --frozen pytest -q \
    dunk/scripts/tests/test_syllabus_acquisition.py 2>&1 | tee -a "$RAW_LOG"
  echo 'adversarial_ambiguity_result=ST5201X processed generically; no layout meaning invented'
  echo
  echo '$ full repository checks'
  find . -type f -name '*.json' -not -path './.git/*' -print0 | xargs -0 -n1 jq empty
  find . -type f -name '*.sh' -not -path './.git/*' -print0 | xargs -0 -n1 bash -n
  find . -type f -name '*.sh' -not -path './.git/*' -print0 | xargs -0 shellcheck --severity=warning
  bash egg/skills/ralph/evaluations/test-ralph-loop.sh >>"$RAW_LOG" 2>&1
  bash egg/scripts/tests/test-profile-hooks.sh >>"$RAW_LOG" 2>&1
  uv run --project dunk/scripts --python 3.10.20 --frozen pytest -q dunk/scripts/tests 2>&1 | tee -a "$RAW_LOG"
  uv run --project dunk/scripts --python 3.10.20 --frozen pytest -q \
    tools/dunk-migrations/test_migrate_docker.py 2>&1 | tee -a "$RAW_LOG"
  for target in . ./egg ./aerion ./dunk; do
    claude plugin validate "$target" --strict >>"$RAW_LOG" 2>&1
  done
  echo 'json_syntax=true bash_syntax=true shellcheck=true'
  echo 'ralph_tests=true profile_hook_tests=true migration_tests=true'
  echo 'strict_repository_and_plugins=true'
  echo 'final_store_validation=true'
} >"$LOG" 2>&1

normalize <"$LOG" >"$EVIDENCE_ROOT/terminal-normalized.txt"
mv "$EVIDENCE_ROOT/terminal-normalized.txt" "$LOG"

if grep -Ei '/Users/|/private/|dunk-issue3-evidence\.|authorization:|proxy-authorization:|x-api-key:|api[_-]?key|cookie:|set-cookie:|https://[^ ]*\?' \
  "$EVIDENCE_ROOT/intake-prepared.md" "$EVIDENCE_ROOT/decision-dashboard.md" \
  "$SESSION_RECEIPT" "$LOG" "$EVIDENCE_ROOT/content.json" >/dev/null; then
  echo 'personal path, temporary path, or secret-bearing input leaked into evidence' >&2
  exit 1
fi

SWIFT_CACHE="$EVIDENCE_ROOT/swift-cache"
mkdir -p "$SWIFT_CACHE"
SWIFT_MODULECACHE_PATH="$SWIFT_CACHE" CLANG_MODULE_CACHE_PATH="$SWIFT_CACHE" \
  swift "$EVIDENCE_DIR/render-evidence.swift" \
    "$EVIDENCE_ROOT/intake-prepared.md" "$EVIDENCE_ROOT/decision-dashboard.md" \
    "$SESSION_RECEIPT" "$LOG" "$EVIDENCE_DIR" "$RENDER_DATE" "$DOMAIN_ID"

printf 'Evidence regenerated for disposable domain %s\n' "$DOMAIN_ID"
printf 'source_sha256=%s\nprepared_sha256=%s\n' "$SOURCE_SHA" "$PREPARED_SHA"
