# Issue #1 evidence

These images demonstrate Dunk's local-first store using a disposable, synthetic **Bayesian Forecasting** domain. They contain no personal learner data or absolute home-directory paths. The disposable vault is intentionally not committed.

## Evidence

| File | What it shows |
|---|---|
| [`obsidian-dashboard.png`](obsidian-dashboard.png) | The generated `dashboard.md` and canonical domain file tree in an Obsidian-style reading view. |
| [`session-receipt.png`](session-receipt.png) | The generated canonical Session Receipt, including independent versus supported evidence, model revision, delayed-retrieval status, calibration, and next review. |
| [`terminal-validation.png`](terminal-validation.png) | Actual test/plugin-validation output plus duplicate replay, stale-revision rejection, and deterministic rebuild results. The checkout root is normalized to `.` before rendering. |

The first two PNGs are deterministic local AppKit renderings of the generated Markdown, not captures of a personal Obsidian workspace or a claim that Obsidian itself is required. [`render-evidence.swift`](render-evidence.swift) reads the generated files verbatim and draws the disposable vault tree; the image footer labels this as rendered fixture output. Open the same disposable vault in Obsidian to inspect the ordinary Markdown directly.

## Reproduce the fixture

Run from the repository root. This uses only `/tmp`; choose another disposable path if preferred.

```bash
EVIDENCE_ROOT=/tmp/dunk-issue1-evidence
VAULT="$EVIDENCE_ROOT/vault"
DOMAIN_ID=bayesian-forecasting-0b6072a9
STORE=dunk/scripts/dln-store.py

rm -rf "$EVIDENCE_ROOT"
mkdir -p "$EVIDENCE_ROOT"
cp docs/evidence/issue-1/session-request.json "$EVIDENCE_ROOT/session-request.json"

python3 "$STORE" init \
  --root "$VAULT" \
  --domain "Bayesian Forecasting" \
  --goal "Calibrate predictions with evidence"

python3 "$STORE" commit \
  --root "$VAULT" \
  --domain-id "$DOMAIN_ID" \
  --expected-revision 0 \
  --request "$EVIDENCE_ROOT/session-request.json"

python3 "$STORE" validate --root "$VAULT" --domain-id "$DOMAIN_ID"
```

The request contains nine fixed, synthetic, stage-valid events and produces:

```text
vault/domains/bayesian-forecasting-0b6072a9/
├── profile.yaml
├── events.jsonl
├── state.json
├── dashboard.md
└── sessions/
    └── session-demo-001.md
```

The generated learner-facing files have these SHA-256 values:

```text
f64b02a94c08740b9633a623bc8f60a0194298b574f6caa7308b5077c4b16b03  dashboard.md
dd5e956460b9d213fbf2298540f1b96f92221620f2b3f7f0ada2a4d0cd371d47  sessions/session-demo-001.md
```

## Reproduce the storage checks

```bash
# Exact replay: succeeds as an idempotent no-op at revision 1.
python3 "$STORE" commit \
  --root "$VAULT" --domain-id "$DOMAIN_ID" --expected-revision 1 \
  --request "$EVIDENCE_ROOT/session-request.json"

# Stale write: exits 3 and leaves revision 1 unchanged.
python3 "$STORE" commit \
  --root "$VAULT" --domain-id "$DOMAIN_ID" --expected-revision 0 \
  --request "$EVIDENCE_ROOT/session-request.json"

DOMAIN="$VAULT/domains/$DOMAIN_ID"
shasum -a 256 "$DOMAIN/state.json" "$DOMAIN/dashboard.md" \
  "$DOMAIN/sessions/session-demo-001.md" > "$EVIDENCE_ROOT/hashes-before.txt"
python3 "$STORE" rebuild --root "$VAULT" --domain-id "$DOMAIN_ID"
shasum -a 256 "$DOMAIN/state.json" "$DOMAIN/dashboard.md" \
  "$DOMAIN/sessions/session-demo-001.md" > "$EVIDENCE_ROOT/hashes-after.txt"
diff -u "$EVIDENCE_ROOT/hashes-before.txt" "$EVIDENCE_ROOT/hashes-after.txt"

UV_CACHE_DIR=/tmp/dunk-uv-cache \
  uv run --project dunk/scripts --python 3.10 --frozen pytest -q dunk/scripts/tests
UV_CACHE_DIR=/tmp/dunk-uv-cache \
  uv run --project dunk/scripts --python 3.10 --frozen pytest -q \
  dunk/scripts/tests/test_ks_merge.py
claude plugin validate ./dunk --strict
```

## Render the Markdown evidence

On macOS, save the actual validation transcript to `$EVIDENCE_ROOT/terminal-validation.txt`, then run:

```bash
swift docs/evidence/issue-1/render-evidence.swift \
  "$VAULT/domains/$DOMAIN_ID/dashboard.md" \
  "$VAULT/domains/$DOMAIN_ID/sessions/session-demo-001.md" \
  "$EVIDENCE_ROOT/terminal-validation.txt" \
  docs/evidence/issue-1
```

AppKit/font differences may change PNG bytes without changing the generated Markdown. On other platforms, open `$VAULT` or `$VAULT/domains` in Obsidian and capture the same generated files, or inspect them with any Markdown viewer.
