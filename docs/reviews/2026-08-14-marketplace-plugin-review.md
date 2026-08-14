# ashford marketplace & plugin review

**Date:** 2026-08-14
**Commit reviewed:** `7c44b10899f62e40ca9218c5675d65fe2a846920` (clean `main`, working tree clean)
**Scope:** marketplace registry + all three plugins (`aerion`, `dunk`, `egg`) — manifests, marketplace metadata, skill/command/agent frontmatter, hooks, MCP configs, invoked scripts, security, portability, installation-cache behaviour, persistent-state paths, release/versioning hygiene, tests, docs, validation automation.

**Ground truth:** Claude Code documentation as of 2026-08-14 — [plugins-reference](https://code.claude.com/docs/en/plugins-reference), [plugins](https://code.claude.com/docs/en/plugins), [plugin-marketplaces](https://code.claude.com/docs/en/plugin-marketplaces), [slash-commands](https://code.claude.com/docs/en/slash-commands) (now redirects to the merged *skills* page), [hooks](https://code.claude.com/docs/en/hooks), [best-practices](https://code.claude.com/docs/en/best-practices).

---

## Contents

- [1. Executive summary](#1-executive-summary)
- [2. Validation performed](#2-validation-performed)
- [3. P0 — Confirmed broken functionality](#3-p0--confirmed-broken-functionality)
- [4. P1 — High](#4-p1--high)
- [5. P2 — Medium](#5-p2--medium)
- [6. P3 — Low / hygiene](#6-p3--low--hygiene)
- [7. What is already correct](#7-what-is-already-correct)
- [8. Implementation workstreams](#8-implementation-workstreams)

---

## 1. Executive summary

The repository is structurally sound: directory layout matches the documented convention, all 11 JSON files parse, all 12 shell scripts pass `bash -n`, both Python test suites pass (52 + 17 tests), and no secrets are committed. The manifests are minimal but legal — `name` is the only required `plugin.json` field, and the marketplace has the required `name` / `owner` / `plugins`.

The problems are concentrated in **wiring**, not structure. Five defects were confirmed to fully disable shipped functionality, four of them proven by direct execution rather than inference:

| # | Defect | Blast radius |
|---|---|---|
| F1 | Both `templates/*.json` omit the `mcpServers` wrapper | Copying a template to a project root loads **zero** MCP servers |
| F2 | `aerion/commands/check-apps.md` frontmatter is invalid YAML | `/check-apps` loses description, `argument-hint`, and `allowed-tools` |
| F3 | dunk's Notion MCP tool names use the wrong scoped prefix (9 sites) | `dln-sync` agent has **no** tools; KS-marker hook never fires |
| F4 | Stop-hook `jq` reads `.content` instead of `.message.content` | Write-back enforcement for both teacher skills is dead code |
| F5 | `learner-profile-load.sh` is wired to no hook | LeetCode learner profile never loads at session start |

Beyond those, the highest-value fixes are: quoting plugin-skill `name` values that contain spaces (F6), removing `permissionMode` from plugin agents where it is documented as unsupported (F7), routing `python-lint.sh` diagnostics to stderr so Claude actually receives them (F8), dropping `--unsafe-fixes` from that same hook (F9), and eliminating the 21 dangling references to a `resume-builder` skill that does not exist (F11).

There is **no CI**. Adding `claude plugin validate ./<plugin> --strict` per plugin would have caught F2 and the missing `version` fields automatically. Note that validating the marketplace root (`claude plugin validate .`) is shallower and did **not** surface F2 — per-plugin invocation is required.

Severity counts: **5 × P0**, **7 × P1**, **19 × P2**, **16 × P3**.

---

## 2. Validation performed

All commands were read-only with respect to the repository; the working tree remains clean at `7c44b10`. Throwaway fixtures were created under `/tmp` and removed.

| Check | Command | Result |
|---|---|---|
| JSON syntax | `jq empty` on all 11 `.json` | 11/11 OK |
| Shell syntax | `bash -n` on all 12 `.sh` | 12/12 OK |
| Manifest validation | `claude plugin validate ./egg` | passes with 1 warning (no `version`) |
| Manifest validation | `claude plugin validate ./aerion` | **1 error** — command frontmatter parse failure (F2) |
| Manifest validation | `claude plugin validate ./dunk` | passes with 1 warning (no `version`) |
| Marketplace validation | `claude plugin validate . --strict` | fails on 3 × missing `version`; **did not detect F2** |
| Strict mode | `claude plugin validate ./<p> --strict` | all three fail |
| ks-merge tests | `uv run pytest tests/test_ks_merge.py` | **52 passed** in 1.35s |
| migration tests | `uv run pytest migrations/test_migrate_docker.py` | **17 passed** in 0.01s |
| Plugin `.mcp.json` shape | `claude --plugin-dir <fixture> mcp list`, bare vs wrapped | **both load** — bare form is tolerated |
| Project `.mcp.json` shape | `claude mcp list` in a project using `templates/mcp-personal.json` | **`[Error] mcpServers: Invalid input: expected record, received undefined`** (F1) |
| Transcript schema | `jq` against a live `~/.claude/projects/**/*.jsonl` | `has_content:false`, `has_message:true` → F4 confirmed |
| YAML isolation | Ruby `Psych` per frontmatter line | `argument-hint: [days] (default: 7)` is the sole failure (F2) |
| Ralph loop logic | replayed lines 5 + 70 in a fixture | F21 and F22 both reproduced |
| Schema URL liveness | `curl -I` | marketplace `$schema` → **HTTP 404** |
| aerion endpoint | `curl` Cloud Run `/sse` | **HTTP 503** |
| Description budget | measured decoded frontmatter values | `global-markets-teacher` = 1554 chars vs 1536 cap |
| Secret scan | regex sweep for `sk-`, `ghp_`, `AIza`, inline API keys | clean |

---

## 3. P0 — Confirmed broken functionality

### F1. Both MCP templates omit the `mcpServers` wrapper → a project using them loads zero servers

**Files:** `templates/mcp-personal.json:1`, `templates/mcp-all.json:1`

Both files are a bare map of server names:

```json
{
  "git": { "command": "uvx", "args": ["mcp-server-git"] },
  "context7": { ... }
}
```

`README.md:30-34` and `CLAUDE.md:78` instruct users to copy a template to a project root `.mcp.json`. A project-scoped `.mcp.json` requires the wrapper. Reproduced:

```
$ cd /tmp/projmcp/bare && claude mcp list
 └ [Error] mcpServers: Invalid input: expected record, received undefined
```

With the wrapper, the same fixture loads correctly. (Plugin-root `.mcp.json` files in `egg`/`dunk`/`aerion` *do* tolerate the bare form — verified with `--plugin-dir` fixtures — so those are a separate, lower-severity concern; see F43a.)

**Fix.** Wrap both templates:

```json
{
  "mcpServers": {
    "git": { "command": "uvx", "args": ["mcp-server-git"] },
    "...": {}
  }
}
```

---

### F2. `/check-apps` frontmatter is invalid YAML → all metadata silently dropped

**File:** `aerion/commands/check-apps.md:3`

```yaml
argument-hint: [days] (default: 7)
```

YAML parses `[days]` as a flow sequence, then fails on the trailing `(default: 7)`. Isolated per-line with Ruby's YAML parser: this line is the *only* failure; quoting it parses cleanly. The official validator is explicit about the consequence:

```
$ claude plugin validate ./aerion
Validating command: .../aerion/commands/check-apps.md
✘ frontmatter: YAML frontmatter failed to parse: YAML Parse error: Unexpected token.
  At runtime this command loads with empty metadata (all frontmatter fields silently dropped).
```

So `description` (line 2), `argument-hint` (line 3) **and** `allowed-tools` (line 4) are all lost. `/check-apps` shows no description in the `/` menu, and every Gmail/Sheets tool call prompts for permission because the pre-approval list never applies.

**Fix.**

```yaml
argument-hint: "[days] (default: 7)"
```

---

### F3. dunk's Notion MCP tool names use the wrong scoped prefix → `dln-sync` has zero tools and the KS hook never fires

**Files:** `dunk/agents/dln-sync.md:14-17`, `dunk/agents/dln-syllabus.md:18-20`, `dunk/hooks/hooks.json:5`, `dunk/scripts/validate-ks-markers.sh:3` (comment)

The scoped form is documented verbatim as `mcp__plugin_<plugin-name>_<server-name>__<tool-name>`. The plugin is `dunk` (`dunk/.claude-plugin/plugin.json:2`) and the server key is `Notion` (`dunk/.mcp.json:10`), so every name must be `mcp__plugin_dunk_Notion__<tool>`. All nine sites instead use `mcp__plugin_Notion_notion__<tool>` — plugin name and server name swapped and re-cased.

Two independent failures follow:

1. **`dln-sync` is inert.** Its `tools:` list (lines 14-17) contains *only* the four malformed Notion names. An agent `tools:` field is an allowlist, so the agent that performs *all* Notion I/O for the DLN running ledger resolves to an empty toolset. Every `dln-dot` / `dln-linear` / `dln-network` sync boundary that dispatches it cannot complete.
2. **The marker-validation hook never runs.** `dunk/hooks/hooks.json:5` matches on the same malformed name. The docs note this failure mode directly: *"A matcher written against the bare server key never fires."* So `validate-ks-markers.sh` — the guard that prevents escaped/unescaped `<!-- KS:start -->` corruption reaching Notion — is unreachable, and the escaping bug it exists to catch ships unchecked.

`dln-syllabus.md:18-20` has the same defect, leaving that agent with only `WebSearch`, `WebFetch`, and the (correctly named) context7/exa tools — it can research but cannot write the syllabus to Notion.

**Fix.** Replace `mcp__plugin_Notion_notion__` with `mcp__plugin_dunk_Notion__` at all nine sites. Then re-verify the tool names themselves against a live `/mcp` listing.

---

### F4. Stop-hook turn counter reads a field that does not exist → write-back enforcement is dead

**Files:** `egg/scripts/learner-profile-check.sh:39-43`, `egg/scripts/markets-profile-check.sh:37-41`

```bash
USER_TURNS=$(jq -r '
  select(.type == "user")
  | select((.content // []) | any(.type != "tool_result"))
  | 1
' "$TRANSCRIPT" 2>/dev/null | wc -l | tr -d ' ')
```

Transcript entries carry their payload under `.message.content`, not `.content` — the scripts' own header comments say so (`learner-profile-check.sh:7` — *"Assistant tool use appears in message.content[]"*). `.content // []` therefore yields `[]`, `any` over an empty array is `false`, `select` emits nothing, and `USER_TURNS` is always `0`. The guard at line 47 (`[ "$USER_TURNS" -lt 2 ]`) then exits 0 unconditionally, so lines 51-81 — the entire enforcement body — are unreachable.

Verified against a live transcript:

```
$ jq -c 'select(.type=="user")|{has_content:has("content"),has_message:has("message")}' <transcript> | head -1
{"has_content":false,"has_message":true}

$ jq -r 'select(.type=="user")|select((.content // [])|any(.type!="tool_result"))|1' <transcript> | wc -l
0                       # current expression
$ jq -r 'select(.type=="user")|select((.message.content // [])|if type=="array" then any(.type!="tool_result") else true end)|1' <transcript> | wc -l
4                       # corrected expression
```

**Fix.** Use `.message.content`, and handle the string case — a plain user prompt has `.message.content` as a *string*, not an array:

```bash
USER_TURNS=$(jq -r '
  select(.type == "user")
  | select((.message.content // []) | if type == "array" then any(.type != "tool_result") else true end)
  | 1
' "$TRANSCRIPT" 2>/dev/null | wc -l | tr -d ' ')
```

Add a regression test that feeds a fixture transcript through the script and asserts exit 2.

---

### F5. `learner-profile-load.sh` is not wired to any hook → the LeetCode profile never loads

**Files:** `egg/hooks/hooks.json:14-23`, `egg/scripts/learner-profile-load.sh` (orphan)

`SessionStart` registers only one command:

```json
"SessionStart": [
  { "hooks": [ { "type": "command",
      "command": "bash ${CLAUDE_PLUGIN_ROOT}/scripts/markets-profile-load.sh" } ] }
]
```

`egg/scripts/learner-profile-load.sh` — 251 lines that create the profile/ledger templates, self-heal structure, sync the ledger, and emit the `=== LEARNER PROFILE ===` and `=== RETEST SUGGESTIONS ===` blocks — is referenced by nothing. Meanwhile `egg/skills/leetcode-teacher/SKILL.md:207` and `references/teaching/learner-profile-spec.md:7` expect that injected context, and the paired `Stop` hook (`egg/hooks/hooks.json:47`) still runs `learner-profile-check.sh`, whose failure message tells the user to write files at `~/.local/share/claude/` that were never created.

Combined with F4, the entire LeetCode persistence loop is non-functional: nothing loads it, and nothing enforces it.

**Fix.** Add the missing `SessionStart` entry alongside the markets one:

```json
"SessionStart": [
  { "hooks": [
      { "type": "command",
        "command": "bash \"${CLAUDE_PLUGIN_ROOT}\"/scripts/learner-profile-load.sh",
        "timeout": 15 },
      { "type": "command",
        "command": "bash \"${CLAUDE_PLUGIN_ROOT}\"/scripts/markets-profile-load.sh",
        "timeout": 15 } ] }
]
```

See F18 for why both should additionally be gated rather than run on every session.

---

## 4. P1 — High

### F6. Plugin skill `name:` values contain spaces → the documented command name is unreachable

**Files:** `egg/skills/doc-generator/SKILL.md:2`, `egg/skills/tech-blog/SKILL.md:2`

```yaml
name: Documentation Generator      # doc-generator/SKILL.md
name: Technical Blog Writer        # tech-blog/SKILL.md
```

For personal/project skills `name` is only a display label, but the docs are explicit that plugin skills differ: *"In a plugin skill, `name` sets the last segment of the command and the plugin prefix stays in place… the frontmatter `name` replaces the directory name in the last segment of the command."* A value containing spaces cannot form a usable command segment, so `/egg:doc-generator` and `/egg:tech-blog` — the names `CLAUDE.md:38` and `CLAUDE.md:40` document — do not resolve.

All 16 other skills correctly use the kebab-case directory name.

**Fix.** Set `name: doc-generator` and `name: tech-blog`; move the prose titles to `displayName`-style headings in the body (which both files already have at line 10 / line 6).

---

### F7. `permissionMode` is not supported on plugin agents

**Files:** `egg/agents/leetcode-profile-sync.md:17`, `dunk/agents/dln-sync.md:20`

```yaml
permissionMode: dontAsk
```

The plugins reference is unambiguous: *"For security reasons, `hooks`, `mcpServers`, and `permissionMode` are not supported for plugin-shipped agents."* The field is silently ignored — `claude plugin validate` does not flag it, which is exactly what makes this dangerous.

Both agents are documented as *"Internal — dispatched programmatically… Not user-facing"* and their design assumes silent file/Notion writes. In practice each write raises a permission prompt mid-teaching, breaking the Socratic flow the skills are built around.

**Fix.** Delete the `permissionMode` lines. To get non-interactive writes, either (a) list the needed tools in the invoking skill's `allowed-tools`, or (b) narrow the agent's `tools` so the user's own permission rules can pre-approve them. Update the skills' fallback instructions to assume a prompt may occur.

---

### F8. `python-lint.sh` sends all diagnostics to stdout and exits 1 → Claude never sees them

**File:** `egg/scripts/python-lint.sh:103-157`

Lines 103-156 `echo` roughly 50 lines of remediation guidance to **stdout**, then line 157 does `exit 1`. Per the hooks reference:

- Exit 0 → *"Stdout written to debug log"* (surfaced to Claude only for `UserPromptSubmit`, `UserPromptExpansion`, `SessionStart` — not `PostToolUse`).
- Exit 1 → *"Transcript shows `<hook name> hook error: Failed with non-blocking status code: <stderr>`"*.
- `PostToolUse` + exit 2 → *"Tool already ran; **shows stderr to Claude**"*.

So the current combination surfaces an error with an **empty** message and discards every diagnostic. The script's own header at line 9 already states the intent — *"exits with code 2 for manual fixes needed"* — so this is a drift between comment and code.

**Fix.** Redirect the diagnostic block to stderr and exit 2:

```bash
{
  echo "Manual Fixes Required"
  echo "File: $FILE_PATH"
  echo "$REMAINING"
  # ...
} >&2
exit 2
```

Alternatively emit structured JSON on stdout with `hookSpecificOutput.additionalContext` and exit 0. Either way, also fix line 9's comment or the code so they agree.

---

### F9. `python-lint.sh` applies `ruff --unsafe-fixes` to arbitrary files with no review

**File:** `egg/scripts/python-lint.sh:76-81`

```bash
FIX_OUTPUT=$(ruff check "$FILE_PATH" \
    --select=$RUFF_RULES --ignore=$RUFF_IGNORE --line-length=$LINE_LENGTH \
    --fix --unsafe-fixes 2>&1) || true
```

Ruff classifies a fix as *unsafe* precisely when it may change program behaviour. This hook applies those fixes automatically, on every `Write|Edit`, before the user or Claude sees the file. Line 29 only checks that the path ends in `.py` — there is no confinement to the project, so any `.py` file Claude touches anywhere on disk is rewritten. The reported diff is also invisible to Claude because of F8, so a semantic change can land completely silently.

`CLAUDE.md:70` documents the hook as *"Auto-formats and auto-fixes first"* without mentioning `--unsafe-fixes`.

**Fix.** Drop `--unsafe-fixes`. If unsafe fixes are genuinely wanted, run them in a separate opt-in path and report the resulting diff to Claude via stderr.

---

### F10. `aerion` depends on a private single-tenant Cloud Run instance, over deprecated SSE, and it is currently down

**File:** `aerion/scripts/google-sheets-mcp.sh:5`

```bash
exec npx -y mcp-remote "https://mcp-google-sheets-595255742975.asia-southeast1.run.app/sse"
```

Three problems in one line:

1. **Not portable.** The URL is hardcoded to the author's own GCP project (number `595255742975`) in `asia-southeast1`. Anyone installing `aerion@ashford` from the public marketplace is pointed at that instance. There is no configuration hook, so the plugin cannot work for anyone else — and if the service accepts unauthenticated invocations, every installer shares one set of Google credentials.
2. **Unavailable.** `curl --max-time 12` against the endpoint returns **HTTP 503**.
3. **Deprecated transport.** The MCP docs state: *"The SSE (Server-Sent Events) transport is deprecated. Use HTTP servers instead, where available."*

**Fix.** Make the endpoint configurable via `userConfig` in `aerion/.claude-plugin/plugin.json` — this is the documented mechanism, and `sensitive: true` stores values in secure storage rather than `settings.json`:

```json
{
  "userConfig": {
    "sheets_mcp_url": {
      "type": "string",
      "title": "Google Sheets MCP endpoint",
      "description": "HTTPS URL of your mcp-google-sheets deployment"
    }
  }
}
```

Then declare the server as `"type": "http"` with `"url": "${user_config.sheets_mcp_url}"` and delete the wrapper script. Document the deployment steps in a README, and set `defaultEnabled: false` so the plugin does not install into a broken state.

---

### F11. A `resume-builder` skill is referenced 21 times but does not exist

**Files:** `README.md:16`, `README.md:50`, `egg/skills/behavioral-interview-prepper/SKILL.md:3,8,13,24,35,94,95,96`, `egg/skills/behavioral-interview-prepper/references/story-mapping.md:77`, `egg/skills/behavioral-interview-prepper/references/candidate-discovery.md:5`, `egg/skills/behavioral-interview-prepper/evaluations/trigger-tests.md:15,26,29,37`, `egg/skills/resume-analyzer/references/candidate-discovery.md:5`, `egg/skills/technical-interview-roadmap/SKILL.md:12,30`, `egg/skills/technical-interview-roadmap/evaluations/trigger-tests.md:17,29,39,51`

The skill was evidently split into `resume-analyzer` + `resume-tailor` (as `CLAUDE.md:39` correctly describes) but the references were never updated. The worst case is behavioural, not cosmetic:

```
egg/skills/behavioral-interview-prepper/SKILL.md:35:
**If required files do not exist:** Prompt the user to run the `resume-builder`
skill first with the target JD.
```

Claude is instructed to send the user to a skill that cannot be invoked. `trigger-tests.md:26,29` also route negative cases to `resume-builder`, so the eval fixtures encode the wrong expectation.

**Fix.** Global rename. Producer of `notes.md` → `resume-analyzer`; producer of `resume.tex` → `resume-tailor`; producer of `cover-letter.md` → `cover-letter`. Update the four `trigger-tests.md` rows and both `README.md` lines.

---

### F12. `EXA_API_KEY` is passed in argv → readable by any local process

**Files:** `egg/scripts/exa-mcp.sh:5`, `dunk/scripts/exa-mcp.sh:5`

```bash
exec npx -y mcp-remote "https://mcp.exa.ai/mcp?exaApiKey=${EXA_API_KEY}&tools=..."
```

The key becomes part of the `npx` command line and is visible in `ps` output, `/proc/<pid>/cmdline`, process monitors, and any crash report or `mcp-remote` error log that echoes its arguments. With no `set -u` and no validation, an unset `EXA_API_KEY` silently produces `exaApiKey=` and the server fails to authenticate with no diagnostic.

The header comment (lines 3-4) justifies the wrapper as a workaround for `${ENV_VAR}` not interpolating in `.mcp.json`. That is accurate for arbitrary env vars, but `userConfig` is now the supported path.

**Fix.** Declare the key via `userConfig` with `sensitive: true`, then use an `http` server entry with the value in `headers` (which is not shell-parsed) rather than the URL query string. If the wrapper must stay, at minimum add:

```bash
set -euo pipefail
: "${EXA_API_KEY:?EXA_API_KEY is not set — configure it before enabling this plugin}"
```

and prefer `mcp-remote --header` over a query parameter.

---

## 5. P2 — Medium

### F13. No plugin declares a `version`; strict validation fails for all three

**Files:** `egg/.claude-plugin/plugin.json`, `aerion/.claude-plugin/plugin.json`, `dunk/.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json:8-25`

```
$ claude plugin validate ./egg --strict
⚠ version: No version specified. Consider adding a version following semver (e.g., "1.0.0")
✘ Validation failed (--strict treats warnings as errors)
```

Omitting `version` is a legitimate strategy — the docs' "Commit-SHA version" approach, appropriate for plugins under active development. But it contradicts `egg/CHANGELOG.md:5`, which declares `## [1.0.0] - 2025-12-17`, and it means `--strict` can never be used in CI.

**Fix.** Pick one strategy and make it consistent. Given the CHANGELOG exists, add explicit `version` fields (`egg` → `1.1.0` or later given the components added since 1.0.0), create `CHANGELOG.md` for `aerion` and `dunk`, and bump on every release. Otherwise delete `egg/CHANGELOG.md` and document the commit-SHA strategy in `CLAUDE.md`.

### F14. `global-markets-teacher` description exceeds the 1536-character listing cap

**File:** `egg/skills/global-markets-teacher/SKILL.md:3`

Measured length of the decoded value: **1554 characters**. The docs cap the combined `description` + `when_to_use` at 1536 characters in the skill listing. The final 18 characters — `interview simulation.` — are truncated, and every skill listing pays 1554 characters of context.

**Fix.** Trim to ~900 characters (in line with `leetcode-teacher`'s 822 and `dln`'s 894), keeping the highest-value triggers first as the docs advise. Move the long firm-name enumeration into the body or into `when_to_use`, remembering that `when_to_use` also counts toward the same cap.

### F15. Unrecognized skill frontmatter keys

**Files:** `egg/skills/doc-generator/SKILL.md:4-7` (`trigger_patterns`), `egg/skills/ml-paper-writing/SKILL.md:4,5,7,8` (`version`, `author`, `tags`, `dependencies`)

None of these are in the documented skill frontmatter set. `license` (line 6 of `ml-paper-writing`) *is* valid. Unrecognized keys are ignored at runtime and reported as warnings — which `--strict` promotes to errors. `trigger_patterns` in particular reads as if it drives activation; it does not, and `doc-generator`'s 51-character `description` is the only signal Claude actually gets.

**Fix.** Move non-standard data under the sanctioned `metadata:` map, or delete it. Fold `trigger_patterns` content into `description` / `when_to_use` so it has an effect.

### F16. Session-state `.processed` rename crosses directories

**File:** `egg/skills/global-markets-teacher/SKILL.md:184`

> If `~/.local/share/claude/markets-session-state.md` exists… Rename the file to `~/.claude/markets-session-state.md.processed`

Source is `~/.local/share/claude/`, destination is `~/.claude/` — and `egg/scripts/markets-session-snapshot.sh:5` writes only to the former. The LeetCode equivalent (`egg/skills/leetcode-teacher/SKILL.md:205`) keeps both in `~/.local/share/claude/`, so this is a one-off typo. Effect: a stale scratchpad may be re-read, or the rename fails because `~/.claude/` semantics differ.

**Fix.** Change the destination to `~/.local/share/claude/markets-session-state.md.processed`.

### F17. Persistent state is hand-rolled under `$HOME` instead of `${CLAUDE_PLUGIN_DATA}`

**Files:** `egg/scripts/learner-profile-load.sh:5-7`, `egg/scripts/markets-profile-load.sh:5-7`, `egg/scripts/learner-session-snapshot.sh:5`, `egg/scripts/markets-session-snapshot.sh:5`, plus ~35 doc references across both teacher skills

`${CLAUDE_PLUGIN_DATA}` appears **zero** times in the repository. It is the documented directory that survives plugin updates, resolves to `~/.claude/plugins/data/{id}/`, is created on first reference, and is cleaned up on uninstall. The current `$HOME/.local/share/claude` layout works but has real drawbacks: it is not cleaned up on uninstall, it collides across marketplaces that pick the same filename, and it is invisible to the `/plugin` data-size UI.

**Fix.** Prefer `${CLAUDE_PLUGIN_DATA}` with a one-time migration from the legacy path (`[ -f "$LEGACY" ] && [ ! -f "$NEW" ] && mv …`). This touches every path reference in both teacher skills, so schedule it as one atomic workstream (WS-G).

### F18. `SessionStart` hook has no matcher → the markets profile is injected into every session of every project

**File:** `egg/hooks/hooks.json:14-23`

With no `matcher`, `SessionStart` fires on `startup`, `resume`, `clear`, `compact`, *and* `fork`. Consequences for anyone who installs `egg` for `/commit` and `code-reviewer`:

- `markets-profile-load.sh:18` creates `~/.local/share/claude/` and two files on first run, unprompted.
- The full profile is `cat`-ed to stdout (line 155) and injected as context in every session, in every project, regardless of topic.
- It re-fires after each compaction, re-injecting the profile the compaction just removed.

**Fix.** Two options, best combined: (a) add `"matcher": "startup|clear"` to stop re-injection on resume/compact/fork; (b) make the scripts no-op unless the profile already exists, and let the teacher skills create it on first genuine use rather than at install time. Also set explicit `timeout` values (F19).

### F19. No hook declares a `timeout`

**Files:** `egg/hooks/hooks.json:7-10,17-21,27-31,35-39,45-49,53-57`; `dunk/hooks/hooks.json:7-11`

Every entry relies on the 600-second default. Hook security best practice #3 is *"Set restrictive `timeout` values."* The `Stop` hooks in particular `grep` an entire transcript file up to four times, which grows without bound over a long session.

**Fix.** Add `"timeout": 10` to the lint and validation hooks and `"timeout": 15` to the profile hooks.

### F20. Shell-form hook commands leave `${CLAUDE_PLUGIN_ROOT}` unquoted

**Files:** `egg/hooks/hooks.json:9,19,29,37,47,55`; `dunk/hooks/hooks.json:9`

```json
"command": "bash ${CLAUDE_PLUGIN_ROOT}/scripts/python-lint.sh"
```

The docs prefer exec form (*"use exec form with `args` so each path is passed as one argument with no quoting"*) and otherwise require quoting (*"wrap the variables in double quotes"*). A plugin root containing a space — realistic for local `--plugin-dir` development under a path like `~/My Projects/` — word-splits and the hook silently fails.

**Fix.** Convert to exec form:

```json
{ "type": "command", "command": "bash",
  "args": ["${CLAUDE_PLUGIN_ROOT}/scripts/python-lint.sh"], "timeout": 10 }
```

### F21. Ralph loop extracts the wrong task number

**File:** `egg/scripts/ralph-loop-template.sh:70`

```bash
NEXT=$(grep -m1 '^\- \[ \]' "$PLAN_FILE" | grep -oE '[0-9]+' | head -1)
```

`grep -oE '[0-9]+' | head -1` takes the first digit run *anywhere* on the line, not the leading task number. Reproduced:

```
$ printf -- '- [x] 1 Setup\n- [ ] Add OAuth2 support\n' > PLAN.md
$ grep -m1 '^\- \[ \]' PLAN.md | grep -oE '[0-9]+' | head -1
2
```

`NEXT=2` then drives the lookup at line 83 and the `sed` at line 88, so the loop can execute or block the wrong task.

**Fix.** Anchor the capture:

```bash
NEXT=$(sed -n 's/^- \[ \] \([0-9]\{1,\}\).*/\1/p' "$PLAN_FILE" | head -1)
```

### F22. Ralph loop aborts silently instead of reporting completion

**File:** `egg/scripts/ralph-loop-template.sh:5,70-78`

`set -euo pipefail` (line 5) plus the line-70 pipeline means that when no unchecked task remains, `grep -m1` exits 1, `pipefail` propagates it to the assignment, and `set -e` terminates the script — **before** the `if [[ -z "$NEXT" ]]` completion branch at lines 72-78. Reproduced:

```
$ bash t.sh; echo "exit=$?"
exit=1          # no "Ralph Loop Complete" printed
```

So a fully successful run exits 1 with no output, which any wrapper reads as failure.

**Fix.** `NEXT=$(… || true)`, or wrap in `if ! NEXT=$(…); then NEXT=""; fi`.

### F23. Ralph loop has no iteration cap around an autonomous `claude -p`

**File:** `egg/scripts/ralph-loop-template.sh:68,143`

`while :; do … echo -e "$PROMPT" | claude -p …` terminates only when `PLAN.md` has no unchecked boxes — and `PLAN.md` is updated by the very Claude invocation inside the loop. If Claude fails to tick a box (a permission denial, a transient API error, a misread protocol), the loop re-runs the identical task forever, consuming tokens without bound. Only the missing-task-file case is defended (lines 85-90).

**Fix.** Add `MAX_ITERATIONS="${MAX_ITERATIONS:-25}"` with a hard break, detect a repeated `NEXT` value across consecutive iterations and mark it blocked, and propagate a non-zero exit from `claude -p` instead of swallowing it with `2>&1 | tee`.

### F24. `echo -e` corrupts injected source content

**File:** `egg/scripts/ralph-loop-template.sh:143` (also `egg/scripts/learner-profile-load.sh:162`, `egg/scripts/markets-profile-load.sh:157`)

`PROMPT` is assembled with literal `\n` sequences and rendered with `echo -e`, so `echo -e` also interprets any backslash escape present in the *file contents* injected at line 126 (`PROMPT+="$(cat "$filepath")"`). A Python file containing `"\t"` or `"\0"` is silently mangled before Claude sees it. `echo -e` is additionally not portable across shells.

**Fix.** Build the prompt with real newlines (`$'\n'` or a heredoc) and emit with `printf '%s'`.

### F25. A personal `hojicha/` directory is hardcoded into published skills

**Files:** `egg/skills/resume-tailor/SKILL.md:8,15,27,28,79,83`; `egg/skills/behavioral-interview-prepper/SKILL.md:8,28,29,30,48,93,197,201,202,203`; `egg/skills/behavioral-interview-prepper/references/candidate-discovery.md:103`

The skills read `hojicha/resume.tex`, `hojicha/fed-res.cls`, `hojicha/candidate-context.md` and write to `hojicha/<company>-<role>-resume/`. `hojicha/` is the author's private directory — and `.gitignore:28` confirms it, listing `hojicha/` under "Misc. Folders". No installer has it, so all three skills fail at step 1 with a missing-file error.

**Fix.** Parameterize the root: default to a discoverable location (e.g. `${CLAUDE_PROJECT_DIR}/resumes/` or a `userConfig` `directory` option), and have the skills prompt for it on first use if absent. Keep `hojicha` only as an example in prose.

### F26. Fragile regex JSON parsing in the lint hook

**File:** `egg/scripts/python-lint.sh:17-21`

```bash
FILE_PATH=$(echo "$TOOL_INPUT" | sed -n 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
```

The leading greedy `.*` means the **last** `"file_path"` in the payload wins. A `PostToolUse` payload also carries `tool_result.content`, so a tool result that echoes a `"file_path"` string hijacks the match. JSON escapes (`\"`, `\\`, `\u`) are also not decoded, so any path containing them resolves incorrectly.

**Fix.** `jq` is already a hard dependency of the sibling hook scripts:

```bash
FILE_PATH=$(jq -r '.tool_input.file_path // .tool_input.path // empty' <<<"$TOOL_INPUT")
```

Keep a `command -v jq` fail-open guard, matching `validate-ks-markers.sh:13-15`.

### F27. `\s` in `grep -E` is not POSIX ERE

**Files:** `egg/scripts/learner-profile-check.sh:30,57,65,67`; `egg/scripts/markets-profile-check.sh:28,50,52`

```bash
grep -E '"name"\s*:\s*"Read"' "$TRANSCRIPT"
```

`\s` is a GNU/PCRE extension, undefined in POSIX ERE. GNU grep and `ugrep` accept it; stock BSD `grep` on macOS is the documented risk case, and these hooks ship to whatever `grep` the user has. If `\s` is not honoured the pattern silently matches nothing and the hook fails open. (Note this is currently masked on the author's machine, where `grep` resolves to `ugrep 7.5.0`.)

**Fix.** Use `[[:space:]]*`, or move the whole match into `jq` (which the scripts already use) and drop the grep-over-JSON approach entirely.

### F28. Predictable `/tmp` paths for KS merge payloads, left behind on failure

**File:** `dunk/skills/dln/references/merge-protocol.md:34,35,42,68`

The protocol writes `/tmp/ks-merge-payload-<page_id_8chars>.json` and `/tmp/ks-merge-ks-<page_id_8chars>.md`. The names are fully predictable from a Notion page ID, so on a shared host another user can pre-create a symlink at those paths and redirect the write, or read the learner's knowledge-state content. Line 68 deliberately preserves both files when Notion sync fails, leaving them indefinitely.

**Fix.** Use `mktemp -d` and place both files inside the private directory, or write under `${CLAUDE_PLUGIN_DATA}/tmp/`. Add a `trap … EXIT` cleanup, and on failure move the files somewhere non-world-readable rather than leaving them in `/tmp`.

### F29. Stop-hook loop guard reads a field absent from the current documented input

**Files:** `egg/scripts/learner-profile-check.sh:18`, `egg/scripts/markets-profile-check.sh:16`

```bash
STOP_HOOK_ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false' 2>/dev/null)
```

The current hooks reference lists the `Stop` input fields as the common set plus `last_assistant_message` and `stop_reason`. `stop_hook_active` is not among them. If it is no longer sent, the guard evaluates to `false` every time and the re-entrancy protection is inert — meaning that once F4 is fixed and these hooks start returning exit 2, they could loop.

**Fix.** Before shipping the F4 fix, verify empirically whether `stop_hook_active` is still delivered (log the raw stdin from a temporary `Stop` hook). If not, implement an independent guard — e.g. a marker file under `${CLAUDE_PLUGIN_DATA}` keyed by `session_id`, cleared on `SessionStart` — and cap the number of blocks per session. This is a prerequisite for F4, not an independent cleanup.

### F30. Unescaped interpolation corrupts the ledger markdown table

**Files:** `egg/scripts/learner-profile-load.sh:122,143`; `egg/scripts/markets-profile-load.sh:117,138`

```bash
echo "| ${LAST_PROFILE_TS} | sync-heal | ${P_PROBLEM} | unknown | ${P_MODE} | ${P_VERDICT} | ${P_GAPS} | ${P_REVIEW} |" >> "$LEDGER"
```

Values come from `cut -d'|'` over profile text the learner (or Claude) authored. A `|` or newline inside any of them adds spurious columns or rows, corrupting the file that `learner-profile-spec.md` designates the source of truth. Subsequent `sort -r | head -1` timestamp detection (lines 93-99) then reads garbage.

**Fix.** Escape `|` as `\|` and strip newlines before interpolation, e.g. `P_GAPS=${P_GAPS//|/\\|}` plus `tr -d '\n'`.

### F31. Non-standard glob patterns in `allowed-tools`

**File:** `aerion/commands/check-apps.md:4`

```yaml
allowed-tools: mcp__plugin_aerion_gmail__*, mcp__plugin_aerion_google-sheets__*, mcp__*gmail*, mcp__*google-sheets*, mcp__*google_sheets*
```

MCP permission rules are documented as `mcp__server` (all tools on a server) or `mcp__server__tool`. Neither a trailing `__*` nor an infix `mcp__*gmail*` is a documented form. This is currently moot because F2 drops the field entirely, but it will start mattering the moment F2 is fixed.

**Fix.** After fixing F2, use `mcp__plugin_aerion_gmail` and `mcp__plugin_aerion_google-sheets` and drop the three infix-glob entries. Verify the resulting names against a live `/mcp` listing.

---

## 6. P3 — Low / hygiene

### F32. Marketplace `$schema` URL returns 404

**File:** `.claude-plugin/marketplace.json:2` — `https://anthropic.com/claude-code/marketplace.schema.json` → **HTTP 404**. Claude Code ignores the field at load time, so nothing breaks at runtime, but editors get no completion or validation. The docs' cited plugin-manifest schema (`https://json.schemastore.org/claude-code-plugin-manifest.json`) returns **HTTP 200**.
**Fix.** Remove the dead URL, and add the working schemastore `$schema` to each `plugin.json`.

### F33. No CI or validation automation

No `.github/` directory and no workflow files exist. F2 and F13 are both machine-detectable.
**Fix.** Add a workflow that runs, per plugin, `claude plugin validate ./<plugin> --strict`, plus `jq empty` over all JSON, `bash -n` (ideally `shellcheck`) over all `.sh`, and both `pytest` suites. Validate each plugin directory individually — validating the marketplace root did **not** catch F2.

### F34. `README.md` and `CLAUDE.md` are materially stale

- `CLAUDE.md:41` claims *"50 reference files"* for `leetcode-teacher`; the actual count is **69** — and `egg/skills/leetcode-teacher/CLAUDE.md:7` already says 69, so the root file is the stale one. (`CLAUDE.md:42`'s count of 28 for `global-markets-teacher` is correct.)
- `README.md:38-50` presents the repo as `egg/` + `templates/` only; `aerion/`, `dunk/`, and `.claude-plugin/marketplace.json` are absent.
- `README.md:8-16` documents 2 commands, 1 agent, 5 real skills. Actual: **6 commands, 4 agents, 18 skills**.
- `CLAUDE.md:31-33` omits `/evaluate-feature-ccskill` and `/ralph`; `CLAUDE.md:31-61` omits `behavioral-interview-prepper`, `ralph`, `technical-interview-roadmap`.
- `CLAUDE.md:18-24`'s component table lists only `egg/…` paths despite naming three plugins.
- `CLAUDE.md:63-73` documents only the lint hook; the `SessionStart`, two `PreCompact`, and two `Stop` hooks are undocumented.
- `README.md:21-26` and `CLAUDE.md:82-84` say installing `egg@ashford` *"delivers all components"*; `aerion@ashford` and `dunk@ashford` require separate installs.
- `CLAUDE.md:10` says *"No … package management"*; `dunk/scripts/{pyproject.toml,uv.lock,.python-version}` exist.
- `CLAUDE.md:22,78` inventory only `egg/.mcp.json`, omitting `dunk`'s `Notion` and `aerion`'s `gmail` / `google-sheets`.
- `CLAUDE.md:88` still frames commands as using a `model` field; no command declares one.

### F35. `egg/skills/leetcode-teacher/CLAUDE.md` is never loaded

The plugins reference states: *"A `CLAUDE.md` file at the plugin root is not loaded as project context. Plugins contribute context through skills, agents, and hooks rather than CLAUDE.md."* A `CLAUDE.md` nested inside a skill directory is likewise not a discovered component — only `SKILL.md` is. The file ships in every install and reads as authoritative guidance that Claude never sees.
**Fix.** Move it to `docs/` in the repo (outside any plugin), or convert it into a `references/` file the skill actually reads.

### F36. `dunk/references/merge-payload-schema.md` is orphaned

A repo-wide grep for `merge-payload-schema` returns **zero** hits. `references/` at the plugin root is not a discovered component location, so the file is unreachable.
**Fix.** Move it under `dunk/skills/dln/references/` and reference it from `merge-protocol.md`, or delete it.

### F37. One-off migration tooling ships to every installer

`dunk/migrations/migrate_docker.py` (484 lines) and `test_migrate_docker.py` (328 lines) are described in-file as *"One-off migration script for the Docker DLN profile."* Everything in the plugin root is copied into `~/.claude/plugins/cache` on install.
**Fix.** Move to a repo-level `tools/` or `dev/` directory outside `dunk/`, or exclude via explicit manifest component paths.

### F38. Release-hygiene gaps

`egg/LICENSE` and `egg/CHANGELOG.md` exist; `aerion` and `dunk` have neither, and there is no repository-root `LICENSE`. No `plugin.json` declares `license`, `homepage`, `repository`, or `keywords`; no marketplace plugin entry declares `category`, `tags`, or `version`. `owner` in `.claude-plugin/marketplace.json:5-7` has only `name` (legal — `email`/`url` are optional).
**Fix.** Add a root `LICENSE`, per-plugin `CHANGELOG.md`, and the discovery metadata. `keywords` must be an array — a string value is a hard load error.

### F39. Skill evaluation coverage is uneven

7 of 18 skills have no `evaluations/` directory: `doc-generator`, `ml-paper-writing`, `mlx-dev`, `tech-blog` (egg); all five `dunk` skills; `job-tracker` (aerion). The 11 that do have `trigger-tests.md`. `doc-generator` is the most exposed — a 51-character description plus the inert `trigger_patterns` (F15) is its entire activation signal.
**Fix.** Add `evaluations/trigger-tests.md` to the remaining skills, prioritizing `doc-generator` and `tech-blog` whose F6 name defects also affect invocation.

### F40. `.gitignore:29` ignores `docs/` — including this report

`docs/` is listed under "Misc. Folders", so `docs/reviews/` is untracked by default and this review will not be committed without `git add -f`.
**Fix.** Narrow the rule (e.g. `docs/ralph/` and `docs/scratch/`) or add `!docs/reviews/`.

### F41. `.gitignore` omits `.pytest_cache/`

Running the test suites creates `dunk/scripts/.pytest_cache`. `.venv/` is covered at line 21; `.pytest_cache/` is not, so it relies on the developer having a global excludes file.
**Fix.** Add `.pytest_cache/` and `.ruff_cache/`.

### F42. `exa-mcp.sh` is byte-identical in two plugins

`egg/scripts/exa-mcp.sh` and `dunk/scripts/exa-mcp.sh` are identical. Duplication is *required* — the docs are explicit that *"paths that traverse outside the plugin root (such as `../shared-utils`) will not work after installation"* — but it invites drift, and both copies must be fixed together for F12.
**Fix.** Either use the documented intra-marketplace symlink (dereferenced into the cache at install time) or add a CI check asserting the two files are identical.

### F43. Unpinned `@latest` MCP dependencies

`egg/.mcp.json:4,12`, `dunk/.mcp.json:4`, `templates/mcp-personal.json:8,12`, `templates/mcp-all.json:8,19` all use `npx -y …@latest`, and `mcp-remote` is unpinned in all three wrapper scripts. Every session pulls the newest published version and executes it, with no lockfile and no review — a standing supply-chain exposure.
**Fix.** Pin to exact versions and bump deliberately. Note that Claude Code's automatic dependency install only runs for a plugin-root `package.json` + lockfile, which none of these plugins has.

### F43a. Plugin `.mcp.json` files omit the documented `mcpServers` wrapper

`egg/.mcp.json:1`, `dunk/.mcp.json:1`, `aerion/.mcp.json:1` are bare maps. Empirically **both forms load** for plugin-root files (verified with `--plugin-dir` fixtures), so this is not currently broken — but every documented example uses the wrapper, and the bare form is an undocumented fallback that could be tightened in a future release. Unlike F1, this is precautionary.
**Fix.** Add the wrapper for forward-compatibility and consistency with F1's fix.

### F44. `chrome-devtools` hardcodes a debug port, inconsistently

`egg/.mcp.json:12` passes `--browserUrl http://127.0.0.1:9222`, which requires Chrome already running with `--remote-debugging-port=9222`; otherwise the server fails at every session start. Both templates omit the flag, so plugin and template behaviour diverge.
**Fix.** Drop the flag (letting the server launch its own browser) or document the prerequisite in `CLAUDE.md` and align the templates.

### F45. `commands/` is the legacy layout

`egg/commands/` (5 files) and `aerion/commands/` (1 file) use the flat-markdown form. The docs now list `commands/` as *"Skills as flat Markdown files. Use `skills/` for new plugins"* and state that custom commands have been merged into skills.
**Fix.** No action required — existing files keep working. Prefer `skills/<name>/SKILL.md` for anything new; migrating `/ralph` would also let it share a directory with `egg/skills/ralph/`, which currently splits one feature across two component types.

### F46. `dunk/scripts/pyproject.toml` is a scaffold placeholder

Lines 2-4: `name = "scripts"`, `description = "Add your description here"`. Line 5 requires `>=3.13`, which is aggressive for a stdlib-only script (`ks-merge.py` imports only `json`, `re`, `sys`, `dataclasses`).
**Fix.** Set a real name/description and relax to `>=3.10` unless a 3.13 feature is actually used — then verify with `uv run --python 3.10 pytest`.

### F47. `color` is not in the documented plugin-agent field list

`egg/agents/code-reviewer.md:34`, `egg/agents/leetcode-profile-sync.md:11`, `dunk/agents/dln-sync.md:12` set `color`. The plugins reference enumerates supported plugin-agent fields as `name`, `description`, `model`, `effort`, `maxTurns`, `tools`, `disallowedTools`, `skills`, `memory`, `background`, `isolation`. `color` is a general subagent field and the validator does not flag it. Harmless; noted for completeness.

### F48. `context7` tool name looks wrong

`dunk/agents/dln-syllabus.md:15` grants `mcp__plugin_dunk_context7__query-docs`. The context7 MCP server exposes `resolve-library-id` and `get-library-docs`; `query-docs` does not appear to exist. The prefix is correct here (unlike F3), so only the tool segment is suspect.
**Fix.** Verify against a live `/mcp` listing and change to `get-library-docs`.

---

## 7. What is already correct

Worth recording so remediation does not regress it:

- **Directory layout** matches the documented convention exactly — `.claude-plugin/` holds only `plugin.json`; `commands/`, `agents/`, `skills/`, `hooks/`, `scripts/`, `.mcp.json` are all at the plugin root.
- **Marketplace required fields** (`name`, `owner.name`, `plugins[].name`, `plugins[].source`) are all present and correctly typed; relative `./egg`-style sources are the documented form.
- **16 of 18 skills** use kebab-case `name` matching the directory, and all 18 have a `description`.
- **No writes inside the plugin cache directory.** A dedicated audit found zero instructions or scripts writing to `${CLAUDE_PLUGIN_ROOT}`-relative paths — exactly right, since that directory is replaced on update. All `${CLAUDE_PLUGIN_ROOT}` uses are reads or executions.
- **No hardcoded `/Users/…` paths** anywhere, and **no committed secrets**.
- **Cross-plugin `@${CLAUDE_PLUGIN_ROOT}/skills/dln/references/…` includes** in the dunk phase skills are the correct pattern for sharing files within one plugin.
- **`validate-ks-markers.sh`** is the best-engineered script in the repo: `#!/usr/bin/env bash`, an explicit `command -v jq` fail-open guard, proper `jq`-based payload parsing, and correctly shaped `hookSpecificOutput` deny JSON. Only its matcher (F3) is wrong.
- **Test suites are real and green** — 52 tests for `ks-merge.py` and 17 for `migrate_docker.py`, both passing.
- **`ks-merge.py`** is stdlib-only, writes nothing, and emits to stdout — a clean, side-effect-free design.
- **`egg/agents/code-reviewer.md`** uses the `<example>`-block description pattern correctly and declares no unsupported fields.

---

## 8. Implementation workstreams

Twelve workstreams partitioned so that **no two touch the same file**. Each can be handed to an independent engineer agent. Dependencies are noted where ordering matters.

### WS-A — MCP configuration and transport (P0/P1)
**Owns:** `templates/mcp-personal.json`, `templates/mcp-all.json`, `egg/.mcp.json`, `dunk/.mcp.json`, `aerion/.mcp.json`, `aerion/scripts/google-sheets-mcp.sh`, `egg/scripts/exa-mcp.sh`, `dunk/scripts/exa-mcp.sh`, `aerion/.claude-plugin/plugin.json`
**Findings:** F1, F10, F12, F42, F43, F43a, F44
Wrap both templates in `mcpServers` (F1 — highest single-value fix). Add the wrapper to the three plugin configs for consistency (F43a). Replace the ps-visible Exa API key with `userConfig` + header auth (F12). Replace the hardcoded, 503-returning Cloud Run SSE endpoint with a configurable `http` server and set `defaultEnabled: false` (F10). Pin all `@latest` versions (F43). Resolve or document the `chrome-devtools` debug-port assumption (F44).
*Note:* owns `aerion/.claude-plugin/plugin.json` for the `userConfig` block; WS-J must not touch that file — it adds `version` to `egg` and `dunk` only, and coordinates aerion's `version` through WS-A.

### WS-B — dunk Notion wiring (P0)
**Owns:** `dunk/agents/dln-sync.md`, `dunk/agents/dln-syllabus.md`, `dunk/hooks/hooks.json`, `dunk/scripts/validate-ks-markers.sh`
**Findings:** F3, F7 (dunk half), F19 (dunk half), F20 (dunk half), F47 (dunk half), F48
Rewrite all nine `mcp__plugin_Notion_notion__` occurrences to `mcp__plugin_dunk_Notion__` (F3). Remove `permissionMode: dontAsk` from `dln-sync.md:20` and adjust the sync protocol for possible prompts (F7). Verify `query-docs` → `get-library-docs` (F48). Convert the hook to exec form with a `timeout` (F19, F20). **Verify against a live `/mcp` listing before committing** — this workstream is entirely about exact tool-name strings.

### WS-C — egg hook wiring and profile scripts (P0)
**Owns:** `egg/hooks/hooks.json`, `egg/scripts/learner-profile-check.sh`, `egg/scripts/markets-profile-check.sh`, `egg/scripts/learner-profile-load.sh`, `egg/scripts/markets-profile-load.sh`, `egg/scripts/learner-session-snapshot.sh`, `egg/scripts/markets-session-snapshot.sh`
**Findings:** F4, F5, F18, F19, F20, F27, F29, F30
Fix the `.content` → `.message.content` jq bug in both check scripts (F4). Wire `learner-profile-load.sh` into `SessionStart` (F5). **Resolve F29 first** — confirm whether `stop_hook_active` is still delivered and implement an independent re-entrancy guard before F4 makes these hooks able to block. Gate `SessionStart` with `"matcher": "startup|clear"` and stop creating `$HOME` files at install time (F18). Convert all six hook entries to exec form with timeouts (F19, F20). Replace `\s` with `[[:space:]]` or move matching into `jq` (F27). Escape `|` and newlines in ledger interpolation (F30). Add fixture-transcript regression tests.

### WS-D — Python lint hook (P1)
**Owns:** `egg/scripts/python-lint.sh` only
**Findings:** F8, F9, F26
Route the diagnostic block to stderr and `exit 2` so Claude actually receives it (F8). Drop `--unsafe-fixes` (F9). Replace the greedy-regex JSON parse with `jq -r '.tool_input.file_path // empty'` behind a `command -v jq` guard (F26). Reconcile the line-9 comment with the code. *Depends on WS-C for the `egg/hooks/hooks.json` matcher/timeout changes — do not edit that file here.*

### WS-E — aerion command and skill (P0)
**Owns:** `aerion/commands/check-apps.md`, `aerion/skills/job-tracker/SKILL.md`, `aerion/skills/job-tracker/references/email-patterns.md`
**Findings:** F2, F31, F39 (aerion half)
Quote `argument-hint: "[days] (default: 7)"` (F2) and re-run `claude plugin validate ./aerion` to confirm the error clears. Replace the glob `allowed-tools` patterns with `mcp__plugin_aerion_gmail` / `mcp__plugin_aerion_google-sheets` (F31). Add `evaluations/trigger-tests.md` (F39). Remove the personal "hojicha Drive folder" assumption if present in the skill body.

### WS-F — Skill frontmatter normalization (P1)
**Owns:** `egg/skills/doc-generator/SKILL.md`, `egg/skills/tech-blog/SKILL.md`, `egg/skills/ml-paper-writing/SKILL.md`, plus new `evaluations/` dirs for those three and `mlx-dev`
**Findings:** F6, F15, F39 (egg half)
Change `name:` to `doc-generator` and `tech-blog` (F6). Move `trigger_patterns` into `description` / `when_to_use`, and relocate `version` / `author` / `tags` / `dependencies` under `metadata:` (F15). Expand `doc-generator`'s 51-character description. Add `evaluations/trigger-tests.md` to all four (F39). Verify with `claude plugin validate ./egg --strict`.

### WS-G — Teacher-skill state paths (P1/P2)
**Owns:** `egg/skills/global-markets-teacher/**`, `egg/skills/leetcode-teacher/SKILL.md`, `egg/skills/leetcode-teacher/references/**`, `egg/skills/leetcode-teacher/evaluations/**`
**Findings:** F14, F16, F17 (doc half)
Trim the 1554-character `global-markets-teacher` description to under the 1536 cap, targeting ~900 (F14). Fix the `~/.claude/` vs `~/.local/share/claude/` `.processed` rename at `SKILL.md:184` (F16). Migrate all ~35 path references to `${CLAUDE_PLUGIN_DATA}` (F17). **Coordinate F17 with WS-C**, which owns the scripts writing those same paths — land the script change and the doc change together, with a one-time legacy-path migration. *Does not own `egg/skills/leetcode-teacher/CLAUDE.md` — that is WS-K.*

### WS-H — Resume/interview chain rename (P1)
**Owns:** `egg/skills/behavioral-interview-prepper/**`, `egg/skills/technical-interview-roadmap/**`, `egg/skills/resume-analyzer/**`, `egg/skills/resume-tailor/**`, `egg/skills/cover-letter/**`
**Findings:** F11, F25
Replace all 21 `resume-builder` references with the correct producer skill (F11), including the four `trigger-tests.md` rows that encode the wrong expectation. Parameterize the hardcoded `hojicha/` root so the skills work for any installer (F25). *`README.md`'s two `resume-builder` lines belong to WS-K.*

### WS-I — Ralph loop hardening (P2)
**Owns:** `egg/scripts/ralph-loop-template.sh`, `egg/commands/ralph.md`, `egg/skills/ralph/**`
**Findings:** F21, F22, F23, F24, F45
Anchor the task-number extraction (F21). Stop `set -e` from aborting before the completion branch (F22). Add `MAX_ITERATIONS`, repeated-task detection, and non-zero-exit propagation around `claude -p` (F23). Replace `echo -e` with real newlines + `printf '%s'` (F24). Consider consolidating `/ralph` into the `ralph` skill directory (F45).

### WS-J — Release hygiene, versioning, CI (P2/P3)
**Owns:** `egg/.claude-plugin/plugin.json`, `dunk/.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`, `egg/CHANGELOG.md`, new `aerion/CHANGELOG.md`, new `dunk/CHANGELOG.md`, new root `LICENSE`, new `aerion/LICENSE`, new `dunk/LICENSE`, `.gitignore`, new `.github/workflows/validate.yml`
**Findings:** F13, F32, F33, F38, F40, F41
Add `version` to the `egg` and `dunk` manifests and reconcile with `egg/CHANGELOG.md` (F13). Replace the 404 `$schema` with the working schemastore URL (F32). Add `license` / `homepage` / `repository` / `keywords` and marketplace `category` / `tags` (F38). Add CI running per-plugin `claude plugin validate ./<p> --strict`, `jq empty`, `bash -n` / `shellcheck`, and both `pytest` suites (F33) — validate each plugin directory individually, since the marketplace-root call missed F2. Narrow `.gitignore:29` so `docs/reviews/` is tracked and add `.pytest_cache/` (F40, F41). *Does not own `aerion/.claude-plugin/plugin.json` — WS-A does; coordinate aerion's `version` through WS-A.*

### WS-K — Documentation accuracy (P3)
**Owns:** `README.md`, `CLAUDE.md`, `egg/skills/leetcode-teacher/CLAUDE.md`, `egg/agents/code-reviewer.md`, `egg/agents/leetcode-profile-sync.md`, `egg/commands/commit.md`, `egg/commands/status.md`, `egg/commands/debug-ccskill.md`, `egg/commands/evaluate-feature-ccskill.md`
**Findings:** F7 (egg half), F34, F35, F47 (egg half)
Remove `permissionMode: dontAsk` from `egg/agents/leetcode-profile-sync.md:17` (F7). Correct the reference-file count 50 → 69, add `aerion/` and `dunk/` to the structure, inventory all 6 commands / 4 agents / 18 skills, document the four non-lint hook events, clarify that each plugin installs separately, and fix the root-relative path ambiguities (F34). Relocate `egg/skills/leetcode-teacher/CLAUDE.md` out of the plugin (F35). **Run last** — it should describe the post-fix state, so schedule after WS-A through WS-J land.

### WS-L — dunk repository cleanup (P3)
**Owns:** `dunk/references/merge-payload-schema.md`, `dunk/migrations/**`, `dunk/scripts/pyproject.toml`, `dunk/scripts/.python-version`, `dunk/skills/dln/references/merge-protocol.md`, `dunk/skills/dln/**`, `dunk/skills/dln-compress/**`, `dunk/skills/dln-dot/**`, `dunk/skills/dln-linear/**`, `dunk/skills/dln-network/**`
**Findings:** F28, F36, F37, F39 (dunk half), F46
Replace the predictable `/tmp` KS-merge paths with `mktemp -d` plus `trap` cleanup (F28). Relocate the orphaned `merge-payload-schema.md` under `dunk/skills/dln/references/` and link it, or delete it (F36). Move `dunk/migrations/` outside the plugin root so it stops shipping to installers (F37). Add `evaluations/trigger-tests.md` to all five dunk skills (F39). Give `pyproject.toml` a real name/description and relax `requires-python` (F46). *Does not own `dunk/scripts/validate-ks-markers.sh` or `dunk/hooks/hooks.json` — WS-B does.*

---

### Suggested sequencing

1. **Wave 1 (parallel, unblock everything):** WS-A, WS-B, WS-E, WS-F — the P0 fixes plus frontmatter, no interdependencies.
2. **Wave 2 (parallel):** WS-C (resolve F29 before F4), WS-D, WS-H, WS-I.
3. **Wave 3:** WS-G (coordinate the `${CLAUDE_PLUGIN_DATA}` migration with WS-C), WS-J, WS-L.
4. **Wave 4:** WS-K, so the docs describe the finished state.

After every wave, re-run: `claude plugin validate ./egg --strict`, `./aerion --strict`, `./dunk --strict`, both `pytest` suites, and the `templates/*.json` project-load check that surfaced F1.
