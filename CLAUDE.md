# CLAUDE.md

Contributor guidance for the ashford Claude Code plugin marketplace.

## Project overview

Ashford contains three independent plugins:

| Plugin | Version | Purpose |
|---|---:|---|
| `egg/` | 1.1.0 | development, writing, interview, and teacher workflows |
| `aerion/` | 1.0.0 | Gmail-to-Google-Sheets job application tracking |
| `dunk/` | 1.0.0 | Dot–Linear–Network learning with Notion persistence |

The marketplace registry is `.claude-plugin/marketplace.json`. Each plugin has its own `<plugin>/.claude-plugin/plugin.json` and must be installed, configured, enabled, versioned, and validated separately.

There is no application compilation step. Plugin components are Markdown, JSON, and shell files discovered by Claude Code. Dunk's Python validation tooling is managed with `uv` from `dunk/scripts/pyproject.toml` and `dunk/scripts/uv.lock`.

## Plugin layout

| Component | Locations | Format |
|---|---|---|
| Commands | `egg/commands/*.md`, `aerion/commands/*.md` | Markdown with YAML frontmatter |
| Agents | `egg/agents/*.md`, `dunk/agents/*.md` | Markdown with YAML frontmatter |
| Skills | `egg/skills/*/SKILL.md`, `aerion/skills/*/SKILL.md`, `dunk/skills/*/SKILL.md` | skill Markdown |
| Hooks | `egg/hooks/hooks.json`, `dunk/hooks/hooks.json` | hook JSON |
| MCP | `egg/.mcp.json`, `aerion/.mcp.json`, `dunk/.mcp.json` | `mcpServers` JSON |
| Executed helpers | `egg/scripts/`, `dunk/scripts/` | shell/Python |
| Project MCP examples | `templates/mcp-personal.json`, `templates/mcp-all.json` | `mcpServers` JSON |

All discoverable component directories are siblings of `.claude-plugin/` at the relevant plugin root. Do not put components inside `.claude-plugin/`. A `CLAUDE.md` inside a plugin is not loaded as plugin context; use `SKILL.md`, agents, hooks, or referenced skill files instead.

## Current inventory

There are 6 commands, 4 agents, and 18 skills across the marketplace.

### Commands

- Egg: `commit`, `status`, `debug-ccskill`, `evaluate-feature-ccskill`, `ralph`
- Aerion: `check-apps`

Plugin commands are namespaced at runtime, such as `/egg:commit` and `/aerion:check-apps`.

### Agents

- Egg: `code-reviewer`, `leetcode-profile-sync`
- Dunk: `dln-sync`, `dln-syllabus`

Plugin agents use supported plugin-agent frontmatter only. In particular, do not add `color`, `permissionMode`, `hooks`, or `mcpServers` to plugin-shipped agents.

### Skills

- Egg: `behavioral-interview-prepper`, `cover-letter`, `doc-generator`, `global-markets-teacher`, `leetcode-teacher`, `ml-paper-writing`, `mlx-dev`, `ralph`, `resume-analyzer`, `resume-tailor`, `tech-blog`, `technical-interview-roadmap`
- Aerion: `job-tracker`
- Dunk: `dln`, `dln-compress`, `dln-dot`, `dln-linear`, `dln-network`

The LeetCode teacher has 69 files under `egg/skills/leetcode-teacher/references/`; the markets teacher has 28 under `egg/skills/global-markets-teacher/references/`. LeetCode-specific maintenance guidance is in `docs/leetcode-teacher-development.md`.

## Install and configuration model

Users first add the marketplace, then install only the plugins they need:

```text
/plugin marketplace add LuqDaMan/ashford
/plugin install egg@ashford
/plugin install aerion@ashford
/plugin install dunk@ashford
```

Installing `egg` does not deliver Aerion or Dunk. Claude Code prompts for manifest `userConfig` when a plugin is enabled:

- Egg and Dunk each declare sensitive `exa_api_key`; it is substituted into the Exa HTTP MCP `x-api-key` header.
- Aerion declares required `sheets_mcp_url`; it is substituted into the Google Sheets HTTP MCP URL. Aerion has `defaultEnabled: false`.

Sensitive values are stored by Claude Code in secure storage rather than repository settings. Do not reintroduce API keys in URLs, argv, tracked files, or shell wrappers.

## MCP servers

- `egg/.mcp.json`: context7 `@4.0.2`, git `mcp-server-git==2026.7.10`, chrome-devtools `@1.7.0`, and hosted Exa HTTP.
- `aerion/.mcp.json`: hosted Gmail HTTP and user-configured Google Sheets HTTP.
- `dunk/.mcp.json`: context7 `@4.0.2`, hosted Exa HTTP, and hosted Notion HTTP.
- `templates/mcp-personal.json`: git, context7, chrome-devtools, Exa.
- `templates/mcp-all.json`: the personal set plus GitLab `@2025.4.25`.

Plugin-scoped MCP tools use `mcp__plugin_<plugin>_<server>__<tool>`. Preserve exact server spelling, including Dunk's `Notion` key. The project templates are copied to `<project>/.mcp.json`; they are not installed or configured through plugin `userConfig`.

## Hooks

### Egg

- `PostToolUse` matching `Write|Edit` runs `egg/scripts/python-lint.sh` with a 10-second timeout. It formats and safely fixes Python, then sends remaining diagnostics on stderr with exit 2. It is a no-op if Ruff or jq is unavailable.
- `SessionStart` matching `startup|clear` runs the LeetCode and markets loaders with 15-second timeouts.
- Two `PreCompact` entries snapshot active LeetCode and markets sessions.
- Two `Stop` entries verify profile and ledger write-back after substantive teaching sessions and use per-session loop guards.

### Dunk

- `PreToolUse` for `mcp__plugin_dunk_Notion__notion-update-page` validates knowledge-state markers before the Notion write.

Hook commands use exec form with explicit `command`, `args`, and `timeout`. Reference scripts as `${CLAUDE_PLUGIN_ROOT}/scripts/<file>`; never assume the current working directory is the plugin root.

## Persistent data and paths

Use paths according to purpose:

- `${CLAUDE_PLUGIN_ROOT}`: read-only installed plugin content and executable helpers. It is replaced on update.
- `${CLAUDE_PLUGIN_DATA}`: plugin-scoped persistent learner profiles, ledgers, session snapshots, and stop guards.
- `${CLAUDE_PROJECT_DIR}`: the user's project and default resume workspace.
- Repository-relative paths such as `egg/skills/...`: maintainer documentation and validation run from the ashford repository root.

Egg's loaders perform a one-time, non-overwriting migration from `~/.local/share/claude/` to `${CLAUDE_PLUGIN_DATA}` for LeetCode and markets profiles, ledgers, and session-state files. Do not add new runtime writes to the legacy directory or to the plugin cache.

## Resume and interview chain

There is no `resume-builder` skill. The migrated workflow is:

1. `resume-analyzer` writes `<application-dir>/notes.md`.
2. `resume-tailor` requires those notes and writes `<application-dir>/resume.tex`.
3. `behavioral-interview-prepper` requires the analyzer notes, tailored resume, and candidate context, then writes `behavioral-prep.md`.
4. `cover-letter` uses analyzer notes when available and writes `cover-letter.md`.

`technical-interview-roadmap` consumes a JD directly and is independent of this chain. Default resume paths are under `${CLAUDE_PROJECT_DIR}/resumes`; explicit user-provided workspaces and files always override the convention.

## Frontmatter conventions

- Commands: use supported fields such as `description`, `argument-hint`, and `allowed-tools`. Existing `commands/` files are the legacy flat-skill layout; prefer `skills/<name>/SKILL.md` for new functionality.
- Agents: use documented plugin-agent fields such as `name`, `description`, `model`, `effort`, `maxTurns`, `tools`, `disallowedTools`, `skills`, `memory`, `background`, and `isolation`.
- Skills: keep `name` kebab-case and equal to the directory name; make `description` sufficient for activation routing.

Use `$ARGUMENTS`, `$1`, and `$2` for command arguments. Use absolute plugin paths via `${CLAUDE_PLUGIN_ROOT}` when a hook or MCP process reads shipped files.

## Validation

The CI workflow in `.github/workflows/validate.yml` validates JSON, shell syntax and ShellCheck, the marketplace, every plugin independently, and both Python suites.

Run relevant checks locally:

```bash
claude plugin validate . --strict
claude plugin validate ./egg --strict
claude plugin validate ./aerion --strict
claude plugin validate ./dunk --strict

find . -type f -name '*.json' -not -path './.git/*' -print0 | xargs -0 -n1 jq empty
find . -type f -name '*.sh' -not -path './.git/*' -print0 | xargs -0 -n1 bash -n
find . -type f -name '*.sh' -not -path './.git/*' -print0 | xargs -0 shellcheck --severity=warning

uv run --project dunk/scripts --python 3.10 --frozen pytest dunk/scripts/tests/test_ks_merge.py
uv run --project dunk/scripts --python 3.10 --frozen pytest tools/dunk-migrations/test_migrate_docker.py
```

For link or routing changes, also run the integrity commands in the affected skill's `evaluations/trigger-tests.md`.
