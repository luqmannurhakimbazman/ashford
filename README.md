# ashford

A Claude Code marketplace containing three separately installable plugins for development, career, and learning workflows.

## Plugins

| Plugin | Version | Components | MCP servers |
|---|---:|---|---|
| `egg` | 1.1.0 | 5 commands, 2 agents, 12 skills, hooks | context7, git, chrome-devtools, Exa |
| `aerion` | 1.0.0 | 1 command, 1 skill | Gmail, configurable Google Sheets |
| `dunk` | 2.2.0 | 1 agent, 5 skills | context7, Exa |

Installing one plugin does not install the other two.

## Install and configure

In Claude Code:

```text
/plugin marketplace add LuqDaMan/ashford
/plugin install egg@ashford
/plugin install aerion@ashford
/plugin install dunk@ashford
```

Install only the plugins you need. The interactive installer uses user scope by default; choose project or local scope when the plugin should be shared with a repository or limited to one checkout.

Claude Code prompts for each manifest's `userConfig` values when the plugin is enabled:

- **egg:** `exa_api_key` (sensitive; stored in secure storage).
- **dunk:** `exa_api_key` (sensitive; stored in secure storage).
- **aerion:** required `sheets_mcp_url`, the HTTPS endpoint of your Google Sheets MCP deployment. Aerion is disabled by default until configured.

The MCP definitions substitute these values as `${user_config.exa_api_key}` and `${user_config.sheets_mcp_url}`. Gmail uses its hosted HTTP MCP endpoint and may request authentication when first used. Dunk does not require Notion or Obsidian authentication. The local MCP servers require `npx`; egg's git server also requires `uvx`.

Plugin commands and directly invoked skills are namespaced, for example `/egg:commit`, `/egg:leetcode-teacher`, and `/aerion:check-apps`. Skills may also activate automatically from their descriptions.

## Component inventory

### egg

**Commands**

- `/egg:commit` — Conventional Commits workflow with diff analysis.
- `/egg:status` — concise project status.
- `/egg:debug-ccskill` — trace and diagnose a Claude skill bug.
- `/egg:evaluate-feature-ccskill` — evaluate a proposed skill change before implementation.
- `/egg:ralph` — iterative task loop setup.

**Agents**

- `code-reviewer` — code quality, security, and performance review.
- `leetcode-profile-sync` — internal profile and ledger I/O for `leetcode-teacher`.

**Skills**

- `behavioral-interview-prepper`
- `cover-letter`
- `doc-generator`
- `global-markets-teacher`
- `leetcode-teacher`
- `ml-paper-writing`
- `mlx-dev`
- `ralph`
- `resume-analyzer`
- `resume-tailor`
- `tech-blog`
- `technical-interview-roadmap`

### aerion

- Command: `/aerion:check-apps`
- Skill: `job-tracker`

Together they classify job-status email and synchronize application stages through the configured Gmail and Google Sheets MCP servers.

### dunk

**Agent:** `dln-syllabus`

**Skills:** `dln`, `dln-dot`, `dln-linear`, `dln-network`, `dln-compress`

The `dln` skill is a DLN-inspired tutoring scaffold organized as acquire/discriminate, relate/abstract, and predict/revise/compress. Dunk 2.2.0 can prepare bounded local or explicitly consented HTTPS PDF/HTML syllabi, retain original and normalized content in a content-addressed store, seal portable proposals, record complete learner decisions, and cite accepted assertions without treating coverage as mastery. It records structured events locally, projects deterministic JSON and Markdown, and generates Syllabus Intake and teaching Session Receipts. PDF extraction is pinned to `pypdf==6.14.2`; OCR is not provided.

Set `DLN_VAULT_ROOT` to the directory that should contain `domains/`, or pass `--root` to the store CLI. When Claude Code exposes `${CLAUDE_PLUGIN_DATA}` to the parent Bash environment, Dunk otherwise defaults to `${CLAUDE_PLUGIN_DATA}/dln-vault`; if it does not, the CLI stops with an actionable configuration error rather than choosing an implicit home directory.

Open the configured root (or its `domains/` directory) directly as an Obsidian vault—no Obsidian plugin, MCP server, or database is required. See [`dunk/LOCAL_STORAGE.md`](dunk/LOCAL_STORAGE.md) for canonical files, backup/recovery, and legacy migration.

## Resume and interview workflow

The former `resume-builder` workflow is now an explicit chain:

1. `resume-analyzer` compares the JD and candidate material and writes `<application-dir>/notes.md`.
2. `resume-tailor` consumes `notes.md` and writes `<application-dir>/resume.tex`.
3. `behavioral-interview-prepper` consumes `notes.md`, `resume.tex`, and candidate context to write `behavioral-prep.md`.
4. `cover-letter` can consume `notes.md` to write `cover-letter.md`.

`technical-interview-roadmap` is independent: it accepts a JD directly and does not require the resume chain. Resume workspaces default to `${CLAUDE_PROJECT_DIR}/resumes` when available; explicit user paths take precedence.

## Hooks and persistent data

Egg registers:

- `PostToolUse` on `Write|Edit`: format and lint Python with Ruff, then report remaining diagnostics.
- `SessionStart` on `startup|clear`: load and repair LeetCode and markets learner state when it exists.
- Two `PreCompact` hooks: snapshot active LeetCode and markets sessions.
- Two `Stop` hooks: require profile/ledger write-back after substantive teaching sessions.

Dunk has no persistence hook. Its canonical per-domain data is user-owned `profile.yaml`, append-only `events.jsonl`, retained `sources/sha256/` bytes, and normalized `prepared/sha256/` documents; generated `state.json`, `dashboard.md`, `syllabus/<source-version-id>.md`, and `sessions/<session-id>.md` can be rebuilt at any time.

Persistent learner files must live outside the replaceable `${CLAUDE_PLUGIN_ROOT}` install cache. Dunk uses an explicit `DLN_VAULT_ROOT` when configured and can use `${CLAUDE_PLUGIN_DATA}/dln-vault` when that variable is available to the invoking Bash process. Egg performs a one-time, non-overwriting migration of its LeetCode and markets files from the legacy `~/.local/share/claude/` path.

## MCP templates

Project-level examples are available at:

| Template | Servers |
|---|---|
| `templates/mcp-personal.json` | git, context7, chrome-devtools, Exa |
| `templates/mcp-all.json` | the personal set plus GitLab |

Copy the selected file to `<project>/.mcp.json` and customize it for that project. `mcp-all.json` reads `GITLAB_PERSONAL_ACCESS_TOKEN` from the environment. The templates are separate from plugin `userConfig`; configure authentication required by any copied HTTP endpoint.

## Repository layout

```text
ashford/
├── .claude-plugin/marketplace.json
├── egg/       # development, writing, interview, and teacher workflows
├── aerion/    # job application tracking
├── dunk/      # DLN learning system
├── templates/ # project-level MCP examples
├── tools/     # repository-only migration tooling
└── docs/      # maintainer guides plus plans, specs, investigations, reviews, and evidence
```

Each plugin keeps its manifest at `<plugin>/.claude-plugin/plugin.json`; discoverable `commands/`, `agents/`, `skills/`, `hooks/`, scripts, and `.mcp.json` live at the plugin root.

## Validation

CI validates the marketplace and every plugin separately, checks JSON and shell files, runs ShellCheck, and executes Dunk's complete Python test suite plus the repository migration tests. Useful local commands are:

```bash
claude plugin validate . --strict
claude plugin validate ./egg --strict
claude plugin validate ./aerion --strict
claude plugin validate ./dunk --strict

uv run --project dunk/scripts --python 3.10 --frozen pytest dunk/scripts/tests
uv run --project dunk/scripts --python 3.10 --frozen pytest tools/dunk-migrations/test_migrate_docker.py
```

## License

MIT
