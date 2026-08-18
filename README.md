# ashford

A Claude Code marketplace containing three separately installable plugins for development, career, and learning workflows.

## Plugins

| Plugin | Version | Components | MCP servers |
|---|---:|---|---|
| `egg` | 1.1.0 | 5 commands, 2 agents, 12 skills, hooks | context7, git, chrome-devtools, Exa |
| `aerion` | 1.0.0 | 1 command, 1 skill | Gmail, configurable Google Sheets |
| `dunk` | 1.0.0 | 2 agents, 5 skills, hooks | context7, Exa, Notion |

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

The MCP definitions substitute these values as `${user_config.exa_api_key}` and `${user_config.sheets_mcp_url}`. Gmail and Notion use their hosted HTTP MCP endpoints and may request their own authentication when first used. The local MCP servers require `npx`; egg's git server also requires `uvx`.

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

**Agents:** `dln-sync`, `dln-syllabus`

**Skills:** `dln`, `dln-dot`, `dln-linear`, `dln-network`, `dln-compress`

The `dln` skill orchestrates the Dot–Linear–Network learning phases; its internal agents handle curriculum research and Notion ledger synchronization.

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

Dunk registers a `PreToolUse` hook that validates knowledge-state markers before Notion page updates.

Persistent learner files live under the plugin-scoped `${CLAUDE_PLUGIN_DATA}` directory, not the replaceable `${CLAUDE_PLUGIN_ROOT}` install cache. Egg performs a one-time, non-overwriting migration of its LeetCode and markets files from the legacy `~/.local/share/claude/` path.

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
└── docs/      # plans, reviews, and maintainer guides
```

Each plugin keeps its manifest at `<plugin>/.claude-plugin/plugin.json`; discoverable `commands/`, `agents/`, `skills/`, `hooks/`, scripts, and `.mcp.json` live at the plugin root.

## Validation

CI validates the marketplace and every plugin separately, checks JSON and shell files, runs ShellCheck, and runs the Python tests. The local validation commands are in [CLAUDE.md](CLAUDE.md#validation).

## License

MIT
