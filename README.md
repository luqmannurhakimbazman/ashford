# kapitan-marketplace

A Claude Code plugin marketplace providing commands, agents, skills, hooks, and MCP server configurations.

## Features

| Feature | Description |
|---------|-------------|
| `/commit` | Conventional Commits 1.0.0 compliant commits with automatic diff analysis |
| `/status` | Project status overview (git state, recent changes, pending tasks) |
| `code-reviewer` agent | Code review for quality, security, and performance |
| `mlx-dev` skill | Apple MLX development guide with critical API patterns and gotchas |
| `doc-generator` skill | Automated documentation generation |
| `ml-paper-writing` skill | ML research paper writing assistance |
| MCP servers | Pre-configured git, context7, gitlab, chrome-devtools, and exa integrations |
| Python linting hook | Auto-lints `.py` files on Write/Edit using ruff (Google style + PEP 8) |

## Install

```bash
/plugin marketplace add LuqDaMan/kapitan-marketplace
/plugin install kapitan-claude-plugin@kapitan-marketplace
```

Commands, agents, skills, hooks, and MCP servers are all available immediately.

## MCP Server Templates

The plugin includes MCP servers but you can also copy a template to your project root for per-project config:

| Template | Includes |
|----------|----------|
| `templates/mcp-personal.json` | git, context7, chrome-devtools, exa |
| `templates/mcp-all.json` | Everything above + gitlab |

GitLab requires `GITLAB_PERSONAL_ACCESS_TOKEN` in your environment.

## Structure

```
kapitan-marketplace/
├── kapitan-claude-plugin/
│   ├── .claude-plugin/plugin.json
│   ├── .mcp.json
│   ├── commands/          # /commit, /status
│   ├── agents/            # code-reviewer
│   ├── skills/            # mlx-dev, doc-generator, ml-paper-writing
│   ├── hooks/hooks.json   # Python linting hook
│   └── scripts/           # Helper scripts
└── templates/             # MCP config templates
```

All components are auto-discovered by convention. See `CLAUDE.md` for contributor guidance.

## License

MIT
