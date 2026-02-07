# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**kapitan-marketplace** is a Claude Code plugin marketplace that provides slash commands, agents, skills, hooks, and MCP server configurations. It's installed via the Claude Code marketplace system (`/plugin marketplace add` + `/plugin install`) and all components are auto-discovered at runtime.

- **Version:** 1.2.0
- **Author:** Luqman Nurhakim
- **No build system** — all components are Markdown/JSON files auto-discovered by Claude Code. No compilation, bundling, or package management needed for the plugin itself.

## Architecture

The plugin lives in `kapitan-claude-plugin/` which contains the `.claude-plugin/plugin.json` manifest. Claude Code discovers all components by convention:

| Component | Location | Format |
|-----------|----------|--------|
| Commands | `kapitan-claude-plugin/commands/*.md` | Markdown with YAML frontmatter |
| Agents | `kapitan-claude-plugin/agents/*.md` | Markdown with YAML frontmatter |
| Skills | `kapitan-claude-plugin/skills/*/SKILL.md` | Markdown with YAML frontmatter |
| MCP Servers | `kapitan-claude-plugin/.mcp.json` | JSON config |
| Hooks | `kapitan-claude-plugin/hooks/hooks.json` | JSON config |
| Helper Scripts | `kapitan-claude-plugin/scripts/` | Shell scripts |

The top-level `.claude-plugin/marketplace.json` is the marketplace registry pointing to the plugin directory. The `templates/` directory has ready-to-copy `.mcp.json` configurations (personal, all).

## Current Components

- **`/commit`** — Conventional Commits 1.0.0 compliant commit creation with diff analysis. Explicitly forbids AI attribution footers.
- **`/status`** — Project status overview (git state, recent changes, pending tasks).
- **`code-reviewer` agent** — Code review for quality, security, and performance.
- **`mlx-dev` skill** — Apple MLX development guide with references in `skills/mlx-dev/references/`.
- **`doc-generator` skill** — Automated documentation generation.
- **`ml-paper-writing` skill** — ML research paper writing assistance.

## Python Linting Hook

A `PostToolUse` hook (`hooks/hooks.json`) runs `scripts/python-lint.sh` on every Write/Edit of `.py` files. It uses `ruff` with these settings:

- **Rules:** E, W, F, D, N, I (pycodestyle, pyflakes, pydocstyle, pep8-naming, isort)
- **Ignored:** D203, D213 (to enforce Google-style docstrings — D211 + D212)
- **Line length:** 100 characters
- **Style:** Google Python Style Guide + PEP 8
- **Behavior:** Auto-formats and auto-fixes first, then reports unfixable issues (exit 2)
- **Requires:** `ruff` installed (`pip install ruff` or `uv pip install ruff`)

## Frontmatter Conventions

**Commands** use: `description`, `allowed-tools`, `argument-hint`, `model` (sonnet/opus/haiku).

**Agents** use: `name`, `description` (with `<example>` blocks for trigger conditions), `model` (inherit/sonnet/opus/haiku), `color`, `tools` (array).

**Skills** use: `name`, `description` (Claude uses this to decide contextual activation).

## Adding New Components

All components are auto-discovered — create files in the right directory and they work immediately. Use `$ARGUMENTS`, `$1`, `$2` for command arguments, `@path/to/file` for file inclusion, and `!`command`` for inline bash in commands. Use `${CLAUDE_PLUGIN_ROOT}` for plugin-relative paths in hooks and MCP configs. Use `${ENV_VAR}` for environment variable injection in MCP server definitions.

## MCP Servers

Defined in `kapitan-claude-plugin/.mcp.json`: context7 (npx), gitlab (npx, needs `GITLAB_PERSONAL_ACCESS_TOKEN`), git (uvx), chrome-devtools (npx), exa (npx via mcp-remote). Two templates in `templates/`: `mcp-personal.json` (everything except gitlab) and `mcp-all.json` (everything). Projects copy a template to their root `.mcp.json` and customize.

## Installation Model

**Primary (Marketplace):** Users run `/plugin marketplace add LuqDaMan/kapitan-marketplace` then `/plugin install kapitan-claude-plugin@kapitan-marketplace`. This delivers all components — commands, agents, skills, hooks, and MCP servers.

**Legacy (Submodule):** Adding the repo as a git submodule at `.claude-plugins/kapitan` only delivers MCP servers. Commands, agents, skills, and hooks are **not** discovered via submodules. Use the marketplace approach for full functionality.
