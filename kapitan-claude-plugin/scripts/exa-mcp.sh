#!/bin/bash
# Wrapper script for exa-mcp-server using the hosted Exa MCP endpoint via mcp-remote.
# Claude Code plugin system doesn't interpolate ${ENV_VAR} in .mcp.json, so we use
# this script to expand EXA_API_KEY at runtime into the URL query parameter.
exec npx -y mcp-remote "https://mcp.exa.ai/mcp?exaApiKey=${EXA_API_KEY}"
