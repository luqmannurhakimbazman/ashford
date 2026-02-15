#!/bin/bash
# Wrapper script for exa-mcp-server that passes the EXA_API_KEY from the environment
exec npx -y exa-mcp-server "exaApiKey=${EXA_API_KEY}"
