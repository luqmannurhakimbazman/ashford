#!/bin/bash
# Wrapper script for mcp-google-sheets.
# Claude Code plugin system doesn't interpolate ${ENV_VAR} in .mcp.json, so we use
# this script to expand env vars at runtime.
export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_SERVICE_ACCOUNT_PATH}"
export DRIVE_FOLDER_ID="${HOJICHA_FOLDER_ID}"
exec uvx mcp-google-sheets@latest
