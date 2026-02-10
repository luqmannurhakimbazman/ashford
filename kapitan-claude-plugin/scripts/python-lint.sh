#!/bin/bash
# Python Coding Standards Enforcement Hook (Auto-Fix Edition)
# Enforces: Google Python Style Guide, PEP 8
# - Max line length: 100 characters
# - 4 spaces for indentation (no tabs)
# - Naming conventions: snake_case for functions, PascalCase for classes
# - Docstrings required for public modules, classes, and functions (Google format)
#
# This hook auto-fixes what it can and exits with code 2 for manual fixes needed.

# Read tool input from stdin (Claude Code passes tool output as JSON via stdin)
TOOL_INPUT=$(cat)

# Extract file path from tool input JSON
# Handle formats like: {"file_path": "/path/to/file.py"} or {"path": "/path/to/file.py"}
FILE_PATH=""
if echo "$TOOL_INPUT" | grep -q '"file_path"'; then
    FILE_PATH=$(echo "$TOOL_INPUT" | sed -n 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
elif echo "$TOOL_INPUT" | grep -q '"path"'; then
    FILE_PATH=$(echo "$TOOL_INPUT" | sed -n 's/.*"path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
fi

# If no path found in JSON, exit silently
if [[ -z "$FILE_PATH" ]]; then
    exit 0
fi

# Check if it's a Python file
if [[ ! "$FILE_PATH" =~ \.py$ ]]; then
    exit 0
fi

# Check if file exists
if [[ ! -f "$FILE_PATH" ]]; then
    exit 0
fi

# Check if ruff is available
if ! command -v ruff &> /dev/null; then
    # ruff not installed — skip silently rather than blocking Claude
    exit 0
fi

# Ruff rule configuration
# E = pycodestyle errors (PEP 8)
# W = pycodestyle warnings
# F = pyflakes (undefined names, unused imports)
# D = pydocstyle (docstring conventions - Google style)
# N = pep8-naming (naming conventions)
# I = isort (import sorting)
RUFF_RULES="E,W,F,D,N,I"

# Ignore D203 and D213 to enforce Google style:
# - D203 (1 blank line before class docstring) conflicts with D211 (no blank line)
# - D213 (summary on second line) conflicts with D212 (summary on first line)
# We want D211 and D212 for Google style.
RUFF_IGNORE="D203,D213"

LINE_LENGTH=100

# Track if auto-fixes were applied
AUTO_FIXED=0

# =============================================================================
# Step 1: Auto-format with ruff format
# =============================================================================
FORMAT_OUTPUT=$(ruff format "$FILE_PATH" --line-length=$LINE_LENGTH 2>&1) || true

if [[ "$FORMAT_OUTPUT" == *"1 file reformatted"* ]]; then
    AUTO_FIXED=1
fi

# =============================================================================
# Step 2: Auto-fix with ruff check --fix
# =============================================================================
FIX_OUTPUT=$(ruff check "$FILE_PATH" \
    --select=$RUFF_RULES \
    --ignore=$RUFF_IGNORE \
    --line-length=$LINE_LENGTH \
    --fix \
    --unsafe-fixes 2>&1) || true

if [[ "$FIX_OUTPUT" == *"Fixed"* ]]; then
    AUTO_FIXED=1
fi

# =============================================================================
# Step 3: Re-check for remaining unfixable issues
# =============================================================================
REMAINING=$(ruff check "$FILE_PATH" \
    --select=$RUFF_RULES \
    --ignore=$RUFF_IGNORE \
    --line-length=$LINE_LENGTH \
    --output-format=concise 2>&1) || true

# =============================================================================
# Output results and exit with appropriate code
# =============================================================================

# Check if there are remaining issues
if [[ -n "$REMAINING" && "$REMAINING" != *"All checks passed"* ]]; then
    # There are unfixable issues - exit 2 to prompt Claude to fix manually
    echo ""
    echo "=========================================="

    if [[ $AUTO_FIXED -eq 1 ]]; then
        echo "Auto-Fixed Issues Applied"
        echo "=========================================="
        echo ""
        echo "The following were automatically fixed:"
        echo "  - Formatted code to $LINE_LENGTH char line length"
        echo "  - Fixed indentation to 4 spaces"
        echo "  - Sorted imports (isort)"
        echo "  - Fixed trailing whitespace"
        echo "  - Applied other auto-fixable corrections"
        echo ""
        echo "=========================================="
    fi

    echo "Manual Fixes Required"
    echo "=========================================="
    echo "File: $FILE_PATH"
    echo ""
    echo "$REMAINING"
    echo ""
    echo "=========================================="
    echo "ACTION REQUIRED:"
    echo "=========================================="
    echo ""
    echo "Please fix the issues above. Common fixes:"
    echo ""
    echo "For missing docstrings (D100, D101, D102, D103, D104):"
    echo "  - Add Google-format docstrings to modules, classes, and functions"
    echo "  - Example:"
    echo '    def my_function(arg1: str) -> bool:'
    echo '        """Short description ending with a period.'
    echo ''
    echo '        Args:'
    echo '            arg1: Description of arg1.'
    echo ''
    echo '        Returns:'
    echo '            Description of return value.'
    echo '        """'
    echo ""
    echo "For docstring formatting (D200, D205, D212, D400, D415):"
    echo "  - First line should be a short summary ending with a period"
    echo "  - Multi-line docstrings: summary on first line after opening quotes"
    echo "  - Blank line between summary and description"
    echo ""
    echo "For naming conventions (N801, N802, N803, N806):"
    echo "  - Functions/methods: snake_case (e.g., my_function)"
    echo "  - Classes: PascalCase (e.g., MyClass)"
    echo "  - Constants: UPPER_SNAKE_CASE (e.g., MAX_SIZE)"
    echo ""
    echo "Please fix these issues and save the file again."
    echo ""
    exit 1
else
    # All clean - either no issues or all were auto-fixed
    if [[ $AUTO_FIXED -eq 1 ]]; then
        echo "Python lint: Auto-fixed issues in $FILE_PATH"
    else
        echo "Python lint: $FILE_PATH passed all checks"
    fi
    exit 0
fi
