---
description: Scan Gmail for job application updates and sync to Job Tracker sheet
argument-hint: [days] (default: 7)
allowed-tools: mcp__plugin_aerion_gmail__*, mcp__plugin_aerion_google-sheets__*, mcp__*gmail*, mcp__*google-sheets*, mcp__*google_sheets*
---

# Check Applications

Scan Gmail for job application status updates and sync them to the Job Tracker Google Sheet.

## Prerequisites

If Gmail or Google Sheets MCP tools are not available, stop immediately and tell the user: "MCP servers are not connected. Please add Gmail and Google Sheets connectors."

**Tool discovery:** Tool name prefixes vary by environment. On local Claude Code they use `mcp__plugin_aerion_gmail__` and `mcp__plugin_aerion_google-sheets__`. On Cowork they use different prefixes. Look up available tools by **function name** (e.g., `gmail_search_messages`, `list_spreadsheets`), not by prefix.

## Steps

### 1. Determine Time Range

Use `$ARGUMENTS` as the number of days to look back. Default to 7 if not provided.

### 2. Search Gmail

Use the Gmail search tool (`gmail_search_messages` or `Search Gmail Emails`) with this query:

```
to:lluqmannurhakim@gmail.com (subject:application OR subject:interview OR subject:offer OR subject:assessment OR subject:"online assessment" OR subject:"phone screen" OR subject:"coding challenge" OR subject:"not moving forward" OR subject:congratulations OR subject:"candidate reference" OR subject:"your submission" OR from:greenhouse.io OR from:lever.co OR from:ashbyhq.com OR from:myworkdayjobs.com OR from:myworkday.com OR from:icims.com OR from:smartrecruiters.com OR from:jobvite.com OR from:successfactors.com OR from:successfactors.eu OR from:taleo.net OR from:hire.jazz.co OR from:breezy.hr OR from:applytojob.com OR from:hackerrankforwork.com OR from:codesignal.com OR from:codility.com OR from:brassring.com OR from:avature.net OR from:phenom.com) newer_than:${days}d
```

### 3. Read Matched Emails

For each search result, use the Gmail read tool (`gmail_read_message` or `Read Gmail Email`) to retrieve the full email content. Read emails in parallel where possible.

### 4. Read Current Sheet

1. Use `list_spreadsheets` to find "Job Tracker" in the hojicha folder.
2. Use `list_sheets` with the spreadsheet ID to discover the sheet tab name (it's `job-tracker`).
3. Use `get_sheet_data` with the spreadsheet ID and sheet name `job-tracker` to read all current rows.

### 5. Classify and Match

Invoke the `job-tracker` skill knowledge to:
- Classify each email into a Stage
- Extract Company, Role, Date, and Note
- Match against existing sheet rows by Company + Role
- Apply stage progression rules (forward-only, except Rejected/Ghosted)

### 6. Present Summary

Show the user:
- **Updates** — existing rows with stage changes (old → new)
- **New applications** — rows to add (Stage = Applied)
- **Ambiguous** — emails that need user input

### 7. Confirm and Write

Ask the user to confirm. On confirmation:
- Use `update_cells` for existing row updates (specify sheet name `job-tracker`)
- Use `add_rows` for new applications (specify sheet name `job-tracker`)
- Do NOT write anything the user rejected or skipped
