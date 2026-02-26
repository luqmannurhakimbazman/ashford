---
description: Scan Gmail for job application updates and sync to Job Tracker sheet
argument-hint: [days] (default: 7)
allowed-tools: mcp__plugin_aerion_gmail__*, mcp__plugin_aerion_google-sheets__*
---

# Check Applications

Scan Gmail for job application status updates and sync them to the Job Tracker Google Sheet.

## Prerequisites

The Gmail MCP requires OAuth authentication via browser. If this is your first time, run `/check-apps` in an **interactive session** (not Cowork) to complete the OAuth flow. Once authenticated, subsequent sessions reuse the token.

If the Gmail or Google Sheets MCP tools are not available, stop immediately and tell the user: "MCP servers are not connected. Run this command in an interactive session first to complete OAuth."

## Steps

### 1. Determine Time Range

Use `$ARGUMENTS` as the number of days to look back. Default to 7 if not provided.

### 2. Search Gmail

Use `mcp__plugin_aerion_gmail__gmail_search_messages` with this query:

```
to:lluqmannurhakim@gmail.com (subject:application OR subject:interview OR subject:offer OR subject:assessment OR subject:"online assessment" OR subject:"phone screen" OR subject:"coding challenge" OR subject:"not moving forward" OR subject:congratulations OR from:greenhouse.io OR from:lever.co OR from:ashbyhq.com OR from:myworkdayjobs.com OR from:myworkday.com OR from:icims.com OR from:smartrecruiters.com OR from:jobvite.com OR from:successfactors.com OR from:successfactors.eu OR from:taleo.net OR from:hire.jazz.co OR from:breezy.hr OR from:applytojob.com OR from:hackerrankforwork.com OR from:codesignal.com OR from:codility.com) newer_than:${days}d
```

### 3. Read Matched Emails

For each search result, use `mcp__plugin_aerion_gmail__gmail_read_message` to retrieve the full email content. Read emails in parallel where possible.

### 4. Read Current Sheet

1. Use `mcp__plugin_aerion_google-sheets__list_spreadsheets` to find "Job Tracker" in the hojicha folder.
2. Use `mcp__plugin_aerion_google-sheets__list_sheets` with the spreadsheet ID to discover the sheet tab name (it's `job-tracker`).
3. Use `mcp__plugin_aerion_google-sheets__get_sheet_data` with the spreadsheet ID and sheet name `job-tracker` to read all current rows.

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
- Use `mcp__plugin_aerion_google-sheets__update_cells` for existing row updates (specify sheet name `job-tracker`)
- Use `mcp__plugin_aerion_google-sheets__add_rows` for new applications (specify sheet name `job-tracker`)
- Do NOT write anything the user rejected or skipped
