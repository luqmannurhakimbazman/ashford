---
description: Scan Gmail for job application updates and sync to Job Tracker sheet
allowed-tools: mcp__workspace__gmail.search, mcp__workspace__gmail.get, mcp__google-sheets__get_sheet_data, mcp__google-sheets__update_cells, mcp__google-sheets__add_rows, mcp__google-sheets__list_spreadsheets, mcp__google-sheets__find_in_spreadsheet
argument-hint: [days] (default: 7)
---

# Check Applications

Scan Gmail for job application status updates and sync them to the Job Tracker Google Sheet.

## Steps

### 1. Determine Time Range

Use `$ARGUMENTS` as the number of days to look back. Default to 7 if not provided.

### 2. Search Gmail

Use `gmail.search` with a query targeting job-related emails:

```
to:lluqmannurhakim@gmail.com (subject:application OR subject:interview OR subject:offer OR subject:assessment OR subject:"online assessment" OR subject:"phone screen" OR subject:"coding challenge" OR subject:"not moving forward" OR subject:congratulations OR from:greenhouse.io OR from:lever.co OR from:ashbyhq.com OR from:myworkdayjobs.com OR from:icims.com OR from:smartrecruiters.com) newer_than:${days}d
```

### 3. Read Matched Emails

For each search result, use `gmail.get` to retrieve the full email content.

### 4. Read Current Sheet

Use `list_spreadsheets` to find "Job Tracker" in the hojicha folder, then `get_sheet_data` to read all current rows.

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
- Use `update_cells` for existing row updates
- Use `add_rows` for new applications
- Do NOT write anything the user rejected or skipped
