---
name: job-tracker
description: This skill should be used when the user runs /check-apps or asks to check job applications, scan Gmail for application updates, update the job tracker, or sync application status. It provides email classification rules, entity extraction logic, and sheet update constraints for the aerion job application tracking workflow.
---

# Job Application Tracker

Track job application status by scanning Gmail and updating the Google Sheets "Job Tracker" in the hojicha Drive folder.

## Target Email

lluqmannurhakim@gmail.com

## Sheet Schema

The Job Tracker sheet has these columns (in order):

| Column | Type | Values |
|--------|------|--------|
| A: Company | Text | Company name |
| B: Role | Text | Role/position title |
| C: Stage | Dropdown | Applied, Behavioral Interview, Online Assessment, Onsite Interview, Rejected, Ghosted, Offered |
| D: Last Contact Date | Date | YYYY-MM-DD format |
| E: Notes | Text | Append-only summary notes |

## Email Classification

Classify each email into a Stage using signals from the subject line, body, and sender. Refer to `references/email-patterns.md` for the full pattern catalog.

**Stage mapping priority:** If an email matches multiple stages, use the most advanced stage. For example, an email mentioning both "application received" and "online assessment link" maps to Online Assessment.

## Stage Progression Rules

Stages have a natural forward order:

```
Applied → Online Assessment → Behavioral Interview → Onsite Interview → Offered
```

**Rules:**
1. A stage update can only move **forward** in the progression above.
2. **Rejected** and **Ghosted** can override any stage (they are terminal states).
3. Never regress a stage (e.g., if currently at Onsite Interview, ignore an old "application received" email).
4. **Ghosted** is only suggested when there is no email activity for >30 days since the Last Contact Date. Always ask the user for confirmation before marking Ghosted.

## Entity Extraction

For each email, extract:
- **Company** — the hiring company name
- **Role** — the position title
- **Stage** — inferred from email classification
- **Date** — the email's sent date (for Last Contact Date)
- **Note** — a one-line summary of the email content (e.g., "Received OA link via HackerRank", "Rejection — position filled")

Use the fallback order in `references/email-patterns.md` for entity extraction when company/role are not obvious.

## Matching Rules

Match emails to existing sheet rows by **Company + Role** (case-insensitive, fuzzy). Examples:
- "Citadel Securities" matches "Citadel" — same company
- "Software Engineer, Infrastructure" matches "SWE Infra" — use judgment on abbreviations

If no match is found and the email indicates a new application, create a new row with Stage = Applied.

## Update Rules

1. **Existing row, stage change:** Update Stage column. Update Last Contact Date. Append to Notes.
2. **Existing row, no stage change:** Update Last Contact Date. Append to Notes.
3. **New application:** Add row with Company, Role, Stage = Applied, Last Contact Date = email date, Notes = summary.
4. **Ambiguous email:** Cannot determine company, role, or stage — flag for user with email subject, sender, and a snippet. Never skip silently.

## Notes Column

- **Append, never overwrite.** Add new notes on a new line prefixed with the date.
- Format: `YYYY-MM-DD: <summary>`
- Example: `2026-02-26: Received OA link via CodeSignal`

## Output Format

After scanning, present results to the user in this format:

### Updates (existing rows)

```
| Company | Role | Old Stage | New Stage | Note |
|---------|------|-----------|-----------|------|
| Citadel | Quant Researcher | Applied | Online Assessment | Received OA link |
```

### New Applications

```
| Company | Role | Note |
|---------|------|------|
| Jane Street | SWE Intern | Application confirmation email |
```

### Ambiguous (needs your input)

```
- Email from recruiter@google.com (subject: "Quick question") — cannot determine role. Skip or provide details?
```

### No Changes

If no relevant emails found, say so.

**After presenting:** Ask the user to confirm before writing any changes to the sheet.
