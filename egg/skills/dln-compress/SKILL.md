---
name: dln-compress
description: >
  Internal format specification for the DLN system. Only relevant when preloaded
  by the dln-sync agent via the skills frontmatter field. Never activated by user
  prompts. Defines the re-anchor payload compression template that dln-sync uses
  to convert raw Notion page-body read-backs into compact structured summaries
  for DLN teaching skills.
---

# DLN Compress — Re-anchor Payload Format

This skill defines the compression format for converting raw Notion page content into a re-anchor payload. Follow these rules exactly when compressing read-back data.

## Output Format

After reading back Notion page content, produce EXACTLY this structure. Be concise — compress lists to key terms, not full sentences. The teaching skill needs just enough to re-orient, not the full raw content.

~~~
## Re-anchor
### Knowledge State
- Concepts: [comma-separated list of concept names]
- Chains: [comma-separated list of chain descriptions, abbreviated]
- Factors: [comma-separated list of factor names]
- Compressed Model: [current model text, verbatim if short, summarized if >3 sentences]
- Open Questions: [comma-separated list]
### Current Session
- Plan: [one-line summary of session plan]
- Progress: [bullet list of completed items with outcomes]
- Adjustments: [any plan changes, or "none"]
### Status
- Write: [success/failed]
- Failed writes: [list of what failed, or "none"]
- Next on plan: [what the teaching skill should do next]
~~~

## Compression Rules

1. **Be concise.** The whole payload should be under 30 lines. Compress aggressively.
2. **Preserve precision.** Don't lose information that the teaching skill needs to make decisions (comprehension check outcomes, specific concepts that struggled, etc.).
3. **"Next on plan"** — derive this from the session plan minus completed items. State the immediate next action.
4. **If Knowledge State sections are empty**, say "none" — don't omit the field.
5. **Verbatim model** — for Compressed Model, if it's 3 sentences or fewer, include verbatim. If longer, summarize.
