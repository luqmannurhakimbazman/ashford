# Merge Payload Schema

JSON contract between the phase skill normalizer (running in the parent conversation) and `ks-merge.py`. The phase skill produces this per the merge protocol (`dln/references/merge-protocol.md`); the script consumes it. All fields are optional — a dispatch only includes what changed.

## Schema

```json
{
  "mastery_updates": [
    {
      "table": "concepts | chains | factors",
      "name": "row identifier — exact match against Concept/Chain/Factor column",
      "status": "not-mastered | partial | mastered",
      "evidence": "string to append to Evidence column (comma-separated)",
      "last_tested": "YYYY-MM-DD",
      "syllabus_topic": "optional — concepts table only"
    }
  ],
  "weakness_queue": [
    {
      "priority": 1,
      "item": "string",
      "type": "concept | chain | factor",
      "phase": "Dot | Linear | Network",
      "severity": "string",
      "source": "string",
      "added": "YYYY-MM-DD"
    }
  ],
  "syllabus_updates": [
    {
      "topic": "string — must match existing syllabus line",
      "status": "checked | unchecked"
    }
  ],
  "section_rewrites": {
    "compressed_model": "COMPLETE replacement text for ## Compressed Model",
    "open_questions": "COMPLETE replacement text for ## Open Questions",
    "interleave_pool": "COMPLETE replacement text for ## Interleave Pool"
  },
  "subsection_rewrites": {
    "calibration_trend": "COMPLETE replacement text for ### Calibration Trend"
  },
  "section_appends": {
    "calibration_concept": "pipe-delimited table row (no header) to append to ### Concept-Level Confidence",
    "calibration_gate": "pipe-delimited table row (no header) to append to ### Gate Predictions",
    "load_session_history": "pipe-delimited table row (no header) to append to ### Session History"
  },
  "load_baseline": {
    "working_batch_size": "number — observed working batch size",
    "hint_tolerance": "string — e.g. 'low (needs <=1 hint per concept)'",
    "recovery_pattern": "string — e.g. 'responds well to different analogies'"
  },
  "engagement": {
    "momentum": "positive | neutral | negative",
    "consecutive_struggles": 0,
    "last_celebration": "string",
    "notes": "string"
  }
}
```

## Rules

- **`section_rewrites`** targets `##`-level headers. Values must be COMPLETE replacement content — not deltas. The script replaces everything between the header and the next `##` header. Partial content causes data loss.
- **`subsection_rewrites`** targets `###`-level headers. Same complete-replacement semantics, but replacement ends at the next `###` or `##` header.
- **`weakness_queue`** is a full rewrite — the script replaces the entire queue, not a merge.
- **`syllabus_updates`** is part of the merge JSON payload. The phase skill includes it when concept mastery changes affect syllabus topic coverage.
- **`mastery_updates`** never delete rows. Existing rows are upserted; new rows are appended.

## Field Reference

| Field | Target Section | Operation |
|-------|---------------|-----------|
| `mastery_updates` | `## Concepts`, `## Chains`, `## Factors` | Row upsert |
| `weakness_queue` | `## Weakness Queue` | Full rewrite |
| `syllabus_updates` | `## Syllabus` | Checkbox toggle |
| `section_rewrites.compressed_model` | `## Compressed Model` | Full replace |
| `section_rewrites.open_questions` | `## Open Questions` | Full replace |
| `section_rewrites.interleave_pool` | `## Interleave Pool` | Full replace |
| `subsection_rewrites.calibration_trend` | `### Calibration Trend` | Full replace |
| `section_appends.calibration_concept` | `### Concept-Level Confidence` | Row append |
| `section_appends.calibration_gate` | `### Gate Predictions` | Row append |
| `section_appends.load_session_history` | `### Session History` | Row append |
| `load_baseline` | `### Baseline` | Key-value update |
| `engagement` | `## Engagement Signals` | Key-value update |
