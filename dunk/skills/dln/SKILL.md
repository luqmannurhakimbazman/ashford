---
name: dln
description: >
  Use when the user wants structured learning in a new or existing domain, says
  "dln", "dln list", "dln reset [domain]", "learn [domain]", "teach me
  [domain] from zero", "continue learning [domain]", "resume [domain]",
  "review [domain]", "dln exam [domain] by [date]", "dln exam [domain]
  status", "dln mock [domain]", "dln cram [N]d [domain]", or refers to the
  Dot-Linear-Network learning workflow. Orchestrates the existing dln-dot,
  dln-linear, and dln-network skill IDs using an authoritative local event store,
  structured evidence, generated Session Receipts, and Obsidian-readable Markdown.
---

# DLN Learn — Local-First Learning Orchestrator

Dunk is a **DLN-inspired pedagogical adaptation** for tutoring operations. It does not implement or validate Alia Wu's computational model. It uses three evidence-driven stages:

| Store stage | Learner-facing stage | Skill ID | Purpose |
|---|---|---|---|
| `acquire` | Acquire/Discriminate | `dln-dot` | Build foundations and test recall, distinctions, and application. |
| `relate` | Relate/Abstract | `dln-linear` | Compare known structures and test abstraction/transfer. |
| `revise` | Predict/Revise/Compress | `dln-network` | Make predictions, expose error, and revise the learner's model. |

The historical Dot/Linear/Network names remain implementation identifiers and backward-compatible user triggers only.

## Required references

Before operating the store or routing a session, read:

- `@references/local-store-schema.md` — exact profile/event/state contract.
- `@references/local-persistence-protocol.md` — CLI lifecycle, revisions, retries, and recovery.
- `@references/syllabus-grounding-protocol.md` — supplied-document intake, approval, bounded grounding, and citations.
- `@references/evidence-protocol.md` — what does and does not count as evidence.
- `@references/session-receipt-format.md` — the sole completion artifact.
- `@references/sync-protocol.md` — local checkpoints and plan adjustment.
- `@references/visual-format.md` — learner-facing visual conventions.

The files `init-template.md`, `merge-payload-schema.md`, and `merge-protocol.md` are legacy KS compatibility references. Do not use them in an active session.

## Local authority

Run the stdlib-only CLI:

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/dln-store.py" <command>
```

Root discovery is `--root`, then `DLN_VAULT_ROOT`, then `${CLAUDE_PLUGIN_DATA}/dln-vault`. If the CLI says the root is unconfigured, ask the user to set `DLN_VAULT_ROOT`; do not choose an implicit directory.

Canonical sources are `profile.yaml` and append-only `events.jsonl`, including immutable syllabus source and approval history. `state.json`, `dashboard.md`, `syllabus/<source-version-id>.md`, and `sessions/<session-id>.md` are deterministic generated projections. Raw syllabus PDFs are not retained. Obsidian can open the ordinary Markdown directly and is never a write authority.

## Command parsing

Extract the command and domain:

- `dln` / `learn <domain>` / `continue <domain>` → standard session.
- `dln list` → list domains.
- `dln reset <domain>` → append a reset event after confirmation.
- `dln exam <domain> by <date>` → configure `profile.exam`.
- `dln exam <domain> status` → show exam configuration plus measured evidence.
- `dln mock <domain>` → run an assessment session using exam configuration.
- `dln cram <N>d <domain>` → update review preferences/exam plan, not proficiency.

If the domain is missing, ask for it. Never guess a domain ID from a slug; resolve it from `list`.

## List

Run `dln-store.py list`. Present each ready domain with domain, goal, revision, and status. For richer stage/review/session information, run `context` for the selected domain only; do not bulk-load raw files. Unavailable domains must show their diagnostic and recovery need rather than stale cached values.

## Initialize

When no matching domain exists:

1. Ask for or infer a concise goal and confirm it.
2. Run `init --domain <domain> --goal <goal>` and retain the returned `domain_id` and revision `0`.
3. Continue through the shared syllabus registration/status flow below before routing the first session.

## Syllabus registration and pending approval

For both new and existing domains, ask whether the learner wants to register a supplied syllabus whenever one is offered. After `init` or `context`, resolve syllabus status before routing teaching:

1. If a supplied syllabus is chosen, require a runtime-resolvable file containing the actual bytes. A transient attachment preview or attachment content without a readable path/byte channel cannot be registered; say so and do not claim grounding.
2. For a readable file, run `ingest-syllabus` with `--adapter st5201x-2026-v1`, `--media-type application/pdf`, the original filename, timestamp, and retained revision. Only the exact supported ST5201X digest can succeed, then reload `context`.
3. If `state.grounding.status` is `approval_required` or `approved_update_pending`, open the pending generated Syllabus Intake Receipt from `state.grounding.pending_sources` and present its assertions. Collect a complete learner decision: accepted, corrected, or deferred. Keep the Week 7–13 row alignment unresolved unless the learner explicitly corrects it.
4. Write the complete approval request privately and run `approve-syllabus` with the retained revision. Reload `context` after success and use its approved `state.grounding`; do not patch `profile.syllabus` on this grounded path.
5. While status is `approved_update_pending`, the prior `active_source` and `active_approval` remain authoritative. Never cite or teach pending-source assertions as settled before approval.
6. If intake fails because bytes are unavailable, unreadable, or do not match the exact adapter digest, offer either a registration retry or the separate ungrounded curriculum path. Never conflate the two.
7. On the explicit no-document/generated-curriculum path only, dispatch `dln-syllabus` with `{domain, goal}`, show its ungrounded flat topics for learner edits, and commit the approved `profile_patch.syllabus`. Label the result generated and non-citable even when research tools were available.

A syllabus or curriculum is planning configuration, not mastery or evidence.

## Resume and route

For every existing domain:

1. Run `context --domain-id <id>`.
2. Retain `profile`, `state`, `state.grounding`, and `state.revision`.
3. If context exits `5`, run `doctor --recover` and retry context. For other errors, follow the persistence protocol and do not start a persistent session from stale conversation memory.
4. Check `state.next_review_date`, subject retrieval status, latest independent/supported evidence, syllabus, goal, current model, and exam configuration.
5. Create one portable session ID and keep it for the entire live session.
6. Route solely from `state.stage`:
   - `acquire` → preload/invoke `dln-dot` as **Acquire/Discriminate**.
   - `relate` → preload/invoke `dln-linear` as **Relate/Abstract**.
   - `revise` → preload/invoke `dln-network` as **Predict/Revise/Compress**.
7. Pass only bounded `context` output (including `state.grounding`), the domain ID, retained revision, session ID, command intent, and whether a delayed review is due. Do not pass raw PDF bytes, full event history, canonical page text, prior chat, or generated Markdown as machine state.

A phase skill owns teaching and structured event construction. The parent owns CLI calls, revision tracking, recovery, and receipt presentation unless the runtime explicitly permits the phase skill to run Bash.

## Review and spacing

A due review begins with an independent retrieval/application task before cues or new teaching. A valid delayed retrieval assessment must link a prior same-subject assessment and record the scheduled date and positive observed delay.

- If a delayed independent retrieval passes, the next review may be expanded according to `profile.review_preferences`.
- If it is partial/fail, shorten the next review and route remediation from the evidence. Supported or non-passing retrievals stay measured evidence and never satisfy the gate, so a passing subject stays `needs-retrieval`.
- If no linked delayed retrieval occurred, say **spacing was not measured**. Do not advance or reset from schedule passage alone.

Review schedules are future intentions. Only completed structured assessment events affect subject evidence.

## Reset

After naming exactly what will reset, require confirmation. Then:

1. Run `context` and retain its current revision.
2. Commit one `domain_reset` event with a new session ID, timestamp, and optional reason.
3. Re-run `context` and report the new generation/stage.

Reset never deletes events, profile ownership fields, or historic receipts. If the learner wants a different domain identity, initialize a new domain instead.

## Goal and syllabus edits

Apply learner-approved goal changes through a revision-checked `profile_patch`. Flat `profile.syllabus` edits are allowed only for the legacy ungrounded curriculum path. When an approved source is active, `state.syllabus` is derived from approved coverage assertions; change grounded values through a complete superseding `approve-syllabus` snapshot, never a profile patch. The syllabus agent never writes. Re-read context after a stale revision and retry the semantically same approved operation once. Added topics do not demote or promote a stage automatically; route any new foundational gap through measured evidence.

## Exam commands

Exam mode is configuration plus ordinary structured assessments—not a separate mastery system.

### Configure

Gather only the needed metadata (date, format, duration, marks, AI policy, target score, priority topics) and write it as `profile_patch.exam`. Date urgency may change task selection and review preferences, but never relabel supported work as independent or bypass evidence gates.

### Status

Show exam configuration, remaining calendar time, completed receipts, subject evidence, delayed retrieval status, and calibration only when measured. Do not calculate readiness from coverage, notes, or self-report alone.

### Mock

Automatic mock-question generation is deferred. If the learner supplies questions/rubrics or `profile.exam` already contains approved tasks, run them as a structured assessment session: record each outcome with honest assistance and novelty, collect confidence before answers when calibration is desired, and close with `session_completed`. Otherwise explain the deferral and offer a regular goal-aligned assessment session. The generated receipt is the record. A self-reported external outcome belongs only in `exam_cycle_closed.self_reported_outcome`.

### Cram

Cram changes prioritization and review timing in `review_preferences`/`exam`; it does not weaken evidence requirements. Explicitly warn when delayed retention cannot be measured before the exam.

### Close exam cycle

After confirmation, commit `exam_cycle_closed` with an immutable snapshot of the prior exam configuration and optional self-reported result, then clear or replace current `profile.exam` in the same revision-checked request. Preserve all assessment history.

## Stage transitions

A phase may propose a transition only after the supporting independent assessments have committed. Commit a `stage_transition` citing those event IDs and a stable rubric ID. Then reload context and route from the returned stage.

Do not transition because of session count, syllabus coverage, delivered material, supported checks, confidence, imported legacy claims, or a polished note/model alone.

## Session completion

At the end of every persistable session:

1. Commit any remaining assessment/model/transition events followed by one terminal `session_completed` event in the same request.
2. Include every same-session learner-visible evidence event in `evidence_event_ids`.
3. Parse the success revision and reload context.
4. Read the generated `sessions/<session-id>.md` receipt.
5. Present that receipt as the sole canonical summary. Do not write a second narrative recap or session log.

If the completion commit fails, state that the session is not durably closed. Keep the pending structured request in conversation and follow recovery/retry rules; never fabricate a receipt.

## Non-negotiable safeguards

- Content delivery, dialogue, plans, notes, syllabus assertions, source approval, citations, and coverage are not evidence.
- Supported and independent performance remain separate.
- Transfer requires `novel`; calibration requires pre-answer confidence plus score; retrieval requires a linked prior assessment and observed delay.
- Omit response time unless explicitly timed.
- Event IDs are stable across retries.
- On unresolved persistence failure, stop writes; do not use prose or generated Markdown as fallback state.
- Never contact a remote workspace for canonical reads/writes. A future exporter may consume generated projections one way, but is outside this active path.
