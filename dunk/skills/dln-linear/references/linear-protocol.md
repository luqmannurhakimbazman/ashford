# Relate/Abstract Protocol

Detailed protocol for the backward-compatible `dln-linear` skill ID. The shared local schema, persistence, evidence, and receipt references are authoritative.

## Objective

Help the learner discover and test structure shared across known subjects. A relation identifies roles and mechanisms; an abstraction states the invariant and boundaries. Neither becomes evidence until evaluated in a predeclared task.

## Comparison cycle

### 1. Prepare

Choose two or more stable measured subjects. Ask the learner to retrieve each briefly before comparison. If one cannot be reconstructed independently, record that result and remediate rather than pretending both structures are available.

### 2. Relate

Prompt for:

- common causal role or constraint;
- decisive difference;
- invariant across surface changes;
- boundary where the analogy fails.

Use `relate-structure-v1`: pass requires a correct shared mechanism and a correct boundary/difference; partial has the direction but misses one; fail uses surface similarity or the wrong mechanism.

### 3. Abstract

Ask the learner to state the smallest domain-independent principle that explains the relation and predict what it would imply in another case.

Use `abstract-principle-v1`: pass requires a structural principle, boundary conditions, and a falsifiable implication; partial is structural but vague/domain-locked; fail merely renames examples.

### 4. Transfer

Present a case without signaling which abstraction applies. Only a genuinely unsignaled new structure is `novelty: novel`; a familiar structure with changed details is `variant`.

Use `abstract-transfer-v1`: pass requires selecting/applying the principle and explaining why; partial needs one non-answer-revealing clarification; if that clarification narrows the applicable principle, record supported evidence.

## Assistance boundary

Showing aligned diagrams, naming the shared factor, selecting the relevant chains, or supplying the key comparison makes the resulting check supported. Close any independent attempt before tutoring. Later checks are new events.

## Retrieval and calibration

Run linked delayed retrieval before displaying prior abstractions. A schedule alone is not evidence. Collect confidence before the answer and only when paired with numeric score/max score. Discuss the observed gap after feedback.

## Interleaving

Alternate comparison families so the learner must choose which structure applies. Include cases where no existing abstraction fits. Task sequencing is instructional metadata, not a separate event kind.

## Foundational failure

When a comparison fails because a prerequisite subject is absent or independently weak:

1. record the observed relate/abstract assessment;
2. identify the prerequisite without changing the outcome;
3. teach/remediate and record a distinct supported check if performed;
4. schedule an independent re-check or propose an evidence-cited transition to `acquire` for broad gaps.

## Gate rubric: `relate-to-revise-v1`

Required already-committed independent evidence:

1. correct structural relation across multiple known subjects;
2. precise abstraction with boundary conditions;
3. at least one `novelty: novel` transfer application;
4. explanation of the abstraction's prediction.

Commit a `stage_transition` only after these evidence events exist. The transition cites them, sets `from: relate`, `to: revise`, and records a concise decision. Count of notes/factors, coverage, supported success, confidence, and imported claims do not qualify.

## Visuals

Side-by-side chains and factor maps are teaching aids. Label mechanisms and differences clearly, but never count a tutor-rendered diagram as learner evidence. A learner's unaided reconstruction or new-case application may be assessed.

## Session end

Close with an atomic `session_completed` commit and present the generated Session Receipt. The receipt separates independent evidence, supported performance, transfer/calibration/retrieval measurement, and next action.
