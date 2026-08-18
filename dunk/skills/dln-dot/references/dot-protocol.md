# Acquire/Discriminate Protocol

Detailed teaching and assessment protocol for the backward-compatible `dln-dot` skill ID. Shared storage and evidence rules in the parent `dln` references are authoritative.

## Objective

Build a small usable foundation, then measure whether the learner can retrieve, distinguish, and apply it. Explanations and demonstrations support learning but do not themselves change proficiency.

## Cycle

### 1. Select targets

Use `context.state.subjects`, syllabus, goal, due retrieval, and exam configuration. Prefer:

1. due independent retrieval;
2. independent partial/fail;
3. supported-only subjects that need an independent check;
4. syllabus-relevant unmeasured foundations.

Keep a batch to one to three interacting concepts. If the learner struggles, reduce elements and change the representation rather than lowering the rubric after seeing the answer.

### 2. Acquire

For each concept:

- give a plain-language definition;
- show a concrete example and a boundary/non-example;
- state why it matters to the learner's goal;
- connect it to one prior measured subject when appropriate.

No event is created for delivery.

### 3. Discriminate

Ask the learner to compare plausible alternatives, diagnose a misconception, or choose which concept applies. Do not identify the relevant concept in a task intended to test discrimination.

### 4. Assess

Fix the task, `context_id`, and rubric before the response. Use one of these rubric IDs (or a documented versioned domain-specific derivative):

- `acquire-recall-v1`: accurately state the essential meaning without cues.
- `discriminate-boundary-v1`: choose the correct alternative and explain the decisive distinction.
- `discriminate-application-v1`: apply the concept to a variant and justify the choice.
- `acquire-delayed-retrieval-v1`: independently retrieve/apply the same subject after a recorded positive delay.

Outcome rules:

- `pass`: satisfies every essential rubric criterion;
- `partial`: correct direction but misses an essential distinction/step;
- `fail`: cannot produce the essential meaning or applies the wrong principle.

Record actual assistance. If a hint is needed, close the independent attempt with its observed result, teach or prompt, and create a separate supported assessment for the re-check.

## Retrieval

Run retrieval before showing notes, state, answers, or the prior receipt. A delayed retrieval event cites a prior same-subject assessment and stores the scheduled date and observed delay. A free-recall attempt with no valid prior event or no positive delay is still an assessment but is not measured delayed retrieval.

## Interleaving

After initial blocked teaching, mix old and new subjects using classification/application tasks. Interleaving is a task-selection strategy, not an event kind. Mark novelty honestly and do not call a familiar shuffled item novel transfer.

## Causal chains

Represent a chain as a stable subject when it can be tested consistently:

```json
{"id":"chain-stable-id","label":"A leads to B through mechanism C","type":"chain"}
```

Assess the direction, intermediate mechanism, and boundary conditions. A diagram the tutor creates is instruction. The learner's unaided reconstruction or application can be evidence.

## Frustration and remediation

Acknowledge difficulty without inferring ability or motivation. Use a different analogy, smaller subtask, or worked example. Then run a distinct supported check. Later, schedule an independent check rather than claiming the remediation established independent proficiency.

## Gate rubric: `acquire-to-relate-v1`

The learner must produce committed independent evidence for:

1. recall/explanation across the core goal-relevant foundation;
2. discrimination of important near-misses;
3. application on at least a variant task;
4. two or more connected subjects when the domain requires relational reasoning.

A gate task may generate several assessment events. Commit those first. If they pass the rubric, commit a separate `stage_transition` citing their IDs. The gate does not require a percentage of notes/syllabus entries and cannot use supported events.

## Feedback

Give outcome feedback after the attempt. Name the criterion met or missed and the next task. Process praise is welcome, but never report a mastery label that is not present in projected state.

## Session end

The terminal completion event cites the learner-visible assessment and transition event IDs. The generated Session Receipt, not this protocol or an improvised recap, is the session record.
