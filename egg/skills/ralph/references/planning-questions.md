# Planning Questions Framework

## Brain-Dump Capture (Claude → User)

Ask these one at a time. Skip questions the user already answered in their initial message.

### Project Understanding
1. What are you building? (One sentence elevator pitch)
2. What does the codebase look like right now? (Existing code, or greenfield?)
3. What does "done" look like? (How will you know Ralph succeeded?)
4. What's the tech stack? (Language, framework, key libraries)

### Scope & Constraints
5. What's explicitly OUT of scope? (Features to skip, corners to cut)
6. Are there hard constraints? (Must use X, can't use Y, must run on Z)
7. How should errors be handled? (Fail fast? Graceful degradation? Retry?)
8. Are there tests? (Existing test suite? Preferred test framework? Coverage target?)

### Domain Knowledge
9. What domain concepts does Claude need to understand? (Business rules, data formats, APIs)
10. Are there existing patterns Claude should follow? (Code style, architecture conventions)

## Assumption Surfacing (Bidirectional)

After brain-dump capture, explicitly say:

> "Now I'll share what I'm assuming about this project. Tell me where I'm wrong, and then ask ME any questions about what I'll be building."

Then list your assumptions as a numbered list. Wait for the user to correct them and ask their questions.

### Common Hidden Assumptions to Surface
- Database choice and schema design
- Authentication/authorization model
- Error response format
- Logging and observability
- Configuration management (env vars, config files)
- Deployment model (where does this run?)
- Concurrency model (single-threaded? async? multi-process?)
- External service dependencies and failure modes

## Context Budget Estimation

After spec and plan are drafted, estimate the per-iteration context load:

1. Count lines in SPEC.md (target: under 200 lines total)
2. For each task, sum: relevant spec sections + codebase files + task details + iteration protocol (~60 lines)
3. Flag any task whose estimated context exceeds ~40% of a context window
4. Suggest decomposition for oversized tasks
