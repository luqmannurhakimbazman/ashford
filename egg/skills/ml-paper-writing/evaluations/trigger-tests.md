# Trigger Tests: ml-paper-writing

**Test types:** `MANUAL` -- requires a live Claude Code session.

## Should Activate `MANUAL`

### 1. Draft an ML paper
- **Query:** "draft an ML paper from the results in this repository"
- **Expected:** ml-paper-writing activates

### 2. Conference submission
- **Query:** "prepare this manuscript for a NeurIPS submission"
- **Expected:** ml-paper-writing activates

### 3. Structure a research argument
- **Query:** "help structure the argument and contributions for my ICLR paper"
- **Expected:** ml-paper-writing activates

### 4. Verify academic citations
- **Query:** "verify the citations in this machine learning paper"
- **Expected:** ml-paper-writing activates

### 5. Camera-ready preparation
- **Query:** "make this ACL paper camera-ready"
- **Expected:** ml-paper-writing activates

## Should NOT Activate `MANUAL`

### 6. Technical blog request
- **Query:** "write a blog post explaining this paper"
- **Expected:** tech-blog activates, NOT ml-paper-writing

### 7. General documentation
- **Query:** "generate API docs for this model-serving package"
- **Expected:** doc-generator activates, NOT ml-paper-writing

### 8. Paper summary
- **Query:** "summarize this paper for me"
- **Expected:** Does NOT activate
