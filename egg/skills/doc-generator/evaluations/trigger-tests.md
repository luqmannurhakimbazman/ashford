# Trigger Tests: doc-generator

**Test types:** `MANUAL` -- requires a live Claude Code session.

## Should Activate `MANUAL`

### 1. Document code
- **Query:** "document this Python module"
- **Expected:** doc-generator activates

### 2. Generate project documentation
- **Query:** "generate docs for this project"
- **Expected:** doc-generator activates

### 3. Create a README
- **Query:** "create a README for this repository"
- **Expected:** doc-generator activates

### 4. Document an API
- **Query:** "write API documentation for these public endpoints"
- **Expected:** doc-generator activates

### 5. Add code comments
- **Query:** "add documentation comments to these exported functions"
- **Expected:** doc-generator activates

## Should NOT Activate `MANUAL`

### 6. Technical blog request
- **Query:** "turn this project into a technical blog post"
- **Expected:** tech-blog activates, NOT doc-generator

### 7. ML paper request
- **Query:** "draft the methods section for my NeurIPS paper"
- **Expected:** ml-paper-writing activates, NOT doc-generator

### 8. General code explanation
- **Query:** "explain why this function returns None"
- **Expected:** Does NOT activate
