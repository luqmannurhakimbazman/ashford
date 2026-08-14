# Trigger Tests: tech-blog

**Test types:** `MANUAL` -- requires a live Claude Code session.

## Should Activate `MANUAL`

### 1. Direct blog request
- **Query:** "write a blog post about attention mechanisms"
- **Expected:** tech-blog activates

### 2. Technical deep dive
- **Query:** "draft a deep dive on variational inference"
- **Expected:** tech-blog activates

### 3. Tutorial post
- **Query:** "write a tutorial post explaining how to implement a B-tree"
- **Expected:** tech-blog activates

### 4. Convert source material
- **Query:** "turn this paper into a technical blog post"
- **Expected:** tech-blog activates

### 5. Static-site article
- **Query:** "create an Astro blog article about this project"
- **Expected:** tech-blog activates

## Should NOT Activate `MANUAL`

### 6. Repository documentation
- **Query:** "write a README for this repository"
- **Expected:** doc-generator activates, NOT tech-blog

### 7. Academic paper request
- **Query:** "prepare this manuscript for an ICLR submission"
- **Expected:** ml-paper-writing activates, NOT tech-blog

### 8. Marketing copy
- **Query:** "write a product launch announcement"
- **Expected:** Does NOT activate
