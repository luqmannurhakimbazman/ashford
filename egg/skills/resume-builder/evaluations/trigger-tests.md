# Trigger Tests

Evaluation scenarios for verifying skill activation and routing behavior.

**Test types:**
- `MANUAL` — requires a live Claude Code session with the skill installed. Cannot be automated.

## Should Activate `MANUAL`

### 1. Direct tailoring request
- **Query:** "tailor my resume for this JD" (pastes job description)
- **Expected:** Skill activates, reads master resume and candidate-context.md, begins Step 1

### 2. Company-specific request
- **Query:** "help me customize my resume for a data engineer role at Stripe"
- **Expected:** Skill activates, asks for JD if not provided, derives slug `stripe-data-engineer-resume`

### 3. Cover letter request
- **Query:** "write a cover letter for this position" (pastes job description)
- **Expected:** Skill activates, follows cover letter workflow (Step 11), outputs `cover-letter.md`

### 4. ATS optimization request
- **Query:** "optimize my resume for ATS"
- **Expected:** Skill activates, begins keyword extraction (Step 2), references `ats-keywords.md`

### 5. Resume with URL JD
- **Query:** "tailor my resume for this role: https://jobs.lever.co/company/12345"
- **Expected:** Skill activates, fetches JD from URL, proceeds with full workflow

### 6. Refactor phrasing
- **Query:** "refactor resume for ML engineer at DeepMind"
- **Expected:** Skill activates (trigger phrase "refactor resume"), begins workflow

## Should NOT Activate `MANUAL`

### 7. General resume advice
- **Query:** "what's a good resume format for software engineers?"
- **Expected:** Skill does NOT activate (general advice, not tailoring to a specific JD)

### 8. Resume review / proofreading
- **Query:** "review my resume for typos and grammar"
- **Expected:** Skill does NOT activate (editing/proofreading, not JD-targeted tailoring)

### 9. Writing a resume from scratch
- **Query:** "help me create a resume from scratch"
- **Expected:** Skill does NOT activate (no master resume tailoring — this is a creation task)

### 10. Interview prep
- **Query:** "prepare me for a Goldman Sachs S&T interview"
- **Expected:** Skill does NOT activate (interview prep, not resume tailoring — should route to global-markets-teacher)
