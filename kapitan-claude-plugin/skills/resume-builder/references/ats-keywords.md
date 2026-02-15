# ATS Keywords & Extraction Strategy

## How ATS Systems Work

Applicant Tracking Systems parse resumes and score them against job descriptions. Key mechanics:

1. **Keyword matching**: Exact and fuzzy matching of terms from the JD
2. **Frequency weighting**: Terms mentioned multiple times in the JD are weighted higher
3. **Section awareness**: Some ATS weight skills sections differently from experience
4. **Acronym handling**: Include both the acronym and full form (e.g., "Natural Language Processing (NLP)")
5. **PDF parsing**: `fed-res.cls` uses `\pdfgentounicode=1` for machine-readable output
6. **Experience duration inference**: Some ATS infer years of experience for a skill based on where it appears in your timeline. If Python appears in a role you held for 3 years, the ATS may credit you with 3 years of Python experience. Place high-priority keywords in your longest-tenure roles for maximum inferred experience.

## Keyword Extraction Process

### 1. Identify Required vs. Preferred

JDs typically separate requirements into:
- **Required/Must-have**: Non-negotiable — these MUST appear in the resume
- **Preferred/Nice-to-have**: Bonus points — include if you have them
- **Implicit**: Not stated but implied by the role (e.g., "ML Engineer" implies Python)

### 2. Categorize Keywords

| Category | What to Look For |
|----------|-----------------|
| Programming languages | Python, R, SQL, TypeScript, C++ |
| Frameworks/libraries | PyTorch, TensorFlow, scikit-learn, React |
| Cloud/infra | AWS, GCP, Docker, Kubernetes |
| Methodologies | Agile, CI/CD, TDD |
| Domain terms | NLP, computer vision, reinforcement learning |
| Soft skills | Leadership, cross-functional, mentoring |
| Qualifications | Degree, years of experience, certifications |

### 3. Count Frequency

Terms repeated across the JD are high-priority. Example:

```
"Python" appears 4 times → Must be prominent
"Docker" appears 1 time → Include but don't force
```

### 4. Map to Resume Content

For each high-priority keyword:
1. Check if it already exists in the master resume
2. If yes, ensure it appears in a prominent position
3. If no but you have the experience, add it naturally to a bullet
4. If no and you lack the experience, skip it (never fabricate)

## Keyword Placement Priority

1. **Skills section** — Direct keyword listing, highest ATS hit rate
2. **Most recent experience bullets** — Contextual keyword usage
3. **Project descriptions** — Additional keyword reinforcement
4. **Education** — Relevant coursework and modules

## Common Pitfalls

- **Keyword stuffing**: ATS may flag unnatural repetition. Use each keyword 2-3 times max across the resume.
- **Missing exact terms**: If JD says "machine learning", don't only write "ML" — use both forms.
- **Ignoring soft skills**: Many ATS also scan for leadership, communication, teamwork.
- **Wrong section**: Technical skills in the Skills section parse better than buried in bullet text for some ATS.

## Abbreviation Expansion

Always expand abbreviations on first use to maximize ATS matching:

1. **Skills section**: Use full forms with abbreviations — "Amazon Web Services (AWS)", "Google Cloud Platform (GCP)", "Natural Language Processing (NLP)"
2. **First mention in bullets**: Expand on first use — "...using Amazon Web Services (AWS) for deployment..."
3. **Subsequent mentions**: Abbreviation alone is fine after first use within the same section
4. **Why this matters**: ATS may search for either "AWS" or "Amazon Web Services" — including both ensures you match regardless of which form the recruiter configured
