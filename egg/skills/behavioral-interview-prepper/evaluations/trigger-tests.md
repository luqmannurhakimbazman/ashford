# Trigger Tests: behavioral-interview-prepper

## Should Trigger

| Input | Why |
|-------|-----|
| "prep behavioral for Google SWE" | Direct trigger phrase + company |
| "behavioral interview prep" | Exact trigger phrase |
| "prep me for interview at Citadel" | Exact trigger phrase with company |
| "generate behavioral answers for this role" | Exact trigger phrase |
| "practice behavioral questions" | Exact trigger phrase |
| "prep my stories for the Amazon interview" | Trigger phrase "prep my stories" |
| "can you make me an answer bank for my interview?" | Trigger phrase "answer bank for interview" |
| "help me with STAR method answers" | Trigger phrase "STAR method answers" |
| "I just finished tailoring my resume, now prep me for the interview" | Post-resume-tailor context |
| "behavioral prep for Jane Street" | Exact trigger phrase + finance company |
| "help me prep questions to ask my interviewer at Google" | Questions-for-interviewer feature |
| "walk me through your resume prep" | Resume walkthrough trigger phrasing |
| "practice my resume walkthrough" | Resume walkthrough feature |
| "help me with my SOAR answers for the Amazon interview" | SOAR format trigger |

## Should NOT Trigger

| Input | Why | Correct Skill |
|-------|-----|---------------|
| "tailor my resume for this JD" | Resume tailoring, not behavioral prep | resume-analyzer |
| "teach me about linked lists" | Coding interview, not behavioral | leetcode-teacher |
| "mock interview for S&T rates desk" | Markets technical interview | global-markets-teacher |
| "write a cover letter" | Cover letter generation | cover-letter |
| "quiz me on derivatives" | Technical finance content | global-markets-teacher |
| "help me with system design interview" | Technical interview, not behavioral | — |

## Edge Cases

| Input | Expected Behavior |
|-------|-------------------|
| "prep behavioral" (no resume-analyzer or resume-tailor output exists) | Trigger skill, but Step 1 should detect missing files and prompt the user to run resume-analyzer, then resume-tailor |
| "behavioral prep for Goldman Sachs trading" | Trigger skill + finance layer (Step 6) should activate |
| "prep behavioral for Anthropic" | Trigger skill, no finance layer |
| "prep my resume walkthrough for Goldman Sachs trading" | Trigger skill + resume walkthrough + finance layer should both activate |
