# Email Patterns for Job Application Tracking

## Known ATS Sender Domains

| Domain | Platform |
|--------|----------|
| `greenhouse.io` | Greenhouse |
| `lever.co` | Lever |
| `ashbyhq.com` | Ashby |
| `myworkdayjobs.com`, `myworkday.com` | Workday |
| `icims.com` | iCIMS |
| `smartrecruiters.com` | SmartRecruiters |
| `jobvite.com` | Jobvite |
| `successfactors.com`, `successfactors.eu` | SAP SuccessFactors |
| `taleo.net` | Oracle Taleo |
| `hire.jazz.co` | JazzHR |
| `breezy.hr` | Breezy HR |
| `applytojob.com` | BambooHR |

## Subject Line Patterns by Stage

### Applied (confirmation)
- "Application received"
- "Thank you for applying"
- "We received your application"
- "Application confirmation"
- "Your application for [Role] at [Company]"
- "Application submitted successfully"

### Online Assessment
- "Online assessment invitation"
- "Complete your assessment"
- "Coding challenge"
- "HackerRank" / "CodeSignal" / "Codility" in subject or body
- "Technical assessment for [Role]"
- "Please complete the following test"

### Behavioral Interview
- "Phone screen"
- "Schedule a call"
- "Introductory call"
- "Recruiter call"
- "First-round interview"
- "We'd like to learn more about you"
- Calendly/calendar invite link in body

### Onsite Interview
- "Onsite interview"
- "Virtual onsite"
- "Final round"
- "Superday"
- "On-site visit"
- "Full loop interview"
- "Meet the team"

### Offered
- "Pleased to offer"
- "Offer letter"
- "Congratulations"
- "We'd like to extend an offer"
- "Compensation package"

### Rejected
- "Not moving forward"
- "Other candidates"
- "Regret to inform"
- "Unfortunately"
- "Position has been filled"
- "After careful consideration"
- "We will not be proceeding"
- "We have decided to pursue"

## Recruiter Email Heuristics

An email is likely from a recruiter (not an ATS bot) when:
- Sender domain matches a company domain (not an ATS platform)
- Email is personalized (uses candidate's first name in body, not just subject)
- Contains scheduling language: "availability", "schedule", "meet", "chat", "connect"
- Contains a calendar link (Calendly, Google Calendar invite, Outlook invite)
- Signed with a person's name and title (e.g., "Jane Smith, Talent Acquisition")

## Entity Extraction Fallback Order

1. **Subject line** — often contains "[Company] - [Role]" or "Your application for [Role] at [Company]"
2. **Email body** — look for "the [Role] position at [Company]" patterns
3. **Sender "on behalf of"** — ATS emails often include "on behalf of [Company]" in headers
4. **Sender domain** — if not an ATS domain, the sender's company domain is the company name
5. **Flag as ambiguous** — if none of the above yield a clear company + role
