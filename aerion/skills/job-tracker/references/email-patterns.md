# Email Patterns for Job Application Tracking

> **Maintenance note:** When adding new ATS/assessment domains here, also update the Gmail search query in `aerion/commands/check-apps.md` to include them as `from:` filters.

## Known ATS Sender Domains

| Domain | Platform |
|--------|----------|
| `greenhouse.io`, `greenhouse-mail.io` | Greenhouse |
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
| `hackerrankforwork.com` | HackerRank (OA platform) |
| `codesignal.com` | CodeSignal (OA platform) |
| `codility.com` | Codility (OA platform) |
| `hirevue.com` | HireVue (OA/video platform) |
| `hackerearth.com` | HackerEarth (OA platform) |
| `karat.com` | Karat (technical interview platform) |
| `shl.com` | SHL (psychometric/aptitude platform) |
| `testgorilla.com` | TestGorilla (assessment platform) |
| `brassring.com` | IBM/Kenexa BrassRing |
| `avature.net` | Avature |
| `phenom.com` | Phenom People |

## Subject Line Patterns by Stage

### Applied (confirmation)
- "Application received"
- "Thank you for applying"
- "We received your application"
- "Application confirmation"
- "Your application for [Role] at [Company]"
- "Application submitted successfully"
- "Your submission" (generic confirmation from some ATS platforms)
- "Candidate reference" (reference check request, may also indicate later stages)

### Online Assessment
- "Online assessment invitation"
- "Complete your assessment"
- "Coding challenge"
- "HackerRank" / "CodeSignal" / "Codility" / "HireVue" / "HackerEarth" / "Karat" / "SHL" / "TestGorilla" in subject or body
- "Technical assessment for [Role]"
- "Please complete the following test"

### Phone Screen
- "Phone screen"
- "Schedule a call"
- "Introductory call"
- "Recruiter call"
- "We'd like to learn more about you"
- Calendly/calendar invite link in body (when sender is a recruiter, not an interviewer)

### Behavioral Interview
- "Behavioral interview"
- "First-round interview"
- "Second-round interview"
- "Tell me about a time"
- "STAR format"
- "Meet with the hiring manager"
- Calendly/calendar invite link in body (when sender is a hiring manager or interviewer)

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
