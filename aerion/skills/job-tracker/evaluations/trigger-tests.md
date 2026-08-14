# Trigger Tests

**Test types:** `MANUAL` -- requires a live Claude Code session.

## Should Activate `MANUAL`

### 1. Direct application check
- **Query:** "check my job applications from the last 7 days"
- **Expected:** job-tracker activates

### 2. Gmail application scan
- **Query:** "scan Gmail for application updates"
- **Expected:** job-tracker activates

### 3. Tracker synchronization
- **Query:** "sync my application statuses to the job tracker"
- **Expected:** job-tracker activates

### 4. Interview invitation check
- **Query:** "did I receive any interview invitations?"
- **Expected:** job-tracker activates

## Should NOT Activate `MANUAL`

### 5. General email search
- **Query:** "find the latest invoice in my Gmail"
- **Expected:** Does NOT activate

### 6. General spreadsheet update
- **Query:** "update my monthly budget spreadsheet"
- **Expected:** Does NOT activate
