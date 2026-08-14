# Trigger Tests

Manual evaluation scenarios for verifying skill activation and routing behavior.

## Should Activate

### 1. Direct teaching request
- **Query:** "teach me about yield curves"
- **Expected:** Skill activates, enters Learning Mode, starts with layman intuition question

### 2. Concept explanation request
- **Query:** "explain contango and backwardation"
- **Expected:** Skill activates, enters Learning Mode, classifies as Commodities

### 3. Instrument deep-dive
- **Query:** "how do credit default swaps work?"
- **Expected:** Skill activates, enters Learning Mode, loads credit.md and derivatives-fundamentals.md

### 4. Recall mode request
- **Query:** "quiz me on options Greeks"
- **Expected:** Skill activates, enters Recall Mode, asks for unprompted explanation

### 5. Mock interview request
- **Query:** "mock interview me for Goldman S&T"
- **Expected:** Skill activates, enters Mock Interview Mode, loads firm-bank-st.md, starts M1

### 6. Headline analysis
- **Query:** "Fed raises rates 50bps, signals more to come — analyze this"
- **Expected:** Skill activates, uses headline-analysis-framework.md, traces cross-asset effects

### 7. Firm-specific prep
- **Query:** "prepare me for a Balyasny interview"
- **Expected:** Skill activates, enters Mock Interview Mode, loads firm-hedge-fund.md

### 8. Asset class teaching
- **Query:** "walk me through how the repo market works"
- **Expected:** Skill activates, enters Learning Mode, loads rates-fixed-income.md

## Should NOT Activate

### 9. Current price query
- **Query:** "what's the S&P 500 at?"
- **Expected:** Skill does NOT activate (data lookup, not teaching)

### 10. General knowledge question
- **Query:** "who is the CEO of Goldman Sachs?"
- **Expected:** Skill does NOT activate (factual question, not teaching/practice)

### 11. Code review request
- **Query:** "review my Python trading bot code"
- **Expected:** Skill does NOT activate (code review, not markets teaching)

### 12. Project work
- **Query:** "build me a portfolio optimization tool"
- **Expected:** Skill does NOT activate (implementation task, not learning)

### 13. LeetCode request
- **Query:** "teach me two sum"
- **Expected:** leetcode-teacher activates instead, NOT global-markets-teacher

## Mode Routing

### 14. Ambiguous intent
- **Query:** "I know about yield curves"
- **Expected:** Asks whether user wants quiz (Recall), teaching (Learning), or mock interview

### 15. Mode transition — upshift
- **Query:** User in Learning Mode gives complete framework with cross-asset implications unprompted
- **Expected:** Offers to switch to Recall Mode (quiz), does NOT force the switch

### 16. Mode transition — downshift
- **Query:** User in Recall Mode cannot explain the core concept at all
- **Expected:** Downshifts to Learning Mode for the specific gap, offers to resume quiz after

### 17. Headline in recall mode
- **Query:** User is in Recall Mode, pastes a Bloomberg headline
- **Expected:** Analyzes headline within Recall Mode context (tests application, not teaches)

## Functional — Learning Mode

### 18. Full learning flow
- **Query:** "walk me through interest rate swaps"
- **Expected:**
  - Starts with layman intuition question (does NOT explain immediately)
  - Follows 6-section structure (Intuition -> Naive Model -> Market Framework -> Alternative Views -> Summary -> Interview)
  - Uses three-tier progressive hints when user is stuck
  - Ends with study notes and profile update

### 19. Weakness calibration
- **Setup:** Learner profile has "recurring" weakness on "confuses modified duration with Macaulay duration"
- **Query:** "teach me about bond duration"
- **Expected:** Actively probes duration distinction during the session

## Functional — Recall Mode

### 20. Full recall flow
- **Query:** "test my recall on carry trades"
- **Expected:**
  - Enters Recall Mode (interviewer persona, neutral acknowledgments)
  - Steps: R1 (framing) -> R2 (explanation) -> R3 (scenarios) -> R4 (precision) -> R5 (classification) -> R6 (variation) -> R7 (debrief/scoring)
  - Provides verdict (Strong Pass / Pass / Borderline / Needs Work)
  - Suggests review schedule based on performance

## Functional — Mock Interview Mode

### 21. Full mock interview flow
- **Query:** "mock interview me for a hedge fund trading role"
- **Expected:**
  - Enters Mock Interview Mode with hedge fund calibration
  - Steps: M1 (resume) -> M2 (market views) -> M3 (trade pitch) -> M4 (scenario) -> M5 (brain teaser) -> M6 (debrief)
  - Maintains professional interviewer persona throughout M1-M5
  - Drops persona for coaching in M6
  - Provides section-by-section scoring and overall verdict

### 22. Firm-type calibration
- **Query:** "mock interview me for Trafigura"
- **Expected:** Loads firm-trading-house.md, emphasizes physical commodities, logistics, Incoterms in M3-M4


## Persistent State `AUTO`

### 23. Description budget
```bash
description=$(sed -n 's/^description: //p' egg/skills/global-markets-teacher/SKILL.md)
[ "${#description}" -le 1536 ]
```
- **Expected:** Exit 0; the decoded plain-scalar description stays within the skill listing cap.

### 24. Legacy state migration
- **Setup:** Set `HOME` and `CLAUDE_PLUGIN_DATA` to temporary fixture directories. Put valid markets profile, ledger, and session-state files only under `$HOME/.local/share/claude/`.
- **Action:** Run `egg/scripts/markets-profile-load.sh` with a SessionStart JSON fixture.
- **Expected:** All three files move to `${CLAUDE_PLUGIN_DATA}`; no legacy file remains; existing files in plugin data are never overwritten.
