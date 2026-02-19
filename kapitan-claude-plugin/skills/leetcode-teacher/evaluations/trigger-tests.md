# Trigger Tests

Manual evaluation scenarios for verifying skill activation and routing behavior.

## Should Activate

### 1. Direct teaching request
- **Query:** "teach me how to solve two sum"
- **Expected:** Skill activates, enters Learning Mode, starts with layman intuition question

### 2. URL-based request
- **Query:** "https://leetcode.com/problems/merge-intervals/ walk me through this"
- **Expected:** Skill activates, attempts URL fetch, enters Learning Mode

### 3. Recall mode request
- **Query:** "quiz me on binary search"
- **Expected:** Skill activates, enters Recall Mode, asks for unprompted reconstruction

### 4. ML implementation request
- **Query:** "implement the Adam optimizer from scratch"
- **Expected:** Skill activates, classifies as ML Implementation, references ml-implementations.md

### 5. Data structure fundamentals
- **Query:** "help me understand how a hash table works"
- **Expected:** Skill activates, classifies as Data Structure Fundamentals, starts with internals

## Should NOT Activate

### 6. General knowledge question
- **Query:** "what sorting algorithm does Python use internally?"
- **Expected:** Skill does NOT activate (general knowledge, not a teaching/practice request)

### 7. Code review request
- **Query:** "review my binary search implementation for bugs"
- **Expected:** Skill does NOT activate (code review, not learning/teaching)

### 8. Project work
- **Query:** "add a search feature to my app using binary search"
- **Expected:** Skill does NOT activate (implementation task, not learning)

## Mode Routing

### 9. Ambiguous intent
- **Query:** "I know two sum, show me"
- **Expected:** Asks whether user wants quiz (Recall) or review (Learning), does NOT assume mode

### 10. Mode transition — upshift
- **Query:** User in Learning Mode gives optimal solution unprompted on first try
- **Expected:** Offers to switch to Recall Mode (quiz), does NOT force the switch

### 11. Mode transition — downshift
- **Query:** User in Recall Mode cannot reconstruct the algorithm at all
- **Expected:** Downshifts to Learning Mode for the specific gap, offers to resume quiz after

## Functional — Learning Mode

### 12. Full learning flow
- **Query:** "walk me through the merge intervals problem"
- **Expected:**
  - Starts with layman intuition question (does NOT give solution immediately)
  - Follows 6-section structure (Intuition -> Brute Force -> Optimal -> Alternatives -> Summary -> Interview)
  - Uses three-tier progressive hints when user is stuck
  - Ends with study notes and profile update

### 13. Weakness calibration
- **Setup:** Learner profile has "recurring" weakness on "hashmap insertion syntax"
- **Query:** "teach me two sum"
- **Expected:** Actively probes hashmap syntax during the session, adjusts scaffolding for that gap

## Functional — Recall Mode

### 14. Full recall flow
- **Query:** "test my recall on merge two sorted lists"
- **Expected:**
  - Enters Recall Mode (interviewer persona, neutral acknowledgments)
  - Steps: R1 (framing) -> R2 (reconstruction) -> R3 (edge cases) -> R4 (complexity) -> R5 (pattern) -> R6 (variation) -> R7 (debrief/scoring)
  - Provides verdict (Strong Pass / Pass / Borderline / Needs Work)
  - Suggests review schedule based on performance
