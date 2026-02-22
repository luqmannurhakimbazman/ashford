# Trigger Tests

Evaluation scenarios for verifying skill activation, routing behavior, and reference pre-loading.

**Test types:**
- `MANUAL` — requires a live Claude Code session with the skill installed. Cannot be automated.
- `AUTO` — can be run via shell commands. See each test for the exact command.

## Should Activate `MANUAL`

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
- **Expected:** Skill activates, classifies as ML Implementation, references `references/ml/ml-implementations.md`

### 5. Data structure fundamentals
- **Query:** "help me understand how a hash table works"
- **Expected:** Skill activates, classifies as Data Structure Fundamentals, starts with internals

## Should NOT Activate `MANUAL`

### 6. General knowledge question
- **Query:** "what sorting algorithm does Python use internally?"
- **Expected:** Skill does NOT activate (general knowledge, not a teaching/practice request)

### 7. Code review request
- **Query:** "review my binary search implementation for bugs"
- **Expected:** Skill does NOT activate (code review, not learning/teaching)

### 8. Project work
- **Query:** "add a search feature to my app using binary search"
- **Expected:** Skill does NOT activate (implementation task, not learning)

## Mode Routing `MANUAL`

### 9. Ambiguous intent
- **Query:** "I know two sum, show me"
- **Expected:** Asks whether user wants quiz (Recall) or review (Learning), does NOT assume mode

### 10. Mode transition — upshift
- **Query:** User in Learning Mode gives optimal solution unprompted on first try
- **Expected:** Offers to switch to Recall Mode (quiz), does NOT force the switch

### 11. Mode transition — downshift
- **Query:** User in Recall Mode cannot reconstruct the algorithm at all
- **Expected:** Downshifts to Learning Mode for the specific gap, offers to resume quiz after

## Reference Pre-Loading (Step 2B) `MANUAL`

Tests that at least one technique reference is loaded immediately after problem classification, before Socratic teaching begins. **Smoke test:** Try test 12 first — it's the fastest to verify since Valid Palindrome is a simple problem where you can quickly tell if reference content (Socratic prompts, two-pointer template) is informing the session.

### 12. Two-pointer / string problem
- **Query:** "teach me valid palindrome"
- **Expected:**
  - Classifies: string manipulation, two pointers
  - Step 2B loads: `references/techniques/string-techniques.md` (primary)
  - Optionally loads: `references/techniques/array-techniques.md` (two pointers)
  - Reference content informs Socratic prompts and code templates in Steps 3-5

### 13. Sliding window problem
- **Query:** "walk me through longest substring without repeating characters"
- **Expected:**
  - Classifies: sliding window, hash map, string
  - Step 2B loads: `references/algorithms/sliding-window.md` (primary, atomic file)
  - Does NOT load the 250-line `references/frameworks/algorithm-frameworks.md` mega-file for a sliding window problem

### 14. Interval scheduling problem
- **Query:** "teach me meeting rooms II"
- **Expected:**
  - Classifies: interval scheduling, heap
  - Step 2B loads: `references/algorithms/interval-scheduling.md` (primary, atomic file)
  - Optionally loads: `references/data-structures/data-structure-fundamentals.md` (heap internals)

### 15. DP problem
- **Query:** "explain word break to me"
- **Expected:**
  - Classifies: DP, backtracking, trie
  - Step 2B loads: `references/algorithms/dynamic-programming-core.md` (primary — contains string DP section)
  - Optionally loads: `references/algorithms/backtracking.md` (secondary)

### 16. Hard multi-technique problem
- **Query:** "teach me trapping rain water"
- **Expected:**
  - Classifies: two pointers + stack (monotonic) + DP
  - Step 2B loads primary + secondary (e.g., `references/techniques/array-techniques.md` + `references/techniques/stack-queue-monotonic.md`)
  - Step 6 (Alternatives): if presenting a technique not yet loaded, loads its reference before presenting

### 17. Graph problem
- **Query:** "walk me through network delay time"
- **Expected:**
  - Classifies: Dijkstra / shortest path
  - Step 2B loads: `references/graphs/dijkstra.md` (atomic file, direct match)
  - Does NOT load the generic `references/graphs/graph-algorithms.md` when a specific algorithm file exists

### 18. Greedy problem
- **Query:** "teach me jump game"
- **Expected:**
  - Classifies: greedy, jump game
  - Step 2B loads: `references/algorithms/jump-game.md` (atomic file, direct match)
  - Does NOT load the generic `references/algorithms/greedy-algorithms.md` when a specific file exists

### 19. Quick-match fallback — browse subdirectory
- **Query:** "teach me counting bits"
- **Expected:**
  - Classifies: bit manipulation, DP
  - Step 2B loads: `references/numeric/bit-manipulation.md` (quick-match hit)
  - If no quick-match, falls back to browsing `references/numeric/` subdirectory

## Functional — Learning Mode `MANUAL`

### 20. Full learning flow
- **Query:** "walk me through the merge intervals problem"
- **Expected:**
  - Step 2B loads at least one reference (interval scheduling or sorting) before Step 3
  - Starts with layman intuition question (does NOT give solution immediately)
  - Follows 6-section structure (Intuition -> Brute Force -> Optimal -> Alternatives -> Summary -> Interview)
  - Uses three-tier progressive hints when user is stuck
  - Ends with study notes and profile update

### 21. Weakness calibration
- **Setup:** Learner profile has "recurring" weakness on "hashmap insertion syntax"
- **Query:** "teach me two sum"
- **Expected:** Actively probes hashmap syntax during the session, adjusts scaffolding for that gap

### 22. Step 6 alternative technique loading
- **Setup:** Primary reference loaded in Step 2B is `references/techniques/array-techniques.md` (two pointers)
- **Query:** User asks about monotonic stack alternative for trapping rain water
- **Expected:** Loads `references/techniques/stack-queue-monotonic.md` before presenting the alternative

## Functional — Recall Mode `MANUAL`

### 23. Full recall flow
- **Query:** "test my recall on merge two sorted lists"
- **Expected:**
  - Enters Recall Mode (interviewer persona, neutral acknowledgments)
  - Steps: R1 (framing) -> R2 (reconstruction) -> R3 (edge cases) -> R4 (complexity) -> R5 (pattern) -> R6 (variation) -> R7 (debrief/scoring)
  - Provides verdict (Strong Pass / Pass / Borderline / Needs Work)
  - Suggests review schedule based on performance

## Reference Path Integrity `AUTO`

Run all three checks from the repo root. All must return zero results.

### 24. No stale bare paths
```bash
grep -rn 'references/[a-z]' egg/skills/leetcode-teacher/ \
  | grep -v 'references/\(frameworks\|data-structures\|techniques\|algorithms\|graphs\|math\|numeric\|problems\|ml\|teaching\)/' \
  | grep -v 'TODO.md'
```
- **Expected:** Zero results (all paths use subdirectory format)

### 25. No broken references
```bash
grep -roh 'references/[a-zA-Z_/-]*\.md' egg/skills/leetcode-teacher/ \
  | sort -u | while read ref; do
  [ -f "egg/skills/leetcode-teacher/$ref" ] || echo "BROKEN: $ref"
done
```
- **Expected:** Zero broken links

### 26. No empty files
```bash
find egg/skills/leetcode-teacher/references/ -name '*.md' -empty
```
- **Expected:** Zero empty files
