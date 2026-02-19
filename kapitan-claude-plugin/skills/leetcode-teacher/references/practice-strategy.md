# Deliberate Practice Strategy

A structured methodology for practicing algorithmic problem solving effectively. Adapted from battle-tested competitive programming advice (SuperJ6's Codeforces guide) and calibrated for LeetCode/interview preparation contexts.

---

## 1. The Goal of Practice

**Core objective:** Come across as many subtle ideas and concepts as quickly as possible, and learn to intuitively recognize when to apply them.

Practice is NOT about:
- Grinding the same problem type until it feels easy
- Spending hours on a single problem to prove you can solve it alone
- Memorizing solutions

Practice IS about:
- **Productive discomfort** — actively generating new insights under time pressure
- **Breadth of exposure** — seeing many problems across different patterns
- **Active insight generation** — every minute should produce new observations, not recycled thoughts

### Discomfort vs. Confusion

There's a critical difference:
- **Productive discomfort:** You have ideas, you're trying combinations, you're making small progress — this is the learning zone
- **Unproductive confusion:** You have no idea what to do, you're re-reading the problem hoping something clicks — this is wasted time

**Rule:** If you're actively generating new observations, keep going. If you're stuck repeating the same thoughts, it's time to seek a hint or read the editorial. Learning from 2-3 problems in the same time beats 90 minutes stuck on one.

---

## 2. Difficulty Calibration

### The 30-40% Rule

**Target a difficulty level where you can solve ~30-40% of problems independently.** This is the sweet spot where:
- You can't solve most problems on your own (so you're learning new things)
- You CAN understand the solution once you see it (so the learning sticks)
- You're maximizing new concepts per hour of practice

### LeetCode Difficulty Mapping

| Your Level | Primary Practice Zone | Solve Rate Target | Progression Signal |
|------------|----------------------|-------------------|-------------------|
| **Beginner** | Easy (top 50) | ~30-40% of Easy | Consistently solve Easy in <15 min |
| **Developing** | Easy-Medium mix | ~30-40% of Medium | Recognize core patterns (two pointers, hash map, sliding window) |
| **Intermediate** | Medium | ~30-40% of Medium | Solve familiar Medium in <20 min, attempt Hard |
| **Advanced** | Medium-Hard mix | ~30-40% of Hard | Identify optimal approach for most Medium without hints |
| **Interview-Ready** | Hard + timed practice | ~30-40% of Hard | Solve Medium in <25 min under interview conditions |

### When to Level Up

Move to the next difficulty tier when:
- You're solving **50%+ of problems** at your current level independently
- The lower end of your range feels boring or routine
- You've done roughly **25-40 problems** at the current level

### Problem Selection

- **Mix topics** — don't practice by topic. Interleaving forces you to identify the pattern yourself, which is exactly the skill interviews test. (This aligns with the interleaving principle in `references/learning-principles.md`.)
- **Prefer recent/popular problems** over obscure ones — they better represent current interview trends
- **Use curated lists for structure:** NeetCode 150 or Blind 75 provide good coverage, but shuffle the order rather than going topic-by-topic
- **After completing a curated list,** move to random LeetCode problems in your difficulty range

---

## 3. Session Design

### Time Budget Per Problem

**Spend 15-20 minutes thinking before seeking hints.** After that:
- If you're still having new ideas → keep going
- If you're recycling the same thoughts → read a hint or the editorial
- If a hint gives you new ideas → think again before reading more

**The exhaustion test:** You should feel mentally tired before giving up. If you quit because you "don't feel like thinking harder," you're not practicing at full intensity. But if you've genuinely run out of angles, move on — learning from the solution of 2-3 problems beats being stuck on one for 90 minutes.

### 90-Minute Session Template

| Phase | Duration | Activity |
|-------|----------|----------|
| **Warm-up** | 10 min | Solve one easy problem or re-solve a previously learned problem from memory |
| **Main problem** | 30 min | Work through a problem at your target difficulty (15-20 min thinking + implementation) |
| **Second problem** | 30 min | Another problem at target difficulty |
| **Reflection** | 10 min | Review both problems: key insights, what you'd do differently, pattern connections |
| **Implementation review** | 10 min | Review your code for conciseness, read others' solutions for implementation tricks |

### The Implementation Rule

**Always implement every problem.** This is non-negotiable because:
1. You often discover you didn't understand the details as well as you thought
2. Turning ideas into code is a separate skill that needs practice
3. You remember solutions better when you've coded them

**Implementation order:**
1. If you have an idea (even uncertain) → implement before reading the editorial
2. If you read the editorial → implement yourself before looking at others' code
3. Looking at others' implementation is a last resort

### Timing Yourself

**Time your implementations.** This builds the speed needed for real interviews and keeps you focused. Track your times — you should see improvement for a fixed difficulty level over weeks.

---

## 4. Post-Problem Reflection Protocol

After every problem (solved or not), spend 2-3 minutes on structured reflection:

### Key Insight Identification
> "What was the ONE idea that unlocked this problem?"

Name it specifically. Not "use DP" but "the state only depends on the previous row, so we can compress to 1D." This is the insight you need to recognize in future problems.

### Mental Trap Analysis
> "Where did my thinking go wrong or stall? What assumption was I making that blocked me?"

Common traps to watch for:
- Fixating on the first approach instead of stepping back
- Missing a constraint that simplifies the problem
- Overcomplicating when a simpler observation exists
- Not drawing out examples early enough

### Generalization Practice
> "If I were teaching someone to solve problems like this, what would I tell them to look for?"

This is the most valuable reflection. You're building your personal problem-solving heuristics — not just categorizing by algorithm, but identifying the **thinking patterns** that lead to solutions.

### Editorial Comparison

When you solve a problem, still read the editorial or top solutions:
- Is there a cleaner approach you missed?
- Are there implementation tricks that make the code more concise?
- Does the editorial use a different framing that's more generalizable?

---

## 5. Problem-Solving Thinking Checklist

A condensed set of meta-strategies to run through when stuck. These complement the pattern-specific frameworks in `references/algorithm-frameworks.md` and `references/problem-patterns.md` — those tell you WHAT technique to use; this checklist helps you THINK your way to discovering which technique applies.

### Representation & Reframing

1. **Change the lens** — View the problem as a graph (pairs → edges), as geometry (coordinates), as bits (binary representation), or as intervals. A different representation often makes the solution obvious.
2. **Rewrite conditions** — Express constraints as formulas, expand them, transform them. Try `|x|` → `±x`, difference arrays, prefix sums, complements. The right algebraic form often reveals structure.
3. **Simplify first** — Reduce to a simpler case (binary values, smaller constraints, 1D instead of 2D), solve that, then generalize. If you can solve the simple version, the full version usually extends naturally.
4. **Reverse the process** — Work backwards, look at the inverse, swap the order of operations. Counting the complement is often easier than counting directly.

### Structure & Properties

5. **Find invariants** — What stays the same after each operation? What property does every valid state share? Invariants constrain the search space and often reveal the answer directly.
6. **Spot monotonicity** — Is there a sorted order, a concavity, a property that only increases? Monotonicity enables binary search, two pointers, greedy, and DP optimizations.
7. **Use constraints as clues** — Unusual constraints (e.g., `n ≤ 20` → bitmask, `n ≤ 10^5` → O(n log n)) hint at the intended approach. If something in the problem statement looks odd, it's probably the key.
8. **Map to canonical form** — Can you normalize equivalent representations? Sorting, hashing to a standard form, or reducing to a known problem structure can simplify both the logic and the counting.

### Decomposition & Scope

9. **Reuse information** — What are you recomputing? Can you cache it (DP), extend it (sliding window, two pointers), or process in dependency order (topological sort, sweep line)?
10. **Make one choice, then recurse** — Fix the first element's assignment, reduce to a smaller instance. This is the heart of greedy, DP, backtracking, and divide-and-conquer. Ask: "What do I know for sure about the optimal first move?"
11. **Break into independent parts** — Can the problem decompose into non-interacting subproblems? Solve x and y coordinates separately, process disjoint intervals independently, or sweep one dimension while querying another.
12. **Change the scope** — Instead of thinking about the entire array, focus on one element's contribution to the answer. Or instead of processing online, think about all queries at once (offline). Zooming in or out often simplifies.

### Practical Tactics

13. **Draw it out and trace examples** — Visualize the problem. Trace through test cases by hand with your current best idea. Small examples reveal patterns that abstract thinking misses.
14. **Don't overcomplicate** — If your approach has too many edge cases or steps, it's probably wrong. Almost every interview problem has a clean, elegant solution. Step back and look for it.
15. **Write down every observation** — Even small, seemingly irrelevant observations. Progress comes from combining observations, and you can't combine what you haven't recorded. If you're not writing new things, you're not thinking new things.

> **Cross-references:** For pattern-specific recognition signals, see `references/problem-patterns.md`. For framework-level thinking (enumeration principle, recursion-as-tree, traversal vs. decomposition), see `references/algorithm-frameworks.md`. For data-structure selection, see `references/data-structure-fundamentals.md`.

---

## 6. Long-Term Study Planning

### Weekly Schedule Template

For consistent improvement, aim for **4-5 practice sessions per week:**

| Day | Session Type | Duration | Focus |
|-----|-------------|----------|-------|
| **Mon** | New problems | 90 min | Standard session (Section 3 template) |
| **Tue** | New problems | 90 min | Standard session, different patterns from Mon |
| **Wed** | Recall practice | 60 min | Re-solve 3-5 previously learned problems from memory (no notes) |
| **Thu** | New problems | 90 min | Standard session |
| **Fri** | Rest or light review | 30 min | Read editorials, review algorithm articles, or skip |
| **Sat** | Timed practice | 90 min | Simulate interview: 2 problems in 45 min (see below) |
| **Sun** | New problems + review | 90 min | Standard session + weekly reflection |

### Environment & Habits

- **Consistent time and place** — Practice at the same time and location each day. This builds the habit of deep focus.
- **Eliminate distractions** — Close social media, messaging apps, and anything that fragments attention during your practice block.
- **Background processing** — For hard problems you couldn't solve, memorize the problem statement at the start of the day and think about it during downtime (commute, exercise, shower). The subconscious often finds angles that forced thinking misses.
- **Minimum 90 minutes per session** — Shorter sessions don't allow enough depth. Longer is better if you can maintain focus.

### Timed Interview Simulation

Once per week, simulate real interview conditions:

| Format | Setup |
|--------|-------|
| **Duration** | 45 minutes total |
| **Problems** | 2 problems (1 Medium + 1 Medium/Hard) |
| **Rules** | No hints, no editorial, no looking up syntax. Talk through your approach out loud. |
| **After** | Score yourself: Did you finish? Were solutions correct? How was your communication? |

This is where you practice performing under pressure — separate from the learning sessions where you're building knowledge.

### Learning New Techniques

When you encounter an algorithm or concept you don't know (in an editorial or during practice):

1. **Learn it immediately** — Don't save it for later. Find an explanation (references in this skill, cp-algorithms, or a tutorial), understand it, implement it in the context of the current problem.
2. **Do 2-3 focused problems** using the new technique to build basic fluency.
3. **Return to mixed practice** — Don't keep grinding the same topic. If the technique is important, you'll see it again naturally, and recognizing it in a mixed context is the real skill.

This aligns with the interleaving principle: topic-grinding gives a false sense of mastery because you already know which tool to use. Mixed practice forces pattern recognition.

### Progression Milestones

| Stage | Target | Key Skills | Duration Estimate |
|-------|--------|------------|-------------------|
| **Foundation** | Solve most Easy problems | Arrays, strings, hash maps, basic recursion, sorting | 4-8 weeks |
| **Core Patterns** | Solve 30-40% of Medium | Two pointers, sliding window, BFS/DFS, binary search, basic DP | 8-16 weeks |
| **Advanced** | Solve 30-40% of Hard | Advanced DP, graph algorithms, segment trees, complex greedy | 16-32 weeks |
| **Interview-Ready** | Solve Medium in <25 min timed | All patterns + speed + communication + edge case awareness | Ongoing maintenance |

**Important:** These are rough guides, not deadlines. Progress is nonlinear — you'll have plateaus and breakthroughs. The key metric is not "how fast am I moving" but "am I practicing effectively every session."

---

## 7. Implementation Discipline

### Planning Before Coding

Before writing any code:
1. **Plan in chunks** — Mentally divide your solution into 3-5 logical blocks (initialization, main loop, edge handling, result construction). Have each chunk clear before you start typing.
2. **Know your data structures** — Decide what you're storing and how before writing the first line.
3. **Identify the tricky part** — Which chunk has the most room for error? Plan that one most carefully.

### Writing Clean Code

- **Concise but readable** — Short variable names are fine in interview code, but the logic flow should be clear
- **Don't rewrite the same thing twice** — If you find yourself copy-pasting with small changes, step back and restructure
- **If you keep rewriting, stop and re-plan** — Repeated edits signal that you didn't fully understand the approach before coding

### Debugging Checklist

When your solution gives wrong output:

1. **Print intermediate state** — Add print statements at key points to see where actual values diverge from expected
2. **Binary search for the bug** — Find the first point in execution where output doesn't match expectation. The bug is between the last correct state and the first incorrect one.
3. **Trace through a small example by hand** — Follow your code line-by-line with a small input. The mistake is usually somewhere you were "sure" couldn't go wrong.
4. **Rewrite, don't patch** — If a section is broken and you can't see why after 2-3 minutes, rewrite it cleanly from your plan rather than adding small fixes on top of broken logic.

---

## 8. Motivation & Mindset

### Finding Fulfillment in Small Steps

Every problem solved and every session completed is genuine progress. Every new observation — even small ones that don't immediately lead to a solution — is one step closer to mastery. The compound effect of daily practice is enormous but invisible day-to-day.

### Discomfort = Learning

If practice feels comfortable, you're not learning efficiently. The productive struggle of working through problems just beyond your current ability is where growth happens. This is the "desirable difficulty" principle from learning science (see `references/learning-principles.md`).

### Track Your Progress

Maintain visibility into your improvement:
- The learner profile tracks weaknesses, session history, and verdict trajectories
- Review your profile periodically ("how am I doing?" triggers Profile Review Mode)
- Celebrate when weaknesses move from `recurring` → `improving` → `resolved`

### When You Feel Stuck

Plateaus are normal and temporary. If you feel stuck for multiple weeks:
1. **Check your practice intensity** — Are you truly exhausting yourself on each problem, or going through the motions?
2. **Check your reflection quality** — Are you doing structured post-problem reflection, or just moving to the next problem?
3. **Check your difficulty calibration** — Are you practicing at the right level (30-40% solve rate)?
4. **Try a different angle** — Sometimes working on a completely different pattern area breaks a plateau

---

## Attribution

Core practice methodology adapted from SuperJ6's ["How to Effectively Practice CP + Problem Solving Guide"](https://codeforces.com/blog/entry/116371) on Codeforces, with significant adaptations for LeetCode/interview preparation contexts: CF ratings mapped to Easy/Medium/Hard tiers, virtual contests adapted to timed practice sessions, upsolving adapted to recall mode, and the 28-item thinking checklist condensed to 15 items with cross-references to existing skill reference files.
