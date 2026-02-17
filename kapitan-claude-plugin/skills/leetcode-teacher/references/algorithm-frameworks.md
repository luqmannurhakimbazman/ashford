# Algorithm Frameworks

Meta-level thinking frameworks that sit above individual patterns. Based on labuladong's "framework-first" approach to algorithmic problem solving.

---

## The Enumeration Principle

**Core insight:** All algorithms are fundamentally brute-force enumeration made intelligent.

Every algorithm problem reduces to searching through a space of possibilities. The two challenges are:

1. **No omissions** — enumerate all possibilities without missing any (master the right framework)
2. **No redundancy** — avoid re-examining the same possibility twice (use information smartly)

### Computer Thinking vs Mathematical Thinking

- **Mathematical thinking:** Find a clever closed-form or shortcut (rare in interviews)
- **Computer thinking:** Systematically enumerate all candidates, prune intelligently

When stuck, ask: *"What am I enumerating? What is the full space of candidates?"* Then ask: *"Where am I doing redundant work? What information can I reuse?"*

This reframes optimization: you are not inventing a new algorithm — you are making brute-force smarter.

| Technique | How It Eliminates Redundancy |
|-----------|------------------------------|
| Memoization / DP | Cache results of overlapping subproblems |
| Sorting + Two Pointers | Skip pairs you can prove are suboptimal |
| Sliding Window | Reuse window state instead of recomputing from scratch |
| Greedy | Prove only one candidate at each step can be optimal |
| Binary Search | Eliminate half the search space per step |
| Backtracking + Pruning | Cut branches that cannot lead to valid solutions |

---

## General Interview Tips

Practical advice for approaching any algorithmic problem in an interview setting.

### Before Coding

- **Clarify assumptions.** Repeat the problem in your own words. Ask about input constraints (size, range, types), edge cases, and expected output format.
- **Ask about time/space constraints.** "Should I optimize for time or space?" and "What's the expected input size?" help determine the required complexity.
- **Work through examples.** Trace through the given example by hand before writing code. Create your own small test case.
- **Think before you code.** Spend 2-3 minutes planning your approach. Interviewers prefer a well-thought-out solution over a hastily coded one.

### While Coding

- **Check off-by-one errors.** These are the most common interview bugs. Double-check loop bounds, index calculations, and boundary conditions.
- **Test with examples after coding.** Mentally trace through your code with the example input to catch bugs before the interviewer does.

### When Stuck

- **Enumerate data structures as weapons.** When you don't see a path forward, systematically consider each data structure: would a hash map help? A heap? A stack? A trie? A sorted array with binary search?
- **Consider augmented data structures.** Combine two structures to get the best of both. Classic example: hash map + doubly-linked list = O(1) LRU cache.
- **Think about the brute force first.** Even if it's too slow, it clarifies what you're optimizing and often reveals the bottleneck.

### Functional vs Imperative

- **Pure functions** (no side effects, return values) are easier to reason about and test. Prefer them when possible.
- **Mutation** (modifying data in place) saves space but makes code harder to debug. Use it deliberately, not by accident.
- In interviews, choose whichever style lets you communicate your thinking most clearly.

---

## Binary Tree Centrality

**Labuladong's thesis:** Binary trees aren't just one data structure among many — they are THE mental model for algorithmic thinking.

**Why?** Two converging observations:

1. **All advanced data structures are tree extensions** — BSTs, heaps, tries, segment trees, graphs, B-trees are all variations on the binary tree. Master the tree, and every other structure becomes a familiar extension.
2. **All brute-force algorithms walk implicit trees** — backtracking walks a decision tree, BFS explores a state-space tree level by level, DP prunes a recursion tree, divide-and-conquer splits and combines at tree nodes.

This means binary tree traversal (pre-order, in-order, post-order) is the **universal skeleton** underlying all recursive algorithms. The only difference between backtracking, DP, merge sort, and tree problems is what code you place at each traversal position.

When a learner struggles with any recursive problem, bring them back to tree thinking: *"Draw the recursion tree. What does each node represent? Where do you make decisions? Where do you combine results?"*

For comprehensive coverage of how each data structure relates to trees, see `data-structure-fundamentals.md`.

---

## Recursion as Tree Traversal

**One perspective:** All recursion is tree traversal. Every recursive function implicitly walks a tree where each node is a function call and children are the recursive subcalls.

### Two Thinking Modes

#### Mode 1: Traversal (Backtracking Style)

Walk the tree with external state. Collect information as you go.

```python
result = []  # External state

def traverse(node):
    if not node:
        return
    # PRE-ORDER: make a choice (entering node)
    result.append(node.val)
    traverse(node.left)
    traverse(node.right)
    # POST-ORDER: undo the choice (leaving node)
    result.pop()
```

**Key rule:** Pre-order position = make choice. Post-order position = undo choice.

Use this mode when: the answer requires accumulating state along a path (backtracking, path enumeration).

#### Mode 2: Decomposition (DP Style)

Each subtree returns a value. Parent combines children's answers.

```python
def solve(node):
    if not node:
        return 0  # Base case
    left_result = solve(node.left)
    right_result = solve(node.right)
    # POST-ORDER: combine results from children
    return combine(left_result, right_result, node.val)
```

Use this mode when: the answer can be built by combining independent sub-answers (DP, divide and conquer).

### The Sorting Insight

- **Quick Sort = pre-order traversal:** Partition (make decision) first, then recurse on halves. The "work" — choosing where each element belongs relative to the pivot — happens **before** the recursive calls.
- **Merge Sort = post-order traversal:** Recurse on halves first, then merge (combine results). The "work" — interleaving two sorted halves — happens **after** the recursive calls return.

This is not a metaphor — the call structure is literally a pre/post-order tree walk.

```python
# Quick sort: pre-order — partition THEN recurse
def quick_sort(arr, lo, hi):
    if lo >= hi: return
    pivot = partition(arr, lo, hi)   # Work happens HERE (pre-order)
    quick_sort(arr, lo, pivot - 1)
    quick_sort(arr, pivot + 1, hi)

# Merge sort: post-order — recurse THEN merge
def merge_sort(arr, lo, hi):
    if lo >= hi: return
    mid = (lo + hi) // 2
    merge_sort(arr, lo, mid)
    merge_sort(arr, mid + 1, hi)
    merge(arr, lo, mid, hi)          # Work happens HERE (post-order)
```

**Why this matters for problem solving:** When you see a divide-and-conquer problem, ask: "Do I need to make a decision before recursing (pre-order/quick sort pattern) or combine results after recursing (post-order/merge sort pattern)?" This instantly narrows your approach.

For full implementations of all 10 sorting algorithms including non-comparison sorts, see `sorting-algorithms.md`.

---

## Recursion Interview Tips

Practical guidance for recursion problems in interviews.

### Base Cases

The number of base cases should match the depth of recursive calls. If you recurse on `n-1` and `n-2` (like Fibonacci), you need base cases for both `n=0` and `n=1`.

### Stack Overflow Awareness

- **Default recursion limits:** Python = 1000, Java ~5000-10000 (stack size dependent), C++ ~10000-50000.
- If N can be large (>1000 in Python), consider converting to an **iterative** solution with an explicit stack.
- Tail-call optimization eliminates stack growth for tail-recursive functions, but only Scheme/Haskell/Scala guarantee it. Python, Java, and C++ do **not** optimize tail calls.

### Recursion for Common Problem Types

- **Permutations/combinations:** Recursion with backtracking is the standard approach. See `references/brute-force-search.md` for the 9-variant framework.
- **Tree problems:** Recursion maps directly to tree traversal. Choose pre-order, in-order, or post-order based on when you need to process the node.
- **DP problems:** Start with recursive (top-down) solution, add memoization, then optionally convert to bottom-up.

### Corner Cases

- `n = 0` (empty input — what should the recursion return?)
- `n = 1` (single element — does the recursive case handle it, or do you need a separate base case?)

### Memoization: The Bridge from Recursion to DP

If your recursive solution has **overlapping subproblems** (the same inputs are computed multiple times), add a cache. This is the conceptual bridge between recursion and dynamic programming:

1. Write the recursive solution (correct but potentially exponential)
2. Identify overlapping subproblems (same arguments → same result)
3. Add memoization (cache results by arguments)
4. Optionally convert to bottom-up DP (fill table iteratively)

### Essential & Recommended Practice Questions

| Problem | Difficulty | Key Technique |
|---------|-----------|---------------|
| Subsets (78) | Medium | Backtracking or bit manipulation |
| Combinations (77) | Medium | Backtracking with start index |
| Generate Parentheses (22) | Medium | Backtracking with validity constraint |
| Permutations (46) | Medium | Backtracking with used set |
| Letter Combinations of a Phone Number (17) | Medium | Backtracking with mapping |
| Sudoku Solver (37) | Hard | Backtracking with constraint checking |

---

### Choosing a Mode

| Signal | Mode | Why |
|--------|------|-----|
| "Find all paths / combinations / permutations" | Traversal | Need to track the path as you walk |
| "Count / minimize / maximize" | Decomposition | Combine sub-answers without tracking full path |
| "Can you reach a state?" with choices at each step | Traversal | Walk the decision tree |
| "What is the value of the optimal substructure?" | Decomposition | Each subproblem returns its optimal value |

---

## The Divide and Conquer Framework

### Why Divide and Conquer Works: The (a+b)² Insight

For superlinear problems, splitting the input reduces total work. Consider: `(a+b)² = a² + 2ab + b² > a² + b²`. The cross-term `2ab` is the work you **eliminate** by dividing. This is why merge sort (O(N log N)) beats insertion sort (O(N²)) — splitting removes the quadratic interaction between halves.

*Socratic prompt: "If sorting N elements takes N² comparisons with brute force, and you split into two halves of N/2, what's the total work for the halves? What happened to the 'cross' comparisons?"*

### D&C vs Plain Recursion

Not every recursive function is divide and conquer. True D&C must satisfy:

1. **Subproblems reduce complexity** — splitting must eliminate superlinear cross-work
2. **Subproblems are independent** — solving one half must not depend on the other (this is what separates D&C from DP)
3. **Combine step is efficient** — merging results must be cheaper than re-solving

Binary search is technically "decrease and conquer" — it discards one half entirely rather than solving both halves independently.

### Pre-Order vs Post-Order D&C

| Style | Work Happens | Example | Sorting Analogy |
|-------|-------------|---------|-----------------|
| Pre-order D&C | Before recursion (partition, then recurse) | Quick Sort, Quick Select | Partition first, halves are independent |
| Post-order D&C | After recursion (recurse, then merge) | Merge Sort, Closest Pair | Solve halves first, combine results |

```python
# Pre-order D&C: partition THEN recurse
def quick_select(arr, lo, hi, k):
    if lo == hi:
        return arr[lo]
    pivot = partition(arr, lo, hi)
    if k == pivot:
        return arr[k]
    elif k < pivot:
        return quick_select(arr, lo, pivot - 1, k)   # Only one side!
    else:
        return quick_select(arr, pivot + 1, hi, k)

# Post-order D&C: recurse THEN combine
def count_inversions(arr, lo, hi):
    if lo >= hi:
        return 0
    mid = (lo + hi) // 2
    left_inv = count_inversions(arr, lo, mid)
    right_inv = count_inversions(arr, mid + 1, hi)
    split_inv = merge_and_count(arr, lo, mid, hi)     # Combine step
    return left_inv + right_inv + split_inv
```

### The Master Theorem (Simplified)

For recurrences of the form `T(N) = a·T(N/b) + O(N^c)`:

| Condition | Result | Intuition |
|-----------|--------|-----------|
| c < log_b(a) | T(N) = O(N^(log_b(a))) | Subproblems dominate — tree has many leaves |
| c = log_b(a) | T(N) = O(N^c · log N) | Work balanced across levels |
| c > log_b(a) | T(N) = O(N^c) | Combine step dominates — root does most work |

**Common cases:**
- Merge sort: `T(N) = 2T(N/2) + O(N)` → a=2, b=2, c=1. `c = log₂(2) = 1` → **O(N log N)**
- Binary search: `T(N) = T(N/2) + O(1)` → a=1, b=2, c=0. `c = log₂(1) = 0` → **O(log N)**
- Karatsuba multiplication: `T(N) = 3T(N/2) + O(N)` → a=3, b=2, c=1. `c < log₂(3) ≈ 1.58` → **O(N^1.58)**

*Socratic prompt: "Merge sort splits into 2 subproblems of size N/2 and does O(N) merge work. What are a, b, and c? Which Master Theorem case applies?"*

---

## The Sliding Window Framework

All sliding window problems answer three questions:

1. **Q1: When do I expand the window?** (What condition makes me move `right` forward?)
2. **Q2: When do I shrink the window?** (What condition makes me move `left` forward?)
3. **Q3: When do I update the result?** (After expanding? After shrinking? Both?)

### Annotated Template

```python
def sliding_window(s, t):
    from collections import defaultdict
    need = defaultdict(int)     # What we need to satisfy
    window = defaultdict(int)   # What the current window contains

    for c in t:
        need[c] += 1

    left = 0
    valid = 0          # Number of characters satisfying the condition
    result = float('inf')  # or 0, depending on min/max

    for right in range(len(s)):
        # --- EXPAND: add s[right] to window ---
        c = s[right]
        window[c] += 1
        if window[c] == need[c]:    # Q1: update validity check
            valid += 1

        # --- SHRINK: while window satisfies condition ---
        while valid == len(need):   # Q2: shrink condition
            # Q3: update result (here, BEFORE shrinking)
            result = min(result, right - left + 1)
            d = s[left]
            if window[d] == need[d]:
                valid -= 1
            window[d] -= 1
            left += 1              # Shrink from left

    return result
```

**Convention:** The window is `[left, right)` — left-closed, right-open. This means `right` points to the next element to add.

### Worked Example: Minimum Window Substring (LC 76)

Given `s = "ADOBECODEBANC"`, `t = "ABC"`:
- Expand right until window contains all of A, B, C → `"ADOBEC"` (valid)
- Shrink left while still valid → `"DOBEC"` (still valid) → `"OBEC"` (invalid, stop)
- Continue expanding right, repeat

Answer the three questions for every sliding window problem and the template writes itself.

---

## The DP Framework

> **Deep dive:** For comprehensive DP coverage including knapsack family, grid/path DP, interval DP, game theory DP, egg drop, and Floyd-Warshall, see `dynamic-programming-core.md`.

### Three Steps to Define Any DP

1. **Clarify the state** — what changes between subproblems? (Index, remaining capacity, last choice, etc.)
2. **Clarify the choices** — at each state, what decisions can you make?
3. **Define dp meaning** — `dp[state]` = the answer to the subproblem defined by that state

### Top-Down vs Bottom-Up

```python
# TOP-DOWN (memoized recursion) — think like mathematical induction
from functools import lru_cache

@lru_cache(maxsize=None)
def dp(i):
    if i == 0: return base_case
    # Assume dp(i-1), dp(i-2), ... are correct (inductive hypothesis)
    return best(dp(i - choice) for choice in choices)

# BOTTOM-UP (tabulation) — fill table from base cases
def dp_bottom_up(n):
    table = [0] * (n + 1)
    table[0] = base_case
    for i in range(1, n + 1):
        table[i] = best(table[i - choice] for choice in choices)
    return table[n]
```

**Mathematical induction analogy:** Top-down DP is exactly mathematical induction. You assume smaller subproblems are solved correctly (inductive hypothesis) and show how to combine them for the current problem (inductive step). The base case is the base case.

### Two Subsequence DP Templates

**Template 1: `dp[i]`** — one sequence, answer involves elements ending at or up to index `i`.

```python
# Example: Longest Increasing Subsequence
for i in range(n):
    for j in range(i):
        if nums[j] < nums[i]:
            dp[i] = max(dp[i], dp[j] + 1)
```

**Template 2: `dp[i][j]`** — two sequences (or one sequence with two pointers), relating prefixes `s[:i]` and `t[:j]`.

```python
# Example: Longest Common Subsequence
for i in range(1, m + 1):
    for j in range(1, n + 1):
        if s[i-1] == t[j-1]:
            dp[i][j] = dp[i-1][j-1] + 1
        else:
            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
```

**How to choose:** One sequence → try `dp[i]` first. Two sequences or comparing a sequence against itself → try `dp[i][j]`.

---

## The Backtracking Framework

### Decision Tree Model

Every backtracking problem is a walk through a decision tree with three components:

1. **Path** — choices made so far (the current partial solution)
2. **Choice list** — options available at the current node
3. **End condition** — when to record a result (leaf node or constraint met)

### Template

```python
def backtrack(path, choice_list):
    if meets_end_condition(path):
        result.append(path[:])
        return
    for choice in choice_list:
        # PRE-ORDER: make choice
        path.append(choice)
        # Recurse with updated choice list
        backtrack(path, updated_choice_list)
        # POST-ORDER: undo choice
        path.pop()
```

### Three Variants

The only difference between combination/permutation variants is how `choice_list` is updated:

| Variant | Problem Example | Start Parameter | Skip Duplicates? |
|---------|----------------|-----------------|-------------------|
| **Unique elements, no reuse** | Subsets, Combinations | `start = i + 1` | No |
| **Duplicate elements, no reuse** | Subsets II, Combination Sum II | `start = i + 1` | Yes: `if i > start and nums[i] == nums[i-1]: continue` |
| **Unique elements, with reuse** | Combination Sum | `start = i` (not `i + 1`) | No |

The `start` parameter controls whether elements before the current index are reconsidered. Sorting + skip-duplicate logic prevents equivalent branches when input has duplicates.

---

## The BFS Framework

### Why BFS Finds Shortest Paths

BFS explores all nodes at distance `d` before any node at distance `d+1`. This layer-by-layer expansion guarantees that the first time you reach a node is via the shortest path (in unweighted graphs).

### BFS vs Dijkstra

| Property | BFS | Dijkstra |
|----------|-----|----------|
| Data structure | Queue (FIFO) | Priority queue (min-heap) |
| Edge weights | All equal (unweighted) | Non-negative, possibly different |
| Visited tracking | Boolean `visited` set | `dist_to[node]` array (relax edges) |
| When to mark visited | When enqueuing | When dequeuing (popping from heap) |
| Time complexity | O(V + E) | O((V + E) log V) |

**Key insight:** Dijkstra is BFS generalized to weighted graphs. The priority queue ensures we always process the closest unvisited node, just as BFS's FIFO queue ensures we process by layer.

### Three Graph Traversal Modes

Labuladong identifies three distinct traversal patterns for graphs, each using `visited` differently:

**Mode 1: Traverse Nodes** — visit each node once. Standard DFS/BFS. Use `visited = set()`.

**Mode 2: Traverse Edges** — visit each edge once. Track `(from, to)` pairs in visited. Needed when parallel edges or edge-specific logic matters.

**Mode 3: Traverse Paths** — track the current path with `on_path = set()`, adding nodes on entry and removing on exit (backtracking). Essential for **cycle detection in directed graphs** (e.g., course schedule, topological sort).

The critical distinction: `visited` prevents revisiting nodes globally, while `on_path` tracks only the current recursion stack. A node can be `visited` but not `on_path` (explored via a different branch). This is why directed cycle detection needs `on_path` — `visited` alone cannot distinguish "already explored elsewhere" from "currently in a cycle."

**Complexity:** Graph traversal is O(V + E), not just O(V), because every edge is examined.

For code templates and detailed examples of all three modes, see `data-structure-fundamentals.md`.

---

## The State Machine Framework

### Stock Problem Generalization

All stock buy/sell problems can be modeled with one state machine:

```
dp[i][k][s]
  i = day (0 to n-1)
  k = max transactions remaining
  s = 0 (not holding) or 1 (holding)
```

### State Transitions

```
dp[i][k][0] = max(dp[i-1][k][0],           # rest (do nothing)
                   dp[i-1][k][1] + prices[i]) # sell

dp[i][k][1] = max(dp[i-1][k][1],           # rest (do nothing)
                   dp[i-1][k-1][0] - prices[i]) # buy (uses one transaction)
```

### Variants Table

| Problem | K | Special Rule | Simplification |
|---------|---|-------------|----------------|
| LC 121: Best Time to Buy and Sell Stock | 1 | — | Track min price, max profit |
| LC 122: Best Time II (unlimited) | infinity | — | Sum all positive diffs |
| LC 123: Best Time III | 2 | — | Full DP with k=2 |
| LC 188: Best Time IV | K (given) | — | General DP |
| LC 309: With Cooldown | infinity | Must wait 1 day after sell | `dp[i][0] = max(rest, dp[i-2][1] + price)` |
| LC 714: With Transaction Fee | infinity | Fee per transaction | Subtract fee on sell |

All six problems are the same framework with minor modifications to the transitions.

---

## The Greedy Framework

### The Optimization Hierarchy

Greedy is at the top of an optimization ladder. Each level skips more enumeration:

| Level | Technique | What It Enumerates | Time |
|-------|-----------|-------------------|------|
| 1 | Backtracking | All valid solutions | O(2^n) or worse |
| 2 | Dynamic Programming | All subproblem states (no redundancy) | O(n²) or O(n·k) |
| 3 | Greedy | One choice per step (no enumeration) | O(n) or O(n log n) |

Each level requires a stronger problem property: DP needs optimal substructure + overlapping subproblems. Greedy needs the **greedy choice property** — at each step, the locally optimal choice is part of some globally optimal solution.

### When Greedy Works vs Fails

**Works (greedy choice property holds):**
- Fractional knapsack — take the highest value/weight ratio first
- Interval scheduling — pick the earliest-ending non-overlapping interval
- Huffman coding — merge the two lowest-frequency nodes

**Fails (must use DP instead):**
- 0/1 knapsack — can't take fractions, local best can lead to global suboptimal
- Longest increasing subsequence — greedy "take the largest increase" doesn't work
- Edit distance — local character matches don't guarantee global minimum

*Socratic prompt: "Why does greedy work for fractional knapsack but not 0/1 knapsack? What specific property breaks?"*

### Proof Techniques

When you claim greedy works, you need a proof sketch. Two standard approaches:

**Exchange argument:** Assume an optimal solution that differs from greedy. Show you can "exchange" a non-greedy choice for the greedy one without worsening the result. Repeat until the optimal solution matches greedy.

**Stay-ahead argument:** Show that after each step, greedy's partial solution is at least as good as any other algorithm's partial solution. By induction, greedy stays ahead through the final step.

*Socratic prompt: "For interval scheduling, if the greedy picks the earliest-ending interval but the optimal solution picks a different one, can you swap them? Does the solution get worse?"*

---

## Bidirectional BFS

An optimization for standard BFS when you know both the start and target states.

### Core Idea

Instead of searching from start to target (expanding a potentially huge frontier), search from **both ends simultaneously** and stop when the frontiers meet. This reduces the search space from O(b^d) to O(b^(d/2)), where `b` is the branching factor and `d` is the distance.

### Template

```python
from collections import deque

def bidirectional_bfs(start, target, get_neighbors):
    if start == target:
        return 0

    # Two frontiers as sets (for O(1) membership check)
    front = {start}
    back = {target}
    visited = {start, target}
    steps = 0

    while front and back:
        steps += 1
        # Always expand the SMALLER frontier for balance
        if len(front) > len(back):
            front, back = back, front

        next_front = set()
        for node in front:
            for neighbor in get_neighbors(node):
                if neighbor in back:
                    return steps          # Frontiers met!
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_front.add(neighbor)
        front = next_front

    return -1  # No path exists
```

### When to Use

- You must know the **target state** in advance (not just "find any goal")
- The branching factor is large (e.g., Word Ladder with 26 possible letter changes per position)
- Standard BFS times out due to exponential frontier growth

**Word Ladder example:** Changing "hit" → "cog" with dictionary lookups. Standard BFS explores O(26^d) states from one end. Bidirectional BFS explores O(26^(d/2)) from each end — dramatically faster.

*Socratic prompt: "If the shortest path is 6 steps and each node has 10 neighbors, how many nodes does standard BFS explore vs bidirectional BFS? Which grows faster?"*

---

## Practical Complexity Analysis

### Input Size → Required Complexity

Use this table to reverse-engineer the expected algorithm complexity from the problem's input constraints:

| Input Size N | Max Acceptable Complexity | Typical Algorithms |
|-------------|--------------------------|-------------------|
| N ≤ 10 | O(N!) or O(2^N) | Brute force, permutations |
| N ≤ 20 | O(2^N) | Backtracking with pruning, bitmask DP |
| N ≤ 100 | O(N³) | Floyd-Warshall, cubic DP |
| N ≤ 1,000 | O(N²) | Quadratic DP, nested loops |
| N ≤ 100,000 | O(N log N) | Sorting-based, divide and conquer |
| N ≤ 1,000,000 | O(N) | Linear scan, hash table, sliding window |
| N ≤ 10^8 | O(N) (tight) | Simple iteration only |
| N ≤ 10^12 | O(log N) or O(√N) | Binary search, math |

### The 10^8 Operations Rule

Most online judges allow roughly **10^8 simple operations per second** (with a typical 1-2 second time limit). Use this to sanity-check your approach:

- Your algorithm is O(N²) and N = 10,000? → 10^8 operations ✓
- Your algorithm is O(N²) and N = 100,000? → 10^10 operations ✗ (need O(N log N))

*Socratic prompt: "The problem says N ≤ 10^5. Your current solution is O(N²). Will it pass? How many operations is that?"*

### Recursive Complexity Formula

For recursive algorithms: **Total work = (number of subproblems) × (work per subproblem)**

- Binary search: 1 subproblem × O(1) work per level × O(log N) levels = **O(log N)**
- Merge sort: 2 subproblems × O(N) merge per level × O(log N) levels = **O(N log N)**
- Naive Fibonacci: 2 subproblems × O(1) per call × O(2^N) total calls = **O(2^N)** (overlapping!)
- Memoized Fibonacci: N unique subproblems × O(1) per call = **O(N)**

### Common Submission Pitfalls

| Pitfall | Effect | Fix |
|---------|--------|-----|
| Print/log statements in loops | 10-100x slowdown | Remove all debug output before submitting |
| Pass-by-value for large objects | Copies entire data structure per call | Pass by reference or use indices |
| String concatenation in loops | O(N²) due to immutable string copies | Use `list.append()` + `''.join()` |
| Unnecessary deep copies | O(N) per copy, often inside O(N) loop | Copy only when mutation is needed |
| Recursive without memoization | Exponential for overlapping subproblems | Add `@lru_cache` or memo dict |

### Space Complexity Essentials

- **Recursion depth** counts as O(depth) space (call stack). Binary tree DFS = O(height), not O(N)
- **Input space** (the input itself) is not counted in auxiliary space complexity
- **In-place** means O(1) auxiliary space — you can modify the input but can't allocate proportional extra space

*Socratic prompt: "Your recursive solution has depth O(N) and allocates an O(N) array at each level. What's the total space complexity?"*

---

## Attribution

The frameworks in this file are inspired by and adapted from labuladong's algorithmic guides (labuladong.online), which provide a comprehensive framework-first approach to algorithm learning. The enumeration principle, recursion-as-tree-traversal model, and framework templates have been restructured and annotated for Socratic teaching use. The divide and conquer framework draws from the "divide-and-conquer" article, the greedy hierarchy from the "greedy" article, bidirectional BFS from the "bfs-framework" article, and complexity analysis from the "complexity-analysis" article.
