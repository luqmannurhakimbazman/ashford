# Algorithm Frameworks

Meta-level thinking frameworks that sit above individual patterns. Based on labuladong's "framework-first" approach to algorithmic problem solving.

## Table of Contents

- [The Enumeration Principle](#the-enumeration-principle)
- [General Interview Tips](#general-interview-tips)
- [Binary Tree Centrality](#binary-tree-centrality)
- [Recursion as Tree Traversal](#recursion-as-tree-traversal)
- [Recursion Interview Tips](#recursion-interview-tips)
- [Practical Complexity Analysis](#practical-complexity-analysis)
- [Technique-Specific Frameworks (Cross-References)](#technique-specific-frameworks)

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

For comprehensive coverage of how each data structure relates to trees, see `references/data-structures/data-structure-fundamentals.md`.

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

For full implementations of all 10 sorting algorithms including non-comparison sorts, see `references/algorithms/sorting-algorithms.md`.

---

### Choosing a Mode

| Signal | Mode | Why |
|--------|------|-----|
| "Find all paths / combinations / permutations" | Traversal | Need to track the path as you walk |
| "Count / minimize / maximize" | Decomposition | Combine sub-answers without tracking full path |
| "Can you reach a state?" with choices at each step | Traversal | Walk the decision tree |
| "What is the value of the optimal substructure?" | Decomposition | Each subproblem returns its optimal value |

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

- **Permutations/combinations:** Recursion with backtracking is the standard approach. See `references/algorithms/brute-force-search.md` for the 9-variant framework.
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

## Technique-Specific Frameworks

The following technique frameworks have been split into dedicated files for focused loading:

| Technique | Reference |
|-----------|-----------|
| Sliding window | `references/algorithms/sliding-window.md` |
| Backtracking | `references/algorithms/backtracking.md` |
| BFS + Bidirectional BFS | `references/algorithms/bfs-framework.md` |
| DP framework | `references/algorithms/dp-framework.md` |
| Divide and conquer | `references/algorithms/divide-and-conquer.md` |
| State machine (stock problems) | `references/algorithms/state-machine.md` |
| Greedy framework | `references/algorithms/greedy-algorithms.md` |

---

## Attribution

The frameworks in this file are inspired by and adapted from labuladong's algorithmic guides (labuladong.online), which provide a comprehensive framework-first approach to algorithm learning. The enumeration principle, recursion-as-tree-traversal model, and framework templates have been restructured and annotated for Socratic teaching use.
