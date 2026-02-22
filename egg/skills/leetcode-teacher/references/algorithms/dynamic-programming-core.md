# Dynamic Programming Core

Comprehensive DP reference covering the framework principles, subsequence/string DP, and shortest path DP. For the high-level DP framework overview, see `references/frameworks/algorithm-frameworks.md`.

## Table of Contents

- [1. DP Framework Principles](#1-dp-framework-principles)
- [2. Subsequence & String DP](#2-subsequence--string-dp)
- [Technique-Specific DP (Cross-References)](#technique-specific-dp-cross-references)
- [9. Shortest Path DP: Floyd-Warshall](#9-shortest-path-dp-floyd-warshall)
- [Practice Questions](#practice-questions)

---

## Quick Reference Table

| DP Family | Reference |
|-----------|-----------|
| Framework & Principles | This file (Section 1) |
| Subsequence & String DP | This file (Section 2) |
| Knapsack Problems | `references/algorithms/knapsack.md` |
| Grid & Path DP | `references/algorithms/grid-dp.md` |
| Game Theory DP | `references/algorithms/game-theory-dp.md` |
| Interval DP + Egg Drop | `references/algorithms/interval-dp.md` |
| House Robber & Stock | `references/algorithms/stock-problems.md` |
| Shortest Path DP (Floyd-Warshall) | This file (Section 9) |
| Subsequence DP (quick-reference) | `references/algorithms/subsequence-dp.md` |
| DP Framework (quick-reference) | `references/algorithms/dp-framework.md` |

---

## 1. DP Framework Principles

### The Three-Step DP Process

Every DP problem is solved by answering three questions in order:

1. **Clarify the state** -- what variables change between subproblems? (index, remaining capacity, last choice, etc.)
2. **Clarify the choices** -- at each state, what decisions can you make?
3. **Define `dp` meaning** -- `dp[state]` = the answer to the subproblem defined by that state

Then find the base case and write the state transition equation.

*Socratic prompt: "What changes between subproblems? That's your state. What decisions do you make at each state? That's your choice list."*

### Overlapping Subproblems & Memoization

The hallmark of DP: the brute-force recursion recomputes the same subproblems exponentially many times.

**Fibonacci example:** Naive recursion computes `fib(18)` multiple times within `fib(20)`. The recursion tree has O(2^n) nodes.

**Fix:** Add a memo (hash map or array) to cache results. Each subproblem is solved exactly once.

```python
# Top-down with memoization
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
```

**Time complexity drops from O(2^n) to O(n)** -- each of the n subproblems is solved once in O(1).

### Top-Down vs Bottom-Up

| Approach | Direction | Implementation | When to Use |
|----------|-----------|---------------|-------------|
| **Top-down** (memoized recursion) | Start from target, recurse to base | `@lru_cache` or memo dict | Natural when recursion structure is clear |
| **Bottom-up** (tabulation) | Start from base cases, build up | Fill `dp` array iteratively | When you need space optimization or iterative control |

```python
# Bottom-up tabulation for Coin Change (LC 322)
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # Base case: 0 coins needed for amount 0
    for i in range(1, amount + 1):
        for coin in coins:
            if i - coin >= 0 and dp[i - coin] != float('inf'):
                dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

**Mathematical induction analogy:** Top-down DP IS mathematical induction. You assume smaller subproblems are correct (inductive hypothesis) and show how to combine them (inductive step). The base case is the base case.

*Socratic prompt: "Can you write the top-down version first? Once that works, can you convert it to bottom-up by reversing the direction?"*

### Space Optimization (Rolling Array)

When the state transition only depends on the previous row/column, you don't need the full DP table.

**Pattern:** If `dp[i]` only depends on `dp[i-1]` (and possibly `dp[i-2]`), replace the array with two variables:

```python
# Fibonacci: O(1) space
def fib(n):
    if n <= 1:
        return n
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr
    return prev1
```

**For 2D DP:** If `dp[i][j]` only depends on `dp[i-1][...]`, use a single 1D array and update it carefully (usually right-to-left for 0-1 knapsack, left-to-right for complete knapsack).

> **Tip:** Always check which previous states your transition actually needs before allocating the full table.

*Socratic prompt: "Look at your state transition. Which previous states does dp[i] actually need? Can you keep only those?"*

### Common DP Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Wrong base case | Off-by-one errors, wrong answer for small inputs | Trace through the smallest cases by hand |
| Wrong state definition | DP doesn't capture enough info to make transitions | Add dimensions (e.g., add a "holding stock" boolean) |
| Missing states | Some subproblems not covered | Ensure every reachable state has a transition |
| Integer overflow in memo | `dp[i] + dp[j]` overflows | Use `float('inf')` or check before adding |
| Forgetting to memoize | Top-down runs in exponential time | Always add `@lru_cache` or memo check |

---

## 2. Subsequence & String DP

### Two Fundamental Templates

**Template 1: `dp[i]` (single sequence)** -- answer involves elements ending at or up to index `i`.

**Template 2: `dp[i][j]` (two sequences or two pointers)** -- compares prefixes `s[:i]` and `t[:j]`, or a single sequence with two endpoints.

| Signal | Template | Example |
|--------|----------|---------|
| One array, "longest subsequence ending at i" | `dp[i]` | LIS (300) |
| Two strings, "similarity/distance/matching" | `dp[i][j]` | LCS (1143), Edit Distance (72) |
| One string, palindrome-related | `dp[i][j]` | Longest Palindromic Subsequence (516) |
| One string, "can it be segmented" | `dp[i]` (boolean) | Word Break (139) |

### Longest Increasing Subsequence (LIS) -- LC 300

**State:** `dp[i]` = length of LIS ending at `nums[i]`.

**Transition:** For each `j < i` where `nums[j] < nums[i]`: `dp[i] = max(dp[i], dp[j] + 1)`.

```python
def length_of_lis(nums):
    n = len(nums)
    dp = [1] * n
    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)  # O(n^2)
```

**O(n log n) optimization -- Patience Sorting:**

```python
import bisect

def length_of_lis_fast(nums):
    tails = []
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)
```

*Socratic prompt: "Why does replacing tails[pos] with a smaller value not break the LIS? What invariant does the tails array maintain?"*

**Related:** Russian Doll Envelopes (354) -- sort by width ascending, height descending for same width, then find LIS on heights.

### Longest Common Subsequence (LCS) -- LC 1143

**State:** `dp[i][j]` = length of LCS of `s1[:i]` and `s2[:j]`.

```python
def longest_common_subsequence(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
```

*Socratic prompt: "When s1[i-1] != s2[j-1], why do we take the max of skipping from either string?"*

### Edit Distance -- LC 72

**State:** `dp[i][j]` = minimum operations to convert `s[:i]` to `t[:j]`.

```python
def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Delete
                    dp[i][j - 1],      # Insert
                    dp[i - 1][j - 1]   # Replace
                )
    return dp[m][n]
```

### Maximum Subarray -- LC 53

**State:** `dp[i]` = max sum of subarray ending at index `i`.

```python
def max_subarray(nums):
    max_sum = curr_sum = nums[0]
    for i in range(1, len(nums)):
        curr_sum = max(nums[i], curr_sum + nums[i])
        max_sum = max(max_sum, curr_sum)
    return max_sum
```

### Word Break -- LC 139

**State:** `dp[i]` = True if `s[:i]` can be segmented into dictionary words.

```python
def word_break(s, word_dict):
    word_set = set(word_dict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[n]
```

### Regular Expression Matching -- LC 10

**State:** `dp[i][j]` = True if `s[:i]` matches pattern `p[:j]`.

```python
def is_match(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for j in range(2, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[i][j] = dp[i][j - 2]
                if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                    dp[i][j] = dp[i][j] or dp[i - 1][j]
            elif p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    return dp[m][n]
```

### Subsequence DP: General Patterns

**Longest Palindromic Subsequence (LC 516):** `dp[i][j]` = LPS of `s[i..j]`. If `s[i] == s[j]`, extend by 2. Otherwise, `max(dp[i+1][j], dp[i][j-1])`. Note: fill diagonally or bottom-up since `dp[i]` depends on `dp[i+1]`.

**Minimum Insertions for Palindrome (LC 1312):** `len(s) - LPS(s)` = minimum insertions needed.

**Two Subsequence Templates Summary:**

| Template | When | Direction | Base |
|----------|------|-----------|------|
| `dp[i]` | Single sequence, ending at i | Left to right | `dp[i] = 1` or `dp[0] = base` |
| `dp[i][j]` | Two sequences or range `[i..j]` | Row by row or diagonal | `dp[i][i] = base` or `dp[0][j]`/`dp[i][0]` |

---

## Technique-Specific DP (Cross-References)

The following DP families have been split into dedicated files for focused loading:

| DP Family | Reference |
|-----------|-----------|
| Knapsack Problems (0-1, complete, bounded) | `references/algorithms/knapsack.md` |
| Grid & Path DP | `references/algorithms/grid-dp.md` |
| Game Theory DP | `references/algorithms/game-theory-dp.md` |
| Interval DP + Egg Drop | `references/algorithms/interval-dp.md` |
| House Robber & Stock Problems | `references/algorithms/stock-problems.md` |

---

## 9. Shortest Path DP: Floyd-Warshall

### All-Pairs Shortest Path

**State:** `dp[i][j]` = shortest path weight from node `i` to node `j`.

**Transition (the k-loop):** For each intermediate node `k`, check if going through `k` improves the path: `dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j])`.

```python
def floyd_warshall(n, edges):
    INF = float('inf')
    dp = [[INF] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 0
    for u, v, w in edges:
        dp[u][v] = w

    # The triple loop: k MUST be the outermost loop
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dp[i][k] != INF and dp[k][j] != INF:
                    dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j])

    return dp
```

**Why k must be outermost:** The DP builds up by considering paths that use only nodes `{0, 1, ..., k}` as intermediates.

**Complexity:** O(n^3) time, O(n^2) space. Suitable for `n <= 400` approximately.

**Comparison with other shortest path algorithms:**

| Algorithm | Problem Type | Negative Weights? | Complexity |
|-----------|-------------|-------------------|------------|
| BFS | Single-source, unweighted | N/A | O(V + E) |
| Dijkstra | Single-source, non-negative | No | O((V+E) log V) |
| Bellman-Ford | Single-source, any | Yes | O(VE) |
| Floyd-Warshall | All-pairs | Yes (no neg cycles) | O(V^3) |

*Socratic prompt: "When would you use Floyd-Warshall instead of running Dijkstra from every node? What if the graph has negative edges?"*

**Example problem:** Find the City With the Smallest Number of Neighbors at a Threshold Distance (LC 1334).

---

## Practice Questions

### Essential

| Problem | DP Family | Key Concept |
|---------|-----------|-------------|
| Climbing Stairs (70) | Framework | Base case + Fibonacci-style transition |
| Coin Change (322) | Knapsack (Complete) | Minimize coins, complete knapsack pattern |
| Longest Increasing Subsequence (300) | Subsequence | `dp[i]` = LIS ending at i, O(n log n) optimization |
| House Robber (198) | House Robber | No-two-adjacent recurrence |

### Recommended

| Problem | DP Family | Key Concept |
|---------|-----------|-------------|
| Jump Game (55) | Grid/Greedy | Can reach end? Track farthest reachable index |
| Unique Paths (62) | Grid | 2D grid path counting |
| Decode Ways (91) | Subsequence | String segmentation with 1-2 digit chunks |
| House Robber II (213) | House Robber | Circular variant: two linear passes |
| Combination Sum IV (377) | Knapsack (Complete) | Count permutations (order matters) |
| Word Break (139) | Subsequence | Boolean segmentation DP |
| Longest Common Subsequence (1143) | Subsequence | Two-sequence `dp[i][j]` |
| Partition Equal Subset Sum (416) | Knapsack (0-1) | Reframe as subset sum = total/2 |

---

## Attribution

The frameworks and problem derivations in this file are inspired by and adapted from labuladong's algorithmic guides (labuladong.online) and the Tech Interview Handbook (techinterviewhandbook.org) dynamic programming cheatsheet. Content has been restructured and annotated for Socratic teaching use with Python code templates, embedded prompts, and cross-references.
