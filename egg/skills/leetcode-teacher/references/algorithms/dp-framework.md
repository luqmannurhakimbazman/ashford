# The DP Framework

Meta-level introduction to dynamic programming: the three-step process, top-down vs bottom-up, and the two subsequence DP templates.

**Prerequisites:** Recursion, memoization concept. See `references/frameworks/algorithm-frameworks.md` for the recursion-as-tree-traversal model (decomposition mode = DP).

---

## Three Steps to Define Any DP

1. **Clarify the state** — what changes between subproblems? (Index, remaining capacity, last choice, etc.)
2. **Clarify the choices** — at each state, what decisions can you make?
3. **Define dp meaning** — `dp[state]` = the answer to the subproblem defined by that state

## Top-Down vs Bottom-Up

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

## Two Subsequence DP Templates

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

## See Also

- `references/algorithms/dynamic-programming-core.md` — Comprehensive DP: knapsack, grid/path, interval, game theory, string DP, egg drop, Floyd-Warshall
- `references/algorithms/knapsack.md` — Knapsack problem family
- `references/algorithms/grid-dp.md` — Grid and path DP
- `references/algorithms/game-theory-dp.md` — Two-player game DP
- `references/algorithms/interval-dp.md` — Interval DP and egg drop
- `references/algorithms/subsequence-dp.md` — Subsequence and string DP patterns
- `references/algorithms/stock-problems.md` — State machine DP for stock problems

---

## Attribution

Extracted from the DP Framework section of `algorithm-frameworks.md`, inspired by labuladong's algorithmic guides (labuladong.online).
