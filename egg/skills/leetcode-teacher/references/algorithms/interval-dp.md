# Interval DP & Egg Drop

DP on intervals (burst balloons, merge stones) and the egg drop problem (closely related interval reasoning).

**Prerequisites:** DP basics. See `references/algorithms/dp-framework.md` for the three-step DP process.

---

## Burst Balloons (LC 312)

**Key reframe:** Instead of thinking about which balloon to burst *first*, think about which balloon to burst *last* in the interval `[i, j]`.

**State:** `dp[i][j]` = max coins from bursting all balloons between indices `i` and `j` (exclusive).

**Transition:** For each `k` in `(i, j)` as the last balloon burst: `dp[i][j] = max(dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j])`.

```python
def max_coins(nums):
    # Add boundary balloons with value 1
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]

    # Fill by increasing interval length
    for length in range(2, n):  # length = j - i
        for i in range(n - length):
            j = i + length
            for k in range(i + 1, j):  # k is the last balloon burst
                dp[i][j] = max(dp[i][j],
                    dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j])

    return dp[0][n - 1]
```

**Why "last burst" works:** If `k` is the last balloon in `[i, j]`, then when we burst `k`, only `nums[i]` and `nums[j]` are adjacent to it (all others in the interval are already gone). This makes subproblems `[i, k]` and `[k, j]` independent.

*Socratic prompt: "Why does thinking about the first balloon to burst make subproblems dependent? What changes when you think about the last?"*

**Related:** Minimum Cost to Merge Stones (1000), Matrix Chain Multiplication.

---

## Egg Drop (Super Egg Drop — LC 887)

**Problem:** With `k` eggs and `n` floors, what's the minimum number of drops needed to find the critical floor (worst case)?

**State:** `dp[k][n]` = minimum drops needed with `k` eggs and `n` floors.

**Transition:** Drop an egg from floor `x`:
- Egg breaks: search below with `dp[k-1][x-1]`
- Egg survives: search above with `dp[k][n-x]`
- Worst case: `max(dp[k-1][x-1], dp[k][n-x])`
- Best choice: minimize over all floors `x`

**Naive O(kn²):**

```python
from functools import lru_cache

def super_egg_drop_naive(k, n):
    @lru_cache(maxsize=None)
    def dp(k, n):
        if k == 1:
            return n  # Must try every floor linearly
        if n == 0:
            return 0
        res = float('inf')
        for x in range(1, n + 1):
            worst = max(dp(k - 1, x - 1), dp(k, n - x))
            res = min(res, worst + 1)
        return res
    return dp(k, n)
```

**O(kn log n) optimization with binary search:** For fixed `k` and `n`, as `x` increases, `dp[k-1][x-1]` monotonically increases and `dp[k][n-x]` monotonically decreases. Binary search for the crossover point.

```python
def super_egg_drop(k, n):
    @lru_cache(maxsize=None)
    def dp(k, n):
        if k == 1:
            return n
        if n == 0:
            return 0
        lo, hi = 1, n
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            breaks = dp(k - 1, mid - 1)   # Egg breaks
            survives = dp(k, n - mid)       # Egg survives
            if breaks < survives:
                lo = mid
            elif breaks > survives:
                hi = mid
            else:
                lo = hi = mid
        return 1 + min(
            max(dp(k - 1, lo - 1), dp(k, n - lo)),
            max(dp(k - 1, hi - 1), dp(k, n - hi))
        )
    return dp(k, n)
```

**Alternative O(kn) "reverse thinking":** Instead of "given k eggs and n floors, minimize drops", ask "given k eggs and m drops, maximize floors you can check". `dp[m][k] = dp[m-1][k-1] + dp[m-1][k] + 1`.

*Socratic prompt: "With 1 egg, why must you start from floor 1 and go up? With 2 eggs and 100 floors, what's your first drop and why?"*

## See Also

- `references/algorithms/dynamic-programming-core.md` — Full DP framework
- `references/algorithms/game-theory-dp.md` — Game theory DP (related minimax reasoning)

---

## Attribution

Extracted from Sections 6 (Interval DP) and 7 (Egg Drop) of `dynamic-programming-core.md`, inspired by labuladong's algorithmic guides (labuladong.online).
