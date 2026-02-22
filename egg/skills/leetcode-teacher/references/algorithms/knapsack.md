# Knapsack Problems

The knapsack family: 0-1, complete, and bounded variants with space optimization.

**Prerequisites:** DP basics. See `references/algorithms/dp-framework.md` for the three-step DP process.

---

## The Knapsack Family

All knapsack problems share the same structure: given items with weights and values, select items to maximize value within a weight capacity.

| Variant | Item Usage | Key Difference | Iteration Order |
|---------|-----------|----------------|-----------------|
| **0-1 Knapsack** | Each item used at most once | Standard DP with 2D table | Inner loop: right-to-left (1D optimization) |
| **Complete Knapsack** | Each item used unlimited times | Same structure, different loop direction | Inner loop: left-to-right (1D optimization) |
| **Bounded Knapsack** | Each item has a count limit | Binary representation trick | Convert to 0-1 knapsack |

## 0-1 Knapsack

**State:** `dp[i][w]` = max value using items `0..i-1` with capacity `w`.

**Transition:** For each item `i` with weight `wt[i]` and value `val[i]`:
- Don't take item `i`: `dp[i][w] = dp[i-1][w]`
- Take item `i` (if `w >= wt[i]`): `dp[i][w] = max(dp[i-1][w], dp[i-1][w - wt[i]] + val[i])`

```python
def knapsack_01(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]  # Don't take item i
            if w >= weights[i - 1]:
                dp[i][w] = max(dp[i][w],
                    dp[i - 1][w - weights[i - 1]] + values[i - 1])
    return dp[n][capacity]
```

**Space-optimized (1D array):** Iterate capacity **right-to-left** to avoid using item `i` twice:

```python
def knapsack_01_optimized(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for i in range(len(weights)):
        for w in range(capacity, weights[i] - 1, -1):  # Right to left!
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]
```

*Socratic prompt: "Why must we iterate right-to-left in the 1D optimization? What goes wrong if we go left-to-right?"*

## Complete Knapsack

Same as 0-1, but each item can be used unlimited times. The only change: iterate capacity **left-to-right**:

```python
def knapsack_complete(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for i in range(len(weights)):
        for w in range(weights[i], capacity + 1):  # Left to right!
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]
```

**Why left-to-right?** When computing `dp[w]`, we want `dp[w - weights[i]]` to already reflect taking item `i` (allowing reuse). Left-to-right ensures this.

**Coin Change (LC 322)** is a complete knapsack problem: coins are items with unlimited supply, "weight" is coin value, minimize number of coins to reach amount.

## Partition Equal Subset Sum (LC 416)

**Reframe as 0-1 knapsack:** Can we select a subset summing to `total_sum / 2`?

```python
def can_partition(nums):
    total = sum(nums)
    if total % 2:
        return False
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        for w in range(target, num - 1, -1):  # 0-1: right-to-left
            dp[w] = dp[w] or dp[w - num]
    return dp[target]
```

## Target Sum (LC 494)

**Transform to knapsack:** If we split nums into positive set P and negative set N, then `sum(P) - sum(N) = target` and `sum(P) + sum(N) = total`. So `sum(P) = (target + total) / 2`. This becomes a 0-1 knapsack counting problem.

```python
def find_target_sum_ways(nums, target):
    total = sum(nums)
    if (target + total) % 2 or abs(target) > total:
        return 0
    bag = (target + total) // 2
    dp = [0] * (bag + 1)
    dp[0] = 1  # One way to make sum 0: take nothing
    for num in nums:
        for w in range(bag, num - 1, -1):
            dp[w] += dp[w - num]
    return dp[bag]
```

*Socratic prompt: "The brute-force is 2^n (try +/- for each element). How does the knapsack transformation reduce this to O(n * sum)?"*

## Knapsack Problem Recognition

| Problem | Knapsack Type | Items | Weight | Value |
|---------|--------------|-------|--------|-------|
| Coin Change (322) | Complete | Coins | Coin value | 1 (minimize count) |
| Coin Change II (518) | Complete | Coins | Coin value | Count combinations |
| Partition Equal Subset (416) | 0-1 | Numbers | Number value | Boolean reachability |
| Target Sum (494) | 0-1 | Numbers | Number value | Count ways |
| Ones and Zeroes (474) | 0-1 (2D) | Strings | (zeros, ones) | Max count |
| Last Stone Weight II (1049) | 0-1 | Stones | Stone weight | Min remaining |

## See Also

- `references/algorithms/dynamic-programming-core.md` — Full DP framework and other DP families
- `references/algorithms/dynamic-programming-advanced.md` — Bounded knapsack optimizations

---

## Attribution

Extracted from Section 3 (Knapsack Problems) of `dynamic-programming-core.md`, inspired by labuladong's algorithmic guides (labuladong.online).
