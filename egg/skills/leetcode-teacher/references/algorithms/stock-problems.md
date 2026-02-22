# Stock Problems & House Robber Pattern

State machine DP for all 6 stock buy/sell variants, plus the House Robber adjacency-constraint family.

**Prerequisites:** DP basics. See `references/algorithms/dp-framework.md` for the three-step DP process and `references/algorithms/state-machine.md` for the state machine framework overview.

---

## State Machine DP (Stock Problems)

### Framework

```python
def max_profit(prices, k, cooldown=0, fee=0):
    n = len(prices)
    if n == 0:
        return 0

    # dp[i][j][s]: day i, j transactions remaining, s=0 not holding / s=1 holding
    dp = [[[-float('inf')] * 2 for _ in range(k + 1)] for _ in range(n + 1)]
    dp[0][k][0] = 0  # Start with k transactions, not holding, 0 profit

    for i in range(1, n + 1):
        for j in range(k + 1):
            # Not holding
            dp[i][j][0] = max(
                dp[i - 1][j][0],                              # Rest
                dp[i - 1][j][1] + prices[i - 1] - fee         # Sell
            )
            # Holding
            if j > 0:
                buy_from = i - 1 - cooldown  # Cooldown after selling
                if buy_from >= 0:
                    dp[i][j][1] = max(
                        dp[i - 1][j][1],                      # Rest
                        dp[buy_from][j - 1][0] - prices[i - 1]  # Buy
                    )

    return max(dp[n][j][0] for j in range(k + 1))
```

### Variant Simplifications

**k=1 (LC 121):**
```python
def max_profit_one(prices):
    min_price = float('inf')
    max_profit = 0
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    return max_profit
```

**k=infinity (LC 122):**
```python
def max_profit_unlimited(prices):
    return sum(max(0, prices[i] - prices[i-1]) for i in range(1, len(prices)))
```

**k=infinity with cooldown (LC 309):**
```python
def max_profit_cooldown(prices):
    n = len(prices)
    if n < 2:
        return 0
    dp_hold = -prices[0]       # Holding stock
    dp_sold = 0                # Just sold (cooldown next)
    dp_rest = 0                # Not holding, free to buy
    for i in range(1, n):
        new_hold = max(dp_hold, dp_rest - prices[i])
        new_sold = dp_hold + prices[i]
        new_rest = max(dp_rest, dp_sold)
        dp_hold, dp_sold, dp_rest = new_hold, new_sold, new_rest
    return max(dp_sold, dp_rest)
```

*Socratic prompt: "For the cooldown variant, draw the state machine diagram. How many states are there? What are the transitions?"*

---

## House Robber Pattern

Linear DP with an adjacency constraint: cannot take two consecutive elements.

### Core Pattern

```python
def rob(nums):
    if not nums:
        return 0
    if len(nums) <= 2:
        return max(nums)
    prev2, prev1 = nums[0], max(nums[0], nums[1])
    for i in range(2, len(nums)):
        curr = max(prev1, prev2 + nums[i])
        prev2, prev1 = prev1, curr
    return prev1
```

**Recurrence:** `dp[i] = max(dp[i-1], dp[i-2] + nums[i])` — either skip house `i` (take previous best) or rob house `i` (add to best before previous).

### Variant: Circular (LC 213)

Houses are in a circle — first and last are adjacent. Solution: run linear House Robber twice — once excluding the first house, once excluding the last. Take the max.

```python
def rob_circular(nums):
    if len(nums) == 1:
        return nums[0]
    return max(rob(nums[1:]), rob(nums[:-1]))
```

### Variant: Tree (LC 337)

Houses form a binary tree — cannot rob a node and its direct children.

```python
def rob_tree(root):
    def dfs(node):
        if not node:
            return (0, 0)  # (rob_this, skip_this)
        left = dfs(node.left)
        right = dfs(node.right)
        rob_this = node.val + left[1] + right[1]   # Rob node, must skip children
        skip_this = max(left) + max(right)          # Skip node, free to rob or skip children
        return (rob_this, skip_this)
    return max(dfs(root))
```

## Key Problems

- House Robber I/II/III (198/213/337)
- Best Time to Buy and Sell Stock I-VI (121/122/123/188/309/714)

---

## Attribution

Extracted from the State Machine DP and House Robber sections of `advanced-patterns.md` and Section 8 of `dynamic-programming-core.md`, inspired by labuladong's algorithmic guides (labuladong.online).
