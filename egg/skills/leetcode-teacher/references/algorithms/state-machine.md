# The State Machine Framework

State machine DP for stock buy/sell problems and similar constraint-based optimization.

**Prerequisites:** Basic DP concepts. See `references/algorithms/dp-framework.md` for the three-step DP process.

---

## Stock Problem Generalization

All stock buy/sell problems can be modeled with one state machine:

```
dp[i][k][s]
  i = day (0 to n-1)
  k = max transactions remaining
  s = 0 (not holding) or 1 (holding)
```

## State Transitions

```
dp[i][k][0] = max(dp[i-1][k][0],           # rest (do nothing)
                   dp[i-1][k][1] + prices[i]) # sell

dp[i][k][1] = max(dp[i-1][k][1],           # rest (do nothing)
                   dp[i-1][k-1][0] - prices[i]) # buy (uses one transaction)
```

## Variants Table

| Problem | K | Special Rule | Simplification |
|---------|---|-------------|----------------|
| LC 121: Best Time to Buy and Sell Stock | 1 | — | Track min price, max profit |
| LC 122: Best Time II (unlimited) | infinity | — | Sum all positive diffs |
| LC 123: Best Time III | 2 | — | Full DP with k=2 |
| LC 188: Best Time IV | K (given) | — | General DP |
| LC 309: With Cooldown | infinity | Must wait 1 day after sell | `dp[i][0] = max(rest, dp[i-2][1] + price)` |
| LC 714: With Transaction Fee | infinity | Fee per transaction | Subtract fee on sell |

All six problems are the same framework with minor modifications to the transitions.

## See Also

- `references/algorithms/stock-problems.md` — Full implementations for all 6 stock variants with code
- `references/algorithms/dynamic-programming-core.md` — House Robber & Stock series deep dive (Section 8)

---

## Attribution

Extracted from the State Machine Framework section of `algorithm-frameworks.md`, inspired by labuladong's algorithmic guides (labuladong.online).
