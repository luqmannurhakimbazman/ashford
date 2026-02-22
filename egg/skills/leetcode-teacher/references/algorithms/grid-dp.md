# Grid & Path DP

DP on 2D grids: minimum path sum, unique paths, dungeon game, and graph-constrained paths.

**Prerequisites:** DP basics. See `references/algorithms/dp-framework.md` for the three-step DP process.

---

## Minimum Path Sum (LC 64)

**State:** `dp[i][j]` = minimum path sum from top-left to cell `(i, j)`.

**Transition:** `dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])` (can only move right or down).

```python
def min_path_sum(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + grid[0][j]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i - 1][j], dp[i][j - 1])
    return dp[m - 1][n - 1]
```

**Space optimization:** Use a 1D array of size `n`, updating left-to-right row by row.

*Socratic prompt: "Why can we only move right or down? What would change if we could move in all 4 directions?"*

## Dungeon Game (LC 174)

**Key twist:** Must solve **backwards** (from bottom-right to top-left) because the minimum HP needed at each cell depends on future cells, not past ones.

**State:** `dp[i][j]` = minimum HP needed to enter cell `(i, j)` and survive to the end.

```python
def calculate_minimum_hp(dungeon):
    m, n = len(dungeon), len(dungeon[0])
    dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
    dp[m][n - 1] = dp[m - 1][n] = 1  # Need at least 1 HP at the end
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            dp[i][j] = max(1, min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j])
    return dp[0][0]
```

## Freedom Trail (LC 514)

**State:** `dp[i][j]` = minimum steps to spell `key[i:]` when the ring pointer is at position `j`.

```python
def find_rotate_steps(ring, key):
    from functools import lru_cache
    n = len(ring)
    char_pos = {}
    for i, c in enumerate(ring):
        char_pos.setdefault(c, []).append(i)

    @lru_cache(maxsize=None)
    def dp(i, j):
        if i == len(key):
            return 0
        res = float('inf')
        for k in char_pos[key[i]]:
            diff = abs(j - k)
            cost = min(diff, n - diff) + 1
            res = min(res, cost + dp(i + 1, k))
        return res

    return dp(0, 0)
```

## Cheapest Flights Within K Stops (LC 787)

**State:** `dp[k][dst]` = cheapest price to reach `dst` using at most `k` edges.

```python
def find_cheapest_price(n, flights, src, dst, k):
    INF = float('inf')
    dp = [[INF] * n for _ in range(k + 2)]
    dp[0][src] = 0
    for i in range(1, k + 2):
        dp[i][src] = 0
        for u, v, price in flights:
            if dp[i - 1][u] != INF:
                dp[i][v] = min(dp[i][v], dp[i - 1][u] + price)
    return dp[k + 1][dst] if dp[k + 1][dst] != INF else -1
```

*Socratic prompt: "This looks like Bellman-Ford with a twist. What does the k-stop limit add to the state space?"*

## See Also

- `references/algorithms/dynamic-programming-core.md` — Full DP framework
- `references/graphs/dijkstra.md` — Dijkstra for weighted shortest paths

---

## Attribution

Extracted from Section 4 (Grid & Path DP) of `dynamic-programming-core.md`, inspired by labuladong's algorithmic guides (labuladong.online).
