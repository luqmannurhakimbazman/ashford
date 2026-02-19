# Dynamic Programming — Advanced Optimizations

Advanced DP techniques beyond the standard families covered in `dynamic-programming-core.md`. Focuses on optimization methods (Divide & Conquer DP, Knuth's optimization), bitmask DP, and advanced angles on knapsack and LIS. Based on cp-algorithms.com.

---

## Quick Reference Table

| Technique | Reduces | From → To | Precondition |
|-----------|---------|-----------|-------------|
| Divide & Conquer DP | Partition DP over rows | O(mN²) → O(mN log N) | Monotone opt(i, j) |
| Knuth's Optimization | Range DP | O(N³) → O(N²) | Quadrangle inequality on C |
| Bitmask DP (Profile) | Grid tiling / state encoding | Exponential → O(N · 2^M · M) | M ≤ ~20 |
| O(N log N) LIS | Longest increasing subseq | O(N²) → O(N log N) | — |
| Monotone Queue Knapsack | Bounded knapsack | O(NW·k) → O(NW) | — |
| Binary Grouping | Bounded knapsack | O(NW·k) → O(NW · Σlog k_i) | — |

---

## 1. Divide & Conquer DP Optimization

### When It Applies

You have a DP recurrence of the form:

```
dp(i, j) = min over 0 ≤ k ≤ j of { dp(i-1, k-1) + C(k, j) }
```

where `i` is the "row" (number of groups/segments) and `j` is the position. The key insight: if the **optimal splitting point** `opt(i, j)` is monotone — `opt(i, j) ≤ opt(i, j+1)` — then divide and conquer reduces each row from O(N²) to O(N log N).

### Precondition: Monotonicity of opt

The cost function C must satisfy the **quadrangle inequality**:

```
C(a, c) + C(b, d) ≤ C(a, d) + C(b, c)    for all a ≤ b ≤ c ≤ d
```

This guarantees opt(i, j) is non-decreasing in j for fixed i.

*Socratic prompt: "The quadrangle inequality says 'crossing costs are cheaper than nesting costs.' Can you think of a cost function that satisfies this? (Hint: what about sum of squared values?)"*

### The Algorithm

For a fixed row `i`, instead of checking all k for each j:

1. **Solve the middle:** compute `dp(i, mid)` by trying all valid k ∈ [opt_lo, opt_hi], find `opt(i, mid)`
2. **Recurse left:** for j < mid, we know opt ≤ opt(i, mid)
3. **Recurse right:** for j > mid, we know opt ≥ opt(i, mid)

Each k value participates in at most O(log N) recursive levels → total O(N log N) per row.

```python
import sys

def solve_dc_dp(n, m, cost_fn):
    """
    Divide & Conquer DP optimization.
    dp[i][j] = min cost to partition arr[0..j] into i groups.
    cost_fn(k, j) = cost of a single group covering [k, j].

    Time: O(m * n * log n).
    """
    INF = float('inf')
    dp_prev = [INF] * n
    dp_curr = [INF] * n

    # Base case: row 0 (one group covering [0, j])
    for j in range(n):
        dp_prev[j] = cost_fn(0, j)

    for i in range(1, m):
        dp_curr = [INF] * n

        def compute(lo, hi, opt_lo, opt_hi):
            if lo > hi:
                return
            mid = (lo + hi) // 2
            best_val, best_k = INF, opt_lo

            for k in range(opt_lo, min(mid, opt_hi) + 1):
                val = (dp_prev[k - 1] if k > 0 else 0) + cost_fn(k, mid)
                if val < best_val:
                    best_val = val
                    best_k = k

            dp_curr[mid] = best_val
            compute(lo, mid - 1, opt_lo, best_k)
            compute(mid + 1, hi, best_k, opt_hi)

        compute(0, n - 1, 0, n - 1)
        dp_prev = dp_curr[:]

    return dp_prev[n - 1]
```

### Example: CSES - Dividing into Segments

Divide an array into exactly m contiguous segments minimizing the total cost, where cost of a segment is (max - min)².

```python
# cost_fn(k, j) = (max(arr[k..j]) - min(arr[k..j]))^2
# This satisfies the quadrangle inequality.
```

### Relationship to Convex Hull Trick

Many D&C DP problems can alternatively be solved with the Convex Hull Trick (CHT). D&C DP is more general — it only requires monotone opt, while CHT requires the DP transitions to form linear functions.

| Method | Requirement | Time per row |
|--------|-------------|-------------|
| Naive | None | O(N²) |
| D&C DP | Monotone opt(i,j) | O(N log N) |
| Convex Hull Trick | Linear transition form | O(N) |
| Knuth's Opt | Range DP + quadrangle ineq | O(N²) total for all |

---

## 2. Knuth's Optimization (Knuth-Yao Speedup)

### When It Applies

**Range DP** of the form:

```
dp(i, j) = min over i ≤ k < j of { dp(i, k) + dp(k+1, j) + C(i, j) }
```

with base case dp(i, i) = 0 (or some constant). This covers problems like:
- Optimal binary search tree construction
- Matrix chain multiplication
- Minimum cost to merge stones
- Breaking a string/rod into pieces

### Preconditions on Cost C(i, j)

For all a ≤ b ≤ c ≤ d:

1. **Monotonicity:** `C(b, c) ≤ C(a, d)` (larger intervals cost at least as much)
2. **Quadrangle inequality:** `C(a, c) + C(b, d) ≤ C(a, d) + C(b, c)`

Together these guarantee: `opt(i, j-1) ≤ opt(i, j) ≤ opt(i+1, j)`

*Socratic prompt: "If C(i, j) = prefix_sum[j+1] - prefix_sum[i] (the sum of a subarray), does it satisfy both conditions? Why?"*

### The Speedup

**Naive range DP:** for each (i, j), test all k from i to j-1 → O(N³).

**With Knuth:** for each (i, j), test k only from `opt(i, j-1)` to `opt(i+1, j)`. The sum telescopes:

```
Σ [opt(i+1, j) - opt(i, j-1)] over all (i, j) = O(N²)
```

So total work is O(N²) instead of O(N³).

```python
def knuth_optimization(n, cost_fn):
    """
    Range DP with Knuth's optimization.
    dp[i][j] = min cost to process range [i, j].
    Transition: dp[i][j] = min over k of (dp[i][k] + dp[k+1][j] + cost(i,j)).

    Time: O(N²). Space: O(N²).
    """
    INF = float('inf')
    dp = [[0] * n for _ in range(n)]
    opt = [[0] * n for _ in range(n)]

    # Base case: single elements
    for i in range(n):
        opt[i][i] = i

    # Fill by increasing interval length
    for length in range(2, n + 1):          # length of [i, j]
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = INF
            c = cost_fn(i, j)

            # Only search k in [opt[i][j-1], opt[i+1][j]]
            lo = opt[i][j - 1]
            hi = opt[i + 1][j] if i + 1 < n else j - 1
            for k in range(lo, min(hi, j - 1) + 1):
                val = dp[i][k] + dp[k + 1][j] + c
                if val <= dp[i][j]:  # use <= to get rightmost opt (needed for upper bound)
                    dp[i][j] = val
                    opt[i][j] = k

    return dp[0][n - 1]
```

### Important Implementation Details

- **Loop order:** process by increasing interval length (length = j - i + 1)
- **Use `<=`** (not `<`) when updating best, so opt[i][j] is the **rightmost** optimum — needed because opt[i+1][j] serves as an upper bound
- **Verify** the quadrangle inequality holds for your specific cost function before applying

### Common Cost Functions

| Problem | C(i, j) | Satisfies QI? |
|---------|---------|--------------|
| Merge stones (sum of elements) | prefix[j+1] - prefix[i] | Yes |
| Optimal BST (frequency-weighted) | Σ freq[i..j] | Yes |
| Arbitrary | Must verify | Check on examples |

---

## 3. Bitmask DP (Profile Dynamics)

### When It Applies

When you need to encode the state of a **row or column boundary** in a grid-based DP. The state is a bitmask of width M (typically M ≤ 20 for 2^M to fit in memory/time).

### The Domino Tiling Problem

**Classic:** count the number of ways to tile an N × M grid with 2×1 dominoes.

**State:** `dp[row][mask]` where mask encodes which cells in the boundary between row `row` and row `row+1` are "protruding" (filled from above, extending into the next row).

- Bit j = 0: cell (row, j) is flush — the cell is covered and nothing protrudes
- Bit j = 1: cell (row, j) has a vertical domino protruding into row+1

**Transition:** for each `(row, mask)`, enumerate valid placements in row+1 that consume the protruding cells and may create new protrusions.

```python
def count_domino_tilings(n, m):
    """
    Count ways to tile an N x M grid with 2x1 dominoes.
    Time: O(N * 2^M * M). Space: O(2^M).
    """
    # Ensure m is the smaller dimension for efficiency
    if n < m:
        n, m = m, n

    dp = [0] * (1 << m)
    dp[0] = 1  # empty boundary = 1 way

    def fill(row_dp, col, mask, next_mask, new_dp):
        """Recursively place dominoes in a row, building next_mask."""
        if col >= m:
            new_dp[next_mask] += row_dp[mask]
            return

        if mask & (1 << col):
            # Cell is protruding from previous row → it's already filled
            # Move to next column, this cell in next_mask stays 0
            fill(row_dp, col + 1, mask, next_mask, new_dp)
        else:
            # Cell is empty. Option 1: place vertical domino (protrude into next row)
            fill(row_dp, col + 1, mask, next_mask | (1 << col), new_dp)

            # Option 2: place horizontal domino (fill this and next column)
            if col + 1 < m and not (mask & (1 << (col + 1))):
                fill(row_dp, col + 2, mask, next_mask, new_dp)

    for row in range(n):
        new_dp = [0] * (1 << m)
        for mask in range(1 << m):
            if dp[mask] > 0:
                fill(dp, 0, mask, 0, new_dp)
        dp = new_dp

    return dp[0]  # all cells covered, no protrusions
```

*Socratic prompt: "The bitmask represents the boundary between two rows. Why does encoding ONLY the boundary suffice — don't we need to know the state of the entire grid above?"*

### TSP and General Bitmask DP

The traveling salesman problem is the canonical non-grid bitmask DP:

```python
def tsp(dist, n):
    """
    TSP via bitmask DP. dp[mask][i] = min cost to visit cities in mask,
    ending at city i.
    Time: O(2^N * N²). Space: O(2^N * N).
    """
    INF = float('inf')
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0  # start at city 0

    for mask in range(1 << n):
        for last in range(n):
            if dp[mask][last] == INF:
                continue
            if not (mask & (1 << last)):
                continue
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue  # already visited
                new_mask = mask | (1 << nxt)
                dp[new_mask][nxt] = min(dp[new_mask][nxt],
                                        dp[mask][last] + dist[last][nxt])

    full = (1 << n) - 1
    return min(dp[full][i] + dist[i][0] for i in range(n))
```

### Bitmask Tricks

| Trick | Code | Purpose |
|-------|------|---------|
| Iterate subsets of mask | `sub = mask; while sub: ...; sub = (sub - 1) & mask` | Enumerate all subsets |
| Lowest set bit | `mask & (-mask)` | Isolate rightmost 1 |
| Remove lowest set bit | `mask & (mask - 1)` | Turn off rightmost 1 |
| Count set bits | `bin(mask).count('1')` or `mask.bit_count()` | Population count |
| Check if power of 2 | `mask and not (mask & (mask - 1))` | Exactly one bit set |

### When Is the Bitmask Dimension M Too Large?

| M | 2^M | Feasibility |
|---|-----|-------------|
| ≤ 15 | 32,768 | Fast |
| 16-20 | 65K-1M | Usually OK with care |
| 21-25 | 2M-33M | Tight, may need tricks |
| > 25 | > 33M | Generally infeasible |

---

## 4. O(N log N) LIS — The Advanced View

The basic O(N²) LIS DP is in `dynamic-programming-core.md`. Here we cover the O(N log N) binary search approach and the patience sorting interpretation.

### The Tails Array

Maintain array `d` where `d[l]` = smallest possible last element of any increasing subsequence of length `l`.

**Key invariant:** `d` is always sorted in increasing order.

For each element `a[i]`:
- Binary search for the first `d[l]` that is ≥ a[i]
- Replace `d[l]` with `a[i]` (or extend if a[i] > all values)

```python
import bisect

def lis_length(arr):
    """O(N log N) LIS using binary search on tails array."""
    tails = []
    for x in arr:
        pos = bisect.bisect_left(tails, x)  # strictly increasing
        if pos == len(tails):
            tails.append(x)
        else:
            tails[pos] = x
    return len(tails)
```

### Recovering the Subsequence

```python
def lis_with_recovery(arr):
    """O(N log N) LIS with subsequence recovery."""
    n = len(arr)
    tails = []       # tails[l] = smallest ending value for length l+1
    indices = []     # indices[l] = index in arr that achieved tails[l]
    parent = [-1] * n

    for i, x in enumerate(arr):
        pos = bisect.bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
            indices.append(i)
        else:
            tails[pos] = x
            indices[pos] = i
        parent[i] = indices[pos - 1] if pos > 0 else -1

    # Trace back from the last element of the LIS
    result = []
    idx = indices[len(tails) - 1]
    while idx != -1:
        result.append(arr[idx])
        idx = parent[idx]
    return result[::-1]
```

### The Patience Sorting Interpretation

The LIS algorithm is equivalent to **patience sorting**:
- Each element is placed on the leftmost pile whose top is ≥ the element
- If no such pile exists, start a new pile
- The number of piles equals the LIS length
- To recover the LIS, track which pile each element was placed on and backtrack

*Socratic prompt: "The tails array is always sorted. Why? What property of the algorithm maintains this invariant?"*

### Variations

| Variant | Modification |
|---------|-------------|
| Longest non-decreasing subsequence | Use `bisect_right` instead of `bisect_left` |
| Longest decreasing subsequence | Negate all elements, find LIS |
| Count of LIS | Track count alongside each tail entry |
| Minimum cover by non-increasing subsequences | By Dilworth's theorem, equals LIS length |

---

## 5. Advanced Knapsack Optimizations

The basic 0-1 and unbounded knapsacks are in `dynamic-programming-core.md`. Here we cover the bounded (multiple) knapsack optimizations.

### Binary Grouping for Bounded Knapsack

When item i has k_i copies, instead of running k_i iterations, group copies into powers of 2:

For k_i = 13: create bundles of size 1, 2, 4, 6 (remainder). Any count 0..13 can be composed from these 4 bundles.

```python
def bounded_knapsack_binary(items, W):
    """
    Bounded knapsack with binary grouping.
    items: list of (weight, value, count).
    Time: O(W * Σ log(k_i)).
    """
    # Expand items into binary groups
    expanded = []
    for w, v, k in items:
        c = 1
        while k > 0:
            take = min(c, k)
            expanded.append((take * w, take * v))
            k -= take
            c *= 2

    # Standard 0-1 knapsack on expanded items
    dp = [0] * (W + 1)
    for w, v in expanded:
        for j in range(W, w - 1, -1):
            dp[j] = max(dp[j], dp[j - w] + v)
    return dp[W]
```

### Monotone Queue Optimization for Bounded Knapsack

The tightest optimization: O(NW) total. For each item (w, v, k), split the capacity dimension into residue classes modulo w:

```python
from collections import deque

def bounded_knapsack_monotone(items, W):
    """
    Bounded knapsack with monotone queue. O(NW) total.
    items: list of (weight, value, count).
    """
    dp = [0] * (W + 1)

    for w, v, k in items:
        if w == 0:
            # Unlimited free value — special case
            for j in range(W + 1):
                dp[j] += k * v
            continue

        for r in range(w):  # residue class
            dq = deque()  # (index, adjusted_value)
            for t in range((W - r) // w + 1):
                j = r + t * w
                val = dp[j] - t * v  # "normalized" value for monotone comparison

                # Remove elements outside window of size k
                while dq and dq[0][0] < t - k:
                    dq.popleft()
                # Maintain decreasing monotone queue
                while dq and dq[-1][1] <= val:
                    dq.pop()
                dq.append((t, val))

                dp[j] = dq[0][1] + t * v  # best value in window + offset

    return dp[W]
```

**Why it works:** within a residue class mod w, the transition `dp[j] = max over t=0..k of (dp[j - t*w] + t*v)` is a sliding window maximum after normalizing by the linear term `t*v`.

*Socratic prompt: "The bounded knapsack's naive O(NWK) becomes O(NW) with a monotone queue. Where exactly does the sliding window appear? What's the window size?"*

### Comparison of Knapsack Methods

| Method | 0-1 | Unbounded | Bounded (k copies) |
|--------|-----|-----------|-------------------|
| Basic | O(NW) | O(NW) | O(NWK) |
| Binary grouping | — | — | O(NW · Σlog k_i) |
| Monotone queue | — | — | O(NW) |
| Loop direction | j decreasing | j increasing | Per residue class |

---

## 6. Largest Zero Submatrix

### The Problem

Given an N × M binary matrix, find the largest rectangular submatrix consisting entirely of zeros.

### Connection to Largest Rectangle in Histogram

Process the matrix row by row. For each row, maintain heights: `h[j]` = number of consecutive zeros above (and including) the current row in column j. Then finding the largest zero rectangle ending at this row reduces to **largest rectangle in a histogram** — solvable in O(M) with a monotone stack.

```python
def largest_zero_submatrix(matrix):
    """
    Find area of largest all-zero rectangle in a binary matrix.
    Time: O(N * M). Space: O(M).
    """
    if not matrix or not matrix[0]:
        return 0
    n, m = len(matrix), len(matrix[0])
    heights = [0] * m
    max_area = 0

    for i in range(n):
        # Update histogram heights
        for j in range(m):
            heights[j] = heights[j] + 1 if matrix[i][j] == 0 else 0

        # Largest rectangle in histogram using monotone stack
        stack = []  # stack of (index, height)
        for j in range(m + 1):
            h = heights[j] if j < m else 0
            start = j
            while stack and stack[-1][1] > h:
                idx, sh = stack.pop()
                area = sh * (j - idx)
                max_area = max(max_area, area)
                start = idx
            stack.append((start, h))

    return max_area
```

*Socratic prompt: "This problem reduces to 'largest rectangle in histogram' applied N times. Why does the histogram approach work here? What does each bar represent?"*

### The cp-algorithms Approach (Direct Stack)

An alternative implementation uses auxiliary arrays `d[j]` (nearest row with a 1 above), `d1[j]` (left boundary), `d2[j]` (right boundary):

```python
def largest_zero_submatrix_direct(matrix):
    """Direct stack approach from cp-algorithms. O(NM)."""
    n, m = len(matrix), len(matrix[0])
    d = [-1] * m  # nearest row above with a 1 in column j
    max_area = 0

    for i in range(n):
        for j in range(m):
            if matrix[i][j] == 1:
                d[j] = i

        # Find left boundaries using stack (left to right)
        d1 = [-1] * m
        stack = []
        for j in range(m):
            while stack and d[stack[-1]] <= d[j]:
                stack.pop()
            d1[j] = stack[-1] if stack else -1
            stack.append(j)

        # Find right boundaries using stack (right to left)
        d2 = [m] * m
        stack = []
        for j in range(m - 1, -1, -1):
            while stack and d[stack[-1]] <= d[j]:
                stack.pop()
            d2[j] = stack[-1] if stack else m
            stack.append(j)

        # Compute maximum area
        for j in range(m):
            area = (i - d[j]) * (d2[j] - d1[j] - 1)
            max_area = max(max_area, area)

    return max_area
```

**Practice:** LeetCode 85 (Maximal Rectangle), LeetCode 84 (Largest Rectangle in Histogram), LeetCode 221 (Maximal Square)

---

## Attribution

Content synthesized from cp-algorithms.com articles on [Intro to DP](https://cp-algorithms.com/dynamic_programming/intro-to-dp.html), [Knapsack](https://cp-algorithms.com/dynamic_programming/knapsack.html), [LIS](https://cp-algorithms.com/dynamic_programming/longest_increasing_subsequence.html), [Divide and Conquer DP](https://cp-algorithms.com/dynamic_programming/divide-and-conquer-dp.html), [Knuth's Optimization](https://cp-algorithms.com/dynamic_programming/knuth-optimization.html), [Profile Dynamics](https://cp-algorithms.com/dynamic_programming/profile-dynamics.html), and [Largest Zero Submatrix](https://cp-algorithms.com/dynamic_programming/zero_matrix.html). Algorithms and complexity analyses are from the original articles; code has been translated to Python with added commentary for the leetcode-teacher reference format. For standard DP families (knapsack basics, LIS O(N²), grid DP, game theory DP, etc.), see `dynamic-programming-core.md`.
