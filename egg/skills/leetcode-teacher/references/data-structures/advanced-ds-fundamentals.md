# Advanced Data Structure Fundamentals

Minimum stacks, minimum queues, and sparse tables — foundational building blocks for range queries and sliding window problems. Based on cp-algorithms.com treatments of stack/queue modifications and static range-minimum queries.

---

## 1. Minimum Stack

### The Problem

A standard stack supports push/pop in O(1), but finding the minimum element requires O(N) scanning. Can we maintain O(1) for all three operations?

### The Trick: Pair Each Element with Its Running Minimum

Store `(value, current_min)` pairs. When pushing, the new minimum is `min(value, stack[-1].min)`. The top pair always holds the stack-wide minimum.

```python
class MinStack:
    def __init__(self):
        self.stack = []  # list of (value, min_so_far)

    def push(self, val):
        current_min = val if not self.stack else min(val, self.stack[-1][1])
        self.stack.append((val, current_min))

    def pop(self):
        return self.stack.pop()[0]

    def get_min(self):
        return self.stack[-1][1]  # O(1)
```

| Operation | Time | Space Overhead |
|-----------|------|----------------|
| push | O(1) | +1 int per element |
| pop | O(1) | — |
| get_min | O(1) | — |

*Socratic prompt: "Why does storing the running minimum work even after pops? What invariant is maintained?"*

**The key insight:** when we pop an element, all elements below it are unchanged — their running minimums were computed without knowledge of the popped element. So the new top's stored minimum is still correct.

### Practice Problems

- LeetCode 155: Min Stack

---

## 2. Minimum Queue

A queue supports enqueue at the back and dequeue from the front. Finding the minimum in O(1) is harder than for stacks because removal happens at the opposite end from insertion.

### Method 1: Monotonic Deque (Amortized O(1))

Maintain a deque that stores elements in **non-decreasing** order. When adding a new element, pop all larger elements from the back — they can never be the minimum while the new (smaller) element exists.

```python
from collections import deque

class MinQueueMonotonic:
    """Min queue using monotonic deque. O(1) amortized all operations.
    Caveat: does not store all elements — cannot iterate over contents."""

    def __init__(self):
        self.q = deque()  # non-decreasing order

    def get_min(self):
        return self.q[0]  # front is always the minimum

    def push(self, val):
        # Remove elements larger than val — they'll never be min
        while self.q and self.q[-1] > val:
            self.q.pop()
        self.q.append(val)

    def pop(self, removed_val):
        # Must pass the actual value being removed from the logical queue
        if self.q and self.q[0] == removed_val:
            self.q.popleft()
```

**Limitation:** `pop` requires knowing the value being removed (the caller must track the logical queue separately). Elements that were "displaced" by a smaller arrival are not stored.

*Socratic prompt: "Why can we safely discard elements larger than the new arrival? Under what condition would a discarded element have been the minimum?"*

### Method 2: Two Min-Stacks (True O(1) Amortized, Stores All Elements)

Simulate a queue with two min-stacks: `s_in` (receives pushes) and `s_out` (serves pops). When `s_out` is empty, transfer all elements from `s_in` — this reversal puts the oldest element on top of `s_out`.

```python
class MinQueueTwoStacks:
    """Min queue using two min-stacks. O(1) amortized, stores all elements."""

    def __init__(self):
        self.s_in = []   # (value, min_so_far)
        self.s_out = []  # (value, min_so_far)

    def push(self, val):
        current_min = val if not self.s_in else min(val, self.s_in[-1][1])
        self.s_in.append((val, current_min))

    def pop(self):
        if not self.s_out:
            self._transfer()
        return self.s_out.pop()[0]

    def get_min(self):
        if not self.s_in and not self.s_out:
            raise IndexError("empty queue")
        if not self.s_in:
            return self.s_out[-1][1]
        if not self.s_out:
            return self.s_in[-1][1]
        return min(self.s_in[-1][1], self.s_out[-1][1])

    def _transfer(self):
        """Move all elements from s_in to s_out, reversing order."""
        while self.s_in:
            val = self.s_in.pop()[0]
            current_min = val if not self.s_out else min(val, self.s_out[-1][1])
            self.s_out.append((val, current_min))
```

**Why amortized O(1)?** Each element is pushed onto `s_in` once, transferred to `s_out` once, and popped from `s_out` once — 3 operations total over its lifetime.

| Method | get_min | push | pop | Stores all? |
|--------|---------|------|-----|-------------|
| Monotonic deque | O(1) | O(1) amortized | O(1)* | No |
| Two min-stacks | O(1) | O(1) | O(1) amortized | Yes |

*\*Requires knowing the removed value.*

### Application: Sliding Window Minimum in O(N)

Both methods solve the classic problem: given array `A` of length `N` and window size `M`, find the minimum of every contiguous subarray of length `M`.

```python
def sliding_window_min(arr, m):
    """O(N) sliding window minimum using monotonic deque."""
    from collections import deque
    dq = deque()  # stores (value, index) in non-decreasing value order
    result = []
    for i, val in enumerate(arr):
        # Remove elements outside the window
        while dq and dq[0][1] <= i - m:
            dq.popleft()
        # Maintain monotonicity
        while dq and dq[-1][0] > val:
            dq.pop()
        dq.append((val, i))
        if i >= m - 1:
            result.append(dq[0][0])
    return result
```

*Socratic prompt: "The monotonic deque approach processes N elements, each entering and leaving the deque at most once. Why does this guarantee O(N) total time, not O(NM)?"*

### Practice Problems

- LeetCode 239: Sliding Window Maximum (use max variant)
- LeetCode 862: Shortest Subarray with Sum at Least K (min-deque on prefix sums)

---

## 3. Sparse Table

### When to Use

When you need **fast range queries on a static (immutable) array** — no updates between queries. Sparse table achieves:

| Query Type | Build | Query | Space |
|------------|-------|-------|-------|
| Range sum | O(N log N) | O(log N) | O(N log N) |
| Range min/max (RMQ) | O(N log N) | **O(1)** | O(N log N) |

The O(1) RMQ is the killer feature — no other structure achieves this without complex preprocessing (like the Bender–Farach-Colton algorithm).

*Socratic prompt: "Why can range minimum queries be answered in O(1) but range sum queries cannot? What property of `min` makes the difference?"*

### The Idea: Precompute All Power-of-Two Ranges

Any interval of length `L` can be covered by two overlapping intervals of length `2^k` where `k = floor(log2(L))`. For **idempotent** operations (min, max, GCD, AND, OR) — where processing an element twice doesn't change the result — this gives O(1) queries.

### Building the Table

`st[k][j]` = answer for the range `[j, j + 2^k - 1]`.

Base case: `st[0][j] = arr[j]` (ranges of length 1).

Transition: `st[k][j] = op(st[k-1][j], st[k-1][j + 2^(k-1)])` — combine two halves.

```python
import math

class SparseTable:
    """Sparse table for O(1) range minimum queries on a static array."""

    def __init__(self, arr):
        n = len(arr)
        self.k = int(math.log2(n)) + 1 if n > 0 else 0
        # st[k][j] = min of arr[j .. j + 2^k - 1]
        self.st = [[0] * n for _ in range(self.k + 1)]
        self.st[0] = arr[:]

        for k in range(1, self.k + 1):
            for j in range(n - (1 << k) + 1):
                self.st[k][j] = min(
                    self.st[k - 1][j],
                    self.st[k - 1][j + (1 << (k - 1))]
                )

        # Precompute floor(log2) for all lengths
        self.log = [0] * (n + 1)
        for i in range(2, n + 1):
            self.log[i] = self.log[i // 2] + 1

    def query_min(self, l, r):
        """Return min(arr[l..r]) in O(1)."""
        length = r - l + 1
        k = self.log[length]
        return min(self.st[k][l], self.st[k][r - (1 << k) + 1])
```

### Why O(1) RMQ Works

For `min`, overlapping is harmless: `min(min(A), min(B)) = min(A ∪ B)` regardless of overlap. So we cover `[L, R]` with just two precomputed ranges:

```
[L, L + 2^k - 1]  and  [R - 2^k + 1, R]    where k = floor(log2(R - L + 1))
```

These two ranges overlap in the middle but together cover `[L, R]` exactly.

**This does NOT work for sum** — overlapping elements would be counted twice. For sum, you need to decompose the range into non-overlapping power-of-two parts, taking O(log N).

### Range Sum Query with Sparse Table

```python
def query_sum(self, l, r):
    """Return sum(arr[l..r]) in O(log N) using non-overlapping decomposition."""
    total = 0
    for k in range(self.k, -1, -1):
        if (1 << k) <= r - l + 1:
            total += self.st[k][l]
            l += 1 << k
    return total
```

### Comparison with Other Static RMQ Structures

| Structure | Build | Query | Space | Updates |
|-----------|-------|-------|-------|---------|
| Sparse Table | O(N log N) | O(1) | O(N log N) | Not supported |
| Segment Tree | O(N) | O(log N) | O(N) | O(log N) |
| Fenwick Tree (BIT) | O(N) | O(log N) | O(N) | O(log N) |
| Sqrt Decomposition | O(N) | O(√N) | O(N) | O(√N) |
| Disjoint Sparse Table | O(N log N) | O(1) | O(N log N) | Not supported |

*Socratic prompt: "When would you choose a sparse table over a segment tree? What about the reverse — when is a segment tree strictly better?"*

### Idempotent Operations That Support O(1) Queries

| Operation | Idempotent? | O(1) with Sparse Table? |
|-----------|-------------|------------------------|
| min / max | Yes | Yes |
| GCD | Yes | Yes |
| Bitwise AND / OR | Yes | Yes |
| Sum | No | No (O(log N)) |
| XOR | No | No (O(log N)) |

### Practice Problems

- LeetCode 2104: Sum of Subarray Ranges (use sparse table for efficient min/max queries)
- CSES: Static Range Minimum Queries
- Codeforces 514D: R2D2 and Droid Army

---

## Attribution

Content synthesized from cp-algorithms.com articles on [Minimum Stack / Minimum Queue](https://cp-algorithms.com/data_structures/stack_queue_modification.html) and [Sparse Table](https://cp-algorithms.com/data_structures/sparse-table.html). Algorithms and complexity analyses are from the original articles; code has been translated to Python with added commentary for the leetcode-teacher reference format.
