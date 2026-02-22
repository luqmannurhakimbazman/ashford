# Advanced Tree Structures

Deep-dive into competitive programming tree-based data structures: DSU (Union-Find), Fenwick Trees (BIT), Segment Trees (advanced), Treaps, Sqrt Decomposition, and supporting techniques. Based on cp-algorithms.com.

---

## 1. Disjoint Set Union (DSU / Union-Find)

### Core Idea

Maintain a collection of disjoint sets. Support two operations efficiently:
- **Find(x):** which set does x belong to? (return the representative/root)
- **Union(x, y):** merge the sets containing x and y

### Naive Implementation

Each element stores a parent pointer. Find follows parents to the root. Union attaches one root to another.

```python
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x, y):
        """Union by rank. Returns False if already in same set."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        # Attach smaller tree under larger tree
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True
```

### Two Key Optimizations

| Optimization | Effect | Standalone Complexity |
|-------------|--------|----------------------|
| **Path compression** | During find, point every node directly to root | O(log N) amortized |
| **Union by rank/size** | Attach smaller tree under larger | O(log N) worst case |
| **Both combined** | Nearly constant per operation | O(α(N)) ≈ O(1) |

α(N) is the inverse Ackermann function — effectively ≤ 4 for any practical N (up to 10^600).

*Socratic prompt: "Path compression flattens the tree during find. Union by rank keeps the tree balanced during union. Why do you need both for near-O(1) performance?"*

### Weighted DSU (Tracking Edge Weights)

Maintain a weight `w[x]` representing the "distance" from x to its root. During find, accumulate weights along the path. Useful for problems like "is the relationship between a and b consistent?"

```python
class WeightedDSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.weight = [0] * n  # weight[x] = distance from x to parent[x]

    def find(self, x):
        if self.parent[x] != x:
            root = self.find(self.parent[x])
            self.weight[x] += self.weight[self.parent[x]]
            self.parent[x] = root
        return self.parent[x]

    def dist(self, x):
        """Distance from x to its root (call find first)."""
        self.find(x)
        return self.weight[x]

    def union(self, x, y, w):
        """Set dist(y) - dist(x) = w. Returns False if contradicts existing."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return self.weight[x] - self.weight[y] == w
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
            x, y = y, x
            w = -w
        self.parent[ry] = rx
        self.weight[ry] = self.weight[x] - self.weight[y] + w
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True
```

### DSU with Rollback (for offline algorithms)

When you need to **undo** union operations (e.g., in segment-tree-over-time problems), use union by rank **without** path compression, and maintain a stack of changes.

```python
class DSURollback:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.history = []  # stack of (node, old_parent, old_rank)

    def find(self, x):
        # No path compression — needed for rollback correctness
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.history.append((ry, self.parent[ry], self.rank[rx]))
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def rollback(self):
        if not self.history:
            return
        ry, old_parent, old_rank_rx = self.history.pop()
        rx = self.parent[ry]
        self.rank[rx] = old_rank_rx
        self.parent[ry] = old_parent
```

### Common Applications

| Problem | DSU Usage |
|---------|-----------|
| Connected components | Classic find/union |
| Kruskal's MST | Sort edges, union endpoints |
| Cycle detection in undirected graph | Union returns false → cycle |
| Dynamic connectivity (offline) | DSU with rollback + segment tree over time |
| Checking bipartiteness | Weighted DSU with parity weights |

**Practice:** LeetCode 547 (Number of Provinces), 684 (Redundant Connection), 721 (Accounts Merge), 1319 (Number of Operations to Make Network Connected)

---

## 2. Fenwick Tree (Binary Indexed Tree / BIT)

### Core Idea

A Fenwick tree supports **prefix sum queries** and **point updates** in O(log N) with minimal memory (just an array of size N+1). It's simpler and faster in practice than a segment tree for these specific operations.

### How It Works

The key insight uses **lowest set bit** arithmetic. Node `i` is responsible for the range `[i - lowbit(i) + 1, i]` where `lowbit(i) = i & (-i)`.

- **Update:** increment index `i`, then all ancestors (add lowbit)
- **Query:** sum prefix [1, i] by walking down (subtract lowbit)

```python
class FenwickTree:
    """1-indexed Fenwick tree for prefix sums and point updates."""

    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)

    def update(self, i, delta):
        """Add delta to position i. O(log N)."""
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)  # move to parent

    def query(self, i):
        """Return sum of arr[1..i]. O(log N)."""
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & (-i)  # move to predecessor
        return total

    def range_query(self, l, r):
        """Return sum of arr[l..r]."""
        return self.query(r) - self.query(l - 1)
```

### Building from an Existing Array in O(N)

```python
def build_fenwick(arr):
    """Build Fenwick tree from 0-indexed arr in O(N)."""
    n = len(arr)
    tree = [0] + arr[:]  # 1-indexed
    for i in range(1, n + 1):
        j = i + (i & (-i))
        if j <= n:
            tree[j] += tree[i]
    return tree
```

### Finding the k-th Smallest Element

With a Fenwick tree storing frequency counts, binary lifting finds the k-th smallest in O(log N):

```python
def kth_smallest(bit, n, k):
    """Find k-th smallest (1-indexed) using binary lifting. O(log N)."""
    pos = 0
    log_n = n.bit_length()
    for i in range(log_n, -1, -1):
        nxt = pos + (1 << i)
        if nxt <= n and bit.tree[nxt] < k:
            k -= bit.tree[nxt]
            pos = nxt
    return pos + 1  # 1-indexed answer
```

### 2D Fenwick Tree

For 2D prefix sums with point updates:

```python
class FenwickTree2D:
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]

    def update(self, r, c, delta):
        i = r
        while i <= self.rows:
            j = c
            while j <= self.cols:
                self.tree[i][j] += delta
                j += j & (-j)
            i += i & (-i)

    def query(self, r, c):
        """Prefix sum of [1..r][1..c]."""
        total = 0
        i = r
        while i > 0:
            j = c
            while j > 0:
                total += self.tree[i][j]
                j -= j & (-j)
            i -= i & (-i)
        return total
```

### Comparison with Segment Tree

| Feature | Fenwick Tree | Segment Tree |
|---------|-------------|--------------|
| Point update + prefix query | O(log N) | O(log N) |
| Range update + range query | Possible with 2 BITs | O(log N) with lazy prop |
| Arbitrary range query (min/max) | Not supported | O(log N) |
| Memory | N + 1 | 4N |
| Constant factor | Very small | Larger |
| Implementation | ~15 lines | ~60 lines |

*Socratic prompt: "A Fenwick tree stores cumulative information using the binary structure of indices. Why can it compute prefix sums but not arbitrary range minimums?"*

**Practice:** LeetCode 307 (Range Sum Query - Mutable), 315 (Count of Smaller Numbers After Self), 493 (Reverse Pairs)

---

## 3. Segment Tree (Advanced)

The basic segment tree is covered in `data-structure-fundamentals.md`. Here we cover **lazy propagation**, **persistent segment trees**, and **merge sort tree** — the advanced variants needed for competitive programming.

### Lazy Propagation (Full Implementation)

Lazy propagation defers range updates to children until they're needed. Each node stores a pending "lazy" value.

```python
class LazySegTree:
    """Segment tree with lazy propagation for range add + range sum."""

    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self._build(arr, 1, 0, self.n - 1)

    def _build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
            return
        mid = (start + end) // 2
        self._build(arr, 2 * node, start, mid)
        self._build(arr, 2 * node + 1, mid + 1, end)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def _push_down(self, node, start, end):
        """Propagate lazy value to children."""
        if self.lazy[node] != 0:
            mid = (start + end) // 2
            self._apply(2 * node, start, mid, self.lazy[node])
            self._apply(2 * node + 1, mid + 1, end, self.lazy[node])
            self.lazy[node] = 0

    def _apply(self, node, start, end, val):
        """Apply a pending add of val to this node."""
        self.tree[node] += val * (end - start + 1)
        self.lazy[node] += val

    def range_update(self, l, r, val, node=1, start=0, end=None):
        """Add val to all elements in [l, r]. O(log N)."""
        if end is None:
            end = self.n - 1
        if l > end or r < start:
            return
        if l <= start and end <= r:
            self._apply(node, start, end, val)
            return
        self._push_down(node, start, end)
        mid = (start + end) // 2
        self.range_update(l, r, val, 2 * node, start, mid)
        self.range_update(l, r, val, 2 * node + 1, mid + 1, end)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def range_query(self, l, r, node=1, start=0, end=None):
        """Return sum of [l, r]. O(log N)."""
        if end is None:
            end = self.n - 1
        if l > end or r < start:
            return 0
        if l <= start and end <= r:
            return self.tree[node]
        self._push_down(node, start, end)
        mid = (start + end) // 2
        return (self.range_query(l, r, 2 * node, start, mid) +
                self.range_query(l, r, 2 * node + 1, mid + 1, end))
```

*Socratic prompt: "Lazy propagation delays work. When does the deferred update actually happen? What triggers it?"*

### Segment Tree with Multiple Operations

When combining different operations (e.g., range set + range add + range sum), the lazy tag must encode the operation type and composition order matters.

**Common combinations and their lazy tag structures:**

| Operations | Lazy Tag | Composition Rule |
|-----------|----------|-----------------|
| Range add + range sum | `add` value | New lazy = old + delta |
| Range set + range sum | `set` value + flag | Set overwrites previous set/add |
| Range add + range set + range sum | `(set_val, add_val)` | Apply set first, then add |
| Range multiply + range add + range sum | `(mult, add)` | `new = mult * old + add` |

### Persistent Segment Tree

Creates a new "version" of the tree after each update, sharing unchanged nodes with previous versions. Each update creates O(log N) new nodes.

```python
class PersistentNode:
    __slots__ = ['left', 'right', 'val']
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class PersistentSegTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.roots = []  # root of each version
        self.roots.append(self._build(arr, 0, self.n - 1))

    def _build(self, arr, l, r):
        if l == r:
            return PersistentNode(arr[l])
        mid = (l + r) // 2
        left = self._build(arr, l, mid)
        right = self._build(arr, mid + 1, r)
        return PersistentNode(left.val + right.val, left, right)

    def update(self, version, idx, val):
        """Create new version with arr[idx] = val. O(log N) new nodes."""
        new_root = self._update(self.roots[version], 0, self.n - 1, idx, val)
        self.roots.append(new_root)
        return len(self.roots) - 1  # new version number

    def _update(self, node, l, r, idx, val):
        if l == r:
            return PersistentNode(val)
        mid = (l + r) // 2
        if idx <= mid:
            new_left = self._update(node.left, l, mid, idx, val)
            return PersistentNode(new_left.val + node.right.val, new_left, node.right)
        else:
            new_right = self._update(node.right, mid + 1, r, idx, val)
            return PersistentNode(node.left.val + new_right.val, node.left, new_right)
```

**Key application:** k-th smallest in any subarray `[l, r]` — build persistent segment tree on sorted value indices, then diff versions `r` and `l-1`.

### Complexity Summary

| Variant | Build | Query | Update | Space |
|---------|-------|-------|--------|-------|
| Basic | O(N) | O(log N) | O(log N) point | O(N) |
| Lazy propagation | O(N) | O(log N) | O(log N) range | O(N) |
| Persistent | O(N) | O(log N) | O(log N) per version | O(N + Q log N) |

**Practice:** LeetCode 307, 315, 493; Codeforces: Persistent segment tree problems

---

## 4. Treap (Randomized BST)

### Core Idea

A treap is a binary search tree where each node has a **key** (BST property) and a **random priority** (heap property). The random priorities ensure the tree is balanced in expectation — O(log N) height.

### Why It Matters

Treaps support **split** and **merge** operations, which enable:
- Insert/delete in O(log N)
- k-th element queries
- Range reversals
- Implicit indexing (acts as a balanced array with O(log N) insert/delete anywhere)

### Split and Merge

**Split(t, key):** split tree into two trees: all keys ≤ key go left, rest go right.

**Merge(l, r):** merge two trees where all keys in l < all keys in r.

```python
import random

class TreapNode:
    __slots__ = ['key', 'priority', 'size', 'left', 'right']
    def __init__(self, key):
        self.key = key
        self.priority = random.random()
        self.size = 1
        self.left = self.right = None

def size(t):
    return t.size if t else 0

def update(t):
    if t:
        t.size = 1 + size(t.left) + size(t.right)

def split(t, key):
    """Split into (left: keys <= key, right: keys > key). O(log N) expected."""
    if not t:
        return None, None
    if t.key <= key:
        t.right, right = split(t.right, key)
        update(t)
        return t, right
    else:
        left, t.left = split(t.left, key)
        update(t)
        return left, t

def merge(left, right):
    """Merge two treaps. All keys in left < all keys in right. O(log N) expected."""
    if not left or not right:
        return left or right
    if left.priority > right.priority:
        left.right = merge(left.right, right)
        update(left)
        return left
    else:
        right.left = merge(left, right.left)
        update(right)
        return right

def insert(root, key):
    left, right = split(root, key)
    return merge(merge(left, TreapNode(key)), right)

def erase(root, key):
    """Remove one occurrence of key."""
    left, mid_right = split(root, key - 1)   # left: keys < key
    mid, right = split(mid_right, key)        # mid: keys == key
    # Remove one node from mid by merging its children
    if mid:
        mid = merge(mid.left, mid.right)
    return merge(merge(left, mid), right)
```

### Implicit Treap (Indexed Sequence)

Replace explicit keys with implicit indices (derived from subtree sizes). This turns the treap into a balanced sequence supporting O(log N) insert/delete/split at any position.

```python
def implicit_split(t, pos):
    """Split by position: first `pos` elements go left."""
    if not t:
        return None, None
    left_size = size(t.left)
    if left_size >= pos:
        t.left, right = implicit_split(t.left, pos)
        update(t)
        return right is None and (t, None) or (t.left, merge_node(right, t))
        # Simplified:
        left_part, t.left = implicit_split(t.left, pos)
        update(t)
        return left_part, t
    else:
        t.right, right_part = implicit_split(t.right, pos - left_size - 1)
        update(t)
        return t, right_part
```

*Socratic prompt: "A treap combines BST and heap properties. The BST property ensures sorted order; the heap property (with random priorities) ensures balance. What would happen if priorities weren't random?"*

### Complexity

| Operation | Expected Time |
|-----------|---------------|
| Split | O(log N) |
| Merge | O(log N) |
| Insert | O(log N) |
| Delete | O(log N) |
| k-th element | O(log N) |
| Range operations | O(log N) |

---

## 5. Sqrt Decomposition

### Core Idea

Divide an array of N elements into blocks of size ~√N. Precompute the answer for each complete block. For a range query, combine O(√N) complete blocks plus at most 2 partial blocks.

```python
import math

class SqrtDecomp:
    """Sqrt decomposition for range sum queries with point updates."""

    def __init__(self, arr):
        self.arr = arr[:]
        self.n = len(arr)
        self.block = max(1, int(math.isqrt(self.n)))
        self.blocks = [0] * ((self.n + self.block - 1) // self.block)
        for i, val in enumerate(arr):
            self.blocks[i // self.block] += val

    def update(self, i, val):
        """Set arr[i] = val. O(1)."""
        self.blocks[i // self.block] += val - self.arr[i]
        self.arr[i] = val

    def query(self, l, r):
        """Return sum of arr[l..r]. O(√N)."""
        total = 0
        bl, br = l // self.block, r // self.block
        if bl == br:
            # Same block — iterate directly
            return sum(self.arr[l:r + 1])
        # Left partial block
        total += sum(self.arr[l:(bl + 1) * self.block])
        # Complete blocks in between
        for b in range(bl + 1, br):
            total += self.blocks[b]
        # Right partial block
        total += sum(self.arr[br * self.block:r + 1])
        return total
```

### When to Use

| Scenario | Sqrt Decomp | Segment Tree |
|----------|-------------|--------------|
| Implementation complexity | Low | Medium-High |
| Query/Update time | O(√N) | O(log N) |
| Range updates (add to range) | O(√N) with lazy blocks | O(log N) with lazy prop |
| Offline/batched operations | Mo's algorithm (√N-based) | Not applicable |
| Constant factor | Small | Larger |

*Socratic prompt: "Why √N specifically? What happens if you use blocks of size N^(1/3) instead? When might that be better?"*

### Mo's Algorithm (Offline Range Queries)

Process range queries offline by sorting them in a specific order that minimizes pointer movement:

```python
def mo_algorithm(arr, queries):
    """Process range queries offline in O((N + Q) * √N)."""
    n = len(arr)
    block = max(1, int(math.isqrt(n)))

    # Sort queries: primary by l // block, secondary by r
    # (with right-endpoint parity trick for better cache behavior)
    indexed = [(l, r, i) for i, (l, r) in enumerate(queries)]
    indexed.sort(key=lambda x: (x[0] // block, x[1] if (x[0] // block) % 2 == 0 else -x[1]))

    answers = [0] * len(queries)
    cur_l, cur_r = 0, -1
    current_answer = 0  # maintain running answer

    def add(idx):
        nonlocal current_answer
        # Add arr[idx] to current window
        current_answer += arr[idx]  # example: sum

    def remove(idx):
        nonlocal current_answer
        # Remove arr[idx] from current window
        current_answer -= arr[idx]

    for l, r, qi in indexed:
        while cur_r < r:
            cur_r += 1
            add(cur_r)
        while cur_l > l:
            cur_l -= 1
            add(cur_l)
        while cur_r > r:
            remove(cur_r)
            cur_r -= 1
        while cur_l < l:
            remove(cur_l)
            cur_l += 1
        answers[qi] = current_answer

    return answers
```

**Practice:** LeetCode queries that can be processed offline; Codeforces: Powerful array, XOR and Favorite Number

---

## 6. Sqrt Tree

### Core Idea

A sqrt tree answers **arbitrary associative range queries in O(1)** after O(N log log N) preprocessing. It generalizes sparse tables to non-idempotent operations (like sum, XOR) while keeping O(1) query time.

### How It Differs from Sparse Table

| Feature | Sparse Table | Sqrt Tree |
|---------|-------------|-----------|
| Idempotent ops (min, max) | O(1) query | O(1) query |
| Non-idempotent ops (sum, xor) | O(log N) query | O(1) query |
| Updates | Not supported | O(√N) per update |
| Build time | O(N log N) | O(N log log N) |

### Structure (Conceptual)

1. Divide array into blocks of size √N
2. Precompute prefix and suffix answers within each block
3. Precompute inter-block answers for all block pairs
4. Recurse on the "between" array to handle multi-block queries

For a query `[l, r]`:
- **Same block:** use prefix/suffix arrays → O(1)
- **Adjacent blocks:** combine suffix + prefix → O(1)
- **Multiple blocks:** suffix + between[block_l+1..block_r-1] + prefix → O(1) via recursive sqrt tree on blocks

### Complexity

| Operation | Time |
|-----------|------|
| Build | O(N log log N) |
| Query | O(1) |
| Point update | O(√N log log N) |
| Space | O(N log log N) |

*Socratic prompt: "The sqrt tree achieves O(1) for non-idempotent queries. What's the trade-off compared to a segment tree or sparse table?"*

---

## 7. Randomized Heap (Mergeable Heap)

### Core Idea

A min-heap where the core operation is **merge**: combining two heaps in O(log N) expected time. All other operations reduce to merge:

- **Insert:** merge heap with a single-node heap
- **Extract-min:** merge left and right children of root
- **Delete arbitrary node:** merge its children, replace it with the result

### The Merge Algorithm

```python
import random

class HeapNode:
    __slots__ = ['val', 'left', 'right']
    def __init__(self, val):
        self.val = val
        self.left = self.right = None

def merge(t1, t2):
    """Merge two min-heaps. O(log N) expected."""
    if not t1 or not t2:
        return t1 or t2
    if t2.val < t1.val:
        t1, t2 = t2, t1
    # Randomly choose which child to merge into
    if random.random() < 0.5:
        t1.left, t1.right = t1.right, t1.left
    t1.left = merge(t1.left, t2)
    return t1

def insert(heap, val):
    return merge(heap, HeapNode(val))

def extract_min(heap):
    """Remove and return the minimum. O(log N) expected."""
    if not heap:
        raise IndexError("empty heap")
    min_val = heap.val
    heap = merge(heap.left, heap.right)
    return min_val, heap
```

### Why Randomization Works

The random child swap ensures the expected path length from root to leaf is O(log N). The proof uses the fact that E[h(T)] ≤ log(n+1) by induction with AM-GM inequality.

### Comparison

| Heap Type | Insert | Extract-Min | Merge | Decrease-Key |
|-----------|--------|-------------|-------|-------------|
| Binary heap | O(log N) | O(log N) | O(N) | O(log N) |
| Fibonacci heap | O(1) amort | O(log N) amort | O(1) | O(1) amort |
| Randomized heap | O(log N) exp | O(log N) exp | O(log N) exp | O(log N) exp |
| Leftist heap | O(log N) | O(log N) | O(log N) | — |

*Socratic prompt: "Standard binary heaps can't merge efficiently — it takes O(N). Why? What structural property of randomized/leftist heaps enables O(log N) merge?"*

---

## 8. Deleting from Data Structures in O(T(N) log N)

### The Problem

You have a data structure that supports **adding** elements in O(T(N)) per operation. How do you support **deletion** efficiently, even if the structure doesn't natively support it?

### The Technique: Segment Tree over Time

Each element has a "lifetime" — the range of queries during which it exists (between its addition and deletion). Build a segment tree over the query timeline. Each element's lifetime splits into O(log Q) nodes.

### Algorithm

1. Map each element to its lifetime interval `[add_time, delete_time]`
2. Insert each element into O(log Q) nodes of the segment tree
3. DFS the segment tree: entering a node → add its elements; leaving → rollback
4. At leaves (actual queries), the data structure reflects the correct state

```python
def solve_with_deletion(operations, n):
    """
    operations: list of (type, args) where type is 'add', 'delete', or 'query'

    The data structure must support:
    - add(element) in O(T(N))
    - rollback() to undo the last add in O(T(N))
    """
    # Phase 1: Determine lifetimes
    # Phase 2: Build segment tree over query indices
    # Phase 3: DFS — add on enter, rollback on leave
    pass  # (See DSU with rollback above for a complete example)
```

### Complexity

If the base data structure supports add in O(T(N)):
- Total time: O(Q · T(N) · log Q) — each element appears in O(log Q) segment tree nodes
- Requires the structure to support **rollback** (undo last operation)

**Caveat:** This technique works **offline** only — all operations must be known in advance.

*Socratic prompt: "This technique requires rollback, which breaks amortized complexity. Why? Think about what happens when you undo an operation that triggered an amortized 'expensive' step."*

### Key Application: Dynamic Connectivity

Given a graph where edges are added and removed over time, answer "how many connected components?" at each query. Solution: DSU with rollback + segment tree over time.

---

## Comparison Table: When to Use What

| Structure | Best For | Build | Query | Update | Space |
|-----------|----------|-------|-------|--------|-------|
| DSU | Connectivity, MST | O(N) | O(α(N)) ≈ O(1) | O(α(N)) | O(N) |
| Fenwick Tree | Prefix sums, inversions | O(N) | O(log N) | O(log N) | O(N) |
| Segment Tree | Any range query + update | O(N) | O(log N) | O(log N) | O(4N) |
| Persistent Seg Tree | Versioned queries | O(N) | O(log N) | O(log N) | O(N + Q log N) |
| Treap | Ordered set with split/merge | O(N log N) | O(log N) | O(log N) | O(N) |
| Sqrt Decomposition | Simple range queries, Mo's algo | O(N) | O(√N) | O(1) | O(N) |
| Sqrt Tree | O(1) query, non-idempotent ops | O(N log log N) | O(1) | O(√N) | O(N log log N) |
| Sparse Table | Static RMQ | O(N log N) | O(1) | N/A | O(N log N) |
| Randomized Heap | Mergeable priority queue | O(1) | O(1) top | O(log N) merge | O(N) |

---

## Attribution

Content synthesized from cp-algorithms.com articles on [DSU](https://cp-algorithms.com/data_structures/disjoint_set_union.html), [Fenwick Tree](https://cp-algorithms.com/data_structures/fenwick.html), [Segment Tree](https://cp-algorithms.com/data_structures/segment_tree.html), [Treap](https://cp-algorithms.com/data_structures/treap.html), [Sqrt Decomposition](https://cp-algorithms.com/data_structures/sqrt_decomposition.html), [Sqrt Tree](https://cp-algorithms.com/data_structures/sqrt-tree.html), [Randomized Heap](https://cp-algorithms.com/data_structures/randomized_heap.html), and [Deleting in O(T(n) log n)](https://cp-algorithms.com/data_structures/deleting_in_log_n.html). Algorithms and complexity analyses are from the original articles; code has been translated to Python with added commentary for the leetcode-teacher reference format.
