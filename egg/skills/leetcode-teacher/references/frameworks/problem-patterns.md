# Problem Patterns Catalog

Quick-reference catalog of algorithmic patterns for problem classification and teaching.

---

## Quick Identification Table

| Pattern | Recognition Signals | Example Problems |
|---------|-------------------|-----------------|
| Two Pointers | Sorted array, pair/triplet finding, palindrome | Two Sum II, 3Sum, Container With Most Water |
| Sliding Window | Contiguous subarray/substring, "at most K", max/min window | Longest Substring Without Repeating Characters, Minimum Window Substring |
| Binary Search | Sorted input, "find minimum/maximum that satisfies", monotonic condition | Search in Rotated Sorted Array, Koko Eating Bananas |
| Dynamic Programming | Overlapping subproblems, optimal substructure, "number of ways", "minimum cost" | Climbing Stairs, Longest Common Subsequence, Coin Change |
| DFS/BFS | Tree/graph traversal, connected components, shortest path (unweighted) | Number of Islands, Binary Tree Level Order Traversal |
| Backtracking | "All combinations", "all permutations", "all valid", constraint satisfaction | Subsets, Permutations, N-Queens |
| Greedy | Local optimal leads to global optimal, interval scheduling, "minimum number of" | Jump Game, Merge Intervals, Task Scheduler |
| Hash Table | O(1) lookup needed, frequency counting, "two sum"-style complement finding | Two Sum, Group Anagrams, LRU Cache |
| Heap / Priority Queue | "Kth largest/smallest", merge K sorted, streaming median | Kth Largest Element, Merge K Sorted Lists, Find Median from Data Stream |
| Union-Find | Connected components, cycle detection, dynamic connectivity | Number of Connected Components, Redundant Connection, Accounts Merge |

---

## Pattern Deep Dives

### Two Pointers

Two-pointer techniques fall into three distinct categories. Recognizing which category applies is the first step to solving any two-pointer problem.

#### Category 1: Left-Right (Opposite Direction)

**Recognition Signals:**
- Input is sorted (or should be sorted)
- Need to find pairs/triplets that satisfy a condition
- Palindrome checking
- Container/trapping problems (shrink from both sides)

**When to Use:**
- Reducing O(n^2) pair search to O(n) on sorted data
- Checking symmetry (palindromes)

**Code Template:**
```python
def two_pointers_left_right(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        current = arr[left] + arr[right]
        if current == target:
            return [left, right]
        elif current < target:
            left += 1       # Need larger sum
        else:
            right -= 1      # Need smaller sum
    return []
```

**Example Problems:** Two Sum II (167), 3Sum (15), Container With Most Water (11), Trapping Rain Water (42), Valid Palindrome (125)

#### Category 2: Fast-Slow (Same Direction)

**Recognition Signals:**
- In-place array modification ("remove duplicates", "move zeroes")
- Partitioning without extra space
- One pointer reads, another writes

**When to Use:**
- Modifying an array in-place in O(1) extra space
- Filtering elements while preserving order

**Code Template:**
```python
def two_pointers_fast_slow(nums):
    slow = 0                     # Write pointer (next position to write)
    for fast in range(len(nums)):  # Read pointer (scans every element)
        if should_keep(nums[fast]):
            nums[slow] = nums[fast]
            slow += 1
    return slow                  # New length of modified array
```

**Example Problems:** Remove Duplicates from Sorted Array (26), Move Zeroes (283), Remove Element (27)

*Socratic prompt: "The slow pointer marks where to write next. The fast pointer reads every element. What invariant does this maintain? What's always true about everything to the left of slow?"*

#### Category 3: Expand-From-Center

**Recognition Signals:**
- Palindromic substrings or subsequences
- "Longest palindrome" in a string
- Symmetry around a center point

**When to Use:**
- Finding palindromes by expanding outward from each possible center
- O(n²) but simple and practical (Manacher's algorithm is O(n) but rarely needed in interviews)

**Code Template:**
```python
def expand_from_center(s, left, right):
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    return s[left + 1:right]     # The palindrome found

def longest_palindrome(s):
    result = ""
    for i in range(len(s)):
        # Odd-length palindromes (center is s[i])
        odd = expand_from_center(s, i, i)
        # Even-length palindromes (center is between s[i] and s[i+1])
        even = expand_from_center(s, i, i + 1)
        result = max(result, odd, even, key=len)
    return result
```

**Example Problems:** Longest Palindromic Substring (5), Palindromic Substrings (647)

#### Category Summary

| Category | Direction | Key Idea | Typical Problems |
|----------|-----------|----------|-----------------|
| Left-Right | ← → (opposite) | Sorted pair search, symmetry check | Two Sum II, 3Sum, Container, Palindrome check |
| Fast-Slow | → → (same) | In-place read/write separation | Remove Duplicates, Move Zeroes, Remove Element |
| Expand-From-Center | ← → (outward from point) | Palindrome expansion | Longest Palindromic Substring |

**Linked list fast-slow:** The fast-slow pattern also applies to linked lists (cycle detection, find middle, kth from end), but the mechanics differ because linked lists lack random access. See `linked-list-techniques.md` for linked list-specific patterns.

**Common Pitfalls (all categories):**
- Forgetting to sort when input isn't pre-sorted (left-right category)
- Off-by-one errors with `left < right` vs `left <= right`
- Not handling duplicates in 3Sum-style problems
- Confusing which pointer is the "reader" vs "writer" in fast-slow

---

### Sliding Window

**Recognition Signals:**
- Contiguous subarray or substring
- "At most K distinct", "longest/shortest with condition"
- Window size is fixed or variable

**When to Use:**
- Finding optimal contiguous subarray/substring
- Reducing O(n*k) to O(n) for fixed-size window operations

**Code Template (Variable Window):**
```python
def sliding_window(s, condition):
    left = 0
    window = {}  # or counter
    result = 0
    for right in range(len(s)):
        # Expand: add s[right] to window
        window[s[right]] = window.get(s[right], 0) + 1
        # Shrink: while window violates condition
        while not condition(window):
            window[s[left]] -= 1
            if window[s[left]] == 0:
                del window[s[left]]
            left += 1
        # Update result
        result = max(result, right - left + 1)
    return result
```

**Example Problems:** Longest Substring Without Repeating Characters (3), Minimum Window Substring (76), Longest Repeating Character Replacement (424)

**Labuladong's Three Key Questions:**

Every sliding window problem is defined by answering these three questions:

1. **Q1 — When to expand?** What should happen when `right` moves forward?
2. **Q2 — When to shrink?** Under what condition should `left` advance?
3. **Q3 — When to update the result?** After expanding, after shrinking, or both?

Use the `[left, right)` convention (left-closed, right-open) for consistent boundary handling. See `algorithm-frameworks.md` for the full annotated template.

**Common Pitfalls:**
- Not knowing when to shrink vs expand (answer Q1-Q3 explicitly)
- Forgetting to clean up the window state when shrinking
- Confusing fixed-size vs variable-size window templates

---

### Binary Search

**Recognition Signals:**
- Sorted array or monotonic function
- "Find minimum X such that condition holds"
- Search space can be halved

**When to Use:**
- Classic sorted array search
- "Binary search on answer" — when the answer space is monotonic
- Finding boundaries (first/last occurrence)

**Code Template (Search on Answer):**
```python
def binary_search_on_answer(lo, hi, feasible):
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if feasible(mid):
            hi = mid       # mid might be answer, search left
        else:
            lo = mid + 1   # mid too small, search right
    return lo
```

**Example Problems:** Search in Rotated Sorted Array (33), Koko Eating Bananas (875), Median of Two Sorted Arrays (4)

**Common Pitfalls:**
- Off-by-one with `lo < hi` vs `lo <= hi`
- Integer overflow with `(lo + hi) // 2` — use `lo + (hi - lo) // 2`
- Not identifying that a problem is binary-searchable (the "monotonic condition" insight)

---

### Dynamic Programming

**Recognition Signals:**
- "How many ways", "minimum cost", "maximum profit"
- Overlapping subproblems (same computation repeated)
- Optimal substructure (optimal solution built from optimal sub-solutions)
- Problem can be broken into stages/decisions

**When to Use:**
- Counting problems, optimization problems
- When brute force is exponential but subproblems overlap

**Code Template (Bottom-Up):**
```python
def dp_template(n):
    # 1. Define state: dp[i] = answer for subproblem of size i
    dp = [0] * (n + 1)
    # 2. Base case
    dp[0] = 1  # or whatever the base is
    # 3. Transition
    for i in range(1, n + 1):
        for choice in choices:
            dp[i] = combine(dp[i], dp[i - choice])
    # 4. Answer
    return dp[n]
```

**Example Problems:** Climbing Stairs (70), Coin Change (322), Longest Common Subsequence (1143), House Robber (198)

**Labuladong's 3-Step Framework:**

1. **Clarify the state** — what variables change between subproblems? (index, capacity, holding status, etc.)
2. **Clarify the choices** — at each state, what decisions can you make?
3. **Define dp meaning** — write `dp[state] = ...` in plain English

**Top-down vs Bottom-up:** Top-down (memoized recursion) is mathematical induction — assume smaller subproblems are correct, show how to combine. Bottom-up fills a table from base cases. Both encode the same recurrence; choose based on comfort and whether you need space optimization (bottom-up makes this easier).

See `algorithm-frameworks.md` for the full framework, induction analogy, and two subsequence DP templates (`dp[i]` vs `dp[i][j]`).

**Common Pitfalls:**
- Not identifying the correct state definition (start with step 1 above)
- Wrong transition — missing cases or double-counting
- Not optimizing space when only previous row/state is needed

---

### DFS / BFS

**Recognition Signals:**
- Tree or graph traversal
- "Connected components", "shortest path" (unweighted)
- "All paths", "level-by-level"

**When to Use:**
- DFS: exhaustive exploration, path finding, topological sort
- BFS: shortest path (unweighted), level-order traversal

**Code Template (BFS Shortest Path):**
```python
from collections import deque

def bfs(graph, start, target):
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        node, dist = queue.popleft()
        if node == target:
            return dist
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1
```

**Example Problems:** Number of Islands (200), Binary Tree Level Order Traversal (102), Word Ladder (127), Clone Graph (133)

**BFS Shortest Path Insight:** BFS explores all nodes at distance `d` before any at distance `d+1` (layer-by-layer). This guarantees the first time you reach a node is via the shortest path. For weighted graphs, use Dijkstra instead (see `advanced-patterns.md`).

**Common Pitfalls:**
- Forgetting the visited set (infinite loops)
- DFS stack overflow on deep graphs — consider iterative DFS
- BFS vs DFS choice: BFS for shortest path, DFS for exhaustive search

---

### Backtracking

**Recognition Signals:**
- "All combinations", "all permutations", "all valid configurations"
- Constraint satisfaction
- Decision tree exploration

**When to Use:**
- Generating all valid solutions
- Problems where you need to explore and prune

**Code Template:**
```python
def backtrack(candidates, path, result, start):
    if is_valid(path):
        result.append(path[:])
        return
    for i in range(start, len(candidates)):
        # Skip duplicates if needed
        if i > start and candidates[i] == candidates[i - 1]:
            continue
        path.append(candidates[i])
        backtrack(candidates, path, result, i + 1)  # i+1 for combinations, i for reuse
        path.pop()
```

**Example Problems:** Subsets (78), Permutations (46), Combination Sum (39), N-Queens (51)

**Labuladong's Decision Tree Model:**

Every backtracking problem walks a decision tree with three components:
- **Path** — choices made so far
- **Choice list** — options available at the current node
- **End condition** — when to record a result

The three common variants differ only in how `start` is passed and whether duplicates are skipped:

| Variant | `start` in recursion | Duplicate skip? |
|---------|---------------------|-----------------|
| Unique, no reuse | `i + 1` | No |
| Duplicates, no reuse | `i + 1` | Yes (sort + `nums[i] == nums[i-1]`) |
| Unique, with reuse | `i` | No |

See `algorithm-frameworks.md` for the full template with pre-order/post-order choice/undo pattern.

**Common Pitfalls:**
- Not pruning early enough (performance)
- Forgetting to undo choices (the "pop" step)
- Duplicate handling in combinations with repeated elements — use sort + skip pattern

---

### Greedy

**Recognition Signals:**
- Interval scheduling / merging
- "Minimum number of operations"
- Local optimal choice leads to global optimal (provable)

**When to Use:**
- When you can prove the greedy choice property
- Interval problems, activity selection

**Code Template (Interval Scheduling):**
```python
def interval_schedule(intervals):
    intervals.sort(key=lambda x: x[1])  # Sort by end time
    count = 0
    end = float('-inf')
    for start_i, end_i in intervals:
        if start_i >= end:
            count += 1
            end = end_i
    return count
```

**Example Problems:** Jump Game (55), Merge Intervals (56), Non-overlapping Intervals (435), Task Scheduler (621)

**Common Pitfalls:**
- Assuming greedy works without proof (many "greedy-looking" problems need DP)
- Wrong sorting criterion
- Not handling ties correctly

---

### Hash Table

**Recognition Signals:**
- Need O(1) lookup
- Frequency counting
- "Find complement", "group by property"

**When to Use:**
- Trading space for time
- Reducing O(n^2) search to O(n)

**How It Works Under the Hood:**

A hash table maps keys to array indices via a hash function: `index = hash(key) % capacity`. Two collision resolution strategies:

- **Chaining:** Each bucket holds a list of entries. Simple, handles high load. Used in most language implementations (Python `dict` uses open addressing, Java `HashMap` uses chaining).
- **Open addressing (linear probing):** If a slot is taken, probe the next slot. Cache-friendly but degrades under high load factor.

The **load factor** (entries / capacity) determines performance. When it exceeds a threshold (~0.75), the table resizes — rehashing all entries into a larger array. This is O(N) but amortized O(1) per operation.

**Why O(1) average but O(N) worst case:** With a good hash function and low load factor, buckets have ~1 entry → O(1). With a bad hash function, all keys collide → O(N) scan.

Understanding these internals helps learners reason about when hash tables DON'T give O(1) — high collision rates, poor hash functions, or adversarial inputs. See `data-structure-fundamentals.md` for implementation details.

**Example Problems:** Two Sum (1), Group Anagrams (49), LRU Cache (146), Longest Consecutive Sequence (128)

**Common Pitfalls:**
- Hash collisions in custom hash functions
- Not handling the "element mapping to itself" case (e.g., Two Sum with duplicate values)
- Forgetting that dict operations are amortized O(1), not worst-case
- Assuming iteration order is meaningful (it depends on hash values and table size)

---

### Heap / Priority Queue

**Recognition Signals:**
- "Kth largest/smallest"
- "Merge K sorted"
- Streaming data, running median
- "Top K frequent"

**When to Use:**
- Maintaining a dynamic set where you need quick access to min/max
- Reducing O(n log n) sort to O(n log k)

**Code Template:**
```python
import heapq

def kth_largest(nums, k):
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    return heap[0]
```

**Example Problems:** Kth Largest Element (215), Merge K Sorted Lists (23), Find Median from Data Stream (295), Top K Frequent Elements (347)

**Common Pitfalls:**
- Python's heapq is a min-heap — negate values for max-heap behavior
- Not realizing a heap can replace sorting when only K elements matter
- Custom comparators require wrapper classes or tuples in Python

---

### Union-Find (Disjoint Set Union)

**Recognition Signals:**
- "Connected components" with dynamic edge additions
- Cycle detection in undirected graphs
- "Are X and Y connected?"

**When to Use:**
- Dynamic connectivity queries
- Kruskal's MST algorithm
- Grouping equivalent items

**Code Template:**
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
```

**How It Works Under the Hood:**

Union-Find is a **tree structure stored in an array**. Each element's parent is stored at `parent[i]`. The root of each tree is the "representative" of that connected component. `find(x)` walks up the tree to the root; `union(x, y)` connects two trees by making one root point to the other.

Two optimizations make it near-O(1) amortized:

- **Path compression:** During `find(x)`, make every node on the path point directly to the root. Flattens the tree on each query.
- **Union by rank:** Always attach the shorter tree under the taller tree's root. Prevents the tree from degenerating into a linked list.

With both optimizations, operations are O(α(N)) amortized — where α is the inverse Ackermann function, effectively constant (≤ 4 for any practical N).

**The tree insight:** UF is fundamentally a forest (collection of trees) stored in an array. Path compression and union by rank are tree-balancing techniques. This connects UF to the binary tree centrality thesis — even "flat" data structure operations reduce to tree operations.

**Example Problems:** Number of Connected Components (323), Redundant Connection (684), Accounts Merge (721)

**Common Pitfalls:**
- Forgetting path compression (degrades to O(n) per query)
- Forgetting union by rank (same issue)
- Not tracking component count separately when needed
- Confusing "connected" (same component) with "adjacent" (direct edge)

---

## Pattern Selection Decision Tree

```
Is the input sorted or can it be sorted?
├── Yes → Two Pointers or Binary Search
│   ├── Looking for pairs/triplets? → Two Pointers
│   └── Searching for a threshold? → Binary Search
└── No
    ├── Contiguous subarray/substring? → Sliding Window
    ├── Tree or graph?
    │   ├── Shortest path (unweighted)? → BFS
    │   ├── Exhaustive exploration? → DFS
    │   └── Dynamic connectivity? → Union-Find
    ├── "All combinations/permutations"? → Backtracking
    ├── Overlapping subproblems?
    │   ├── Yes → Dynamic Programming
    │   └── No, but local optimal = global optimal? → Greedy
    ├── Need O(1) lookup? → Hash Table
    └── "Kth element" or "Top K"? → Heap
```

---

## Interleaving: Multi-Pattern Problems

Some problems combine patterns. Use these examples to practice interleaving:

| Problem | Patterns Combined |
|---------|------------------|
| Trapping Rain Water (42) | Two Pointers + Stack (or DP) |
| Word Ladder (127) | BFS + Hash Table |
| Alien Dictionary (269) | Topological Sort (DFS) + Hash Table |
| Minimum Window Substring (76) | Sliding Window + Hash Table |
| Course Schedule II (210) | DFS + Topological Sort |
| Kth Smallest in BST (230) | DFS + Heap (or in-order traversal) |
| Longest Increasing Subsequence (300) | DP + Binary Search |

---

## Further Reading

- **`algorithm-frameworks.md`** — Meta-level frameworks (enumeration principle, recursion-as-tree, full sliding window / DP / backtracking / BFS / state machine templates with derivations)
- **`advanced-patterns.md`** — 9 advanced patterns beyond this catalog (N-Sum, LRU/LFU Cache, stock problems, subsequence DP, house robber, interval scheduling, bipartite graphs, Dijkstra)
