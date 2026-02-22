# Advanced Patterns (Miscellaneous)

Remaining advanced patterns not covered by dedicated technique files. For the full original catalog of 13 patterns, see the dedicated files linked below.

---

## Quick Reference Table

| Pattern | Reference |
|---------|-----------|
| N-Sum Generalized | `references/algorithms/n-sum.md` |
| LRU Cache, LFU Cache, Random Set O(1) | `references/algorithms/lru-lfu-cache.md` |
| State Machine DP (Stock Problems) + House Robber | `references/algorithms/stock-problems.md` |
| Subsequence DP | `references/algorithms/subsequence-dp.md` |
| Interval Scheduling (Greedy) | `references/algorithms/interval-scheduling.md` |
| Dijkstra's Algorithm | `references/graphs/dijkstra.md` |

**Remaining patterns in this file:**

| Pattern | Recognition Signals | Key Problems |
|---------|-------------------|--------------|
| Find Median from Stream | "Median of a stream of numbers", "running median" | Find Median from Data Stream (295) |
| Remove Duplicate Letters | "Smallest result after removing duplicates", "remove to make monotonic" | Remove Duplicate Letters (316) |
| Exam Room | "Maximize distance to closest person", "seat assignment" | Exam Room (855) |
| Bipartite Graph | "2-colorable", "possible bipartition", "is graph bipartite" | Is Graph Bipartite (785), Possible Bipartition (886) |

---

## Find Median from Data Stream

### Key Insight

Maintain two heaps that split the data into a smaller half and a larger half:
- **Max-heap (left):** stores the smaller half (top = largest of small half)
- **Min-heap (right):** stores the larger half (top = smallest of large half)

The median is either the top of the max-heap (odd total) or the average of both tops (even total).

### Template

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.small = []    # Max-heap (negate values for Python's min-heap)
        self.large = []    # Min-heap

    def addNum(self, num):
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, -heapq.heappop(self.small))
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2
```

### Invariants

1. `len(small)` is equal to or one more than `len(large)`
2. Every element in `small` ≤ every element in `large`
3. The median is always accessible from the heap tops in O(1)

*Socratic prompt: "If you just sorted the stream each time, that's O(N log N) per query. How do the two heaps give you O(log N) insert and O(1) median?"*

**Example problems:** Find Median from Data Stream (295), Sliding Window Median (480)

---

## Remove Duplicate Letters

### Key Insight

Combine **greedy** + **monotonic stack**: build the smallest possible string by keeping a monotonically increasing stack of characters, but only pop a character if it appears later in the string (so we won't lose it).

### Template

```python
def remove_duplicate_letters(s):
    last_occurrence = {ch: i for i, ch in enumerate(s)}
    stack = []
    in_stack = set()

    for i, ch in enumerate(s):
        if ch in in_stack:
            continue

        while stack and stack[-1] > ch and last_occurrence[stack[-1]] > i:
            in_stack.remove(stack.pop())

        stack.append(ch)
        in_stack.add(ch)

    return ''.join(stack)
```

### The Three Conditions for Popping

A character on the stack gets popped only when ALL three hold:
1. The current character is smaller (greedy: we want lexicographically smallest)
2. The stack-top character appears again later (safe to remove)
3. The current character is not already in the stack

*Socratic prompt: "Why is condition 2 essential? What happens if we pop a character that doesn't appear later?"*

**Example problems:** Remove Duplicate Letters (316), Smallest Subsequence of Distinct Characters (1081)

---

## Exam Room

### Key Insight

Model the problem as choosing the longest "gap" between seated students. When a new student sits, they choose the midpoint of the longest gap.

### Template

```python
import bisect

class ExamRoom:
    def __init__(self, n):
        self.n = n
        self.students = []

    def seat(self):
        if not self.students:
            self.students.append(0)
            return 0

        max_dist = self.students[0]
        best_seat = 0

        for i in range(1, len(self.students)):
            dist = (self.students[i] - self.students[i - 1]) // 2
            if dist > max_dist:
                max_dist = dist
                best_seat = self.students[i - 1] + dist

        if self.n - 1 - self.students[-1] > max_dist:
            best_seat = self.n - 1

        bisect.insort(self.students, best_seat)
        return best_seat

    def leave(self, p):
        self.students.remove(p)
```

**Trade-off:** O(N) per operation. For O(log N), use `SortedList` from `sortedcontainers`.

*Socratic prompt: "Why do we pick the midpoint of the largest gap? What would happen if we always sat at the leftmost empty seat?"*

**Example problems:** Exam Room (855), Maximize Distance to Closest Person (849)

---

## Bipartite Graph Detection

### Key Insight

A graph is bipartite if and only if it is 2-colorable: you can assign each node one of two colors such that no edge connects two same-colored nodes. Equivalently, the graph has no odd-length cycles.

### Template (DFS)

```python
def is_bipartite(graph):
    n = len(graph)
    color = [0] * n  # 0 = uncolored, 1 = color A, -1 = color B

    def dfs(node, c):
        color[node] = c
        for neighbor in graph[node]:
            if color[neighbor] == c:
                return False
            if color[neighbor] == 0:
                if not dfs(neighbor, -c):
                    return False
        return True

    for i in range(n):
        if color[i] == 0:
            if not dfs(i, 1):
                return False
    return True
```

### Common Applications

- **Possible Bipartition (LC 886):** Build a graph of dislikes, check bipartiteness.
- **Is Graph Bipartite (LC 785):** Direct application of the template.
