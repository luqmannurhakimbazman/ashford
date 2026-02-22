# Advanced Shortest Path Algorithms

Seven shortest path algorithm families covering dense/sparse Dijkstra variants, negative-weight handling, 0-1 BFS, D'Esopo-Pape, all-pairs shortest paths, and fixed-length paths via matrix exponentiation. Extends the basic Dijkstra and Bellman-Ford overviews in `graph-algorithms.md` and builds on the DP foundation in `dynamic-programming-core.md`.

---

## Quick Reference Table

| Pattern | Recognition Signals | Key Problems | Section |
|---------|-------------------|--------------|---------|
| Dijkstra (Dense, O(n^2)) | "Shortest path", dense graph (m ~ n^2), adjacency matrix, small n | Network Delay Time (743), Path With Minimum Effort (1631) | 1 |
| Dijkstra (Sparse, Heap) | "Shortest path", sparse graph, large n, adjacency list, non-negative weights | Network Delay Time (743), Cheapest Flights Within K Stops (787), Swim in Rising Water (778) | 2 |
| Bellman-Ford | "Negative weights", "detect negative cycle", "at most K edges", SPFA | Cheapest Flights Within K Stops (787), Negative Weight Cycle detection | 3 |
| 0-1 BFS | Edge weights only 0 or 1, "minimum flips/swaps", grid with free/costly moves | Minimum Cost to Make at Least One Valid Path (1368), Shortest Path in Binary Matrix variant | 4 |
| D'Esopo-Pape | Fast heuristic SSSP, negative edges allowed, no negative cycles, practical speed | General SSSP in competitive programming | 5 |
| Floyd-Warshall | "All pairs shortest path", "between every pair", small n (< 500), transitive closure | Find the City With Smallest Number of Neighbors (1334), Shortest Path Visiting All Nodes (847) | 6 |
| Fixed-Length Paths (Matrix Exp.) | "Exactly k edges", "path of length k", adjacency matrix exponentiation | Knight Dialer (935), Count Paths of Length K | 7 |

---

## Master Comparison: When to Use Which Algorithm

| Algorithm | Weights | Negative Edges | Negative Cycles | Time | Space | Best For |
|-----------|---------|---------------|-----------------|------|-------|----------|
| Dijkstra (Dense) | Non-negative | No | N/A | O(n^2 + m) | O(n) | Dense graphs, small n |
| Dijkstra (Heap) | Non-negative | No | N/A | O(m log n) | O(n + m) | Sparse graphs, large n |
| Bellman-Ford | Any | Yes | Detects | O(n * m) | O(n) | Negative weights, K-edge constraint |
| SPFA | Any | Yes | Detects | O(m) avg, O(n*m) worst | O(n) | Practical negative-weight graphs |
| 0-1 BFS | {0, 1} only | No | N/A | O(m) | O(n) | Binary-weight graphs |
| D'Esopo-Pape | Any | Yes | No (undefined) | O(m) avg, exp worst | O(n) | Fast heuristic, random graphs |
| Floyd-Warshall | Any | Yes | Detects | O(n^3) | O(n^2) | All-pairs, small n |
| Matrix Exp. | Any | Yes | N/A | O(n^3 log k) | O(n^2) | Exactly k edges |

*Socratic prompt: "You have a graph with 500 nodes, 1000 edges, and some negative weights but no negative cycles. Which algorithm do you pick and why?"*

*Socratic prompt: "If someone says 'just use Dijkstra', what assumptions are they making about the graph?"*

---

## 1. Dijkstra's Algorithm (Dense Graphs, O(n^2))

### Core Insight

On dense graphs where m is close to n^2, the overhead of maintaining a heap is wasted — a simple linear scan to find the minimum-distance unvisited vertex is just as efficient. The algorithm greedily selects the closest unprocessed vertex, marks it final, and relaxes its outgoing edges.

The correctness proof hinges on one invariant: once a vertex is marked, its distance is optimal and will never change. This works because all edge weights are non-negative — no future path through unvisited vertices can be shorter.

*Socratic prompt: "Why does this greedy choice fail when edges can have negative weights? Can you construct a small counterexample?"*

### Template

```python
def dijkstra_dense(adj_matrix: list[list[float]], src: int) -> tuple[list[float], list[int]]:
    """Dijkstra's algorithm for dense graphs using adjacency matrix.

    Time:  O(n^2 + m) — optimal when m ~ n^2.
    Space: O(n) for dist/prev arrays (matrix is input).

    Args:
        adj_matrix: n x n matrix where adj_matrix[u][v] = weight (float('inf') if no edge).
        src: Source vertex index.

    Returns:
        (dist, prev) — shortest distances and predecessor array for path reconstruction.
    """
    n = len(adj_matrix)
    INF = float('inf')
    dist = [INF] * n
    prev = [-1] * n
    visited = [False] * n

    dist[src] = 0

    for _ in range(n):
        # Find unvisited vertex with minimum distance — O(n) scan
        u = -1
        for v in range(n):
            if not visited[v] and (u == -1 or dist[v] < dist[u]):
                u = v

        if dist[u] == INF:
            break  # Remaining vertices are unreachable

        visited[u] = True

        # Relax all edges from u
        for v in range(n):
            if adj_matrix[u][v] < INF and dist[u] + adj_matrix[u][v] < dist[v]:
                dist[v] = dist[u] + adj_matrix[u][v]
                prev[v] = u

    return dist, prev


def reconstruct_path(prev: list[int], target: int) -> list[int]:
    """Reconstruct shortest path from source to target using predecessor array."""
    path = []
    cur = target
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]
```

| Operation | Time | Space |
|-----------|------|-------|
| Find min unvisited | O(n) per iteration | O(1) |
| Relax edges from u | O(n) per iteration (matrix row scan) | O(1) |
| Total (n iterations) | O(n^2 + m) | O(n) |

*Socratic prompt: "When n = 1000 and m = 500000, is this version faster or slower than the heap version? At what m/n ratio does the crossover happen?"*

### Practice Problems

| Problem (LC#) | Key Twist |
|--------------|-----------|
| Network Delay Time (743) | Direct SSSP; try both dense and sparse versions to compare |
| Path With Minimum Effort (1631) | Minimax path — modify relaxation to track max edge on path |

---

## 2. Dijkstra's Algorithm (Sparse Graphs, Heap-Based)

### Core Insight

For sparse graphs (m is much less than n^2), the O(n) linear scan to find the minimum is wasteful. A min-heap (priority queue) reduces extraction to O(log n). The trick: since Python's `heapq` lacks a decrease-key operation, we allow duplicate entries and skip stale ones with a simple distance check.

The key line is `if d_v != dist[u]: continue` — this discards old (distance, vertex) pairs that were superseded by a later relaxation. Without this guard, the algorithm would process vertices multiple times with suboptimal distances.

*Socratic prompt: "Why is the 'lazy deletion' approach (allowing duplicates + skipping stale entries) simpler than implementing decrease-key, and does it change the asymptotic complexity?"*

### Template

```python
import heapq


def dijkstra_sparse(adj: list[list[tuple[int, int]]], src: int) -> tuple[list[float], list[int]]:
    """Dijkstra's algorithm for sparse graphs using a min-heap.

    Time:  O(m log n) — each edge triggers at most one heap push.
    Space: O(n + m) for adjacency list + heap.

    Args:
        adj: Adjacency list where adj[u] = [(v, weight), ...].
        src: Source vertex index.

    Returns:
        (dist, prev) — shortest distances and predecessor array.
    """
    n = len(adj)
    INF = float('inf')
    dist = [INF] * n
    prev = [-1] * n

    dist[src] = 0
    heap = [(0, src)]  # (distance, vertex)

    while heap:
        d_v, u = heapq.heappop(heap)

        # Skip stale entries (lazy deletion)
        if d_v != dist[u]:
            continue

        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                heapq.heappush(heap, (dist[v], v))

    return dist, prev
```

### Using `SortedList` (Decrease-Key Alternative)

For problems requiring true decrease-key (e.g., when heap size matters), use `sortedcontainers.SortedList`:

```python
from sortedcontainers import SortedList


def dijkstra_sorted_list(adj: list[list[tuple[int, int]]], src: int) -> list[float]:
    """Dijkstra using SortedList for O(log n) add/remove.

    Time:  O(m log n).
    Space: O(n).
    """
    n = len(adj)
    INF = float('inf')
    dist = [INF] * n
    dist[src] = 0

    sl = SortedList([(0, src)])

    while sl:
        d_v, u = sl.pop(0)  # Extract minimum

        if d_v > dist[u]:
            continue

        for v, w in adj[u]:
            new_dist = dist[u] + w
            if new_dist < dist[v]:
                if dist[v] != INF:
                    sl.discard((dist[v], v))  # Remove old entry
                dist[v] = new_dist
                sl.add((dist[v], v))

    return dist
```

| Variant | Extract-Min | Decrease-Key | Overall |
|---------|------------|-------------|---------|
| Min-heap (lazy deletion) | O(log n) | N/A (push new) | O(m log n) |
| SortedList | O(log n) | O(log n) | O(m log n) |
| Fibonacci heap (theory) | O(1) amortized | O(1) amortized | O(m + n log n) |

*Socratic prompt: "In competitive programming, the min-heap version is almost always preferred over SortedList. Why might that be, despite identical asymptotic complexity?"*

### Practice Problems

| Problem (LC#) | Key Twist |
|--------------|-----------|
| Network Delay Time (743) | Textbook Dijkstra on sparse graph |
| Cheapest Flights Within K Stops (787) | Add state dimension: (cost, node, stops_left) |
| Swim in Rising Water (778) | Minimax path — use max instead of sum in relaxation |
| Path With Maximum Probability (1514) | Max-product path — negate log of probabilities or use max-heap |
| Shortest Path in Binary Matrix (1091) | BFS on unweighted grid (Dijkstra overkill but works) |
| Minimum Cost to Reach Destination in Time (1928) | Two-dimensional state: (node, time_remaining) |

---

## 3. Bellman-Ford Algorithm

### Core Insight

Bellman-Ford trades speed for generality: it handles **negative edge weights** by relaxing all edges n-1 times. After k phases, all shortest paths using at most k edges are correct. Since any shortest path has at most n-1 edges, n-1 phases suffice.

The critical bonus: a **negative cycle** exists if and only if any edge can still be relaxed during the n-th phase. This makes Bellman-Ford the go-to algorithm for negative cycle detection.

*Socratic prompt: "Why exactly n-1 phases? What's the relationship between phase count and path length?"*

### Template

```python
def bellman_ford(n: int, edges: list[tuple[int, int, int]], src: int
                 ) -> tuple[list[float], list[int], bool]:
    """Bellman-Ford algorithm with negative cycle detection.

    Time:  O(n * m) — n-1 relaxation phases over all m edges.
    Space: O(n) for dist/prev arrays.

    Args:
        n: Number of vertices.
        edges: List of (u, v, weight) directed edges.
        src: Source vertex index.

    Returns:
        (dist, prev, has_negative_cycle).
    """
    INF = float('inf')
    dist = [INF] * n
    prev = [-1] * n
    dist[src] = 0

    # Phase 1..n-1: relax all edges
    for i in range(n - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] < INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                updated = True
        if not updated:
            break  # Early termination — no changes means we're done

    # Phase n: check for negative cycles
    has_negative_cycle = False
    for u, v, w in edges:
        if dist[u] < INF and dist[u] + w < dist[v]:
            has_negative_cycle = True
            break

    return dist, prev, has_negative_cycle
```

### Negative Cycle Extraction

```python
def find_negative_cycle(n: int, edges: list[tuple[int, int, int]]
                        ) -> list[int] | None:
    """Find and return a negative cycle if one exists.

    Time:  O(n * m).
    Space: O(n).

    Returns:
        List of vertices forming the cycle, or None.
    """
    INF = float('inf')
    dist = [0] * n  # Start all at 0 to detect cycles reachable from any vertex
    prev = [-1] * n
    last_relaxed = -1

    for i in range(n):
        last_relaxed = -1
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                last_relaxed = v

    if last_relaxed == -1:
        return None  # No negative cycle

    # Walk back n times to guarantee we're on the cycle
    v = last_relaxed
    for _ in range(n):
        v = prev[v]

    # Trace the cycle
    cycle = []
    cur = v
    while True:
        cycle.append(cur)
        cur = prev[cur]
        if cur == v:
            cycle.append(v)
            break

    return cycle[::-1]
```

### Bellman-Ford with K-Edge Constraint

A powerful variant for "cheapest flight with at most K stops":

```python
def bellman_ford_k_edges(n: int, edges: list[tuple[int, int, int]],
                         src: int, dst: int, k: int) -> float:
    """Shortest path using at most k edges.

    Time:  O(k * m).
    Space: O(n).

    Key insight: Run only k relaxation phases instead of n-1.
    Must copy dist array each phase to prevent using paths with more edges.
    """
    INF = float('inf')
    dist = [INF] * n
    dist[src] = 0

    for _ in range(k):
        dist_copy = dist[:]  # Freeze current phase distances
        for u, v, w in edges:
            if dist_copy[u] < INF and dist_copy[u] + w < dist[v]:
                dist[v] = dist_copy[u] + w

    return dist[dst]
```

*Socratic prompt: "Why do we need to copy the dist array in the K-edge variant but not in standard Bellman-Ford? What goes wrong without the copy?"*

### SPFA (Shortest Path Faster Algorithm)

SPFA is a queue-based optimization of Bellman-Ford that only relaxes edges from vertices whose distances recently changed:

```python
from collections import deque


def spfa(adj: list[list[tuple[int, int]]], src: int) -> tuple[list[float], bool]:
    """SPFA — queue-optimized Bellman-Ford.

    Time:  O(m) average, O(n * m) worst case.
    Space: O(n).

    Returns:
        (dist, has_negative_cycle).
    """
    n = len(adj)
    INF = float('inf')
    dist = [INF] * n
    dist[src] = 0
    in_queue = [False] * n
    count = [0] * n  # Times each vertex enters the queue

    q = deque([src])
    in_queue[src] = True

    while q:
        u = q.popleft()
        in_queue[u] = False

        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if not in_queue[v]:
                    q.append(v)
                    in_queue[v] = True
                    count[v] += 1
                    if count[v] >= n:
                        return dist, True  # Negative cycle detected

    return dist, False
```

| Variant | Time | Negative Edges | Negative Cycle Detection |
|---------|------|---------------|--------------------------|
| Standard Bellman-Ford | O(n * m) | Yes | Yes (n-th phase check) |
| Early termination | O(n * m) worst, often faster | Yes | Yes |
| K-edge constrained | O(k * m) | Yes | N/A |
| SPFA | O(m) avg, O(n * m) worst | Yes | Yes (queue count >= n) |

### Practice Problems

| Problem (LC#) | Key Twist |
|--------------|-----------|
| Cheapest Flights Within K Stops (787) | K-edge Bellman-Ford with dist copy |
| Negative Weight Cycle (no LC) | Run n-th phase, trace back predecessors |
| Network Delay Time (743) | Works but slower than Dijkstra for non-negative weights |

---

## 4. 0-1 BFS (Deque Trick)

### Core Insight

When all edge weights are either 0 or 1, we can exploit a deque instead of a heap. The key observation: at any point the deque contains vertices with at most two distinct distance values d and d+1. Weight-0 edges keep us at the same distance tier (push front), while weight-1 edges advance to the next tier (push back). This maintains the sorted order invariant without any heap operations.

*Socratic prompt: "Regular BFS works on unweighted graphs. 0-1 BFS handles {0,1} weights. What would you need for {0,1,2} weights? (Hint: think about Dial's algorithm.)"*

### Template

```python
from collections import deque


def bfs_01(adj: list[list[tuple[int, int]]], src: int) -> list[float]:
    """0-1 BFS for graphs with edge weights in {0, 1}.

    Time:  O(V + E) — each vertex/edge processed at most once.
    Space: O(V) for deque and dist array.

    Args:
        adj: Adjacency list where adj[u] = [(v, w)] with w in {0, 1}.
        src: Source vertex.

    Returns:
        Shortest distances from src to all vertices.
    """
    n = len(adj)
    INF = float('inf')
    dist = [INF] * n
    dist[src] = 0

    dq = deque([src])

    while dq:
        u = dq.popleft()

        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if w == 0:
                    dq.appendleft(v)  # Same distance tier — front
                else:
                    dq.append(v)      # Next distance tier — back

    return dist
```

### Grid Application: Minimum Flips

Many grid problems reduce to 0-1 BFS. For example, "minimum flips to create a path" — moving in the arrow's direction costs 0, against it costs 1:

```python
def min_cost_path(grid: list[list[int]]) -> int:
    """Minimum cost to reach (m-1, n-1) from (0, 0).

    Grid values: 1=right, 2=left, 3=down, 4=up.
    Following the arrow costs 0; changing direction costs 1.

    Time:  O(m * n).
    Space: O(m * n).
    """
    m, n = len(grid), len(grid[0])
    INF = float('inf')
    dist = [[INF] * n for _ in range(m)]
    dist[0][0] = 0

    # direction_map[grid_val] = (dr, dc)
    direction_map = {1: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0)}
    all_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    dq = deque([(0, 0)])

    while dq:
        r, c = dq.popleft()

        for dr, dc in all_dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n:
                # Cost 0 if moving in the arrow's direction, else 1
                cost = 0 if direction_map[grid[r][c]] == (dr, dc) else 1
                if dist[r][c] + cost < dist[nr][nc]:
                    dist[nr][nc] = dist[r][c] + cost
                    if cost == 0:
                        dq.appendleft((nr, nc))
                    else:
                        dq.append((nr, nc))

    return dist[m - 1][n - 1]
```

| Operation | Time | Space |
|-----------|------|-------|
| Process each vertex | O(1) amortized | O(1) |
| Process each edge | O(1) | O(1) |
| Total | O(V + E) | O(V) |

*Socratic prompt: "0-1 BFS is O(V+E) while Dijkstra is O(E log V). For a grid of size 1000x1000 with binary weights, how much faster is 0-1 BFS in practice?"*

### Extension: Dial's Algorithm

For edge weights in {0, 1, ..., k}, maintain k+1 buckets and cycle through them. This generalizes 0-1 BFS to small integer weights with O(V + E) time.

### Practice Problems

| Problem (LC#) | Key Twist |
|--------------|-----------|
| Minimum Cost to Make at Least One Valid Path in a Grid (1368) | Grid arrows; following = 0, changing = 1 |
| Shortest Path in Binary Matrix (1091) | Unweighted (all 1s), but BFS is simpler |
| Minimum Number of Flips to Convert Binary Matrix to Zero Matrix (1284) | State-space BFS with 0/1 transitions |
| Map of Highest Peak (1765) | Multi-source 0-1 BFS on grid |

---

## 5. D'Esopo-Pape Algorithm

### Core Insight

D'Esopo-Pape is a heuristic SSSP algorithm that partitions vertices into three sets based on processing status:
- **M0**: Already processed (dequeued at least once)
- **M1**: Currently in the deque (being processed)
- **M2**: Never seen (not yet reached)

The trick: when a previously-processed vertex (M0) gets improved, it goes to the **front** of the deque for immediate re-processing. New vertices (M2) go to the **back**. This heuristic often outperforms Dijkstra and Bellman-Ford on random graphs, but has **exponential worst-case** time on adversarial inputs.

*Socratic prompt: "Why does pushing re-discovered vertices to the front help in practice? What kind of graph structure would make this heuristic backfire?"*

### Template

```python
from collections import deque


def desopo_pape(adj: list[list[tuple[int, int]]], src: int) -> tuple[list[float], list[int]]:
    """D'Esopo-Pape shortest path algorithm.

    Time:  O(m) average on random graphs, exponential worst case.
    Space: O(n).

    Handles negative edge weights but NOT negative cycles.

    Args:
        adj: Adjacency list where adj[u] = [(v, weight), ...].
        src: Source vertex.

    Returns:
        (dist, prev) — shortest distances and predecessor array.
    """
    n = len(adj)
    INF = float('inf')
    dist = [INF] * n
    prev = [-1] * n
    # State: 0 = processed (M0), 1 = in queue (M1), 2 = unseen (M2)
    state = [2] * n

    dist[src] = 0
    dq = deque([src])
    state[src] = 1

    while dq:
        u = dq.popleft()
        state[u] = 0  # Mark as processed

        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u

                if state[v] == 2:
                    # Never seen — add to back
                    dq.append(v)
                    state[v] = 1
                elif state[v] == 0:
                    # Previously processed — re-add to front
                    dq.appendleft(v)
                    state[v] = 1
                # If state[v] == 1 (already in queue), do nothing

    return dist, prev
```

| Operation | Time (Average) | Time (Worst) | Space |
|-----------|---------------|-------------|-------|
| Overall | O(m) | Exponential | O(n) |

### Comparison: D'Esopo-Pape vs Dijkstra vs Bellman-Ford

| Property | Dijkstra (Heap) | Bellman-Ford | D'Esopo-Pape |
|----------|----------------|-------------|--------------|
| Negative edges | No | Yes | Yes |
| Negative cycles | N/A | Detects | Undefined (loops) |
| Worst-case time | O(m log n) | O(n * m) | Exponential |
| Practical speed | Good | Slow | Often fastest |
| Reliable for contests | Yes | Yes | Risky |

*Socratic prompt: "Given that D'Esopo-Pape has exponential worst-case, when would you still choose it over Bellman-Ford?"*

### Practice Problems

| Problem (LC#) | Key Twist |
|--------------|-----------|
| (No standard LC problems) | Primarily used in competitive programming where random tests dominate |

---

## 6. Floyd-Warshall (All-Pairs Shortest Paths)

### Core Insight

Floyd-Warshall computes shortest paths between **every pair** of vertices in O(n^3). The DP recurrence considers whether the shortest path from i to j benefits from routing through an intermediate vertex k:

`dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])`

The outer loop iterates over intermediate vertices k = 0, 1, ..., n-1. After phase k, `dist[i][j]` holds the shortest path using only vertices {0, 1, ..., k} as intermediaries. This is a textbook example of the "add one element at a time" DP strategy.

*Socratic prompt: "Why must k be the outer loop? What breaks if you put i or j as the outer loop instead?"*

### Template

```python
def floyd_warshall(dist: list[list[float]]) -> list[list[float]]:
    """Floyd-Warshall all-pairs shortest paths (in-place).

    Time:  O(n^3).
    Space: O(n^2) — modifies dist in-place.

    Args:
        dist: n x n matrix. dist[i][j] = edge weight, float('inf') if no edge,
              0 on diagonal. Modified in-place.

    Returns:
        The modified dist matrix with shortest paths.
    """
    n = len(dist)
    INF = float('inf')

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] < INF and dist[k][j] < INF:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist
```

### Path Reconstruction

```python
def floyd_warshall_with_paths(weight: list[list[float]]
                              ) -> tuple[list[list[float]], list[list[int]]]:
    """Floyd-Warshall with path reconstruction.

    Time:  O(n^3).
    Space: O(n^2).

    Returns:
        (dist, next_hop) — dist[i][j] is shortest distance,
        next_hop[i][j] is the next vertex after i on the path to j.
    """
    n = len(weight)
    INF = float('inf')

    dist = [row[:] for row in weight]
    next_hop = [[-1] * n for _ in range(n)]

    # Initialize next_hop
    for i in range(n):
        for j in range(n):
            if i != j and dist[i][j] < INF:
                next_hop[i][j] = j

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] < INF and dist[k][j] < INF:
                    new_dist = dist[i][k] + dist[k][j]
                    if new_dist < dist[i][j]:
                        dist[i][j] = new_dist
                        next_hop[i][j] = next_hop[i][k]

    return dist, next_hop


def reconstruct_fw_path(next_hop: list[list[int]], u: int, v: int) -> list[int]:
    """Reconstruct path from u to v using next_hop matrix."""
    if next_hop[u][v] == -1:
        return []  # No path exists

    path = [u]
    while u != v:
        u = next_hop[u][v]
        path.append(u)
    return path
```

### Negative Cycle Detection

```python
def has_negative_cycle(dist: list[list[float]]) -> bool:
    """Check for negative cycle after running Floyd-Warshall.

    A negative cycle exists iff any diagonal element is negative.
    """
    return any(dist[i][i] < 0 for i in range(len(dist)))
```

### Transitive Closure Variant

```python
def transitive_closure(adj: list[list[bool]]) -> list[list[bool]]:
    """Compute transitive closure: can vertex i reach vertex j?

    Time:  O(n^3).
    Space: O(n^2).

    Replace min with OR, addition with AND.
    """
    n = len(adj)
    reach = [row[:] for row in adj]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j])

    return reach
```

| Operation | Time | Space |
|-----------|------|-------|
| All-pairs shortest paths | O(n^3) | O(n^2) |
| Path reconstruction | O(path length) per query | O(n^2) for next_hop |
| Negative cycle detection | O(n) post-processing | O(1) |
| Transitive closure | O(n^3) | O(n^2) |

*Socratic prompt: "Floyd-Warshall is O(n^3). Running Dijkstra from every vertex is O(n * m log n). When is Floyd-Warshall actually faster?"*

### Practice Problems

| Problem (LC#) | Key Twist |
|--------------|-----------|
| Find the City With the Smallest Number of Neighbors at a Threshold Distance (1334) | APSP then count reachable cities per vertex |
| Shortest Path Visiting All Nodes (847) | Bitmask DP, not pure Floyd-Warshall, but APSP as preprocessing |
| Course Schedule IV (1462) | Transitive closure variant — can course i be prerequisite of j? |
| Evaluate Division (399) | Floyd-Warshall on weighted graph of variable ratios |
| Minimum Cost to Convert String I (2976) | APSP on 26-node character graph |

---

## 7. Fixed-Length Paths (Matrix Exponentiation)

### Core Insight

To find the shortest path of **exactly k edges**, we define a modified matrix multiplication where:
- Standard addition is replaced by **min**
- Standard multiplication is replaced by **addition**

If G is the adjacency matrix (with `float('inf')` for missing edges), then G^k under this "min-plus" semiring gives the shortest k-edge paths between all pairs. Using binary exponentiation, we compute G^k in O(n^3 log k) time.

This is a direct analogy to counting paths of length k via standard matrix exponentiation on the adjacency matrix — but in the (min, +) semiring instead of the (+, *) semiring.

*Socratic prompt: "Regular matrix multiplication counts paths of length k. How does switching from (+, *) to (min, +) change the meaning from counting to optimization?"*

### Template

```python
def min_plus_multiply(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Multiply two matrices in the (min, +) semiring.

    C[i][j] = min over p of (A[i][p] + B[p][j]).

    Time:  O(n^3).
    Space: O(n^2).
    """
    n = len(A)
    INF = float('inf')
    C = [[INF] * n for _ in range(n)]

    for i in range(n):
        for p in range(n):
            if A[i][p] == INF:
                continue  # Optimization: skip infinite entries
            for j in range(n):
                if B[p][j] < INF:
                    C[i][j] = min(C[i][j], A[i][p] + B[p][j])

    return C


def identity_matrix(n: int) -> list[list[float]]:
    """Identity for (min, +) semiring: 0 on diagonal, inf elsewhere."""
    INF = float('inf')
    I = [[INF] * n for _ in range(n)]
    for i in range(n):
        I[i][i] = 0
    return I


def matrix_exp_min_plus(G: list[list[float]], k: int) -> list[list[float]]:
    """Compute G^k in (min, +) semiring via binary exponentiation.

    Time:  O(n^3 log k).
    Space: O(n^2).

    Args:
        G: n x n adjacency matrix (float('inf') for no edge).
        k: Exact number of edges in the path.

    Returns:
        Result[i][j] = shortest path from i to j using exactly k edges.
    """
    n = len(G)
    result = identity_matrix(n)

    base = [row[:] for row in G]  # Copy to avoid mutation

    while k > 0:
        if k & 1:
            result = min_plus_multiply(result, base)
        base = min_plus_multiply(base, base)
        k >>= 1

    return result
```

### Counting Paths of Length K (Standard Semiring)

For comparison, counting paths uses normal matrix multiplication:

```python
def matrix_multiply_mod(A: list[list[int]], B: list[list[int]],
                        mod: int) -> list[list[int]]:
    """Standard matrix multiplication with modular arithmetic.

    Used for counting paths of length k.
    """
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for p in range(n):
            if A[i][p] == 0:
                continue
            for j in range(n):
                C[i][j] = (C[i][j] + A[i][p] * B[p][j]) % mod
    return C


def count_paths_of_length_k(adj: list[list[int]], k: int,
                             mod: int = 10**9 + 7) -> list[list[int]]:
    """Count paths of exactly k edges between all pairs.

    Time:  O(n^3 log k).
    Space: O(n^2).
    """
    n = len(adj)
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    base = [row[:] for row in adj]

    while k > 0:
        if k & 1:
            result = matrix_multiply_mod(result, base, mod)
        base = matrix_multiply_mod(base, base, mod)
        k >>= 1

    return result
```

### Paths with At Most K Edges

Add a self-loop of weight 0 to each vertex. Then a path using exactly k edges in the modified graph corresponds to a path using at most k edges in the original graph (the extra edges are self-loops contributing 0 cost).

```python
def shortest_at_most_k_edges(G: list[list[float]], k: int) -> list[list[float]]:
    """Shortest paths using at most k edges.

    Time:  O(n^3 log k).
    Space: O(n^2).

    Trick: Add 0-weight self-loops so "wasting" an edge costs nothing.
    """
    n = len(G)
    G_mod = [row[:] for row in G]
    for i in range(n):
        G_mod[i][i] = 0  # Add self-loops

    return matrix_exp_min_plus(G_mod, k)
```

| Operation | Time | Space |
|-----------|------|-------|
| Single min-plus multiply | O(n^3) | O(n^2) |
| Matrix exponentiation | O(n^3 log k) | O(n^2) |
| Shortest path exactly k edges | O(n^3 log k) | O(n^2) |
| Count paths of length k | O(n^3 log k) | O(n^2) |

*Socratic prompt: "Knight Dialer (LC 935) asks for the number of distinct phone numbers of length n. How does this reduce to matrix exponentiation on a 10-node graph?"*

### Practice Problems

| Problem (LC#) | Key Twist |
|--------------|-----------|
| Knight Dialer (935) | Count paths of length n on digit-adjacency graph; standard (+, *) semiring |
| Number of Ways to Arrive at Destination (1976) | Can combine shortest path + count via modified matrix or Dijkstra |
| Cheapest Flights Within K Stops (787) | At-most-k-edges variant (but Bellman-Ford is simpler here) |

---

## Common Pitfalls

1. **Using Dijkstra with negative edges** — Dijkstra's greedy invariant breaks. The marked vertex may not have its final shortest distance. Always check for negative weights first.

2. **Forgetting the stale-entry check in heap Dijkstra** — Without `if d_v != dist[u]: continue`, you process vertices with outdated distances, producing wrong results and TLE.

3. **Not copying dist in K-edge Bellman-Ford** — Without `dist_copy = dist[:]`, relaxations within the same phase chain together, effectively allowing paths longer than k edges.

4. **Floyd-Warshall with k as inner loop** — The intermediate-vertex k **must** be the outermost loop. Any other order computes garbage.

5. **Integer overflow in Floyd-Warshall** — Always guard `dist[i][k] < INF and dist[k][j] < INF` before adding, or `INF + INF` overflows (especially in languages without arbitrary precision).

6. **0-1 BFS on non-binary weights** — The deque invariant requires weights in {0, 1}. For weights in {0, 1, ..., k}, use Dial's algorithm or Dijkstra.

---

## Attribution

Content synthesized from cp-algorithms.com articles on [Dijkstra (Dense)](https://cp-algorithms.com/graph/dijkstra.html), [Dijkstra (Sparse)](https://cp-algorithms.com/graph/dijkstra_sparse.html), [Bellman-Ford](https://cp-algorithms.com/graph/bellman_ford.html), [0-1 BFS](https://cp-algorithms.com/graph/01_bfs.html), [D'Esopo-Pape](https://cp-algorithms.com/graph/desopo_pape.html), [Floyd-Warshall](https://cp-algorithms.com/graph/all-pair-shortest-path-floyd-warshall.html), and [Fixed-Length Paths](https://cp-algorithms.com/graph/fixed_length_paths.html). All code has been translated to Python with added docstrings, complexity annotations, and commentary for the leetcode-teacher reference format.
