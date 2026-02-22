# Graph Algorithms

Eight graph algorithm families covering cycle detection, topological sort, union-find, shortest paths, and minimum spanning trees. Builds on the graph fundamentals in `data-structure-fundamentals.md`.

---

## Graph Representations for Interviews

### Three Common Representations

**1. Adjacency list** — most common in interviews for sparse graphs:
```python
graph = {0: [1, 2], 1: [0, 3], 2: [0], 3: [1]}
```

**2. Adjacency matrix** — useful for dense graphs or when you need O(1) edge lookups:
```python
matrix = [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]
```

**3. Hash table of hash tables** — simplest to set up in interviews, handles weighted edges naturally:
```python
graph = {0: {1: 5, 2: 3}, 1: {0: 5, 3: 2}}  # {node: {neighbor: weight}}
```

### 2D Matrix as an Implicit Graph

Many problems present a 2D matrix where cells are nodes and neighbors are the 4 (or 8) adjacent cells. No explicit graph construction needed — just use boundary checks:

```python
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up

def get_neighbors(r, c, rows, cols):
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc
```

## Time Complexity

| Algorithm | Time | Space |
|-----------|------|-------|
| DFS | O(\|V\| + \|E\|) | O(\|V\|) |
| BFS | O(\|V\| + \|E\|) | O(\|V\|) |
| Topological Sort | O(\|V\| + \|E\|) | O(\|V\|) |
| Dijkstra | O((\|V\| + \|E\|) log \|V\|) | O(\|V\|) |
| Bellman-Ford | O(\|V\| \* \|E\|) | O(\|V\|) |

## Interview Tips

- **Always track visited nodes** to avoid infinite loops in graphs with cycles. Unlike trees, graphs can have cycles — forgetting `visited` is the most common bug.
- A problem may **look like a tree** in the diagram but actually have cycles. Always ask: "Can there be cycles?"
- For disconnected graphs, iterate over all nodes to ensure every component is explored.

## Corner Cases

- Empty graph (no nodes)
- Graph with one or two nodes
- Disconnected graph (multiple components)
- Graph with cycles
- Graph with self-loops
- Complete graph (every node connected to every other)

## Algorithm Frequency in Interviews

| Frequency | Algorithms |
|-----------|-----------|
| **Common** | BFS, DFS |
| **Uncommon** | Topological Sort, Dijkstra |
| **Almost never** | Bellman-Ford, Floyd-Warshall, Prim's, Kruskal's |

## DFS Template for 2D Matrix

```python
def dfs_matrix(matrix, r, c, visited):
    """DFS on a 2D matrix with visited tracking."""
    rows, cols = len(matrix), len(matrix[0])
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return
    if (r, c) in visited or matrix[r][c] == 0:  # boundary/visited/condition check
        return
    visited.add((r, c))
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        dfs_matrix(matrix, r + dr, c + dc, visited)
```

## BFS Template for 2D Matrix

```python
from collections import deque

def bfs_matrix(matrix, start_r, start_c):
    """BFS on a 2D matrix — useful for shortest path in unweighted grids."""
    rows, cols = len(matrix), len(matrix[0])
    visited = {(start_r, start_c)}
    queue = deque([(start_r, start_c, 0)])  # (row, col, distance)

    while queue:
        r, c, dist = queue.popleft()
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                if matrix[nr][nc] != 0:  # passable cell
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))
```

## Essential & Recommended Practice Questions

| Problem | Difficulty | Key Technique |
|---------|-----------|---------------|
| 01 Matrix (542) | Medium | Multi-source BFS |
| Flood Fill (733) | Easy | DFS/BFS on 2D matrix |
| Number of Islands (200) | Medium | DFS/BFS flood fill or Union-Find |
| Rotting Oranges (994) | Medium | Multi-source BFS with level tracking |
| Clone Graph (133) | Medium | DFS/BFS with hash map for visited |
| Pacific Atlantic Water Flow (417) | Medium | Reverse BFS/DFS from ocean borders |
| Course Schedule (207) | Medium | Cycle detection (DFS or Kahn's) |
| Alien Dictionary (269) | Hard | Build graph from char ordering + topo sort |

---

## Quick Reference Table

| Algorithm | Key Insight | When to Use | Complexity |
|-----------|-------------|-------------|------------|
| Cycle Detection (Directed) | `visited[]` + `onPath[]` (DFS) or in-degree count (BFS) | "Can you finish all courses?", "detect deadlock" | O(V + E) |
| Topological Sort | Reverse postorder (DFS) or Kahn's in-degree (BFS) | "Order of courses", "build order", "alien dictionary" | O(V + E) |
| Union-Find | Path compression + union by rank for near O(1) ops | "Connected components", "redundant connection", "accounts merge" | O(α(N)) per op |
| Eulerian Path | Hierholzer's: follow edges, splice at dead-ends | "Visit every edge exactly once", "reconstruct itinerary" | O(E) |
| Dijkstra | BFS with priority queue for weighted shortest path | "Shortest/cheapest path", "network delay" | O((V+E) log V) |
| A* | Dijkstra + heuristic for goal-directed search | "Shortest path to specific target", grid navigation | O((V+E) log V) |
| Kruskal's MST | Sort edges + union-find greedy | "Minimum cost to connect all points" | O(E log E) |
| Prim's MST | Expand from one node, always pick cheapest edge | Same as Kruskal, better for dense graphs | O(E log V) |

---

## 1. Cycle Detection (Directed Graphs)

### DFS Approach: visited[] + onPath[]

Two separate tracking arrays:
- `visited[v]` — has node `v` been explored at all? (prevents re-exploring)
- `on_path[v]` — is node `v` on the CURRENT recursion path? (detects cycles)

A cycle exists if we encounter a node that is already on the current path.

```python
def has_cycle(graph, n):
    """graph: adjacency list, n: number of nodes."""
    visited = [False] * n
    on_path = [False] * n

    def dfs(node):
        visited[node] = True
        on_path[node] = True

        for neighbor in graph[node]:
            if on_path[neighbor]:
                return True         # Cycle: neighbor is on current path
            if not visited[neighbor]:
                if dfs(neighbor):
                    return True

        on_path[node] = False       # Backtrack: leaving this path
        return False

    # Check all components (graph may be disconnected)
    for i in range(n):
        if not visited[i]:
            if dfs(i):
                return True
    return False
```

**Why two arrays?** A node can be `visited` but NOT on the current path — it was explored via a different branch. Only `on_path` membership indicates a back-edge (cycle).

### BFS Approach: In-Degree Count

If we process all nodes with in-degree 0 (Kahn's algorithm) and the count of processed nodes < total nodes, a cycle exists.

```python
from collections import deque

def has_cycle_bfs(graph, n):
    in_degree = [0] * n
    for u in range(n):
        for v in graph[u]:
            in_degree[v] += 1

    queue = deque(i for i in range(n) if in_degree[i] == 0)
    count = 0

    while queue:
        node = queue.popleft()
        count += 1
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return count != n   # True if cycle exists
```

*Socratic prompt: "Why does a cycle prevent all nodes from reaching in-degree 0? What happens to nodes inside a cycle during Kahn's algorithm?"*

### Problems

| Problem | Key Twist |
|---------|-----------|
| Course Schedule (207) | Direct cycle detection — can we take all courses? |
| Course Schedule II (210) | If no cycle, return a valid ordering (topological sort) |

---

## 2. Topological Sort

### When It Applies

Topological sort works on **directed acyclic graphs (DAGs)** only. It produces a linear ordering of nodes such that for every edge u → v, u comes before v.

### DFS Approach: Reverse Postorder

Add each node to the result in postorder (after visiting all descendants), then reverse.

```python
def topological_sort_dfs(graph, n):
    visited = [False] * n
    on_path = [False] * n
    order = []
    has_cycle = False

    def dfs(node):
        nonlocal has_cycle
        visited[node] = True
        on_path[node] = True

        for neighbor in graph[node]:
            if on_path[neighbor]:
                has_cycle = True
                return
            if not visited[neighbor]:
                dfs(neighbor)

        on_path[node] = False
        order.append(node)   # Postorder: add after all descendants

    for i in range(n):
        if not visited[i]:
            dfs(i)

    if has_cycle:
        return []            # No valid topological order
    return order[::-1]       # Reverse postorder
```

### BFS Approach: Kahn's Algorithm (In-Degree)

Process nodes with in-degree 0 first. When a node is processed, decrement its neighbors' in-degrees.

```python
from collections import deque

def topological_sort_bfs(graph, n):
    in_degree = [0] * n
    for u in range(n):
        for v in graph[u]:
            in_degree[v] += 1

    queue = deque(i for i in range(n) if in_degree[i] == 0)
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != n:
        return []            # Cycle detected
    return order
```

### DFS vs BFS Comparison

| Aspect | DFS (Reverse Postorder) | BFS (Kahn's) |
|--------|------------------------|---------------|
| Cycle detection | Via `on_path` array | Via count check |
| Produces ordering | One valid ordering | One valid ordering (different from DFS) |
| Intuition | "Finish deepest nodes first, then reverse" | "Always process nodes with no dependencies" |
| Better for | When you already have DFS infrastructure | When you need to process "in order of availability" |

*Socratic prompt: "Why does reverse postorder give a valid topological order? Think about what it means that a node is added AFTER all its descendants."*

### Problems

| Problem | Key Twist |
|---------|-----------|
| Course Schedule II (210) | Build graph from prerequisites, topological sort |
| Alien Dictionary (269) | Build graph from character ordering between words |
| Parallel Courses (1136) | Topological sort with level tracking (BFS) for min semesters |

---

## 3. Union-Find (Disjoint Set Union)

### Key Insight

Union-Find efficiently tracks connected components. Two optimizations make it near-constant time:
1. **Path compression:** Point every node directly to the root during `find`
2. **Union by rank:** Attach the shorter tree under the taller tree

### Template

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n                    # Number of connected components

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])   # Path compression
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False                  # Already connected
        # Union by rank
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.count -= 1
        return True                       # Merged two components

    def connected(self, x, y):
        return self.find(x) == self.find(y)
```

### Complexity

With both optimizations: O(α(N)) per operation, where α is the inverse Ackermann function — effectively O(1) for all practical input sizes.

### When to Use Union-Find vs BFS/DFS

| Use Union-Find When | Use BFS/DFS When |
|---------------------|------------------|
| Edges arrive incrementally (online) | Graph is fully known upfront |
| Need component count or "are X and Y connected?" | Need shortest path or traversal order |
| Undirected graph connectivity | Directed graph problems (topological sort, cycle detection) |

*Socratic prompt: "Without path compression, `find` could take O(N) in the worst case. Draw a tree where this happens. How does path compression fix it?"*

### Problems

| Problem | Key Twist |
|---------|-----------|
| Number of Connected Components (323) | Count components after all unions |
| Redundant Connection (684) | The edge that creates a cycle — first union that returns False |
| Accounts Merge (721) | Union accounts with shared emails, then group by root |
| Number of Islands (200) | Union adjacent land cells (alternative to BFS/DFS flood fill) |
| Satisfiability of Equality Equations (990) | Union variables with `==`, then check `!=` constraints |

---

## 4. Eulerian Path (Hierholzer's Algorithm)

### Key Concept

An **Eulerian path** visits every EDGE exactly once. An **Eulerian circuit** is an Eulerian path that starts and ends at the same node.

### Existence Conditions

| Graph Type | Eulerian Circuit | Eulerian Path |
|------------|-----------------|---------------|
| Undirected | All vertices have even degree | Exactly 0 or 2 vertices have odd degree |
| Directed | All vertices: in-degree == out-degree | At most one vertex: out - in = 1 (start), at most one: in - out = 1 (end), all others: in == out |

### Hierholzer's Algorithm

```python
def find_eulerian_path(graph):
    """
    graph: dict of node -> list of neighbors (adjacency list).
    Modifies graph in place by removing edges as they're traversed.
    """
    result = []

    def dfs(node):
        while graph[node]:
            neighbor = graph[node].pop()   # Remove edge as we traverse
            dfs(neighbor)
        result.append(node)                # Postorder: add after dead-end

    # Start from the node with out-degree > in-degree (or any node for circuit)
    start = find_start_node(graph)
    dfs(start)
    return result[::-1]                    # Reverse postorder
```

**Why postorder + reverse?** Hierholzer's may reach dead-ends before traversing all edges. By adding nodes in postorder and reversing, dead-end detours get spliced correctly into the path.

*Socratic prompt: "Why can't we just do a regular DFS and record nodes in preorder? What goes wrong at a dead-end that isn't the final destination?"*

### Problems

| Problem | Key Twist |
|---------|-----------|
| Reconstruct Itinerary (332) | Directed Eulerian path; sort neighbors lexicographically for smallest itinerary |

---

## 5. Dijkstra Extensions

### State Object Pattern

For problems with extra constraints (limited stops, fuel, etc.), extend the state beyond just (distance, node):

```python
import heapq

def dijkstra_with_state(graph, start, target, max_stops):
    """State: (cost, node, stops_remaining)."""
    heap = [(0, start, max_stops)]
    # visited tracks (node, stops) to avoid re-processing
    visited = set()

    while heap:
        cost, node, stops = heapq.heappop(heap)
        if node == target:
            return cost
        if (node, stops) in visited:
            continue
        visited[node, stops] = True

        if stops > 0:
            for neighbor, weight in graph[node]:
                new_cost = cost + weight
                if (neighbor, stops - 1) not in visited:
                    heapq.heappush(heap, (new_cost, neighbor, stops - 1))

    return -1  # Unreachable
```

### Problems

| Problem | State Extension |
|---------|----------------|
| Network Delay Time (743) | Standard Dijkstra — answer = max(all distances) |
| Cheapest Flights Within K Stops (787) | State = (cost, node, stops_left) |
| Path with Maximum Probability (1514) | Max-heap with probabilities (negate for min-heap) |
| Swim in Rising Water (778) | Edge weight = max(current_max, neighbor elevation) |

---

## 6. A* Algorithm

### Key Insight

A* is Dijkstra with a heuristic that guides the search toward the goal. The priority is `f(x) = g(x) + h(x)`:
- `g(x)` = actual cost from start to current node
- `h(x)` = estimated cost from current node to goal (heuristic)

If `h(x)` is **admissible** (never overestimates), A* finds the optimal path.

### Template

```python
import heapq

def a_star(grid, start, goal):
    """Grid-based A* with Manhattan distance heuristic."""
    rows, cols = len(grid), len(grid[0])

    def heuristic(r, c):
        return abs(r - goal[0]) + abs(c - goal[1])   # Manhattan distance

    # (f, g, row, col)
    heap = [(heuristic(*start), 0, start[0], start[1])]
    g_score = {start: 0}

    while heap:
        f, g, r, c = heapq.heappop(heap)

        if (r, c) == goal:
            return g

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1:
                new_g = g + 1
                if new_g < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = new_g
                    f = new_g + heuristic(nr, nc)
                    heapq.heappush(heap, (f, new_g, nr, nc))

    return -1  # Unreachable
```

### When A* Helps vs Plain Dijkstra

| Scenario | Use Dijkstra | Use A* |
|----------|-------------|--------|
| Find shortest path to ALL nodes | Yes | No (A* targets one goal) |
| Find shortest path to ONE target | Works, but explores more | Yes, much faster with good heuristic |
| Non-grid graph, hard to define heuristic | Yes | Only if a good heuristic exists |
| Grid with obstacles | Works | A* with Manhattan/Euclidean heuristic is much faster |

### Common Heuristics

| Heuristic | Grid Type | Formula |
|-----------|-----------|---------|
| Manhattan | 4-directional | `|r1 - r2| + |c1 - c2|` |
| Euclidean | Any-direction | `sqrt((r1-r2)^2 + (c1-c2)^2)` |
| Chebyshev | 8-directional | `max(|r1-r2|, |c1-c2|)` |

*Socratic prompt: "If the heuristic overestimates the true cost, A* might find a suboptimal path. Why? Can you construct an example?"*

### Problems

| Problem | Key Twist |
|---------|-----------|
| Shortest Path in Binary Matrix (1091) | 8-directional grid, Chebyshev heuristic |
| Sliding Puzzle (773) | State = board configuration, heuristic = sum of Manhattan distances |

---

## 7. Kruskal's MST

### Key Insight

Sort all edges by weight. Greedily add the cheapest edge that doesn't create a cycle (use Union-Find to check).

### Template

```python
def kruskal(n, edges):
    """
    n: number of nodes
    edges: list of (weight, u, v)
    Returns: total weight of MST, or -1 if graph is disconnected
    """
    edges.sort()                          # Sort by weight
    uf = UnionFind(n)
    mst_weight = 0
    edges_used = 0

    for weight, u, v in edges:
        if uf.union(u, v):               # No cycle — add this edge
            mst_weight += weight
            edges_used += 1
            if edges_used == n - 1:       # MST complete
                break

    return mst_weight if edges_used == n - 1 else -1
```

**Why it works:** The cut property guarantees that the lightest edge crossing any cut belongs to some MST. By processing edges in sorted order and skipping those that create cycles, Kruskal's greedily builds the MST.

### Problems

| Problem | Key Twist |
|---------|-----------|
| Min Cost to Connect All Points (1584) | Generate all point-pair edges, then Kruskal |
| Connecting Cities With Minimum Cost (1135) | Direct application |

---

## 8. Prim's MST

### Key Insight

Start from any node. Repeatedly add the cheapest edge that connects a visited node to an unvisited node. Use a min-heap for efficiency.

### Template

```python
import heapq

def prim(n, graph):
    """
    graph: adjacency list, graph[u] = [(weight, v), ...]
    Returns: total weight of MST
    """
    visited = [False] * n
    heap = [(0, 0)]                       # (weight, node) — start from node 0
    mst_weight = 0
    edges_used = 0

    while heap and edges_used < n:
        weight, node = heapq.heappop(heap)
        if visited[node]:
            continue
        visited[node] = True
        mst_weight += weight
        edges_used += 1

        for w, neighbor in graph[node]:
            if not visited[neighbor]:
                heapq.heappush(heap, (w, neighbor))

    return mst_weight if edges_used == n else -1
```

### Kruskal vs Prim Comparison

| Aspect | Kruskal's | Prim's |
|--------|----------|--------|
| Approach | Sort edges, add cheapest non-cycle edge | Grow tree from one node, add cheapest crossing edge |
| Data structure | Union-Find | Min-heap + visited |
| Complexity | O(E log E) | O(E log V) |
| Better for | Sparse graphs (fewer edges to sort) | Dense graphs (fewer heap operations) |
| Edge list required? | Yes | No (works with adjacency list) |
| Similarity | Like Kruskal is to edges... | ...Prim is to vertices (like Dijkstra without cumulative cost) |

*Socratic prompt: "Prim's looks a lot like Dijkstra. What's the key difference? Hint: what does the priority queue key represent in each?"*

**Answer:** In Dijkstra, the key is the total distance from the source. In Prim's, the key is just the edge weight. Dijkstra finds shortest paths; Prim's finds the MST.

---

## Decision Tree: Which Graph Algorithm?

```
What does the problem ask?
├── "Can you complete all tasks?" / "Is there a valid ordering?"
│   ├── Cycle detection → DFS with visited[] + onPath[]
│   └── If no cycle → Topological sort (DFS or BFS)
├── "Are X and Y connected?" / "How many groups?"
│   └── Union-Find (or BFS/DFS for one-shot)
├── "Visit every edge exactly once"
│   └── Eulerian path → Hierholzer's algorithm
├── "Shortest/cheapest path in weighted graph"
│   ├── Non-negative weights → Dijkstra
│   ├── Negative weights → Bellman-Ford
│   └── Specific target with good heuristic → A*
├── "Connect all nodes with minimum total cost"
│   ├── Sparse graph → Kruskal's (sort edges + union-find)
│   └── Dense graph → Prim's (heap-based expansion)
└── "Is graph 2-colorable?" / "Split into two groups"
    └── Bipartite detection (see advanced-patterns.md)
```

---

## Attribution

The algorithms and frameworks in this file are inspired by and adapted from labuladong's algorithmic guides (labuladong.online), particularly the graph algorithm articles from Chapter 1 "Data Structure Algorithms" covering topological sort, union-find, Eulerian paths, Dijkstra extensions, and minimum spanning trees. Templates have been restructured and annotated for Socratic teaching use.
