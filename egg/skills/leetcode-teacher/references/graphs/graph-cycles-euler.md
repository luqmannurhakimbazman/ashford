# Graph Cycles and Euler Paths

Cycle detection (directed and undirected), negative cycle finding via Bellman-Ford, and Euler path/circuit construction via Hierholzer's algorithm. These extend the graph traversal foundations in `graph-algorithms.md` — DFS coloring, BFS, and shortest-path algorithms are prerequisites.

---

## Quick Reference Table

| Pattern | Recognition Signals | Key Problems | Section |
|---------|-------------------|--------------|---------|
| Cycle Detection (Directed) | "detect cycle in directed graph", "course prerequisites", "deadlock detection", three-color DFS | Course Schedule (207), Course Schedule II (210) | 1 |
| Cycle Detection (Undirected) | "detect cycle in undirected graph", "redundant connection", parent-tracking DFS / Union-Find | Redundant Connection (684), Graph Valid Tree (261) | 1 |
| Negative Cycle (Bellman-Ford) | "negative weight cycle", "arbitrage", "currency exchange", "relax N times" | Cheapest Flights Within K Stops (787), Bellman-Ford (743) | 2 |
| Euler Path / Circuit | "visit every edge exactly once", "reconstruct itinerary", "domino arrangement", degree parity checks | Reconstruct Itinerary (332), Cracking the Safe (753), Valid Arrangement of Pairs (2097) | 3 |

---

## Corner Cases

- **Self-loops:** A self-loop is a cycle of length 1 in directed graphs. In undirected graphs, a self-loop also creates a cycle. Handle separately in parent-tracking approaches.
- **Parallel edges (multigraph):** Multiple edges between the same pair of nodes. For undirected cycle detection, track edge indices rather than just parent node to avoid false positives.
- **Disconnected graph:** Cycle detection must iterate over all components. Euler paths require the graph to be connected (ignoring isolated vertices).
- **Empty graph / single node:** No cycle, trivially Eulerian (zero edges).
- **Graph with no edges but multiple nodes:** No cycle. Not Eulerian unless you consider the trivial case.

---

## 1. Cycle Detection in Graphs

### Core Insight

**Directed graphs:** A cycle exists if and only if DFS encounters a **back edge** — an edge to a vertex that is currently on the recursion stack. The three-color scheme (white/gray/black) distinguishes "not visited", "in progress", and "done" vertices. A gray-to-gray edge is a back edge.

**Undirected graphs:** A cycle exists if DFS visits a neighbor that has already been visited and is **not** the parent of the current node. Since every edge is traversed in both directions, you must skip the parent edge to avoid false cycle detection.

*Socratic prompt: "In a directed graph, why does finding an edge to a 'black' (fully processed) vertex NOT indicate a cycle, while an edge to a 'gray' vertex does? What's the structural difference?"*

### Template: Directed Graph Cycle Detection (Three-Color DFS)

```python
def find_cycle_directed(n, adj):
    """Find a cycle in a directed graph using three-color DFS.

    Args:
        n: Number of vertices (0-indexed).
        adj: Adjacency list where adj[v] is a list of neighbors of v.

    Returns:
        A list of vertices forming the cycle, or an empty list if acyclic.
        The cycle is returned as [v0, v1, ..., vk, v0] where v0 = vk.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    parent = [-1] * n
    cycle_start = -1
    cycle_end = -1

    def dfs(v):
        nonlocal cycle_start, cycle_end
        color[v] = GRAY
        for u in adj[v]:
            if color[u] == WHITE:
                parent[u] = v
                if dfs(u):
                    return True
            elif color[u] == GRAY:
                # Back edge found: u -> ... -> v -> u
                cycle_end = v
                cycle_start = u
                return True
        color[v] = BLACK
        return False

    for v in range(n):
        if color[v] == WHITE:
            if dfs(v):
                break

    if cycle_start == -1:
        return []  # Acyclic

    cycle = [cycle_start]
    v = cycle_end
    while v != cycle_start:
        cycle.append(v)
        v = parent[v]
    cycle.append(cycle_start)
    cycle.reverse()
    return cycle
```

**Why three colors instead of two?**
- In directed graphs, a visited node may have been fully processed (black) — an edge to it is a **cross edge** or **forward edge**, not a back edge.
- Only an edge to a gray node (still on the recursion stack) proves a cycle.
- In undirected graphs, two colors suffice because there are no cross/forward edges in DFS trees.

### Template: Undirected Graph Cycle Detection (Parent-Tracking DFS)

```python
def find_cycle_undirected(n, adj):
    """Find a cycle in an undirected graph using parent-tracking DFS.

    Args:
        n: Number of vertices (0-indexed).
        adj: Adjacency list where adj[v] is a list of neighbors of v.

    Returns:
        A list of vertices forming the cycle, or an empty list if acyclic.
    """
    visited = [False] * n
    parent = [-1] * n
    cycle_start = -1
    cycle_end = -1

    def dfs(v, par):
        nonlocal cycle_start, cycle_end
        visited[v] = True
        for u in adj[v]:
            if u == par:
                continue  # Skip the edge we came from
            if visited[u]:
                cycle_end = v
                cycle_start = u
                return True
            parent[u] = v
            if dfs(u, v):
                return True
        return False

    for v in range(n):
        if not visited[v]:
            if dfs(v, -1):
                break

    if cycle_start == -1:
        return []  # Acyclic

    cycle = [cycle_start]
    v = cycle_end
    while v != cycle_start:
        cycle.append(v)
        v = parent[v]
    cycle.append(cycle_start)
    return cycle
```

**Parallel edge gotcha:** If the graph has multiple edges between the same pair of nodes, skipping by parent *node* ID causes a bug — you'd skip all edges to the parent, not just the one you traversed. Fix: track edge indices and skip the specific edge.

```python
def find_cycle_undirected_multigraph(n, adj):
    """Cycle detection for undirected multigraph with parallel edges.

    adj[v] is a list of (neighbor, edge_id) tuples.
    """
    visited = [False] * n
    parent = [-1] * n
    cycle_start = -1
    cycle_end = -1

    def dfs(v, from_edge):
        nonlocal cycle_start, cycle_end
        visited[v] = True
        for u, edge_id in adj[v]:
            if edge_id == from_edge:
                continue  # Skip the exact edge we came from
            if visited[u]:
                cycle_end = v
                cycle_start = u
                return True
            parent[u] = v
            if dfs(u, edge_id):
                return True
        return False

    for v in range(n):
        if not visited[v]:
            if dfs(v, -1):
                break

    if cycle_start == -1:
        return []

    cycle = [cycle_start]
    v = cycle_end
    while v != cycle_start:
        cycle.append(v)
        v = parent[v]
    cycle.append(cycle_start)
    return cycle
```

### Iterative Cycle Detection (Avoids Recursion Limit)

For large graphs where Python's default recursion limit (1000) is insufficient:

```python
def has_cycle_directed_iterative(n, adj):
    """Detect cycle in directed graph using Kahn's algorithm (BFS topological sort).

    Returns True if the graph has a cycle, False otherwise.
    Key idea: if topological sort cannot process all vertices, a cycle exists.
    """
    in_degree = [0] * n
    for v in range(n):
        for u in adj[v]:
            in_degree[u] += 1

    from collections import deque
    queue = deque(v for v in range(n) if in_degree[v] == 0)
    processed = 0

    while queue:
        v = queue.popleft()
        processed += 1
        for u in adj[v]:
            in_degree[u] -= 1
            if in_degree[u] == 0:
                queue.append(u)

    return processed != n  # True if cycle exists
```

### Complexity

| Variant | Time | Space |
|---------|------|-------|
| Directed (three-color DFS) | O(\|V\| + \|E\|) | O(\|V\|) |
| Undirected (parent-tracking DFS) | O(\|V\| + \|E\|) | O(\|V\|) |
| Directed (Kahn's / BFS topological) | O(\|V\| + \|E\|) | O(\|V\|) |

*Socratic prompt: "Course Schedule (LC 207) asks if you can finish all courses given prerequisites. How does this reduce to cycle detection? Why does a cycle make it impossible?"*

*Socratic prompt: "For undirected graphs, Union-Find can also detect cycles. When you add an edge (u, v) and find(u) == find(v), why does that prove a cycle? What advantage does Union-Find have over DFS for dynamic edge insertion?"*

### Practice Problems

#### Essential

| # | Problem | Difficulty | Key Technique |
|---|---------|-----------|---------------|
| 207 | Course Schedule | Medium | Directed cycle detection (three-color DFS or Kahn's) |
| 210 | Course Schedule II | Medium | Topological sort — fails if cycle |
| 684 | Redundant Connection | Medium | Undirected cycle via Union-Find (or DFS) |
| 261 | Graph Valid Tree | Medium | n-1 edges + no cycle (Union-Find or DFS) |

#### Recommended

| # | Problem | Difficulty | Key Technique |
|---|---------|-----------|---------------|
| 802 | Find Eventual Safe States | Medium | Reverse graph + cycle detection (three-color) |
| 1059 | All Paths from Source Lead to Destination | Medium | Directed DFS with cycle check |
| 685 | Redundant Connection II | Hard | Directed graph: cycle + in-degree analysis |

---

## 2. Finding Negative Cycles (Bellman-Ford)

### Core Insight

The standard Bellman-Ford algorithm runs `|V| - 1` iterations of edge relaxation to find shortest paths. If a **V-th iteration** still relaxes some edge, a negative-weight cycle is reachable from the source. To find *any* negative cycle in the graph (even if unreachable from a source), initialize all distances to zero instead of infinity.

**Why zero initialization?** With all distances at zero, every vertex is treated as a potential source. If any cycle has negative total weight, the relaxation will keep reducing distances along that cycle indefinitely — the V-th iteration will still find an improvement.

*Socratic prompt: "Standard Bellman-Ford initializes dist[source] = 0 and all others to infinity. Why would that miss a negative cycle that is not reachable from the source? How does initializing all to zero fix this?"*

### Template: Detect Any Negative Cycle

```python
def find_negative_cycle(n, edges):
    """Find a negative-weight cycle in a directed weighted graph.

    Args:
        n: Number of vertices (0-indexed).
        edges: List of (u, v, weight) tuples.

    Returns:
        A list of vertices forming the negative cycle, or an empty list
        if no negative cycle exists. Cycle is [v0, v1, ..., vk, v0].
    """
    INF = float('inf')
    dist = [0] * n       # Zero-init to detect ALL negative cycles
    parent = [-1] * n
    last_relaxed = -1     # Track which vertex was relaxed on the N-th iteration

    for i in range(n):
        last_relaxed = -1
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = max(-INF, dist[u] + w)  # Prevent underflow
                parent[v] = u
                last_relaxed = v

    if last_relaxed == -1:
        return []  # No negative cycle

    # Walk back N steps to guarantee we're inside the cycle
    v = last_relaxed
    for _ in range(n):
        v = parent[v]

    # Now v is on the cycle; trace until we return to v
    cycle = []
    u = v
    while True:
        cycle.append(u)
        u = parent[u]
        if u == v:
            cycle.append(v)
            break
    cycle.reverse()
    return cycle
```

**Why walk back N steps?** The vertex `last_relaxed` may not be on the cycle itself — it could be reachable from the cycle. Walking N steps through parent pointers guarantees entering the cycle, because the cycle has at most N vertices.

### Template: Detect Negative Cycle Reachable from Source

```python
def negative_cycle_from_source(n, edges, source):
    """Detect negative cycle reachable from a specific source.

    Args:
        n: Number of vertices.
        edges: List of (u, v, weight) tuples.
        source: Source vertex.

    Returns:
        True if a negative cycle is reachable from source, False otherwise.
    """
    INF = float('inf')
    dist = [INF] * n
    dist[source] = 0

    for i in range(n - 1):
        for u, v, w in edges:
            if dist[u] < INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # N-th iteration: check for further relaxation
    for u, v, w in edges:
        if dist[u] < INF and dist[u] + w < dist[v]:
            return True
    return False
```

### Application: Currency Arbitrage

A classic application of negative cycle detection. Model currencies as vertices and exchange rates as edge weights. Take `log` of rates — a negative cycle in the log-space means multiplying rates around the cycle yields a gain (arbitrage).

```python
import math

def detect_arbitrage(rates):
    """Detect arbitrage opportunity in currency exchange rates.

    Args:
        rates: n x n matrix where rates[i][j] is the exchange rate from
               currency i to currency j (1 unit of i = rates[i][j] units of j).

    Returns:
        True if an arbitrage cycle exists, False otherwise.
    """
    n = len(rates)
    # Convert to negative-log edges: arbitrage iff negative cycle in log-space
    edges = []
    for i in range(n):
        for j in range(n):
            if rates[i][j] > 0:
                edges.append((i, j, -math.log(rates[i][j])))

    # Bellman-Ford with zero-init to find any negative cycle
    dist = [0.0] * n
    for i in range(n):
        for u, v, w in edges:
            if dist[u] + w < dist[v] - 1e-9:
                dist[v] = dist[u] + w
                if i == n - 1:
                    return True  # Relaxation on N-th iteration
    return False
```

*Socratic prompt: "Why do we take the negative log of exchange rates? If rate[A->B] * rate[B->C] * rate[C->A] > 1, what does that look like in log-space?"*

### Complexity

| Aspect | Value |
|--------|-------|
| Time | O(\|V\| * \|E\|) — N iterations over all edges |
| Space | O(\|V\| + \|E\|) |
| Cycle reconstruction | O(\|V\|) additional |

*Socratic prompt: "Bellman-Ford runs |V| - 1 iterations for shortest paths. Why exactly |V| - 1? What property of shortest paths in a graph without negative cycles guarantees convergence by then?"*

### Practice Problems

#### Essential

| # | Problem | Difficulty | Key Technique |
|---|---------|-----------|---------------|
| 787 | Cheapest Flights Within K Stops | Medium | Bellman-Ford with iteration limit |
| 743 | Network Delay Time | Medium | Bellman-Ford / Dijkstra for shortest paths |

#### Recommended

| # | Problem | Difficulty | Key Technique |
|---|---------|-----------|---------------|
| — | CSES: Cycle Finding | — | Direct negative cycle detection and reconstruction |
| — | CSES: High Score | — | Longest path with negative cycle detection (negate weights) |
| — | Currency arbitrage (classic) | — | Negative log transform + Bellman-Ford |

---

## 3. Euler Path and Circuit (Hierholzer's Algorithm)

### Core Insight

An **Euler path** traverses every edge exactly once. An **Euler circuit** (cycle) is an Euler path that starts and ends at the same vertex. The existence conditions depend on whether the graph is directed or undirected:

**Undirected graph:**
- **Euler circuit exists** iff every vertex has **even degree** and the graph is connected (ignoring isolated vertices).
- **Euler path exists** iff exactly **0 or 2** vertices have **odd degree** and the graph is connected. The two odd-degree vertices are the start and end of the path.

**Directed graph:**
- **Euler circuit exists** iff every vertex has **in-degree == out-degree** and the graph is connected.
- **Euler path exists** iff at most one vertex has **out-degree - in-degree == 1** (start), at most one has **in-degree - out-degree == 1** (end), and all others have equal in/out degree. The graph must be connected.

*Socratic prompt: "Why must all vertices have even degree for an Euler circuit in an undirected graph? Think about what happens every time the path enters a vertex — it must also leave. What does this imply about degree parity?"*

### Template: Hierholzer's Algorithm (Directed Graph)

```python
from collections import defaultdict, deque

def find_euler_path_directed(edges, n):
    """Find an Euler path or circuit in a directed graph using Hierholzer's algorithm.

    Args:
        edges: List of (u, v) directed edges.
        n: Number of vertices (0-indexed).

    Returns:
        List of vertices forming the Euler path/circuit, or empty list if
        no Euler path exists.
    """
    # Build adjacency list with mutable edge lists
    adj = defaultdict(deque)
    in_deg = [0] * n
    out_deg = [0] * n

    for u, v in edges:
        adj[u].append(v)
        out_deg[u] += 1
        in_deg[v] += 1

    # Determine start vertex and check existence
    start_nodes = 0  # Vertices with out - in == 1
    end_nodes = 0    # Vertices with in - out == 1
    start = 0

    for v in range(n):
        diff = out_deg[v] - in_deg[v]
        if diff == 1:
            start_nodes += 1
            start = v  # Euler path must start here
        elif diff == -1:
            end_nodes += 1
        elif diff != 0:
            return []  # No Euler path/circuit possible

    # Valid configs: (0 start, 0 end) for circuit or (1 start, 1 end) for path
    if not (start_nodes == 0 and end_nodes == 0) and \
       not (start_nodes == 1 and end_nodes == 1):
        return []

    # Hierholzer's algorithm (iterative with stack)
    stack = [start]
    path = []

    while stack:
        v = stack[-1]
        if adj[v]:
            u = adj[v].popleft()  # Take and remove an edge
            stack.append(u)
        else:
            path.append(stack.pop())

    path.reverse()

    # Verify all edges were used
    if len(path) != len(edges) + 1:
        return []  # Graph was not connected

    return path
```

**Why `popleft` from the adjacency deque?** Each edge must be used exactly once. By popping edges as we traverse them, we guarantee no edge is revisited. The deque provides O(1) popleft.

**Why the stack-based approach?** When we reach a dead end (no more edges from current vertex), we add that vertex to the path and backtrack. This naturally handles branching — sub-tours are spliced into the main path.

### Template: Hierholzer's Algorithm (Undirected Graph)

```python
from collections import defaultdict

def find_euler_path_undirected(n, edges):
    """Find an Euler path or circuit in an undirected graph.

    Args:
        n: Number of vertices (0-indexed).
        edges: List of (u, v) undirected edges.

    Returns:
        List of vertices forming the Euler path/circuit, or empty list if
        no Euler path exists.
    """
    adj = defaultdict(list)
    degree = [0] * n
    used_edge = [False] * len(edges)

    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
        degree[u] += 1
        degree[v] += 1

    # Check existence and find start
    odd_count = sum(1 for v in range(n) if degree[v] % 2 == 1)
    if odd_count not in (0, 2):
        return []  # No Euler path/circuit

    # Start at an odd-degree vertex (if path) or any vertex with edges (if circuit)
    start = 0
    for v in range(n):
        if degree[v] % 2 == 1:
            start = v
            break
        if degree[v] > 0:
            start = v  # Fallback: any vertex with edges

    # Hierholzer's (iterative)
    # Use pointer array to skip already-used edges efficiently
    ptr = defaultdict(int)  # Current index into adj[v]
    stack = [start]
    path = []

    while stack:
        v = stack[-1]
        found = False
        while ptr[v] < len(adj[v]):
            u, edge_id = adj[v][ptr[v]]
            ptr[v] += 1
            if not used_edge[edge_id]:
                used_edge[edge_id] = True
                stack.append(u)
                found = True
                break
        if not found:
            path.append(stack.pop())

    path.reverse()

    # Verify all edges used
    if len(path) != len(edges) + 1:
        return []  # Disconnected graph

    return path
```

**Key difference from directed:** Each undirected edge appears in both adjacency lists, so we track `used_edge[edge_id]` to mark an edge as consumed from both directions simultaneously.

### Template: Reconstruct Itinerary (LeetCode 332)

A classic Euler path problem with lexicographic ordering:

```python
from collections import defaultdict

def find_itinerary(tickets):
    """LC 332: Reconstruct Itinerary.

    Given airline tickets as [from, to] pairs, reconstruct the itinerary
    starting from 'JFK'. Use all tickets exactly once. If multiple valid
    itineraries exist, return the lexicographically smallest one.

    Args:
        tickets: List of [from, to] airport code pairs.

    Returns:
        List of airport codes forming the itinerary.
    """
    adj = defaultdict(list)
    for src, dst in sorted(tickets, reverse=True):
        # Sort in reverse so we can pop() from end for lex smallest
        adj[src].append(dst)

    stack = ["JFK"]
    itinerary = []

    while stack:
        while adj[stack[-1]]:
            next_airport = adj[stack[-1]].pop()
            stack.append(next_airport)
        itinerary.append(stack.pop())

    return itinerary[::-1]
```

**Why reverse sort + pop?** We want the lexicographically smallest neighbor first. Sorting in reverse and using `pop()` (which takes from the end) gives us the smallest available neighbor in O(1). This is equivalent to sorting normally and using `popleft()` from a deque, but avoids the import.

*Socratic prompt: "In LC 332, why can't you just greedily always pick the lexicographically smallest next airport with a simple DFS? What goes wrong, and how does Hierholzer's post-order insertion fix it?"*

### Existence Conditions Summary

| Graph Type | Euler Circuit | Euler Path (non-circuit) |
|------------|--------------|-------------------------|
| **Undirected** | All vertices have even degree; graph is connected | Exactly 2 vertices with odd degree; graph is connected |
| **Directed** | in-degree == out-degree for all vertices; graph is connected | Exactly 1 vertex with out - in == 1 (start), 1 with in - out == 1 (end); rest equal; connected |

### Complexity

| Aspect | Value |
|--------|-------|
| Time | O(\|E\|) — each edge is visited exactly once |
| Space | O(\|E\|) — for adjacency list and stack |
| Existence check | O(\|V\| + \|E\|) — degree counting + connectivity |

*Socratic prompt: "The Domino Problem asks: given N dominoes with numbers on each half, can you arrange them in a line so adjacent halves match? How do you model this as an Euler path problem? What are the vertices and what are the edges?"*

### Practice Problems

#### Essential

| # | Problem | Difficulty | Key Technique |
|---|---------|-----------|---------------|
| 332 | Reconstruct Itinerary | Hard | Hierholzer's with lexicographic ordering |
| 2097 | Valid Arrangement of Pairs | Hard | Directed Euler path (Hierholzer's) |
| 753 | Cracking the Safe | Hard | Euler circuit on de Bruijn graph |

#### Recommended

| # | Problem | Difficulty | Key Technique |
|---|---------|-----------|---------------|
| — | CSES: Mail Delivery | — | Euler circuit in undirected graph |
| — | CSES: Teleporters Path | — | Euler path in directed graph |
| — | CSES: De Bruijn Sequence | — | Euler circuit on de Bruijn graph |
| 1743 | Restore the Array From Adjacent Pairs | Medium | Path reconstruction (related: Euler-like traversal) |

---

## Method Comparison

| Method | When to Use | Time | Key Check |
|--------|------------|------|-----------|
| Three-color DFS | Directed cycle detection/reconstruction | O(\|V\| + \|E\|) | Gray-to-gray edge = back edge |
| Parent-tracking DFS | Undirected cycle detection | O(\|V\| + \|E\|) | Visited non-parent neighbor |
| Kahn's (BFS topo sort) | Directed cycle existence (no reconstruction) | O(\|V\| + \|E\|) | Processed count < \|V\| |
| Bellman-Ford N-th iter | Negative cycle detection + reconstruction | O(\|V\| * \|E\|) | Relaxation on iteration N |
| Hierholzer's | Euler path/circuit construction | O(\|E\|) | Degree parity conditions met |

**Decision flowchart:**
1. Need to **detect a cycle**?
   - Directed graph -> Three-color DFS (or Kahn's for existence only)
   - Undirected graph -> Parent-tracking DFS (or Union-Find for dynamic edges)
2. Need to find a **negative-weight cycle**?
   - Bellman-Ford with zero-init (for any cycle) or standard init (from source)
3. Need to **traverse every edge exactly once**?
   - Check Euler existence conditions (degree parity + connectivity)
   - Build the path with Hierholzer's algorithm

---

## Attribution

Content synthesized from cp-algorithms.com articles on [Finding a Cycle](https://cp-algorithms.com/graph/finding-cycle.html), [Finding Negative Cycle](https://cp-algorithms.com/graph/finding-negative-cycle-in-graph.html), and [Euler Path](https://cp-algorithms.com/graph/euler_path.html). Algorithms and complexity analyses are from the original articles; code has been translated to Python with added commentary for the leetcode-teacher reference format.
