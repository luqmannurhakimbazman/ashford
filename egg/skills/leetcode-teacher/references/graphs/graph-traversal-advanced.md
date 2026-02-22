# Advanced Graph Traversal

Advanced graph traversal techniques covering BFS/DFS applications, connectivity analysis, bridge and articulation point detection, and strongly connected components. Builds on the basic BFS/DFS templates in `graph-algorithms.md` and graph representations in `data-structure-fundamentals.md`.

---

## Quick Reference Table

| Pattern | Recognition Signals | Key Problems | Section |
|---------|-------------------|--------------|---------|
| BFS (Advanced) | "Shortest path unweighted", "minimum moves", "level-order", "0-1 BFS" | Word Ladder (127), Shortest Path in Binary Matrix (1091), Rotting Oranges (994) | 1 |
| DFS (Edge Classification & Timestamps) | "Detect cycle in directed graph", "topological order", "entry/exit time", "ancestor check" | Course Schedule (207), Course Schedule II (210), Alien Dictionary (269) | 2 |
| Connected Components | "Number of islands", "count groups", "connected regions" | Number of Islands (200), Number of Connected Components (323), Accounts Merge (721) | 3 |
| Bridges | "Critical connection", "edge whose removal disconnects" | Critical Connections in a Network (1192) | 4 |
| Bridge Finding Online | "Dynamic edge additions", "maintain bridge count online" | -- (competitive programming) | 5 |
| Articulation Points | "Cut vertex", "node whose removal disconnects", "biconnected components" | Critical Connections (1192) — related | 6 |
| Strongly Connected Components | "Mutually reachable", "condensation DAG", "2-SAT" | -- (competitive programming), Course Schedule IV (1462) — related | 7 |
| Strong Orientation | "Direct edges to make strongly connected", "bridgeless graph" | -- (competitive programming) | 8 |

---

## Topic Connection Map

These topics build on each other. Use this map to navigate:

```
BFS (shortest paths) ──→ 0-1 BFS (deque trick)
DFS (timestamps) ──→ Edge Classification ──→ Cycle Detection
       │
       ├──→ Bridges (low[to] > tin[v])
       ├──→ Articulation Points (low[to] >= tin[v])
       ├──→ Strong Orientation (orient bridgeless graph)
       │
       └──→ Strongly Connected Components
                ├── Kosaraju (two DFS passes)
                └── Tarjan (single DFS + stack)
                         │
                    Condensation DAG ──→ DP on DAG
```

---

## 1. BFS (Advanced Applications)

### Core Insight

BFS explores vertices in layers from a source, guaranteeing that the first time a vertex is reached, it is via the shortest path (in terms of edge count). This "fire-spreading" property makes BFS the go-to algorithm for shortest paths in unweighted graphs and for any problem that asks for "minimum moves" or "minimum steps."

The 0-1 BFS variant handles graphs where edges have weight 0 or 1 by using a deque instead of a queue: weight-0 edges push to the front, weight-1 edges push to the back. This preserves the shortest-first ordering without needing Dijkstra.

*Socratic prompt: "Why does BFS guarantee shortest paths in unweighted graphs but not in weighted graphs? What invariant does the queue maintain that breaks when edges have different weights?"*

### Template

```python
from collections import deque
from typing import List, Optional


def bfs_shortest_path(adj: List[List[int]], source: int, target: int) -> List[int]:
    """Find shortest path in an unweighted graph using BFS.

    Time:  O(V + E) where V = vertices, E = edges.
    Space: O(V) for visited, distance, and parent arrays.

    Args:
        adj: Adjacency list representation of the graph.
        source: Starting vertex.
        target: Destination vertex.

    Returns:
        List of vertices on the shortest path, or empty list if unreachable.
    """
    n = len(adj)
    visited = [False] * n
    dist = [0] * n
    parent = [-1] * n

    queue = deque([source])
    visited[source] = True

    while queue:
        v = queue.popleft()
        for u in adj[v]:
            if not visited[u]:
                visited[u] = True
                queue.append(u)
                dist[u] = dist[v] + 1
                parent[u] = v

    # Reconstruct path
    if not visited[target]:
        return []
    path = []
    cur = target
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]


def bfs_01(adj: List[List[tuple]], source: int, n: int) -> List[int]:
    """0-1 BFS for graphs with edge weights 0 or 1.

    Uses a deque: weight-0 neighbors go to front, weight-1 to back.

    Time:  O(V + E).
    Space: O(V).

    Args:
        adj: Adjacency list of (neighbor, weight) pairs. weight in {0, 1}.
        source: Starting vertex.
        n: Number of vertices.

    Returns:
        List of shortest distances from source to every vertex.
    """
    INF = float('inf')
    dist = [INF] * n
    dist[source] = 0
    dq = deque([source])

    while dq:
        v = dq.popleft()
        for u, w in adj[v]:
            if dist[v] + w < dist[u]:
                dist[u] = dist[v] + w
                if w == 0:
                    dq.appendleft(u)
                else:
                    dq.append(u)
    return dist
```

| Operation | Time | Space |
|-----------|------|-------|
| Standard BFS | O(V + E) | O(V) |
| 0-1 BFS | O(V + E) | O(V) |
| Multi-source BFS | O(V + E) | O(V) |
| Shortest cycle (directed) | O(V * (V + E)) | O(V) |

### BFS vs Dijkstra Comparison

| Feature | BFS | 0-1 BFS | Dijkstra |
|---------|-----|---------|----------|
| Edge weights | All equal (unweighted) | 0 or 1 only | Non-negative |
| Data structure | Queue | Deque | Priority queue (min-heap) |
| Time complexity | O(V + E) | O(V + E) | O((V + E) log V) |
| When to use | Unweighted shortest path | Binary-weight graphs | General non-negative weights |

*Socratic prompt: "In 0-1 BFS, why do we push weight-0 neighbors to the front of the deque and weight-1 neighbors to the back? What invariant about the deque does this maintain?"*

### Practice Problems

| Problem (LC#) | Key Twist |
|--------------|-----------|
| Word Ladder (127) | BFS on implicit graph; each word is a node, edges connect words differing by one letter |
| Shortest Path in Binary Matrix (1091) | 8-directional BFS on grid |
| Rotting Oranges (994) | Multi-source BFS; all rotten oranges start in queue simultaneously |
| Minimum Knight Moves (1197) | BFS with large state space; pruning or bidirectional BFS helps |
| Shortest Bridge (934) | DFS to find one island, then BFS to reach the other |
| Open the Lock (752) | BFS on state space with deadends as blocked nodes |
| Sliding Puzzle (773) | BFS on board states represented as strings |
| Jump Game III (1306) | BFS/DFS on index graph |

---

## 2. DFS (Edge Classification & Timestamps)

### Core Insight

DFS assigns each vertex an entry time (`tin`) when first discovered and an exit time (`tout`) when fully processed. These timestamps enable O(1) ancestor queries: vertex `u` is an ancestor of `v` if and only if `tin[u] <= tin[v]` and `tout[u] >= tout[v]` (the interval of `u` contains the interval of `v`).

In directed graphs, DFS classifies every edge into exactly one of four types based on the color (state) of the destination vertex:

- **Tree edge**: leads to an unvisited vertex (white/color 0).
- **Back edge**: leads to an ancestor currently on the recursion stack (gray/color 1). These indicate cycles.
- **Forward edge**: leads to a descendant already fully processed (black/color 2, and `tin[u] < tin[v]`).
- **Cross edge**: leads to a vertex in a different subtree already fully processed (black/color 2, and `tin[u] > tin[v]`).

In undirected graphs, every non-tree edge is a back edge.

*Socratic prompt: "Why are forward edges and cross edges impossible in an undirected graph DFS? Think about what happens when DFS first encounters an edge to an already-visited vertex."*

### Template

```python
from typing import List, Tuple, Set


class DFSWithTimestamps:
    """DFS with entry/exit timestamps and edge classification.

    Time:  O(V + E).
    Space: O(V) for arrays + O(V) recursion stack.
    """

    WHITE, GRAY, BLACK = 0, 1, 2  # unvisited, in-stack, finished

    def __init__(self, n: int, adj: List[List[int]], directed: bool = True):
        self.n = n
        self.adj = adj
        self.directed = directed
        self.color = [self.WHITE] * n
        self.tin = [0] * n
        self.tout = [0] * n
        self.parent = [-1] * n
        self.timer = 0

        # Edge classification results
        self.tree_edges: List[Tuple[int, int]] = []
        self.back_edges: List[Tuple[int, int]] = []
        self.forward_edges: List[Tuple[int, int]] = []
        self.cross_edges: List[Tuple[int, int]] = []

    def dfs(self, v: int) -> None:
        """Run DFS from vertex v, classifying all edges."""
        self.color[v] = self.GRAY
        self.tin[v] = self.timer
        self.timer += 1

        for u in self.adj[v]:
            if self.color[u] == self.WHITE:
                self.tree_edges.append((v, u))
                self.parent[u] = v
                self.dfs(u)
            elif self.color[u] == self.GRAY:
                self.back_edges.append((v, u))
            elif self.directed:
                # BLACK vertex in directed graph
                if self.tin[v] < self.tin[u]:
                    self.forward_edges.append((v, u))
                else:
                    self.cross_edges.append((v, u))

        self.color[v] = self.BLACK
        self.tout[v] = self.timer
        self.timer += 1

    def run(self) -> None:
        """Run DFS from all unvisited vertices."""
        for v in range(self.n):
            if self.color[v] == self.WHITE:
                self.dfs(v)

    def is_ancestor(self, u: int, v: int) -> bool:
        """Check if u is an ancestor of v in O(1) using timestamps."""
        return self.tin[u] <= self.tin[v] and self.tout[u] >= self.tout[v]

    def has_cycle(self) -> bool:
        """A directed graph has a cycle iff DFS finds a back edge."""
        return len(self.back_edges) > 0
```

### Edge Classification Summary

| Edge Type | Condition | Meaning | Exists In |
|-----------|-----------|---------|-----------|
| Tree | `color[u] == WHITE` | Discovery edge; forms DFS tree | Directed & Undirected |
| Back | `color[u] == GRAY` | Points to ancestor on stack; creates cycle | Directed & Undirected |
| Forward | `color[u] == BLACK` and `tin[v] < tin[u]` | Points to descendant already finished | Directed only |
| Cross | `color[u] == BLACK` and `tin[v] > tin[u]` | Points to vertex in different subtree | Directed only |

| Operation | Time | Space |
|-----------|------|-------|
| Full DFS with timestamps | O(V + E) | O(V) |
| Ancestor query (after DFS) | O(1) | O(1) |
| Cycle detection | O(V + E) | O(V) |

*Socratic prompt: "If you run DFS on a directed graph and sort vertices by decreasing exit time, you get a topological order. Why does this work? What role do back edges play in proving no valid topological order exists when there is a cycle?"*

### Practice Problems

| Problem (LC#) | Key Twist |
|--------------|-----------|
| Course Schedule (207) | Cycle detection in directed graph via DFS coloring |
| Course Schedule II (210) | Topological sort by decreasing exit time |
| Alien Dictionary (269) | Build directed graph from ordering constraints, then topological sort |
| Find Eventual Safe States (802) | Vertices with no path to a cycle; detect via DFS coloring |
| Reconstruct Itinerary (332) | Eulerian path via DFS with sorted adjacency lists |
| All Ancestors of a Node (2192) | DFS from each node or reverse graph + DFS |

---

## 3. Connected Components

### Core Insight

A connected component is a maximal set of vertices in an undirected graph such that every pair is connected by a path. Finding all components is a direct application of BFS or DFS: start a traversal from any unvisited vertex, mark everything reachable as one component, then repeat from the next unvisited vertex.

The key insight is that any traversal (BFS or DFS) from a vertex will visit exactly its entire connected component and nothing else. The number of times you restart traversal equals the number of components.

*Socratic prompt: "If you add one edge to a graph, the number of connected components either stays the same or decreases by exactly one. Why can it never decrease by more than one?"*

### Template

```python
from collections import deque
from typing import List


def find_connected_components_dfs(n: int, adj: List[List[int]]) -> List[List[int]]:
    """Find all connected components using DFS.

    Time:  O(V + E).
    Space: O(V).

    Args:
        n: Number of vertices.
        adj: Adjacency list.

    Returns:
        List of components, each component is a list of vertex indices.
    """
    visited = [False] * n
    components = []

    def dfs(v: int, comp: List[int]) -> None:
        visited[v] = True
        comp.append(v)
        for u in adj[v]:
            if not visited[u]:
                dfs(u, comp)

    for v in range(n):
        if not visited[v]:
            comp = []
            dfs(v, comp)
            components.append(comp)

    return components


def find_connected_components_iterative(n: int, adj: List[List[int]]) -> List[List[int]]:
    """Find all connected components using iterative DFS (avoids stack overflow).

    Time:  O(V + E).
    Space: O(V).
    """
    visited = [False] * n
    components = []

    for start in range(n):
        if visited[start]:
            continue
        comp = []
        stack = [start]
        while stack:
            v = stack.pop()
            if visited[v]:
                continue
            visited[v] = True
            comp.append(v)
            for u in reversed(adj[v]):  # reversed to match recursive DFS order
                if not visited[u]:
                    stack.append(u)
        components.append(comp)

    return components


def find_components_bfs(n: int, adj: List[List[int]]) -> List[List[int]]:
    """Find all connected components using BFS.

    Time:  O(V + E).
    Space: O(V).
    """
    visited = [False] * n
    components = []

    for start in range(n):
        if visited[start]:
            continue
        comp = []
        queue = deque([start])
        visited[start] = True
        while queue:
            v = queue.popleft()
            comp.append(v)
            for u in adj[v]:
                if not visited[u]:
                    visited[u] = True
                    queue.append(u)
        components.append(comp)

    return components
```

### Recursive vs Iterative DFS Comparison

| Approach | Pros | Cons |
|----------|------|------|
| Recursive DFS | Clean, simple code | Stack overflow on deep graphs (Python default limit ~1000) |
| Iterative DFS (stack) | No stack overflow | Slightly more code; visit order may differ |
| BFS (queue) | Level-order; no stack overflow | Slightly more memory for wide graphs |

| Operation | Time | Space |
|-----------|------|-------|
| Find all components | O(V + E) | O(V) |
| Count components | O(V + E) | O(V) |
| Check if two nodes connected | O(V + E) naive; O(α(V)) with Union-Find | O(V) |

*Socratic prompt: "When would you prefer Union-Find over DFS/BFS for connected components? Think about when edges arrive one at a time versus all at once."*

### Practice Problems

| Problem (LC#) | Key Twist |
|--------------|-----------|
| Number of Islands (200) | 2D grid; each cell is a node, flood-fill connected components |
| Number of Connected Components in Undirected Graph (323) | Direct application of component counting |
| Accounts Merge (721) | Union-Find or DFS on email-to-account graph |
| Number of Provinces (547) | Adjacency matrix input; count components |
| Making a Large Island (827) | Track component sizes, then try flipping each 0 |
| Redundant Connection (684) | Find the edge that creates a cycle (one extra edge beyond a tree) |

---

## 4. Bridges (Critical Connections)

### Core Insight

A bridge is an edge whose removal increases the number of connected components. The algorithm uses DFS with two key arrays:

- `tin[v]`: the entry time when vertex `v` is first visited.
- `low[v]`: the earliest entry time reachable from the subtree rooted at `v` (through tree edges and at most one back edge).

**Bridge condition**: Edge `(v, to)` is a bridge if and only if `low[to] > tin[v]`. This means the subtree rooted at `to` has no back edge that reaches `v` or any of `v`'s ancestors, so removing `(v, to)` disconnects `to`'s subtree.

The `low[v]` value is computed as the minimum of:
1. `tin[v]` itself
2. `tin[p]` for every back edge from `v` to ancestor `p`
3. `low[to]` for every tree-edge child `to`

*Socratic prompt: "Why do we use `low[to] > tin[v]` for bridges but `low[to] >= tin[v]` for articulation points? What is the subtle difference in what disconnection means for edges vs vertices?"*

### Template

```python
from typing import List, Tuple


def find_bridges(n: int, adj: List[List[int]]) -> List[Tuple[int, int]]:
    """Find all bridges in an undirected graph using DFS.

    A bridge is an edge whose removal disconnects the graph.
    Uses Tarjan's bridge-finding algorithm with tin/low arrays.

    Time:  O(V + E).
    Space: O(V).

    Args:
        n: Number of vertices.
        adj: Adjacency list for undirected graph.

    Returns:
        List of bridge edges as (u, v) tuples.
    """
    tin = [-1] * n
    low = [-1] * n
    visited = [False] * n
    bridges = []
    timer = [0]  # mutable container for closure

    def dfs(v: int, parent: int) -> None:
        visited[v] = True
        tin[v] = low[v] = timer[0]
        timer[0] += 1

        parent_skipped = False
        for to in adj[v]:
            if to == parent and not parent_skipped:
                # Skip the first occurrence of the parent edge
                # (handles multigraphs correctly)
                parent_skipped = True
                continue
            if visited[to]:
                # Back edge: update low with tin of ancestor
                low[v] = min(low[v], tin[to])
            else:
                # Tree edge: recurse, then update low from child
                dfs(to, v)
                low[v] = min(low[v], low[to])
                if low[to] > tin[v]:
                    bridges.append((v, to))

    for i in range(n):
        if not visited[i]:
            dfs(i, -1)

    return bridges
```

| Operation | Time | Space |
|-----------|------|-------|
| Find all bridges | O(V + E) | O(V) |
| Check if specific edge is a bridge | O(V + E) | O(V) |
| Count 2-edge-connected components | O(V + E) | O(V) |

### Bridge vs Non-Bridge Visual Intuition

```
Bridge example:          No bridge (cycle):
  1 --- 2 --- 3           1 --- 2
              |           |     |
              4           4 --- 3
  Edge (2,3) is a        No bridges: every edge
  bridge: removing it    is part of a cycle
  disconnects 3,4
```

*Socratic prompt: "If every edge in a graph lies on at least one cycle, can the graph have any bridges? Why or why not?"*

### Practice Problems

| Problem (LC#) | Key Twist |
|--------------|-----------|
| Critical Connections in a Network (1192) | Direct application of bridge finding |
| Minimize Malware Spread II (928) | Removing nodes; related to articulation points and bridges |

---

## 5. Bridge Finding Online

### Core Insight

The offline bridge algorithm requires the full graph upfront. The online variant handles edges arriving one at a time, maintaining the bridge count dynamically. It uses two Disjoint Set Union (DSU) structures:

1. **DSU for 2-edge-connected components** (`dsu_2ecc`): vertices in the same 2-edge-connected component (connected even after removing any single edge) are merged.
2. **DSU for connected components** (`dsu_cc`): tracks which tree each vertex belongs to.

The key observation is that bridges partition the graph into 2-edge-connected components. If we contract each such component into a single vertex and keep only the bridges as edges, we get a forest (acyclic graph). Adding a new edge falls into three cases:

- **Different connected components**: the new edge becomes a bridge; merge the two trees.
- **Same 2-edge-connected component**: no structural change.
- **Same tree, different 2-edge-connected components**: the new edge creates a cycle, destroying all bridges on the path between the two endpoints. Compress the entire path into one 2-edge-connected component.

*Socratic prompt: "When a new edge creates a cycle in the bridge tree, every edge on the path between its endpoints stops being a bridge. Why does this happen, and how does the LCA help us find all these edges efficiently?"*

### Template

```python
from typing import List


class OnlineBridgeFinder:
    """Online bridge finding with DSU.

    Supports adding edges one at a time and maintaining bridge count.

    Time:  O(n log n + m log n) overall for n vertices and m edge additions.
    Space: O(n).
    """

    def __init__(self, n: int):
        self.n = n
        self.parent = [-1] * n       # parent in the bridge forest
        self.dsu_2ecc = list(range(n))  # DSU for 2-edge-connected components
        self.dsu_cc = list(range(n))    # DSU for connected components
        self.dsu_cc_size = [1] * n      # size of each connected component
        self.bridges = 0
        self.lca_iteration = 0
        self.last_visit = [0] * n

    def find_2ecc(self, v: int) -> int:
        """Find representative of the 2-edge-connected component of v."""
        if v == -1:
            return -1
        if self.dsu_2ecc[v] != v:
            self.dsu_2ecc[v] = self.find_2ecc(self.dsu_2ecc[v])
        return self.dsu_2ecc[v]

    def find_cc(self, v: int) -> int:
        """Find representative of the connected component of v."""
        v = self.find_2ecc(v)
        if self.dsu_cc[v] != v:
            self.dsu_cc[v] = self.find_cc(self.dsu_cc[v])
        return self.dsu_cc[v]

    def _make_root(self, v: int) -> None:
        """Re-root the tree so that v becomes the root."""
        root = v
        child = -1
        while v != -1:
            p = self.find_2ecc(self.parent[v])
            self.parent[v] = child
            self.dsu_cc[v] = root
            child = v
            v = p
        self.dsu_cc_size[root] = self.dsu_cc_size[child]

    def _merge_path(self, a: int, b: int) -> None:
        """Compress all 2ecc components on the path from a to b via LCA.

        Every edge on this path was a bridge; now they are all part of a cycle.
        """
        self.lca_iteration += 1
        path_a = []
        path_b = []
        lca = -1

        while lca == -1:
            if a != -1:
                a = self.find_2ecc(a)
                path_a.append(a)
                if self.last_visit[a] == self.lca_iteration:
                    lca = a
                    break
                self.last_visit[a] = self.lca_iteration
                a = self.parent[a]
            if b != -1:
                b = self.find_2ecc(b)
                path_b.append(b)
                if self.last_visit[b] == self.lca_iteration:
                    lca = b
                    break
                self.last_visit[b] = self.lca_iteration
                b = self.parent[b]

        # Compress path_a up to lca
        for v in path_a:
            self.dsu_2ecc[v] = lca
            if v == lca:
                break
            self.bridges -= 1

        # Compress path_b up to lca
        for v in path_b:
            self.dsu_2ecc[v] = lca
            if v == lca:
                break
            self.bridges -= 1

    def add_edge(self, a: int, b: int) -> None:
        """Add an undirected edge (a, b) and update bridge count.

        Time: O(log n) amortized per edge addition.
        """
        a = self.find_2ecc(a)
        b = self.find_2ecc(b)
        if a == b:
            return  # Same 2-edge-connected component; nothing changes

        ca = self.find_cc(a)
        cb = self.find_cc(b)

        if ca != cb:
            # Different connected components: new bridge
            self.bridges += 1
            if self.dsu_cc_size[ca] > self.dsu_cc_size[cb]:
                a, b = b, a
                ca, cb = cb, ca
            self._make_root(a)
            self.parent[a] = b
            self.dsu_cc[a] = cb
            self.dsu_cc_size[cb] += self.dsu_cc_size[ca]
        else:
            # Same tree: merge the path, destroying bridges along it
            self._merge_path(a, b)
```

| Operation | Time (amortized) | Space |
|-----------|-----------------|-------|
| Add edge | O(log n) | O(n) total |
| Query bridge count | O(1) | -- |
| Total for m edges | O(n log n + m log n) | O(n) |

*Socratic prompt: "Why do we always re-root the smaller tree when merging two connected components? What total work bound does this give us across all merge operations?"*

### Practice Problems

| Problem (LC#) | Key Twist |
|--------------|-----------|
| -- | This is primarily a competitive programming technique |
| Critical Connections in a Network (1192) | Can be solved offline; online variant useful if edges arrive dynamically |

---

## 6. Articulation Points (Cut Vertices)

### Core Insight

An articulation point is a vertex whose removal (along with all its incident edges) increases the number of connected components. The algorithm is nearly identical to bridge finding, with two conditions:

1. **Non-root vertex `v`**: `v` is an articulation point if it has any child `to` in the DFS tree such that `low[to] >= tin[v]`. This means the subtree rooted at `to` cannot reach any ancestor of `v` without going through `v`.

2. **Root vertex**: The root is an articulation point if and only if it has more than one child in the DFS tree.

Note the critical difference from bridges: bridges use `low[to] > tin[v]` (strict), while articulation points use `low[to] >= tin[v]` (non-strict). For bridges, the subtree must not reach `v` at all. For articulation points, reaching `v` itself is not enough because removing `v` would still disconnect the subtree.

*Socratic prompt: "Consider a vertex v with two children in the DFS tree, where both subtrees have back edges to v but not to v's ancestors. Is v an articulation point? Why does `>= tin[v]` catch this but `> tin[v]` would not?"*

### Template

```python
from typing import List, Set


def find_articulation_points(n: int, adj: List[List[int]]) -> Set[int]:
    """Find all articulation points (cut vertices) in an undirected graph.

    Time:  O(V + E).
    Space: O(V).

    Args:
        n: Number of vertices.
        adj: Adjacency list for undirected graph.

    Returns:
        Set of articulation point vertex indices.
    """
    tin = [-1] * n
    low = [-1] * n
    visited = [False] * n
    cutpoints = set()
    timer = [0]

    def dfs(v: int, parent: int) -> None:
        visited[v] = True
        tin[v] = low[v] = timer[0]
        timer[0] += 1
        children = 0

        for to in adj[v]:
            if to == parent:
                continue
            if visited[to]:
                # Back edge
                low[v] = min(low[v], tin[to])
            else:
                # Tree edge
                children += 1
                dfs(to, v)
                low[v] = min(low[v], low[to])

                # Non-root articulation point condition
                if low[to] >= tin[v] and parent != -1:
                    cutpoints.add(v)

        # Root articulation point condition
        if parent == -1 and children > 1:
            cutpoints.add(v)

    for i in range(n):
        if not visited[i]:
            dfs(i, -1)

    return cutpoints
```

### Bridge vs Articulation Point Comparison

| Property | Bridge | Articulation Point |
|----------|--------|--------------------|
| What is it? | An edge | A vertex |
| Condition | `low[to] > tin[v]` (strict) | `low[to] >= tin[v]` (non-strict) |
| Root special case | No | Yes: root is AP iff it has > 1 DFS child |
| Removal effect | Increases components by exactly 1 | May increase components by 1 or more |
| Can exist in a cycle? | No | Yes (vertex shared by two cycles) |

| Operation | Time | Space |
|-----------|------|-------|
| Find all articulation points | O(V + E) | O(V) |
| Find biconnected components | O(V + E) | O(V + E) |

*Socratic prompt: "A bridge's endpoints are always articulation points (unless they have degree 1). Can you prove this using the bridge and articulation point conditions? What about the converse -- is every articulation point adjacent to a bridge?"*

### Practice Problems

| Problem (LC#) | Key Twist |
|--------------|-----------|
| Critical Connections in a Network (1192) | Bridge variant; articulation point logic is closely related |
| Minimize Malware Spread (924) | Removing nodes from initial infected set; related to cut vertices |
| Minimize Malware Spread II (928) | Which single node removal minimizes spread; articulation point reasoning |
| Biconnected Components | Partition edges into biconnected components using articulation points + stack |

---

## 7. Strongly Connected Components (SCC)

### Core Insight

A strongly connected component of a directed graph is a maximal set of vertices where every vertex can reach every other vertex. The condensation graph (contract each SCC into one node) is always a DAG, enabling topological sort and DP on the condensed structure.

Two classic O(V + E) algorithms exist:

**Kosaraju's algorithm** uses two DFS passes:
1. DFS on the original graph, recording vertices by exit time.
2. DFS on the transposed (reversed) graph in decreasing exit time order. Each DFS tree in pass 2 is one SCC.

The key theorem: if there is an edge from SCC C to SCC C' in the condensation, then the maximum exit time in C is greater than the maximum exit time in C'.

**Tarjan's algorithm** uses a single DFS pass with a stack:
- Maintain a stack of "unclaimed" vertices.
- Track `tin[v]` (entry time) and `low[v]` (minimum entry time reachable from `v`'s subtree via tree edges and back edges to unclaimed vertices).
- When `low[v] == tin[v]`, vertex `v` is the root of an SCC; pop all vertices from the stack down to `v`.

*Socratic prompt: "Kosaraju's algorithm reverses all edges and processes vertices by decreasing exit time. Why does this guarantee that each DFS in the second pass visits exactly one SCC? Hint: think about what happens to the condensation DAG when you reverse edges."*

### Template: Kosaraju's Algorithm

```python
from typing import List, Tuple


def kosaraju_scc(n: int, adj: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
    """Find SCCs using Kosaraju's two-pass algorithm.

    Time:  O(V + E).
    Space: O(V + E) for the transposed graph.

    Args:
        n: Number of vertices.
        adj: Adjacency list for directed graph.

    Returns:
        (components, comp_id) where components[i] is list of vertices in SCC i,
        and comp_id[v] is the SCC index of vertex v.
    """
    visited = [False] * n
    order = []  # vertices sorted by exit time

    # Pass 1: DFS on original graph, record exit order
    def dfs1(v: int) -> None:
        visited[v] = True
        for u in adj[v]:
            if not visited[u]:
                dfs1(u)
        order.append(v)

    for v in range(n):
        if not visited[v]:
            dfs1(v)

    # Build transposed graph
    adj_rev = [[] for _ in range(n)]
    for v in range(n):
        for u in adj[v]:
            adj_rev[u].append(v)

    # Pass 2: DFS on transposed graph in reverse exit-time order
    visited = [False] * n
    components = []
    comp_id = [-1] * n

    def dfs2(v: int, comp: List[int]) -> None:
        visited[v] = True
        comp.append(v)
        comp_id[v] = len(components)
        for u in adj_rev[v]:
            if not visited[u]:
                dfs2(u, comp)

    for v in reversed(order):
        if not visited[v]:
            comp = []
            dfs2(v, comp)
            components.append(comp)

    return components, comp_id
```

### Template: Tarjan's Algorithm

```python
from typing import List, Tuple


def tarjan_scc(n: int, adj: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
    """Find SCCs using Tarjan's single-pass algorithm.

    Time:  O(V + E).
    Space: O(V).

    Args:
        n: Number of vertices.
        adj: Adjacency list for directed graph.

    Returns:
        (components, comp_id) where components[i] is list of vertices in SCC i,
        and comp_id[v] is the SCC index of vertex v.
    """
    tin = [-1] * n
    low = [-1] * n
    comp_id = [-1] * n  # -1 means unclaimed
    stack = []
    timer = [0]
    components = []

    def dfs(v: int) -> None:
        tin[v] = low[v] = timer[0]
        timer[0] += 1
        stack.append(v)

        for u in adj[v]:
            if tin[u] == -1:
                # Unvisited: tree edge
                dfs(u)
                low[v] = min(low[v], low[u])
            elif comp_id[u] == -1:
                # Visited but unclaimed: back edge to vertex still on stack
                low[v] = min(low[v], tin[u])
            # If comp_id[u] != -1, u is already in a finished SCC (cross edge)

        # If v is an SCC root, pop the entire SCC from the stack
        if low[v] == tin[v]:
            comp = []
            while True:
                u = stack.pop()
                comp_id[u] = len(components)
                comp.append(u)
                if u == v:
                    break
            components.append(comp)

    for v in range(n):
        if tin[v] == -1:
            dfs(v)

    return components, comp_id
```

### Template: Condensation Graph

```python
def build_condensation(
    n: int, adj: List[List[int]], comp_id: List[int], num_components: int
) -> List[List[int]]:
    """Build the condensation DAG from the original graph and SCC labels.

    Time:  O(V + E).
    Space: O(V + E) for the condensation graph.

    Args:
        n: Number of vertices in original graph.
        adj: Original adjacency list.
        comp_id: SCC index for each vertex.
        num_components: Total number of SCCs.

    Returns:
        Adjacency list of the condensation DAG.
    """
    adj_cond = [[] for _ in range(num_components)]
    edge_set = set()  # avoid duplicate edges in condensation

    for v in range(n):
        for u in adj[v]:
            if comp_id[v] != comp_id[u]:
                edge = (comp_id[v], comp_id[u])
                if edge not in edge_set:
                    edge_set.add(edge)
                    adj_cond[comp_id[v]].append(comp_id[u])

    return adj_cond
```

### Kosaraju vs Tarjan Comparison

| Feature | Kosaraju's | Tarjan's |
|---------|-----------|----------|
| DFS passes | 2 | 1 |
| Needs transposed graph | Yes | No |
| Extra space | O(V + E) for reverse graph | O(V) for stack |
| Conceptual simplicity | Higher (two clean passes) | Lower (subtle stack logic) |
| Time complexity | O(V + E) | O(V + E) |
| Component order | Reverse topological order of condensation | Reverse topological order of condensation |
| Interview preference | More common (easier to explain) | More common in competitive programming |

| Operation | Time | Space |
|-----------|------|-------|
| Find all SCCs (either algorithm) | O(V + E) | O(V + E) |
| Build condensation DAG | O(V + E) | O(V + E) |
| DP on condensation | O(V + E) | O(V) |

*Socratic prompt: "After computing SCCs and building the condensation DAG, how would you find the minimum number of edges to add so that the entire directed graph becomes strongly connected? Hint: think about sources and sinks in the DAG."*

### Practice Problems

| Problem (LC#) | Key Twist |
|--------------|-----------|
| Course Schedule IV (1462) | Reachability queries; SCC condensation + DAG reachability |
| Satisfiability of Equality Equations (990) | Union-Find or SCC-like reasoning on equality/inequality constraints |
| 2-SAT problems | Model as implication graph; SCC determines satisfiability (competitive programming) |
| Minimum Number of Vertices to Reach All Nodes (1557) | Find sources in a DAG (related to condensation) |
| Longest Path in DAG | After condensation, DP on the DAG for longest path |

---

## 8. Strong Orientation

### Core Insight

A strong orientation assigns a direction to every edge of an undirected graph such that the resulting directed graph is strongly connected (every vertex can reach every other). By Robbins' theorem, a connected undirected graph has a strong orientation if and only if it has no bridges. The intuition: a bridge directed either way creates a one-way bottleneck that prevents traversal in the other direction.

The algorithm is beautifully simple:
1. Run DFS from any vertex.
2. Orient tree edges away from the root (parent to child).
3. Orient back edges from descendant to ancestor.

This works because:
- Every vertex can be reached from the root via tree edges (directed downward).
- Every vertex can reach the root via back edges (directed upward). Since the graph is bridgeless, every subtree must have at least one back edge reaching above, ensuring a path back to the root.

*Socratic prompt: "Why does the absence of bridges guarantee that every vertex has a back-edge path to the root? What would happen if some subtree had no back edge to an ancestor?"*

### Template

```python
from typing import List, Tuple


def strong_orientation(
    n: int, edges: List[Tuple[int, int]]
) -> Tuple[bool, List[Tuple[int, int]]]:
    """Orient edges of an undirected graph to make it strongly connected.

    First checks that the graph is bridgeless (necessary and sufficient
    condition by Robbins' theorem). If bridgeless, orients edges via DFS.

    Time:  O(V + E).
    Space: O(V + E).

    Args:
        n: Number of vertices.
        edges: List of undirected edges as (u, v) tuples.

    Returns:
        (success, directed_edges) where success is True if orientation exists,
        and directed_edges is the list of directed edges.
    """
    # Build adjacency list with edge indices
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))

    tin = [-1] * n
    low = [-1] * n
    timer = [0]
    edge_used = [False] * len(edges)
    orientation = [None] * len(edges)  # (from, to) for each edge
    has_bridge = [False]

    def dfs(v: int) -> None:
        tin[v] = low[v] = timer[0]
        timer[0] += 1

        for to, edge_idx in adj[v]:
            if edge_used[edge_idx]:
                continue
            edge_used[edge_idx] = True

            # Orient this edge from v to 'to'
            orientation[edge_idx] = (v, to)

            if tin[to] == -1:
                # Tree edge
                dfs(to)
                low[v] = min(low[v], low[to])
                if low[to] > tin[v]:
                    has_bridge[0] = True
            else:
                # Back edge
                low[v] = min(low[v], tin[to])

    # Check connectivity: all vertices must be reachable
    dfs(0)
    if has_bridge[0] or any(tin[v] == -1 for v in range(n)):
        return False, []

    return True, [o for o in orientation if o is not None]


def check_bridgeless(n: int, adj: List[List[int]]) -> bool:
    """Check if an undirected graph is bridgeless (2-edge-connected).

    Time:  O(V + E).
    Space: O(V).

    A graph can be strongly oriented iff it is connected and bridgeless
    (Robbins' theorem, 1939).
    """
    tin = [-1] * n
    low = [-1] * n
    timer = [0]

    def dfs(v: int, parent: int) -> bool:
        tin[v] = low[v] = timer[0]
        timer[0] += 1
        for to in adj[v]:
            if to == parent:
                continue
            if tin[to] != -1:
                low[v] = min(low[v], tin[to])
            else:
                if not dfs(to, v):
                    return False
                low[v] = min(low[v], low[to])
                if low[to] > tin[v]:
                    return False  # Found a bridge
        return True

    if not dfs(0, -1):
        return False
    return all(tin[v] != -1 for v in range(n))
```

### When Can You Strongly Orient?

| Graph Property | Strong Orientation Exists? | Reason |
|---------------|---------------------------|--------|
| Connected, no bridges | Yes | Robbins' theorem |
| Connected, has bridges | No | Bridge creates one-way bottleneck |
| Disconnected | No | Cannot reach between components regardless of orientation |
| Single vertex | Yes (trivially) | Already strongly connected |
| Tree (n > 1) | No | Every edge is a bridge |
| Cycle | Yes | Orient all edges in one direction |

| Operation | Time | Space |
|-----------|------|-------|
| Check if bridgeless | O(V + E) | O(V) |
| Compute strong orientation | O(V + E) | O(V + E) |

*Socratic prompt: "Given a graph with bridges, you cannot make the whole graph strongly connected. But you can still orient edges to maximize the number of strongly connected pairs. How would you approach this? Hint: think about what happens within each 2-edge-connected component."*

### Practice Problems

| Problem (LC#) | Key Twist |
|--------------|-----------|
| -- | Primarily a competitive programming and graph theory topic |
| Critical Connections in a Network (1192) | Finding bridges is the prerequisite check for strong orientation |
| Robbins' theorem applications | Orient a network for fault tolerance (real-world application) |

---

## Attribution

Content synthesized from cp-algorithms.com articles on [Breadth First Search](https://cp-algorithms.com/graph/breadth-first-search.html), [Depth First Search](https://cp-algorithms.com/graph/depth-first-search.html), [Search for Connected Components](https://cp-algorithms.com/graph/search-for-connected-components.html), [Finding Bridges](https://cp-algorithms.com/graph/bridge-searching.html), [Finding Bridges Online](https://cp-algorithms.com/graph/bridge-searching-online.html), [Finding Articulation Points](https://cp-algorithms.com/graph/cutpoints.html), [Strongly Connected Components](https://cp-algorithms.com/graph/strongly-connected-components.html), and [Strong Orientation](https://cp-algorithms.com/graph/strong-orientation.html). Algorithms and complexity analyses are from the original articles; code has been translated to Python with added commentary for the leetcode-teacher reference format.
