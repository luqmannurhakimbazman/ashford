# Graph Special Topics

Advanced graph techniques for competitive programming and hard interview problems: topological sorting with applications, edge/vertex connectivity, re-rooting (tree painting), 2-SAT via implication graphs, and Heavy-Light Decomposition for path queries. Builds on the base graph templates in `graph-algorithms.md` and the traversal patterns in `graph-traversal-advanced.md`. For tree-specific structures (LCA, Euler tour), see `advanced-tree-structures.md`.

---

## Quick Reference Table

| Pattern | Recognition Signals | Key Problems | Section |
|---------|-------------------|--------------|---------|
| Topological Sort (Kahn's BFS + DFS) | "Course prerequisites", "task ordering", "DAG longest path", "cycle detection in directed graph" | Course Schedule (207), Course Schedule II (210), Alien Dictionary (269) | 1 |
| Edge & Vertex Connectivity | "Minimum edges/vertices to disconnect", "edge-disjoint paths", "network reliability" | Critical Connections (1192), Network Delay Time (743) | 2 |
| Tree Painting (Re-rooting) | "Answer for every root", "re-root DP", "sum of distances to all nodes" | Sum of Distances in Tree (834), Count Nodes at Distance K | 3 |
| 2-SAT | "Boolean variables with pairwise constraints", "implication graph", "at most one / at least one" | -- (competitive programming: Codeforces, CSES) | 4 |
| Heavy-Light Decomposition | "Path queries on tree", "update/query path values", "segment tree on tree" | SPOJ QTREE, CSES Path Queries II | 5 |

---

## Topic Connection Map

```
Topological Sort (Kahn's / DFS)
    │
    ├── Longest path in DAG (DP on topo order)
    │
    └── 2-SAT ←── SCC (Kosaraju / Tarjan)
                      │
                      └── Edge/Vertex Connectivity ←── Max Flow (Ford-Fulkerson)

Tree Painting (Re-rooting DP)
    │
    └── Heavy-Light Decomposition ←── Euler Tour + Segment Tree
                                          │
                                          └── LCA (advanced-tree-structures.md)
```

---

## 1. Topological Sort (Kahn's BFS + DFS)

### Core Insight

A topological ordering of a DAG is a linear ordering of vertices such that for every directed edge (u, v), vertex u appears before v. This ordering exists **if and only if** the graph has no cycles. Two classic approaches: DFS-based (reverse post-order) and Kahn's BFS (iterative in-degree reduction).

**Key invariant (DFS):** The exit time of vertex v is always greater than the exit time of any vertex reachable from v. Reversing the exit-time order gives a valid topological order.

**Key invariant (Kahn's):** At each step, a vertex with in-degree 0 has no remaining unprocessed prerequisites — it is safe to place next.

*Socratic prompt: "In Kahn's algorithm, why does a vertex with in-degree 0 have no unsatisfied dependencies? What does in-degree represent in a prerequisite graph?"*

### Template: DFS-Based Topological Sort

```python
def topo_sort_dfs(n: int, adj: list[list[int]]) -> list[int]:
    """Topological sort via DFS (reverse post-order).

    Args:
        n: Number of vertices (0 to n-1).
        adj: Adjacency list for a directed graph.

    Returns:
        A list of vertices in topological order,
        or an empty list if a cycle is detected.

    Time: O(V + E). Space: O(V).
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    order = []
    has_cycle = False

    def dfs(v: int) -> None:
        nonlocal has_cycle
        if has_cycle:
            return
        color[v] = GRAY
        for u in adj[v]:
            if color[u] == GRAY:
                has_cycle = True
                return
            if color[u] == WHITE:
                dfs(u)
        color[v] = BLACK
        order.append(v)

    for i in range(n):
        if color[i] == WHITE:
            dfs(i)

    if has_cycle:
        return []
    order.reverse()
    return order
```

*Socratic prompt: "Why do we use three colors (WHITE, GRAY, BLACK) instead of a simple visited boolean? What does encountering a GRAY node during DFS tell us?"*

### Template: Kahn's Algorithm (BFS)

```python
from collections import deque

def topo_sort_kahn(n: int, adj: list[list[int]]) -> list[int]:
    """Topological sort via Kahn's algorithm (BFS in-degree reduction).

    Args:
        n: Number of vertices (0 to n-1).
        adj: Adjacency list for a directed graph.

    Returns:
        A list of vertices in topological order,
        or an empty list if a cycle is detected.

    Time: O(V + E). Space: O(V).
    """
    in_degree = [0] * n
    for u in range(n):
        for v in adj[u]:
            in_degree[v] += 1

    queue = deque(v for v in range(n) if in_degree[v] == 0)
    order = []

    while queue:
        v = queue.popleft()
        order.append(v)
        for u in adj[v]:
            in_degree[u] -= 1
            if in_degree[u] == 0:
                queue.append(u)

    if len(order) != n:
        return []  # Cycle detected: not all vertices processed
    return order
```

*Socratic prompt: "If Kahn's algorithm processes fewer than n vertices, why does that guarantee a cycle exists? Think about what remains in the graph."*

### Application: Longest Path in a DAG

DP on topological order computes the longest (or shortest) path in O(V + E). This is impossible for general graphs (NP-hard) but trivial for DAGs because topological order ensures all predecessors are processed first.

```python
def longest_path_dag(n: int, adj: list[list[tuple[int, int]]]) -> list[int]:
    """Longest path from any source to each vertex in a weighted DAG.

    Args:
        n: Number of vertices.
        adj: Adjacency list of (neighbor, weight) pairs.

    Returns:
        dist[v] = length of longest path ending at v.

    Time: O(V + E). Space: O(V).
    """
    order = topo_sort_dfs(n, [list({u for u, _ in adj[v]}) for v in range(n)])
    if not order:
        return []  # Cycle

    dist = [0] * n
    for v in order:
        for u, w in adj[v]:
            if dist[v] + w > dist[u]:
                dist[u] = dist[v] + w
    return dist
```

*Socratic prompt: "Why can we compute longest path in O(V+E) on a DAG but not on a general graph? What property of DAGs makes the DP work?"*

### Application: Counting Paths in a DAG

```python
def count_paths_dag(n: int, adj: list[list[int]], src: int, dst: int) -> int:
    """Count the number of distinct paths from src to dst in a DAG.

    Time: O(V + E). Space: O(V).
    """
    order = topo_sort_dfs(n, adj)
    if not order:
        return 0

    cnt = [0] * n
    cnt[src] = 1
    for v in order:
        for u in adj[v]:
            cnt[u] += cnt[v]
    return cnt[dst]
```

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| DFS-based topo sort | O(V + E) | O(V) |
| Kahn's BFS topo sort | O(V + E) | O(V) |
| Longest path in DAG | O(V + E) | O(V) |
| Count paths in DAG | O(V + E) | O(V) |

### When to Use Which

| Scenario | Use |
|----------|-----|
| Need cycle detection | DFS (three-color) |
| Need all valid orderings (level-based) | Kahn's (can use min-heap for lexicographic order) |
| Parallel scheduling (find critical path) | Kahn's with level tracking |
| DP on DAG (longest path, count paths) | Either; DFS order is natural |

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Course Schedule (207) | Cycle detection in prerequisite graph (Kahn's or DFS) |
| Course Schedule II (210) | Return a valid topological order |
| Alien Dictionary (269) | Build graph from character ordering, topo sort |
| Longest Increasing Path in a Matrix (329) | Matrix cells as DAG, DFS with memoization (implicit topo sort) |
| Parallel Courses (1136) | Kahn's with level tracking for minimum semesters |
| CSES Longest Flight Route | Longest path in DAG (DP on topo order) |
| CSES Game Routes | Count paths in DAG |
| SPOJ TOPOSORT | Direct topological sort application |

---

## 2. Edge and Vertex Connectivity

### Core Insight

**Edge connectivity** (lambda) is the minimum number of edges whose removal disconnects the graph. **Vertex connectivity** (kappa) is the minimum number of vertices whose removal disconnects the graph. Whitney's inequality relates these to minimum degree (delta):

```
kappa <= lambda <= delta
```

The Ford-Fulkerson theorem connects edge connectivity to maximum flow: "The maximum number of edge-disjoint paths between two vertices equals the minimum number of edges separating them."

*Socratic prompt: "Why must vertex connectivity be at most edge connectivity? If removing k vertices disconnects the graph, can you always find k edges to remove instead?"*

### Template: Edge Connectivity via Max Flow

```python
from collections import defaultdict, deque

def bfs_augment(graph: dict[int, dict[int, int]], source: int,
                sink: int, parent: dict[int, int]) -> int:
    """BFS to find an augmenting path in the residual graph.

    Returns the bottleneck capacity, or 0 if no path exists.

    Time: O(V + E). Space: O(V).
    """
    visited = {source}
    queue = deque([source])
    parent.clear()

    while queue:
        u = queue.popleft()
        for v, cap in graph[u].items():
            if v not in visited and cap > 0:
                visited.add(v)
                parent[v] = u
                if v == sink:
                    # Trace back to find bottleneck
                    flow = float('inf')
                    node = sink
                    while node != source:
                        prev = parent[node]
                        flow = min(flow, graph[prev][node])
                        node = prev
                    return flow
                queue.append(v)
    return 0


def max_flow_edmonds_karp(n: int, edges: list[tuple[int, int]],
                          source: int, sink: int) -> int:
    """Edmonds-Karp max flow (BFS-based Ford-Fulkerson).

    Each undirected edge has capacity 1 for edge connectivity.

    Time: O(V * E^2). Space: O(V + E).
    """
    graph = defaultdict(lambda: defaultdict(int))
    for u, v in edges:
        graph[u][v] += 1
        graph[v][u] += 1

    total_flow = 0
    parent = {}
    while True:
        aug = bfs_augment(graph, source, sink, parent)
        if aug == 0:
            break
        total_flow += aug
        # Update residual graph
        node = sink
        while node != source:
            prev = parent[node]
            graph[prev][node] -= aug
            graph[node][prev] += aug
            node = prev
    return total_flow


def edge_connectivity(n: int, edges: list[tuple[int, int]]) -> int:
    """Compute the edge connectivity of an undirected graph.

    Fix vertex 0 as source, try all other vertices as sink.
    Edge connectivity = min over all sinks of max_flow(0, sink).

    Time: O(V^2 * E^2) using Edmonds-Karp. Space: O(V + E).
    """
    if n <= 1:
        return 0
    min_cut = float('inf')
    for t in range(1, n):
        flow = max_flow_edmonds_karp(n, edges, 0, t)
        min_cut = min(min_cut, flow)
    return min_cut
```

*Socratic prompt: "Why is it sufficient to fix one vertex as the source and try all others as the sink? Could the minimum cut not involve vertex 0 at all?"*

### Template: Vertex Connectivity via Vertex Splitting

To compute vertex connectivity, split each vertex v (except source and sink) into v_in and v_out connected by an edge of capacity 1. This transforms vertex cuts into edge cuts.

```python
def vertex_connectivity(n: int, edges: list[tuple[int, int]]) -> int:
    """Compute the vertex connectivity of an undirected graph.

    Vertex splitting: each vertex v becomes v_in (2*v) and v_out (2*v+1)
    with an internal edge of capacity 1. Original edges get capacity infinity.

    Time: O(V^3 * E^2). Space: O(V + E).
    """
    if n <= 1:
        return 0

    INF = n + 1  # Effectively infinity for capacity-1 internal edges
    min_cut = float('inf')

    for s in range(n):
        for t in range(s + 1, n):
            # Build split graph: 2*n nodes
            split_edges = []
            for v in range(n):
                if v != s and v != t:
                    # Internal edge: v_in -> v_out, capacity 1
                    split_edges.append((2 * v, 2 * v + 1, 1))
                else:
                    # Source and sink are not split (infinite internal cap)
                    split_edges.append((2 * v, 2 * v + 1, INF))

            for u, v in edges:
                # Original edge: u_out -> v_in and v_out -> u_in, capacity INF
                split_edges.append((2 * u + 1, 2 * v, INF))
                split_edges.append((2 * v + 1, 2 * u, INF))

            # Run max flow on split graph from s_out to t_in
            graph = defaultdict(lambda: defaultdict(int))
            for a, b, cap in split_edges:
                graph[a][b] += cap

            parent = {}
            total = 0
            while True:
                aug = bfs_augment(graph, 2 * s + 1, 2 * t, parent)
                if aug == 0:
                    break
                total += aug
                node = 2 * t
                while node != 2 * s + 1:
                    prev = parent[node]
                    graph[prev][node] -= aug
                    graph[node][prev] += aug
                    node = prev
            min_cut = min(min_cut, total)

    return min_cut
```

### Bridge Detection (Edge Connectivity = 1)

For the special case of finding bridges (edges whose removal disconnects the graph), Tarjan's bridge-finding algorithm runs in O(V + E):

```python
def find_bridges(n: int, adj: list[list[int]]) -> list[tuple[int, int]]:
    """Find all bridges in an undirected graph using Tarjan's algorithm.

    A bridge is an edge whose removal increases the number of
    connected components.

    Time: O(V + E). Space: O(V).
    """
    disc = [-1] * n
    low = [-1] * n
    bridges = []
    timer = [0]

    def dfs(u: int, parent: int) -> None:
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        for v in adj[u]:
            if disc[v] == -1:
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.append((u, v))
            elif v != parent:
                low[u] = min(low[u], disc[v])

    for i in range(n):
        if disc[i] == -1:
            dfs(i, -1)

    return bridges
```

*Socratic prompt: "In Tarjan's bridge algorithm, what does `low[v] > disc[u]` mean intuitively? Why does it imply (u, v) is a bridge?"*

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Edge connectivity (Edmonds-Karp) | O(V^2 * E^2) | O(V + E) |
| Vertex connectivity (splitting) | O(V^3 * E^2) | O(V + E) |
| Bridge detection (Tarjan) | O(V + E) | O(V) |
| Articulation point detection | O(V + E) | O(V) |

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Critical Connections in a Network (1192) | Find all bridges (Tarjan's algorithm) |
| Minimum Number of Days to Disconnect Island (1568) | Vertex connectivity of grid graph (answer is 0, 1, or 2) |
| CSES Flight Routes Check | Check if graph is strongly connected (related to connectivity) |
| Codeforces 118E | Edge connectivity and minimum cut |

---

## 3. Tree Painting (Re-rooting Technique)

### Core Insight

Many tree problems ask: "Compute f(v) for every vertex v, where f(v) is the answer when v is the root." A naive approach computes f(v) from scratch for each root in O(N) each, giving O(N^2) total. The **re-rooting technique** computes f(v) for all vertices in O(N) total by:

1. **Root arbitrarily** (say at vertex 0) and compute f(0) using a standard DFS.
2. **Re-root DFS:** When moving the root from parent p to child c, update f(c) from f(p) using the relationship between their subtrees. The subtree of c "loses" c's children and "gains" everything else.

*Socratic prompt: "When you move the root from vertex p to its child c, how does the 'subtree' of each vertex change? What does c gain that it didn't have before?"*

### Template: Sum of Distances in Tree

The classic re-rooting problem: for each vertex v, compute the sum of distances from v to all other vertices.

```python
def sum_of_distances(n: int, edges: list[list[int]]) -> list[int]:
    """Sum of distances from each vertex to all other vertices.

    Two-pass re-rooting technique:
      Pass 1 (post-order): compute subtree sizes and root answer.
      Pass 2 (pre-order): re-root to compute answer for every vertex.

    Time: O(N). Space: O(N).
    """
    if n == 1:
        return [0]

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    subtree_size = [1] * n
    dist_sum = [0] * n

    # Pass 1: Root at 0, compute subtree sizes and dist_sum[0]
    parent = [-1] * n
    order = []  # BFS order for iterative post-order
    visited = [False] * n
    from collections import deque
    queue = deque([0])
    visited[0] = True
    while queue:
        v = queue.popleft()
        order.append(v)
        for u in adj[v]:
            if not visited[u]:
                visited[u] = True
                parent[u] = v
                queue.append(u)

    # Post-order: process leaves first (reverse BFS order)
    for v in reversed(order):
        for u in adj[v]:
            if parent[u] == v:  # u is child of v
                subtree_size[v] += subtree_size[u]
                dist_sum[v] += dist_sum[u] + subtree_size[u]

    # Pass 2: Re-root from parent to child
    # When moving root from p to c:
    #   dist_sum[c] = dist_sum[p] - subtree_size[c] + (n - subtree_size[c])
    # Explanation: c's subtree gets 1 closer (size[c] nodes),
    #              everything else gets 1 farther (n - size[c] nodes).
    for v in order:
        for u in adj[v]:
            if parent[u] == v:  # u is child of v
                dist_sum[u] = dist_sum[v] - subtree_size[u] + (n - subtree_size[u])

    return dist_sum
```

*Socratic prompt: "In the re-rooting formula `dist_sum[c] = dist_sum[p] - size[c] + (n - size[c])`, why do size[c] nodes get closer and (n - size[c]) nodes get farther? Draw a small tree and verify."*

### Generic Re-rooting Framework

```python
def reroot_dp(n: int, adj: list[list[int]],
              leaf_val,
              merge,
              finalize,
              add_edge,
              remove_edge) -> list:
    """Generic re-rooting DP framework.

    Args:
        n: Number of vertices.
        adj: Adjacency list (undirected tree).
        leaf_val: Base value for a leaf node.
        merge(accumulated, child_val): Combine child results.
        finalize(accumulated, vertex): Finalize after all children merged.
        add_edge(val): Transform value when moving across an edge (child -> parent).
        remove_edge(val): Inverse of add_edge (parent -> child direction).

    Returns:
        answer[v] for each vertex v as root.

    Time: O(N). Space: O(N).
    """
    parent = [-1] * n
    order = []
    visited = [False] * n
    from collections import deque
    queue = deque([0])
    visited[0] = True
    while queue:
        v = queue.popleft()
        order.append(v)
        for u in adj[v]:
            if not visited[u]:
                visited[u] = True
                parent[u] = v
                queue.append(u)

    # Pass 1: Compute dp_down[v] = answer for subtree rooted at v
    dp_down = [leaf_val] * n
    for v in reversed(order):
        accumulated = leaf_val
        for u in adj[v]:
            if parent[u] == v:
                accumulated = merge(accumulated, add_edge(dp_down[u]))
        dp_down[v] = finalize(accumulated, v)

    # Pass 2: Re-root
    answer = [None] * n
    answer[0] = dp_down[0]

    for v in order:
        for u in adj[v]:
            if parent[u] == v:
                # Remove u's contribution from v, add complement to u
                v_without_u = remove_edge(dp_down[v], add_edge(dp_down[u]))
                dp_up = add_edge(v_without_u)
                answer[u] = finalize(merge(dp_down[u], dp_up), u)

    return answer
```

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Re-rooting (two-pass DFS/BFS) | O(N) | O(N) |
| Naive per-root DFS | O(N^2) | O(N) |

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Sum of Distances in Tree (834) | Classic re-rooting: sum of all pairwise distances per root |
| Minimum Height Trees (310) | Find roots that minimize tree height (related to re-rooting / pruning leaves) |
| CSES Tree Distances I | Maximum distance from each vertex (re-root with max tracking) |
| CSES Tree Distances II | Sum of distances (direct re-rooting application) |
| Codeforces 1187E | Tree painting: assign weights based on subtree sizes, re-root for optimal root |

---

## 4. 2-SAT (Implication Graph + SCC)

### Core Insight

2-SAT solves Boolean satisfiability where each clause has exactly two literals. While general SAT is NP-complete, 2-SAT is solvable in **O(n + m)** by reducing it to Strongly Connected Components on an **implication graph**.

**The key reduction:** Each clause `(a OR b)` is equivalent to two implications: `(NOT a => b)` AND `(NOT b => a)`. Build a directed graph of these implications, then:
- A solution exists **iff** no variable x and its negation NOT x are in the same SCC.
- The assignment is extracted from the topological order of SCCs.

*Socratic prompt: "Why is the clause (a OR b) equivalent to (NOT a => b) AND (NOT b => a)? Try the truth table: when is (a OR b) false, and what do the implications say in that case?"*

### Template: 2-SAT Solver

```python
class TwoSAT:
    """2-SAT solver using Kosaraju's SCC algorithm.

    Variable x is represented as node 2*x (positive) and 2*x+1 (negative).
    For variable x:
        - True literal  = 2*x
        - False literal = 2*x + 1
        - Negation: literal ^ 1

    Time: O(n + m) where n = variables, m = clauses. Space: O(n + m).
    """

    def __init__(self, n: int):
        """Initialize solver for n Boolean variables (0 to n-1)."""
        self.n = n
        self.adj = [[] for _ in range(2 * n)]
        self.adj_rev = [[] for _ in range(2 * n)]

    def _lit(self, var: int, negated: bool) -> int:
        """Convert variable + sign to literal index."""
        return 2 * var + (1 if negated else 0)

    def add_clause(self, a: int, neg_a: bool, b: int, neg_b: bool) -> None:
        """Add clause (a OR b), where neg_a/neg_b indicate negation.

        Example: add_clause(0, False, 1, True) means (x0 OR NOT x1).
        """
        lit_a = self._lit(a, neg_a)
        lit_b = self._lit(b, neg_b)
        # (a OR b) => (NOT a => b) AND (NOT b => a)
        self.adj[lit_a ^ 1].append(lit_b)
        self.adj[lit_b ^ 1].append(lit_a)
        self.adj_rev[lit_b].append(lit_a ^ 1)
        self.adj_rev[lit_a].append(lit_b ^ 1)

    def add_implication(self, a: int, neg_a: bool, b: int, neg_b: bool) -> None:
        """Add implication (a => b), equivalent to clause (NOT a OR b)."""
        self.add_clause(a, not neg_a, b, neg_b)

    def set_true(self, var: int) -> None:
        """Force variable to be True. Equivalent to clause (var OR var)."""
        self.add_clause(var, False, var, False)

    def set_false(self, var: int) -> None:
        """Force variable to be False. Equivalent to clause (NOT var OR NOT var)."""
        self.add_clause(var, True, var, True)

    def at_most_one(self, literals: list[tuple[int, bool]]) -> None:
        """At most one of the given literals is true.

        For each pair (a, b): add clause (NOT a OR NOT b).
        Warning: O(k^2) clauses for k literals. For large k, use auxiliary variables.
        """
        for i in range(len(literals)):
            for j in range(i + 1, len(literals)):
                a_var, a_neg = literals[i]
                b_var, b_neg = literals[j]
                # NOT a OR NOT b
                self.add_clause(a_var, not a_neg, b_var, not b_neg)

    def solve(self) -> list[bool] | None:
        """Solve the 2-SAT instance.

        Returns:
            List of n booleans (assignment), or None if unsatisfiable.
        """
        num_nodes = 2 * self.n

        # Kosaraju's SCC: Pass 1 - DFS on original graph, record finish order
        visited = [False] * num_nodes
        finish_order = []

        def dfs1(v: int) -> None:
            stack = [(v, 0)]
            visited[v] = True
            while stack:
                node, idx = stack[-1]
                if idx < len(self.adj[node]):
                    stack[-1] = (node, idx + 1)
                    nxt = self.adj[node][idx]
                    if not visited[nxt]:
                        visited[nxt] = True
                        stack.append((nxt, 0))
                else:
                    stack.pop()
                    finish_order.append(node)

        for i in range(num_nodes):
            if not visited[i]:
                dfs1(i)

        # Pass 2 - DFS on reversed graph in reverse finish order
        comp = [-1] * num_nodes
        comp_id = 0

        def dfs2(v: int, c: int) -> None:
            stack = [v]
            comp[v] = c
            while stack:
                node = stack.pop()
                for nxt in self.adj_rev[node]:
                    if comp[nxt] == -1:
                        comp[nxt] = c
                        stack.append(nxt)

        for v in reversed(finish_order):
            if comp[v] == -1:
                dfs2(v, comp_id)
                comp_id += 1

        # Check satisfiability and extract assignment
        assignment = [False] * self.n
        for i in range(self.n):
            if comp[2 * i] == comp[2 * i + 1]:
                return None  # x and NOT x in same SCC => unsatisfiable
            # In Kosaraju's, later comp_id = earlier in topological order.
            # Assign True if comp[x] > comp[NOT x] (x comes later topo => x is true).
            assignment[i] = comp[2 * i] > comp[2 * i + 1]

        return assignment
```

*Socratic prompt: "Why does the SCC-based assignment avoid contradictions? If x is assigned True, what does `comp[x] > comp[NOT x]` guarantee about reachability in the implication graph?"*

### Common 2-SAT Modeling Patterns

| Constraint | How to Model |
|-----------|-------------|
| `a OR b` | `add_clause(a, False, b, False)` |
| `a AND b` | `set_true(a)` and `set_true(b)` |
| `a XOR b` (exactly one) | `add_clause(a, False, b, False)` and `add_clause(a, True, b, True)` |
| `a = b` (equivalence) | `add_implication(a, False, b, False)` and `add_implication(b, False, a, False)` |
| `a => b` | `add_implication(a, False, b, False)` |
| At most one of {a, b, c} | Pairwise `(NOT a OR NOT b)`, `(NOT a OR NOT c)`, `(NOT b OR NOT c)` |

### Usage Example

```python
def solve_example():
    """Example: 3 variables, clauses (x0 OR x1), (NOT x1 OR x2), (NOT x0 OR NOT x2).

    One satisfying assignment: x0=True, x1=True, x2=True.
    """
    solver = TwoSAT(3)
    solver.add_clause(0, False, 1, False)   # x0 OR x1
    solver.add_clause(1, True, 2, False)    # NOT x1 OR x2
    solver.add_clause(0, True, 2, True)     # NOT x0 OR NOT x2

    result = solver.solve()
    if result is None:
        print("UNSATISFIABLE")
    else:
        for i, val in enumerate(result):
            print(f"x{i} = {val}")
```

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Build implication graph | O(m) | O(n + m) |
| Kosaraju's SCC | O(n + m) | O(n + m) |
| Total 2-SAT solve | O(n + m) | O(n + m) |

where n = number of variables, m = number of clauses.

*Socratic prompt: "2-SAT is in P while 3-SAT is NP-complete. What structural property of 2-clause implications makes the SCC approach work, and why does it break for 3 literals per clause?"*

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| CSES Giant Pizza | Direct 2-SAT: each person wants at least one topping preference |
| Codeforces 776D (The Door Problem) | Doors controlled by switches, model switch states as 2-SAT variables |
| Codeforces 1215F (Radio Stations) | Frequency assignment with interference constraints |
| Kattis illumination | Light placement with coverage constraints |
| UVA Rectangles | Rectangle placement with overlap constraints |

---

## 5. Heavy-Light Decomposition (HLD)

### Core Insight

HLD partitions a tree's edges into **heavy** and **light** categories, forming disjoint chains (heavy paths) such that any root-to-leaf path crosses at most **O(log N)** light edges. By laying each heavy path contiguously in a flat array, we can answer path queries using a segment tree with O(log^2 N) per query.

**Why O(log N) chain changes?** Moving down a light edge reduces the subtree size by at least half (otherwise the edge would be heavy). Since the subtree size starts at N and halves at each light edge, there are at most log N light edges on any root-to-leaf path.

*Socratic prompt: "Why does moving down a light edge halve the subtree size? What is the definition of a heavy child, and what does it imply about all other children?"*

### Template: HLD Construction + Path Queries

```python
import sys
from collections import deque

class HLD:
    """Heavy-Light Decomposition with segment tree for path queries.

    Supports:
        - Point updates on vertices.
        - Path queries (max/sum/etc.) between any two vertices.

    Time: O(N) construction, O(log^2 N) per query/update.
    Space: O(N).
    """

    def __init__(self, n: int, adj: list[list[int]], root: int = 0):
        """Build HLD on tree with n vertices.

        Args:
            n: Number of vertices.
            adj: Adjacency list (undirected tree).
            root: Root vertex.
        """
        self.n = n
        self.root = root
        self.parent = [-1] * n
        self.depth = [0] * n
        self.subtree_size = [1] * n
        self.heavy_child = [-1] * n
        self.chain_head = [0] * n
        self.pos = [0] * n  # Position in flattened array
        self.flat = [0] * n  # Flattened array for segment tree

        self._build(adj)
        self._seg = [0] * (4 * n)  # Segment tree

    def _build(self, adj: list[list[int]]) -> None:
        """Construct HLD in two passes (iterative to avoid stack overflow)."""
        n = self.n
        root = self.root

        # Pass 1: BFS to compute parent, depth, subtree_size, heavy_child
        order = []
        visited = [False] * n
        queue = deque([root])
        visited[root] = True
        while queue:
            v = queue.popleft()
            order.append(v)
            for u in adj[v]:
                if not visited[u]:
                    visited[u] = True
                    self.parent[u] = v
                    self.depth[u] = self.depth[v] + 1
                    queue.append(u)

        # Post-order: compute subtree sizes and heavy children
        for v in reversed(order):
            max_child_size = 0
            for u in adj[v]:
                if self.parent[u] == v:  # u is child of v
                    self.subtree_size[v] += self.subtree_size[u]
                    if self.subtree_size[u] > max_child_size:
                        max_child_size = self.subtree_size[u]
                        self.heavy_child[v] = u

        # Pass 2: Assign chain heads and positions (DFS-like via stack)
        cur_pos = 0
        stack = [(root, root)]  # (vertex, chain_head)
        # We need to process heavy child first for contiguous positions
        while stack:
            v, head = stack.pop()
            self.chain_head[v] = head
            self.pos[v] = cur_pos
            cur_pos += 1

            # Push light children first (processed later), heavy child last (processed next)
            light_children = []
            for u in adj[v]:
                if self.parent[u] == v and u != self.heavy_child[v]:
                    light_children.append(u)

            # Light children start new chains
            for u in light_children:
                stack.append((u, u))

            # Heavy child continues the current chain (pushed last = processed first)
            if self.heavy_child[v] != -1:
                stack.append((self.heavy_child[v], head))

    # --- Segment Tree Operations (max query example) ---

    def _seg_update(self, node: int, lo: int, hi: int, idx: int, val: int) -> None:
        if lo == hi:
            self._seg[node] = val
            return
        mid = (lo + hi) // 2
        if idx <= mid:
            self._seg_update(2 * node, lo, mid, idx, val)
        else:
            self._seg_update(2 * node + 1, mid + 1, hi, idx, val)
        self._seg[node] = max(self._seg[2 * node], self._seg[2 * node + 1])

    def _seg_query(self, node: int, lo: int, hi: int, ql: int, qr: int) -> int:
        if ql > hi or qr < lo:
            return 0  # Identity for max; use float('-inf') if values can be negative
        if ql <= lo and hi <= qr:
            return self._seg[node]
        mid = (lo + hi) // 2
        return max(
            self._seg_query(2 * node, lo, mid, ql, qr),
            self._seg_query(2 * node + 1, mid + 1, hi, ql, qr)
        )

    # --- Public API ---

    def update(self, v: int, val: int) -> None:
        """Set value of vertex v.

        Time: O(log N).
        """
        self._seg_update(1, 0, self.n - 1, self.pos[v], val)

    def query_path(self, u: int, v: int) -> int:
        """Query max value on the path from u to v.

        Climb from both endpoints toward their LCA, querying
        each heavy chain segment along the way.

        Time: O(log^2 N) — O(log N) chain hops x O(log N) segment tree query.
        """
        result = 0  # Identity for max
        while self.chain_head[u] != self.chain_head[v]:
            # Move the deeper chain head upward
            if self.depth[self.chain_head[u]] < self.depth[self.chain_head[v]]:
                u, v = v, u
            # Query from chain_head[u] to u
            result = max(result, self._seg_query(
                1, 0, self.n - 1,
                self.pos[self.chain_head[u]], self.pos[u]
            ))
            u = self.parent[self.chain_head[u]]

        # u and v are now on the same chain
        if self.depth[u] > self.depth[v]:
            u, v = v, u
        result = max(result, self._seg_query(
            1, 0, self.n - 1,
            self.pos[u], self.pos[v]
        ))
        return result

    def query_subtree(self, v: int) -> int:
        """Query max value in subtree of v.

        Time: O(log N) — single segment tree query on contiguous range.
        """
        return self._seg_query(
            1, 0, self.n - 1,
            self.pos[v], self.pos[v] + self.subtree_size[v] - 1
        )
```

*Socratic prompt: "In the query function, why do we always move the endpoint whose chain head is deeper? What invariant does this maintain, and how does it find the LCA?"*

### Usage Example

```python
def hld_example():
    """Demonstrate HLD: assign values to vertices, query max on paths."""
    #       0
    #      / \
    #     1   2
    #    / \
    #   3   4
    n = 5
    adj = [[] for _ in range(n)]
    edges = [(0, 1), (0, 2), (1, 3), (1, 4)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    hld = HLD(n, adj, root=0)

    # Assign values
    values = [10, 20, 5, 15, 25]
    for i, val in enumerate(values):
        hld.update(i, val)

    # Query max on path 3 -> 2: goes through 3 -> 1 -> 0 -> 2
    print(hld.query_path(3, 2))  # max(15, 20, 10, 5) = 20

    # Query max on path 3 -> 4: goes through 3 -> 1 -> 4
    print(hld.query_path(3, 4))  # max(15, 20, 25) = 25
```

### Edge-Based HLD

For edge-weighted trees, store edge weights on the child vertex (each non-root vertex has exactly one parent edge). Exclude the LCA vertex from queries since it doesn't correspond to an edge on the path.

```python
def query_path_edges(hld: HLD, u: int, v: int) -> int:
    """Query max edge weight on path u -> v.

    Same as vertex query but skip the LCA vertex.
    Edge weight is stored on the child endpoint.

    Time: O(log^2 N).
    """
    result = 0
    while hld.chain_head[u] != hld.chain_head[v]:
        if hld.depth[hld.chain_head[u]] < hld.depth[hld.chain_head[v]]:
            u, v = v, u
        result = max(result, hld._seg_query(
            1, 0, hld.n - 1,
            hld.pos[hld.chain_head[u]], hld.pos[u]
        ))
        u = hld.parent[hld.chain_head[u]]

    if hld.depth[u] > hld.depth[v]:
        u, v = v, u
    # Skip LCA (pos[u]): edges start from pos[u]+1
    if hld.pos[u] + 1 <= hld.pos[v]:
        result = max(result, hld._seg_query(
            1, 0, hld.n - 1,
            hld.pos[u] + 1, hld.pos[v]
        ))
    return result
```

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| HLD construction | O(N) | O(N) |
| Path query (with segment tree) | O(log^2 N) | -- |
| Path update (point) | O(log N) | -- |
| Path update (range on chain) | O(log^2 N) | -- |
| Subtree query | O(log N) | -- |

**Optimization:** Using Euler tour + segment tree with lazy propagation can reduce path queries to O(log N) amortized for certain operations.

*Socratic prompt: "Why is the path query O(log^2 N) and not O(log N)? What are the two logarithmic factors, and can either be eliminated?"*

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| SPOJ QTREE (Query on a Tree) | Max edge weight on path, single edge updates (classic HLD) |
| CSES Path Queries II | Max vertex value on path with point updates |
| Codeforces 1254D (Tree Queries) | Expected values on tree paths |
| Codeforces 1017G (The Tree) | Propagation of marks on tree using HLD |
| SPOJ GRASSPLA | Path sum queries on tree |
| CSES Path Queries | Subtree sum queries (HLD or Euler tour) |

---

## Interview Tips

1. **Topological sort is the most interview-relevant topic here.** Know both Kahn's and DFS approaches cold. The others (2-SAT, HLD) are primarily competitive programming topics.

2. **Recognize DAG DP.** Any time you see "directed acyclic graph" and need shortest/longest path or counting, think topological sort + DP.

3. **Re-rooting is underrated.** Problems that say "compute X for every node as root" are a strong signal. The key formula is always: "what changes when we shift the root by one edge?"

4. **Bridge/articulation point detection** (edge connectivity = 1) is the most common connectivity problem in interviews. Full min-cut via max flow is rare.

5. **2-SAT modeling is the hard part.** The solver itself is mechanical (SCC). The skill is translating problem constraints into 2-literal clauses.

6. **HLD is a rare interview topic** but essential for competitive programming tree problems. If you see "path query with updates on a tree," HLD is likely the intended approach.

---

## Attribution

Content synthesized from cp-algorithms.com articles on [Topological Sort](https://cp-algorithms.com/graph/topological-sort.html), [Edge/Vertex Connectivity](https://cp-algorithms.com/graph/edge_vertex_connectivity.html), [Tree Painting](https://cp-algorithms.com/graph/tree_painting.html), [2-SAT](https://cp-algorithms.com/graph/2SAT.html), and [Heavy-Light Decomposition](https://cp-algorithms.com/graph/hld.html). Algorithms and complexity analyses are from the original articles; code has been translated to Python with added commentary for the leetcode-teacher reference format.
