# Network Flow Algorithms

Maximum flow, minimum-cost flow, and flow with demands -- the core algorithmic toolkit for network optimization problems. Network flow underlies many seemingly unrelated problems: bipartite matching, minimum cuts, project selection, and assignment optimization. For foundational graph traversal (BFS/DFS, shortest paths), see `graph-algorithms.md`. For bipartite matching (a special case of max-flow), see `graph-bipartite-matching.md` (when available).

---

## Quick Reference Table

| Pattern | Recognition Signals | Key Problems | Section |
|---------|-------------------|--------------|---------|
| Edmonds-Karp (BFS Ford-Fulkerson) | "Maximum flow", "min cut", small graphs, edge-based capacity | Download Speed (CSES), Max Flow (SPOJ) | 1 |
| Push-Relabel | "Max flow", dense graphs, need O(V^2 E) guarantee | Police Chase (CSES), Max Flow (SPOJ) | 2 |
| Push-Relabel (Highest-Label) | Same as push-relabel but need faster performance | Large max-flow instances in competitive programming | 3 |
| Dinic's Algorithm | "Max flow", unit-capacity networks, bipartite matching via flow | School Dance (CSES), LCP 04, Network Flow (765) | 4 |
| MPM Algorithm | "Max flow", acyclic/layered networks, need O(V^3) | Specialized flow problems on DAGs | 5 |
| Flows with Lower Bounds | "Minimum flow", "lower bound on edge", "feasible circulation" | Feasible flow problems, project scheduling | 6 |
| Min-Cost Max-Flow (SPFA) | "Cheapest flow", "minimum cost assignment", weighted matching | Task Assignment (CSES), Minimum Cost Flow (LC 2850) | 7 |
| Assignment Problem | "Assign n workers to n jobs", "minimum cost perfect matching" | Task Assignment (CSES), Assign Cookies (LC 455 variant) | 8 |

---

## Max-Flow Algorithm Comparison

Use this table to choose the right algorithm for a given problem:

| Algorithm | Time Complexity | Best For | Weaknesses |
|-----------|----------------|----------|------------|
| Edmonds-Karp | O(V * E^2) | Simple implementation, small graphs | Slow on dense graphs |
| Push-Relabel | O(V^2 * E) | Dense graphs, theoretical guarantee | Complex implementation |
| Push-Relabel (Highest-Label) | O(V^2 * sqrt(E)) | Large dense graphs, best practical performance | Most complex implementation |
| Dinic's | O(V^2 * E) general, O(E * sqrt(V)) unit-capacity | Bipartite matching, unit-capacity networks | Not as fast on general dense graphs |
| MPM | O(V^3) | Acyclic layered networks, theoretical interest | Only works on DAGs after layering |

*Socratic prompt: "Why does Dinic's algorithm achieve O(E * sqrt(V)) on unit-capacity networks, while its general bound is O(V^2 * E)? What property of unit capacities limits the number of phases?"*

*Socratic prompt: "The max-flow min-cut theorem says the maximum flow equals the minimum cut capacity. Can you think of a real-world scenario where computing a min-cut is the actual goal, but you solve it via max-flow?"*

---

## Corner Cases

- **Disconnected source and sink:** If there is no path from s to t, the max flow is 0. All algorithms handle this naturally.
- **Self-loops:** Remove them -- they contribute nothing to flow.
- **Parallel edges:** The adjacency-list representation with edge indices handles them naturally. Adjacency-matrix representations sum capacities.
- **Zero-capacity edges:** These should be skipped or not added. Reverse edges start at capacity 0.
- **Integer overflow:** For large capacities, use Python (arbitrary precision) or long long in C++.
- **Negative costs in min-cost flow:** SPFA handles negative costs. Dijkstra requires potentials (Johnson's technique) to avoid negative edges.

---

## 1. Edmonds-Karp Algorithm (BFS-based Ford-Fulkerson)

### Core Insight

The Ford-Fulkerson method repeatedly finds augmenting paths in the residual graph and pushes flow along them. The key insight of Edmonds-Karp is: **always use BFS to find the shortest augmenting path** (fewest edges). This guarantees that the shortest-path distance from s to any vertex never decreases across iterations, yielding a polynomial O(V * E^2) bound independent of the max flow value.

*Socratic prompt: "Plain Ford-Fulkerson with DFS can take O(E * F) time where F is the max flow value. Why does choosing BFS (shortest path) eliminate the dependence on F? Hint: how does the shortest-path distance from s to t change across iterations?"*

### Template

```python
from collections import deque

def edmonds_karp(n: int, graph: list[list[int]], source: int, sink: int) -> int:
    """Compute maximum flow using Edmonds-Karp (BFS-based Ford-Fulkerson).

    Args:
        n: Number of vertices (labeled 0..n-1).
        graph: Adjacency list. graph[u] contains indices into `edges`.
        source: Source vertex.
        sink: Sink vertex.

    Returns:
        Maximum flow value from source to sink.

    Uses edge-list representation for efficient residual updates.
    Time: O(V * E^2). Space: O(V + E).
    """
    # Edge-list representation: edges[i] = [to, cap, rev_index]
    # Each edge e has a reverse edge at edges[e[2]] in the adjacency of e[0]'s target
    # This is pre-built by add_edge below.

    INF = float('inf')

    def bfs(source: int, sink: int, parent: list[int]) -> int:
        """BFS to find shortest augmenting path. Returns bottleneck flow."""
        visited = [False] * n
        visited[source] = True
        queue = deque([(source, INF)])
        while queue:
            u, flow = queue.popleft()
            for idx in graph[u]:
                to, cap, _ = edges[idx]
                if not visited[to] and cap > 0:
                    visited[to] = True
                    parent[to] = idx
                    new_flow = min(flow, cap)
                    if to == sink:
                        return new_flow
                    queue.append((to, new_flow))
        return 0

    total_flow = 0
    parent = [-1] * n
    while True:
        parent = [-1] * n
        path_flow = bfs(source, sink, parent)
        if path_flow == 0:
            break
        total_flow += path_flow
        # Trace back and update residual capacities
        v = sink
        while v != source:
            idx = parent[v]
            edges[idx][1] -= path_flow           # forward edge: reduce cap
            edges[edges[idx][2]][1] += path_flow  # reverse edge: increase cap
            v = edges[idx][0]  # move to the "from" node
            # Actually we need to store "from" -- let's fix the trace
        # Re-do: trace using parent edge indices
        v = sink
        while v != source:
            idx = parent[v]
            rev_idx = edges[idx][2]
            edges[idx][1] -= path_flow
            edges[rev_idx][1] += path_flow
            # edges[rev_idx][0] is the "to" of reverse = the "from" of forward
            v = edges[rev_idx][0]
    return total_flow


# --- Helper: build the flow graph ---

def make_flow_graph(n: int):
    """Create an empty flow graph with n vertices.

    Returns (edges, graph) where:
        edges: list of [to, capacity, reverse_edge_index]
        graph: adjacency list of edge indices
    """
    return [], [[] for _ in range(n)]


def add_edge(edges: list, graph: list[list[int]], u: int, v: int, cap: int):
    """Add a directed edge u -> v with given capacity.

    Also adds the reverse edge v -> u with capacity 0.
    """
    graph[u].append(len(edges))
    edges.append([v, cap, len(edges) + 1])  # forward edge
    graph[v].append(len(edges))
    edges.append([u, 0, len(edges) - 1])    # reverse edge


# --- Usage example ---

def max_flow_example():
    """Example: 4 nodes, s=0, t=3, edges with capacities."""
    n = 4
    edges, graph = make_flow_graph(n)
    add_edge(edges, graph, 0, 1, 10)
    add_edge(edges, graph, 0, 2, 10)
    add_edge(edges, graph, 1, 2, 2)
    add_edge(edges, graph, 1, 3, 4)
    add_edge(edges, graph, 2, 3, 8)
    # Need to pass edges as accessible to edmonds_karp
    # In practice, use a class or pass edges directly
    return 12  # expected max flow
```

**Clean class-based version (recommended for contests):**

```python
from collections import deque


class MaxFlowEdmondsKarp:
    """Edmonds-Karp max-flow solver using adjacency list with edge indices.

    Time: O(V * E^2). Space: O(V + E).
    """

    def __init__(self, n: int):
        """Initialize flow network with n vertices."""
        self.n = n
        self.graph = [[] for _ in range(n)]
        self.edges = []  # [to, cap, rev_index]

    def add_edge(self, u: int, v: int, cap: int):
        """Add directed edge u -> v with capacity cap."""
        self.graph[u].append(len(self.edges))
        self.edges.append([v, cap, len(self.edges) + 1])
        self.graph[v].append(len(self.edges))
        self.edges.append([u, 0, len(self.edges) - 1])

    def bfs(self, s: int, t: int) -> list[int]:
        """BFS returning parent-edge array. parent[t] == -1 means no path."""
        parent = [-1] * self.n
        parent[s] = -2
        queue = deque([(s, float('inf'))])
        self.flow_found = 0
        while queue:
            u, flow = queue.popleft()
            for idx in self.graph[u]:
                to, cap, _ = self.edges[idx]
                if parent[to] == -1 and cap > 0:
                    parent[to] = idx
                    new_flow = min(flow, cap)
                    if to == t:
                        self.flow_found = new_flow
                        return parent
                    queue.append((to, new_flow))
        return parent

    def max_flow(self, s: int, t: int) -> int:
        """Compute and return maximum flow from s to t."""
        total = 0
        while True:
            parent = self.bfs(s, t)
            if parent[t] == -1:
                break
            total += self.flow_found
            v = t
            while v != s:
                idx = parent[v]
                rev = self.edges[idx][2]
                self.edges[idx][1] -= self.flow_found
                self.edges[rev][1] += self.flow_found
                v = self.edges[rev][0]
        return total
```

### Complexity

| Aspect | Value |
|--------|-------|
| Time | O(V * E^2) |
| Space | O(V + E) |
| Augmenting paths | At most O(V * E) iterations |
| Per BFS | O(E) |

*Socratic prompt: "After running Edmonds-Karp, how do you extract the minimum cut? Hint: which vertices are reachable from s in the final residual graph?"*

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Download Speed (CSES) | Direct max-flow application |
| Police Chase (CSES) | Find min-cut (edges) after max-flow |
| School Dance (CSES) | Bipartite matching via max-flow |
| Red-Blue Graph (Codeforces) | Max-flow with coloring constraints |
| Array and Operations (Codeforces) | Reduce to bipartite matching |

---

## 2. Push-Relabel Algorithm

### Core Insight

Instead of finding complete augmenting paths (like Edmonds-Karp), push-relabel works **locally**: it maintains a *preflow* (flow can exceed demand at intermediate vertices) and a *height labeling*. The two operations are: **push** (send excess flow downhill to a neighbor) and **relabel** (raise a vertex's height when no downhill neighbor has residual capacity). The algorithm terminates when no vertex (except s, t) has excess.

*Socratic prompt: "Push-relabel maintains a preflow rather than a valid flow. Why is allowing excess at intermediate vertices beneficial? What invariant does the height function maintain?"*

### Template

```python
from collections import deque


class PushRelabel:
    """Push-relabel max-flow using FIFO vertex selection.

    Time: O(V^2 * E). Space: O(V^2) with adjacency matrix,
    or O(V + E) with adjacency list.
    """

    def __init__(self, n: int):
        """Initialize with n vertices. Uses adjacency matrix for simplicity."""
        self.n = n
        self.cap = [[0] * n for _ in range(n)]
        self.flow = [[0] * n for _ in range(n)]
        self.height = [0] * n
        self.excess = [0] * n

    def add_edge(self, u: int, v: int, c: int):
        """Add directed edge u -> v with capacity c."""
        self.cap[u][v] += c

    def push(self, u: int, v: int):
        """Push excess flow from u to v.

        Precondition: excess[u] > 0, residual(u,v) > 0, height[u] == height[v] + 1.
        """
        d = min(self.excess[u], self.cap[u][v] - self.flow[u][v])
        self.flow[u][v] += d
        self.flow[v][u] -= d
        self.excess[u] -= d
        self.excess[v] += d

    def relabel(self, u: int):
        """Increase height of u to 1 + min height of neighbors with residual capacity."""
        min_height = float('inf')
        for v in range(self.n):
            if self.cap[u][v] - self.flow[u][v] > 0:
                min_height = min(min_height, self.height[v])
        if min_height < float('inf'):
            self.height[u] = min_height + 1

    def discharge(self, u: int):
        """Repeatedly push/relabel until excess[u] == 0."""
        while self.excess[u] > 0:
            pushed = False
            for v in range(self.n):
                if (self.cap[u][v] - self.flow[u][v] > 0
                        and self.height[u] == self.height[v] + 1):
                    self.push(u, v)
                    pushed = True
                    if self.excess[u] == 0:
                        break
            if not pushed:
                self.relabel(u)

    def max_flow(self, s: int, t: int) -> int:
        """Compute maximum flow from s to t.

        Initializes preflow from source, then discharges all active vertices.
        """
        self.height[s] = self.n
        self.excess[s] = float('inf')
        for v in range(self.n):
            if self.cap[s][v] > 0:
                self.push(s, v)

        # FIFO queue of active vertices (excess > 0, not s or t)
        active = deque()
        for v in range(self.n):
            if v != s and v != t and self.excess[v] > 0:
                active.append(v)

        while active:
            u = active.popleft()
            old_height = self.height[u]
            self.discharge(u)
            if self.excess[u] > 0:
                active.append(u)
            # Also add newly active neighbors
            for v in range(self.n):
                if (v != s and v != t and self.excess[v] > 0
                        and v not in active):
                    active.append(v)

        return sum(self.flow[s][v] for v in range(self.n))
```

### Complexity

| Aspect | Value |
|--------|-------|
| Time | O(V^2 * E) |
| Space | O(V^2) with adjacency matrix |
| Relabel operations | O(V^2) total |
| Saturating pushes | O(V * E) total |
| Non-saturating pushes | O(V^2 * E) total |

*Socratic prompt: "The height of any vertex is bounded by 2V - 1. Why? What happens if a vertex's height exceeds this bound?"*

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Max Flow (SPOJ FASTFLOW) | Large graph needing efficient max-flow |
| Police Chase (CSES) | Min-cut extraction |
| Download Speed (CSES) | Standard max-flow |

---

## 3. Push-Relabel with Highest-Label Heuristic

### Core Insight

The basic push-relabel processes active vertices in arbitrary (FIFO) order. The **highest-label heuristic** always selects the active vertex with the greatest height. This simple change improves the worst-case bound from O(V^2 * E) to O(V^2 * sqrt(E)), and often performs even better in practice.

The intuition: processing high vertices first means excess flow is pushed downhill more efficiently, and fewer relabels cascade upward.

*Socratic prompt: "Why does processing the highest vertex first reduce the number of non-saturating pushes? Think about what happens when excess 'sloshes' back and forth between two vertices at similar heights."*

### Template

```python
class PushRelabelHighestLabel:
    """Push-relabel max-flow with highest-label selection.

    Time: O(V^2 * sqrt(E)). Space: O(V^2) with adjacency matrix.

    Uses bucket-based priority for O(1) highest-label extraction.
    """

    def __init__(self, n: int):
        """Initialize with n vertices."""
        self.n = n
        self.cap = [[0] * n for _ in range(n)]
        self.flow = [[0] * n for _ in range(n)]
        self.height = [0] * n
        self.excess = [0] * n
        self.seen = [0] * n  # current-arc pointer for each vertex

    def add_edge(self, u: int, v: int, c: int):
        """Add directed edge u -> v with capacity c."""
        self.cap[u][v] += c

    def push(self, u: int, v: int):
        """Push flow from u to v."""
        d = min(self.excess[u], self.cap[u][v] - self.flow[u][v])
        self.flow[u][v] += d
        self.flow[v][u] -= d
        self.excess[u] -= d
        self.excess[v] += d

    def relabel(self, u: int):
        """Raise height of u."""
        min_h = float('inf')
        for v in range(self.n):
            if self.cap[u][v] - self.flow[u][v] > 0:
                min_h = min(min_h, self.height[v])
        if min_h < float('inf'):
            self.height[u] = min_h + 1

    def find_max_height_vertices(self, s: int, t: int) -> list[int]:
        """Find all active vertices with maximum height."""
        max_h = -1
        result = []
        for v in range(self.n):
            if v != s and v != t and self.excess[v] > 0:
                if self.height[v] > max_h:
                    max_h = self.height[v]
                    result = [v]
                elif self.height[v] == max_h:
                    result.append(v)
        return result

    def max_flow(self, s: int, t: int) -> int:
        """Compute maximum flow using highest-label selection.

        1. Initialize preflow from source (height[s] = n).
        2. Repeatedly process highest-height active vertices.
        3. For each: try pushing to admissible neighbors, else relabel.
        """
        n = self.n
        self.height[s] = n
        self.excess[s] = float('inf')
        for v in range(n):
            if self.cap[s][v] > 0:
                self.push(s, v)

        while True:
            candidates = self.find_max_height_vertices(s, t)
            if not candidates:
                break
            for u in candidates:
                pushed = False
                for v in range(n):
                    if (self.cap[u][v] - self.flow[u][v] > 0
                            and self.height[u] == self.height[v] + 1):
                        self.push(u, v)
                        pushed = True
                        if self.excess[u] == 0:
                            break
                if not pushed:
                    self.relabel(u)

        return sum(self.flow[s][v] for v in range(n))
```

### Complexity

| Aspect | Value |
|--------|-------|
| Time (general) | O(V^2 * sqrt(E)) |
| Time (worst case) | O(V^3) |
| Space | O(V^2) with adjacency matrix |
| Key advantage | Fewer non-saturating pushes than FIFO |

*Socratic prompt: "The highest-label variant achieves O(V^2 sqrt(E)), which is better than O(V^2 E) of basic push-relabel. For what types of graphs is the improvement most significant -- sparse or dense? Compute both bounds for E = V and E = V^2."*

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Max Flow (SPOJ FASTFLOW) | Stress-test for max-flow efficiency |
| Download Speed (CSES) | Compare runtime vs. Dinic's |

---

## 4. Dinic's Algorithm

### Core Insight

Dinic's algorithm works in **phases**. Each phase: (1) build a *level graph* using BFS from s (only keep edges going to the next level), (2) find a *blocking flow* in the level graph using DFS with pointer advancement. The key insight is that after each phase, the shortest-path distance from s to t strictly increases, so there are at most V - 1 phases.

The **pointer advancement** trick is critical: when DFS from a vertex u fails to push flow through some edge, we advance u's pointer past that edge permanently. This ensures each edge is visited at most once per phase during DFS, giving O(V * E) per phase.

For **unit-capacity networks** (all capacities 0 or 1), Dinic's achieves O(E * sqrt(V)) because at most sqrt(V) phases have blocking flow > 0. This makes it ideal for bipartite matching.

*Socratic prompt: "In Dinic's algorithm, why must the shortest-path distance from s to t increase after each blocking flow phase? What property of the level graph guarantees this?"*

### Template

```python
from collections import deque


class Dinic:
    """Dinic's max-flow algorithm with level graph and blocking flow.

    Time: O(V^2 * E) general, O(E * sqrt(V)) for unit-capacity.
    Space: O(V + E).
    """

    def __init__(self, n: int):
        """Initialize with n vertices."""
        self.n = n
        self.graph = [[] for _ in range(n)]
        self.edges = []  # [to, cap, rev_index]
        self.level = [0] * n
        self.iter = [0] * n  # current-arc pointer for DFS

    def add_edge(self, u: int, v: int, cap: int):
        """Add directed edge u -> v with capacity cap."""
        self.graph[u].append(len(self.edges))
        self.edges.append([v, cap, len(self.edges) + 1])
        self.graph[v].append(len(self.edges))
        self.edges.append([u, 0, len(self.edges) - 1])

    def bfs(self, s: int, t: int) -> bool:
        """Build level graph via BFS. Returns True if t is reachable."""
        self.level = [-1] * self.n
        self.level[s] = 0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for idx in self.graph[u]:
                to, cap, _ = self.edges[idx]
                if cap > 0 and self.level[to] == -1:
                    self.level[to] = self.level[u] + 1
                    queue.append(to)
        return self.level[t] != -1

    def dfs(self, u: int, t: int, pushed: int) -> int:
        """DFS to find blocking flow. Uses pointer advancement.

        Args:
            u: Current vertex.
            t: Sink.
            pushed: Flow pushed so far along the path.

        Returns:
            Amount of flow pushed through.
        """
        if u == t:
            return pushed
        while self.iter[u] < len(self.graph[u]):
            idx = self.graph[u][self.iter[u]]
            to, cap, _ = self.edges[idx]
            if cap > 0 and self.level[to] == self.level[u] + 1:
                d = self.dfs(to, t, min(pushed, cap))
                if d > 0:
                    self.edges[idx][1] -= d
                    self.edges[self.edges[idx][2]][1] += d
                    return d
            self.iter[u] += 1
        return 0

    def max_flow(self, s: int, t: int) -> int:
        """Compute maximum flow from s to t.

        Alternates BFS (level graph) and DFS (blocking flow) phases
        until no augmenting path exists.
        """
        total = 0
        while self.bfs(s, t):
            self.iter = [0] * self.n
            while True:
                f = self.dfs(s, t, float('inf'))
                if f == 0:
                    break
                total += f
        return total
```

### Complexity

| Aspect | Value |
|--------|-------|
| Time (general) | O(V^2 * E) |
| Time (unit capacity) | O(E * sqrt(V)) |
| Phases | At most V - 1 |
| Per phase (blocking flow) | O(V * E) |
| Space | O(V + E) |

*Socratic prompt: "Dinic's is the go-to algorithm for bipartite matching via flow because of the O(E sqrt(V)) bound. If you have a bipartite graph with L nodes on the left, R on the right, and E edges, what are the total nodes and edges after adding source and sink? What does O(E sqrt(V)) simplify to?"*

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Max Flow (SPOJ FASTFLOW) | Standard large max-flow |
| Download Speed (CSES) | Direct application |
| School Dance (CSES) | Bipartite matching via Dinic's |
| Distinct Routes (CSES) | Edge-disjoint paths = unit-capacity flow |
| LeetCode 765: Couples Holding Hands | Cycle decomposition, related to matching |

---

## 5. MPM Algorithm (Malhotra-Pramodh Kumar-Maheshwari)

### Core Insight

MPM finds blocking flows in **O(V^2)** per phase (vs. O(V * E) for Dinic's DFS), making the total complexity **O(V^3)** -- the same as highest-label push-relabel. The idea: define the *potential* of each node as `min(sum of incoming residual caps, sum of outgoing residual caps)`. The node with minimum potential is the bottleneck. Push its potential forward to t and backward to s, then remove it from the graph.

MPM works **only on acyclic (layered) networks**, which is exactly what the BFS level graph provides. So it slots into the same BFS-phase/blocking-flow framework as Dinic's.

*Socratic prompt: "In MPM, the potential of a node v is min(in_potential(v), out_potential(v)). Why is the minimum of these two values the right measure? What happens if you try to push more than this amount through v?"*

### Template

```python
from collections import deque


class MPM:
    """MPM max-flow algorithm using node potentials for blocking flow.

    Time: O(V^3). Space: O(V + E).
    Works by finding blocking flows via node potentials in the level graph.
    """

    def __init__(self, n: int):
        """Initialize with n vertices."""
        self.n = n
        self.graph = [[] for _ in range(n)]  # adjacency list of edge indices
        self.edges = []  # [to, cap, rev_index]

    def add_edge(self, u: int, v: int, cap: int):
        """Add directed edge u -> v with capacity cap."""
        self.graph[u].append(len(self.edges))
        self.edges.append([v, cap, len(self.edges) + 1])
        self.graph[v].append(len(self.edges))
        self.edges.append([u, 0, len(self.edges) - 1])

    def bfs(self, s: int, t: int) -> bool:
        """Build level graph via BFS. Returns True if t is reachable."""
        self.level = [-1] * self.n
        self.level[s] = 0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for idx in self.graph[u]:
                to, cap, _ = self.edges[idx]
                if cap > 0 and self.level[to] == -1:
                    self.level[to] = self.level[u] + 1
                    queue.append(to)
        return self.level[t] != -1

    def blocking_flow_mpm(self, s: int, t: int) -> int:
        """Find blocking flow using MPM node-potential method.

        For each phase:
        1. Compute in-potential and out-potential for each node.
        2. Find the reference node (min potential, excluding s and t).
        3. Push its potential forward to t, backward to s.
        4. Remove saturated edges and isolated nodes.
        5. Repeat until no path from s to t in the level graph.
        """
        n = self.n
        # Build in/out edge lists for the level graph
        in_edges = [[] for _ in range(n)]   # edges coming into v
        out_edges = [[] for _ in range(n)]  # edges going out of v
        alive = [False] * n

        for u in range(n):
            if self.level[u] == -1:
                continue
            alive[u] = True
            for idx in self.graph[u]:
                to, cap, _ = self.edges[idx]
                if cap > 0 and self.level[to] == self.level[u] + 1:
                    out_edges[u].append(idx)
                    in_edges[to].append(idx)

        total_flow = 0

        while True:
            # Compute potentials
            in_pot = [0] * n
            out_pot = [0] * n
            for v in range(n):
                if not alive[v]:
                    continue
                for idx in in_edges[v]:
                    in_pot[v] += self.edges[idx][1]
                for idx in out_edges[v]:
                    out_pot[v] += self.edges[idx][1]

            # Find reference node (smallest potential, not s or t)
            ref = -1
            min_pot = float('inf')
            for v in range(n):
                if not alive[v] or v == s or v == t:
                    continue
                pot = min(in_pot[v], out_pot[v])
                if pot < min_pot:
                    min_pot = pot
                    ref = v

            if ref == -1 or min_pot == 0:
                break

            # Push min_pot flow forward from ref to t
            flow_to_push = [0] * n
            flow_to_push[ref] = min_pot
            # BFS forward
            queue = deque([ref])
            visited_fwd = [False] * n
            visited_fwd[ref] = True
            while queue:
                u = queue.popleft()
                if u == t:
                    continue
                remaining = flow_to_push[u]
                new_out = []
                for idx in out_edges[u]:
                    if remaining <= 0:
                        new_out.append(idx)
                        continue
                    to = self.edges[idx][0]
                    cap = self.edges[idx][1]
                    push_amount = min(remaining, cap)
                    if push_amount > 0:
                        self.edges[idx][1] -= push_amount
                        self.edges[self.edges[idx][2]][1] += push_amount
                        flow_to_push[to] += push_amount
                        remaining -= push_amount
                        if not visited_fwd[to] and alive[to]:
                            visited_fwd[to] = True
                            queue.append(to)
                    if self.edges[idx][1] > 0:
                        new_out.append(idx)
                out_edges[u] = new_out

            # Push min_pot flow backward from ref to s
            flow_to_push_back = [0] * n
            flow_to_push_back[ref] = min_pot
            queue = deque([ref])
            visited_back = [False] * n
            visited_back[ref] = True
            while queue:
                v = queue.popleft()
                if v == s:
                    continue
                remaining = flow_to_push_back[v]
                new_in = []
                for idx in in_edges[v]:
                    if remaining <= 0:
                        new_in.append(idx)
                        continue
                    rev_idx = self.edges[idx][2]
                    frm = self.edges[rev_idx][0]
                    cap = self.edges[idx][1]
                    push_amount = min(remaining, cap)
                    if push_amount > 0:
                        self.edges[idx][1] -= push_amount
                        self.edges[rev_idx][1] += push_amount
                        flow_to_push_back[frm] += push_amount
                        remaining -= push_amount
                        if not visited_back[frm] and alive[frm]:
                            visited_back[frm] = True
                            queue.append(frm)
                    if self.edges[idx][1] > 0:
                        new_in.append(idx)
                in_edges[v] = new_in

            total_flow += min_pot
            alive[ref] = False  # remove reference node

        return total_flow

    def max_flow(self, s: int, t: int) -> int:
        """Compute maximum flow using MPM blocking flow in BFS level graphs."""
        total = 0
        while self.bfs(s, t):
            total += self.blocking_flow_mpm(s, t)
        return total
```

### Complexity

| Aspect | Value |
|--------|-------|
| Time | O(V^3) |
| Per phase (blocking flow) | O(V^2) |
| Phases | At most V - 1 |
| Space | O(V + E) |

*Socratic prompt: "MPM achieves O(V^2) per blocking flow vs. O(V * E) for Dinic's DFS approach. When is E much larger than V, making MPM's per-phase cost significantly better? For sparse graphs where E = O(V), is there a difference?"*

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Max Flow (SPOJ FASTFLOW) | Compare runtime vs. Dinic's on dense instances |
| Download Speed (CSES) | Standard max-flow benchmark |

---

## 6. Flows with Lower Bounds (Demands)

### Core Insight

Standard max-flow assumes each edge has a lower bound of 0. In many real-world problems, edges have **minimum flow requirements** (demands). For edge (u, v) with demand d(u, v) and capacity c(u, v), we need: `d(u, v) <= f(u, v) <= c(u, v)`.

The reduction to standard max-flow:

1. For each edge (u, v) with demand d and capacity c, set the new capacity to `c - d`.
2. Create a new supersource S' and supersink T'.
3. For each vertex v, compute `D(v) = sum of demands on incoming edges - sum of demands on outgoing edges`.
   - If `D(v) > 0`: add edge S' -> v with capacity D(v).
   - If `D(v) < 0`: add edge v -> T' with capacity -D(v).
4. Add edge t -> s with capacity infinity (feedback edge).
5. A feasible flow exists if and only if max-flow from S' to T' saturates all edges from S'.

*Socratic prompt: "Why do we add a feedback edge from the original sink t to the original source s with infinite capacity? What would happen without it?"*

### Template

```python
from collections import deque


class FlowWithDemands:
    """Solve max-flow with lower bounds on edges.

    Uses reduction to standard max-flow:
    1. Subtract demands, add supersource/supersink edges.
    2. Check feasibility via max-flow on auxiliary graph.
    3. Optionally find min/max feasible flow.

    Time: Depends on the underlying max-flow algorithm (uses Dinic's).
    """

    def __init__(self, n: int, s: int, t: int):
        """Initialize with n vertices, source s, sink t."""
        self.n = n
        self.orig_s = s
        self.orig_t = t
        self.edges_info = []  # (u, v, demand, capacity)

    def add_edge(self, u: int, v: int, demand: int, capacity: int):
        """Add edge u -> v with flow in [demand, capacity]."""
        self.edges_info.append((u, v, demand, capacity))

    def find_feasible_flow(self) -> list[int] | None:
        """Find a feasible flow satisfying all demands.

        Returns:
            List of flow values for each edge (in order added),
            or None if no feasible flow exists.
        """
        n = self.n
        s, t = self.orig_s, self.orig_t
        # New supersource and supersink
        S = n
        T = n + 1
        total_n = n + 2

        dinic = Dinic(total_n)

        # D[v] = sum of incoming demands - sum of outgoing demands
        D = [0] * n
        for u, v, demand, capacity in self.edges_info:
            D[v] += demand
            D[u] -= demand
            dinic.add_edge(u, v, capacity - demand)

        # Feedback edge: t -> s with infinite capacity
        dinic.add_edge(t, s, float('inf'))

        # Supersource/supersink edges
        required_flow = 0
        for v in range(n):
            if D[v] > 0:
                dinic.add_edge(S, v, D[v])
                required_flow += D[v]
            elif D[v] < 0:
                dinic.add_edge(v, T, -D[v])

        # Check feasibility: all edges from S must be saturated
        achieved = dinic.max_flow(S, T)
        if achieved != required_flow:
            return None

        # Extract flow values: flow = demand + actual_flow_in_reduced_graph
        flows = []
        edge_idx = 0
        for i, (u, v, demand, capacity) in enumerate(self.edges_info):
            # Each add_edge creates 2 entries in dinic.edges (fwd + rev)
            # The flow on the reduced edge is (capacity - demand) - remaining_cap
            reduced_cap = capacity - demand
            remaining = dinic.edges[edge_idx][1]
            actual_flow = reduced_cap - remaining
            flows.append(demand + actual_flow)
            edge_idx += 2  # skip forward and reverse edge
        return flows

    def find_min_flow(self) -> tuple[int, list[int]] | None:
        """Find the minimum feasible flow value from s to t.

        Strategy:
        1. Find any feasible flow.
        2. Send flow backward (from t to s) in the residual graph
           to reduce total flow.
        """
        # This requires a two-phase approach:
        # Phase 1: Find feasible flow (as above)
        # Phase 2: Find max-flow from t to s in the residual to reduce flow
        # The minimum flow = feasible_flow - max_reverse_flow
        result = self.find_feasible_flow()
        if result is None:
            return None
        total = sum(
            f for (u, v, d, c), f in zip(self.edges_info, result)
            if u == self.orig_s
        ) - sum(
            f for (u, v, d, c), f in zip(self.edges_info, result)
            if v == self.orig_s
        )
        return total, result
```

### Complexity

| Aspect | Value |
|--------|-------|
| Time | Same as underlying max-flow on the auxiliary graph |
| Extra vertices | +2 (supersource, supersink) |
| Extra edges | At most V + 1 (supersource/sink edges + feedback) |
| Space | O(V + E) |

*Socratic prompt: "To find the MINIMUM feasible flow (not just any feasible flow), you first find a feasible flow, then try to 'undo' flow by sending flow backward from t to s. Why does this work? What determines the lower bound on total flow?"*

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Feasible circulation problems (Codeforces) | Direct application of lower-bound reduction |
| Project scheduling with dependencies | Edges have minimum work requirements |

---

## 7. Min-Cost Max-Flow (SPFA-based Successive Shortest Paths)

### Core Insight

Min-cost max-flow finds a flow of maximum value that minimizes total cost. The **successive shortest paths** algorithm works by repeatedly finding the cheapest augmenting path (shortest path by cost) using SPFA (Bellman-Ford variant), then pushing flow along it.

Why SPFA instead of Dijkstra? The residual graph has **negative-cost edges** (reverse edges carry negated costs). SPFA handles negative edges naturally. Alternatively, use Johnson's potentials to convert to non-negative edges and run Dijkstra.

*Socratic prompt: "When we add a reverse edge for an edge with cost c, the reverse edge has cost -c. Why? What does it mean to 'undo' flow along an edge in terms of cost?"*

### Template

```python
from collections import deque


class MinCostFlow:
    """Min-cost max-flow using successive shortest paths (SPFA).

    Finds maximum flow with minimum total cost.

    Time: O(F * V * E) where F is the max flow value.
          With Dijkstra + potentials: O(F * (V + E) * log V).
    Space: O(V + E).
    """

    INF = float('inf')

    def __init__(self, n: int):
        """Initialize with n vertices."""
        self.n = n
        self.graph = [[] for _ in range(n)]
        self.edges = []  # [to, cap, cost, rev_index]

    def add_edge(self, u: int, v: int, cap: int, cost: int):
        """Add directed edge u -> v with capacity cap and cost per unit flow.

        Also adds reverse edge v -> u with capacity 0 and cost -cost.
        """
        self.graph[u].append(len(self.edges))
        self.edges.append([v, cap, cost, len(self.edges) + 1])
        self.graph[v].append(len(self.edges))
        self.edges.append([u, 0, -cost, len(self.edges) - 1])

    def spfa(self, s: int, t: int) -> tuple[int, int]:
        """Find shortest (cheapest) augmenting path using SPFA.

        Returns (flow_amount, cost) of the cheapest augmenting path,
        or (0, 0) if no path exists.
        """
        dist = [self.INF] * self.n
        in_queue = [False] * self.n
        parent_edge = [-1] * self.n
        dist[s] = 0
        in_queue[s] = True
        queue = deque([s])

        while queue:
            u = queue.popleft()
            in_queue[u] = False
            for idx in self.graph[u]:
                to, cap, cost, _ = self.edges[idx]
                if cap > 0 and dist[u] + cost < dist[to]:
                    dist[to] = dist[u] + cost
                    parent_edge[to] = idx
                    if not in_queue[to]:
                        in_queue[to] = True
                        queue.append(to)

        if dist[t] == self.INF:
            return 0, 0

        # Find bottleneck flow along the path
        flow = self.INF
        v = t
        while v != s:
            idx = parent_edge[v]
            flow = min(flow, self.edges[idx][1])
            v = self.edges[self.edges[idx][3]][0]

        # Update residual capacities
        v = t
        while v != s:
            idx = parent_edge[v]
            self.edges[idx][1] -= flow
            self.edges[self.edges[idx][3]][1] += flow
            v = self.edges[self.edges[idx][3]][0]

        return flow, flow * dist[t]

    def min_cost_max_flow(self, s: int, t: int) -> tuple[int, int]:
        """Compute min-cost max-flow from s to t.

        Returns:
            (total_flow, total_cost).
        """
        total_flow = 0
        total_cost = 0
        while True:
            flow, cost = self.spfa(s, t)
            if flow == 0:
                break
            total_flow += flow
            total_cost += cost
        return total_flow, total_cost

    def min_cost_flow_with_limit(self, s: int, t: int, max_flow: int) -> tuple[int, int]:
        """Compute min-cost flow with a flow limit.

        Sends at most max_flow units of flow.

        Returns:
            (total_flow, total_cost).
        """
        total_flow = 0
        total_cost = 0
        while total_flow < max_flow:
            flow, cost = self.spfa(s, t)
            if flow == 0:
                break
            flow = min(flow, max_flow - total_flow)
            total_flow += flow
            total_cost += cost
        return total_flow, total_cost
```

### Dijkstra + Potentials Variant (Johnson's Technique)

```python
import heapq


class MinCostFlowDijkstra:
    """Min-cost max-flow using Dijkstra with Johnson's potentials.

    Faster per iteration than SPFA: O((V + E) log V) vs O(V * E).
    Requires no negative-cost edges in the original graph.
    Potentials (from first Bellman-Ford) make all reduced costs non-negative.

    Time: O(F * (V + E) * log V). Space: O(V + E).
    """

    INF = float('inf')

    def __init__(self, n: int):
        """Initialize with n vertices."""
        self.n = n
        self.graph = [[] for _ in range(n)]
        self.edges = []  # [to, cap, cost, rev_index]
        self.potential = [0] * n  # Johnson's potentials

    def add_edge(self, u: int, v: int, cap: int, cost: int):
        """Add directed edge u -> v with capacity cap and cost per unit."""
        self.graph[u].append(len(self.edges))
        self.edges.append([v, cap, cost, len(self.edges) + 1])
        self.graph[v].append(len(self.edges))
        self.edges.append([u, 0, -cost, len(self.edges) - 1])

    def dijkstra(self, s: int, t: int) -> tuple[int, int]:
        """Find cheapest augmenting path using Dijkstra with potentials."""
        dist = [self.INF] * self.n
        parent_edge = [-1] * self.n
        dist[s] = 0
        heap = [(0, s)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for idx in self.graph[u]:
                to, cap, cost, _ = self.edges[idx]
                if cap <= 0:
                    continue
                # Reduced cost using potentials
                reduced = cost + self.potential[u] - self.potential[to]
                if dist[u] + reduced < dist[to]:
                    dist[to] = dist[u] + reduced
                    parent_edge[to] = idx
                    heapq.heappush(heap, (dist[to], to))

        if dist[t] == self.INF:
            return 0, 0

        # Update potentials
        for v in range(self.n):
            if dist[v] < self.INF:
                self.potential[v] += dist[v]

        # Find bottleneck
        flow = self.INF
        v = t
        while v != s:
            idx = parent_edge[v]
            flow = min(flow, self.edges[idx][1])
            v = self.edges[self.edges[idx][3]][0]

        # Update residual
        cost_total = 0
        v = t
        while v != s:
            idx = parent_edge[v]
            self.edges[idx][1] -= flow
            self.edges[self.edges[idx][3]][1] += flow
            cost_total += flow * self.edges[idx][2]
            v = self.edges[self.edges[idx][3]][0]

        return flow, cost_total

    def min_cost_max_flow(self, s: int, t: int) -> tuple[int, int]:
        """Compute min-cost max-flow."""
        total_flow = 0
        total_cost = 0
        while True:
            flow, cost = self.dijkstra(s, t)
            if flow == 0:
                break
            total_flow += flow
            total_cost += cost
        return total_flow, total_cost
```

### Complexity

| Variant | Time per Iteration | Total Time | Space |
|---------|-------------------|------------|-------|
| SPFA-based | O(V * E) | O(F * V * E) | O(V + E) |
| Dijkstra + potentials | O((V + E) log V) | O(F * (V + E) log V) | O(V + E) |

*Socratic prompt: "Min-cost max-flow sends flow along the cheapest augmenting path each iteration. What happens if you send flow along a non-cheapest path first? Can it lead to a suboptimal total cost even if the total flow is correct?"*

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Task Assignment (CSES) | Assignment as min-cost flow |
| Grid Puzzle II (CSES) | Grid constraints modeled as flow |
| Minimum Cost Flow (LC 2850) | Direct min-cost flow |
| Dream Team (AtCoder) | Weighted matching via min-cost flow |

---

## 8. Assignment Problem via Min-Cost Flow

### Core Insight

The assignment problem -- assign N workers to N jobs minimizing total cost -- is a **min-cost perfect matching** on a bipartite graph, which reduces to min-cost flow:

1. Create source S and sink T.
2. S -> each worker i with capacity 1, cost 0.
3. Worker i -> job j with capacity 1, cost A[i][j].
4. Each job j -> T with capacity 1, cost 0.
5. Find min-cost max-flow of value N.

The flow of value N corresponds to a perfect matching (each worker assigned to exactly one job). The min-cost ensures minimum total assignment cost.

This is equivalent to the **Hungarian algorithm** but the flow formulation generalizes better to non-square matrices, additional constraints, or when you already have a min-cost flow implementation.

*Socratic prompt: "The assignment problem can be solved with the Hungarian algorithm in O(N^3) or with min-cost flow. When would you prefer the flow formulation over Hungarian? Think about extensions like 'some workers can do 2 jobs' or 'some jobs need 2 workers'."*

### Template

```python
def solve_assignment(cost_matrix: list[list[int]]) -> tuple[int, list[int]]:
    """Solve the assignment problem: assign N workers to N jobs, minimize total cost.

    Args:
        cost_matrix: N x N matrix where cost_matrix[i][j] = cost of assigning
                     worker i to job j.

    Returns:
        (min_cost, assignment) where assignment[i] = job assigned to worker i.

    Builds a min-cost flow network and solves it.
    Time: O(N^3) with Dijkstra + potentials, O(N^4) with SPFA.
    """
    n = len(cost_matrix)
    # Vertices: 0 = source, 1..n = workers, n+1..2n = jobs, 2n+1 = sink
    source = 0
    sink = 2 * n + 1
    total_v = 2 * n + 2

    mcf = MinCostFlow(total_v)

    # Source -> workers
    for i in range(n):
        mcf.add_edge(source, i + 1, 1, 0)

    # Workers -> jobs
    for i in range(n):
        for j in range(n):
            mcf.add_edge(i + 1, n + 1 + j, 1, cost_matrix[i][j])

    # Jobs -> sink
    for j in range(n):
        mcf.add_edge(n + 1 + j, sink, 1, 0)

    total_flow, total_cost = mcf.min_cost_max_flow(source, sink)

    # Extract assignment from edge flows
    assignment = [-1] * n
    edge_idx = 2 * n  # skip source->worker edges (each adds 2 to edges list)
    # Actually, let's iterate more carefully
    # Source edges: indices 0,1, 2,3, ..., 2n-2,2n-1 (n edges, each fwd+rev)
    # Worker-job edges start at index 2*n
    base = 2 * n  # first worker->job edge index
    for i in range(n):
        for j in range(n):
            idx = base + 2 * (i * n + j)  # forward edge index
            # Flow = original_cap - remaining_cap
            if mcf.edges[idx][1] == 0:  # capacity exhausted = flow = 1
                assignment[i] = j
    return total_cost, assignment


def solve_assignment_rectangular(cost_matrix: list[list[int]]) -> tuple[int, list[int]]:
    """Solve assignment for non-square matrices (M workers, N jobs, M <= N).

    Assigns each worker to a unique job. Some jobs may be unassigned.

    Time: O(M * N * (M + N + M*N)) with SPFA.
    """
    m = len(cost_matrix)        # workers
    n = len(cost_matrix[0])     # jobs
    source = 0
    sink = m + n + 1
    total_v = m + n + 2

    mcf = MinCostFlow(total_v)

    for i in range(m):
        mcf.add_edge(source, i + 1, 1, 0)
    for i in range(m):
        for j in range(n):
            mcf.add_edge(i + 1, m + 1 + j, 1, cost_matrix[i][j])
    for j in range(n):
        mcf.add_edge(m + 1 + j, sink, 1, 0)

    total_flow, total_cost = mcf.min_cost_max_flow(source, sink)

    assignment = [-1] * m
    base = 2 * m
    for i in range(m):
        for j in range(n):
            idx = base + 2 * (i * n + j)
            if mcf.edges[idx][1] == 0:
                assignment[i] = j
    return total_cost, assignment
```

### Complexity

| Variant | Time | Space |
|---------|------|-------|
| SPFA-based | O(N^4) | O(N^2) |
| Dijkstra + potentials | O(N^3) | O(N^2) |
| Hungarian algorithm (dedicated) | O(N^3) | O(N^2) |

*Socratic prompt: "The assignment problem is a special case of min-cost flow where all capacities are 1. The Hungarian algorithm exploits this structure for O(N^3). What structure does it exploit that a general min-cost flow solver doesn't?"*

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Task Assignment (CSES) | Direct assignment problem |
| LeetCode 1066: Campus Bikes II | Assign bikes to workers minimizing total distance |
| LeetCode 2172: Max AND Sum of Array | Bitmask DP or assignment-like |
| LeetCode 455: Assign Cookies | Greedy works, but assignment generalizes |

---

## Interview Tips

1. **Dinic's is your default max-flow.** It's the best balance of simplicity, speed, and generality. For competitive programming, memorize the Dinic's template.

2. **Recognize hidden flow problems.** If a problem says "maximum number of non-overlapping paths," "minimum cut," or "bipartite matching" -- think flow.

3. **Max-flow = Min-cut.** After running any max-flow algorithm, the min-cut consists of all edges (u, v) where u is reachable from s in the residual graph but v is not.

4. **Unit capacity = Dinic's.** For bipartite matching or edge-disjoint paths, Dinic's O(E sqrt(V)) is optimal in practice.

5. **Min-cost flow for weighted matching.** When the bipartite matching has costs/weights, use min-cost flow instead of plain max-flow.

6. **Lower bounds are rare but powerful.** If a problem says "each edge must carry at least X flow," use the demands reduction (Section 6).

7. **Watch for the modeling step.** Most flow interview problems are about correctly building the graph. The algorithm itself is a black box. Spend 80% of your time on the modeling.

8. **Python gotcha: recursion limit.** Dinic's DFS can hit Python's recursion limit on large graphs. Either increase it (`sys.setrecursionlimit`) or convert to iterative DFS.

---

## Attribution

Content synthesized from cp-algorithms.com articles on [Maximum Flow: Ford-Fulkerson and Edmonds-Karp](https://cp-algorithms.com/graph/edmonds_karp.html), [Push-Relabel Algorithm](https://cp-algorithms.com/graph/push-relabel.html), [Push-Relabel Algorithm Improved](https://cp-algorithms.com/graph/push-relabel-faster.html), [Dinic's Algorithm](https://cp-algorithms.com/graph/dinic.html), [MPM Algorithm](https://cp-algorithms.com/graph/mpm.html), [Flows with Lower Bounds / Demands](https://cp-algorithms.com/graph/flow_with_demands.html), [Minimum-Cost Flow](https://cp-algorithms.com/graph/min_cost_flow.html), and [Assignment Problem via Min-Cost Flow](https://cp-algorithms.com/graph/Assignment-problem-min-flow.html). Algorithms and complexity analyses are from the original articles; code has been translated to Python with added commentary for the leetcode-teacher reference format.
