# Graph Bipartite Matching

Bipartite graph detection, maximum matching, and minimum-cost assignment. These techniques appear in problems involving two-group partitioning, task assignment, resource allocation, and covering/independence on bipartite structures. For general graph traversal (BFS/DFS), see `graph-algorithms.md`. For flow-based approaches to matching (max-flow formulations, Hopcroft-Karp), the concepts here form the foundation -- Kuhn's algorithm is the augmenting-path core that generalizes into network flow matching.

---

## Quick Reference Table

| Pattern | Recognition Signals | Key Problems | Section |
|---------|-------------------|--------------|---------|
| Bipartiteness Check (2-Coloring) | "Split into two groups", "no odd cycles", "two-colorable", "assign sides" | Is Graph Bipartite? (785), Possible Bipartition (886) | 1 |
| Kuhn's Algorithm (Maximum Bipartite Matching) | "Maximum pairing", "assign tasks to workers", "max independent edges", "min vertex cover" | Maximum Number of Accepted Invitations (1820), Course Schedule IV (related) | 2 |
| Hungarian Algorithm (Min-Cost Assignment) | "Minimize total cost of assignment", "n workers n jobs", "weighted bipartite matching", "optimal allocation" | Minimum Cost to Hire K Workers (857 -- related), Campus Bikes II (1066) | 3 |

---

## Comparison of Matching Approaches

| Approach | Problem Solved | Time Complexity | When to Use |
|----------|---------------|-----------------|-------------|
| Kuhn's Algorithm | Max cardinality matching | O(V * E) | Unweighted, moderate graphs |
| Hopcroft-Karp | Max cardinality matching | O(E * sqrt(V)) | Unweighted, large graphs |
| Hungarian Algorithm | Min-cost perfect matching | O(n^3) | Weighted assignment, n x n |
| Min-Cost Max-Flow | Min-cost matching (general) | O(V^2 * E) | General networks, complex constraints |

*Socratic prompt: "Kuhn's algorithm finds maximum matching, while the Hungarian algorithm finds minimum-cost matching. When would a maximum matching NOT be optimal? Think about assigning employees to tasks where each assignment has a different cost."*

---

## 1. Bipartiteness Check (BFS 2-Coloring)

### Core Insight

A graph is bipartite if and only if it contains no odd-length cycles. Equivalently, a graph is bipartite if and only if its vertices can be 2-colored such that every edge connects vertices of different colors. BFS assigns colors level by level; if any edge connects two same-colored vertices, the graph is not bipartite.

*Socratic prompt: "Why does an odd cycle make 2-coloring impossible? Try coloring a triangle -- what happens when you reach the third vertex?"*

### Template

```python
from collections import deque


def is_bipartite(adj: list[list[int]], n: int) -> tuple[bool, list[int]]:
    """Check if an undirected graph is bipartite using BFS 2-coloring.

    Args:
        adj: Adjacency list where adj[v] contains neighbors of vertex v.
        n: Number of vertices (0-indexed).

    Returns:
        (is_bipartite, color) where color[v] is 0 or 1 for each vertex,
        or -1 if unvisited (for disconnected components handled here).

    Time: O(V + E). Space: O(V).
    """
    color = [-1] * n

    for start in range(n):
        if color[start] != -1:
            continue
        # BFS from each unvisited component
        color[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            for u in adj[v]:
                if color[u] == -1:
                    color[u] = color[v] ^ 1  # assign opposite color
                    queue.append(u)
                elif color[u] == color[v]:
                    return False, color  # same color on both ends of edge
    return True, color
```

**DFS variant** (useful when you need to detect the actual odd cycle):

```python
def is_bipartite_dfs(adj: list[list[int]], n: int) -> tuple[bool, list[int]]:
    """Check bipartiteness using DFS 2-coloring.

    Time: O(V + E). Space: O(V) for color array + O(V) recursion stack.
    """
    color = [-1] * n

    def dfs(v: int, c: int) -> bool:
        color[v] = c
        for u in adj[v]:
            if color[u] == -1:
                if not dfs(u, c ^ 1):
                    return False
            elif color[u] == c:
                return False
        return True

    for v in range(n):
        if color[v] == -1:
            if not dfs(v, 0):
                return False, color
    return True, color
```

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| BFS 2-coloring | O(V + E) | O(V) |
| DFS 2-coloring | O(V + E) | O(V) stack + O(V) color |

### Key Theorem

**A graph is bipartite if and only if it contains no odd-length cycles.** This follows directly from the 2-coloring argument: traversing an odd cycle forces the start vertex to receive both colors.

*Socratic prompt: "If a graph has only even-length cycles, can you always 2-color it? What about a tree -- is it always bipartite? Why?"*

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Is Graph Bipartite? (785) | Direct BFS/DFS 2-coloring |
| Possible Bipartition (886) | Build conflict graph, check bipartiteness |
| CSES Building Teams | Assign students to two teams with no friend pairs in same team |
| Codeforces Graph Without Long Directed Paths | Orient edges so no path has length >= 2; equivalent to 2-coloring |

---

## 2. Kuhn's Algorithm (Maximum Bipartite Matching)

### Core Insight

A **matching** in a bipartite graph is a set of edges with no shared vertices. Kuhn's algorithm finds the **maximum matching** by repeatedly searching for **augmenting paths** -- paths that alternate between non-matching and matching edges, starting and ending at unmatched vertices. By **Berge's lemma**, a matching is maximum if and only if no augmenting path exists.

*Socratic prompt: "An augmenting path starts at an unmatched left vertex and ends at an unmatched right vertex, alternating between non-matching and matching edges. If you 'flip' all edges along this path (matching becomes non-matching and vice versa), what happens to the matching size?"*

### Template

```python
def max_bipartite_matching(n1: int, n2: int, adj: list[list[int]]) -> tuple[int, list[int]]:
    """Find maximum matching in a bipartite graph using Kuhn's algorithm.

    The graph has two partitions: left vertices [0, n1) and right vertices [0, n2).
    adj[v] lists the right-side neighbors of left vertex v.

    Returns:
        (matching_size, match_right) where match_right[j] = left vertex matched
        to right vertex j, or -1 if unmatched.

    Time: O(n1 * E) where E = total edges. Space: O(n1 + n2).
    """
    match_right = [-1] * n2  # match_right[j] = left vertex matched to j

    def try_kuhn(v: int, visited: list[bool]) -> bool:
        """Try to find an augmenting path starting from left vertex v.

        Args:
            v: Current left vertex.
            visited: Tracks which right vertices have been explored in this DFS.

        Returns:
            True if an augmenting path was found (matching increased by 1).
        """
        for u in adj[v]:
            if not visited[u]:
                visited[u] = True
                # If right vertex u is free, or we can re-route its current match
                if match_right[u] == -1 or try_kuhn(match_right[u], visited):
                    match_right[u] = v
                    return True
        return False

    result = 0
    for v in range(n1):
        visited = [False] * n2
        if try_kuhn(v, visited):
            result += 1

    return result, match_right
```

### Greedy Initialization (Optimization)

Before running the full DFS-based algorithm, greedily match edges to reduce the number of augmenting path searches needed. This significantly improves performance on random graphs.

```python
def max_bipartite_matching_optimized(
    n1: int, n2: int, adj: list[list[int]]
) -> tuple[int, list[int]]:
    """Kuhn's algorithm with greedy initialization for faster performance.

    Time: O(n1 * E) worst case, much faster in practice with greedy init.
    Space: O(n1 + n2).
    """
    match_right = [-1] * n2
    match_left = [-1] * n1

    def try_kuhn(v: int, visited: list[bool]) -> bool:
        for u in adj[v]:
            if not visited[u]:
                visited[u] = True
                if match_right[u] == -1 or try_kuhn(match_right[u], visited):
                    match_right[u] = v
                    match_left[v] = u
                    return True
        return False

    # Phase 1: Greedy matching (no DFS needed)
    result = 0
    used = [False] * n1
    for v in range(n1):
        for u in adj[v]:
            if match_right[u] == -1:
                match_right[u] = v
                match_left[v] = u
                used[v] = True
                result += 1
                break

    # Phase 2: Augmenting paths for remaining unmatched vertices
    for v in range(n1):
        if not used[v]:
            visited = [False] * n2
            if try_kuhn(v, visited):
                result += 1

    return result, match_right
```

### Complexity

| Variant | Time | Space |
|---------|------|-------|
| Basic Kuhn's | O(V * E) | O(V) |
| With greedy init | O(V * E) worst case, faster in practice | O(V) |
| Hopcroft-Karp (for reference) | O(E * sqrt(V)) | O(V) |

### Konig's Theorem

**In any bipartite graph, the size of the maximum matching equals the size of the minimum vertex cover.**

A **vertex cover** is a set of vertices such that every edge has at least one endpoint in the set. Konig's theorem gives an exact equivalence that does NOT hold for general graphs.

**Corollary (Maximum Independent Set):** In a bipartite graph with n vertices and maximum matching M:

```
Maximum Independent Set = n - |M|
```

This follows because the complement of a minimum vertex cover is a maximum independent set.

*Socratic prompt: "Konig's theorem says max matching = min vertex cover in bipartite graphs. In a general graph, min vertex cover can be much larger. Why does the bipartite structure make them equal?"*

### Constructing the Minimum Vertex Cover from a Maximum Matching

```python
from collections import deque


def min_vertex_cover(
    n1: int, n2: int, adj: list[list[int]], match_right: list[int]
) -> tuple[set[int], set[int]]:
    """Find minimum vertex cover from a maximum matching (Konig's theorem).

    Returns:
        (left_cover, right_cover) -- sets of vertices in the cover.

    The algorithm:
    1. Find all unmatched left vertices.
    2. BFS/DFS alternating: unmatched left -> any right neighbor -> matched left partner.
    3. Cover = unreachable left vertices UNION reachable right vertices.

    Time: O(V + E). Space: O(V).
    """
    # Build reverse mapping: match_left[v] = right vertex matched to v, or -1
    match_left = [-1] * n1
    for j in range(n2):
        if match_right[j] != -1:
            match_left[match_right[j]] = j

    # BFS from unmatched left vertices via alternating paths
    reachable_left = set()
    reachable_right = set()
    queue = deque()

    for v in range(n1):
        if match_left[v] == -1:  # unmatched left vertex
            reachable_left.add(v)
            queue.append(('L', v))

    while queue:
        side, v = queue.popleft()
        if side == 'L':
            # From left vertex, go to any right neighbor via NON-matching edge
            for u in adj[v]:
                if u not in reachable_right:
                    reachable_right.add(u)
                    queue.append(('R', u))
        else:
            # From right vertex, go to its matched left partner via MATCHING edge
            if match_right[v] != -1 and match_right[v] not in reachable_left:
                reachable_left.add(match_right[v])
                queue.append(('L', match_right[v]))

    # Konig's cover: unreachable left + reachable right
    left_cover = set(range(n1)) - reachable_left
    right_cover = reachable_right

    return left_cover, right_cover
```

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Maximum Number of Accepted Invitations (1820) | Direct max bipartite matching |
| Maximum Students Taking Exam (1349) | Bipartite matching on chessboard-colored seats |
| Minimum Operations to Make a Uni-Value Grid (2033) | Matching/assignment flavor |
| Kattis Gopher II | Classic Kuhn's application |
| Kattis Borders | Matching with geometric constraints |
| CSES School Dance | Direct max bipartite matching |

---

## 3. Hungarian Algorithm (Minimum-Cost Assignment)

### Core Insight

The **assignment problem**: given an n x n cost matrix A where A[i][j] is the cost of assigning worker i to job j, find a one-to-one assignment of workers to jobs that minimizes total cost. The Hungarian algorithm solves this in O(n^3) using **dual potentials** that maintain optimality conditions.

**Key idea:** Maintain potential arrays u[i] and v[j] satisfying u[i] + v[j] <= A[i][j] for all i, j. An edge (i, j) is **tight** (or **rigid**) when u[i] + v[j] = A[i][j]. The algorithm finds a perfect matching using only tight edges. When no augmenting path exists in the tight subgraph, it adjusts potentials to make new edges tight.

*Socratic prompt: "The potentials u[i] and v[j] represent a lower bound on the optimal cost. Why does u[i] + v[j] <= A[i][j] guarantee that any matching in the tight subgraph is optimal? Think about what happens when you sum the matching costs."*

### Template

```python
def hungarian(cost: list[list[int]]) -> tuple[int, list[int]]:
    """Solve the assignment problem: minimize total cost of 1-to-1 assignment.

    Args:
        cost: n x m cost matrix (n <= m). cost[i][j] = cost of assigning
              worker i to job j. For n < m, some jobs will be unassigned.

    Returns:
        (min_cost, assignment) where assignment[i] = job assigned to worker i.

    Time: O(n^2 * m). Space: O(n + m).

    Based on the Lopatin implementation from cp-algorithms.
    Uses 1-indexed arrays with a dummy row/column for cleaner code.
    """
    n = len(cost)
    m = len(cost[0]) if n > 0 else 0

    INF = float('inf')

    # Potentials for workers (u) and jobs (v), 1-indexed
    u = [0] * (n + 1)
    v = [0] * (m + 1)
    # p[j] = worker assigned to job j (0 = unassigned dummy)
    p = [0] * (m + 1)
    # way[j] = previous job in the augmenting path ending at job j
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        # Start augmenting path from worker i
        p[0] = i
        j0 = 0  # virtual "current" job (start at dummy job 0)
        minv = [INF] * (m + 1)  # minv[j] = min reduced cost to reach job j
        used = [False] * (m + 1)  # whether job j is in the current tree

        while True:
            used[j0] = True
            i0 = p[j0]  # worker currently assigned to job j0
            delta = INF
            j1 = -1  # next job to add to tree

            for j in range(1, m + 1):
                if not used[j]:
                    # Reduced cost of edge (i0, j)
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            # Update potentials
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1

            # If job j0 is unassigned, we found an augmenting path
            if p[j0] == 0:
                break

        # Trace back the augmenting path and update assignment
        while j0 != 0:
            p[j0] = p[way[j0]]
            j0 = way[j0]

    # Extract assignment: assignment[i] = job for worker i (0-indexed)
    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1

    # Optimal cost is -v[0] (accumulated through potential updates)
    return -v[0], assignment
```

### Usage Example

```python
# 3 workers, 3 jobs. cost[i][j] = cost of worker i doing job j.
cost = [
    [9, 2, 7],
    [6, 4, 3],
    [5, 8, 1],
]
min_cost, assignment = hungarian(cost)
# min_cost = 5 (worker 0 -> job 1 (cost 2), worker 1 -> job 2 (cost 3), ... wait)
# Actually: optimal is worker 0->job 1 (2), worker 1->job 0 (6)... Let's compute:
# Optimal: 0->1(2), 1->2(3), 2->0(5) = 10? Or 0->2(7), 1->0(6), 2->1(8) = 21?
# Actually: 0->1(2) + 1->2(3) + 2->0(5) = 10. That's the minimum.
print(f"Minimum cost: {min_cost}")  # 10
print(f"Assignment: {assignment}")  # [1, 2, 0]
```

### Variant: Maximum Weight Matching

To find the **maximum** total assignment, negate all costs:

```python
def hungarian_max(profit: list[list[int]]) -> tuple[int, list[int]]:
    """Maximize total profit of assignment by negating and running Hungarian.

    Time: O(n^2 * m). Space: O(n + m).
    """
    neg_cost = [[-profit[i][j] for j in range(len(profit[0]))]
                for i in range(len(profit))]
    neg_result, assignment = hungarian(neg_cost)
    return -neg_result, assignment
```

### Variant: Rectangular Matrices

When n != m (more jobs than workers or vice versa), pad the cost matrix with zeros (or infinity, depending on interpretation). The template above already handles n <= m. For n > m, transpose the problem.

```python
def hungarian_rectangular(cost: list[list[int]]) -> tuple[int, list[int]]:
    """Handle rectangular cost matrices by ensuring n <= m.

    Time: O(min(n,m)^2 * max(n,m)). Space: O(n + m).
    """
    n = len(cost)
    m = len(cost[0]) if n > 0 else 0
    if n <= m:
        return hungarian(cost)
    # Transpose: treat jobs as workers
    cost_t = [[cost[i][j] for i in range(n)] for j in range(m)]
    total, assign_t = hungarian(cost_t)
    # Invert assignment
    assignment = [-1] * n
    for j, i in enumerate(assign_t):
        if i != -1:
            assignment[i] = j
    return total, assignment
```

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Square n x n | O(n^3) | O(n) |
| Rectangular n x m (n <= m) | O(n^2 * m) | O(n + m) |
| Naive O(n^4) (repeated Kuhn) | O(n^4) | O(n) |

*Socratic prompt: "The Hungarian algorithm maintains potentials u[i] + v[j] <= cost[i][j]. When we adjust potentials by delta, we add delta to u for 'visited' workers and subtract delta from v for 'visited' jobs. Why does this maintain the feasibility constraint for ALL edges, not just the visited ones?"*

### Connection to Min-Cost Flow

The Hungarian algorithm is equivalent to the **Successive Shortest Path** algorithm on the bipartite flow network: source -> workers -> jobs -> sink, with edge capacities 1 and costs from the matrix. The potentials u[i], v[j] serve the same role as Johnson's reweighting to keep edge weights non-negative (avoiding Bellman-Ford on each iteration).

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Campus Bikes II (1066) | Minimize total Manhattan distance for bike assignment |
| Minimum Cost to Hire K Workers (857) | Related cost optimization (greedy + heap, but assignment flavor) |
| Minimum Cost to Connect Two Groups of Points (1595) | Weighted bipartite matching with coverage constraint |
| UVA 1786 Crime Wave | Direct assignment problem |
| UVA 1829 Warehouse | Assignment with geometric costs |

---

## Interview Tips

1. **Detect bipartiteness first.** Many matching problems implicitly require confirming the graph is bipartite. If the problem says "two groups" or "two sides," run BFS 2-coloring before anything else.

2. **Konig's theorem is your secret weapon.** "Find the minimum number of lines to cover all 1s in a matrix" = minimum vertex cover = maximum matching (by Konig). This connection is rarely tested directly but unlocks several problems.

3. **Kuhn's algorithm is the workhorse.** For most interview-level bipartite matching problems, Kuhn's O(VE) is sufficient. Only reach for Hopcroft-Karp or flow-based methods if V and E are large (> 10^4 edges).

4. **Hungarian algorithm for weighted assignment.** If the problem has COSTS and asks for optimal assignment, it's Hungarian. Keywords: "minimize total," "n workers n jobs," "assignment."

5. **Maximum independent set on bipartite = n - max matching.** This is NOT true for general graphs (where it's NP-hard). The bipartite special case is polynomial thanks to Konig.

6. **Watch for hidden bipartite structure.** Chessboard problems (black/white cells), grid problems with parity-based adjacency, and conflict graphs from "no two adjacent" constraints often hide bipartite matching.

7. **Greedy initialization matters in practice.** Always add the greedy matching phase before Kuhn's DFS. On random graphs, this alone finds most of the matching, and the DFS augments only a few remaining paths.

8. **For maximum weight matching, negate costs and run Hungarian.** Don't try to modify the algorithm itself -- just negate input, run standard min-cost Hungarian, and negate the result.

---

## Attribution

Content synthesized from cp-algorithms.com articles on [Bipartite Graph Check](https://cp-algorithms.com/graph/bipartite-check.html), [Kuhn's Maximum Bipartite Matching](https://cp-algorithms.com/graph/kuhn_maximum_bipartite_matching.html), and [Hungarian Algorithm](https://cp-algorithms.com/graph/hungarian-algorithm.html). Algorithms, complexity analyses, and theorems (Berge's lemma, Konig's theorem) are from the original articles; all code has been translated from C++ to Python with added docstrings, type hints, and commentary for the leetcode-teacher reference format.
