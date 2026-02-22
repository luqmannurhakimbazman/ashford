# Graph MST & Spanning Tree Algorithms

Minimum spanning tree construction, counting, and encoding algorithms for competitive programming and interviews. Covers Prim's and Kruskal's MST algorithms, second-best MST, spanning tree counting (Kirchhoff's theorem), and labeled tree encoding (Prufer sequences). For foundational graph traversal (DFS, BFS, Union-Find, Dijkstra) and basic MST usage, see `graph-algorithms.md`. For matrix determinant computation used in Kirchhoff's theorem, see `linear-algebra-gauss.md`. For combinatorics (Cayley's formula connections), see `combinatorics.md`.

---

## Quick Reference Table

| Pattern | Recognition Signals | Key Problems | Section |
|---------|-------------------|--------------|---------|
| Prim's MST | "Minimum cost to connect all points", dense graph, adjacency matrix given | Min Cost to Connect All Points (1584), Minimum Spanning Tree | 1 |
| Kruskal's MST | "Minimum cost to connect all", edge list given, sparse graph | Min Cost to Connect All Points (1584), Connecting Cities With Minimum Cost (1135) | 2 |
| Kruskal's with DSU | Same as Kruskal + "dynamic connectivity", "are nodes connected?" | Redundant Connection (684), Number of Islands II (305) | 3 |
| Second-Best MST | "Second minimum spanning tree", "swap one edge", "nearly optimal tree" | Find Critical and Pseudo-Critical Edges (1489) | 4 |
| Kirchhoff's Theorem | "Count spanning trees", "number of spanning trees", determinant of Laplacian | -- (competitive programming) | 5 |
| Prufer Code | "Encode labeled tree", "Cayley's formula", "n^(n-2) labeled trees" | -- (competitive programming) | 6 |

---

## Prim vs Kruskal Decision Table

| Criterion | Prim's | Kruskal's |
|-----------|--------|-----------|
| Best for | Dense graphs (many edges) | Sparse graphs (few edges) |
| Data structure | Priority queue (min-heap) | Sorted edge list + DSU |
| Grows MST from | A single source vertex | Global sorted edges |
| Time (dense, adj matrix) | O(V^2) | O(E log E) -- worse for dense |
| Time (sparse, adj list + heap) | O(E log V) | O(E log E) ~= O(E log V) |
| Edge list required? | No (works from adj list) | Yes |
| Easy to implement? | Slightly more complex (heap) | Simpler (sort + union-find) |
| Interview preference | Less common | More common (simpler DSU) |

*Socratic prompt: "Given a complete graph on V vertices, how many edges are there? Which algorithm would you choose and why?"*

---

## 1. Prim's Minimum Spanning Tree

### Core Insight

Prim's algorithm grows the MST one vertex at a time. Start from any vertex; at each step, add the cheapest edge that connects a vertex already in the MST to one not yet included. This greedy choice is correct because of the **cut property**: for any cut of the graph, the minimum-weight crossing edge belongs to some MST.

*Socratic prompt: "If you pick any vertex and always extend the tree by the cheapest neighboring edge, why does this guarantee a minimum spanning tree? What if two edges have the same weight?"*

### Template (Dense Graphs -- O(V^2))

Best when the graph is represented as an adjacency matrix or is nearly complete.

```python
import math


def prim_dense(adj_matrix: list[list[float]]) -> tuple[float, list[tuple[int, int]]]:
    """Prim's MST for dense graphs using adjacency matrix.

    Args:
        adj_matrix: adj_matrix[u][v] = weight of edge (u,v), math.inf if no edge.

    Returns:
        (total_cost, list of MST edges as (u, v) pairs)

    Time: O(V^2). Space: O(V).
    """
    n = len(adj_matrix)
    INF = math.inf

    # min_edge[v] = (weight of cheapest edge from v to MST, neighbor in MST)
    min_edge = [(INF, -1)] * n
    min_edge[0] = (0, -1)  # start from vertex 0

    selected = [False] * n
    total_cost = 0
    mst_edges = []

    for _ in range(n):
        # Find unselected vertex with minimum edge to MST
        v = -1
        for j in range(n):
            if not selected[j] and (v == -1 or min_edge[j][0] < min_edge[v][0]):
                v = j

        if min_edge[v][0] == INF:
            # Graph is disconnected -- no MST exists
            return INF, []

        selected[v] = True
        total_cost += min_edge[v][0]
        if min_edge[v][1] != -1:
            mst_edges.append((min_edge[v][1], v))

        # Update minimum edges for all unselected neighbors of v
        for to in range(n):
            if not selected[to] and adj_matrix[v][to] < min_edge[to][0]:
                min_edge[to] = (adj_matrix[v][to], v)

    return total_cost, mst_edges
```

### Template (Sparse Graphs -- O(E log V))

Uses a min-heap for efficient extraction. Preferred when the graph is sparse (E << V^2).

```python
import heapq
from collections import defaultdict


def prim_sparse(
    n: int, edges: list[tuple[int, int, int]]
) -> tuple[int, list[tuple[int, int, int]]]:
    """Prim's MST for sparse graphs using adjacency list + min-heap.

    Args:
        n: number of vertices (0-indexed).
        edges: list of (u, v, weight) undirected edges.

    Returns:
        (total_cost, list of MST edges as (u, v, weight) tuples)

    Time: O(E log V). Space: O(V + E).
    """
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((w, v))
        adj[v].append((w, u))

    total_cost = 0
    mst_edges = []
    visited = [False] * n

    # (weight, vertex, parent)
    heap = [(0, 0, -1)]

    while heap and len(mst_edges) < n - 1:
        w, v, parent = heapq.heappop(heap)
        if visited[v]:
            continue
        visited[v] = True
        total_cost += w
        if parent != -1:
            mst_edges.append((parent, v, w))

        for edge_w, to in adj[v]:
            if not visited[to]:
                heapq.heappush(heap, (edge_w, to, v))

    if len(mst_edges) != n - 1:
        return -1, []  # disconnected graph

    return total_cost, mst_edges
```

### Complexity

| Variant | Time | Space | Best For |
|---------|------|-------|----------|
| Dense (adj matrix) | O(V^2) | O(V) | Complete / dense graphs |
| Sparse (heap) | O(E log V) | O(V + E) | Sparse graphs |
| Sparse (Fibonacci heap) | O(E + V log V) | O(V + E) | Theoretical best |

*Socratic prompt: "Why does the dense version NOT use a heap? What happens to heap performance when E is close to V^2?"*

*Socratic prompt: "In the sparse version, we might push the same vertex multiple times into the heap. Why is that okay? How does the `visited` check handle duplicates?"*

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Min Cost to Connect All Points (1584) | Coordinates given -- build complete graph, Prim O(V^2) is ideal |
| Connecting Cities With Minimum Cost (1135) | Direct edge list -- Prim or Kruskal both work |
| Minimum Spanning Tree (Kattis/SPOJ) | Standard MST |

---

## 2. Kruskal's Minimum Spanning Tree

### Core Insight

Kruskal's algorithm builds the MST by processing edges in order of increasing weight. For each edge, if its two endpoints are in different components, add it to the MST (merging the components). Otherwise, skip it (adding it would create a cycle). The correctness follows from the **cycle property**: the heaviest edge in any cycle is never in the MST.

*Socratic prompt: "If you sort all edges by weight and greedily pick edges that don't form cycles, why does this give a minimum spanning tree? What invariant holds after each step?"*

### Template (Simple -- O(E log E + V^2))

Uses a component-ID array. Simple but slow for the merge step.

```python
def kruskal_simple(
    n: int, edges: list[tuple[int, int, int]]
) -> tuple[int, list[tuple[int, int, int]]]:
    """Kruskal's MST with simple component tracking.

    Args:
        n: number of vertices (0-indexed).
        edges: list of (u, v, weight) undirected edges.

    Returns:
        (total_cost, list of MST edges as (u, v, weight) tuples)

    Time: O(E log E + V^2). Space: O(V + E).
    """
    edges_sorted = sorted(edges, key=lambda e: e[2])
    tree_id = list(range(n))  # tree_id[v] = component ID of vertex v
    total_cost = 0
    mst_edges = []

    for u, v, w in edges_sorted:
        if tree_id[u] != tree_id[v]:
            total_cost += w
            mst_edges.append((u, v, w))

            # Merge: relabel all vertices in v's component to u's component
            old_id = tree_id[v]
            new_id = tree_id[u]
            for i in range(n):
                if tree_id[i] == old_id:
                    tree_id[i] = new_id

            if len(mst_edges) == n - 1:
                break

    return total_cost, mst_edges
```

### Complexity (Simple Version)

| Operation | Time |
|-----------|------|
| Sort edges | O(E log E) |
| Merge components (V-1 merges, each O(V)) | O(V^2) |
| **Total** | **O(E log E + V^2)** |

*Socratic prompt: "The merge step scans all V vertices each time. Can you think of a data structure that makes 'are u and v in the same set?' and 'merge their sets' nearly O(1)?"*

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Min Cost to Connect All Points (1584) | Build all-pairs edges, sort, apply Kruskal |
| Connecting Cities With Minimum Cost (1135) | Direct edge list input |

---

## 3. Kruskal's MST with Disjoint Set Union (DSU)

### Core Insight

Replace the O(V) merge in basic Kruskal's with a **Disjoint Set Union** (Union-Find) data structure. With **path compression** and **union by rank**, each find/union operation runs in amortized O(alpha(V)) time, where alpha is the inverse Ackermann function (effectively constant, <= 4 for any practical input).

*Socratic prompt: "What does path compression do? After `find(x)`, what changes about x's position in the tree? Why does this speed up future queries?"*

### DSU Template

```python
class DSU:
    """Disjoint Set Union with path compression and union by rank.

    Time: O(alpha(n)) amortized per operation. Space: O(n).
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, v: int) -> int:
        """Find root of v's component with path compression."""
        if self.parent[v] != v:
            self.parent[v] = self.find(self.parent[v])
        return self.parent[v]

    def union(self, a: int, b: int) -> bool:
        """Merge components of a and b. Returns True if they were different.

        Uses union by rank to keep the tree shallow.
        """
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1
        return True
```

### Kruskal + DSU Template

```python
def kruskal_dsu(
    n: int, edges: list[tuple[int, int, int]]
) -> tuple[int, list[tuple[int, int, int]]]:
    """Kruskal's MST using DSU for near-linear performance.

    Args:
        n: number of vertices (0-indexed).
        edges: list of (u, v, weight) undirected edges.

    Returns:
        (total_cost, list of MST edges as (u, v, weight) tuples)

    Time: O(E log E + E * alpha(V)) ≈ O(E log E). Space: O(V + E).
    """
    edges_sorted = sorted(edges, key=lambda e: e[2])
    dsu = DSU(n)
    total_cost = 0
    mst_edges = []

    for u, v, w in edges_sorted:
        if dsu.union(u, v):
            total_cost += w
            mst_edges.append((u, v, w))
            if len(mst_edges) == n - 1:
                break  # MST complete -- early termination

    return total_cost, mst_edges
```

### Complexity

| Operation | Time |
|-----------|------|
| Sort edges | O(E log E) |
| E union/find operations | O(E * alpha(V)) ≈ O(E) |
| **Total** | **O(E log E)** ≈ **O(E log V)** |

> **Note:** O(E log E) = O(E log V) because E <= V^2, so log E <= 2 log V.

### DSU Variants for Interviews

| Variant | Modification | Use Case |
|---------|-------------|----------|
| Union by rank | Attach shorter tree under taller | Standard -- keeps height O(log n) |
| Union by size | Attach smaller set under larger | When you need component sizes |
| Path compression only | Skip rank, just compress paths | Simpler code, still very fast |
| Weighted DSU | Store edge weights along parent pointers | "Distance to root" queries |

*Socratic prompt: "Union by rank keeps the tree height at O(log n). Path compression flattens it further. Why do you need BOTH for the O(alpha(n)) guarantee? What happens with only one?"*

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Min Cost to Connect All Points (1584) | Standard Kruskal + DSU |
| Redundant Connection (684) | Find the edge that creates a cycle using DSU |
| Number of Islands II (305) | Online connectivity -- add cells one by one, DSU tracks components |
| Accounts Merge (721) | Union-Find to group accounts by shared emails |
| Connecting Cities With Minimum Cost (1135) | Standard Kruskal + DSU |
| Number of Connected Components (323) | DSU to count components |

---

## 4. Second-Best Minimum Spanning Tree

### Core Insight

The second-best MST differs from the optimal MST by **exactly one edge swap**: remove one MST edge and add one non-MST edge. The key observation is that for each non-MST edge (u, v), adding it to the MST creates a cycle, and the best swap removes the **heaviest edge on the path from u to v in the MST**. We want the swap that increases total weight the least.

*Socratic prompt: "Why does the second-best MST differ from the best by exactly one edge? Can it differ by two or more swaps?"*

### Approach 1: Brute Force (O(V * E))

For each of the V-1 MST edges, remove it and recompute the MST. Take the minimum.

```python
def second_best_mst_brute(
    n: int, edges: list[tuple[int, int, int]]
) -> int:
    """Find the cost of the second-best MST by trying each edge removal.

    Args:
        n: number of vertices (0-indexed).
        edges: list of (u, v, weight) undirected edges.

    Returns:
        Cost of the second-best MST, or -1 if none exists.

    Time: O(V * E * alpha(V)). Space: O(V + E).
    """
    import math

    # First, find the MST
    mst_cost, mst_edges = kruskal_dsu(n, edges)
    mst_edge_set = set()
    for u, v, w in mst_edges:
        mst_edge_set.add((min(u, v), max(u, v), w))

    best_second = math.inf

    # Try removing each MST edge and rebuilding
    for skip_u, skip_v, skip_w in mst_edges:
        remaining = [
            (u, v, w) for u, v, w in edges
            if not (min(u, v) == min(skip_u, skip_v)
                    and max(u, v) == max(skip_u, skip_v)
                    and w == skip_w)
        ]
        cost, result = kruskal_dsu(n, remaining)
        if len(result) == n - 1:
            best_second = min(best_second, cost)

    return best_second if best_second != math.inf else -1
```

### Approach 2: LCA + Path Maximum (O(E log V))

The optimal approach uses **binary lifting** on the MST to find the maximum edge weight on the path between any two vertices in O(log V) time per query.

```python
import math
from collections import defaultdict


def second_best_mst_lca(
    n: int, edges: list[tuple[int, int, int]]
) -> int:
    """Find the cost of the second-best MST using LCA with binary lifting.

    For each non-MST edge (u, v, w), compute:
        delta = w - max_weight_on_path(u, v, in MST)
    The second-best MST cost = MST cost + min(delta) over all non-MST edges.

    Args:
        n: number of vertices (0-indexed).
        edges: list of (u, v, weight) undirected edges.

    Returns:
        Cost of the second-best MST.

    Time: O(E log V) for sorting + O(V log V) for LCA preprocess
          + O(E log V) for queries = O(E log V).
    Space: O(V log V + E).
    """
    # Step 1: Build MST using Kruskal
    edges_sorted = sorted(edges, key=lambda e: e[2])
    dsu = DSU(n)
    mst_cost = 0
    mst_adj = defaultdict(list)  # adjacency list of MST
    mst_edge_set = set()

    for u, v, w in edges_sorted:
        if dsu.union(u, v):
            mst_cost += w
            mst_adj[u].append((v, w))
            mst_adj[v].append((u, w))
            mst_edge_set.add((min(u, v), max(u, v)))

    # Step 2: Binary lifting preprocess on MST
    LOG = max(1, n.bit_length())
    depth = [0] * n
    # up[v][k] = 2^k-th ancestor of v
    up = [[0] * LOG for _ in range(n)]
    # max_edge[v][k] = maximum edge weight on path from v to up[v][k]
    max_edge = [[0] * LOG for _ in range(n)]

    # BFS to set depth and direct parent (up[v][0])
    visited = [False] * n
    visited[0] = True
    queue = [0]
    head = 0
    while head < len(queue):
        v = queue[head]
        head += 1
        for to, w in mst_adj[v]:
            if not visited[to]:
                visited[to] = True
                depth[to] = depth[v] + 1
                up[to][0] = v
                max_edge[to][0] = w
                queue.append(to)

    # Fill binary lifting tables
    for k in range(1, LOG):
        for v in range(n):
            up[v][k] = up[up[v][k - 1]][k - 1]
            max_edge[v][k] = max(max_edge[v][k - 1],
                                 max_edge[up[v][k - 1]][k - 1])

    def get_max_on_path(u: int, v: int) -> int:
        """Return the maximum edge weight on the MST path from u to v."""
        if depth[u] < depth[v]:
            u, v = v, u

        result = 0
        diff = depth[u] - depth[v]

        # Lift u up to same depth as v
        for k in range(LOG):
            if (diff >> k) & 1:
                result = max(result, max_edge[u][k])
                u = up[u][k]

        if u == v:
            return result

        # Lift both until they meet
        for k in range(LOG - 1, -1, -1):
            if up[u][k] != up[v][k]:
                result = max(result, max_edge[u][k], max_edge[v][k])
                u = up[u][k]
                v = up[v][k]

        result = max(result, max_edge[u][0], max_edge[v][0])
        return result

    # Step 3: For each non-MST edge, compute delta
    best_delta = math.inf
    for u, v, w in edges:
        key = (min(u, v), max(u, v))
        if key not in mst_edge_set:
            path_max = get_max_on_path(u, v)
            if w > path_max:  # strict inequality for distinct second-best
                best_delta = min(best_delta, w - path_max)

    return mst_cost + best_delta if best_delta != math.inf else -1
```

### Complexity

| Approach | Time | Space |
|----------|------|-------|
| Brute force (remove each MST edge) | O(V * E * alpha(V)) | O(V + E) |
| LCA + binary lifting | O(E log V) | O(V log V + E) |

*Socratic prompt: "When you add a non-MST edge (u,v) to the MST, a unique cycle forms. Why is removing the heaviest edge on that cycle the best swap? What if the heaviest edge has the same weight as the new edge?"*

*Socratic prompt: "Binary lifting stores the maximum edge for every 2^k ancestor. How does this let you find the path maximum in O(log V) instead of O(V)?"*

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Find Critical and Pseudo-Critical Edges in MST (1489) | Classify each edge: critical (in ALL MSTs), pseudo-critical (in SOME MST), or neither |
| Second Minimum Spanning Tree (various OJs) | Direct application of approach 2 |

---

## 5. Kirchhoff's Theorem (Matrix Tree Theorem)

### Core Insight

Kirchhoff's theorem counts the **number of spanning trees** in a graph using linear algebra. Construct the **Laplacian matrix** L = D - A (degree matrix minus adjacency matrix), delete any one row and the corresponding column, then compute the **determinant** of the resulting (n-1) x (n-1) matrix. This determinant equals the number of spanning trees.

*Socratic prompt: "The Laplacian matrix L has the property that every row and column sums to zero. Why does that mean det(L) = 0? Why must we delete a row and column before computing the determinant?"*

### Matrix Construction

For an undirected graph with n vertices:
- **Adjacency matrix A:** A[i][j] = number of edges between i and j
- **Degree matrix D:** D[i][i] = degree of vertex i, D[i][j] = 0 for i != j
- **Laplacian L:** L = D - A, so L[i][i] = degree(i) and L[i][j] = -A[i][j]

### Template

```python
def count_spanning_trees(n: int, edges: list[tuple[int, int]]) -> float:
    """Count the number of spanning trees using Kirchhoff's theorem.

    Constructs the Laplacian matrix, deletes the last row and column,
    and computes the determinant of the resulting matrix.

    Args:
        n: number of vertices (0-indexed).
        edges: list of (u, v) undirected edges (unweighted).

    Returns:
        Number of spanning trees (as float; round for integer answer).

    Time: O(V^3) for Gaussian elimination. Space: O(V^2).
    """
    # Build Laplacian matrix
    L = [[0.0] * n for _ in range(n)]
    for u, v in edges:
        L[u][v] -= 1
        L[v][u] -= 1
        L[u][u] += 1
        L[v][v] += 1

    # Delete last row and last column -> (n-1) x (n-1) matrix
    size = n - 1
    matrix = [[L[i][j] for j in range(size)] for i in range(size)]

    # Gaussian elimination to compute determinant
    det = 1.0
    for col in range(size):
        # Find pivot
        pivot = -1
        for row in range(col, size):
            if abs(matrix[row][col]) > 1e-9:
                pivot = row
                break
        if pivot == -1:
            return 0  # singular matrix -> 0 spanning trees

        if pivot != col:
            matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
            det *= -1

        det *= matrix[col][col]

        # Eliminate below
        for row in range(col + 1, size):
            if abs(matrix[row][col]) > 1e-9:
                factor = matrix[row][col] / matrix[col][col]
                for k in range(col, size):
                    matrix[row][k] -= factor * matrix[col][k]

    return round(abs(det))
```

### Modular Version (for Large Answers)

When the answer must be computed modulo a prime p:

```python
def count_spanning_trees_mod(
    n: int, edges: list[tuple[int, int]], mod: int
) -> int:
    """Count spanning trees modulo a prime using Kirchhoff's theorem.

    Uses modular arithmetic throughout to avoid floating-point errors.

    Args:
        n: number of vertices (0-indexed).
        edges: list of (u, v) undirected edges.
        mod: a prime modulus.

    Returns:
        Number of spanning trees mod `mod`.

    Time: O(V^3). Space: O(V^2).
    """
    # Build Laplacian matrix mod p
    L = [[0] * n for _ in range(n)]
    for u, v in edges:
        L[u][v] = (L[u][v] - 1) % mod
        L[v][u] = (L[v][u] - 1) % mod
        L[u][u] = (L[u][u] + 1) % mod
        L[v][v] = (L[v][v] + 1) % mod

    size = n - 1
    matrix = [[L[i][j] for j in range(size)] for i in range(size)]

    det = 1
    for col in range(size):
        pivot = -1
        for row in range(col, size):
            if matrix[row][col] != 0:
                pivot = row
                break
        if pivot == -1:
            return 0

        if pivot != col:
            matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
            det = (-det) % mod

        det = det * matrix[col][col] % mod
        inv = pow(matrix[col][col], mod - 2, mod)  # Fermat's little theorem

        for row in range(col + 1, size):
            if matrix[row][col] != 0:
                factor = matrix[row][col] * inv % mod
                for k in range(col, size):
                    matrix[row][k] = (matrix[row][k] - factor * matrix[col][k]) % mod

    return det % mod
```

### Complexity

| Aspect | Value |
|--------|-------|
| Time | O(V^3) -- dominated by Gaussian elimination |
| Space | O(V^2) -- storing the Laplacian |

### Worked Example

For the complete graph K4 (4 vertices, all pairs connected):

```
Laplacian L:
 [ 3  -1  -1  -1]
 [-1   3  -1  -1]
 [-1  -1   3  -1]
 [-1  -1  -1   3]

Delete last row and column:
 [ 3  -1  -1]
 [-1   3  -1]
 [-1  -1   3]

det = 3*(9-1) - (-1)*(-3-1) + (-1)*(1+3) = 3*8 - 1*4 - 4 = 24 - 4 - 4 = 16
```

K4 has 16 spanning trees. In general, Kn has n^(n-2) spanning trees (Cayley's formula).

*Socratic prompt: "K3 (triangle) has how many spanning trees? Verify by both enumeration and the determinant formula. What about K5?"*

### Multigraphs and Weighted Variants

Kirchhoff's theorem naturally handles **multigraphs** (multiple edges between the same pair). If there are k edges between u and v, set A[u][v] = k. The Laplacian is still L = D - A.

For **weighted** graphs (counting the product of edge weights over all spanning trees), use the weighted Laplacian where L[i][j] = -w(i,j) for edges and L[i][i] = sum of all edge weights incident to i.

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| CODECHEF: MSTNUM | Direct application -- count spanning trees |
| SPOJ: OILCOMP | Count spanning trees of a derived graph |
| Codeforces: various | Often combined with DP or construction |

---

## 6. Prufer Code (Labeled Tree Encoding)

### Core Insight

A **Prufer sequence** is a unique encoding of a labeled tree on n vertices as a sequence of n-2 integers, each in [0, n-1]. This establishes a **bijection** between labeled trees and such sequences, immediately proving **Cayley's formula**: the number of labeled trees on n vertices is n^(n-2).

Key properties:
- Vertex i appears in the Prufer code exactly (degree(i) - 1) times
- Leaves (degree 1) never appear in the code
- The last two vertices remaining during encoding always include vertex n-1

*Socratic prompt: "If a vertex has degree 3 in the tree, how many times does it appear in the Prufer sequence? What about a leaf?"*

### Encoding: Tree to Prufer Code

Repeatedly remove the leaf with the smallest label, and record its neighbor.

```python
def tree_to_prufer(n: int, parent: list[int]) -> list[int]:
    """Encode a labeled tree as its Prufer sequence (naive O(n log n)).

    Args:
        n: number of vertices (0-indexed).
        parent: parent[i] = parent of vertex i in the rooted tree.
                parent[root] = -1 (root is typically n-1).

    Returns:
        Prufer sequence of length n-2.

    Time: O(n log n) using a heap. Space: O(n).
    """
    import heapq

    degree = [0] * n
    adj = [[] for _ in range(n)]
    for i in range(n):
        if parent[i] != -1:
            adj[i].append(parent[i])
            adj[parent[i]].append(i)
            degree[i] += 1
            degree[parent[i]] += 1

    # Use heap to always pick smallest leaf
    leaves = []
    for i in range(n):
        if degree[i] == 1:
            heapq.heappush(leaves, i)

    removed = [False] * n
    code = []

    for _ in range(n - 2):
        leaf = heapq.heappop(leaves)
        removed[leaf] = True

        # Find the neighbor that is not removed
        for neighbor in adj[leaf]:
            if not removed[neighbor]:
                code.append(neighbor)
                degree[neighbor] -= 1
                if degree[neighbor] == 1:
                    heapq.heappush(leaves, neighbor)
                break

    return code
```

### Linear-Time Encoding (O(n))

```python
def tree_to_prufer_linear(n: int, adj: list[list[int]]) -> list[int]:
    """Encode a labeled tree as its Prufer sequence in O(n) time.

    Uses a pointer technique: maintain a pointer that scans forward
    for the next leaf, skipping non-leaves.

    Args:
        n: number of vertices (0-indexed).
        adj: adjacency list of the tree.

    Returns:
        Prufer sequence of length n-2.

    Time: O(n). Space: O(n).
    """
    degree = [len(adj[v]) for v in range(n)]
    ptr = 0  # pointer to current candidate leaf
    # Find the first leaf (smallest label with degree 1)
    while degree[ptr] != 1:
        ptr += 1

    leaf = ptr
    code = []

    for _ in range(n - 2):
        # Find neighbor of leaf (the one with degree > 0 after removal)
        neighbor = -1
        for v in adj[leaf]:
            if degree[v] > 0:  # not yet fully removed
                neighbor = v
                break

        code.append(neighbor)
        degree[leaf] = 0
        degree[neighbor] -= 1

        if degree[neighbor] == 1 and neighbor < ptr:
            # New leaf is smaller than pointer -- process immediately
            leaf = neighbor
        else:
            # Advance pointer to next leaf
            ptr += 1
            while ptr < n and degree[ptr] != 1:
                ptr += 1
            leaf = ptr

    return code
```

### Decoding: Prufer Code to Tree

```python
def prufer_to_tree(code: list[int], n: int) -> list[tuple[int, int]]:
    """Decode a Prufer sequence into a labeled tree.

    Args:
        code: Prufer sequence of length n-2, values in [0, n-1].
        n: number of vertices.

    Returns:
        List of (u, v) edges forming the tree.

    Time: O(n). Space: O(n).
    """
    degree = [1] * n  # every vertex starts with degree 1
    for v in code:
        degree[v] += 1

    # Find first leaf (smallest vertex with degree 1)
    ptr = 0
    while degree[ptr] != 1:
        ptr += 1

    leaf = ptr
    edges = []

    for v in code:
        edges.append((leaf, v))
        degree[leaf] -= 1
        degree[v] -= 1

        if degree[v] == 1 and v < ptr:
            leaf = v
        else:
            ptr += 1
            while ptr < n and degree[ptr] != 1:
                ptr += 1
            leaf = ptr

    # Connect the last two remaining vertices (one is always n-1)
    edges.append((leaf, n - 1))
    return edges
```

### Complexity

| Operation | Naive | Linear |
|-----------|-------|--------|
| Encode (tree -> code) | O(n log n) | O(n) |
| Decode (code -> tree) | O(n log n) | O(n) |
| Space | O(n) | O(n) |

### Cayley's Formula and Applications

Since every Prufer sequence of length n-2 with values in [0, n-1] corresponds to exactly one labeled tree, and there are n^(n-2) such sequences:

> **Cayley's formula:** The number of labeled trees on n vertices is n^(n-2).

| n | n^(n-2) | Labeled trees |
|---|---------|---------------|
| 1 | 1 | 1 |
| 2 | 1 | 1 |
| 3 | 3 | 3 |
| 4 | 16 | 16 |
| 5 | 125 | 125 |

*Socratic prompt: "For n=3, enumerate all 3 labeled trees on vertices {0,1,2}. Write down the Prufer code for each. Do you get all possible length-1 sequences over {0,1,2}?"*

### Connecting Components Formula

Given a forest with k components of sizes s_1, s_2, ..., s_k, the number of ways to add exactly k-1 edges to make it connected is:

```
n^(k-2) * s_1 * s_2 * ... * s_k
```

where n is the total number of vertices.

```python
def ways_to_connect_components(component_sizes: list[int]) -> int:
    """Number of ways to add edges to connect k components into a tree.

    Based on the generalized Cayley formula.

    Args:
        component_sizes: list of sizes [s_1, s_2, ..., s_k].

    Returns:
        Number of ways to connect the components.

    Time: O(k). Space: O(1).
    """
    k = len(component_sizes)
    n = sum(component_sizes)
    if k <= 1:
        return 1

    result = pow(n, k - 2)
    for s in component_sizes:
        result *= s
    return result
```

*Socratic prompt: "If you have 3 isolated vertices, Cayley's formula gives 3^1 = 3 labeled trees. Verify this matches the 3 possible edges you can use to form a tree."*

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Cayley's formula verification problems | Encode/decode Prufer sequences |
| Codeforces: "Spanning Trees" variants | Use Prufer to count or enumerate |
| Count ways to connect forest | Generalized Cayley formula |

---

## Interview Tips

1. **MST is rare in interviews but critical in contests.** Most interview problems that involve MST are straightforward applications of Kruskal + DSU. The hard part is recognizing that a problem requires MST.

2. **Default to Kruskal + DSU.** It's simpler to code, handles sparse graphs well, and the DSU data structure is useful for many other problems (connected components, cycle detection in undirected graphs).

3. **Use Prim only when given a dense graph or adjacency matrix.** Min Cost to Connect All Points (1584) with coordinates is a classic case: the complete graph has O(V^2) edges, making Prim's O(V^2) approach ideal.

4. **Know the DSU template cold.** Union-Find appears in far more problems than MST alone: redundant connections, accounts merge, number of islands variants, etc.

5. **Second-best MST and Kirchhoff's theorem are competitive programming topics.** You are unlikely to see them in standard interviews, but knowing them demonstrates depth. LeetCode 1489 (Find Critical and Pseudo-Critical Edges in MST) is the closest interview-style problem.

6. **Cayley's formula (n^(n-2)) is a useful fact for combinatorics problems.** Even if you never code Prufer sequences, knowing this formula helps verify answers and reason about tree counting.

7. **Edge cases to always check:** disconnected graphs (MST doesn't exist), self-loops (ignore them), parallel edges (keep the lightest), single-vertex graph (MST cost = 0, no edges).

---

## Attribution

Content synthesized from cp-algorithms.com articles on [Prim's MST](https://cp-algorithms.com/graph/mst_prim.html), [Kruskal's MST](https://cp-algorithms.com/graph/mst_kruskal.html), [Kruskal's with DSU](https://cp-algorithms.com/graph/mst_kruskal_with_dsu.html), [Second-Best MST](https://cp-algorithms.com/graph/second_best_mst.html), [Kirchhoff's Theorem](https://cp-algorithms.com/graph/kirchhoff-theorem.html), and [Prufer Code](https://cp-algorithms.com/graph/pruefer_code.html). Algorithms and complexity analyses are from the original articles; code has been translated to Python with added commentary for the leetcode-teacher reference format.
