# LCA and RMQ: Lowest Common Ancestor & Range Minimum Queries

Five algorithms for computing Lowest Common Ancestor (LCA) on trees and the closely related Range Minimum Query (RMQ) problem, ranging from practical O(N log N)/O(1) methods to theoretically optimal O(N)/O(1) solutions. For the sparse table data structure used in several approaches, see `advanced-ds-fundamentals.md`. For general tree traversal patterns and DSU (used in Tarjan's offline LCA), see `advanced-tree-structures.md`. For graph traversal fundamentals (DFS/BFS), see `graph-algorithms.md`.

---

## Quick Reference Table

| Pattern | Recognition Signals | Key Problems | Section |
|---------|-------------------|--------------|---------|
| LCA via Euler Tour + RMQ | "lowest common ancestor", "distance between tree nodes", "path queries" | LCA of a Binary Tree (236), Distance Between Nodes | 1 |
| LCA with Binary Lifting | "kth ancestor", "jump up tree", "online LCA queries", "ancestor at distance k" | Kth Ancestor of a Tree Node (1483), LCA of Binary Tree (236) | 2 |
| Farach-Colton-Bender (O(1) LCA) | "O(1) LCA", "many LCA queries", "optimal preprocessing" | LCA with 10^6 queries, Tree distance batch | 3 |
| Linear RMQ via Cartesian Tree | "O(1) range minimum", "static array min queries", "offline RMQ" | Range Minimum Query (static), Sliding Window Maximum variant | 4 |
| Tarjan's Offline LCA | "all queries known in advance", "offline", "batch LCA", "union-find on tree" | Offline LCA batch, Tree Queries | 5 |

---

## LCA Approach Comparison

| Algorithm | Preprocessing | Query | Space | Online? | Best For |
|-----------|:------------:|:-----:|:-----:|:-------:|----------|
| Euler Tour + Sparse Table | O(N log N) | O(1) | O(N log N) | Yes | Most interview/contest problems |
| Binary Lifting | O(N log N) | O(log N) | O(N log N) | Yes | When you also need kth ancestor |
| Farach-Colton-Bender | O(N) | O(1) | O(N) | Yes | Theoretical optimality, huge N |
| Linear RMQ (Cartesian Tree) | O(N) | O(1) | O(N) | Yes | Static array RMQ, theoretical |
| Tarjan's Offline | O(N + Q) | O(1) amortized | O(N + Q) | **No** | All queries known upfront |

*Socratic prompt: "When would you prefer binary lifting over the Euler tour + sparse table approach, even though binary lifting has O(log N) queries instead of O(1)? What additional capability does binary lifting give you?"*

---

## Corner Cases

- **Root queries:** LCA(root, v) = root for all v. Ensure your implementation handles this without special-casing.
- **Same node:** LCA(v, v) = v. The Euler tour approach handles this naturally since first[v] = first[v] and the range is a single element.
- **Unrooted trees:** Pick any node as root. The LCA depends on the root choice, but distances between nodes do not.
- **Disconnected components:** LCA is undefined across components. Check connectivity first.
- **1-indexed vs 0-indexed:** Off-by-one errors in Euler tour indices are the most common bug. Be consistent.
- **Sparse table query direction:** When two indices give the same minimum height, return either -- both are the LCA.

---

## 1. LCA via Euler Tour + RMQ (Sparse Table)

### Core Insight

The LCA of two nodes u and v is the **shallowest node** visited between the first occurrences of u and v in an Euler tour. By recording depths during the DFS Euler tour, LCA reduces to a Range Minimum Query on the depth array. Using a sparse table for RMQ gives O(1) queries after O(N log N) preprocessing.

*Socratic prompt: "Why is the shallowest node between the first occurrences of u and v in the Euler tour guaranteed to be their LCA? What would happen if some node shallower than the LCA appeared between them?"*

### How It Works

1. **Euler tour:** DFS the tree, recording every node when you enter it and when you return from each child. For a tree with N nodes, the Euler tour has length 2N - 1.
2. **Record depths:** For each position in the Euler tour, store the depth of that node.
3. **First occurrence:** For each node v, store the index of its first appearance in the Euler tour.
4. **RMQ:** To find LCA(u, v), query the minimum depth in the range [first[u], first[v]] (or [first[v], first[u]] if first[v] < first[u]). The node at that minimum-depth position is the LCA.

### Template

```python
import math

class LCAEulerTour:
    """LCA using Euler Tour + Sparse Table RMQ.

    Preprocessing: O(N log N) time and space.
    Query: O(1) per LCA query.

    The tree is given as an adjacency list (0-indexed).
    """

    def __init__(self, adj, root=0):
        """Build Euler tour and sparse table from adjacency list.

        Args:
            adj: List of lists, adj[u] contains neighbors of u.
            root: Root node (default 0).
        """
        self.n = len(adj)
        self.euler = []       # Euler tour node sequence
        self.depth = []       # Depth at each Euler tour position
        self.first = [-1] * self.n  # First occurrence of node in Euler tour

        # Build Euler tour via iterative DFS
        self._build_euler_tour(adj, root)

        # Build sparse table on depths
        self._build_sparse_table()

    def _build_euler_tour(self, adj, root):
        """Construct Euler tour iteratively (avoids recursion limit)."""
        parent = [-1] * self.n
        depth_arr = [0] * self.n
        visited = [False] * self.n
        stack = [(root, False)]  # (node, returning)

        while stack:
            node, returning = stack.pop()
            self.euler.append(node)
            self.depth.append(depth_arr[node])

            if self.first[node] == -1:
                self.first[node] = len(self.euler) - 1

            if not returning:
                visited[node] = True
                # Push children in reverse order so leftmost is processed first
                children = []
                for child in adj[node]:
                    if child != parent[node] and not visited[child]:
                        parent[child] = node
                        depth_arr[child] = depth_arr[node] + 1
                        children.append(child)
                # Push return marker, then children (reversed for correct order)
                for child in reversed(children):
                    stack.append((node, True))   # return to parent after child
                    stack.append((child, False))  # process child

    def _build_sparse_table(self):
        """Build sparse table for range minimum query on depths."""
        m = len(self.euler)
        if m == 0:
            return
        self.log = [0] * (m + 1)
        for i in range(2, m + 1):
            self.log[i] = self.log[i // 2] + 1

        k = self.log[m] + 1
        # table[j][i] = index in euler with min depth in range [i, i + 2^j - 1]
        self.table = [[0] * m for _ in range(k)]

        for i in range(m):
            self.table[0][i] = i

        for j in range(1, k):
            for i in range(m - (1 << j) + 1):
                left = self.table[j - 1][i]
                right = self.table[j - 1][i + (1 << (j - 1))]
                self.table[j][i] = left if self.depth[left] <= self.depth[right] else right

    def _rmq(self, l, r):
        """Return index in Euler tour with minimum depth in [l, r]."""
        length = r - l + 1
        k = self.log[length]
        left = self.table[k][l]
        right = self.table[k][r - (1 << k) + 1]
        return left if self.depth[left] <= self.depth[right] else right

    def lca(self, u, v):
        """Return LCA of nodes u and v. O(1) per query.

        Args:
            u: First node (0-indexed).
            v: Second node (0-indexed).

        Returns:
            The LCA node.
        """
        l = self.first[u]
        r = self.first[v]
        if l > r:
            l, r = r, l
        idx = self._rmq(l, r)
        return self.euler[idx]

    def dist(self, u, v):
        """Return distance (number of edges) between u and v.

        dist(u, v) = depth[u] + depth[v] - 2 * depth[LCA(u, v)]
        """
        w = self.lca(u, v)
        return (self.depth[self.first[u]] + self.depth[self.first[v]]
                - 2 * self.depth[self.first[w]])
```

### Complexity

| Aspect | Value |
|--------|-------|
| Preprocessing Time | O(N log N) for sparse table construction |
| Preprocessing Space | O(N log N) for sparse table |
| Euler Tour Size | 2N - 1 entries |
| Query Time | O(1) per LCA query |
| Distance Query | O(1) using depth[u] + depth[v] - 2 * depth[LCA] |

*Socratic prompt: "The Euler tour has length 2N - 1. Why exactly 2N - 1, and not 2N or 3N? Can you prove this by induction on the tree structure?"*

### Practice Problems

| # | Problem | Difficulty | Key Technique |
|---|---------|-----------|---------------|
| 236 | Lowest Common Ancestor of a Binary Tree | Medium | Euler tour + RMQ or recursive |
| 235 | LCA of a Binary Search Tree | Medium | BST property shortcut |
| 1257 | Smallest Common Region | Medium | LCA on general tree |
| -- | Distance Queries (CSES) | Medium | LCA + depth for path distance |

---

## 2. LCA with Binary Lifting

### Core Insight

Precompute `up[v][j]` = the 2^j-th ancestor of node v. To find LCA(u, v), first bring both nodes to the same depth, then jump both upward in decreasing powers of 2 until they meet. Binary lifting also directly solves "kth ancestor" queries, which Euler tour + RMQ cannot.

*Socratic prompt: "How does the recurrence up[v][j] = up[up[v][j-1]][j-1] work? What does it mean to 'jump 2^j steps by making two jumps of 2^(j-1) steps'?"*

### How It Works

1. **DFS:** Compute depth, parent, and entry/exit times (tin/tout) for each node.
2. **Jump table:** `up[v][0] = parent[v]`. For j >= 1: `up[v][j] = up[up[v][j-1]][j-1]`.
3. **Ancestor check:** Node u is an ancestor of v iff `tin[u] <= tin[v]` and `tout[u] >= tout[v]` (interval containment).
4. **LCA query:** If u is ancestor of v (or vice versa), return the ancestor. Otherwise, jump u upward from the highest power of 2 down to 0, skipping jumps that would overshoot past the LCA.

### Template

```python
import math

class LCABinaryLifting:
    """LCA using Binary Lifting.

    Preprocessing: O(N log N) time and space.
    Query: O(log N) per LCA query.
    Also supports kth ancestor queries in O(log N).

    The tree is given as an adjacency list (0-indexed).
    """

    def __init__(self, adj, root=0):
        """Build binary lifting table from adjacency list.

        Args:
            adj: List of lists, adj[u] contains neighbors of u.
            root: Root node (default 0).
        """
        self.n = len(adj)
        self.LOG = max(1, self.n.bit_length())
        self.depth = [0] * self.n
        self.tin = [0] * self.n
        self.tout = [0] * self.n
        self.up = [[0] * self.n for _ in range(self.LOG)]
        self.timer = 0

        self._dfs(adj, root)
        self._build_table(adj)

    def _dfs(self, adj, root):
        """Iterative DFS to compute depth, tin, tout, and parent."""
        stack = [(root, -1, False)]
        while stack:
            node, par, returning = stack.pop()
            if returning:
                self.tout[node] = self.timer
                self.timer += 1
                continue
            self.up[0][node] = par if par != -1 else root
            self.tin[node] = self.timer
            self.timer += 1
            # Push return marker
            stack.append((node, par, True))
            for child in adj[node]:
                if child != par:
                    self.depth[child] = self.depth[node] + 1
                    stack.append((child, node, False))

    def _build_table(self, adj):
        """Build jump table: up[j][v] = 2^j-th ancestor of v."""
        for j in range(1, self.LOG):
            for v in range(self.n):
                self.up[j][v] = self.up[j - 1][self.up[j - 1][v]]

    def is_ancestor(self, u, v):
        """Check if u is an ancestor of v using DFS timestamps.

        u is ancestor of v iff the DFS interval of u contains that of v.
        """
        return self.tin[u] <= self.tin[v] and self.tout[u] >= self.tout[v]

    def lca(self, u, v):
        """Return LCA of nodes u and v. O(log N) per query.

        Args:
            u: First node (0-indexed).
            v: Second node (0-indexed).

        Returns:
            The LCA node.
        """
        if self.is_ancestor(u, v):
            return u
        if self.is_ancestor(v, u):
            return v

        # Jump u upward until just below the LCA
        for j in range(self.LOG - 1, -1, -1):
            if not self.is_ancestor(self.up[j][u], v):
                u = self.up[j][u]
        return self.up[0][u]

    def kth_ancestor(self, v, k):
        """Return the kth ancestor of node v, or -1 if it doesn't exist.

        Args:
            v: Node (0-indexed).
            k: Number of edges to go up.

        Returns:
            The kth ancestor, or -1 if k > depth[v].
        """
        if k > self.depth[v]:
            return -1
        for j in range(self.LOG):
            if k & (1 << j):
                v = self.up[j][v]
        return v

    def dist(self, u, v):
        """Return distance (number of edges) between u and v."""
        w = self.lca(u, v)
        return self.depth[u] + self.depth[v] - 2 * self.depth[w]
```

### Complexity

| Aspect | Value |
|--------|-------|
| Preprocessing Time | O(N log N) |
| Preprocessing Space | O(N log N) for the jump table |
| LCA Query Time | O(log N) |
| Kth Ancestor Query | O(log N) |
| Is-Ancestor Check | O(1) using tin/tout |

*Socratic prompt: "Binary lifting can answer 'kth ancestor' in O(log N). Can the Euler tour + RMQ approach answer kth ancestor queries at all? Why or why not?"*

### Practice Problems

| # | Problem | Difficulty | Key Technique |
|---|---------|-----------|---------------|
| 1483 | Kth Ancestor of a Tree Node | Hard | Binary lifting jump table |
| 236 | Lowest Common Ancestor of a Binary Tree | Medium | Binary lifting or recursive |
| 2846 | Minimum Edge Weight Equilibrium Queries | Hard | LCA + path frequency |
| 1740 | Find Distance in a Binary Tree | Medium | LCA + depth |
| -- | Company Queries I (CSES) | Easy | Kth ancestor via binary lifting |
| -- | Company Queries II (CSES) | Easy | LCA via binary lifting |

---

## 3. Farach-Colton-Bender Algorithm (Optimal O(N)/O(1) LCA)

### Core Insight

The Euler tour depth array has a special property: consecutive elements differ by exactly +1 or -1 (moving to a child increases depth by 1, returning to parent decreases by 1). This **plus-minus-1 constraint** means blocks of size K = 0.5 * log2(N) have at most sqrt(N) distinct "shapes," enabling O(N) total preprocessing for all block types.

*Socratic prompt: "Why do consecutive depths in the Euler tour differ by exactly +/-1? What tree operation causes each +1 and each -1?"*

### How It Works

1. **Euler tour + depth array:** Same as Section 1. Array length is 2N - 1.
2. **Block decomposition:** Divide the depth array into blocks of size K = floor(log2(2N-1) / 2).
3. **Block minimums:** Compute the minimum of each block. Store in array B.
4. **Sparse table on B:** Build a standard sparse table on B for inter-block queries. Since there are O(N/K) = O(N/log N) blocks, the sparse table uses O((N/log N) * log(N/log N)) = O(N) space.
5. **Block type classification:** Each block is characterized by its sequence of +1/-1 differences. With K-1 differences, there are 2^(K-1) = O(sqrt(N)) possible block types.
6. **Intra-block lookup tables:** For each distinct block type, precompute all O(K^2) range minimum answers. Total: O(sqrt(N) * K^2) = O(sqrt(N) * log^2(N)) = O(N) work.
7. **Query:** For a range [l, r]:
   - If l and r are in the same block: use the intra-block lookup.
   - Otherwise: combine the suffix of l's block, the sparse table answer for full blocks between them, and the prefix of r's block. All three are O(1) lookups.

### Template

```python
import math

class FarachColtonBenderLCA:
    """LCA with O(N) preprocessing and O(1) queries.

    Uses the Farach-Colton-Bender algorithm exploiting the +/-1 property
    of Euler tour depth arrays.

    Args:
        adj: Adjacency list (0-indexed).
        root: Root node (default 0).
    """

    def __init__(self, adj, root=0):
        self.n = len(adj)
        self.euler = []
        self.depths = []
        self.first = [-1] * self.n

        # Step 1: Build Euler tour
        self._build_euler_tour(adj, root)
        m = len(self.euler)

        # Step 2: Block decomposition
        self.block_size = max(1, int(math.log2(m + 1)) // 2) if m > 1 else 1
        self.num_blocks = (m + self.block_size - 1) // self.block_size

        # Step 3: Compute block minimums (index of min in each block)
        self.block_min = [0] * self.num_blocks
        self.block_min_idx = [0] * self.num_blocks
        for b in range(self.num_blocks):
            start = b * self.block_size
            end = min(start + self.block_size, m)
            best = start
            for i in range(start, end):
                if self.depths[i] < self.depths[best]:
                    best = i
            self.block_min[b] = self.depths[best]
            self.block_min_idx[b] = best

        # Step 4: Sparse table on block minimums
        self._build_sparse_table_blocks()

        # Step 5: Classify blocks and precompute intra-block RMQ
        self._precompute_blocks(m)

    def _build_euler_tour(self, adj, root):
        """Iterative DFS to build Euler tour."""
        parent = [-1] * self.n
        depth_arr = [0] * self.n
        visited = [False] * self.n
        stack = [(root, False)]

        while stack:
            node, returning = stack.pop()
            self.euler.append(node)
            self.depths.append(depth_arr[node])
            if self.first[node] == -1:
                self.first[node] = len(self.euler) - 1
            if not returning:
                visited[node] = True
                children = []
                for child in adj[node]:
                    if child != parent[node] and not visited[child]:
                        parent[child] = node
                        depth_arr[child] = depth_arr[node] + 1
                        children.append(child)
                for child in reversed(children):
                    stack.append((node, True))
                    stack.append((child, False))

    def _build_sparse_table_blocks(self):
        """Sparse table on block minimum values for inter-block queries."""
        nb = self.num_blocks
        if nb == 0:
            self.sp = []
            self.sp_log = []
            return
        self.sp_log = [0] * (nb + 1)
        for i in range(2, nb + 1):
            self.sp_log[i] = self.sp_log[i // 2] + 1
        k = self.sp_log[nb] + 1
        # sp[j][i] = block index with smallest min in blocks [i, i + 2^j - 1]
        self.sp = [[0] * nb for _ in range(k)]
        for i in range(nb):
            self.sp[0][i] = i
        for j in range(1, k):
            for i in range(nb - (1 << j) + 1):
                l = self.sp[j - 1][i]
                r = self.sp[j - 1][i + (1 << (j - 1))]
                self.sp[j][i] = l if self.block_min[l] <= self.block_min[r] else r

    def _block_type(self, b, m):
        """Compute the type (bitmask) of block b based on +/-1 differences."""
        start = b * self.block_size
        end = min(start + self.block_size, m)
        mask = 0
        for i in range(start + 1, end):
            if self.depths[i] > self.depths[i - 1]:  # +1 step
                mask |= (1 << (i - start - 1))
        return mask

    def _precompute_blocks(self, m):
        """Precompute intra-block RMQ for each distinct block type."""
        bs = self.block_size
        num_types = 1 << (bs - 1) if bs > 1 else 1

        # For each block, record its type
        self.btypes = [0] * self.num_blocks
        for b in range(self.num_blocks):
            self.btypes[b] = self._block_type(b, m)

        # Precompute RMQ for each type: lookup[type][i][j] = position of min
        # in relative coordinates within a block of that type, range [i, j]
        self.lookup = {}
        for b in range(self.num_blocks):
            t = self.btypes[b]
            if t in self.lookup:
                continue
            # Build the depth pattern for this type
            pattern = [0] * bs
            for i in range(1, bs):
                if t & (1 << (i - 1)):
                    pattern[i] = pattern[i - 1] + 1
                else:
                    pattern[i] = pattern[i - 1] - 1

            # Precompute all range minimums within this block type
            table = [[0] * bs for _ in range(bs)]
            for i in range(bs):
                table[i][i] = i
                for j in range(i + 1, bs):
                    table[i][j] = table[i][j - 1]
                    if pattern[j] < pattern[table[i][j]]:
                        table[i][j] = j
            self.lookup[t] = table

    def _query_block(self, b, l, r):
        """Query min index within block b, relative positions [l, r].

        Returns absolute index in the Euler tour.
        """
        t = self.btypes[b]
        rel_pos = self.lookup[t][l][r]
        return b * self.block_size + rel_pos

    def _query_blocks_sparse(self, bl, br):
        """Query which block in [bl, br] has the overall minimum.

        Returns absolute index of the minimum in the Euler tour.
        """
        if bl > br:
            return -1
        length = br - bl + 1
        k = self.sp_log[length]
        l = self.sp[k][bl]
        r = self.sp[k][br - (1 << k) + 1]
        best_block = l if self.block_min[l] <= self.block_min[r] else r
        return self.block_min_idx[best_block]

    def _rmq(self, l, r):
        """Range minimum query on depths[l..r], returns index."""
        m = len(self.euler)
        bl = l // self.block_size
        br = r // self.block_size

        if bl == br:
            # Same block
            return self._query_block(bl, l - bl * self.block_size,
                                     r - bl * self.block_size)

        # Suffix of l's block
        best = self._query_block(bl, l - bl * self.block_size,
                                  min(self.block_size - 1,
                                      m - 1 - bl * self.block_size))

        # Prefix of r's block
        cand = self._query_block(br, 0, r - br * self.block_size)
        if self.depths[cand] < self.depths[best]:
            best = cand

        # Full blocks between bl+1 and br-1
        if bl + 1 <= br - 1:
            cand = self._query_blocks_sparse(bl + 1, br - 1)
            if cand != -1 and self.depths[cand] < self.depths[best]:
                best = cand

        return best

    def lca(self, u, v):
        """Return LCA of u and v in O(1).

        Args:
            u: First node (0-indexed).
            v: Second node (0-indexed).

        Returns:
            The LCA node.
        """
        l = self.first[u]
        r = self.first[v]
        if l > r:
            l, r = r, l
        return self.euler[self._rmq(l, r)]
```

### Complexity

| Aspect | Value |
|--------|-------|
| Preprocessing Time | O(N) |
| Preprocessing Space | O(N) |
| Query Time | O(1) |
| Block Size | K = floor(log(N) / 2) |
| Number of Block Types | O(sqrt(N)) due to +/-1 constraint |

*Socratic prompt: "The key insight is that there are only O(sqrt(N)) distinct block types. Why? What would happen if consecutive depths could differ by more than 1 -- how many block types would there be?"*

### When to Use

In practice, the Euler tour + sparse table approach (Section 1) is almost always sufficient and simpler to implement. Farach-Colton-Bender is primarily important for:
- Achieving theoretically optimal O(N) preprocessing
- Problems with extremely large N where the log factor in sparse table matters
- As a building block inside other optimal algorithms (e.g., suffix trees)

### Practice Problems

| # | Problem | Difficulty | Key Technique |
|---|---------|-----------|---------------|
| 236 | Lowest Common Ancestor of a Binary Tree | Medium | Any LCA method |
| -- | LCA Queries (CSES) | Medium | Optimal LCA for large N |
| -- | Distance Queries (CSES) | Medium | LCA + depth |

---

## 4. Linear RMQ via Cartesian Trees

### Core Insight

Any static RMQ problem can be solved optimally in O(N) preprocessing / O(1) query by reducing it to an LCA problem on a **Cartesian tree**, then applying the Farach-Colton-Bender algorithm. A Cartesian tree is a binary tree where (1) the root is the minimum element, (2) the left subtree is the Cartesian tree of elements to the left, and (3) similarly for the right. The minimum of any range [l, r] is the LCA of nodes l and r in the Cartesian tree.

*Socratic prompt: "Why does the LCA in a Cartesian tree correspond to the range minimum? What property of the Cartesian tree's in-order traversal guarantees this?"*

### How It Works

1. **Build Cartesian tree** from array A in O(N) using a stack.
2. **Reduce to LCA:** RMQ(l, r) = LCA(node_l, node_r) in the Cartesian tree.
3. **Apply FCB:** The Cartesian tree's Euler tour depth array satisfies the +/-1 property, so Farach-Colton-Bender gives O(N) / O(1).

### Template: Cartesian Tree Construction

```python
def build_cartesian_tree(arr):
    """Build a Cartesian tree from array arr.

    The Cartesian tree satisfies:
    - Heap property: parent value <= child values (min-heap).
    - BST property on indices: in-order traversal gives original array order.

    Returns:
        (root, left, right, parent) where left[i], right[i], parent[i]
        are the left child, right child, and parent of node i (-1 if none).

    Time: O(N) using a stack-based algorithm.
    """
    n = len(arr)
    left = [-1] * n
    right = [-1] * n
    parent = [-1] * n
    stack = []  # Stack of indices, maintaining increasing arr values

    for i in range(n):
        last_popped = -1
        while stack and arr[stack[-1]] >= arr[i]:
            last_popped = stack.pop()
        if last_popped != -1:
            left[i] = last_popped       # Last popped becomes left child
            parent[last_popped] = i
        if stack:
            right[stack[-1]] = i         # New node becomes right child of stack top
            parent[i] = stack[-1]
        stack.append(i)

    root = stack[0]  # Bottom of stack is the root (global minimum)
    return root, left, right, parent
```

### Template: Full Linear RMQ

```python
class LinearRMQ:
    """O(N) preprocessing, O(1) query Range Minimum Query.

    Combines Cartesian tree reduction with Farach-Colton-Bender LCA.
    For most practical purposes, a sparse table RMQ (O(N log N) / O(1))
    is simpler and fast enough.
    """

    def __init__(self, arr):
        """Build linear RMQ structure.

        Args:
            arr: List of comparable elements.
        """
        self.arr = arr
        n = len(arr)
        if n == 0:
            return

        # Build Cartesian tree
        root, left, right, par = build_cartesian_tree(arr)

        # Build adjacency list from Cartesian tree
        adj = [[] for _ in range(n)]
        for i in range(n):
            if left[i] != -1:
                adj[i].append(left[i])
                adj[left[i]].append(i)
            if right[i] != -1:
                adj[i].append(right[i])
                adj[right[i]].append(i)

        # Use Farach-Colton-Bender LCA on the Cartesian tree
        self._lca = FarachColtonBenderLCA(adj, root)

    def query(self, l, r):
        """Return index of minimum element in arr[l..r].

        Args:
            l: Left bound (inclusive, 0-indexed).
            r: Right bound (inclusive, 0-indexed).

        Returns:
            Index i in [l, r] such that arr[i] is minimized.
        """
        return self._lca.lca(l, r)
```

### Practical Alternative: Sparse Table RMQ

For most interview and contest settings, a simple sparse table gives O(N log N) / O(1) and is far easier to implement.

```python
class SparseTableRMQ:
    """O(N log N) preprocessing, O(1) query RMQ.

    Simpler and usually preferred over linear RMQ in practice.
    See advanced-ds-fundamentals.md for full treatment.
    """

    def __init__(self, arr):
        """Build sparse table for range minimum index queries."""
        n = len(arr)
        self.arr = arr
        self.log = [0] * (n + 1)
        for i in range(2, n + 1):
            self.log[i] = self.log[i // 2] + 1
        k = self.log[n] + 1 if n > 0 else 1
        self.table = [[0] * n for _ in range(k)]
        for i in range(n):
            self.table[0][i] = i
        for j in range(1, k):
            for i in range(n - (1 << j) + 1):
                l = self.table[j - 1][i]
                r = self.table[j - 1][i + (1 << (j - 1))]
                self.table[j][i] = l if arr[l] <= arr[r] else r

    def query(self, l, r):
        """Return index of minimum element in arr[l..r]."""
        length = r - l + 1
        k = self.log[length]
        left = self.table[k][l]
        right = self.table[k][r - (1 << k) + 1]
        return left if self.arr[left] <= self.arr[right] else right
```

### Complexity

| Aspect | Linear RMQ | Sparse Table RMQ |
|--------|-----------|-----------------|
| Preprocessing Time | O(N) | O(N log N) |
| Preprocessing Space | O(N) | O(N log N) |
| Query Time | O(1) | O(1) |
| Implementation Complexity | High | Low |
| Practical Speed | Slower (large constants) | Faster in practice |

*Socratic prompt: "A Cartesian tree is built with a stack in O(N). Why does each element get pushed and popped at most once? What invariant does the stack maintain?"*

### Practice Problems

| # | Problem | Difficulty | Key Technique |
|---|---------|-----------|---------------|
| 239 | Sliding Window Maximum | Hard | Monotonic deque (related structure) |
| 2104 | Sum of Subarray Ranges | Medium | Stack-based min/max (Cartesian tree idea) |
| -- | Static Range Minimum Queries (CSES) | Easy | Sparse table or linear RMQ |
| -- | Range Minimum Query (SPOJ RMQSQ) | Easy | Direct RMQ |

---

## 5. Tarjan's Offline LCA

### Core Insight

If all LCA queries are known in advance, process them during a single DFS using **Union-Find (DSU)**. When DFS finishes processing a subtree rooted at v, union it with v's parent. At that point, any query (v, w) where w has already been visited can be answered: the LCA is the current representative of w's set (which is the deepest ancestor of w that has finished processing its subtree and includes v).

*Socratic prompt: "Why does find(w) give the correct LCA when we process query (v, w) during DFS at node v? What does the DSU representative correspond to in terms of the DFS state?"*

### How It Works

1. Group all queries by both endpoints: for query (u, v), store it at both u and v.
2. DFS from the root. When visiting node v:
   - Mark v as visited.
   - Recursively process all children. After returning from child c, union c's set with v's set, and set the representative's ancestor to v.
   - For each query (v, w): if w is already visited, the answer is `ancestor[find(w)]`.

### Template

```python
class TarjanOfflineLCA:
    """Tarjan's offline LCA using DFS + Union-Find.

    All queries must be known before processing.
    Time: O((N + Q) * alpha(N)) which is effectively O(N + Q).
    Space: O(N + Q).
    """

    def __init__(self, adj, root=0):
        """Initialize with tree adjacency list.

        Args:
            adj: List of lists, adj[u] contains neighbors of u.
            root: Root node (default 0).
        """
        self.adj = adj
        self.root = root
        self.n = len(adj)

    def solve(self, queries):
        """Answer all LCA queries offline.

        Args:
            queries: List of (u, v) pairs.

        Returns:
            List of LCA answers, one per query.
        """
        n = self.n
        # DSU with path compression and union by rank
        parent = list(range(n))
        rank = [0] * n
        ancestor = list(range(n))  # ancestor[find(v)] = current LCA candidate

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path splitting
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1

        # Group queries by node
        query_map = [[] for _ in range(n)]
        for idx, (u, v) in enumerate(queries):
            query_map[u].append((v, idx))
            query_map[v].append((u, idx))

        visited = [False] * n
        answers = [0] * len(queries)

        # Iterative DFS with post-order processing
        # Use a stack storing (node, parent, child_index, phase)
        # phase 0 = pre-visit, phase 1 = post-child-union, phase 2 = answer queries
        stack = [(self.root, -1, 0)]
        children_of = [[] for _ in range(n)]

        # First pass: compute children (remove parent edges)
        par = [-1] * n
        visit_order = []
        s = [self.root]
        seen = [False] * n
        seen[self.root] = True
        while s:
            node = s.pop()
            visit_order.append(node)
            for child in self.adj[node]:
                if not seen[child]:
                    seen[child] = True
                    par[child] = node
                    children_of[node].append(child)
                    s.append(child)

        # Process in reverse DFS order (post-order)
        # We need true post-order: process children before parent
        post_order = []
        s = [(self.root, False)]
        while s:
            node, processed = s.pop()
            if processed:
                post_order.append(node)
                continue
            s.append((node, True))
            for child in children_of[node]:
                s.append((child, False))

        for node in post_order:
            visited[node] = True
            # Union with parent and set ancestor
            for child in children_of[node]:
                union(node, child)
                ancestor[find(node)] = node

            # Answer queries where the other endpoint is already visited
            for other, idx in query_map[node]:
                if visited[other]:
                    answers[idx] = ancestor[find(other)]

        return answers
```

### Complexity

| Aspect | Value |
|--------|-------|
| Time | O((N + Q) * alpha(N)) ~ O(N + Q) |
| Space | O(N + Q) |
| Online? | No -- all queries must be known upfront |
| DFS Passes | 1 |

*Socratic prompt: "Tarjan's algorithm is offline -- it needs all queries upfront. When would this be acceptable in a contest? Can you think of a problem where queries arrive one at a time, making this approach impossible?"*

### When to Prefer Tarjan's

- All queries are known in advance (batch processing).
- You want the simplest optimal-time algorithm (no sparse tables, no block decomposition).
- Memory is tight: O(N + Q) vs O(N log N) for binary lifting / sparse table.
- You already have a DSU implementation available.

### Practice Problems

| # | Problem | Difficulty | Key Technique |
|---|---------|-----------|---------------|
| 236 | Lowest Common Ancestor of a Binary Tree | Medium | Any LCA method (online ok here) |
| -- | LCA Queries (CSES) | Medium | Offline with Tarjan's |
| -- | Distance Queries (CSES) | Medium | Offline LCA + depth |

---

## Decision Flowchart

1. **Do you need kth ancestor queries too?** -> **Binary Lifting** (Section 2)
2. **Are all queries known upfront and memory is tight?** -> **Tarjan's Offline** (Section 5)
3. **Is N extremely large (> 10^7) and preprocessing must be O(N)?** -> **Farach-Colton-Bender** (Section 3)
4. **General LCA with O(1) queries?** -> **Euler Tour + Sparse Table** (Section 1) -- best default choice
5. **Static array RMQ (not on a tree)?** -> **Sparse Table RMQ** (Section 4) for practice; **Linear RMQ** for theoretical optimality

*Socratic prompt: "You're given a tree with 10^5 nodes and 10^6 online LCA queries. Which algorithm would you choose and why? What if the tree had 10^8 nodes instead?"*

---

## Practice Questions

### Essential

| # | Problem | Difficulty | Key Technique |
|---|---------|-----------|---------------|
| 236 | Lowest Common Ancestor of a Binary Tree | Medium | Euler tour + RMQ or binary lifting |
| 235 | LCA of a Binary Search Tree | Medium | BST property (no preprocessing needed) |
| 1483 | Kth Ancestor of a Tree Node | Hard | Binary lifting |
| 239 | Sliding Window Maximum | Hard | Monotonic deque / RMQ |

### Recommended

| # | Problem | Difficulty | Key Technique |
|---|---------|-----------|---------------|
| 1257 | Smallest Common Region | Medium | LCA on general tree |
| 1740 | Find Distance in a Binary Tree | Medium | LCA + depth |
| 2846 | Minimum Edge Weight Equilibrium Queries | Hard | LCA + path queries |
| 2104 | Sum of Subarray Ranges | Medium | Cartesian tree / stack-based |
| 1676 | LCA of a Binary Tree IV | Medium | Batch LCA |

### Competitive Programming

| Problem | Source | Key Technique |
|---------|--------|---------------|
| Distance Queries | CSES | LCA + depth for path distance |
| Company Queries I | CSES | Kth ancestor (binary lifting) |
| Company Queries II | CSES | LCA (binary lifting or Euler tour) |
| RMQSQ | SPOJ | Static RMQ |

---

## Attribution

Content synthesized from cp-algorithms.com articles on [LCA via Euler Tour + RMQ](https://cp-algorithms.com/graph/lca.html), [LCA with Binary Lifting](https://cp-algorithms.com/graph/lca_binary_lifting.html), [Farach-Colton-Bender Algorithm](https://cp-algorithms.com/graph/lca_farachcoltonbender.html), [Linear RMQ](https://cp-algorithms.com/graph/rmq_linear.html), and [Tarjan's Offline LCA](https://cp-algorithms.com/graph/lca_tarjan.html). Algorithms and complexity analyses are from the original articles; code has been translated to Python with added commentary for the leetcode-teacher reference format.
