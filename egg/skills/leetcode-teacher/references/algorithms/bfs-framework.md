# The BFS Framework

Breadth-first search for shortest paths in unweighted graphs, plus bidirectional BFS optimization.

**Prerequisites:** Queue (FIFO) data structure, graph representation. See `references/data-structures/data-structure-fundamentals.md` for graph representations.

---

## Why BFS Finds Shortest Paths

BFS explores all nodes at distance `d` before any node at distance `d+1`. This layer-by-layer expansion guarantees that the first time you reach a node is via the shortest path (in unweighted graphs).

## BFS vs Dijkstra

| Property | BFS | Dijkstra |
|----------|-----|----------|
| Data structure | Queue (FIFO) | Priority queue (min-heap) |
| Edge weights | All equal (unweighted) | Non-negative, possibly different |
| Visited tracking | Boolean `visited` set | `dist_to[node]` array (relax edges) |
| When to mark visited | When enqueuing | When dequeuing (popping from heap) |
| Time complexity | O(V + E) | O((V + E) log V) |

**Key insight:** Dijkstra is BFS generalized to weighted graphs. The priority queue ensures we always process the closest unvisited node, just as BFS's FIFO queue ensures we process by layer.

## Three Graph Traversal Modes

Labuladong identifies three distinct traversal patterns for graphs, each using `visited` differently:

**Mode 1: Traverse Nodes** — visit each node once. Standard DFS/BFS. Use `visited = set()`.

**Mode 2: Traverse Edges** — visit each edge once. Track `(from, to)` pairs in visited. Needed when parallel edges or edge-specific logic matters.

**Mode 3: Traverse Paths** — track the current path with `on_path = set()`, adding nodes on entry and removing on exit (backtracking). Essential for **cycle detection in directed graphs** (e.g., course schedule, topological sort).

The critical distinction: `visited` prevents revisiting nodes globally, while `on_path` tracks only the current recursion stack. A node can be `visited` but not `on_path` (explored via a different branch). This is why directed cycle detection needs `on_path` — `visited` alone cannot distinguish "already explored elsewhere" from "currently in a cycle."

**Complexity:** Graph traversal is O(V + E), not just O(V), because every edge is examined.

For code templates and detailed examples of all three modes, see `references/data-structures/data-structure-fundamentals.md`.

---

## Bidirectional BFS

An optimization for standard BFS when you know both the start and target states.

### Core Idea

Instead of searching from start to target (expanding a potentially huge frontier), search from **both ends simultaneously** and stop when the frontiers meet. This reduces the search space from O(b^d) to O(b^(d/2)), where `b` is the branching factor and `d` is the distance.

### Template

```python
from collections import deque

def bidirectional_bfs(start, target, get_neighbors):
    if start == target:
        return 0

    # Two frontiers as sets (for O(1) membership check)
    front = {start}
    back = {target}
    visited = {start, target}
    steps = 0

    while front and back:
        steps += 1
        # Always expand the SMALLER frontier for balance
        if len(front) > len(back):
            front, back = back, front

        next_front = set()
        for node in front:
            for neighbor in get_neighbors(node):
                if neighbor in back:
                    return steps          # Frontiers met!
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_front.add(neighbor)
        front = next_front

    return -1  # No path exists
```

### When to Use

- You must know the **target state** in advance (not just "find any goal")
- The branching factor is large (e.g., Word Ladder with 26 possible letter changes per position)
- Standard BFS times out due to exponential frontier growth

**Word Ladder example:** Changing "hit" → "cog" with dictionary lookups. Standard BFS explores O(26^d) states from one end. Bidirectional BFS explores O(26^(d/2)) from each end — dramatically faster.

*Socratic prompt: "If the shortest path is 6 steps and each node has 10 neighbors, how many nodes does standard BFS explore vs bidirectional BFS? Which grows faster?"*

## See Also

- `references/graphs/graph-algorithms.md` — Full graph algorithm families
- `references/graphs/dijkstra.md` — Weighted shortest paths
- `references/algorithms/brute-force-search.md` — State-space BFS applications (sliding puzzle, Open the Lock)

---

## Attribution

Extracted from the BFS Framework and Bidirectional BFS sections of `algorithm-frameworks.md`, inspired by labuladong's algorithmic guides (labuladong.online).
