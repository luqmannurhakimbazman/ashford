# Dijkstra's Algorithm

Shortest path in weighted graphs with non-negative edges. BFS generalized to weighted graphs.

**Prerequisites:** BFS, priority queues/heaps. See `references/algorithms/bfs-framework.md` for BFS vs Dijkstra comparison.

---

## Key Insight

BFS finds shortest paths when all edges have equal weight. Dijkstra generalizes BFS to non-negative weighted graphs by replacing the FIFO queue with a priority queue (min-heap). At each step, process the unvisited node with the smallest known distance.

## Template

```python
import heapq

def dijkstra(graph, start):
    # graph: adjacency list, graph[u] = [(v, weight), ...]
    dist = {start: 0}
    heap = [(0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float('inf')):
            continue  # Stale entry, skip

        for v, w in graph[u]:
            new_dist = d + w
            if new_dist < dist.get(v, float('inf')):
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    return dist
```

## When NOT to Use Dijkstra

- **Negative edge weights:** Use Bellman-Ford instead. Dijkstra's greedy assumption (closest unvisited node has final distance) breaks with negative edges.
- **Unweighted graphs:** Use plain BFS — simpler and faster (O(V+E) vs O((V+E) log V)).

## Common Applications

| Problem | Setup |
|---------|-------|
| Network Delay Time (743) | Dijkstra from source, answer = max distance |
| Cheapest Flights Within K Stops (787) | Modified Dijkstra/BFS with stop count in state |
| Path with Maximum Probability (1514) | Max-heap Dijkstra (negate log-probabilities or use max-heap) |
| Swim in Rising Water (778) | Dijkstra where edge weight = max elevation on path |

## See Also

- `references/graphs/graph-algorithms.md` — Full graph algorithm families
- `references/graphs/graph-shortest-paths-advanced.md` — Bellman-Ford, 0-1 BFS, Floyd-Warshall, APSP
- `references/algorithms/bfs-framework.md` — BFS for unweighted graphs

---

## Attribution

Extracted from the Dijkstra's Algorithm section of `advanced-patterns.md`, inspired by labuladong's algorithmic guides (labuladong.online).
