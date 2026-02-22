# Reference Routing Table

The single source of truth for technique → reference file mappings. Step 2B loads this file to resolve each identified technique to its reference. Steps 4-6 also reference this file for mid-session technique discovery.

## Technique → Reference Mapping

| Technique Domain | Reference |
|-----------------|-----------|
| **Frameworks & Meta-Level** | |
| Enumeration principle, recursion as tree traversal, interview tips, complexity analysis | `references/frameworks/algorithm-frameworks.md` |
| Problem pattern families (frequency, sliding window, prefix sum, etc.) | `references/frameworks/problem-patterns.md` |
| Socratic question banks by stage | `references/frameworks/socratic-questions.md` |
| Learning principles (Make It Stick) | `references/frameworks/learning-principles.md` |
| **Data Structures** | |
| Data structure internals (hash table, heap, trie, linked list, tree, graph) | `references/data-structures/data-structure-fundamentals.md` |
| Min-stack, min-queue, sparse table, static RMQ | `references/data-structures/advanced-ds-fundamentals.md` |
| DSU/Union-Find, Fenwick tree/BIT, segment tree, treap, sqrt decomposition | `references/data-structures/advanced-tree-structures.md` |
| **Techniques (DS-Specific Patterns)** | |
| Two pointers, prefix sums, difference arrays, 2D traversal | `references/techniques/array-techniques.md` |
| Linked list pointer mechanics, fast/slow, dummy node | `references/techniques/linked-list-techniques.md` |
| Monotonic stack/queue, expression evaluation | `references/techniques/stack-queue-monotonic.md` |
| Matrix rotation, spiral traversal, search in 2D | `references/techniques/matrix-techniques.md` |
| String manipulation, palindromes, anagrams | `references/techniques/string-techniques.md` |
| Hashing, suffix array/automaton, Aho-Corasick, Manacher, Lyndon | `references/techniques/string-algorithms-advanced.md` |
| **Algorithms (Paradigms)** | |
| Sliding window | `references/algorithms/sliding-window.md` |
| Backtracking, subsets/combinations/permutations | `references/algorithms/backtracking.md` |
| BFS framework, bidirectional BFS | `references/algorithms/bfs-framework.md` |
| DP framework (meta-level intro) | `references/algorithms/dp-framework.md` |
| Divide and conquer | `references/algorithms/divide-and-conquer.md` |
| State machine framework | `references/algorithms/state-machine.md` |
| Binary search framework | `references/algorithms/binary-search-framework.md` |
| Backtracking variants, grid DFS (islands), state-space BFS (puzzles) | `references/algorithms/brute-force-search.md` |
| Sorting algorithms | `references/algorithms/sorting-algorithms.md` |
| **DP Families** | |
| DP core (framework principles, subsequence/string DP, Floyd-Warshall) | `references/algorithms/dynamic-programming-core.md` |
| Knapsack (0-1, complete, bounded) | `references/algorithms/knapsack.md` |
| Grid & path DP | `references/algorithms/grid-dp.md` |
| Game theory DP | `references/algorithms/game-theory-dp.md` |
| Interval DP + egg drop | `references/algorithms/interval-dp.md` |
| House robber & stock problems (state machine DP) | `references/algorithms/stock-problems.md` |
| Subsequence DP (quick-reference) | `references/algorithms/subsequence-dp.md` |
| D&C DP, Knuth optimization, bitmask DP, O(N log N) LIS, bounded knapsack | `references/algorithms/dynamic-programming-advanced.md` |
| **Greedy** | |
| Greedy framework, proof techniques, greedy vs DP checklist | `references/algorithms/greedy-algorithms.md` |
| Interval scheduling + meeting rooms | `references/algorithms/interval-scheduling.md` |
| Jump game | `references/algorithms/jump-game.md` |
| Gas station + video stitching | `references/algorithms/gas-station.md` |
| **Misc Algorithm Patterns** | |
| N-Sum generalized | `references/algorithms/n-sum.md` |
| LRU Cache, LFU Cache, Random Set O(1) | `references/algorithms/lru-lfu-cache.md` |
| Find median from stream, remove duplicate letters, exam room, bipartite | `references/algorithms/advanced-patterns.md` |
| **Graphs** | |
| Graph fundamentals (BFS, DFS, topological sort, basic algorithms) | `references/graphs/graph-algorithms.md` |
| Dijkstra's algorithm | `references/graphs/dijkstra.md` |
| BFS/DFS details, bridges, articulation points, SCC, strong orientation | `references/graphs/graph-traversal-advanced.md` |
| Dijkstra variants, Bellman-Ford, 0-1 BFS, Floyd-Warshall, APSP | `references/graphs/graph-shortest-paths-advanced.md` |
| MST variants, Kirchhoff theorem, Prüfer code | `references/graphs/graph-mst-trees.md` |
| Cycle finding, negative cycles, Eulerian path/circuit | `references/graphs/graph-cycles-euler.md` |
| LCA (binary lifting, Tarjan, Farach-Colton-Bender), RMQ | `references/graphs/graph-lca-rmq.md` |
| Max flow (Dinic, Edmonds-Karp, push-relabel), min-cost flow | `references/graphs/graph-network-flow.md` |
| Bipartite check, maximum matching (Kuhn), assignment (Hungarian) | `references/graphs/graph-bipartite-matching.md` |
| Topological sort (detailed), 2-SAT, HLD, edge/vertex connectivity | `references/graphs/graph-special-topics.md` |
| **Math** | |
| Math techniques (number theory basics, modular arithmetic) | `references/math/math-techniques.md` |
| Number theory advanced | `references/math/number-theory-advanced.md` |
| Modular arithmetic advanced | `references/math/modular-arithmetic-advanced.md` |
| Combinatorics | `references/math/combinatorics.md` |
| Linear algebra (Gaussian elimination, determinants, rank) | `references/math/linear-algebra-gauss.md` |
| Probability & randomized algorithms | `references/math/probability-random.md` |
| Geometry | `references/math/geometry.md` |
| **Numeric** | |
| Bit manipulation | `references/numeric/bit-manipulation.md` |
| Bit representations | `references/numeric/bit-representations.md` |
| Numerical search (ternary search, Newton's method) | `references/numeric/numerical-search.md` |
| Numerical integration | `references/numeric/numerical-integration.md` |
| **Problems** | |
| Classic interview problems (trapping rain water, ugly numbers, intervals) | `references/problems/classic-interview-problems.md` |
| Brain teasers & games | `references/problems/brain-teasers-games.md` |
| **ML** | |
| ML implementations (optimizers, layers, losses) | `references/ml/ml-implementations.md` |
| ML special handling (Socratic questions, numerical walkthroughs) | `references/ml/ml-special-handling.md` |
