# Domain-Topic Mapping

Maps company engineering domains to the DSA topics and problem patterns most commonly tested in interviews for those domains. Use this when JD signals alone are insufficient — the company's engineering domain provides strong signal about what they test.

Pattern names align with leetcode-teacher's `references/frameworks/problem-patterns.md` taxonomy.

---

## Web & Application Engineering

### Frontend / Full-Stack Web

| Topic | Priority | Why Tested | Key Patterns |
|-------|----------|-----------|--------------|
| Hash Table | High | State management, caching, deduplication | Hash Table |
| DFS/BFS | High | DOM traversal, component trees, dependency resolution | DFS/BFS |
| Sliding Window | Medium | Virtualized lists, viewport rendering, streaming data | Sliding Window |
| Dynamic Programming | Medium | Layout optimization, memoization patterns | Dynamic Programming |
| Two Pointers | Low | String manipulation, diff algorithms | Two Pointers |

### Backend / API Engineering

| Topic | Priority | Why Tested | Key Patterns |
|-------|----------|-----------|--------------|
| Hash Table | High | Caching, routing, rate limiting, deduplication | Hash Table |
| DFS/BFS | High | Service dependency graphs, topological ordering | DFS/BFS |
| Heap / Priority Queue | High | Request scheduling, load balancing, priority queues | Heap / Priority Queue |
| Greedy | Medium | Resource allocation, connection pooling, scheduling | Greedy |
| Binary Search | Medium | Configuration lookup, threshold detection | Binary Search |
| Sliding Window | Medium | Rate limiting windows, log analysis | Sliding Window |

### Mobile Engineering

| Topic | Priority | Why Tested | Key Patterns |
|-------|----------|-----------|--------------|
| Hash Table | High | Caching, local storage efficiency | Hash Table |
| DFS/BFS | High | View hierarchy traversal, navigation graphs | DFS/BFS |
| Two Pointers | Medium | List diffing, string manipulation | Two Pointers |
| Dynamic Programming | Medium | Layout computation, animation interpolation | Dynamic Programming |
| Greedy | Low | Battery/memory optimization heuristics | Greedy |

---

## Data & ML Engineering

### Data Engineering / Pipelines

| Topic | Priority | Why Tested | Key Patterns |
|-------|----------|-----------|--------------|
| Hash Table | High | Join operations, deduplication, lookup tables | Hash Table |
| Sliding Window | High | Windowed aggregation, streaming computation | Sliding Window |
| Heap / Priority Queue | High | Merge K sorted streams, top-K queries | Heap / Priority Queue |
| Greedy | Medium | Pipeline scheduling, resource allocation | Greedy |
| Two Pointers | Medium | Merge operations on sorted datasets | Two Pointers |
| Binary Search | Medium | Partitioning, range queries | Binary Search |

### Machine Learning Engineering

| Topic | Priority | Why Tested | Key Patterns |
|-------|----------|-----------|--------------|
| Dynamic Programming | High | Sequence models, optimization, Viterbi-style problems | Dynamic Programming |
| Hash Table | High | Feature hashing, embedding lookups, memoization | Hash Table |
| DFS/BFS | High | Model graph traversal, search in feature space | DFS/BFS |
| Binary Search | Medium | Hyperparameter tuning logic, threshold optimization | Binary Search |
| Heap / Priority Queue | Medium | Beam search, top-K predictions, nearest neighbors | Heap / Priority Queue |
| Sliding Window | Medium | Time-series feature extraction, windowed statistics | Sliding Window |
| Backtracking | Low | Feature selection, architecture search | Backtracking |

### Data Science / Analytics

| Topic | Priority | Why Tested | Key Patterns |
|-------|----------|-----------|--------------|
| Hash Table | High | Aggregation, grouping, counting | Hash Table |
| Two Pointers | Medium | Sorted data analysis, merge operations | Two Pointers |
| Sliding Window | Medium | Rolling statistics, time-windowed analysis | Sliding Window |
| Dynamic Programming | Medium | Optimization problems, sequence analysis | Dynamic Programming |
| Greedy | Low | Heuristic-based analysis decisions | Greedy |

---

## Infrastructure & Platform

### Cloud / DevOps / SRE

| Topic | Priority | Why Tested | Key Patterns |
|-------|----------|-----------|--------------|
| DFS/BFS | High | Infrastructure dependency graphs, network routing | DFS/BFS |
| Union-Find | High | Network connectivity, cluster detection | Union-Find |
| Greedy | High | Resource scheduling, bin packing, cost optimization | Greedy |
| Heap / Priority Queue | Medium | Job scheduling, priority-based routing | Heap / Priority Queue |
| Binary Search | Medium | Capacity planning, threshold detection | Binary Search |
| Hash Table | Medium | Configuration lookup, service discovery | Hash Table |

### Distributed Systems / Storage

| Topic | Priority | Why Tested | Key Patterns |
|-------|----------|-----------|--------------|
| Hash Table | High | Consistent hashing, partition mapping | Hash Table |
| DFS/BFS | High | Replication graphs, consistency propagation | DFS/BFS |
| Union-Find | High | Partition detection, cluster membership | Union-Find |
| Binary Search | Medium | Log-structured merge trees, range queries | Binary Search |
| Sliding Window | Medium | Windowed replication, log compaction | Sliding Window |
| Dynamic Programming | Low | Optimal replication strategies | Dynamic Programming |

### Networking / Security

| Topic | Priority | Why Tested | Key Patterns |
|-------|----------|-----------|--------------|
| DFS/BFS | High | Network traversal, shortest path routing | DFS/BFS |
| Hash Table | High | Packet classification, firewall rule lookup | Hash Table |
| Binary Search | Medium | IP range lookup, ACL matching | Binary Search |
| Greedy | Medium | Routing optimization, bandwidth allocation | Greedy |
| Union-Find | Medium | Network segmentation, connectivity analysis | Union-Find |

---

## Finance & Trading

### Quantitative Trading / HFT

| Topic | Priority | Why Tested | Key Patterns |
|-------|----------|-----------|--------------|
| Dynamic Programming | High | Options pricing, optimal execution, strategy optimization | Dynamic Programming |
| Binary Search | High | Order book operations, price level lookup | Binary Search |
| Heap / Priority Queue | High | Order matching, priority queues, streaming top-K | Heap / Priority Queue |
| Hash Table | High | Symbol lookup, position tracking, caching | Hash Table |
| Sliding Window | Medium | Moving averages, VWAP calculation, signal detection | Sliding Window |
| Greedy | Medium | Trade scheduling, execution optimization | Greedy |
| Two Pointers | Medium | Time-series comparison, pair analysis | Two Pointers |

### Fintech / Payments

| Topic | Priority | Why Tested | Key Patterns |
|-------|----------|-----------|--------------|
| Hash Table | High | Transaction deduplication, account lookup | Hash Table |
| DFS/BFS | High | Fraud detection graphs, payment routing | DFS/BFS |
| Dynamic Programming | Medium | Fee optimization, currency conversion | Dynamic Programming |
| Greedy | Medium | Payment splitting, change-making | Greedy |
| Binary Search | Low | Rate lookup, threshold detection | Binary Search |

### Risk / Compliance

| Topic | Priority | Why Tested | Key Patterns |
|-------|----------|-----------|--------------|
| DFS/BFS | High | Relationship graphs, exposure propagation | DFS/BFS |
| Hash Table | High | Entity resolution, rule matching | Hash Table |
| Dynamic Programming | Medium | Scenario analysis, Monte Carlo paths | Dynamic Programming |
| Union-Find | Medium | Entity grouping, connected exposure | Union-Find |
| Greedy | Low | Greedy hedging heuristics | Greedy |

---

## Specialized Domains

### Robotics / Autonomous Systems

| Topic | Priority | Why Tested | Key Patterns |
|-------|----------|-----------|--------------|
| DFS/BFS | High | Path planning, state space search | DFS/BFS |
| Dynamic Programming | High | Optimal control, trajectory planning | Dynamic Programming |
| Binary Search | Medium | Sensor data processing, threshold detection | Binary Search |
| Greedy | Medium | Real-time decision making under constraints | Greedy |
| Heap / Priority Queue | Medium | A* search, priority-based planning | Heap / Priority Queue |

### Gaming / Simulation

| Topic | Priority | Why Tested | Key Patterns |
|-------|----------|-----------|--------------|
| DFS/BFS | High | Pathfinding, game tree search | DFS/BFS |
| Dynamic Programming | High | Game state evaluation, scoring optimization | Dynamic Programming |
| Backtracking | Medium | Puzzle solving, constraint propagation | Backtracking |
| Greedy | Medium | AI decision heuristics | Greedy |
| Hash Table | Medium | State caching, transposition tables | Hash Table |

### Crypto / Blockchain

| Topic | Priority | Why Tested | Key Patterns |
|-------|----------|-----------|--------------|
| Hash Table | High | Merkle trees, address lookup, transaction indexing | Hash Table |
| DFS/BFS | High | Chain traversal, block graph analysis | DFS/BFS |
| Dynamic Programming | High | Gas optimization, MEV strategies, advanced subsequence/combinatorial problems. OA ground truth (Crypto.com 2026-02-28) confirmed 2D DP + subsequence dedup + modular arithmetic at Hard level for an early career role | Dynamic Programming |
| Greedy | Medium | Transaction ordering, fee optimization | Greedy |
| Union-Find | High | Account clustering, address grouping, entity resolution across chains. OA ground truth (Crypto.com 2026-02-28) confirmed Union-Find + prime factorization (LC 952 equivalent) at Hard level for an early career role | Union-Find |

---

## How to Use This Reference

1. **Identify the company's primary engineering domain** from Step 3 research
2. **Find the matching domain section** above
3. **Cross-reference with JD signals** from `jd-signal-mapping.md`
4. **Build the topic roadmap** by combining domain priorities with JD-specific signals
5. **When domains overlap** (e.g., ML + Finance), merge the tables and take the higher priority for each pattern
