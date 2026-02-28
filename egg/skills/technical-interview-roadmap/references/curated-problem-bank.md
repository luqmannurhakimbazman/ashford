# Curated Problem Bank

~90 LeetCode problems organized by pattern, with difficulty and domain relevance tags. All pattern names align with leetcode-teacher's `references/frameworks/problem-patterns.md` taxonomy.

**Selection criteria for this bank:** High interview frequency, clear pattern demonstration, good difficulty progression within each pattern, and broad domain coverage.

---

## How to Use This Bank

1. **Filter by pattern** — match patterns from the topic roadmap (Step 5)
2. **Filter by difficulty** — match role level calibration
3. **Filter by domain tags** — prefer problems tagged with the target company's domain
4. **Select 15-25 problems** total across all patterns
5. **Ensure progression** — within each pattern, include Easy → Medium → Hard
6. **Deduplicate** — some problems appear in multiple pattern sections (e.g., LC 76, 42). Count each problem only once regardless of which pattern section it was selected from

**Domain Tags Key:**
- `general` — universally relevant, no domain bias
- `web` — web/API/frontend engineering
- `data` — data engineering, pipelines, ETL
- `ml` — machine learning, AI, data science
- `finance` — quantitative finance, trading, fintech
- `infra` — infrastructure, distributed systems, cloud
- `systems` — low-level systems, performance-critical

---

## Hash Table

| # | Problem | Difficulty | Domain Tags | Key Concept |
|---|---------|-----------|-------------|-------------|
| 1 | Two Sum (1) | Easy | general | Complement lookup in single pass |
| 49 | Group Anagrams (49) | Medium | general, data | Canonical key hashing |
| 128 | Longest Consecutive Sequence (128) | Medium | general | Set-based sequence detection |
| 146 | LRU Cache (146) | Medium | web, infra, systems | Hash map + doubly linked list |
| 380 | Insert Delete GetRandom O(1) (380) | Medium | infra, systems | Array + hash map for O(1) operations |
| 560 | Subarray Sum Equals K (560) | Medium | data, ml | Prefix sum + hash map |
| 76 | Minimum Window Substring (76) | Hard | general | Hash map + sliding window (multi-pattern) |
| 41 | First Missing Positive (41) | Hard | systems | Array-as-hash-map trick |

---

## Two Pointers

| # | Problem | Difficulty | Domain Tags | Key Concept |
|---|---------|-----------|-------------|-------------|
| 125 | Valid Palindrome (125) | Easy | general | Opposite-direction two pointers |
| 167 | Two Sum II (167) | Medium | general, finance | Sorted array pair search |
| 15 | 3Sum (15) | Medium | general | Sort + two pointers for triplet |
| 11 | Container With Most Water (11) | Medium | general | Greedy shrink from both sides |
| 42 | Trapping Rain Water (42) | Hard | infra, systems | Two pointers with running max |
| 283 | Move Zeroes (283) | Easy | general | Fast-slow in-place modification |
| 26 | Remove Duplicates from Sorted Array (26) | Easy | general | Fast-slow pointer, sorted array |

---

## Sliding Window

| # | Problem | Difficulty | Domain Tags | Key Concept |
|---|---------|-----------|-------------|-------------|
| 121 | Best Time to Buy and Sell Stock (121) | Easy | finance | Track min price, max profit |
| 3 | Longest Substring Without Repeating (3) | Medium | general, web | Variable window with set/map |
| 424 | Longest Repeating Character Replacement (424) | Medium | general | Window with frequency + max count |
| 567 | Permutation in String (567) | Medium | general | Fixed window with frequency match |
| 239 | Sliding Window Maximum (239) | Hard | data, finance, systems | Monotonic deque for window max |
| 76 | Minimum Window Substring (76) | Hard | general | Variable window with target coverage |
| 438 | Find All Anagrams in a String (438) | Medium | general | Fixed window frequency comparison |

---

## Binary Search

| # | Problem | Difficulty | Domain Tags | Key Concept |
|---|---------|-----------|-------------|-------------|
| 704 | Binary Search (704) | Easy | general | Classic sorted array search |
| 33 | Search in Rotated Sorted Array (33) | Medium | general | Modified binary search with rotation |
| 153 | Find Minimum in Rotated Sorted Array (153) | Medium | general | Binary search on rotation point |
| 875 | Koko Eating Bananas (875) | Medium | general | Binary search on answer |
| 4 | Median of Two Sorted Arrays (4) | Hard | data, ml | Binary search on partition |
| 34 | Find First and Last Position (34) | Medium | general | Left/right boundary binary search |
| 981 | Time Based Key-Value Store (981) | Medium | web, infra | Binary search on timestamps |

---

## Dynamic Programming

| # | Problem | Difficulty | Domain Tags | Key Concept |
|---|---------|-----------|-------------|-------------|
| 70 | Climbing Stairs (70) | Easy | general | 1D DP, Fibonacci structure |
| 198 | House Robber (198) | Medium | general | 1D DP with skip constraint |
| 322 | Coin Change (322) | Medium | finance, general | Unbounded knapsack variant |
| 300 | Longest Increasing Subsequence (300) | Medium | data, finance | Subsequence DP + binary search optimization |
| 1143 | Longest Common Subsequence (1143) | Medium | general, ml | 2D DP on two sequences |
| 518 | Coin Change II (518) | Medium | finance | Counting combinations (unbounded knapsack) |
| 139 | Word Break (139) | Medium | ml, general | String segmentation DP |
| 152 | Maximum Product Subarray (152) | Medium | finance, data | Track min and max simultaneously |
| 72 | Edit Distance (72) | Medium | ml, general | 2D DP, string distance |
| 91 | Decode Ways (91) | Medium | general | 1D DP with conditional transitions |
| 312 | Burst Balloons (312) | Hard | finance | Interval DP |
| 10 | Regular Expression Matching (10) | Hard | systems, ml | 2D DP with wildcard |

---

## DFS/BFS

| # | Problem | Difficulty | Domain Tags | Key Concept |
|---|---------|-----------|-------------|-------------|
| 226 | Invert Binary Tree (226) | Easy | general | Recursive tree traversal |
| 104 | Maximum Depth of Binary Tree (104) | Easy | general | DFS depth computation |
| 200 | Number of Islands (200) | Medium | general, infra | Grid BFS/DFS, connected components |
| 102 | Binary Tree Level Order Traversal (102) | Medium | general | BFS level-by-level |
| 133 | Clone Graph (133) | Medium | infra | BFS/DFS with visited map |
| 207 | Course Schedule (207) | Medium | general, infra | Topological sort, cycle detection |
| 210 | Course Schedule II (210) | Medium | infra | Topological ordering |
| 994 | Rotting Oranges (994) | Medium | general | Multi-source BFS |
| 417 | Pacific Atlantic Water Flow (417) | Medium | general | Multi-source DFS from boundaries |
| 127 | Word Ladder (127) | Hard | ml, general | BFS shortest transformation |
| 124 | Binary Tree Maximum Path Sum (124) | Hard | general | DFS with global max tracking |
| 297 | Serialize and Deserialize Binary Tree (297) | Hard | infra, web | Tree encoding/decoding |

---

## Backtracking

| # | Problem | Difficulty | Domain Tags | Key Concept |
|---|---------|-----------|-------------|-------------|
| 78 | Subsets (78) | Medium | general | Power set generation |
| 46 | Permutations (46) | Medium | general | Full permutation enumeration |
| 39 | Combination Sum (39) | Medium | general | Combinations with reuse |
| 79 | Word Search (79) | Medium | general | Grid backtracking with visited |
| 51 | N-Queens (51) | Hard | general | Constraint satisfaction |
| 131 | Palindrome Partitioning (131) | Medium | general | Partitioning with validation |
| 17 | Letter Combinations of a Phone Number (17) | Medium | general, web | Multi-choice enumeration |

---

## Greedy

| # | Problem | Difficulty | Domain Tags | Key Concept |
|---|---------|-----------|-------------|-------------|
| 55 | Jump Game (55) | Medium | general | Reachability tracking |
| 45 | Jump Game II (45) | Medium | general | BFS-style greedy jumps |
| 56 | Merge Intervals (56) | Medium | data, infra, finance | Sort + merge overlapping |
| 435 | Non-overlapping Intervals (435) | Medium | data, infra | Interval scheduling (max non-overlap) |
| 621 | Task Scheduler (621) | Medium | infra, systems | Greedy with cooldown |
| 763 | Partition Labels (763) | Medium | general, data | Last occurrence tracking |
| 134 | Gas Station (134) | Medium | general | Circular greedy |
| 135 | Candy (135) | Hard | general | Two-pass greedy |

---

## Stack / Monotonic Stack

| # | Problem | Difficulty | Domain Tags | Key Concept |
|---|---------|-----------|-------------|-------------|
| 20 | Valid Parentheses (20) | Easy | general | Matching brackets with stack |
| 155 | Min Stack (155) | Medium | general, systems | Stack with O(1) min tracking |
| 739 | Daily Temperatures (739) | Medium | data, finance | Monotonic decreasing stack |
| 496 | Next Greater Element I (496) | Easy | general | Monotonic stack + hash map |
| 84 | Largest Rectangle in Histogram (84) | Hard | data, systems | Monotonic stack for max area |

---

## Heap / Priority Queue

| # | Problem | Difficulty | Domain Tags | Key Concept |
|---|---------|-----------|-------------|-------------|
| 215 | Kth Largest Element in an Array (215) | Medium | data, finance | Min-heap of size K |
| 347 | Top K Frequent Elements (347) | Medium | data, web, ml | Frequency count + heap |
| 23 | Merge K Sorted Lists (23) | Hard | data, systems | K-way merge with min-heap |
| 295 | Find Median from Data Stream (295) | Hard | data, finance | Two-heap median tracking |
| 355 | Design Twitter (355) | Medium | web | Merge K sorted feeds |
| 973 | K Closest Points to Origin (973) | Medium | ml, general | Distance-based heap selection |
| 621 | Task Scheduler (621) | Medium | infra, systems | Max-heap + cooldown (multi-pattern) |

---

## Union-Find

| # | Problem | Difficulty | Domain Tags | Key Concept |
|---|---------|-----------|-------------|-------------|
| 200 | Number of Islands (200) | Medium | general, infra | Union adjacent land cells (alt approach) |
| 323 | Number of Connected Components (323) | Medium | infra | Classic union-find |
| 684 | Redundant Connection (684) | Medium | infra | Cycle detection via union-find |
| 721 | Accounts Merge (721) | Medium | web, data | Group equivalent items |
| 128 | Longest Consecutive Sequence (128) | Medium | general | Union consecutive elements (alt approach) |
| 1101 | Earliest Moment When Everyone Becomes Friends (1101) | Medium | general, data | Time-ordered union with earliest threshold |
| 261 | Graph Valid Tree (261) | Medium | infra | N-1 edges + single component check |

---

## Multi-Pattern Problems

These problems combine multiple patterns and are good for advanced practice:

| # | Problem | Difficulty | Patterns | Domain Tags | Key Concept |
|---|---------|-----------|----------|-------------|-------------|
| 42 | Trapping Rain Water (42) | Hard | Two Pointers + DP | infra, systems | Two-pointer with running state |
| 76 | Minimum Window Substring (76) | Hard | Sliding Window + Hash Table | general | Window with hash map tracking |
| 127 | Word Ladder (127) | Hard | BFS + Hash Table | ml, general | BFS with preprocessed adjacency |
| 297 | Serialize/Deserialize Binary Tree (297) | Hard | DFS/BFS + Design | infra, web | Tree encoding with traversal |
| 295 | Find Median from Data Stream (295) | Hard | Heap + Design | data, finance | Two-heap invariant |
| 146 | LRU Cache (146) | Medium | Hash Table + Design | web, infra, systems | O(1) cache with ordering |

---

## Problem Selection Cheat Sheet

### By Role Level

| Level | Suggested Distribution | Focus |
|-------|----------------------|-------|
| Junior / Entry | 5 Easy + 15 Medium + 3-5 Hard | Build pattern fluency AND prepare for worst-case OA difficulty |
| Mid | 3 Easy + 14 Medium + 3-5 Hard | Deepen medium, ensure hard readiness |
| Senior | 0 Easy + 12 Medium + 8 Hard | Optimize and handle edge cases |

> **These distributions are starting points.** Real OAs routinely exceed expected difficulty regardless of role level. Always include Hard problems — companies use shared assessment platforms (HackerRank, Codility, CodeSignal) with problem pools not calibrated to role level.

### By Time Available

| Timeline | Total Problems | Strategy |
|----------|---------------|----------|
| 1 week | 15 | Tier 1 only, Easy-Medium, skip Tier 3 |
| 2 weeks | 20 | Tier 1-2, full difficulty range |
| 3+ weeks | 25 | Full coverage including Tier 3 stretch |

### Must-Include for Any Company

These problems are so frequently tested that they should be included regardless of company:

1. Two Sum (1) — Hash Table
2. Best Time to Buy and Sell Stock (121) — Sliding Window / Greedy
3. Valid Parentheses (20) — Stack / Monotonic Stack
4. Merge Intervals (56) — Greedy
5. Number of Islands (200) — DFS/BFS
6. LRU Cache (146) — Hash Table + Design
7. Coin Change (322) — Dynamic Programming
8. Binary Search (704) — Binary Search
