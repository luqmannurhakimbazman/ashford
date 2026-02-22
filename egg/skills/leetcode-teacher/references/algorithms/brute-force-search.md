# Brute Force Search

Deep-dive into backtracking, 2D grid DFS, and BFS applications. Covers the 9-variant unified framework for permutations/combinations/subsets, the Ball-Box Model for enumeration perspectives, constraint satisfaction backtracking, island problems, and state-space BFS. Builds on the base templates in `algorithm-frameworks.md`.

---

## Quick Reference Table

| Topic | Key Insight | When to Use | Complexity |
|-------|-------------|-------------|------------|
| 9-Variant Framework | 3x3 matrix: {subset, combination, permutation} x {unique, duplicates, reuse} | Any enumeration of subsets/combinations/permutations | O(2^N) or O(N!) |
| Ball-Box Model | Two perspectives: box chooses ball (for-loop) vs ball chooses box (include/exclude) | Choosing the right enumeration strategy | Varies |
| DFS vs Backtracking | Choice inside vs outside the for loop changes root node behavior | Understanding subtle recursion differences | Same |
| Constraint Satisfaction | `isValid` check before placing, single vs all solutions | Sudoku, N-Queens, crossword puzzles | Exponential with pruning |
| 2D Grid DFS | Grid = graph, each cell has 4 neighbors, flood fill to mark visited | Island counting, area, closed regions | O(M x N) |
| State-Space BFS | Abstract problem state as graph node, transitions as edges | Sliding puzzle, Open the Lock, game boards | O(V + E) of state graph |
| Augmented-State BFS | Encode extra info (direction, turns, keys) into state | Mazes with constraints, Connect-Two | O(states x transitions) |

---

## Part I: Backtracking Deep Dives

### 1. Core Backtracking Recap

The base backtracking template lives in `algorithm-frameworks.md` (The Backtracking Framework section). Every backtracking problem is a walk through a decision tree with three components:

1. **Path** — choices made so far
2. **Choice list** — options available at the current node
3. **End condition** — when to record a result

This file goes deeper into variants, perspectives, and advanced applications.

---

### 2. The 9-Variant Unified Framework

All subset/combination/permutation problems are one template with three knobs:

| | **Subsets** (variable length) | **Combinations** (fixed length k) | **Permutations** (length N) |
|---|---|---|---|
| **Unique elements, no reuse** | LC 78 | LC 77 | LC 46 |
| **Duplicate elements, no reuse** | LC 90 | LC 40 | LC 47 |
| **Unique elements, with reuse** | (not standard) | LC 39 | (not standard) |

The three knobs that distinguish all nine variants:

1. **Collection type** — subsets (all sizes), combinations (fixed size k), or permutations (all elements, order matters)
2. **Element uniqueness** — are input elements unique, or can there be duplicates?
3. **Reuse allowed** — can each element be used more than once?

#### Subset/Combination Variants (use `start` parameter)

```python
def backtrack(nums, start, path, result):
    # For subsets: collect at every node
    result.append(path[:])
    # For combinations: collect only when len(path) == k
    # if len(path) == k:
    #     result.append(path[:])
    #     return

    for i in range(start, len(nums)):
        # --- DEDUP (duplicate elements, no reuse) ---
        # if i > start and nums[i] == nums[i - 1]:
        #     continue

        path.append(nums[i])

        # No reuse: next start = i + 1
        backtrack(nums, i + 1, path, result)
        # With reuse: next start = i (not i + 1)
        # backtrack(nums, i, path, result)

        path.pop()
```

**Key mechanics:**

- `start = i + 1` → each element used at most once, no backward selection
- `start = i` → same element can be reused (Combination Sum)
- Sort + `if i > start and nums[i] == nums[i-1]: continue` → skip duplicate branches at the same tree level

#### Permutation Variants (use `used` array)

```python
def backtrack(nums, path, used, result):
    if len(path) == len(nums):
        result.append(path[:])
        return

    for i in range(len(nums)):
        if used[i]:
            continue
        # --- DEDUP (duplicate elements, no reuse) ---
        # if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
        #     continue

        used[i] = True
        path.append(nums[i])
        backtrack(nums, path, used, result)
        path.pop()
        used[i] = False
```

**Key mechanics:**

- Permutations consider all indices (no `start`), but skip `used[i] == True`
- Dedup for duplicate elements: sort first, then `if i > 0 and nums[i] == nums[i-1] and not used[i-1]: continue` — this ensures only the first occurrence of a duplicate is chosen at each tree level

#### Why `not used[i-1]` for Permutation Dedup

When `nums[i] == nums[i-1]` and `used[i-1]` is `False`, it means `nums[i-1]` was **not chosen in the current path** — so choosing `nums[i]` at this position would create a duplicate permutation (we should have chosen `nums[i-1]` first). Skipping ensures duplicates are always picked in left-to-right order.

#### Summary Decision Tree

```
Is order important?
├── Yes → PERMUTATION (use `used[]` array, iterate from 0)
│   ├── Unique elements → LC 46 template
│   └── Duplicate elements → sort + dedup with `not used[i-1]`
└── No → SUBSET or COMBINATION (use `start` parameter)
    ├── Variable length → SUBSET
    │   ├── Unique → LC 78 (collect at every node)
    │   └── Duplicates → LC 90 (sort + skip `i > start`)
    └── Fixed length k → COMBINATION
        ├── Unique, no reuse → LC 77 (`start = i + 1`)
        ├── Duplicates, no reuse → LC 40 (sort + skip)
        └── Unique, with reuse → LC 39 (`start = i`)
```

*Socratic prompt: "You have a subsets problem with duplicate elements. What two things do you add to the base template?"* (Answer: sort the input + skip when `i > start and nums[i] == nums[i-1]`)

---

### 3. The Ball-Box Model

Labuladong's key insight: every enumeration problem can be viewed from two perspectives, and **choosing the right perspective often simplifies the solution**.

#### Perspective 1: Box Chooses Ball (for-loop enumeration)

For each "position" (box), iterate through all candidates (balls) and pick one.

```python
# Box perspective: for each position, which element goes here?
def backtrack(boxes, box_idx, balls, used):
    if box_idx == len(boxes):
        record(boxes)
        return
    for ball in balls:
        if not used[ball]:
            boxes[box_idx] = ball
            used[ball] = True
            backtrack(boxes, box_idx + 1, balls, used)
            used[ball] = False
```

This is the **standard backtracking template** — the for loop at each tree node picks from available choices.

**When to prefer:** Permutations, combinations, most standard problems. The for-loop naturally generates the choice list.

#### Perspective 2: Ball Chooses Box (include/exclude branching)

For each "element" (ball), decide which "position" (box) it goes to — or whether to include/exclude it.

```python
# Ball perspective: for each element, include or exclude?
def backtrack(nums, idx, path, result):
    if idx == len(nums):
        result.append(path[:])
        return
    # Branch 1: include nums[idx]
    path.append(nums[idx])
    backtrack(nums, idx + 1, path, result)
    path.pop()
    # Branch 2: exclude nums[idx]
    backtrack(nums, idx + 1, path, result)
```

This creates a **binary decision tree** — each node branches into "take" or "skip."

**When to prefer:**
- **Subsets** — the ball perspective naturally generates all 2^N subsets via include/exclude
- **Partition problems** — "which group does this element belong to?" (e.g., Partition to K Equal Sum Subsets)
- **Problems where the number of "boxes" is small** — each element chooses among few destinations

#### Comparison Table

| Aspect | Box Chooses Ball (for-loop) | Ball Chooses Box (include/exclude) |
|--------|---------------------------|-----------------------------------|
| Tree shape | N-ary tree (branching = choice list size) | Binary tree (include/exclude) |
| Tree depth | Number of positions to fill | Number of elements to consider |
| Natural fit | Permutations, combinations | Subsets, partition problems |
| Pruning advantage | Prune when choice list shrinks | Prune when running totals fail early |
| Standard template | `for choice in choices: ...` | Two recursive calls (take/skip) |

*Socratic prompt: "For Partition to K Equal Sum Subsets, would you iterate over subsets and assign elements, or iterate over elements and assign to subsets? Which gives better pruning?"* (Answer: iterate over elements — ball perspective — and assign each to one of K subsets. This prunes earlier because you check subset sums incrementally.)

---

### 4. DFS vs Backtracking Distinction

These terms are often used interchangeably, but labuladong identifies a subtle structural difference.

#### Pattern A: DFS (choice outside the for loop)

The "choice" for the current node happens **before** entering the for loop. The root node's choice is executed.

```python
def dfs(node):
    if not node:
        return
    # Choice happens HERE (root included)
    visit(node)
    for child in node.children:
        dfs(child)
```

#### Pattern B: Backtracking (choice inside the for loop)

The "choice" happens **inside** the for loop. The root node is never explicitly chosen — it's the starting point.

```python
def backtrack(node):
    if is_leaf(node):
        record()
        return
    for child in node.children:
        # Choice happens HERE (root skipped)
        make_choice(child)
        backtrack(child)
        undo_choice(child)
```

#### The Root Node Difference

| | DFS (Pattern A) | Backtracking (Pattern B) |
|---|---|---|
| Root node | Processed (visited/chosen) | Not processed (starting point only) |
| Choice location | Before/outside the for loop | Inside the for loop |
| Undo (backtrack) | Not needed at current level | Required after each child |
| Typical use | Tree/graph traversal | Enumeration problems |

**Why this matters:** In permutation/combination problems, the "root" of the decision tree represents "no choices made yet." If you accidentally process the root (DFS style), you get an off-by-one in your enumeration. Standard backtracking templates place the choice inside the for loop to avoid this.

**Practical example — N-ary tree paths:**

```python
# DFS style: root is part of the path
def dfs(node, path):
    path.append(node.val)        # Process current node
    if not node.children:
        result.append(path[:])   # Leaf reached
    for child in node.children:
        dfs(child, path)
    path.pop()                   # Undo for parent

# Backtracking style: root is starting context
def backtrack(node):
    if not node.children:
        result.append(path[:])
        return
    for child in node.children:
        path.append(child.val)   # Choose child
        backtrack(child)
        path.pop()               # Unchoose child
```

Both produce the same result for non-root nodes, but differ in whether the root's value appears. For enumeration problems, backtracking style (Pattern B) is standard.

---

### 5. Constraint Satisfaction Backtracking

Some backtracking problems are not about generating all combinations, but about placing elements under constraints. The key additions: an `isValid` check before each placement, and optionally returning a single solution (return `bool`) vs collecting all solutions.

#### N-Queens (LC 51, 52)

Place N queens on an NxN board so no two attack each other.

```python
def solve_n_queens(n):
    result = []
    board = [['.'] * n for _ in range(n)]

    def is_valid(board, row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        # Check upper-left diagonal
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1
        # Check upper-right diagonal
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1
        return True

    def backtrack(row):
        if row == n:
            result.append([''.join(r) for r in board])
            return
        for col in range(n):
            if not is_valid(board, row, col):
                continue
            board[row][col] = 'Q'
            backtrack(row + 1)
            board[row][col] = '.'

    backtrack(0)
    return result
```

**Key design decisions:**
- Place one queen per row (row = tree level, column = choice)
- `isValid` checks column + two diagonals (only upward — rows below haven't been filled)
- For LC 52 (count only), replace `result.append(...)` with a counter

#### Sudoku Solver (LC 37)

Fill a 9x9 grid so each row, column, and 3x3 box contains digits 1-9.

```python
def solve_sudoku(board):
    def is_valid(board, row, col, num):
        # Check row
        if num in board[row]:
            return False
        # Check column
        if any(board[r][col] == num for r in range(9)):
            return False
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if board[r][c] == num:
                    return False
        return True

    def backtrack(board):
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    continue
                for num in '123456789':
                    if not is_valid(board, i, j, num):
                        continue
                    board[i][j] = num
                    if backtrack(board):  # Single solution: return bool
                        return True
                    board[i][j] = '.'
                return False  # No valid number for this cell → backtrack
        return True  # All cells filled

    backtrack(board)
```

**Key design decisions:**
- Find the first empty cell, try digits 1-9
- `return True/False` pattern for single-solution problems — return immediately when a valid completion is found
- `return False` after trying all digits = dead end → triggers backtracking

#### Single Solution vs All Solutions

| Pattern | Return Type | Base Case | After Recursive Call |
|---------|-------------|-----------|---------------------|
| All solutions | `void` | `result.append(...)` | Continue loop |
| Single solution | `bool` | `return True` | `if backtrack(...): return True` |

---

### 6. Practical Backtracking Exercises

#### Generate Parentheses (LC 22)

Generate all valid combinations of `n` pairs of parentheses.

```python
def generate_parenthesis(n):
    result = []

    def backtrack(path, left, right):
        # End condition: used all parentheses
        if len(path) == 2 * n:
            result.append(''.join(path))
            return
        # Can add '(' if we haven't used all left parens
        if left < n:
            path.append('(')
            backtrack(path, left + 1, right)
            path.pop()
        # Can add ')' if it won't create invalid sequence
        if right < left:
            path.append(')')
            backtrack(path, left, right + 1)
            path.pop()

    backtrack([], 0, 0)
    return result
```

**Key insight:** This is a **binary branching** problem (ball perspective — each position chooses `(` or `)`). The constraints `left < n` and `right < left` are the pruning conditions that ensure validity without post-hoc checking.

*Socratic prompt: "Why is the condition `right < left` and not `right < n`? What invalid string would `right < n` allow?"*

#### Partition to K Equal Sum Subsets (LC 698)

Partition an array into `k` subsets with equal sums.

```python
def can_partition_k_subsets(nums, k):
    total = sum(nums)
    if total % k != 0:
        return False
    target = total // k
    nums.sort(reverse=True)  # Pruning: try large numbers first
    buckets = [0] * k

    def backtrack(idx):
        if idx == len(nums):
            return all(b == target for b in buckets)
        for j in range(k):
            if buckets[j] + nums[idx] > target:
                continue
            # Pruning: skip duplicate buckets
            if j > 0 and buckets[j] == buckets[j - 1]:
                continue
            buckets[j] += nums[idx]
            if backtrack(idx + 1):
                return True
            buckets[j] -= nums[idx]
        return False

    return backtrack(0)
```

**Key insight:** This uses the **ball perspective** — each number (ball) chooses which bucket (box) to go into. The box perspective (each bucket selects numbers) has worse pruning because you don't know which numbers are left.

**Pruning techniques:**
1. **Sort descending** — large numbers fail faster, pruning more branches
2. **Skip duplicate buckets** — if `buckets[j] == buckets[j-1]`, assigning to either gives the same outcome
3. **Sum check** — `buckets[j] + nums[idx] > target` prunes immediately

---

## Part II: 2D Grid DFS (Island Problems)

### 7. Grid-as-Graph Framework

A 2D matrix is an implicit graph where each cell is a node and adjacent cells (up, down, left, right) are neighbors. Grid DFS is structurally identical to tree/graph DFS.

#### Binary Tree Analogy

A binary tree node has 2 children. A grid cell has 4 neighbors. The DFS template is the same — just more recursive calls.

```python
# Binary tree DFS
def traverse(node):
    if not node:
        return
    traverse(node.left)
    traverse(node.right)

# 2D grid DFS (4-directional)
def dfs(grid, i, j):
    m, n = len(grid), len(grid[0])
    if i < 0 or i >= m or j < 0 or j >= n:
        return  # Out of bounds
    if grid[i][j] == '0':
        return  # Water or already visited
    grid[i][j] = '0'  # Mark visited (flood fill)
    dfs(grid, i + 1, j)  # Down
    dfs(grid, i - 1, j)  # Up
    dfs(grid, i, j + 1)  # Right
    dfs(grid, i, j - 1)  # Left
```

#### Flood Fill Technique

Instead of maintaining a separate `visited` set, directly modify the grid to mark cells as visited. This is the "flood fill" technique — equivalent to `visited.add()` but saves space.

**When to use flood fill:** When the grid values can be safely overwritten (e.g., `'1'` → `'0'`).

**When to use a separate visited set:** When original grid values must be preserved, or when the DFS needs to "un-visit" cells (backtracking on grids).

---

### 8. Island Problem Variants

All island problems use the same grid DFS skeleton. The variants differ in what you count, how you handle borders, and what extra information you track.

#### Number of Islands (LC 200)

Count connected components of `'1'` cells.

```python
def num_islands(grid):
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                count += 1
                dfs(grid, i, j)  # Flood fill entire island
    return count
```

**Pattern:** Outer loop finds unvisited land → DFS floods entire island → increment counter.

#### Closed Islands / Number of Enclaves (LC 1254, LC 1020)

Islands that touch the border are "open." Closed islands are entirely surrounded by water.

```python
def closed_island(grid):
    m, n = len(grid), len(grid[0])
    # Step 1: Flood fill all border-connected land
    for i in range(m):
        for j in range(n):
            if (i == 0 or i == m - 1 or j == 0 or j == n - 1) and grid[i][j] == 0:
                dfs(grid, i, j)  # Mark border land as water
    # Step 2: Count remaining islands
    count = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 0:
                count += 1
                dfs(grid, i, j)
    return count
```

**Key insight:** First eliminate border-touching islands (they can't be closed), then count what remains.

For **Number of Enclaves** (LC 1020): same approach, but count individual cells instead of connected components.

#### Max Area of Island (LC 695)

Return the area (cell count) of the largest island.

```python
def max_area_of_island(grid):
    def dfs(grid, i, j):
        m, n = len(grid), len(grid[0])
        if i < 0 or i >= m or j < 0 or j >= n:
            return 0
        if grid[i][j] == 0:
            return 0
        grid[i][j] = 0
        return 1 + dfs(grid, i+1, j) + dfs(grid, i-1, j) + \
                   dfs(grid, i, j+1) + dfs(grid, i, j-1)

    max_area = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                max_area = max(max_area, dfs(grid, i, j))
    return max_area
```

**Key insight:** DFS returns the area count (decomposition mode — each subtree returns its size).

#### Count Sub Islands (LC 1905)

Grid B's island is a "sub island" of grid A if every land cell in B's island is also land in A.

```python
def count_sub_islands(grid1, grid2):
    m, n = len(grid1), len(grid1[0])

    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n:
            return True
        if grid2[i][j] == 0:
            return True
        grid2[i][j] = 0
        # Check all 4 directions; use `&` not `and` to avoid short-circuit
        res = grid1[i][j] == 1
        res = dfs(i+1, j) and res
        res = dfs(i-1, j) and res
        res = dfs(i, j+1) and res
        res = dfs(i, j-1) and res
        return res

    count = 0
    for i in range(m):
        for j in range(n):
            if grid2[i][j] == 1:
                if dfs(i, j):
                    count += 1
    return count
```

**Key insight:** During DFS, check if every cell of grid2's island also has land in grid1. Must avoid short-circuit evaluation — even if one cell fails, continue DFS to flood-fill the entire island.

#### Distinct Islands (LC 694)

Count islands with distinct shapes. Two islands have the same shape if one can be translated (not rotated) to match the other.

```python
def num_distinct_islands(grid):
    m, n = len(grid), len(grid[0])
    shapes = set()

    def dfs(grid, i, j, path, direction):
        if i < 0 or i >= m or j < 0 or j >= n:
            return
        if grid[i][j] == 0:
            return
        grid[i][j] = 0
        path.append(direction)
        dfs(grid, i + 1, j, path, 'D')  # Down
        dfs(grid, i - 1, j, path, 'U')  # Up
        dfs(grid, i, j + 1, path, 'R')  # Right
        dfs(grid, i, j - 1, path, 'L')  # Left
        path.append('B')  # Backtrack marker

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                path = []
                dfs(grid, i, j, path, 'S')  # S = start
                shapes.add(tuple(path))

    return len(shapes)
```

**Key insight:** Serialize the DFS traversal path (including backtrack markers) as a shape fingerprint. Two islands with the same shape produce identical serialized paths when traversed in the same order. The backtrack marker `'B'` is essential — without it, different tree structures can produce the same sequence.

#### Minesweeper (LC 529)

Click a cell: if mine → game over. If empty with no adjacent mines → reveal and recursively reveal neighbors. If empty with adjacent mines → show count.

```python
def update_board(board, click):
    i, j = click
    if board[i][j] == 'M':
        board[i][j] = 'X'
        return board

    m, n = len(board), len(board[0])
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def dfs(i, j):
        # Count adjacent mines (8 directions)
        count = sum(
            1 for di, dj in dirs
            if 0 <= i+di < m and 0 <= j+dj < n
            and board[i+di][j+dj] in ('M', 'X')
        )
        if count > 0:
            board[i][j] = str(count)
        else:
            board[i][j] = 'B'  # Blank
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and board[ni][nj] == 'E':
                    dfs(ni, nj)

    dfs(i, j)
    return board
```

**Key insight:** 8-directional DFS (not 4). Recursion only continues into cells with zero adjacent mines. Cells with adjacent mines show the count and stop recursion — this is the "boundary" of the reveal.

#### Island Problems Summary

| Problem | LC # | Special Technique |
|---------|------|-------------------|
| Number of Islands | 200 | Basic flood fill + count components |
| Closed Islands | 1254 | Eliminate border islands first |
| Number of Enclaves | 1020 | Eliminate border islands, count remaining cells |
| Max Area of Island | 695 | DFS returns area (decomposition mode) |
| Count Sub Islands | 1905 | Cross-reference two grids during DFS |
| Distinct Islands | 694 | Serialize DFS path as shape fingerprint |
| Minesweeper | 529 | 8-directional, count-based boundary |

---

## Part III: BFS Applications

### 9. BFS Template Recap

The base BFS template lives in `algorithm-frameworks.md` (The BFS Framework section). BFS explores all nodes at distance `d` before any at distance `d+1`, guaranteeing shortest paths in unweighted graphs.

This section covers advanced applications where BFS operates on **abstract state spaces** rather than explicit graphs.

---

### 10. State-Space BFS

Many puzzle and game problems are not given as explicit graphs. The key insight: **define the problem state as a node, and valid transitions as edges.** Then BFS finds the shortest sequence of moves.

#### Open the Lock (LC 752)

A 4-digit combination lock. Each move turns one wheel up or down. Find minimum moves from "0000" to target, avoiding deadends.

```python
from collections import deque

def open_lock(deadends, target):
    dead = set(deadends)
    if "0000" in dead:
        return -1

    visited = {"0000"}
    queue = deque(["0000"])
    steps = 0

    while queue:
        for _ in range(len(queue)):
            state = queue.popleft()
            if state == target:
                return steps
            # Generate all 8 neighbor states (4 digits x 2 directions)
            for i in range(4):
                digit = int(state[i])
                for delta in (1, -1):
                    new_digit = (digit + delta) % 10
                    neighbor = state[:i] + str(new_digit) + state[i+1:]
                    if neighbor not in visited and neighbor not in dead:
                        visited.add(neighbor)
                        queue.append(neighbor)
        steps += 1

    return -1
```

**State definition:** 4-digit string (e.g., "0000", "0001", ...)
**Transitions:** 8 neighbors per state (4 positions x up/down)
**State space size:** 10^4 = 10,000 states (small enough for BFS)

**Optimization:** This is a textbook case for **bidirectional BFS** (see `algorithm-frameworks.md`), searching from both "0000" and target simultaneously.

#### Sliding Puzzle (LC 773)

A 2x3 board with tiles 1-5 and one empty space (0). Slide tiles to reach `[[1,2,3],[4,5,0]]`.

```python
from collections import deque

def sliding_puzzle(board):
    # Serialize board state as string
    start = ''.join(str(board[i][j]) for i in range(2) for j in range(3))
    target = "123450"

    # Adjacency: for each position (0-5), which positions can swap with it?
    neighbors = [
        [1, 3],     # pos 0 → can swap with pos 1, 3
        [0, 2, 4],  # pos 1 → can swap with pos 0, 2, 4
        [1, 5],     # pos 2
        [0, 4],     # pos 3
        [1, 3, 5],  # pos 4
        [2, 4],     # pos 5
    ]

    visited = {start}
    queue = deque([start])
    steps = 0

    while queue:
        for _ in range(len(queue)):
            state = queue.popleft()
            if state == target:
                return steps
            zero_idx = state.index('0')
            for neighbor_idx in neighbors[zero_idx]:
                # Swap zero with neighbor
                state_list = list(state)
                state_list[zero_idx], state_list[neighbor_idx] = \
                    state_list[neighbor_idx], state_list[zero_idx]
                new_state = ''.join(state_list)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append(new_state)
        steps += 1

    return -1
```

**Key insight:** Serialize the entire board as a string to use as a hashable state in the visited set. Pre-compute the adjacency list for the 2x3 grid (which positions can the empty space swap with?).

**State space size:** 6! = 720 possible board configurations (very manageable).

**Extension — Huarong Dao (华容道):** Same concept but larger board (typically 4x5 with multi-cell pieces). The state must encode positions of all pieces. State space is much larger but the BFS approach is identical.

---

### 11. BFS with Augmented State

Sometimes the minimum-distance BFS needs extra information beyond just position. Encode this extra info into the state.

#### When Position Alone Is Insufficient

Standard BFS state: `(row, col)` — each cell visited at most once.

But some problems need:
- **Direction + turn count** — "minimum turns to reach the exit" (different directions reaching the same cell are different states)
- **Keys collected** — "shortest path collecting all keys" (same cell with different key sets are different states)
- **Resources remaining** — "reach target with at most K stops"

#### Augmented State Pattern

```python
from collections import deque

def bfs_augmented(grid, start, target):
    # State: (row, col, extra_info)
    initial_state = (start[0], start[1], initial_extra)
    visited = {initial_state}
    queue = deque([initial_state])
    steps = 0

    while queue:
        for _ in range(len(queue)):
            r, c, extra = queue.popleft()
            if (r, c, extra) == target_state:
                return steps
            for nr, nc, new_extra in get_neighbors(r, c, extra, grid):
                state = (nr, nc, new_extra)
                if state not in visited:
                    visited.add(state)
                    queue.append(state)
        steps += 1

    return -1
```

**Examples:**

- **Shortest Path with Keys (LC 864):** State = `(row, col, keys_bitmask)`. A cell can be revisited with a different set of keys.
- **Shortest Path with Obstacle Elimination (LC 1293):** State = `(row, col, obstacles_remaining)`. Same cell with different remaining eliminations are different states.
- **Minimum Moves to Reach Target with a Knight:** State = `(row, col)` is sufficient since there are no constraints beyond position.

**Cost of augmented state:** The state space grows multiplicatively. With a grid of M x N and K possible extra states, the total space is O(M x N x K). Ensure K is bounded.

---

### 12. DFS vs BFS for Mazes

#### When to Use BFS

- Finding the **shortest path** (minimum steps/moves)
- All transitions have **equal cost** (unweighted edges)
- You need the **optimal** answer, not just any answer

#### When to Use DFS

- Finding **any** path (existence, not optimality)
- **Counting** all paths or valid configurations
- Problems with **backtracking** requirements (try/undo)
- Memory-constrained scenarios (DFS uses O(depth) stack vs BFS's O(width) queue)

#### Path Recording in BFS

BFS doesn't naturally track the path (unlike DFS's recursion stack). To reconstruct the shortest path:

```python
from collections import deque

def bfs_with_path(grid, start, target):
    m, n = len(grid), len(grid[0])
    parent = {start: None}
    queue = deque([start])

    while queue:
        r, c = queue.popleft()
        if (r, c) == target:
            # Reconstruct path
            path = []
            node = (r, c)
            while node:
                path.append(node)
                node = parent[node]
            return path[::-1]
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and (nr, nc) not in parent:
                if grid[nr][nc] != '#':  # Not a wall
                    parent[(nr, nc)] = (r, c)
                    queue.append((nr, nc))

    return []  # No path
```

**Key insight:** Use a `parent` dictionary (also serves as `visited`). After reaching the target, follow parent pointers backward to reconstruct the path.

#### Summary: DFS vs BFS for Grid/Maze Problems

| Criterion | DFS | BFS |
|-----------|-----|-----|
| Shortest path guarantee | No | Yes (unweighted) |
| Memory usage | O(depth) | O(width) — can be O(M x N) |
| Finds any path | Yes (fast) | Yes (but explores more) |
| All paths / counting | Natural fit | Awkward |
| Implementation | Recursion or stack | Queue |
| Path reconstruction | Free (recursion stack) | Needs parent array |

---

## Attribution

The content in this file is inspired by and adapted from labuladong's algorithmic guides (labuladong.online), specifically Chapter 2: Brute Force Search. The 9-variant framework, Ball-Box Model, DFS/backtracking distinction, constraint satisfaction patterns, island problem templates, and state-space BFS techniques have been restructured and annotated for Socratic teaching use.

### Source Articles

1. [Backtracking Framework](https://labuladong.online/algo/essential-technique/backtrack-framework/)
2. [Ball-Box Model: Two Perspectives of Backtracking](https://labuladong.online/algo/practice-in-action/two-views-of-backtracking/)
3. [Subset/Combination/Permutation Variants](https://labuladong.online/algo/essential-technique/permutation-combination-subset-all-in-one/)
4. [DFS and Backtracking Are Not the Same Thing](https://labuladong.online/algo/essential-technique/backtrack-vs-dfs/)
5. [Sudoku Solver](https://labuladong.online/algo/practice-in-action/sudoku/)
6. [N-Queens](https://labuladong.online/algo/practice-in-action/nqueens/)
7. [Generate Parentheses](https://labuladong.online/algo/practice-in-action/generate-parentheses/)
8. [Partition to K Equal Sum Subsets](https://labuladong.online/algo/practice-in-action/partition-to-k-equal-sum-subsets/)
9. [DFS on 2D Grids / Island Problems](https://labuladong.online/algo/frequency-interview/island-dfs-summary/)
10. [Minesweeper](https://labuladong.online/algo/practice-in-action/minesweeper/)
11. [BFS Framework](https://labuladong.online/algo/essential-technique/bfs-framework/)
12. [Sliding Puzzle](https://labuladong.online/algo/practice-in-action/sliding-puzzle/)
13. [Open the Lock](https://labuladong.online/algo/practice-in-action/open-the-lock/)
14. [State-Space BFS for Puzzles and Games](https://labuladong.online/algo/practice-in-action/state-space-bfs/)

### Related LeetCode Problems

| # | Problem | Section |
|---|---------|---------|
| 22 | Generate Parentheses | Practical Backtracking |
| 37 | Sudoku Solver | Constraint Satisfaction |
| 39 | Combination Sum | 9-Variant Framework |
| 40 | Combination Sum II | 9-Variant Framework |
| 46 | Permutations | 9-Variant Framework |
| 47 | Permutations II | 9-Variant Framework |
| 51 | N-Queens | Constraint Satisfaction |
| 52 | N-Queens II | Constraint Satisfaction |
| 77 | Combinations | 9-Variant Framework |
| 78 | Subsets | 9-Variant Framework |
| 90 | Subsets II | 9-Variant Framework |
| 200 | Number of Islands | Island Problems |
| 216 | Combination Sum III | 9-Variant Framework |
| 529 | Minesweeper | Island Problems |
| 694 | Number of Distinct Islands | Island Problems |
| 695 | Max Area of Island | Island Problems |
| 698 | Partition to K Equal Sum Subsets | Practical Backtracking |
| 752 | Open the Lock | State-Space BFS |
| 773 | Sliding Puzzle | State-Space BFS |
| 864 | Shortest Path to Get All Keys | Augmented-State BFS |
| 1020 | Number of Enclaves | Island Problems |
| 1254 | Number of Closed Islands | Island Problems |
| 1293 | Shortest Path with Obstacle Elimination | Augmented-State BFS |
| 1905 | Count Sub Islands | Island Problems |
