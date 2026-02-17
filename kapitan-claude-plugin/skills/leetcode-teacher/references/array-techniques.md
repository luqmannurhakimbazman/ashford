# Array Techniques

Array fundamentals and essential manipulation patterns. Covers core concepts, interview tips, and five key patterns that transform brute-force O(N) or O(N^2) range operations into elegant O(1) or O(N) solutions.

---

## Array Fundamentals

### Common Terms

- **Subarray** — a contiguous sequence of elements within an array. Example: `[2, 3, 6, 1, 5, 4]` → `[3, 6, 1]` is a subarray.
- **Subsequence** — a sequence derived by deleting zero or more elements without changing order. Example: `[2, 3, 6, 1, 5, 4]` → `[3, 1, 5]` is a subsequence (not necessarily contiguous).

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Access `arr[i]` | O(1) | Direct index calculation |
| Search (unsorted) | O(N) | Linear scan |
| Search (sorted) | O(log N) | Binary search |
| Insert (at end) | O(1) amortized | Dynamic array resize |
| Insert (at index) | O(N) | Shift elements right |
| Remove (at end) | O(1) | No shifting needed |
| Remove (at index) | O(N) | Shift elements left |

### Interview Tips

- **Slicing creates copies.** In Python, `arr[i:j]` is O(j - i). Avoid slicing inside loops — pass indices instead.
- **Beware off-by-one errors.** Clarify whether indices are inclusive or exclusive. Fencepost errors are the #1 bug in array problems.
- **Clarify duplicates.** Ask: "Can the array contain duplicates?" This often changes the approach entirely (e.g., Two Sum with duplicates, binary search with duplicates).
- **In-place vs extra space.** Many interviewers prefer in-place solutions. Ask about space constraints.
- **Pre-sorting unlocks techniques.** If the array isn't required to stay in order, sorting enables binary search, two pointers, and duplicate skipping.

### Corner Cases

- Empty array
- Array with 1 or 2 elements
- Array with all identical elements
- Array with duplicates (affects uniqueness assumptions)
- Very large or very small numbers (overflow)

---

## Core Array Techniques

### Sliding Window

Maintain a window `[left, right]` that expands and shrinks based on a condition. See `references/algorithm-frameworks.md` for the full sliding window template with the three-question framework (when to expand, when to shrink, when to update).

### Two Pointers

**Opposite direction:** Start pointers at both ends, move inward. Used for sorted array problems and palindrome checks.

```python
def two_sum_sorted(nums, target):
    """Two Sum II (sorted array). O(N) time, O(1) space."""
    left, right = 0, len(nums) - 1
    while left < right:
        total = nums[left] + nums[right]
        if total == target:
            return [left, right]
        elif total < target:
            left += 1
        else:
            right -= 1
```

**Same direction:** Both pointers move forward, often at different speeds. Used for removing duplicates, partitioning.

```python
def remove_duplicates(nums):
    """Remove duplicates in-place from sorted array. O(N) time, O(1) space."""
    if not nums:
        return 0
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1
```

### Traversing from the Right

Some problems become simpler when you iterate from right to left. Useful for "next greater element" and cumulative computations from the end.

### Sorting the Array First

Sorting (O(N log N)) can simplify an O(N^2) or harder problem. After sorting, you can apply binary search or two pointers. Ask: "Would sorting help here, and is it allowed?"

### Index as a Hash Key

For arrays of integers in range `[0, N-1]` or `[1, N]`, use the array index itself as a hash key. This achieves O(1) space for problems that would otherwise need a hash set.

```python
def find_duplicates(nums):
    """Find duplicates in [1, N] range. O(N) time, O(1) space."""
    result = []
    for num in nums:
        idx = abs(num) - 1
        if nums[idx] < 0:
            result.append(abs(num))  # Already seen
        else:
            nums[idx] = -nums[idx]   # Mark as seen
    return result
```

### Traversing the Array More Than Once

Two passes through the array is still O(N). Don't be afraid to make a first pass to collect information and a second pass to compute the answer. Example: "Product of Array Except Self" uses left-pass and right-pass prefix products.

---

## Essential & Recommended Practice Questions

### Essential (Do These First)

| Problem | Difficulty | Key Technique |
|---------|-----------|---------------|
| Two Sum (1) | Easy | Hash map |
| Best Time to Buy and Sell Stock (121) | Easy | Track min price |
| Product of Array Except Self (238) | Medium | Left/right prefix products |
| Maximum Subarray (53) | Medium | Kadane's algorithm |
| Contains Duplicate (217) | Easy | Hash set |
| Maximum Product Subarray (152) | Medium | Track min and max products |

### Recommended (Build Depth)

| Problem | Difficulty | Key Technique |
|---------|-----------|---------------|
| 3Sum (15) | Medium | Sort + two pointers |
| Container With Most Water (11) | Medium | Two pointers (opposite) |
| Sliding Window Maximum (239) | Hard | Monotonic deque |
| Trapping Rain Water (42) | Hard | Two pointers or stack |
| First Missing Positive (41) | Hard | Index as hash key |
| Subarray Sum Equals K (560) | Medium | Prefix sum + hash map |

---

## Quick Reference Table

| Pattern | Key Insight | When to Use | Complexity |
|---------|-------------|-------------|------------|
| Prefix Sum (1D) | Precompute cumulative sums for O(1) range queries | "Sum of subarray", "range sum query" | O(N) build, O(1) query |
| Prefix Sum (2D) | Extend to matrix with inclusion-exclusion | "Sum of submatrix", "2D range sum" | O(MN) build, O(1) query |
| Difference Array | O(1) range increment, O(N) reconstruction | "Increment range [i,j] by val", "booking", "car pooling" | O(1) update, O(N) reconstruct |
| 2D Traversal | Spiral, rotation, diagonal templates | "Spiral order", "rotate image", "diagonal traverse" | O(MN) |
| Rabin-Karp | Rolling hash for O(N) substring matching | "Find pattern in string", "repeated DNA sequences" | O(N) average |

---

## 1. Prefix Sum (1D)

### Key Insight

If you need the sum of a subarray `nums[i..j]` multiple times, precompute a prefix sum array once. Then any range sum is a single subtraction: `prefix[j+1] - prefix[i]`.

**Why it works:** `prefix[k]` stores the sum of `nums[0..k-1]`. The sum of `nums[i..j]` equals "sum of first j+1 elements" minus "sum of first i elements."

### Template

```python
class PrefixSum:
    def __init__(self, nums):
        # prefix[0] = 0, prefix[i] = nums[0] + ... + nums[i-1]
        self.prefix = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            self.prefix[i + 1] = self.prefix[i] + nums[i]

    def query(self, i, j):
        """Return sum of nums[i..j] inclusive."""
        return self.prefix[j + 1] - self.prefix[i]
```

### Recognition Signals

- "Sum of elements between index i and j" (repeated queries)
- "Number of subarrays with sum equal to k"
- "Product of array except self" (use prefix product from left and right)

### Socratic Prompts

- *"If I ask you for the sum of nums[2..5], then nums[1..4], then nums[3..7] — what work are you repeating each time?"*
- *"Can you precompute something once so each query takes O(1)?"*

### Problems

| Problem | Key Twist |
|---------|-----------|
| Range Sum Query - Immutable (303) | Direct prefix sum application |
| Subarray Sum Equals K (560) | Prefix sum + hash map: `count[prefix - k]` |
| Product of Array Except Self (238) | Left prefix product * right prefix product |
| Continuous Subarray Sum (523) | Prefix sum mod k + hash map |
| Path Sum III (437) | Prefix sum on tree paths |

---

## 2. Prefix Sum (2D)

### Key Insight

Extend 1D prefix sums to a matrix. `prefix[i][j]` stores the sum of all elements in the submatrix from `(0,0)` to `(i-1,j-1)`. Use inclusion-exclusion to query any rectangular region in O(1).

### Template

```python
class NumMatrix:
    def __init__(self, matrix):
        if not matrix:
            return
        m, n = len(matrix), len(matrix[0])
        # prefix[i][j] = sum of matrix[0..i-1][0..j-1]
        self.prefix = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                self.prefix[i + 1][j + 1] = (
                    matrix[i][j]
                    + self.prefix[i][j + 1]
                    + self.prefix[i + 1][j]
                    - self.prefix[i][j]       # Subtract double-counted region
                )

    def sum_region(self, r1, c1, r2, c2):
        """Return sum of matrix[r1..r2][c1..c2] inclusive."""
        return (
            self.prefix[r2 + 1][c2 + 1]
            - self.prefix[r1][c2 + 1]
            - self.prefix[r2 + 1][c1]
            + self.prefix[r1][c1]             # Add back double-subtracted corner
        )
```

### The Inclusion-Exclusion Diagram

```
To get sum of region (r1,c1) to (r2,c2):

+-------+-------+
|   A   |   B   |
|       |       |
+-------+-------+  <- r1
|   C   | QUERY |
|       |       |
+-------+-------+  <- r2
        ^       ^
       c1      c2

Answer = Total(A+B+C+Q) - Top(A+B) - Left(A+C) + Corner(A)
```

*Socratic prompt: "Why do we add back the corner region? What happens if we don't?"*

### Problems

| Problem | Key Twist |
|---------|-----------|
| Range Sum Query 2D - Immutable (304) | Direct 2D prefix sum |
| Matrix Block Sum (1314) | Prefix sum + clamp boundaries |

---

## 3. Difference Array

### Key Insight

If you need to increment all elements in a range `[i, j]` by `val` multiple times, don't update each element. Instead, maintain a **difference array** where `diff[i] += val` and `diff[j+1] -= val`. After all updates, reconstruct the result with a prefix sum over `diff`.

**Why it works:** The difference array records where increments start and stop. A prefix sum over it reconstructs the actual values — each `+val` at position `i` "carries forward" until the `-val` at `j+1` cancels it.

### Template

```python
class Difference:
    def __init__(self, nums):
        self.diff = [0] * (len(nums) + 1)
        # Initialize diff from original array
        self.diff[0] = nums[0]
        for i in range(1, len(nums)):
            self.diff[i] = nums[i] - nums[i - 1]

    def increment(self, i, j, val):
        """Add val to all elements in nums[i..j] inclusive. O(1)."""
        self.diff[i] += val
        if j + 1 < len(self.diff):
            self.diff[j + 1] -= val

    def result(self):
        """Reconstruct the array after all increments. O(N)."""
        res = [0] * (len(self.diff) - 1)
        res[0] = self.diff[0]
        for i in range(1, len(res)):
            res[i] = res[i - 1] + self.diff[i]
        return res
```

### Recognition Signals

- "Apply multiple range increments, then query final state"
- "Booking" / "flight bookings" / "car pooling" (add passengers over an interval)
- Any problem with many range updates but only one final read

### Socratic Prompts

- *"If you have 1000 range updates on an array of size 10000, what's the brute force cost? Can you record each update in O(1) and reconstruct once?"*
- *"What's the relationship between a difference array and a prefix sum array?"* (They're inverses!)

### Problems

| Problem | Key Twist |
|---------|-----------|
| Range Addition (370) | Direct difference array application |
| Car Pooling (1094) | Passengers board at `from`, exit at `to` — difference array on timeline |
| Corporate Flight Bookings (1109) | Bookings are range increments on flight seats |

---

## 4. 2D Array Traversal Patterns

### Spiral Order

Traverse a matrix in spiral order by maintaining four boundaries and shrinking them inward.

```python
def spiral_order(matrix):
    if not matrix:
        return []
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # Traverse right across top row
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1

        # Traverse down right column
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1

        # Traverse left across bottom row (if rows remain)
        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1

        # Traverse up left column (if columns remain)
        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1

    return result
```

### Rotate Image (90 degrees clockwise)

**Key insight:** Transpose + reverse each row = 90-degree clockwise rotation.

```python
def rotate(matrix):
    n = len(matrix)
    # Step 1: Transpose (swap matrix[i][j] with matrix[j][i])
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # Step 2: Reverse each row
    for row in matrix:
        row.reverse()
```

*Socratic prompt: "What does transposing do geometrically? After transposing, what single operation completes the rotation?"*

### Diagonal Traversal

```python
def find_diagonal_order(matrix):
    if not matrix:
        return []
    m, n = len(matrix), len(matrix[0])
    result = []
    for d in range(m + n - 1):
        if d % 2 == 0:  # Going up
            r = min(d, m - 1)
            c = d - r
            while r >= 0 and c < n:
                result.append(matrix[r][c])
                r -= 1
                c += 1
        else:            # Going down
            c = min(d, n - 1)
            r = d - c
            while c >= 0 and r < m:
                result.append(matrix[r][c])
                r += 1
                c -= 1
    return result
```

### Problems

| Problem | Pattern |
|---------|---------|
| Spiral Matrix (54) | Four-boundary spiral traversal |
| Spiral Matrix II (59) | Generate spiral-filled matrix |
| Rotate Image (48) | Transpose + reverse rows |
| Diagonal Traverse (498) | Alternating up/down diagonals |

---

## 5. Rabin-Karp String Matching

### Key Insight

Instead of comparing characters one by one (O(M) per position), compute a **rolling hash** of the pattern and slide it across the text. If hashes match, verify the actual strings (to handle collisions).

**Rolling hash formula:** When sliding the window one character right, remove the leftmost character's contribution and add the new character's:

```
new_hash = (old_hash - text[i] * BASE^(M-1)) * BASE + text[i + M]
```

All arithmetic is done modulo a large prime to prevent overflow.

### Template

```python
def rabin_karp(text, pattern):
    n, m = len(text), len(pattern)
    if m > n:
        return -1

    BASE = 256
    MOD = 10**9 + 7

    # Precompute BASE^(m-1) % MOD
    power = pow(BASE, m - 1, MOD)

    # Compute hash of pattern and first window
    p_hash = 0
    t_hash = 0
    for i in range(m):
        p_hash = (p_hash * BASE + ord(pattern[i])) % MOD
        t_hash = (t_hash * BASE + ord(text[i])) % MOD

    for i in range(n - m + 1):
        if t_hash == p_hash:
            # Hash match — verify characters (handle collisions)
            if text[i:i + m] == pattern:
                return i

        # Roll the hash forward
        if i < n - m:
            t_hash = ((t_hash - ord(text[i]) * power) * BASE + ord(text[i + m])) % MOD
            t_hash = (t_hash + MOD) % MOD  # Ensure non-negative

    return -1
```

### Recognition Signals

- "Find first occurrence of pattern in text" (when you want average O(N))
- "Find repeated substrings of length K" (hash each window, check for duplicates)
- "Longest duplicate substring" (binary search on length + Rabin-Karp)

### Socratic Prompts

- *"Why is naive string matching O(NM)? What work is repeated when you slide the window by one?"*
- *"If two strings have the same hash, are they necessarily equal? What do we call this situation?"*

### Problems

| Problem | Key Twist |
|---------|-----------|
| Repeated DNA Sequences (187) | Fixed-length window, hash all windows, find duplicates |
| Find the Index of the First Occurrence (28) | Classic pattern matching |
| Longest Duplicate Substring (1044) | Binary search on length + Rabin-Karp |

---

## Pattern Connections

| If You Know... | Then You Can Solve... |
|----------------|----------------------|
| Prefix sum | Subarray sum problems → hash map + prefix sum for "count subarrays with sum K" |
| Difference array | Any "batch range update" → inverse of prefix sum |
| 2D prefix sum | Extend any 1D prefix sum solution to matrices |
| Rabin-Karp | Replace O(NM) string matching with O(N) average; combine with binary search for "longest repeated X" |

*Socratic prompt: "Prefix sum and difference array are inverses. If prefix sum answers range queries, what does the difference array answer?"*

---

## Attribution

The patterns and techniques in this file are inspired by and adapted from labuladong's algorithmic guides (labuladong.online), particularly the prefix sum and difference array articles from Chapter 1 "Data Structure Algorithms." Templates have been restructured and annotated for Socratic teaching use.
