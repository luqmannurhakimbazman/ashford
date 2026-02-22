# Classic Interview Problems

Frequently-tested interview problems with elegant solutions. Each has a distinct core trick. For interval scheduling (greedy approach), see `greedy-algorithms.md`. For two-pointer and prefix sum techniques, see `array-techniques.md`. For binary search on answer, see `binary-search-framework.md`.

---

## Quick Reference Table

| Pattern | Recognition Signals | Key Problems | Section |
|---------|-------------------|--------------|---------|
| Trapping Rain Water | "Trap water between bars", "container with most water" | Trapping Rain Water (42), Container With Most Water (11) | 1 |
| Ugly Numbers | "Only prime factors 2, 3, 5", "nth ugly number", "super ugly" | Ugly Number (263), Ugly Number II (264), Super Ugly (313), Ugly Number III (1201) | 2 |
| Missing/Duplicate Elements | "Find missing number", "find duplicate", "array as hash map" | Set Mismatch (645), Find Missing (268), Find Duplicate (287) | 3 |
| Pancake Sorting | "Sort using only flip operations", "reverse prefix" | Pancake Sorting (969) | 4 |
| Perfect Rectangle | "Tiles form exact rectangle", "no overlap no gap" | Perfect Rectangle (391) | 5 |
| Consecutive Subsequences | "Split into consecutive subsequences", "hand of straights" | Split Array (659), Hand of Straights (846) | 6 |
| Interval Problems | "Merge intervals", "interval intersection", "remove covered" | Merge Intervals (56), Interval Intersection (986), Remove Covered (1288) | 7 |
| String Multiplication | "Multiply two number strings", "big number multiplication" | Multiply Strings (43) | 8 |

---

## 1. Trapping Rain Water

### Problem (LC 42)

Given n non-negative integers representing an elevation map, compute how much water it can trap after raining.

### Core Insight

Water at position i = `min(max_left[i], max_right[i]) - height[i]`. The water level at each bar is determined by the shorter of the tallest bars on each side.

### Approach 1: Brute Force → Memo Table

**Brute force:** For each position, scan left and right to find the max heights. O(n²).

**Memo optimization:** Precompute `max_left[i]` and `max_right[i]` arrays. O(n) time, O(n) space.

```python
def trap_memo(height: list[int]) -> int:
    """Precompute max heights from both sides."""
    n = len(height)
    if n == 0:
        return 0

    max_left = [0] * n
    max_right = [0] * n

    max_left[0] = height[0]
    for i in range(1, n):
        max_left[i] = max(max_left[i - 1], height[i])

    max_right[n - 1] = height[n - 1]
    for i in range(n - 2, -1, -1):
        max_right[i] = max(max_right[i + 1], height[i])

    water = 0
    for i in range(n):
        water += min(max_left[i], max_right[i]) - height[i]
    return water
```

### Approach 2: Two Pointers (Optimal)

**Key insight:** You don't need both max arrays simultaneously. Use two pointers from the ends, tracking the running max from each side.

```python
def trap(height: list[int]) -> int:
    """Two pointers: O(n) time, O(1) space."""
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0

    while left < right:
        left_max = max(left_max, height[left])
        right_max = max(right_max, height[right])

        if left_max < right_max:
            water += left_max - height[left]
            left += 1
        else:
            water += right_max - height[right]
            right -= 1

    return water
```

**Why two pointers work:** When `left_max < right_max`, we KNOW the water at `left` is bounded by `left_max` regardless of what's between — there's already a taller bar on the right.

*Socratic prompt: "If left_max is 3 and right_max is 5, do we need to know the exact right_max to compute water at the left pointer? What's the minimum information needed?"*

### Container With Most Water (LC 11)

Different problem — find two lines that form a container holding the most water. The width shrinks as pointers move inward, so always move the shorter line.

```python
def max_area(height: list[int]) -> int:
    """Two pointers: move the shorter side inward."""
    left, right = 0, len(height) - 1
    result = 0
    while left < right:
        area = min(height[left], height[right]) * (right - left)
        result = max(result, area)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return result
```

*Socratic prompt: "Why do we move the shorter pointer? What happens to the area if we move the taller one instead?"*

### Comparison

| Problem | Water Calculation | Technique | Complexity |
|---------|------------------|-----------|------------|
| Trapping Rain Water (42) | Sum of water at each bar | Two pointers or memo | O(n) / O(1) |
| Container With Most Water (11) | Max rectangle between two bars | Two pointers (greedy) | O(n) / O(1) |

---

## 2. Ugly Numbers

### Definition

An **ugly number** is a positive integer whose only prime factors are 2, 3, and 5. The sequence: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, ...

### Check Ugly (LC 263)

```python
def is_ugly(n: int) -> bool:
    """Divide out all factors of 2, 3, 5. If result is 1, it's ugly."""
    if n <= 0:
        return False
    for p in [2, 3, 5]:
        while n % p == 0:
            n //= p
    return n == 1
```

### Nth Ugly Number (LC 264)

**Core insight:** Each ugly number is generated by multiplying a previous ugly number by 2, 3, or 5. Use three pointers to track which ugly number each prime should multiply next.

```python
def nth_ugly_number(n: int) -> int:
    """Three-pointer merge: O(n) time, O(n) space."""
    ugly = [0] * n
    ugly[0] = 1
    p2 = p3 = p5 = 0  # Pointers for each prime factor

    for i in range(1, n):
        next2 = ugly[p2] * 2
        next3 = ugly[p3] * 3
        next5 = ugly[p5] * 5
        ugly[i] = min(next2, next3, next5)
        # Advance ALL pointers that match (handles duplicates like 6 = 2*3 = 3*2)
        if ugly[i] == next2:
            p2 += 1
        if ugly[i] == next3:
            p3 += 1
        if ugly[i] == next5:
            p5 += 1

    return ugly[n - 1]
```

**Why three pointers?** Think of it as merging three sorted sequences:
- `1×2, 2×2, 3×2, 4×2, 5×2, 6×2, 8×2, ...`
- `1×3, 2×3, 3×3, 4×3, 5×3, ...`
- `1×5, 2×5, 3×5, 4×5, ...`

Each sequence is a sorted list of ugly numbers multiplied by a prime. We merge them.

*Socratic prompt: "Why do we check ALL three conditions (not just one) when advancing pointers? What happens if ugly[i] equals both next2 and next3?"*

### Super Ugly Number (LC 313)

Generalize to k primes instead of just {2, 3, 5}. Use k pointers or a min-heap.

```python
import heapq

def nth_super_ugly(n: int, primes: list[int]) -> int:
    """Heap-based merge for k prime factors."""
    ugly = [1]
    # Heap entries: (value, prime, pointer_index)
    heap = [(p, p, 0) for p in primes]
    heapq.heapify(heap)

    while len(ugly) < n:
        val, prime, idx = heapq.heappop(heap)
        if val != ugly[-1]:  # Skip duplicates
            ugly.append(val)
        heapq.heappush(heap, (prime * ugly[idx + 1], prime, idx + 1))

    return ugly[n - 1]
```

### Ugly Number III (LC 1201)

Find the nth number divisible by a, b, or c. Use binary search + inclusion-exclusion.

```python
from math import gcd

def nth_ugly_number_iii(n: int, a: int, b: int, c: int) -> int:
    """Binary search on answer with inclusion-exclusion."""
    def lcm(x, y):
        return x // gcd(x, y) * y

    ab = lcm(a, b)
    ac = lcm(a, c)
    bc = lcm(b, c)
    abc = lcm(ab, c)

    def count(x):
        """Count numbers ≤ x divisible by a, b, or c."""
        return x // a + x // b + x // c - x // ab - x // ac - x // bc + x // abc

    lo, hi = 1, 2 * 10**9
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if count(mid) < n:
            lo = mid + 1
        else:
            hi = mid
    return lo
```

*Socratic prompt: "For Ugly Number III, why can't we use the pointer-merge approach from LC 264? What's fundamentally different about this problem?"*

---

## 3. Missing/Duplicate Elements

### Core Insight: Array as Hash Map

When an array contains numbers in range `[1, n]` and has length n, you can use **index mapping**: place number `i` at index `i-1`. After rearranging, any mismatch reveals the missing or duplicate element.

### Set Mismatch (LC 645)

Array has one duplicate and one missing number.

```python
def find_error_nums(nums: list[int]) -> list[int]:
    """Use index mapping to find duplicate and missing."""
    n = len(nums)
    duplicate = missing = -1

    # Mark visited by negating the value at the corresponding index
    for num in nums:
        idx = abs(num) - 1
        if nums[idx] < 0:
            duplicate = abs(num)
        else:
            nums[idx] = -nums[idx]

    # The index that's still positive is the missing number
    for i in range(n):
        if nums[i] > 0:
            missing = i + 1
            break

    return [duplicate, missing]
```

**Why negate?** We use the sign of `nums[i]` as a visited flag for number `i+1`. If we try to negate an already-negative value, we've found the duplicate. The index with a positive value corresponds to the missing number.

### Alternative: XOR + Sum

```python
def find_error_nums_math(nums: list[int]) -> list[int]:
    """Use sum and sum-of-squares to find both values."""
    n = len(nums)
    actual_sum = sum(nums)
    expected_sum = n * (n + 1) // 2
    actual_sq_sum = sum(x * x for x in nums)
    expected_sq_sum = sum(i * i for i in range(1, n + 1))

    diff = expected_sum - actual_sum          # missing - duplicate
    sq_diff = expected_sq_sum - actual_sq_sum  # missing² - duplicate²
    # missing² - duplicate² = (missing + duplicate)(missing - duplicate)
    total = sq_diff // diff  # missing + duplicate

    missing = (diff + total) // 2
    duplicate = (total - diff) // 2
    return [duplicate, missing]
```

*Socratic prompt: "The sign-flipping trick uses the array itself as a hash map. What constraint on the values makes this possible? Could you use this trick if values ranged from 0 to 2n?"*

---

## 4. Pancake Sorting

### Problem (LC 969)

Sort an array using only "pancake flips" — reverse the first k elements. Return the sequence of k values used.

### Core Insight

To place the largest unsorted element: find it, flip it to the front, then flip it to its correct position. Repeat for the next largest.

```python
def pancake_sort(arr: list[int]) -> list[int]:
    """Sort by repeatedly flipping the largest element into place."""
    result = []
    n = len(arr)

    for size in range(n, 1, -1):
        # Find index of the max element in arr[0..size-1]
        max_idx = arr.index(size)

        if max_idx == size - 1:
            continue  # Already in place

        # Flip max to front (if not already there)
        if max_idx > 0:
            result.append(max_idx + 1)
            arr[:max_idx + 1] = arr[:max_idx + 1][::-1]

        # Flip max to its correct position
        result.append(size)
        arr[:size] = arr[:size][::-1]

    return result
```

**Complexity:** O(n²) time (n iterations, each with an O(n) scan and O(n) reverse). At most 2n flips.

*Socratic prompt: "Walk through pancake sorting [3, 2, 4, 1]. How do you get 4 to the end? Then how do you get 3 to position 3?"*

---

## 5. Perfect Rectangle

### Problem (LC 391)

Given n axis-aligned rectangles, determine if they form an exact cover of a rectangular region (no overlaps, no gaps).

### Core Insight

Two conditions must hold:
1. **Area check:** Sum of all small rectangles' areas = area of the bounding rectangle
2. **Corner check:** Every corner point should appear an even number of times, EXCEPT the four corners of the bounding rectangle (which appear exactly once)

```python
def is_rectangle_cover(rectangles: list[list[int]]) -> bool:
    """Check area equality and corner parity."""
    area = 0
    corners = set()
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')

    for x1, y1, x2, y2 in rectangles:
        area += (x2 - x1) * (y2 - y1)

        min_x = min(min_x, x1)
        min_y = min(min_y, y1)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)

        # Toggle corners: add if not present, remove if present
        for point in [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]:
            if point in corners:
                corners.remove(point)
            else:
                corners.add(point)

    # Check: exactly 4 corners remain, matching the bounding rectangle
    expected_corners = {(min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)}
    expected_area = (max_x - min_x) * (max_y - min_y)

    return area == expected_area and corners == expected_corners
```

**Why corner toggling works:** Interior corners of adjacent rectangles always appear an even number of times (they're shared). Only the four outer corners of the bounding rectangle appear an odd number of times. The XOR-like toggle (add/remove from set) captures this parity.

*Socratic prompt: "Why isn't the area check alone sufficient? Can you construct a counterexample where areas match but there's an overlap AND a gap?"*

---

## 6. Consecutive Subsequences

### Problem (LC 659)

Given a sorted array of integers, determine if it can be split into subsequences of length ≥ 3 where each subsequence is consecutive (e.g., [1,2,3] or [3,4,5,6]).

### Core Insight

Use two hash maps:
- `freq`: remaining count of each number
- `need`: count of subsequences that "need" this number as their next element

For each number, either extend an existing subsequence or start a new one (requiring the next two numbers to exist).

```python
from collections import Counter

def is_possible(nums: list[int]) -> bool:
    """Greedy: extend existing subsequences or start new length-3 ones."""
    freq = Counter(nums)
    need = Counter()  # need[x] = number of subsequences waiting for x

    for num in nums:
        if freq[num] == 0:
            continue

        freq[num] -= 1

        if need[num] > 0:
            # Extend an existing subsequence
            need[num] -= 1
            need[num + 1] += 1
        elif freq[num + 1] > 0 and freq[num + 2] > 0:
            # Start a new subsequence of length 3
            freq[num + 1] -= 1
            freq[num + 2] -= 1
            need[num + 3] += 1
        else:
            return False

    return True
```

**Greedy choice:** Always prefer extending an existing subsequence over starting a new one. This is optimal because extending doesn't consume extra numbers, while starting a new one requires two additional consecutive numbers.

*Socratic prompt: "Why do we prioritize extending an existing subsequence over starting a new one? Can you construct a case where starting a new one first leads to failure?"*

---

## 7. Interval Problems

### Overview

Three core interval operations, all requiring sorted input. For greedy interval scheduling (max non-overlapping), see `greedy-algorithms.md`.

### Interview Clarification Tips

Before solving any interval problem, clarify with the interviewer:
- Are intervals `[a, b]` always guaranteed to have `a < b` strictly, or can `a == b` (zero-length intervals)?
- Do intervals `[1, 2]` and `[2, 3]` overlap? (Problem-dependent — some define overlap as sharing a point, others require a range.)

### Corner Cases

- No intervals (empty input)
- Single interval
- Two intervals (overlapping vs non-overlapping)
- Intervals where one starts exactly where another ends (e.g., `[1, 2]` and `[2, 3]`)
- Duplicate intervals (e.g., `[1, 3]` and `[1, 3]`)
- One interval fully consumed by another (e.g., `[1, 5]` contains `[2, 3]`)
- All intervals non-overlapping
- All intervals overlapping into one merged interval

### Utility Functions

```python
def is_overlap(a, b):
    """Check if two intervals [a0, a1] and [b0, b1] overlap."""
    return a[0] <= b[1] and b[0] <= a[1]

def merge_two(a, b):
    """Merge two overlapping intervals into one."""
    return [min(a[0], b[0]), max(a[1], b[1])]
```

### Foundational Technique

**Sort by starting point.** Nearly every interval problem begins with sorting intervals by their start value. This ensures you only need to compare each interval with the previous one (or a running state), reducing the problem from O(n^2) comparisons to O(n).

### Essential & Recommended Practice Questions

| Problem | Difficulty | Key Technique |
|---------|-----------|---------------|
| Insert Interval (57) | Medium | Binary search or linear scan to find insertion point |
| Merge Intervals (56) | Medium | Sort + merge adjacent |
| Meeting Rooms (252) | Easy | Sort + check any overlap |
| Meeting Rooms II (253) | Medium | Min-heap or sweep line for concurrent meetings |
| Non-overlapping Intervals (435) | Medium | Greedy — sort by end, count removals (see `greedy-algorithms.md`) |

### Merge Intervals (LC 56)

Sort by start time. If the current interval overlaps with the previous, merge them.

```python
def merge(intervals: list[list[int]]) -> list[list[int]]:
    """Merge overlapping intervals."""
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
```

### Interval List Intersections (LC 986)

Two sorted interval lists — find all intersections. Use two pointers.

```python
def interval_intersection(
    first: list[list[int]], second: list[list[int]]
) -> list[list[int]]:
    """Two-pointer intersection of sorted interval lists."""
    i = j = 0
    result = []
    while i < len(first) and j < len(second):
        lo = max(first[i][0], second[j][0])
        hi = min(first[i][1], second[j][1])
        if lo <= hi:
            result.append([lo, hi])
        # Advance the pointer with the smaller endpoint
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return result
```

### Remove Covered Intervals (LC 1288)

Remove intervals that are completely covered by another. Sort by start ascending, then by end descending (so longer intervals come first at the same start).

```python
def remove_covered_intervals(intervals: list[list[int]]) -> int:
    """Count intervals that are not covered by any other."""
    # Sort by start asc, then end desc
    intervals.sort(key=lambda x: (x[0], -x[1]))
    count = 0
    max_end = 0
    for _, end in intervals:
        if end > max_end:
            count += 1
            max_end = end
        # If end <= max_end, this interval is covered → skip
    return count
```

### Comparison

| Problem | Sort By | Technique | Key Condition |
|---------|---------|-----------|---------------|
| Merge (56) | Start asc | Extend merged[-1] | `start ≤ prev_end` |
| Intersection (986) | Pre-sorted | Two pointers | `lo ≤ hi` |
| Remove Covered (1288) | Start asc, end desc | Track max end | `end ≤ max_end` → covered |
| Non-overlapping (435) | End asc | Greedy count | `start ≥ prev_end` (see `greedy-algorithms.md`) |

*Socratic prompt: "For merge intervals, why do we sort by start time? Could we sort by end time instead? What would change?"*

---

## 8. String Multiplication

### Problem (LC 43)

Multiply two non-negative integers represented as strings. Return the product as a string. Cannot use built-in big integer libraries.

### Core Insight

Simulate grade-school multiplication. The key observation: digit `num1[i] × num2[j]` contributes to result positions `i+j` and `i+j+1`.

```python
def multiply(num1: str, num2: str) -> str:
    """Grade-school multiplication: O(m*n) time."""
    m, n = len(num1), len(num2)
    result = [0] * (m + n)

    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            product = int(num1[i]) * int(num2[j])
            p1, p2 = i + j, i + j + 1
            total = product + result[p2]

            result[p2] = total % 10
            result[p1] += total // 10

    # Convert to string, skip leading zeros
    result_str = ''.join(map(str, result)).lstrip('0')
    return result_str or '0'
```

**Why positions i+j and i+j+1?** In grade-school multiplication, multiplying the i-th digit from the top (0-indexed from left) with the j-th digit from the bottom produces a partial product. Its ones digit goes to position `i+j+1` and any carry goes to position `i+j` in the result array.

*Socratic prompt: "Try multiplying 12 × 34 by hand using the position formula. Verify that num1[0]×num2[0] = 1×3 = 3 lands at position 0+0+1 = 1 (tens place of the result 408)."*

---

## Attribution

Content synthesized from labuladong's algorithmic guide, Chapter 4 — "Other Common Techniques: Classic Interview Problems." Reorganized and augmented with Socratic teaching prompts, cross-references, and code templates for the leetcode-teacher skill.
