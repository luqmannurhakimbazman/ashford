# Binary Search Framework

A unified framework for binary search that eliminates off-by-one errors. Based on labuladong's "always use closed intervals" approach.

---

## The Core Problem: Off-by-One Errors

Binary search is easy to understand but notoriously hard to implement correctly. The bugs come from:
1. Should `right` start at `len(nums) - 1` or `len(nums)`?
2. Is the condition `left < right` or `left <= right`?
3. Should we return `left` or `right`?
4. Should we set `left = mid + 1` or `left = mid`?

**The solution:** Pick ONE interval convention and derive everything from it. This framework uses **closed intervals `[left, right]`** consistently.

---

## The Unified Framework: Closed Interval `[left, right]`

### Why Closed Intervals?

With `[left, right]`:
- The search space is `nums[left..right]` inclusive
- The loop runs while `left <= right` (search space is non-empty when `left == right`)
- When shrinking, we skip `mid` entirely: `left = mid + 1` or `right = mid - 1`

This convention is consistent across all three variants. No special cases.

### Variant 1: Basic Binary Search (Find Exact Value)

```python
def binary_search(nums, target):
    left, right = 0, len(nums) - 1    # [left, right] closed interval

    while left <= right:               # Search space non-empty
        mid = left + (right - left) // 2   # Prevent overflow
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1             # Target is in [mid+1, right]
        else:
            right = mid - 1            # Target is in [left, mid-1]

    return -1                          # Not found
```

### Variant 2: Find Left Boundary (First Occurrence)

When duplicates exist, find the **leftmost** index where `nums[index] == target`.

```python
def left_bound(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            right = mid - 1            # Don't return — shrink right to find leftmost
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    # After loop: left is the insertion point for target
    if left >= len(nums) or nums[left] != target:
        return -1
    return left
```

**Key insight:** When we find `target`, we don't return immediately. We shrink `right = mid - 1` to keep searching left. After the loop, `left` points to the first occurrence.

### Variant 3: Find Right Boundary (Last Occurrence)

```python
def right_bound(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            left = mid + 1             # Don't return — shrink left to find rightmost
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    # After loop: right is the last occurrence
    if right < 0 or nums[right] != target:
        return -1
    return right
```

### Side-by-Side Comparison

| Aspect | Basic Search | Left Boundary | Right Boundary |
|--------|-------------|---------------|----------------|
| Init | `left=0, right=n-1` | `left=0, right=n-1` | `left=0, right=n-1` |
| Loop | `left <= right` | `left <= right` | `left <= right` |
| Found target | `return mid` | `right = mid - 1` | `left = mid + 1` |
| `nums[mid] < target` | `left = mid + 1` | `left = mid + 1` | `left = mid + 1` |
| `nums[mid] > target` | `right = mid - 1` | `right = mid - 1` | `right = mid - 1` |
| Return | `mid` or `-1` | `left` (check bounds) | `right` (check bounds) |

*Socratic prompt: "The only difference between the three variants is what happens when `nums[mid] == target`. Why does that single change produce left boundary vs right boundary?"*

---

## The `[left, right)` Alternative Convention

Some textbooks use a half-open interval `[left, right)` where `right` is excluded.

| Aspect | Closed `[left, right]` | Half-open `[left, right)` |
|--------|----------------------|--------------------------|
| Init right | `len(nums) - 1` | `len(nums)` |
| Loop condition | `left <= right` | `left < right` |
| Shrink right | `right = mid - 1` | `right = mid` |
| Shrink left | `left = mid + 1` | `left = mid + 1` |

**When to use each:** Pick one and stick with it. The closed interval convention is recommended because it's more intuitive (both endpoints are valid) and the three variants are more uniform.

---

## Binary Search on Answer

### The Pattern

Many optimization problems can be reframed as: "What is the minimum/maximum value X such that some condition is feasible?"

Instead of directly computing the answer, binary search over the space of possible answers and check feasibility for each candidate.

### Recognition Signals

- "Minimize the maximum" or "maximize the minimum"
- "Find the minimum capacity/speed/time such that..."
- The answer space is monotonic: if X works, then X+1 also works (or vice versa)

### Template

```python
def binary_search_on_answer(lo, hi, feasible):
    """
    Find the minimum value in [lo, hi] for which feasible(x) is True.
    Assumes: if feasible(x) is True, then feasible(x+1) is also True.
    """
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if feasible(mid):
            hi = mid - 1       # mid works — try smaller
        else:
            lo = mid + 1       # mid doesn't work — need bigger
    return lo                  # Smallest feasible value
```

### Worked Example: Koko Eating Bananas (LC 875)

**Problem:** Koko eats bananas at speed `k` (bananas/hour). She has `h` hours. Find the minimum `k`.

**Insight:** If she can finish at speed `k`, she can also finish at speed `k+1`. Monotonic! Binary search on `k`.

```python
import math

def min_eating_speed(piles, h):
    def feasible(speed):
        # Can Koko finish all piles in h hours at this speed?
        hours = sum(math.ceil(p / speed) for p in piles)
        return hours <= h

    lo, hi = 1, max(piles)        # Speed range: [1, max pile size]
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if feasible(mid):
            hi = mid - 1
        else:
            lo = mid + 1
    return lo
```

*Socratic prompt: "Why is the answer space `[1, max(piles)]`? What would speed = max(piles) mean? What about speed = 1?"*

### Problems

| Problem | Answer Space | Feasibility Check |
|---------|-------------|-------------------|
| Koko Eating Bananas (875) | `[1, max(piles)]` | Can finish in h hours at this speed? |
| Split Array Largest Sum (410) | `[max(nums), sum(nums)]` | Can split into m subarrays each ≤ mid? |
| Capacity To Ship (1011) | `[max(weights), sum(weights)]` | Can ship in d days with this capacity? |
| Magnetic Force Between Balls (1552) | `[1, max_dist]` | Can place m balls with min distance ≥ mid? |
| Minimize Max Distance to Gas Station (774) | `[0, max_gap]` | Can add k stations so max gap ≤ mid? |

---

## Practical Applications

### Random Pick with Weight (LC 528)

Use prefix sums + left-boundary binary search to pick random indices proportional to weight.

```python
import random

class RandomPickWeight:
    def __init__(self, w):
        self.prefix = []
        total = 0
        for weight in w:
            total += weight
            self.prefix.append(total)
        self.total = total

    def pick_index(self):
        target = random.random() * self.total
        # Find leftmost index where prefix[index] > target
        lo, hi = 0, len(self.prefix) - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if self.prefix[mid] <= target:
                lo = mid + 1
            else:
                hi = mid - 1
        return lo
```

### Advantage Shuffle (LC 870) — "Field Upgrade Strategy"

Sort both arrays. For each opponent value (largest first), assign the smallest value that beats it. If nothing beats it, assign the weakest card.

### Search in Rotated Sorted Array (LC 33)

Binary search with an extra check: determine which half is sorted, then decide which half contains the target.

```python
def search_rotated(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] == target:
            return mid
        # Determine which half is sorted
        if nums[lo] <= nums[mid]:          # Left half is sorted
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:                              # Right half is sorted
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1
```

*Socratic prompt: "In a rotated sorted array, at least one half is always sorted. How do you determine which one? Why does that help?"*

---

## Decision Tree: Which Variant to Use

```
What are you searching for?
├── An exact value in a sorted array?
│   └── Basic Binary Search
├── The first/leftmost occurrence of a value?
│   └── Left Boundary
├── The last/rightmost occurrence of a value?
│   └── Right Boundary
├── The insertion point (where value would go)?
│   └── Left Boundary (returns insertion point naturally)
├── "Minimize the maximum" / "maximize the minimum"?
│   └── Binary Search on Answer
└── Searching in a modified sorted structure (rotated, matrix)?
    └── Basic Binary Search + determine which portion is sorted
```

---

## Common Pitfalls

| Pitfall | Cause | Fix |
|---------|-------|-----|
| Integer overflow in `(left + right) / 2` | Sum exceeds int range | Use `left + (right - left) // 2` |
| Infinite loop | `left = mid` without +1 in certain conditions | Always use `mid + 1` or `mid - 1` with closed intervals |
| Off-by-one on boundary search | Wrong return value after loop | Return `left` for left boundary, `right` for right boundary |
| Wrong search space for "search on answer" | Bounds too tight or too loose | lo = absolute minimum possible, hi = absolute maximum possible |
| Forgetting to check bounds after loop | `left` or `right` might be out of array | Always check `0 <= left < n` before accessing `nums[left]` |

---

## Attribution

The unified binary search framework in this file is inspired by and adapted from labuladong's algorithmic guides (labuladong.online), particularly the "Binary Search in Detail" and "Binary Search on Answer" articles from Chapter 1 "Data Structure Algorithms." The closed-interval convention and side-by-side comparison have been restructured and annotated for Socratic teaching use.
