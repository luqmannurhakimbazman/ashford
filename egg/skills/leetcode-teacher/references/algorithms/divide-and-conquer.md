# The Divide and Conquer Framework

Splitting problems to eliminate superlinear cross-work, with the Master Theorem for complexity analysis.

**Prerequisites:** Recursion, merge sort concept. See `references/frameworks/algorithm-frameworks.md` for the sorting insight (quick sort = pre-order, merge sort = post-order).

---

## Why Divide and Conquer Works: The (a+b)² Insight

For superlinear problems, splitting the input reduces total work. Consider: `(a+b)² = a² + 2ab + b² > a² + b²`. The cross-term `2ab` is the work you **eliminate** by dividing. This is why merge sort (O(N log N)) beats insertion sort (O(N²)) — splitting removes the quadratic interaction between halves.

*Socratic prompt: "If sorting N elements takes N² comparisons with brute force, and you split into two halves of N/2, what's the total work for the halves? What happened to the 'cross' comparisons?"*

## D&C vs Plain Recursion

Not every recursive function is divide and conquer. True D&C must satisfy:

1. **Subproblems reduce complexity** — splitting must eliminate superlinear cross-work
2. **Subproblems are independent** — solving one half must not depend on the other (this is what separates D&C from DP)
3. **Combine step is efficient** — merging results must be cheaper than re-solving

Binary search is technically "decrease and conquer" — it discards one half entirely rather than solving both halves independently.

## Pre-Order vs Post-Order D&C

| Style | Work Happens | Example | Sorting Analogy |
|-------|-------------|---------|-----------------|
| Pre-order D&C | Before recursion (partition, then recurse) | Quick Sort, Quick Select | Partition first, halves are independent |
| Post-order D&C | After recursion (recurse, then merge) | Merge Sort, Closest Pair | Solve halves first, combine results |

```python
# Pre-order D&C: partition THEN recurse
def quick_select(arr, lo, hi, k):
    if lo == hi:
        return arr[lo]
    pivot = partition(arr, lo, hi)
    if k == pivot:
        return arr[k]
    elif k < pivot:
        return quick_select(arr, lo, pivot - 1, k)   # Only one side!
    else:
        return quick_select(arr, pivot + 1, hi, k)

# Post-order D&C: recurse THEN combine
def count_inversions(arr, lo, hi):
    if lo >= hi:
        return 0
    mid = (lo + hi) // 2
    left_inv = count_inversions(arr, lo, mid)
    right_inv = count_inversions(arr, mid + 1, hi)
    split_inv = merge_and_count(arr, lo, mid, hi)     # Combine step
    return left_inv + right_inv + split_inv
```

## The Master Theorem (Simplified)

For recurrences of the form `T(N) = a·T(N/b) + O(N^c)`:

| Condition | Result | Intuition |
|-----------|--------|-----------|
| c < log_b(a) | T(N) = O(N^(log_b(a))) | Subproblems dominate — tree has many leaves |
| c = log_b(a) | T(N) = O(N^c · log N) | Work balanced across levels |
| c > log_b(a) | T(N) = O(N^c) | Combine step dominates — root does most work |

**Common cases:**
- Merge sort: `T(N) = 2T(N/2) + O(N)` → a=2, b=2, c=1. `c = log₂(2) = 1` → **O(N log N)**
- Binary search: `T(N) = T(N/2) + O(1)` → a=1, b=2, c=0. `c = log₂(1) = 0` → **O(log N)**
- Karatsuba multiplication: `T(N) = 3T(N/2) + O(N)` → a=3, b=2, c=1. `c < log₂(3) ≈ 1.58` → **O(N^1.58)**

*Socratic prompt: "Merge sort splits into 2 subproblems of size N/2 and does O(N) merge work. What are a, b, and c? Which Master Theorem case applies?"*

## See Also

- `references/algorithms/sorting-algorithms.md` — Full sorting algorithm implementations
- `references/frameworks/algorithm-frameworks.md` — Recursion as tree traversal (pre-order vs post-order)

---

## Attribution

Extracted from the Divide and Conquer Framework section of `algorithm-frameworks.md`, inspired by labuladong's algorithmic guides (labuladong.online).
