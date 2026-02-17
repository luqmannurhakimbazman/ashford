# Sorting Algorithms

All 10 major sorting algorithms with labuladong's framework-first approach. Covers complexity analysis, stability, and the tree-traversal insight that unifies divide-and-conquer sorts.

---

## Quick Reference Table

| Algorithm | Time (Best) | Time (Avg) | Time (Worst) | Space | Stable? | In-Place? |
|-----------|-------------|-----------|-------------|-------|---------|-----------|
| Selection Sort | O(N^2) | O(N^2) | O(N^2) | O(1) | No | Yes |
| Bubble Sort | O(N) | O(N^2) | O(N^2) | O(1) | Yes | Yes |
| Insertion Sort | O(N) | O(N^2) | O(N^2) | O(1) | Yes | Yes |
| Shell Sort | O(N log N) | O(N^1.3)* | O(N^2) | O(1) | No | Yes |
| Quick Sort | O(N log N) | O(N log N) | O(N^2) | O(log N) | No | Yes |
| Merge Sort | O(N log N) | O(N log N) | O(N log N) | O(N) | Yes | No |
| Heap Sort | O(N log N) | O(N log N) | O(N log N) | O(1) | No | Yes |
| Counting Sort | O(N + K) | O(N + K) | O(N + K) | O(K) | Yes | No |
| Bucket Sort | O(N + K) | O(N + K) | O(N^2) | O(N + K) | Yes* | No |
| Radix Sort | O(N * D) | O(N * D) | O(N * D) | O(N + K) | Yes | No |

*Shell sort complexity depends on gap sequence. Bucket sort stability depends on inner sort.

---

## Interview Tips

### Know Your Language's Default Sort

| Language | Default Sort | Type |
|----------|-------------|------|
| Python 3.11+ | Powersort (Timsort variant) | Stable, O(N log N) |
| Java | Timsort for objects, Dual-Pivot Quicksort for primitives | Stable (objects), unstable (primitives) |
| C++ | Introsort (quicksort + heapsort + insertion sort) | Unstable, O(N log N) guaranteed |
| JavaScript | Implementation-dependent (often Timsort) | Stable (ES2019+) |

**In interviews:** Use your language's built-in sort confidently. You don't need to implement sorting unless the problem specifically asks for it. Know the time complexity (O(N log N)) and stability guarantees.

### When Counting Sort Beats Comparison Sort

If the input values have a **limited range** (e.g., ages 0-150, grades A-F, numbers 1-1000), counting sort achieves O(N + K) which beats O(N log N). Example: H-Index (LC 274) — citation counts are bounded by the number of papers, making counting sort ideal.

### Corner Cases for Sorting Problems

- Array with all identical elements (should remain unchanged)
- Array with exactly two elements (boundary for divide-and-conquer)
- Array with one element (already sorted, check base case handling)
- Empty array (check for edge case handling)
- Array already sorted in ascending or descending order
- Array with negative numbers (affects counting/radix sort)

### Essential & Recommended Practice Questions

| Problem | Difficulty | Key Technique |
|---------|-----------|---------------|
| Binary Search (704) | Easy | Prerequisite: searching sorted arrays |
| Search in Rotated Sorted Array (33) | Medium | Modified binary search |
| Kth Largest Element in an Array (215) | Medium | Quickselect or heap |
| Merge Intervals (56) | Medium | Sort by start, then merge |
| Median of Two Sorted Arrays (4) | Hard | Binary search on partition |
| Sort Colors (75) | Medium | Dutch national flag / counting sort |

---

## The Comparison Sort Lower Bound

**Theorem:** Any comparison-based sort requires at least O(N log N) comparisons in the worst case.

**Why?** N elements have N! permutations. Each comparison splits possibilities in half. To distinguish all N! outcomes, you need at least log2(N!) ≈ N log N comparisons (by Stirling's approximation).

**Implication:** Selection, bubble, insertion, and shell sort are suboptimal. Quick, merge, and heap sort are optimal among comparison sorts. Counting, bucket, and radix sort beat the bound by not using comparisons — they exploit the structure of the data instead.

**Socratic prompt:** *"If the lower bound is O(N log N), how can counting sort be O(N)? What assumption does it make that comparison sorts don't?"*

---

## The Tree-Traversal Insight for Sorts

Labuladong's key observation connecting sorting to binary trees:

- **Quick sort = pre-order tree traversal:** Partition first (make a decision at the root), then recurse on left and right halves
- **Merge sort = post-order tree traversal:** Recurse on left and right halves first, then merge (combine results at the root)

```python
# Quick sort structure = pre-order
def quick_sort(arr, lo, hi):
    if lo >= hi:
        return
    # PRE-ORDER: partition (root decision)
    pivot = partition(arr, lo, hi)
    # Recurse on children
    quick_sort(arr, lo, pivot - 1)
    quick_sort(arr, pivot + 1, hi)

# Merge sort structure = post-order
def merge_sort(arr, lo, hi):
    if lo >= hi:
        return
    mid = (lo + hi) // 2
    # Recurse on children first
    merge_sort(arr, lo, mid)
    merge_sort(arr, mid + 1, hi)
    # POST-ORDER: merge (combine children's results)
    merge(arr, lo, mid, hi)
```

This is not a metaphor — the call structure is literally a tree walk. Understanding this unlocks both sorts and helps identify which pattern to use in other divide-and-conquer problems.

**Socratic prompt:** *"If quick sort is pre-order and merge sort is post-order, what would an 'in-order' sort look like? Does that even make sense?"*

---

## Elementary Sorts: O(N^2)

### Selection Sort

**Idea:** Find the minimum element in the unsorted portion, swap it to the front. Repeat.

```python
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
```

**Properties:** Always O(N^2) regardless of input. Not stable (swap can move equal elements past each other). Simple but never preferred in practice.

### Bubble Sort

**Idea:** Repeatedly pass through the array, swapping adjacent elements that are out of order. Stop when no swaps occur.

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break  # Already sorted — O(N) best case
```

**Properties:** O(N) best case (already sorted). Stable (only swaps adjacent elements). Useful mainly for educational purposes.

### Insertion Sort

**Idea:** Build the sorted portion one element at a time. For each new element, find its correct position in the sorted portion and insert.

```python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]  # Shift right
            j -= 1
        arr[j + 1] = key
```

**Properties:** O(N) best case (nearly sorted). Stable. **The best elementary sort in practice** — used as the base case in hybrid sorts (Timsort, introsort) because it's fast on small or nearly-sorted arrays due to low overhead and good cache behavior.

**When insertion sort shines:**
- Small arrays (N < ~20)
- Nearly sorted data
- Online sorting (elements arrive one at a time)

**Socratic prompt:** *"Why is insertion sort better than selection sort for nearly-sorted data? What about the inner loop changes?"*

---

## Shell Sort

**Idea:** Insertion sort with decreasing gap sequences. First sort elements that are `gap` positions apart, then reduce the gap until gap = 1 (which is regular insertion sort). Early passes with large gaps move elements large distances cheaply.

```python
def shell_sort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            key = arr[i]
            j = i
            while j >= gap and arr[j - gap] > key:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = key
        gap //= 2
```

**Properties:** Sub-quadratic for good gap sequences (Knuth's sequence: 1, 4, 13, 40, ... gives ~O(N^1.3)). Not stable. In-place. A practical improvement over insertion sort for medium-sized arrays.

---

## Efficient Comparison Sorts: O(N log N)

### Quick Sort

**Idea:** Choose a pivot. Partition the array so elements < pivot go left, elements > pivot go right. Recurse on both halves.

```python
def quick_sort(arr, lo, hi):
    if lo >= hi:
        return
    pivot_idx = partition(arr, lo, hi)
    quick_sort(arr, lo, pivot_idx - 1)
    quick_sort(arr, pivot_idx + 1, hi)

def partition(arr, lo, hi):
    pivot = arr[hi]  # Choose last element as pivot
    i = lo
    for j in range(lo, hi):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[hi] = arr[hi], arr[i]
    return i
```

**Avoiding worst case (O(N^2)):** Worst case occurs when pivot is always the min or max (already sorted input with last-element pivot). Mitigations:
- **Random pivot:** `pivot_idx = random.randint(lo, hi)`, swap to end
- **Median-of-three:** Choose median of first, middle, last elements
- **Shuffle input:** Randomize before sorting

**Why quick sort is fastest in practice:** Despite O(N^2) worst case, quick sort's inner loop is simple (one comparison, one increment), cache-friendly (sequential access), and has low constant factors. It typically outperforms merge sort by 2-3x on random data.

### Merge Sort

**Idea:** Divide the array in half, recursively sort each half, merge the two sorted halves.

```python
def merge_sort(arr, lo, hi):
    if lo >= hi:
        return
    mid = (lo + hi) // 2
    merge_sort(arr, lo, mid)
    merge_sort(arr, mid + 1, hi)
    merge(arr, lo, mid, hi)

def merge(arr, lo, mid, hi):
    temp = arr[lo:hi + 1]  # O(N) extra space
    i, j = 0, mid - lo + 1
    for k in range(lo, hi + 1):
        if i > mid - lo:
            arr[k] = temp[j]; j += 1
        elif j > hi - lo:
            arr[k] = temp[i]; i += 1
        elif temp[i] <= temp[j]:
            arr[k] = temp[i]; i += 1
        else:
            arr[k] = temp[j]; j += 1
```

**Properties:** Always O(N log N) — no worst case. Stable. Requires O(N) extra space. Preferred when stability matters or worst-case guarantees are needed. Python's `sorted()` uses Timsort (merge sort + insertion sort hybrid).

**When to prefer merge sort over quick sort:**
- Stability required (preserving equal-element order)
- Worst-case guarantee needed (no O(N^2) risk)
- Sorting linked lists (merge sort needs O(1) extra space on linked lists)
- External sorting (merging sorted runs from disk)

### Heap Sort

**Idea:** Build a max-heap from the array. Repeatedly extract the max and place it at the end.

```python
def heap_sort(arr):
    n = len(arr)
    # Build max-heap (bottom-up) — O(N)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    # Extract elements one by one — O(N log N)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]  # Move max to end
        heapify(arr, i, 0)               # Restore heap property

def heapify(arr, n, i):
    largest = i
    left, right = 2 * i + 1, 2 * i + 2
    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)
```

**Properties:** O(N log N) guaranteed. In-place (O(1) extra space). Not stable. Combines the best of merge sort (guaranteed O(N log N)) and quick sort (in-place), but slower in practice due to poor cache behavior (jumps around the array in heap operations).

**Socratic prompt:** *"Heap sort is O(N log N) and in-place — sounds perfect. Why isn't it the default sort in language libraries?"*

---

## Non-Comparison Sorts: O(N)

These sorts exploit data structure rather than element comparisons, bypassing the O(N log N) lower bound.

### Counting Sort

**Idea:** Count occurrences of each value, then reconstruct the sorted array from counts.

```python
def counting_sort(arr, max_val):
    count = [0] * (max_val + 1)
    for x in arr:
        count[x] += 1
    idx = 0
    for val in range(max_val + 1):
        for _ in range(count[val]):
            arr[idx] = val
            idx += 1
```

**Constraint:** Only works when values are non-negative integers in a small range [0, K]. Space is O(K).

**When it works:** Sorting ages (0-150), grades (0-100), ASCII characters (0-127).

### Bucket Sort

**Idea:** Distribute elements into buckets (ranges), sort each bucket (often with insertion sort), then concatenate.

```python
def bucket_sort(arr, num_buckets=10):
    if not arr:
        return arr
    min_val, max_val = min(arr), max(arr)
    bucket_range = (max_val - min_val) / num_buckets + 1
    buckets = [[] for _ in range(num_buckets)]
    for x in arr:
        idx = int((x - min_val) / bucket_range)
        buckets[idx].append(x)
    result = []
    for bucket in buckets:
        result.extend(sorted(bucket))  # Sort each bucket
    return result
```

**Constraint:** Works best when input is uniformly distributed. Worst case O(N^2) if all elements land in one bucket.

### Radix Sort

**Idea:** Sort by each digit/character position, from least significant to most significant (LSD) or vice versa (MSD), using a stable sub-sort (counting sort) at each step.

```python
def radix_sort(arr):
    if not arr:
        return arr
    max_val = max(arr)
    exp = 1
    while max_val // exp > 0:
        counting_sort_by_digit(arr, exp)
        exp *= 10

def counting_sort_by_digit(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    for x in arr:
        idx = (x // exp) % 10
        count[idx] += 1
    for i in range(1, 10):
        count[i] += count[i - 1]
    for i in range(n - 1, -1, -1):  # Traverse right-to-left for stability
        idx = (arr[i] // exp) % 10
        output[count[idx] - 1] = arr[i]
        count[idx] -= 1
    for i in range(n):
        arr[i] = output[i]
```

**Complexity:** O(N * D) where D = number of digits. Space O(N + K) where K = radix (10 for decimal).

**Socratic prompt:** *"Radix sort processes digits from least significant to most significant. Why does this order matter? What would happen if you went the other direction?"*

---

## Choosing the Right Sort

| Situation | Best Choice | Why |
|-----------|-------------|-----|
| General purpose | Quick sort (or language default) | Fastest average case, good cache behavior |
| Stability required | Merge sort / Timsort | Preserves order of equal elements |
| Worst-case guarantee needed | Merge sort or heap sort | Always O(N log N) |
| Nearly sorted data | Insertion sort / Timsort | O(N) for nearly sorted |
| Small arrays (N < 20) | Insertion sort | Low overhead |
| Integer values in small range | Counting sort | O(N + K), hard to beat |
| Uniform distribution, known range | Bucket sort | O(N) expected |
| Fixed-length keys (strings, IDs) | Radix sort | O(N * D), avoids comparisons |
| Memory constrained | Heap sort or quick sort | O(1) extra space |
| Sorting linked lists | Merge sort | O(1) extra space on linked lists |

---

## Attribution

The frameworks in this file are inspired by and adapted from labuladong's algorithmic guides (labuladong.online), particularly the "Getting Started: Sorting" curriculum. The tree-traversal insight for sorts and the framework-first approach have been restructured and annotated for Socratic teaching use.
