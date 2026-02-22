# N-Sum Generalized

Recursive reduction of N-Sum to 2Sum with sorting and duplicate skipping.

**Prerequisites:** Two pointers, sorting. See `references/techniques/array-techniques.md` for two-pointer basics.

---

## Key Insight

Reduce N-Sum to (N-1)-Sum recursively until you hit 2Sum as the base case. Sort once, then skip duplicates at every recursion level.

## Template

```python
def n_sum(nums, n, target):
    nums.sort()
    results = []

    def helper(start, n, target, path):
        # Base case: 2Sum with two pointers
        if n == 2:
            lo, hi = start, len(nums) - 1
            while lo < hi:
                s = nums[lo] + nums[hi]
                if s == target:
                    results.append(path + [nums[lo], nums[hi]])
                    lo += 1
                    while lo < hi and nums[lo] == nums[lo - 1]:
                        lo += 1  # Skip duplicates
                elif s < target:
                    lo += 1
                else:
                    hi -= 1
            return

        # Recursive case: fix one element, reduce to (n-1)Sum
        for i in range(start, len(nums) - n + 1):
            if i > start and nums[i] == nums[i - 1]:
                continue  # Skip duplicates
            # Pruning: if smallest possible sum > target, stop
            if nums[i] * n > target:
                break
            # Pruning: if largest possible sum < target, skip
            if nums[i] + nums[-1] * (n - 1) < target:
                continue
            helper(i + 1, n - 1, target - nums[i], path + [nums[i]])

    helper(0, n, target, [])
    return results
```

## Why This Works

Each level fixes one number and reduces the problem. Sorting enables both two-pointer base case and duplicate skipping. The pruning bounds prevent wasted work.

## Key Problems

- Two Sum (1), 3Sum (15), 4Sum (18), kSum variants

## See Also

- `references/techniques/array-techniques.md` — Two-pointer technique fundamentals

---

## Attribution

Extracted from the N-Sum Generalized section of `advanced-patterns.md`, inspired by labuladong's algorithmic guides (labuladong.online).
