# Subsequence DP

Templates and patterns for subsequence and string DP problems: LIS, LCS, edit distance, and more.

**Prerequisites:** DP basics. See `references/algorithms/dp-framework.md` for the two subsequence DP templates overview.

---

## Template 1: `dp[i]` (Single Sequence)

Use when the answer involves elements ending at or up to index `i` in one sequence.

```python
# Longest Increasing Subsequence (LIS)
def lis(nums):
    n = len(nums)
    dp = [1] * n  # dp[i] = length of LIS ending at nums[i]
    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
```

**O(n log n) optimization:** Maintain a `tails` array where `tails[k]` is the smallest tail of all increasing subsequences of length `k+1`. Binary search for insertion point.

## Template 2: `dp[i][j]` (Two Sequences)

Use when comparing two sequences or a sequence against itself.

```python
# Longest Common Subsequence (LCS)
def lcs(s, t):
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

# Edit Distance
def edit_distance(s, t):
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],      # Delete
                                    dp[i][j - 1],      # Insert
                                    dp[i - 1][j - 1])  # Replace
    return dp[m][n]
```

## How to Choose

| Signal | Template | Reason |
|--------|----------|--------|
| One array/string, answer about a subsequence | `dp[i]` | State is position in single sequence |
| Two strings, similarity/distance/matching | `dp[i][j]` | State is position in both sequences |
| One string, palindrome-related | `dp[i][j]` | Compare string against itself (two pointers) |

## Key Problems

- LIS (300), LCS (1143), Edit Distance (72), Max Subarray (53), Word Break (139), Regex Matching (10)
- Longest Palindromic Subsequence (516), Min Insertions for Palindrome (1312)

## See Also

- `references/algorithms/dynamic-programming-core.md` — Full string/subsequence DP coverage (Section 2) with word break, regex matching, palindromic subsequences, and O(n log n) LIS
- `references/algorithms/dp-framework.md` — High-level DP templates

---

## Attribution

Extracted from the Subsequence DP section of `advanced-patterns.md`, inspired by labuladong's algorithmic guides (labuladong.online).
