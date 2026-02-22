# The Sliding Window Framework

Technique for substring/subarray problems with a contiguous window constraint. All sliding window problems answer three questions.

**Prerequisites:** Array traversal basics. See `references/techniques/array-techniques.md` for prefix sums and two-pointer foundations.

---

## The Three Questions

1. **Q1: When do I expand the window?** (What condition makes me move `right` forward?)
2. **Q2: When do I shrink the window?** (What condition makes me move `left` forward?)
3. **Q3: When do I update the result?** (After expanding? After shrinking? Both?)

## Annotated Template

```python
def sliding_window(s, t):
    from collections import defaultdict
    need = defaultdict(int)     # What we need to satisfy
    window = defaultdict(int)   # What the current window contains

    for c in t:
        need[c] += 1

    left = 0
    valid = 0          # Number of characters satisfying the condition
    result = float('inf')  # or 0, depending on min/max

    for right in range(len(s)):
        # --- EXPAND: add s[right] to window ---
        c = s[right]
        window[c] += 1
        if window[c] == need[c]:    # Q1: update validity check
            valid += 1

        # --- SHRINK: while window satisfies condition ---
        while valid == len(need):   # Q2: shrink condition
            # Q3: update result (here, BEFORE shrinking)
            result = min(result, right - left + 1)
            d = s[left]
            if window[d] == need[d]:
                valid -= 1
            window[d] -= 1
            left += 1              # Shrink from left

    return result
```

**Convention:** The window is `[left, right)` — left-closed, right-open. This means `right` points to the next element to add.

## Worked Example: Minimum Window Substring (LC 76)

Given `s = "ADOBECODEBANC"`, `t = "ABC"`:
- Expand right until window contains all of A, B, C → `"ADOBEC"` (valid)
- Shrink left while still valid → `"DOBEC"` (still valid) → `"OBEC"` (invalid, stop)
- Continue expanding right, repeat

Answer the three questions for every sliding window problem and the template writes itself.

## See Also

- `references/frameworks/algorithm-frameworks.md` — Enumeration principle (sliding window eliminates redundancy by reusing window state)
- `references/techniques/string-techniques.md` — String-specific sliding window applications
- `references/techniques/array-techniques.md` — Array-specific sliding window applications

---

## Attribution

Extracted from the Sliding Window Framework section of `algorithm-frameworks.md`, inspired by labuladong's algorithmic guides (labuladong.online).
