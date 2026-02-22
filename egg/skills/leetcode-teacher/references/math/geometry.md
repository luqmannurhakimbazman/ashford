# Geometry

Computational geometry techniques for coding interviews: distance calculations, overlap detection, and area computation. These problems appear less frequently than other categories but have elegant, formulaic solutions once you know the patterns. For related 2D array techniques, see `matrix-techniques.md`. For problems that combine geometry with other patterns (e.g., sorting by distance), see `classic-interview-problems.md`.

---

## Quick Reference Table

| Pattern | Recognition Signals | Key Problems | Section |
|---------|-------------------|--------------|---------|
| Distance Between Points | "Closest points", "k nearest", "distance" | K Closest Points to Origin (973) | 1 |
| Overlapping Circles | "Two circles overlap", "circle intersection" | -- (technique) | 2 |
| Overlapping Rectangles | "Rectangle overlap", "intersection area" | Rectangle Overlap (836) | 3 |
| Rectangle Area | "Union area", "total area of two rectangles" | Rectangle Area (223) | 4 |

---

## Corner Cases

- **Zero dimensions:** A rectangle with zero width or height is a line/point, not a rectangle. Check if `x1 == x2` or `y1 == y2`.
- **Identical points:** Distance is zero. Ensure your logic handles this (e.g., K Closest when a point is at the origin).
- **Integer overflow:** Squared distances can overflow in fixed-width languages. In Python this isn't an issue, but in Java/C++ use `long` for `dx*dx + dy*dy`.
- **Negative coordinates:** All formulas work with negative coordinates — no special handling needed, but be mindful of coordinate system orientation.

---

## 1. Distance Between Two Points

### Core Insight

The Euclidean distance between points `(x1, y1)` and `(x2, y2)` is `sqrt((x2-x1)^2 + (y2-y1)^2)`. But for **comparison purposes** (closer/farther), you can skip the square root — comparing squared distances preserves order and avoids floating point issues.

```python
def squared_distance(p1, p2):
    """Squared Euclidean distance — use for comparison, skip sqrt."""
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
```

**Why skip sqrt?** `sqrt` is monotonically increasing, so `sqrt(a) < sqrt(b)` if and only if `a < b`. Skipping it avoids floating point precision issues and is faster.

*Socratic prompt: "If you need to find the k closest points, do you actually need the exact distance? What's sufficient for comparison?"*

### Application: K Closest Points to Origin (LC 973)

Find the k closest points to `(0, 0)`. Use a max-heap of size k or quickselect.

```python
import heapq

def k_closest(points: list[list[int]], k: int) -> list[list[int]]:
    """K closest points using min-heap on squared distance."""
    return heapq.nsmallest(k, points, key=lambda p: p[0] ** 2 + p[1] ** 2)
```

**Alternative — Quickselect O(n) average:**

```python
import random

def k_closest_quickselect(points: list[list[int]], k: int) -> list[list[int]]:
    """K closest using quickselect for O(n) average time."""
    def dist(p):
        return p[0] ** 2 + p[1] ** 2

    def partition(lo, hi):
        pivot = dist(points[hi])
        store = lo
        for i in range(lo, hi):
            if dist(points[i]) <= pivot:
                points[store], points[i] = points[i], points[store]
                store += 1
        points[store], points[hi] = points[hi], points[store]
        return store

    lo, hi = 0, len(points) - 1
    while lo < hi:
        pivot_idx = random.randint(lo, hi)
        points[pivot_idx], points[hi] = points[hi], points[pivot_idx]
        mid = partition(lo, hi)
        if mid == k:
            break
        elif mid < k:
            lo = mid + 1
        else:
            hi = mid - 1
    return points[:k]
```

| Approach | Time | Space | Notes |
|----------|------|-------|-------|
| Sort | O(n log n) | O(n) | Simple but overkill |
| Min-heap (nsmallest) | O(n log k) | O(k) | Good for streaming |
| Quickselect | O(n) avg | O(1) | Best average case |

*Socratic prompt: "When would you prefer the heap approach over quickselect? Think about worst-case guarantees and streaming data."*

---

## 2. Overlapping Circles

### Core Insight

Two circles overlap if the distance between their centers is less than the sum of their radii. Use squared distances to avoid sqrt.

```python
def circles_overlap(x1, y1, r1, x2, y2, r2):
    """Check if two circles overlap (including touching)."""
    dist_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    radii_sum = r1 + r2
    return dist_sq <= radii_sum ** 2
```

**Three cases:**
- `dist > r1 + r2`: No overlap (circles are apart)
- `dist == r1 + r2`: Tangent (touching at exactly one point)
- `dist < r1 + r2`: Overlapping

*Socratic prompt: "What if one circle is entirely inside the other? Does this formula still detect it? (Hint: consider `dist < |r1 - r2|`.)"*

---

## 3. Overlapping Rectangles

### Core Insight

Two axis-aligned rectangles **do NOT overlap** if one is completely to the left, right, above, or below the other. It's easier to check for **non-overlap** and negate.

A rectangle is defined by bottom-left `(x1, y1)` and top-right `(x2, y2)`.

### Template

```python
def is_overlap(rec1: list[int], rec2: list[int]) -> bool:
    """Check if two axis-aligned rectangles overlap.

    Each rectangle: [x1, y1, x2, y2] (bottom-left, top-right).
    """
    # Non-overlap conditions (any one means no overlap):
    # rec1 is left of rec2:  rec1[2] <= rec2[0]
    # rec1 is right of rec2: rec1[0] >= rec2[2]
    # rec1 is below rec2:    rec1[3] <= rec2[1]
    # rec1 is above rec2:    rec1[1] >= rec2[3]
    return not (rec1[2] <= rec2[0] or  # left
                rec1[0] >= rec2[2] or  # right
                rec1[3] <= rec2[1] or  # below
                rec1[1] >= rec2[3])    # above
```

**Why check non-overlap?** There are only 4 non-overlap conditions but many overlap configurations (partial, full, corner, edge). Negating is simpler and less error-prone.

*Socratic prompt: "Why do we use `<=` instead of `<` for the non-overlap check? What's the difference between overlapping and merely touching at an edge?"*

### Application: Rectangle Overlap (LC 836)

Direct application of the template above. The problem uses `[x1, y1, x2, y2]` format with bottom-left and top-right corners.

---

## 4. Rectangle Area

### Core Insight

To compute the total area covered by two (possibly overlapping) rectangles: `Area1 + Area2 - Overlap`. The overlap is itself a rectangle whose coordinates are the intersection of the two rectangles' ranges on each axis.

### Application: Rectangle Area (LC 223)

```python
def compute_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> int:
    """Total area covered by two rectangles (union area)."""
    area1 = (ax2 - ax1) * (ay2 - ay1)
    area2 = (bx2 - bx1) * (by2 - by1)

    # Overlap dimensions (0 if no overlap)
    overlap_width = max(0, min(ax2, bx2) - max(ax1, bx1))
    overlap_height = max(0, min(ay2, by2) - max(ay1, by1))
    overlap_area = overlap_width * overlap_height

    return area1 + area2 - overlap_area
```

**Key formula for overlap on one axis:** `max(0, min(right1, right2) - max(left1, left2))`. If this is negative or zero, there's no overlap on that axis (and therefore no 2D overlap).

*Socratic prompt: "Why does `min(right1, right2) - max(left1, left2)` give the overlap width? Draw two intervals on a number line and trace through the formula."*

---

## Practice Questions

### Essential

| Problem | Key Concept |
|---------|-------------|
| Rectangle Overlap (836) | 4-condition non-overlap check, negate |

### Recommended

| Problem | Key Concept |
|---------|-------------|
| Rectangle Area (223) | Union area = A1 + A2 - overlap |
| K Closest Points to Origin (973) | Squared distance comparison, heap or quickselect |

---

## Attribution

Content synthesized from the Tech Interview Handbook (techinterviewhandbook.org) geometry cheatsheet. Organized with Socratic teaching prompts, code templates, and cross-references for the leetcode-teacher skill.
