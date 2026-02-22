# Interval Scheduling

Greedy interval selection, meeting rooms (scan line), merge intervals, and interval coverage problems.

**Prerequisites:** Sorting. See `references/algorithms/greedy-algorithms.md` for the greedy framework and proof techniques.

---

## Maximum Non-Overlapping Intervals

**Greedy strategy:** Sort by **end time**. Greedily pick the interval that finishes earliest and doesn't overlap with the last picked.

**Why sort by end time?** Picking the earliest-ending interval leaves the most room for future intervals. Sorting by start time doesn't guarantee this — an early-starting but long interval can block many short ones.

```python
def interval_schedule(intervals):
    """Return maximum number of non-overlapping intervals."""
    intervals.sort(key=lambda x: x[1])  # Sort by end time
    count = 0
    end = float('-inf')
    for s, e in intervals:
        if s >= end:       # No overlap with last picked
            count += 1
            end = e
    return count
```

**Proof (exchange argument):** If OPT picks interval A instead of greedy's B (where B ends earlier), swapping A for B can't cause overlaps with later intervals (B ends earlier, so it leaves more room). Therefore OPT with the swap is still valid and has the same count.

---

## Non-overlapping Intervals (LC 435)

**Min removals** = `total intervals - max non-overlapping`.

```python
def erase_overlap_intervals(intervals):
    return len(intervals) - interval_schedule(intervals)
```

## Minimum Arrows to Burst Balloons (LC 452)

Find minimum points that pierce all intervals. Equivalent to finding groups of mutually overlapping intervals.

```python
def find_min_arrow_shots(points):
    if not points:
        return 0
    points.sort(key=lambda x: x[1])  # Sort by end
    arrows = 1
    end = points[0][1]
    for s, e in points[1:]:
        if s > end:  # No overlap: need new arrow
            arrows += 1
            end = e
    return arrows
```

*Socratic prompt: "How is this different from max non-overlapping intervals? Why do we use `s > end` instead of `s >= end`?"*

## Merge Intervals (LC 56)

Sort by **start time**, then extend greedily.

```python
def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:  # Overlaps with last merged
            merged[-1][1] = max(merged[-1][1], e)  # Extend end
        else:
            merged.append([s, e])
    return merged
```

---

## Meeting Rooms & Scan Line Technique

### Minimum Meeting Rooms (LC 253)

**Approach 1: Scan Line (Difference Array)**

Mark `+1` at each meeting start and `-1` at each meeting end. The running sum at any point = simultaneous meetings. The maximum running sum = minimum rooms.

```python
def min_meeting_rooms(intervals):
    events = []
    for start, end in intervals:
        events.append((start, 1))   # Meeting starts: +1 room
        events.append((end, -1))    # Meeting ends: -1 room
    events.sort()  # Sort by time; ties: end (-1) before start (+1)
    max_rooms = curr_rooms = 0
    for time, delta in events:
        curr_rooms += delta
        max_rooms = max(max_rooms, curr_rooms)
    return max_rooms
```

**Approach 2: Two Sorted Arrays**

```python
def min_meeting_rooms_v2(intervals):
    starts = sorted(i[0] for i in intervals)
    ends = sorted(i[1] for i in intervals)
    rooms = 0
    end_ptr = 0
    for start in starts:
        if start < ends[end_ptr]:
            rooms += 1
        else:
            end_ptr += 1
    return rooms
```

**Approach 3: Min-Heap**

```python
import heapq

def min_meeting_rooms_heap(intervals):
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[0])
    heap = [intervals[0][1]]
    for start, end in intervals[1:]:
        if start >= heap[0]:
            heapq.heappop(heap)
        heapq.heappush(heap, end)
    return len(heap)
```

*Socratic prompt: "The scan line approach counts overlaps at each point in time. Why is the maximum overlap equal to the minimum rooms needed?"*

### Scan Line: General Pattern

Applications: Min meeting rooms (253), Maximum population year (1854), Car pooling (1094), My Calendar (729/731/732).

---

## Interval Problem Variants Summary

| Problem | Sort By | Greedy Action | Key |
|---------|---------|--------------|-----|
| Max non-overlapping (435) | End time | Pick earliest-ending | `s >= end` |
| Min arrows (452) | End time | New arrow when no overlap | `s > end` |
| Merge intervals (56) | Start time | Extend end when overlap | `s <= merged[-1][1]` |
| Insert interval (57) | Already sorted | Find overlap position | Binary search or linear scan |
| Remove covered (1288) | Start asc, end desc | Track max end | Skip if `end <= max_end` |
| Meeting Rooms II (253) | Start time (heap) or scan line | Count max overlap | See above |

## See Also

- `references/algorithms/greedy-algorithms.md` — Greedy framework and proof techniques
- `references/algorithms/gas-station.md` — Gas station circular route problem
- `references/algorithms/jump-game.md` — Jump game (related interval coverage)

---

## Attribution

Merged from the Interval Scheduling sections of `advanced-patterns.md` and `greedy-algorithms.md` (Sections 2-3), inspired by labuladong's algorithmic guides (labuladong.online).
