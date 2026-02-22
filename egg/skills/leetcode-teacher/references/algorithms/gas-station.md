# Gas Station & Video Stitching

Circular route greedy (gas station) and interval coverage greedy (video stitching, minimum taps).

**Prerequisites:** Greedy basics, prefix sums. See `references/algorithms/greedy-algorithms.md` for the greedy framework.

---

## Gas Station (LC 134)

**Problem:** Circular route with `n` gas stations. `gas[i]` fuel gained, `cost[i]` fuel spent to next station. Find starting station to complete the circuit (or -1 if impossible).

**Key observations:**
1. If `sum(gas) < sum(cost)`, no solution exists (not enough total fuel)
2. If a solution exists, it's unique

### Approach 1: Greedy (Single Pass)

If starting from station `start`, the tank goes negative at station `i`, then no station between `start` and `i` can be a valid start either (they'd have even less fuel at `i`). So skip to `i + 1`.

```python
def can_complete_circuit(gas, cost):
    n = len(gas)
    total_tank = 0
    curr_tank = 0
    start = 0
    for i in range(n):
        diff = gas[i] - cost[i]
        total_tank += diff
        curr_tank += diff
        if curr_tank < 0:
            start = i + 1
            curr_tank = 0
    return start if total_tank >= 0 else -1
```

### Approach 2: Graph/Prefix Sum Method

Compute `diff[i] = gas[i] - cost[i]`. The prefix sum of `diff` shows accumulated fuel surplus. The valid starting point is right after the minimum prefix sum (the "deepest valley").

```python
def can_complete_circuit_graph(gas, cost):
    n = len(gas)
    total = 0
    min_sum = float('inf')
    min_idx = 0
    for i in range(n):
        total += gas[i] - cost[i]
        if total < min_sum:
            min_sum = total
            min_idx = i
    if total < 0:
        return -1
    return (min_idx + 1) % n
```

*Socratic prompt: "Why does starting after the 'deepest valley' in the prefix sum work? What does the valley represent physically?"*

---

## Video Stitching / Interval Coverage

### Video Stitching (LC 1024)

**Problem:** Given clips `[start, end]` and a target time `T`, find the minimum number of clips to cover `[0, T]`.

**Greedy strategy:** Sort by start time. At each step, among all clips that start at or before the current coverage end, pick the one that extends coverage the farthest.

```python
def video_stitching(clips, time):
    clips.sort()
    res = 0
    cur_end = 0
    next_end = 0
    i = 0
    while cur_end < time:
        while i < len(clips) and clips[i][0] <= cur_end:
            next_end = max(next_end, clips[i][1])
            i += 1
        if next_end == cur_end:
            return -1
        cur_end = next_end
        res += 1
    return res
```

**Connection to Jump Game II:** This is structurally identical! Each "clip" is like a jump range.

### Minimum Number of Taps (LC 1326)

Same problem, different framing: taps at positions `0..n`, each covers `[i - ranges[i], i + ranges[i]]`. Convert to intervals, then apply video stitching.

```python
def min_taps(n, ranges):
    clips = []
    for i, r in enumerate(ranges):
        clips.append([max(0, i - r), i + r])
    return video_stitching(clips, n)
```

*Socratic prompt: "Can you see why Video Stitching, Jump Game II, and Minimum Taps are all the same problem in disguise? What's the common structure?"*

### The Interval Coverage Family

| Problem | Input | Target | Core Idea |
|---------|-------|--------|-----------|
| Video Stitching (1024) | Clips `[s, e]` | Cover `[0, T]` | Greedy extend coverage |
| Jump Game II (45) | `nums[i]` = max jump | Reach index `n-1` | BFS levels = greedy extend |
| Min Taps (1326) | Tap ranges | Cover `[0, n]` | Convert to clips |
| Set Cover | Sets | Cover universe | NP-hard in general (greedy is approximation) |

## See Also

- `references/algorithms/jump-game.md` — Jump game (structurally identical)
- `references/algorithms/interval-scheduling.md` — Interval scheduling and meeting rooms
- `references/algorithms/greedy-algorithms.md` — Greedy framework

---

## Attribution

Extracted from Sections 5-6 (Gas Station, Video Stitching) of `greedy-algorithms.md`, inspired by labuladong's algorithmic guides (labuladong.online).
