# Jump Game

Greedy reachability and minimum jumps using BFS-level insight.

**Prerequisites:** Greedy basics. See `references/algorithms/greedy-algorithms.md` for the greedy framework.

---

## Jump Game I (LC 55) — Can Reach End?

**Greedy insight:** Track the farthest reachable index. If at any point `i > farthest`, you're stuck.

```python
def can_jump(nums):
    farthest = 0
    for i in range(len(nums)):
        if i > farthest:
            return False  # Can't reach index i
        farthest = max(farthest, i + nums[i])
    return True
```

## Jump Game II (LC 45) — Min Jumps

**BFS-level insight:** Think of it as BFS where each "level" is the range of indices reachable in one more jump. Count levels until you reach the end.

```python
def jump(nums):
    n = len(nums)
    jumps = 0
    cur_end = 0     # End of current BFS level
    farthest = 0    # Farthest reachable in next level
    for i in range(n - 1):
        farthest = max(farthest, i + nums[i])
        if i == cur_end:  # Finished current level
            jumps += 1
            cur_end = farthest
            if cur_end >= n - 1:
                break
    return jumps
```

**Why this is greedy, not DP:** At each level boundary, we commit to jumping to the farthest point reachable. We never backtrack or reconsider. The BFS-level structure proves optimality: any solution needs at least as many "levels" as we use.

*Socratic prompt: "Draw the BFS tree for nums = [2,3,1,1,4]. What are the levels? Why does each level correspond to one jump?"*

## See Also

- `references/algorithms/gas-station.md` — Gas station (related circular greedy)
- `references/algorithms/interval-scheduling.md` — Interval coverage (structurally identical to Jump Game II)
- `references/algorithms/greedy-algorithms.md` — Greedy framework and proof techniques

---

## Attribution

Extracted from Section 4 (Jump Game) of `greedy-algorithms.md`, inspired by labuladong's algorithmic guides (labuladong.online).
