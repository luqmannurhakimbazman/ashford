# Brain Teasers & Games

Mathematical brain teasers and game theory problems that appear in coding interviews. These have deceptively simple solutions once you find the key insight. For game theory problems requiring full DP (e.g., multi-pile stone games, prediction), see `dynamic-programming-core.md`. For probability-based puzzles, see `probability-random.md`.

---

## Quick Reference Table

| Pattern | Recognition Signals | Key Problems | Section |
|---------|-------------------|--------------|---------|
| Nim Game (Modulo) | "Two players take turns", "remove 1-3 items", "last to take wins" | Nim Game (292) | 1 |
| Stone Game (First-Player Advantage) | "Even piles", "two players pick from ends", "optimal play" | Stone Game (877) | 2 |
| Bulb Switcher (Square Root) | "Toggle switches", "n rounds of toggling", "how many on" | Bulb Switcher (319) | 3 |
| Random Map Generation | "Generate random board", "minesweeper", "random placement" | -- (system design) | 4 |

---

## 1. Nim Game

### Problem (LC 292)

There is a heap of n stones. Two players take turns removing 1-3 stones. The player who removes the last stone wins. Given n, determine if the first player can win (assuming both play optimally).

### Core Insight

**If n is a multiple of 4, the first player loses. Otherwise, the first player wins.**

Why? If `n % 4 == 0`, whatever the first player takes (1, 2, or 3), the second player can take enough to make the total removal 4. The second player keeps reducing by 4 until the pile hits 0 — and the first player is the one who can't move.

If `n % 4 != 0`, the first player takes `n % 4` stones on the first move, leaving a multiple of 4 for the opponent. Now the opponent is in the losing position.

### Solution

```python
def can_win_nim(n: int) -> bool:
    """First player wins iff n is not a multiple of 4."""
    return n % 4 != 0
```

**Complexity:** O(1) time, O(1) space. No DP, no recursion — pure math.

### The DP Approach (For Understanding)

Before seeing the O(1) insight, consider the DP formulation:

```python
def can_win_nim_dp(n: int) -> bool:
    """DP approach: dp[i] = can the current player win with i stones?"""
    if n <= 3:
        return True
    dp = [False] * (n + 1)
    dp[1] = dp[2] = dp[3] = True
    for i in range(4, n + 1):
        # Current player wins if ANY move leaves opponent in losing state
        dp[i] = not dp[i - 1] or not dp[i - 2] or not dp[i - 3]
    return dp[n]
```

Examining the DP table reveals the pattern: `F T T T F T T T F T T T ...` — every 4th position is False.

*Socratic prompt: "Before looking at the O(1) solution, fill in dp[1] through dp[8] by hand. Do you see a pattern? Can you prove why it repeats every 4?"*

---

## 2. Stone Game

### Problem (LC 877)

Two players pick stones from either end of an array of even length. Each pile has a distinct positive count. The player with the most stones wins. Can the first player always win?

### Core Insight

**The first player always wins.** With an even number of piles, the first player can always choose to take all even-indexed piles OR all odd-indexed piles. One of these sums must be larger (piles are distinct), so the first player can guarantee a win.

### Why This Works

Color the piles alternating black and white:

```
piles: [5, 3, 4, 5]
color:  B  W  B  W
```

- Sum of black (even-indexed): 5 + 4 = 9
- Sum of white (odd-indexed): 3 + 5 = 8

The first player always gets to "choose a color." If they take the leftmost pile, the opponent faces `[W, B, W]` and must expose another B on one end. The first player can consistently take all B piles.

### Solution

```python
def stone_game(piles: list[int]) -> bool:
    """First player always wins with even number of distinct piles."""
    return True
```

**Complexity:** O(1). The answer is literally always `True`.

### The General Case (DP)

When piles aren't even-length or values aren't distinct, you need interval DP. See `dynamic-programming-core.md` for the full game theory DP framework.

```python
def stone_game_dp(piles: list[int]) -> bool:
    """Interval DP for the general stone game."""
    n = len(piles)
    # dp[i][j] = max(score_first - score_second) for piles[i..j]
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = piles[i]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = max(
                piles[i] - dp[i + 1][j],   # Take left
                piles[j] - dp[i][j - 1]    # Take right
            )
    return dp[0][n - 1] > 0
```

*Socratic prompt: "Imagine you have piles [2, 1, 3, 4]. Can the first player guarantee all even-indexed piles? Walk through the choices."*

---

## 3. Bulb Switcher

### Problem (LC 319)

There are n bulbs, all off initially. In round i (1-indexed), you toggle every i-th bulb. After n rounds, how many bulbs are on?

### Core Insight

**Bulb k is on iff k is a perfect square.** The answer is `⌊√n⌋`.

Why? Bulb k is toggled once for each of its divisors. Most numbers have divisors in pairs (e.g., 12: {1,12}, {2,6}, {3,4}), so they're toggled an even number of times → off. Perfect squares have one unpaired divisor (e.g., 9: {1,9}, {3,3}), so they're toggled an odd number of times → on.

### Solution

```python
def bulb_switch(n: int) -> int:
    """Count perfect squares up to n."""
    return int(n ** 0.5)
```

**Complexity:** O(1) time, O(1) space.

### Verification for Small n

| Bulb | Toggled by rounds | # Toggles | On? |
|------|------------------|-----------|-----|
| 1 | 1 | 1 | Yes (1 = 1²) |
| 2 | 1, 2 | 2 | No |
| 3 | 1, 3 | 2 | No |
| 4 | 1, 2, 4 | 3 | Yes (4 = 2²) |
| 5 | 1, 5 | 2 | No |
| 6 | 1, 2, 3, 6 | 4 | No |
| 7 | 1, 7 | 2 | No |
| 8 | 1, 2, 4, 8 | 4 | No |
| 9 | 1, 3, 9 | 3 | Yes (9 = 3²) |

*Socratic prompt: "Why do perfect squares have an odd number of divisors? Think about how divisors pair up. What's special about the square root?"*

---

## 4. Random Map Generation (Minesweeper)

### Problem

Generate a random Minesweeper board: place `m` mines randomly on an `M × N` grid.

### Core Insight

This is a **random sampling problem** — choose m cells from M×N total cells without replacement. Flatten the 2D grid to 1D, shuffle, pick the first m positions.

### Template

```python
import random

def generate_minesweeper(M: int, N: int, m: int) -> list[list[bool]]:
    """Place m mines randomly on an M×N grid using Fisher-Yates shuffle."""
    total = M * N
    # Create array of indices, shuffle, take first m as mines
    indices = list(range(total))
    # Partial Fisher-Yates: only need first m elements
    for i in range(min(m, total)):
        j = random.randint(i, total - 1)
        indices[i], indices[j] = indices[j], indices[i]

    mine_set = set(indices[:m])
    board = [[False] * N for _ in range(M)]
    for idx in mine_set:
        r, c = divmod(idx, N)
        board[r][c] = True
    return board
```

### 2D ↔ 1D Conversion

A common technique in grid problems:
- **Flatten:** `index = row * num_cols + col`
- **Unflatten:** `row, col = divmod(index, num_cols)`

This lets you apply 1D algorithms (shuffling, sampling) to 2D grids.

*Socratic prompt: "Why use Fisher-Yates shuffle instead of randomly picking positions one by one? What's the problem with the naive approach when m is close to M×N?"*

For the full Fisher-Yates shuffle algorithm and proof of uniform randomness, see `probability-random.md`.

---

## When Brain Teasers Appear in Interviews

These problems test **mathematical reasoning**, not coding ability. The interviewer wants to see:

1. **Pattern recognition** — Can you spot the mathematical structure?
2. **Proof sketching** — Can you explain WHY the pattern holds?
3. **Edge case awareness** — What happens at n=0, n=1?

**Interview tip:** If you recognize the trick immediately, don't just blurt the answer. Walk through small examples first, "discover" the pattern, and explain the reasoning. This demonstrates problem-solving process, not just memorization.

---

## Attribution

Content synthesized from labuladong's algorithmic guide, Chapter 4 — "Other Common Techniques: Brain Teasers and Games." Reorganized and augmented with Socratic teaching prompts, cross-references, and code templates for the leetcode-teacher skill.
