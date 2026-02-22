# Game Theory DP

Two-player optimal play problems: stone games, predict the winner, minimax on intervals.

**Prerequisites:** DP basics, interval thinking. See `references/algorithms/dp-framework.md` for the three-step DP process.

---

## Stone Game Framework

**Setup:** Two players pick from ends of an array. Both play optimally. Who wins, or by how much?

**State:** `dp[i][j]` = a tuple `(first_score, second_score)` representing the best outcome for the current player when stones `i..j` remain.

**Key insight (minimax):** When it's your turn, you maximize your score. Your opponent then plays optimally on the remaining subproblem.

```python
def stone_game(piles):
    n = len(piles)
    # dp[i][j] = (first_player_score, second_player_score) for piles[i..j]
    dp = [[(0, 0)] * n for _ in range(n)]

    # Base case: one pile left, first player takes it
    for i in range(n):
        dp[i][i] = (piles[i], 0)

    # Fill diagonally (increasing gap size)
    for gap in range(1, n):
        for i in range(n - gap):
            j = i + gap
            # First player picks left: gets piles[i], becomes second player
            pick_left = piles[i] + dp[i + 1][j][1]
            left_other = dp[i + 1][j][0]
            # First player picks right: gets piles[j], becomes second player
            pick_right = piles[j] + dp[i][j - 1][1]
            right_other = dp[i][j - 1][0]

            if pick_left >= pick_right:
                dp[i][j] = (pick_left, left_other)
            else:
                dp[i][j] = (pick_right, right_other)

    first, second = dp[0][n - 1]
    return first > second  # Does first player win?
```

**Simplified version for LC 877 (Stone Game):** First player always wins when n is even (mathematical proof exists), but the DP approach generalizes to all variants.

**LC 486 (Predict the Winner):** Same framework, return `first >= second`.

*Socratic prompt: "After the first player picks, why does the second player's score equal dp[subproblem][0] (the first player position of the subproblem)? Think about role switching."*

## See Also

- `references/algorithms/interval-dp.md` — Interval DP (burst balloons uses similar interval reasoning)
- `references/algorithms/dynamic-programming-core.md` — Full DP framework
- `references/problems/brain-teasers-games.md` — Game theory brain teasers

---

## Attribution

Extracted from Section 5 (Game Theory DP) of `dynamic-programming-core.md`, inspired by labuladong's algorithmic guides (labuladong.online).
