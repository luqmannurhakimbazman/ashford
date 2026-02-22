# Greedy Algorithms

Greedy algorithm framework and proof techniques. For specific greedy problem families, see the dedicated files linked below.

---

## Quick Reference Table

| Pattern | Reference |
|---------|-----------|
| Greedy Framework (this file) | When greedy works, proof techniques, greedy vs DP checklist |
| Interval Scheduling + Meeting Rooms | `references/algorithms/interval-scheduling.md` |
| Jump Game | `references/algorithms/jump-game.md` |
| Gas Station + Video Stitching | `references/algorithms/gas-station.md` |

---

## Greedy Framework

### What Makes Greedy Work

A greedy algorithm makes the **locally optimal choice** at each step, hoping to find the **global optimum**. This only works when the problem has the **greedy choice property**: the locally best choice is always part of some globally optimal solution.

### The Optimization Hierarchy

| Level | Technique | What It Enumerates | Typical Time |
|-------|-----------|-------------------|--------------|
| 1 | Backtracking | All valid solutions | O(2^n) or worse |
| 2 | Dynamic Programming | All subproblem states (no redundancy) | O(n^2) or O(n*k) |
| 3 | Greedy | One choice per step (no enumeration) | O(n) or O(n log n) |

Each level requires a stronger property. Greedy is fastest but most restrictive.

### When Greedy Works vs Fails

**Works (greedy choice property holds):**
- Fractional knapsack -- take highest value/weight ratio first
- Interval scheduling -- pick earliest-ending non-overlapping interval
- Huffman coding -- merge two lowest-frequency nodes
- Activity selection -- sort by end time, pick greedily

**Fails (must use DP instead):**
- 0/1 knapsack -- can't take fractions, greedy leads to suboptimal
- Longest increasing subsequence -- greedy "take the largest increase" fails
- Edit distance -- local character matches don't guarantee global minimum
- Coin change (arbitrary denominations) -- greedy "largest coin first" fails for some coin sets

*Socratic prompt: "Why does greedy work for fractional knapsack but not 0/1 knapsack? What specific property breaks?"*

### Proof Techniques

When you claim greedy works, you should have a proof sketch. Two standard approaches:

**Exchange Argument:**
1. Assume an optimal solution `OPT` that differs from the greedy solution `G`
2. Find the first point where they differ
3. Show you can "exchange" OPT's choice for G's choice without worsening the result
4. Repeat until OPT matches G -- therefore G is optimal

**Stays-Ahead Argument:**
1. Show that after each step, greedy's partial solution is at least as good as any other algorithm's partial solution
2. By induction, greedy "stays ahead" through the final step
3. Therefore the final greedy solution is optimal

*Socratic prompt: "For interval scheduling, if greedy picks the earliest-ending interval but the optimal picks a different one, can you swap them? Does the solution get worse?"*

### Greedy vs DP Decision Checklist

| Question | If Yes | If No |
|----------|--------|-------|
| Does the locally best choice always lead to a globally best solution? | Greedy | DP |
| Can you prove the exchange argument? | Greedy | DP |
| Are there overlapping subproblems? | DP (or greedy if above holds) | Maybe greedy or divide-and-conquer |
| Does the problem ask for count/all solutions? | DP or backtracking | Greedy (if asking for one optimal) |

---

## Attribution

The frameworks and problem derivations in this file are inspired by and adapted from labuladong's algorithmic guides (labuladong.online), specifically the greedy algorithm chapter. Content has been restructured for Socratic teaching use with Python code templates and cross-references.
