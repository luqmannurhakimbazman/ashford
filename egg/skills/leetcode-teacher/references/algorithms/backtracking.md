# The Backtracking Framework

Decision tree model for enumeration problems: subsets, combinations, permutations, and constraint satisfaction.

**Prerequisites:** Recursion basics, tree traversal (pre-order/post-order). See `references/frameworks/algorithm-frameworks.md` for the recursion-as-tree-traversal mental model.

---

## Decision Tree Model

Every backtracking problem is a walk through a decision tree with three components:

1. **Path** — choices made so far (the current partial solution)
2. **Choice list** — options available at the current node
3. **End condition** — when to record a result (leaf node or constraint met)

## Template

```python
def backtrack(path, choice_list):
    if meets_end_condition(path):
        result.append(path[:])
        return
    for choice in choice_list:
        # PRE-ORDER: make choice
        path.append(choice)
        # Recurse with updated choice list
        backtrack(path, updated_choice_list)
        # POST-ORDER: undo choice
        path.pop()
```

## Three Variants

The only difference between combination/permutation variants is how `choice_list` is updated:

| Variant | Problem Example | Start Parameter | Skip Duplicates? |
|---------|----------------|-----------------|-------------------|
| **Unique elements, no reuse** | Subsets, Combinations | `start = i + 1` | No |
| **Duplicate elements, no reuse** | Subsets II, Combination Sum II | `start = i + 1` | Yes: `if i > start and nums[i] == nums[i-1]: continue` |
| **Unique elements, with reuse** | Combination Sum | `start = i` (not `i + 1`) | No |

The `start` parameter controls whether elements before the current index are reconsidered. Sorting + skip-duplicate logic prevents equivalent branches when input has duplicates.

## See Also

- `references/algorithms/brute-force-search.md` — 9-variant unified framework for subsets/combinations/permutations, Ball-Box Model, constraint satisfaction (Sudoku, N-Queens), grid DFS, state-space BFS
- `references/frameworks/algorithm-frameworks.md` — Recursion as tree traversal (traversal mode = backtracking)

---

## Attribution

Extracted from the Backtracking Framework section of `algorithm-frameworks.md`, inspired by labuladong's algorithmic guides (labuladong.online).
