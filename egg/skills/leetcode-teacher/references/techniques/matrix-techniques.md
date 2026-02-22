# Matrix Techniques

A matrix is a 2-dimensional array. Many matrix problems can be framed as either **graph problems** (cells are nodes, neighbors are edges) or **dynamic programming** on a 2D grid. Understanding which framing applies is the first step.

---

## Corner Cases

- Empty matrix (0 rows or 0 columns)
- 1x1 matrix (single cell)
- Single row matrix
- Single column matrix

---

## Techniques

### Creating an Empty N x M Matrix

```python
# Correct: each row is an independent list
matrix = [[0] * m for _ in range(n)]

# WRONG: all rows share the same list object
matrix = [[0] * m] * n  # Mutating one row mutates all!
```

*Socratic prompt: "Why does `[[0] * m] * n` cause bugs? What does Python actually create in memory?"*

### Copying a Matrix

```python
# Shallow copy (sufficient for matrices of primitives)
copy = [row[:] for row in matrix]
```

### Transposing a Matrix

```python
# Using zip (elegant Python trick)
transposed = list(zip(*matrix))
# Note: zip returns tuples; use [list(row) for row in zip(*matrix)] if you need lists

# Manual transpose
transposed = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
```

*Socratic prompt: "What does the `*` in `zip(*matrix)` do? How does unpacking the rows achieve transposition?"*

### Rotating a Matrix 90 Degrees Clockwise (LC 48)

**Key insight:** Rotate = transpose + reverse each row.

```python
def rotate(matrix):
    """Rotate matrix 90 degrees clockwise in-place."""
    n = len(matrix)
    # Step 1: Transpose
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # Step 2: Reverse each row
    for row in matrix:
        row.reverse()
```

### Game Board Modeling

Matrices naturally model game boards (Tic-Tac-Toe, Sudoku, Connect Four). A powerful technique: **transpose the board to reuse horizontal verification logic for vertical checks**.

For example, in Sudoku validation, if you have a function that checks rows, you can check columns by transposing the board and checking rows again. This avoids writing separate column-checking logic.

---

## Spiral Traversal

### Spiral Matrix (LC 54)

Traverse the matrix in spiral order by maintaining four boundaries: top, bottom, left, right.

```python
def spiral_order(matrix):
    """Return elements in spiral order."""
    if not matrix:
        return []
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # Traverse right
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1
        # Traverse down
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1
        # Traverse left
        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1
        # Traverse up
        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1

    return result
```

*Socratic prompt: "Why do we need the `if top <= bottom` and `if left <= right` checks before the left and up traversals?"*

### Spiral Matrix II (LC 59)

Generate an n x n matrix filled with elements 1 to n^2 in spiral order. Same boundary technique, but filling instead of reading.

---

## Set Matrix Zeroes (LC 73)

If an element is 0, set its entire row and column to 0.

**Key insight:** Use the first row and first column as markers to avoid O(mn) extra space.

```python
def set_zeroes(matrix):
    """Set entire row and column to 0 if element is 0. O(1) extra space."""
    m, n = len(matrix), len(matrix[0])
    first_row_zero = any(matrix[0][j] == 0 for j in range(n))
    first_col_zero = any(matrix[i][0] == 0 for i in range(m))

    # Use first row/col as markers
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0

    # Zero out cells based on markers
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0

    # Handle first row and column
    if first_row_zero:
        for j in range(n):
            matrix[0][j] = 0
    if first_col_zero:
        for i in range(m):
            matrix[i][0] = 0
```

*Socratic prompt: "Why do we handle the first row and first column separately? What goes wrong if we don't?"*

---

## Essential & Recommended Practice Questions

| Problem | Difficulty | Key Technique |
|---------|-----------|---------------|
| Spiral Matrix (54) | Medium | Boundary tracking |
| Set Matrix Zeroes (73) | Medium | First row/col as markers |
| Valid Sudoku (36) | Medium | Transpose for column checks, box indexing |
| Rotate Image (48) | Medium | Transpose + reverse rows |
| Spiral Matrix II (59) | Medium | Boundary filling |
| Word Search (79) | Medium | Matrix as graph, DFS backtracking |

---

## Attribution

Interview-oriented content (corner cases, essential questions, techniques) adapted from the Tech Interview Handbook matrix cheatsheet. Code templates restructured and annotated for Socratic teaching use.
