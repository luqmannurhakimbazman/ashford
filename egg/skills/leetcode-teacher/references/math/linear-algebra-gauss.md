# Linear Algebra — Gaussian Elimination, Determinants & Rank

Gaussian elimination is the workhorse of computational linear algebra: solving linear systems (Ax = b), computing determinants, finding matrix rank, and more. All algorithms here run in O(n³) time. The unifying theme is reducing a matrix to row echelon form through elementary row operations. Numerical stability (partial pivoting) is a recurring concern.

---

## Quick Reference Table

| Technique | What It Computes | Complexity | Key Insight |
|-----------|-----------------|------------|-------------|
| Gaussian Elimination | Solution to Ax = b | O(n²m) | Forward elimination + back substitution |
| Bitset Gaussian (mod 2) | GF(2) linear system | O(n²m / 64) | XOR replaces arithmetic, bitwise parallelism |
| Determinant (Gauss) | det(A) for square A | O(n³) | det(triangular) = product of diagonal |
| LU Decomposition (Crout) | A = LU factorization | O(n³) | Factor once, solve/det cheaply |
| Matrix Rank | Number of independent rows | O(n²m) | Count non-zero pivots after elimination |

---

## 1. Gaussian Elimination (Solving Ax = b)

### Problem Framing

Given n linear equations in m unknowns, form the **augmented matrix** [A | b] of size n × (m + 1). The goal is to reduce it to row echelon form, then back-substitute.

### Core Algorithm

1. **Forward elimination:** for each column, find the pivot (row with max absolute value in that column — partial pivoting), swap it to the current row, then eliminate all entries below.
2. **Back substitution:** starting from the last pivot, solve for each variable.

### Partial Pivoting

Always swap the row with the largest absolute value in the current column to the pivot position. This prevents division by near-zero values and dramatically improves numerical stability.

*Socratic prompt: "Why does dividing by a tiny pivot amplify floating-point errors? What's the worst case if you don't pivot?"*

### Implementation

```python
def gauss_elimination(a, b):
    """Solve Ax = b via Gaussian elimination with partial pivoting.

    Args:
        a: n x m coefficient matrix (list of lists, will be modified).
        b: n-element RHS vector (will be modified).

    Returns:
        List of m floats (unique solution), None (no solution),
        or "infinite" (infinitely many solutions).
    """
    EPS = 1e-9
    n = len(a)
    m = len(a[0])

    # Build augmented matrix
    aug = [a[i][:] + [b[i]] for i in range(n)]

    where = [-1] * m  # where[j] = row that has pivot in column j
    row = 0
    for col in range(m):
        if row >= n:
            break
        # Partial pivoting: find row with max |aug[r][col]|
        pivot = max(range(row, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < EPS:
            continue  # No pivot in this column — free variable
        aug[row], aug[pivot] = aug[pivot], aug[row]
        where[col] = row

        # Eliminate below and above (Gauss-Jordan style)
        for i in range(n):
            if i == row:
                continue
            if abs(aug[i][col]) < EPS:
                continue
            ratio = aug[i][col] / aug[row][col]
            for j in range(col, m + 1):
                aug[i][j] -= ratio * aug[row][j]
        row += 1

    # Extract solution
    x = [0.0] * m
    for j in range(m):
        if where[j] == -1:
            # Free variable detected — infinite solutions
            return "infinite"
        x[j] = aug[where[j]][m] / aug[where[j]][j]

    # Verify: check rows with no pivot for consistency
    for i in range(row, n):
        if abs(aug[i][m]) > EPS:
            return None  # Inconsistent equation: 0 = nonzero

    return x
```

### Degenerate Cases

| Case | Condition | Result |
|------|-----------|--------|
| Unique solution | rank(A) = m = n | Single solution vector |
| No solution | Some row becomes [0 0 ... 0 | c] with c ≠ 0 | Inconsistent system |
| Infinite solutions | rank(A) < m (free variables exist) | Family of solutions |

The `where[]` array tracks which column has a pivot in which row. Columns without pivots correspond to **free variables** — these can take any value, giving infinitely many solutions.

*Socratic prompt: "After elimination, you see a row [0, 0, 0, 0 | 0]. Is the system inconsistent, or does it have infinite solutions? What if the last entry were 5 instead of 0?"*

---

## 2. Bitset Optimization (Mod 2 / GF(2))

### When It Applies

When all arithmetic is modulo 2 (e.g., XOR-based systems, light-switching puzzles), addition and subtraction both become XOR. This means:
- No need for division (pivots are always 1)
- Row elimination = XOR of entire rows
- Rows can be stored as **integers** (bitsets), giving ~64x speedup

### Implementation Sketch

```python
def gauss_mod2(equations, m):
    """Solve a system of linear equations over GF(2).

    Args:
        equations: list of (int, int) pairs (row_bits, rhs_bit).
            row_bits: integer where bit j = coefficient of x_j.
            rhs_bit: 0 or 1.
        m: number of variables.

    Returns:
        List of 0/1 solution, None (no solution), or "infinite".
    """
    # Each equation stored as single integer: coefficients | rhs in bit m
    rows = []
    for bits, rhs in equations:
        rows.append(bits | (rhs << m))

    pivot_row = {}  # col -> row index
    for col in range(m):
        # Find a row with bit col set
        found = -1
        for i in range(len(rows)):
            if i not in pivot_row.values() and (rows[i] >> col) & 1:
                found = i
                break
        if found == -1:
            continue
        pivot_row[col] = found
        # Eliminate col from all other rows
        for i in range(len(rows)):
            if i != found and (rows[i] >> col) & 1:
                rows[i] ^= rows[found]

    # Extract solution
    x = [0] * m
    for col in range(m):
        if col not in pivot_row:
            return "infinite"
        r = pivot_row[col]
        x[col] = (rows[r] >> m) & 1

    # Check consistency
    for i in range(len(rows)):
        if i not in pivot_row.values():
            coeff_bits = rows[i] & ((1 << m) - 1)
            rhs_bit = (rows[i] >> m) & 1
            if coeff_bits == 0 and rhs_bit == 1:
                return None

    return x
```

**Speedup:** standard Gauss on an n × m system is O(n · m · n) = O(n²m). With bitset rows, each row operation is O(m / 64) instead of O(m), giving O(n² · m / 64).

*Socratic prompt: "In GF(2), why can XOR replace both addition and subtraction? What property of modular arithmetic makes this work?"*

---

## 3. Determinant via Gaussian Elimination

### Key Insight

The determinant of an **upper triangular** matrix is the product of its diagonal entries. Gaussian elimination converts any square matrix to upper triangular form. Two adjustments:

1. **Row swaps flip the sign:** each swap multiplies the determinant by -1. Track parity.
2. **Don't normalize pivot rows:** unlike solving Ax = b, we don't divide the pivot row by the pivot element (dividing a row by c divides det by c, which we'd need to undo).

### Implementation

```python
def determinant_gauss(a):
    """Compute determinant of square matrix a via Gaussian elimination.

    Uses partial pivoting for numerical stability.
    Time: O(n³). Does not modify the input.
    """
    EPS = 1e-9
    n = len(a)
    mat = [row[:] for row in a]  # Copy
    sign = 1

    for col in range(n):
        # Partial pivoting
        pivot = max(range(col, n), key=lambda r: abs(mat[r][col]))
        if abs(mat[pivot][col]) < EPS:
            return 0.0  # Singular matrix

        if pivot != col:
            mat[col], mat[pivot] = mat[pivot], mat[col]
            sign *= -1

        # Eliminate below (don't divide pivot row)
        for i in range(col + 1, n):
            if abs(mat[i][col]) < EPS:
                continue
            ratio = mat[i][col] / mat[col][col]
            for j in range(col, n):
                mat[i][j] -= ratio * mat[col][j]

    # det = sign * product of diagonal
    det = sign
    for i in range(n):
        det *= mat[i][i]
    return det
```

### Why This Works

Elementary row operations affect the determinant as follows:

| Operation | Effect on det |
|-----------|--------------|
| Swap two rows | Multiplies by -1 |
| Multiply a row by scalar c | Multiplies by c |
| Add multiple of one row to another | No change |

We only use swaps (tracked by `sign`) and row additions (no effect), so `det = sign × ∏ diagonal`.

*Socratic prompt: "Swapping two rows negates the determinant. Can you prove this from the definition of determinant as a sum over permutations?"*

---

## 4. Determinant via LU Decomposition (Crout's Method)

### The Factorization

Decompose A = LU where:
- **L** is lower triangular with **1s on the diagonal** (unit lower triangular)
- **U** is upper triangular

Since det(L) = 1 (product of diagonal 1s) and det(U) = ∏ U[i][i]:

```
det(A) = det(L) · det(U) = ∏ U[i][i]
```

### Doolittle's Formulas

Compute L and U column-by-column:

```
U[i][j] = A[i][j] - Σ_{k=0}^{i-1} L[i][k] · U[k][j]    (for i ≤ j)
L[i][j] = (A[i][j] - Σ_{k=0}^{j-1} L[i][k] · U[k][j]) / U[j][j]    (for i > j)
```

### Implementation

```python
def lu_decomposition(a):
    """Compute LU decomposition A = LU (Crout/Doolittle method).

    L is unit lower triangular, U is upper triangular.
    Returns (L, U) or None if a zero pivot is encountered.
    Time: O(n³).
    """
    EPS = 1e-9
    n = len(a)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        # Compute U[i][j] for j >= i
        for j in range(i, n):
            s = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = a[i][j] - s

        if abs(U[i][i]) < EPS:
            return None  # Zero pivot — need permutation (PA = LU)

        L[i][i] = 1.0

        # Compute L[j][i] for j > i
        for j in range(i + 1, n):
            s = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (a[j][i] - s) / U[i][i]

    return L, U


def determinant_lu(a):
    """Compute determinant via LU decomposition.

    det(A) = det(L) * det(U) = 1 * product(U[i][i]).
    Returns 0 if matrix is singular.
    """
    result = lu_decomposition(a)
    if result is None:
        return 0.0
    _, U = result
    det = 1.0
    for i in range(len(U)):
        det *= U[i][i]
    return det
```

### LU vs Direct Gaussian for Determinants

| Aspect | Gaussian Elimination | LU Decomposition |
|--------|---------------------|------------------|
| Complexity | O(n³) | O(n³) |
| Reuse | One-shot | Factor once, solve multiple RHS in O(n²) |
| Pivoting | Built-in partial pivoting | Needs PA = LU variant for stability |
| Practical use | Simpler to implement | Better when solving Ax = b for many b |

*Socratic prompt: "Why is det(L) always 1 when L is unit lower triangular? What would change if we allowed arbitrary diagonal entries in L?"*

---

## 5. Matrix Rank

### Definition

The **rank** of a matrix is the maximum number of linearly independent rows (equivalently, columns). It equals the number of non-zero rows after Gaussian elimination — i.e., the number of pivots.

### Algorithm

Run Gaussian elimination. For each column, try to find a pivot row. If the column has no valid pivot (all remaining entries ≈ 0), skip it (this column corresponds to a free variable). The rank is the total number of pivots found.

### Implementation

```python
def matrix_rank(a):
    """Compute the rank of matrix a via Gaussian elimination.

    Time: O(n * m * min(n, m)). Does not modify the input.
    """
    EPS = 1e-9
    n = len(a)
    if n == 0:
        return 0
    m = len(a[0])
    mat = [row[:] for row in a]  # Copy

    rank = 0
    for col in range(m):
        if rank >= n:
            break
        # Find pivot in this column among rows [rank, n)
        pivot = -1
        best = EPS
        for r in range(rank, n):
            if abs(mat[r][col]) > best:
                best = abs(mat[r][col])
                pivot = r
        if pivot == -1:
            continue  # No pivot in this column — skip

        mat[rank], mat[pivot] = mat[pivot], mat[rank]

        # Eliminate below
        for i in range(rank + 1, n):
            if abs(mat[i][col]) < EPS:
                continue
            ratio = mat[i][col] / mat[rank][col]
            for j in range(col, m):
                mat[i][j] -= ratio * mat[rank][j]

        rank += 1

    return rank
```

### Rank and Its Implications

| Condition | Meaning |
|-----------|---------|
| rank(A) = min(n, m) | Full rank — A has maximum possible rank |
| rank(A) = n (for n × m, n ≤ m) | All rows independent; system Ax = b is consistent for any b in range(A) |
| rank(A) = m (for n × m, n ≥ m) | All columns independent; Ax = b has at most one solution |
| rank(A) < min(n, m) | Rank-deficient; null space is non-trivial |
| nullity = m - rank(A) | Dimension of the null space (by rank-nullity theorem) |

*Socratic prompt: "If rank(A) < min(m, n), what does that tell you about the null space? How many free variables does the system Ax = 0 have?"*

---

## Corner Cases & Numerical Pitfalls

| Case | Issue | Handling |
|------|-------|---------|
| Empty matrix (0 × 0) | No rows or columns | Rank = 0, det = 1 (by convention) |
| 1 × 1 matrix [[v]] | Trivial | det = v, rank = (1 if v ≠ 0 else 0) |
| Zero matrix | All entries 0 | det = 0, rank = 0 |
| Near-singular | Pivots close to 0 | Use EPS threshold; partial pivoting critical |
| Large magnitude differences | E.g., [1e18, 1; 1, 1e-18] | Partial pivoting helps; consider **scaled pivoting** |
| Integer-only problems | No floating-point error | Use exact integer arithmetic (avoid EPS entirely) |

**EPS convention:** Use `EPS = 1e-9` for floating-point problems. For integer problems where all intermediate values fit in 64-bit integers, avoid floating-point entirely and compare with `== 0`.

---

## Practice Problems

| Source | Problem | Technique |
|--------|---------|-----------|
| SPOJ XMAX | XOR Maximization | Gaussian elimination over GF(2), greedy on MSB |
| SPOJ MATCHB | Finding Matches | System of linear equations |
| Codeforces 21D | Traveling Graph | Gaussian elimination for minimum spanning subgraph |
| Codeforces 24E | Berland Feasts | Rank of binary matrix |
| Codeforces 79D | Password | GF(2) Gaussian elimination with BFS |
| UVa 10828 | Back to Kernighan-Ritchie | Gaussian elimination on probability system |
| SPOJ LightSwitching | Light Switching | XOR-based system, bitset optimization |

---

## Attribution

Content synthesized from cp-algorithms.com articles on [Gaussian Elimination](https://cp-algorithms.com/linear_algebra/linear-system-gauss.html), [Determinant (Gauss)](https://cp-algorithms.com/linear_algebra/determinant-gauss.html), [Determinant (Kraut)](https://cp-algorithms.com/linear_algebra/determinant-kraut.html), and [Matrix Rank](https://cp-algorithms.com/linear_algebra/rank-matrix.html). Algorithms and complexity analyses are from the original articles; code has been translated to Python with added Socratic commentary for the leetcode-teacher reference format.
