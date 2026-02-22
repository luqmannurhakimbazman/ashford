# Combinatorics

Combinatorial techniques for coding interviews and competitive programming: counting formulas, enumeration under constraints, and symmetry-based counting. These techniques appear in problems involving arrangements, selections, paths, and equivalence classes. For modular arithmetic (needed for large answers), see `math-techniques.md`. For DP approaches to counting, see `dynamic-programming-core.md`. For number theory (Euler's totient, used in Burnside's), see `number-theory-advanced.md`.

---

## Quick Reference Table

| Pattern | Recognition Signals | Key Problems | Section |
|---------|-------------------|--------------|---------|
| Factorial Divisors (Legendre) | "Trailing zeros in n!", "largest power dividing n!" | Factorial Trailing Zeroes (172), Preimage Size (793) | 1 |
| Binomial Coefficients | "Choose k from n", "Pascal's triangle", "C(n,k) mod p" | Pascal's Triangle (118), Unique Paths (62) | 2 |
| Catalan Numbers | "Valid parentheses count", "binary trees", "non-crossing" | Generate Parentheses (22), Unique BSTs (96) | 3 |
| Inclusion-Exclusion | "Count elements NOT in any set", "coprime", "derangements" | Count Primes (204), K-th Smallest in Sorted Matrix | 4 |
| Burnside / Polya | "Distinct under rotation/reflection", "necklaces", "coloring" | -- (competitive programming) | 5 |
| Stars and Bars | "Distribute n identical items into k bins", "non-negative integer solutions" | Combination Sum IV (377), Coin Change 2 (518) | 6 |
| Generating Combinations | "Next combination", "iterate all k-subsets" | Next Permutation (31), Combinations (77) | 7 |
| Bishops on Chessboard | "Non-attacking bishops", "diagonal independence" | N-Queens (51) — related pattern | 8 |
| Balanced Bracket Sequences | "Valid bracket count", "k-th sequence", "next sequence" | Generate Parentheses (22), Valid Parentheses (20) | 9 |
| Counting Labeled Graphs | "How many graphs on n vertices", "connected graphs" | -- (competitive programming) | 10 |

---

## Topic Connection Map

Many combinatorics topics build on each other. Use this map to navigate:

```
Stars & Bars ──→ Binomial Coefficients ──→ Catalan Numbers ──→ Bracket Sequences
                         │                       │
                    Legendre's Formula       Binary Trees / BSTs
                    (factorial divisors)
                         │
                 Inclusion-Exclusion ──→ Derangements, Coprimality
                         │
                 Burnside / Polya ←── Euler's Totient (number-theory-advanced.md)
                         │
                  Labeled Graphs ──→ Connected Component DP
```

---

## 1. Factorial Divisors (Legendre's Formula)

### Core Insight

Find the largest power x such that k^x divides n!. For **prime** k, Legendre's formula gives:

```
v_p(n!) = floor(n/p) + floor(n/p^2) + floor(n/p^3) + ...
```

Every p-th number contributes one factor of p, every p²-th contributes a second, etc. The sum has only O(log_p n) non-zero terms.

*Socratic prompt: "Why does floor(n/5) + floor(n/25) + floor(n/125) + ... give the number of trailing zeros in n!? What does each term count?"*

### Template

```python
def legendre(n: int, p: int) -> int:
    """Largest x such that p^x divides n!. p must be prime.

    Time: O(log_p(n)). Space: O(1).
    """
    count = 0
    pk = p
    while pk <= n:
        count += n // pk
        pk *= p
    return count
```

Equivalent compact form (repeated division):

```python
def legendre_compact(n: int, p: int) -> int:
    """Same as legendre() but uses repeated division."""
    count = 0
    while n:
        n //= p
        count += n
    return count
```

### Composite k

Factor k into primes: k = p1^a1 * p2^a2 * ... * pm^am. Then:

```
answer = min(legendre(n, pi) // ai)  for each prime factor pi
```

```python
from collections import Counter

def factorize(k: int) -> dict[int, int]:
    """Return prime factorization as {prime: exponent}."""
    factors = Counter()
    d = 2
    while d * d <= k:
        while k % d == 0:
            factors[d] += 1
            k //= d
        d += 1
    if k > 1:
        factors[k] += 1
    return factors

def max_power_dividing_factorial(n: int, k: int) -> int:
    """Largest x such that k^x divides n!.

    Time: O(sqrt(k) + number_of_prime_factors * log(n)).
    """
    return min(
        legendre_compact(n, p) // a
        for p, a in factorize(k).items()
    )
```

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Legendre (prime k) | O(log_p n) | O(1) |
| Composite k | O(sqrt(k) + d * log n) | O(d) |

where d = number of distinct prime factors of k.

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Factorial Trailing Zeroes (172) | Legendre with p=5 (factors of 2 always exceed factors of 5) |
| Preimage Size of Factorial Zeroes Function (793) | Binary search on inverse of Legendre |

---

## 2. Binomial Coefficients

### Core Insight

C(n, k) = number of ways to choose k items from n (order doesn't matter). Equivalently, the coefficient of x^k in (1+x)^n.

### Key Formulas

| Formula | Name |
|---------|------|
| C(n,k) = n! / (k! * (n-k)!) | Definition |
| C(n,k) = C(n-1,k-1) + C(n-1,k) | Pascal's recurrence |
| C(n,k) = C(n, n-k) | Symmetry |
| C(n,k) = (n/k) * C(n-1, k-1) | Multiplicative formula |
| sum C(n,k) for k=0..n = 2^n | Row sum |
| sum C(n,k)^2 for k=0..n = C(2n, n) | Vandermonde |
| C(n,0) + C(n,2) + ... = C(n,1) + C(n,3) + ... = 2^(n-1) | Even/odd sum |

*Socratic prompt: "Pascal's recurrence says C(n,k) = C(n-1,k-1) + C(n-1,k). Can you explain this combinatorially? Think about one specific element — either it's chosen or it's not."*

### Computation Methods

| Method | Time | Space | When to Use |
|--------|------|-------|-------------|
| Direct multiplication | O(k) | O(1) | Single query, small k |
| Pascal's triangle DP | O(n^2) | O(n^2) or O(n) | Many queries, n ≤ ~5000 |
| Factorial + mod inverse | O(n) precompute, O(1) query | O(n) | Mod prime, many queries |
| Lucas's theorem | O(p + log_p n) | O(p) | Large n, small prime mod |

### Templates

**Direct computation (no mod, small values):**

```python
def comb_direct(n: int, k: int) -> int:
    """Compute C(n,k) without modular arithmetic.

    Time: O(k). Space: O(1).
    """
    if k < 0 or k > n:
        return 0
    if k > n - k:
        k = n - k
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result
```

**Precomputed factorials with mod inverse (prime mod):**

```python
MOD = 10**9 + 7

def precompute_factorials(n: int) -> tuple[list[int], list[int]]:
    """Precompute factorial and inverse factorial arrays mod MOD.

    Time: O(n). Space: O(n).
    """
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i % MOD
    inv_fact = [1] * (n + 1)
    inv_fact[n] = pow(fact[n], MOD - 2, MOD)
    for i in range(n - 1, -1, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
    return fact, inv_fact

# Usage:
# fact, inv_fact = precompute_factorials(MAXN)
# def C(n, k): return fact[n] * inv_fact[k] % MOD * inv_fact[n-k] % MOD if 0<=k<=n else 0
```

**Lucas's theorem (large n, small prime mod):**

```python
def lucas(n: int, k: int, p: int) -> int:
    """C(n, k) mod p using Lucas's theorem. p must be prime.

    Time: O(p + log_p(n)). Space: O(p).
    """
    if k == 0:
        return 1
    # Precompute small factorials mod p
    fact = [1] * p
    for i in range(1, p):
        fact[i] = fact[i - 1] * i % p
    inv_fact = [1] * p
    inv_fact[p - 1] = pow(fact[p - 1], p - 2, p)
    for i in range(p - 2, -1, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % p

    result = 1
    while n > 0 or k > 0:
        ni, ki = n % p, k % p
        if ki > ni:
            return 0
        result = result * fact[ni] % p * inv_fact[ki] % p * inv_fact[ni - ki] % p
        n //= p
        k //= p
    return result
```

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Pascal's Triangle (118) | Build the triangle row by row |
| Pascal's Triangle II (119) | O(k) space — single row |
| Unique Paths (62) | C(m+n-2, m-1) — lattice paths |
| Kth Smallest Instructions (1643) | Lexicographic rank using binomial coefficients |

---

## 3. Catalan Numbers

### Core Insight

The n-th Catalan number counts a remarkable family of combinatorial structures. It appears whenever you have a sequence of n "open" and n "close" operations that must remain balanced.

**Sequence:** 1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, ...

### Key Formulas

| Formula | Name |
|---------|------|
| C_n = C(2n, n) / (n+1) | Closed form |
| C_n = C(2n, n) - C(2n, n-1) | Ballot / reflection |
| C_n = sum C_k * C_{n-1-k} for k=0..n-1 | Recurrence (C_0 = 1) |
| C_n ~ 4^n / (n^(3/2) * sqrt(pi)) | Asymptotic growth |

### Combinatorial Interpretations

All of the following are counted by C_n:

1. **Balanced parentheses** with n pairs
2. **Monotonic lattice paths** from (0,0) to (n,n) staying below the diagonal
3. **Triangulations** of a convex (n+2)-gon
4. **Full binary trees** with n+1 leaves (equivalently n internal nodes)
5. **Binary search trees** with n nodes (structurally distinct)
6. **Non-crossing partitions** of {1, ..., n}
7. **Stack-sortable permutations** of length n
8. **Ways to parenthesize** a product of n+1 factors
9. **Dyck paths** of length 2n (paths that never go below x-axis)
10. **Mountain ranges** with n upstrokes and n downstrokes

*Socratic prompt: "Can you see why balanced parentheses and Dyck paths are the same problem in disguise? What does '(' map to? What does ')' map to?"*

### Proof of Closed Form (Reflection Principle)

Total paths from (0,0) to (2n, 0) with +1/-1 steps = C(2n, n). Bad paths (those dipping below 0) biject to ALL paths from (0,0) to (2n, -2), which equals C(2n, n-1). So C_n = C(2n,n) - C(2n, n-1) = C(2n,n)/(n+1).

### Templates

**Recurrence (small n, or need entire table):**

```python
def catalan_table(n: int, mod: int = 0) -> list[int]:
    """Compute Catalan numbers C_0 through C_n.

    Time: O(n^2). Space: O(n).
    """
    C = [0] * (n + 1)
    C[0] = 1
    for i in range(1, n + 1):
        for j in range(i):
            C[i] += C[j] * C[i - 1 - j]
        if mod:
            C[i] %= mod
    return C
```

**Closed form (single value, mod prime):**

```python
def catalan(n: int, mod: int = 10**9 + 7) -> int:
    """Compute n-th Catalan number mod prime.

    Time: O(n) for factorial precomputation. Space: O(n).
    """
    if n <= 1:
        return 1
    # C_n = C(2n, n) / (n + 1) mod prime
    fact = [1] * (2 * n + 1)
    for i in range(1, 2 * n + 1):
        fact[i] = fact[i - 1] * i % mod
    inv = pow(fact[2 * n], mod - 2, mod)
    numerator = fact[2 * n] * pow(fact[n], mod - 2, mod) % mod
    numerator = numerator * pow(fact[n], mod - 2, mod) % mod
    return numerator * pow(n + 1, mod - 2, mod) % mod
```

### Complexity

| Method | Time | Space |
|--------|------|-------|
| Recurrence | O(n^2) | O(n) |
| Closed form (single) | O(n) | O(n) |
| Closed form (precomputed) | O(1) per query | O(n) |

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Unique Binary Search Trees (96) | C_n directly — number of structurally distinct BSTs |
| Unique Binary Search Trees II (95) | Generate all BSTs — recursive Catalan decomposition |
| Generate Parentheses (22) | Enumerate all C_n balanced bracket strings |
| Different Ways to Add Parentheses (241) | Catalan-style recursive splitting |

---

## 4. Inclusion-Exclusion Principle

### Core Insight

Count elements in the **union** of overlapping sets by alternating between adding and subtracting intersection sizes:

```
|A1 ∪ A2 ∪ ... ∪ An| = Σ|Ai| - Σ|Ai∩Aj| + Σ|Ai∩Aj∩Ak| - ...
```

**Complement form** (elements in NONE of the sets):

```
|Ā1 ∩ Ā2 ∩ ... ∩ Ān| = Σ_{S⊆[n]} (-1)^|S| * |∩_{i∈S} Ai|
```

*Socratic prompt: "If you have three overlapping circles in a Venn diagram, why isn't the union just |A|+|B|+|C|? What gets counted too many times?"*

### Key Applications

**1. Count integers in [1, r] coprime to n:**

Let A_p = multiples of prime p in [1, r]. Use IE over the distinct prime factors of n.

```python
def count_coprime(n: int, r: int) -> int:
    """Count integers in [1, r] coprime to n.

    Time: O(2^d) where d = number of distinct prime factors of n.
    """
    primes = []
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            primes.append(d)
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        primes.append(temp)

    total = 0
    for mask in range(1, 1 << len(primes)):
        product = 1
        bits = 0
        for i in range(len(primes)):
            if mask >> i & 1:
                product *= primes[i]
                bits += 1
        if bits % 2 == 1:
            total += r // product
        else:
            total -= r // product
    return r - total
```

**2. Derangements (permutations with no fixed points):**

```
D_n = n! * Σ_{k=0}^{n} (-1)^k / k!
```

Equivalent recurrence: D_n = (n-1) * (D_{n-1} + D_{n-2}), with D_0 = 1, D_1 = 0.

```python
def derangements(n: int, mod: int = 10**9 + 7) -> int:
    """Count derangements of n elements.

    Time: O(n). Space: O(1).
    """
    if n == 0:
        return 1
    if n == 1:
        return 0
    a, b = 1, 0  # D_0, D_1
    for i in range(2, n + 1):
        a, b = b, (i - 1) * (a + b) % mod
    return b
```

**3. Stars-and-bars with upper bounds (see Section 6).**

### The General Pattern (Bitmask IE)

Most IE problems follow this structure:

```python
def inclusion_exclusion(bad_properties, universe_count_fn):
    """
    bad_properties: list of "bad" constraints
    universe_count_fn(subset): count of items satisfying ALL
        constraints in subset simultaneously.

    Time: O(2^k * cost_of_count_fn).
    """
    k = len(bad_properties)
    result = 0
    for mask in range(1 << k):
        subset = [bad_properties[i] for i in range(k) if mask >> i & 1]
        count = universe_count_fn(subset)
        if bin(mask).count('1') % 2 == 0:
            result += count
        else:
            result -= count
    return result
```

### Complexity

| Variant | Time |
|---------|------|
| k sets | O(2^k * per-intersection cost) |
| Coprime counting | O(2^d) where d = distinct prime factors |
| Derangements | O(n) |

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Count Primes (204) | Sieve (not IE), but IE can count coprimes in a range |
| Find the Count of Good Integers (3272) | IE over digit constraints |
| K-Inverse Pairs Array (629) | DP with IE flavor for permutation counting |

---

## 5. Burnside's Lemma / Polya Enumeration

### Core Insight

Count **distinct objects under symmetry** (equivalence classes). If a group G of symmetries acts on your objects, the number of distinct objects is:

```
|distinct| = (1/|G|) * Σ_{g ∈ G} |Fix(g)|
```

where Fix(g) = number of objects unchanged by symmetry g.

**Polya's specialization** for coloring problems: if you color n positions with k colors, then Fix(g) = k^(number of cycles in g's permutation representation).

```
|distinct| = (1/|G|) * Σ_{g ∈ G} k^C(g)
```

*Socratic prompt: "You have a necklace with 4 beads and 3 colors. Without symmetry, there are 3^4 = 81 colorings. But rotating a necklace shouldn't create a 'new' necklace. How many are truly distinct?"*

### Necklace Problem (Rotations Only)

Count distinct necklaces of n beads with k colors:

```
Answer = (1/n) * Σ_{d|n} φ(n/d) * k^d
```

where φ is Euler's totient function.

```python
from math import gcd

def euler_totient(n: int) -> int:
    """Euler's totient function φ(n)."""
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result

def count_necklaces(n: int, k: int) -> int:
    """Count distinct necklaces of n beads with k colors (rotations only).

    Time: O(sqrt(n) * d(n)) where d(n) = number of divisors.
    """
    total = 0
    for d in range(1, n + 1):
        if n % d == 0:
            total += euler_totient(n // d) * k**d
    return total // n
```

### General Burnside Template

```python
def count_cycles(perm: list[int]) -> int:
    """Count cycles in a permutation (0-indexed).

    Time: O(n). Space: O(n).
    """
    n = len(perm)
    visited = [False] * n
    cycles = 0
    for i in range(n):
        if not visited[i]:
            cycles += 1
            j = i
            while not visited[j]:
                visited[j] = True
                j = perm[j]
    return cycles

def burnside(symmetries: list[list[int]], k: int) -> int:
    """Count distinct colorings under a symmetry group.

    symmetries: list of permutations (each is a list mapping index -> new index).
    k: number of colors.

    Time: O(|G| * n). Space: O(n).
    """
    total = sum(k ** count_cycles(perm) for perm in symmetries)
    return total // len(symmetries)
```

### Example: Cube Face Coloring

Color 6 faces of a cube with k colors. The rotation group has 24 elements. Group the rotations by cycle structure:

| Rotation Type | Count | Cycles | Contribution |
|---------------|-------|--------|-------------|
| Identity | 1 | 6 | k^6 |
| Face rotations (90°/270°) | 6 | 3 | 6 * k^3 |
| Face rotations (180°) | 3 | 4 | 3 * k^4 |
| Vertex rotations (120°/240°) | 8 | 2 | 8 * k^2 |
| Edge rotations (180°) | 6 | 3 | 6 * k^3 |

Answer = (k^6 + 6k^3 + 3k^4 + 8k^2 + 6k^3) / 24

### Complexity

| Operation | Time |
|-----------|------|
| Necklaces (rotations only) | O(sqrt(n) * d(n)) |
| General Burnside | O(|G| * n) |
| Cycle counting | O(n) per permutation |

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Necklace coloring (Codeforces) | Direct necklace formula |
| SPOJ TRANSP | Burnside on transposition group |
| USACO Moocryptography | Symmetry counting |

---

## 6. Stars and Bars

### Core Insight

Count the number of ways to distribute n identical objects into k distinct bins. This is equivalent to counting non-negative integer solutions to x1 + x2 + ... + xk = n.

**Visual:** n stars and (k-1) bars form a row of n+k-1 symbols. Choose where to place the bars.

*Socratic prompt: "If you have 5 identical candies and 3 children, you can represent a distribution like ★★|★|★★ (2, 1, 2). How many total symbols are there? How many bar positions do you choose?"*

### Formulas

| Constraint | Formula | Substitution |
|------------|---------|-------------|
| x_i ≥ 0 | C(n + k - 1, k - 1) | Direct |
| x_i ≥ 1 | C(n - 1, k - 1) | y_i = x_i - 1, sum becomes n - k |
| x_i ≥ a_i | C(n - Σa_i + k - 1, k - 1) | y_i = x_i - a_i |
| 0 ≤ x_i ≤ b | IE on stars-and-bars (see below) | Subtract overcounts |

### Template: Basic Stars and Bars

```python
from math import comb

def stars_and_bars(n: int, k: int, min_per_bin: int = 0) -> int:
    """Count ways to distribute n identical items into k distinct bins.

    Each bin gets at least min_per_bin items.

    Time: O(k) for comb computation. Space: O(1).
    """
    n -= k * min_per_bin  # shift to >= 0 case
    if n < 0:
        return 0
    return comb(n + k - 1, k - 1)
```

### Template: Upper-Bounded Stars and Bars (IE)

When each x_i ≤ b, use inclusion-exclusion:

```python
def stars_and_bars_bounded(n: int, k: int, b: int) -> int:
    """Count non-negative integer solutions to x1+...+xk = n with 0 ≤ xi ≤ b.

    Time: O(k). Space: O(1).
    """
    result = 0
    for i in range(k + 1):
        remaining = n - i * (b + 1)
        if remaining < 0:
            break
        term = comb(k, i) * comb(remaining + k - 1, k - 1)
        if i % 2 == 0:
            result += term
        else:
            result -= term
    return result
```

### Complexity

| Variant | Time |
|---------|------|
| Basic (no upper bound) | O(k) |
| With upper bounds (IE) | O(min(k, n/(b+1))) |

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Combination Sum IV (377) | DP, but stars-and-bars intuition for ordered sums |
| Coin Change 2 (518) | Unordered — similar flavor, needs DP |
| Distribute Candies Among Children II (2929) | Direct stars-and-bars with upper bounds via IE |
| Number of Ways to Reach a Position After Exactly k Steps (2400) | Lattice path count via binomial |

---

## 7. Generating Combinations

### Core Insight

Generate all k-element subsets of {1, ..., n} in a specific order. Two approaches: lexicographic (natural ordering) and Gray code (minimal changes between consecutive subsets).

### Method 1: Lexicographic Order

Find the rightmost element that can be incremented, increment it, then fill remaining positions with consecutive values.

```python
def next_combination(a: list[int], n: int) -> bool:
    """Advance a to the next combination in lex order (1-indexed, in-place).

    a: current combination, sorted ascending, values in [1, n].
    Returns False if a is the last combination.

    Time: O(k) per call.
    """
    k = len(a)
    for i in range(k - 1, -1, -1):
        if a[i] < n - k + i + 1:
            a[i] += 1
            for j in range(i + 1, k):
                a[j] = a[j - 1] + 1
            return True
    return False

def all_combinations_lex(n: int, k: int):
    """Generate all C(n,k) combinations in lexicographic order.

    Time: O(k * C(n,k)). Space: O(k).
    """
    a = list(range(1, k + 1))  # first combination
    yield a[:]
    while next_combination(a, n):
        yield a[:]
```

### Method 2: Gray Code Order

Adjacent combinations differ by exactly one element swapped. Use the standard binary Gray code and filter for codes with exactly k set bits.

```python
def gray_code(i: int) -> int:
    """Convert integer to Gray code."""
    return i ^ (i >> 1)

def combinations_gray(n: int, k: int):
    """Generate k-subsets in Gray code order (differ by 1 swap each step).

    Time: O(n * 2^n) — iterates all 2^n codes, keeps those with k bits.
    """
    for i in range(1 << n):
        code = gray_code(i)
        if bin(code).count('1') == k:
            yield [j + 1 for j in range(n) if code >> j & 1]
```

*Socratic prompt: "Why would you want Gray-code order instead of lexicographic? Think about problems where changing one element is expensive."*

### Complexity

| Method | Per-step | Total |
|--------|----------|-------|
| Lexicographic | O(k) | O(k * C(n,k)) |
| Gray code (filter) | O(n) | O(n * 2^n) |
| Gray code (recursive) | O(n) amortized | O(n * C(n,k)) |

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Combinations (77) | Generate all C(n,k) subsets — backtracking or iterative |
| Next Permutation (31) | Same "find rightmost incrementable" pattern for permutations |
| Subsets (78) | All 2^n subsets — bitmask iteration |

---

## 8. Bishops on a Chessboard

### Core Insight

Count ways to place K non-attacking bishops on an N×N board. The key observation: diagonals split into two independent color classes (like black/white squares). A bishop on a "black" diagonal never attacks one on a "white" diagonal.

Solve each color independently with DP, then combine: answer = Σ D_black[i] * D_white[K-i] for i = 0..K.

*Socratic prompt: "Why can we split the board into two independent sub-problems? What geometric property of diagonals makes this work?"*

### Diagonal Structure

Number diagonals 1 to 2N-1 for each direction. Diagonals of the same parity belong to the same color class. The number of squares on diagonal i:

```python
def squares_on_diagonal(i: int) -> int:
    """Number of squares on the i-th diagonal (1-indexed)."""
    if i % 2 == 1:  # odd diagonals
        return (i // 4) * 2 + 1
    else:            # even diagonals
        return ((i - 1) // 4) * 2 + 2
```

### DP Recurrence

For diagonals of one color (step by 2):

```
D[i][j] = D[i-2][j] + D[i-2][j-1] * (squares(i) - (j-1))
```

- D[i-2][j]: skip diagonal i (place no bishop)
- D[i-2][j-1] * (squares(i) - (j-1)): place one bishop on diagonal i; there are (squares(i) - (j-1)) available squares (j-1 rows already occupied by previous bishops)

### Template

```python
def bishop_placements(n: int, k: int) -> int:
    """Count ways to place k non-attacking bishops on an n×n board.

    Time: O(n * k). Space: O(n * k).
    """
    if k > 2 * n - 1:
        return 0

    def sq(i):
        """Squares on diagonal i (1-indexed)."""
        return (i // 4) * 2 + 1 if i % 2 else ((i - 1) // 4) * 2 + 2

    max_diag = 2 * n
    D = [[0] * (k + 1) for _ in range(max_diag)]
    for i in range(max_diag):
        D[i][0] = 1

    if max_diag > 1:
        D[1][1] = 1

    for i in range(2, max_diag):
        for j in range(1, k + 1):
            D[i][j] = D[i - 2][j]
            avail = sq(i) - (j - 1)
            if avail > 0:
                D[i][j] += D[i - 2][j - 1] * avail

    # Combine black diagonals (index 2N-1) and white diagonals (index 2N-2)
    result = 0
    for i in range(k + 1):
        result += D[max_diag - 1][i] * D[max_diag - 2][k - i]
    return result
```

### Connection to N-Queens

The N-Queens problem restricts pieces on both diagonals AND rows/columns. Bishops only require diagonal non-attack, making the independent color decomposition possible. N-Queens cannot be split this way because row/column constraints couple the two diagonal directions.

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Bishop placements | O(N * K) | O(N * K) |
| With space optimization | O(N * K) | O(K) |

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| N-Queens (51) | Related but harder — no diagonal independence |
| N-Queens II (52) | Count solutions — backtracking required |
| Bishops placement (Codeforces) | Direct application |

---

## 9. Balanced Bracket Sequences

### Core Insight

Balanced bracket sequences with n pairs are counted by the n-th Catalan number C_n. Beyond counting, we can efficiently find the k-th sequence, compute the rank of a sequence, and find the next sequence in lexicographic order.

### Counting

```
Single bracket type, n pairs: C_n = C(2n, n) / (n + 1)
Multiple bracket types (k types), n pairs: C_n * k^n
```

### Validation

```python
def is_balanced(s: str) -> bool:
    """Check if a bracket sequence is balanced.

    Time: O(n). Space: O(1) for single type, O(n) for multiple types.
    """
    depth = 0
    for ch in s:
        if ch == '(':
            depth += 1
        else:
            depth -= 1
        if depth < 0:
            return False
    return depth == 0
```

### DP Table for Rank/Unrank Operations

d[i][j] = number of balanced sequences of length i with current balance j (number of unmatched open brackets).

```python
def bracket_dp(n: int) -> list[list[int]]:
    """Precompute d[i][j] for bracket sequence indexing.

    d[i][j] = number of valid completions of length i with balance j.

    Time: O(n^2). Space: O(n^2).
    """
    d = [[0] * (n + 1) for _ in range(2 * n + 1)]
    d[0][0] = 1
    for i in range(1, 2 * n + 1):
        for j in range(n + 1):
            if j > 0:
                d[i][j] += d[i - 1][j - 1]  # place '('
            if j + 1 <= n:
                d[i][j] += d[i - 1][j + 1]  # place ')'
    return d
```

### Finding the k-th Sequence (0-indexed)

```python
def kth_bracket_sequence(n: int, k: int) -> str:
    """Find the k-th balanced bracket sequence (0-indexed, lex order).

    Time: O(n) given precomputed d table.
    """
    d = bracket_dp(n)
    result = []
    depth = 0
    for i in range(2 * n):
        remaining = 2 * n - i - 1
        # Count sequences starting with '(' at this position
        open_count = d[remaining][depth + 1] if depth + 1 <= n else 0
        if k < open_count:
            result.append('(')
            depth += 1
        else:
            k -= open_count
            result.append(')')
            depth -= 1
    return ''.join(result)
```

### Finding the Next Sequence (Lexicographic)

```python
def next_bracket_sequence(s: str) -> str | None:
    """Return the lexicographically next balanced bracket sequence, or None.

    Time: O(n). Space: O(n).
    """
    n = len(s)
    depth = 0
    for i in range(n - 1, -1, -1):
        if s[i] == '(':
            depth -= 1
        else:
            depth += 1
        # Can we replace '(' with ')' here and still complete?
        if s[i] == '(' and depth > 0:
            depth -= 1  # changed '(' to ')'
            open_remaining = (n - i - 1 - depth) // 2
            close_remaining = n - i - 1 - open_remaining
            return s[:i] + ')' + '(' * open_remaining + ')' * close_remaining
    return None  # already the last sequence
```

*Socratic prompt: "The next-sequence algorithm scans right-to-left looking for a '(' to flip to ')'. Why right-to-left? What makes a position 'flippable'?"*

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Counting (Catalan) | O(n) | O(n) |
| DP table | O(n^2) | O(n^2) |
| k-th sequence | O(n) | O(n) |
| Next sequence | O(n) | O(n) |
| Rank of sequence | O(n) | O(n) |

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| Generate Parentheses (22) | Enumerate all — backtracking with balance tracking |
| Valid Parentheses (20) | Validation — stack for multiple bracket types |
| Longest Valid Parentheses (32) | DP or stack — longest balanced substring |
| Score of Parentheses (856) | Depth-weighted scoring of balanced sequence |

---

## 10. Counting Labeled Graphs

### Core Insight

Count the number of graphs (labeled) on n vertices. Each of the C(n, 2) = n(n-1)/2 possible edges is independently present or absent.

### Key Formulas

**All labeled graphs:**

```
G_n = 2^(n*(n-1)/2)
```

**Connected labeled graphs C_n:** Use the recurrence based on isolating the component containing vertex 1:

```
C_n = G_n - (1/n) * Σ_{k=1}^{n-1} k * C(n,k) * C_k * G_{n-k}
```

**Labeled graphs with exactly m connected components D[n][m]:**

```
D[n][m] = Σ_{s=1}^{n} C(n-1, s-1) * C_s * D[n-s][m-1]
```

Base: D[0][0] = 1.

*Socratic prompt: "There are 2^(n(n-1)/2) labeled graphs on n vertices. For n=4, that's 2^6 = 64 graphs. How many of those are connected? Can you reason about why most large random graphs are connected?"*

### Template

```python
from math import comb

def count_labeled_graphs(n: int) -> tuple[list[int], list[int]]:
    """Count all and connected labeled graphs on 1..n vertices.

    Returns (G, C) where G[i] = total graphs, C[i] = connected graphs.

    Time: O(n^2). Space: O(n).
    """
    G = [0] * (n + 1)
    C = [0] * (n + 1)

    for i in range(n + 1):
        G[i] = 1 << (i * (i - 1) // 2)

    C[0] = 0
    C[1] = 1

    for i in range(2, n + 1):
        total = 0
        for k in range(1, i):
            total += k * comb(i, k) * C[k] * G[i - k]
        C[i] = G[i] - total // i

    return G, C

def count_labeled_k_components(n: int, max_k: int) -> list[list[int]]:
    """Count labeled graphs on n vertices with exactly m components.

    Returns D where D[i][m] = graphs on i vertices with m components.

    Time: O(n^2 * max_k). Space: O(n * max_k).
    """
    _, C = count_labeled_graphs(n)

    D = [[0] * (max_k + 1) for _ in range(n + 1)]
    D[0][0] = 1

    for i in range(1, n + 1):
        for m in range(1, min(i, max_k) + 1):
            for s in range(1, i + 1):
                D[i][m] += comb(i - 1, s - 1) * C[s] * D[i - s][m - 1]

    return D
```

### Key Values

| n | G_n (all) | C_n (connected) |
|---|-----------|----------------|
| 1 | 1 | 1 |
| 2 | 2 | 1 |
| 3 | 8 | 4 |
| 4 | 64 | 38 |
| 5 | 1024 | 728 |
| 6 | 32768 | 26704 |

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| G_n (all graphs) | O(1) per value | O(n) |
| C_n (connected) | O(n^2) | O(n) |
| D[n][m] (m components) | O(n^2 * m) | O(n * m) |

### Practice Problems

| Problem | Key Twist |
|---------|-----------|
| -- (competitive programming) | Direct application in counting problems |
| Connected graph counting (Codeforces) | The C_n recurrence |
| Chromatic polynomial problems | Related graph enumeration |

---

## Interview Tips

1. **Start with small cases.** Combinatorics problems often click after computing n=0,1,2,3 by hand. Look up the sequence in OEIS if stuck.

2. **Recognize Catalan.** If the answer sequence starts 1, 1, 2, 5, 14, 42 — it's almost certainly Catalan. Know the top 5 Catalan interpretations cold.

3. **Think "complement."** If counting "good" objects is hard, count ALL objects minus "bad" ones. This is the inclusion-exclusion mindset.

4. **Mod arithmetic is non-negotiable.** Almost every combinatorics contest problem requires mod 10^9+7. Precompute factorials and inverse factorials. See `math-techniques.md` Section 1.

5. **Stars and bars is your friend for distribution problems.** "How many ways to distribute n items into k groups?" — if items are identical and groups are distinct, it's stars and bars.

6. **Inclusion-exclusion has exponential cost.** Only use when the number of "bad" properties is small (≤ ~20). For larger problems, look for DP or Mobius inversion.

7. **Burnside is rare in interviews but devastating in contests.** If a problem mentions "distinct under rotation" or "up to symmetry," Burnside is the tool. Count cycles.

8. **Cross-reference your formulas.** Catalan = ballot problem = reflection principle = bracket sequences. Stars and bars + IE = bounded distribution. The same formula appears in many guises.

---

## Attribution

Content synthesized from cp-algorithms.com articles on [Factorial Divisors](https://cp-algorithms.com/algebra/factorial-divisors.html), [Binomial Coefficients](https://cp-algorithms.com/combinatorics/binomial-coefficients.html), [Catalan Numbers](https://cp-algorithms.com/combinatorics/catalan-numbers.html), [Inclusion-Exclusion Principle](https://cp-algorithms.com/combinatorics/inclusion-exclusion.html), [Burnside's Lemma](https://cp-algorithms.com/combinatorics/burnside.html), [Stars and Bars](https://cp-algorithms.com/combinatorics/stars_and_bars.html), [Generating Combinations](https://cp-algorithms.com/combinatorics/generating_combinations.html), [Bishops on a Chessboard](https://cp-algorithms.com/combinatorics/bishops-on-chessboard.html), [Balanced Bracket Sequences](https://cp-algorithms.com/combinatorics/bracket_sequences.html), and [Counting Labeled Graphs](https://cp-algorithms.com/combinatorics/counting_labeled_graphs.html). Algorithms and complexity analyses are from the original articles; code has been translated to Python with added commentary for the leetcode-teacher reference format.
