# Math Techniques

Essential mathematical techniques for coding interviews: modular arithmetic, fast exponentiation, GCD/LCM, prime number algorithms, and factorial properties. For bit manipulation (a related "math-adjacent" topic), see `bit-manipulation.md`. For number theory in DP contexts, see `dynamic-programming-core.md`.

---

## Quick Reference Table

| Pattern | Recognition Signals | Key Problems | Section |
|---------|-------------------|--------------|---------|
| Modular Arithmetic | "Return answer mod 10^9+7", "large number modulo" | -- (ubiquitous in contests) | 1 |
| Fast Exponentiation | "Compute a^b mod m", "pow(a, b) efficiently" | Pow(x, n) (50), Super Pow (372) | 2 |
| GCD / LCM | "Greatest common divisor", "least common multiple" | -- (building block) | 3 |
| Sieve of Eratosthenes | "Count primes", "find all primes up to n" | Count Primes (204) | 4 |
| Factorial Zeros | "Trailing zeros in n!", "preimage of factorial trailing zeros" | Trailing Zeroes (172), Preimage Size (793) | 5 |

---

## Corner Cases

- **Negative numbers:** Check if the problem input can be negative. Many math problems assume positive integers — verify constraints.
- **Floats:** Never compare floats with `==`. Use epsilon comparison: `abs(x - y) <= 1e-6` (see Comparing Floats section below).
- **Multiplication overflow:** `a * b` can overflow in fixed-width languages even if both factors fit. In Python this isn't an issue (arbitrary precision), but in Java/C++ multiply before mod.
- **Division by zero:** Always guard against division/modulo by zero. Check denominators before dividing.
- **Multiply by 1 / add 0:** These are identity operations. Ensure your logic doesn't accidentally skip them or break on them.

---

## Common Formulas

Quick reference for frequently needed math formulas in interviews:

| Formula | Expression | Notes |
|---------|-----------|-------|
| Check even | `num % 2 == 0` | Equivalent to `num & 1 == 0` |
| Sum of 1 to N | `N * (N + 1) // 2` | Gauss's formula, avoids O(N) loop |
| Sum of GP (2^0 to 2^n) | `2**(n + 1) - 1` | Geometric progression with ratio 2 |
| Permutations P(N, K) | `N! / (N - K)!` | Order matters |
| Combinations C(N, K) | `N! / (K! * (N - K)!)` | Order doesn't matter |

*Socratic prompt: "If someone asks you to sum the first 100 positive integers in O(1), what formula would you use? Can you prove why it works?"*

---

## 1. Modular Arithmetic

### Core Insight

When a problem says "return the answer modulo 10^9 + 7," you must apply modular arithmetic **at every step** to prevent overflow. The key properties:

| Property | Formula |
|----------|---------|
| Addition | `(a + b) % m = ((a % m) + (b % m)) % m` |
| Subtraction | `(a - b) % m = ((a % m) - (b % m) + m) % m` |
| Multiplication | `(a * b) % m = ((a % m) * (b % m)) % m` |
| **Division** | `(a / b) % m = (a * b^(m-2)) % m` (Fermat's little theorem, m prime) |

**Critical:** Division does NOT distribute over modulo directly. You must use the modular multiplicative inverse (Fermat's little theorem when m is prime).

> **Non-prime modulus?** When m is not prime, Fermat's little theorem doesn't apply. Use the Extended Euclidean Algorithm instead — see `modular-arithmetic-advanced.md` Section 1 and `number-theory-advanced.md` Section 1 for the general modular inverse.

### Why 10^9 + 7?

- It's prime (so Fermat's little theorem works for modular inverse)
- It fits in a 32-bit signed integer
- `(10^9 + 7)^2` fits in a 64-bit integer (so multiplication of two mod values won't overflow)

### Template

```python
MOD = 10**9 + 7

def add_mod(a, b):
    return (a + b) % MOD

def sub_mod(a, b):
    return (a - b + MOD) % MOD

def mul_mod(a, b):
    return (a * b) % MOD

def div_mod(a, b):
    """Division via modular inverse (Fermat's little theorem)."""
    return mul_mod(a, pow(b, MOD - 2, MOD))
```

*Socratic prompt: "Why can't you just compute the full answer and take modulo at the end? What goes wrong with very large intermediate values?"*

---

## 2. Fast Exponentiation

### Core Insight

Compute `a^b` in **O(log b)** instead of O(b) using repeated squaring. The idea: decompose the exponent in binary.

```
a^13 = a^(1101₂) = a^8 * a^4 * a^1
```

At each step, square the base and check if the current bit of the exponent is 1.

### Template

```python
def fast_pow(a: int, b: int, mod: int = None) -> int:
    """Compute a^b (mod m) in O(log b) time."""
    result = 1
    a = a % mod if mod else a
    while b > 0:
        if b & 1:  # Current bit is 1
            result = result * a
            if mod:
                result %= mod
        b >>= 1   # Move to next bit
        a = a * a
        if mod:
            a %= mod
    return result
```

**Note:** Python's built-in `pow(a, b, mod)` already uses fast exponentiation. Use it in interviews unless asked to implement from scratch.

### Application: Super Pow (LC 372)

Compute `a^b` where `b` is represented as an array of digits.

```python
def super_pow(a: int, b: list[int]) -> int:
    """a^[1,5,6,4] = a^1564, computed digit by digit."""
    MOD = 1337
    result = 1
    for digit in b:
        # result = result^10 * a^digit
        result = pow(result, 10, MOD) * pow(a, digit, MOD) % MOD
    return result
```

*Socratic prompt: "If you compute a^1564 directly, how many multiplications is that? With repeated squaring, how many? What's the general relationship?"*

### Problems

| Problem | Key Twist |
|---------|-----------|
| Pow(x, n) (50) | Handle negative exponents: `x^(-n) = 1 / x^n` |
| Super Pow (372) | Exponent as digit array, decompose: `a^(10k+d) = (a^k)^10 * a^d` |

### Comparing Floats

Floating point arithmetic introduces rounding errors. Never use `==` to compare floats:

```python
# BAD: will fail due to floating point precision
if x == y:
    ...

# GOOD: epsilon comparison
EPSILON = 1e-6
if abs(x - y) <= EPSILON:
    ...
```

**When this matters:** Sqrt(x) (LC 69) when using Newton's method, geometric distance calculations, and any problem involving division that produces non-integer results.

*Socratic prompt: "Why does 0.1 + 0.2 != 0.3 in most programming languages? How does this affect algorithm correctness?"*

---

## 3. GCD and LCM

### Core Insight

The **Euclidean algorithm** computes GCD in O(log(min(a, b))) time. LCM follows directly from GCD.

### Template

```python
def gcd(a: int, b: int) -> int:
    """Euclidean algorithm: gcd(a, b) = gcd(b, a % b)."""
    while b:
        a, b = b, a % b
    return a

def lcm(a: int, b: int) -> int:
    """LCM via GCD: lcm(a, b) = a * b / gcd(a, b)."""
    return a // gcd(a, b) * b  # Divide first to avoid overflow
```

**Why it works:** If `d` divides both `a` and `b`, then `d` also divides `a % b` (since `a % b = a - k*b` for some integer k). So `gcd(a, b) = gcd(b, a % b)`, and we recurse until one value is 0.

**Python shortcut:** `math.gcd(a, b)` (Python 3.5+), `math.lcm(a, b)` (Python 3.9+).

*Socratic prompt: "Trace through gcd(48, 18) step by step. How many steps does it take? Why is this O(log n)?"*

---

## 4. Sieve of Eratosthenes

### Core Insight

To find all primes up to n, start with all numbers marked as prime. For each prime p found, mark all its multiples `p^2, p^2+p, p^2+2p, ...` as composite. Only need to sieve up to `√n`.

### Template

```python
def count_primes(n: int) -> int:
    """Count primes less than n using Sieve of Eratosthenes."""
    if n <= 2:
        return 0
    is_prime = [True] * n
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            # Mark multiples of i starting from i*i
            for j in range(i * i, n, i):
                is_prime[j] = False
    return sum(is_prime)
```

**Complexity:** O(n log log n) time, O(n) space.

### Why Start from i*i?

All multiples of `i` smaller than `i*i` have already been marked by smaller primes. For example, when `i = 5`, `5*2`, `5*3`, `5*4` were already marked when processing 2 and 3.

### Optimization: Sieve Only Odd Numbers

```python
def count_primes_optimized(n: int) -> int:
    """Optimized sieve: skip even numbers, halve space."""
    if n <= 2:
        return 0
    # is_prime[i] represents whether (2*i + 1) is prime
    size = (n - 1) // 2
    is_prime = [True] * size
    count = 1  # Count 2 as prime
    for i in range(size):
        if is_prime[i]:
            val = 2 * i + 3
            count += 1
            # Mark odd multiples of val starting from val*val
            start = (val * val - 3) // 2
            for j in range(start, size, val):
                is_prime[j] = False
    return count
```

*Socratic prompt: "Why do we only need to sieve up to √n? If a number n has no prime factor ≤ √n, what can we conclude?"*

### Problems

| Problem | Key Twist |
|---------|-----------|
| Count Primes (204) | Direct sieve application |
| Ugly Number II (264) | Related: generate numbers with only factors 2, 3, 5 (see `classic-interview-problems.md`) |

---

## 5. Factorial Trailing Zeros

### Core Insight

Trailing zeros in `n!` come from factors of 10 = 2 × 5. Since there are always more factors of 2 than 5 in `n!`, **count the factors of 5**.

### Counting Factors of 5

Naively: check every number from 1 to n for divisibility by 5. But some numbers contribute multiple 5s (25 = 5², 125 = 5³, etc.).

**Formula:** `trailing_zeros(n) = ⌊n/5⌋ + ⌊n/25⌋ + ⌊n/125⌋ + ...`

### Template (LC 172)

```python
def trailing_zeroes(n: int) -> int:
    """Count trailing zeros in n! by counting factors of 5."""
    count = 0
    power_of_5 = 5
    while power_of_5 <= n:
        count += n // power_of_5
        power_of_5 *= 5
    return count
```

**Complexity:** O(log₅ n) time, O(1) space.

*Socratic prompt: "How many trailing zeros does 100! have? Walk through the formula step by step. Why do we count ⌊100/25⌋ separately from ⌊100/5⌋?"*

### Application: Preimage Size of Factorial Zeros (LC 793)

Given a target number of trailing zeros `k`, how many non-negative integers `n` have `trailing_zeroes(n!) = k`? The answer is always 0 or 5.

**Why?** `trailing_zeroes(n)` is a non-decreasing step function that increases by 1 at every multiple of 5, but jumps by 2 at multiples of 25 (and more at 125, etc.). So either exactly 5 consecutive values of n share the same trailing zero count, or the count jumps over `k` entirely.

```python
def preimage_size_fzf(k: int) -> int:
    """Binary search for the range of n where trailing_zeroes(n!) = k."""
    def trailing_zeroes(n):
        count = 0
        while n >= 5:
            n //= 5
            count += n
        return count

    def left_bound(target):
        lo, hi = 0, 5 * target + 1
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if trailing_zeroes(mid) < target:
                lo = mid + 1
            else:
                hi = mid
        return lo

    # If left bound for k equals left bound for k+1, then no n has exactly k zeros
    return left_bound(k + 1) - left_bound(k)
```

*Socratic prompt: "Why is the answer always 0 or 5? Think about what happens to trailing_zeroes(n) as n goes from 24 to 25 to 26."*

### Problems

| Problem | Key Twist |
|---------|-----------|
| Factorial Trailing Zeroes (172) | Count factors of 5 |
| Preimage Size of Factorial Zeroes (793) | Binary search on the trailing_zeroes function |

---

## Interview Tips

- **Always ask about negative numbers.** Many math problems have subtle behavior changes with negative inputs (e.g., Pow(x, n) with negative exponent, integer division rounding direction).
- **Watch for overflow/underflow.** In languages with fixed-width integers, intermediate multiplication results can overflow. Apply modular arithmetic at every step, not just at the end.
- **Guard against division/modulo by zero.** Even if the problem constraints say "non-zero," defensive checks show good engineering habits.
- **Mention time complexity optimizations.** Interviewers love hearing "I can compute this in O(log n) using fast exponentiation instead of O(n)."

---

## Practice Questions

### Essential

| Problem | Key Concept |
|---------|-------------|
| Sqrt(x) (69) | Binary search or Newton's method; float comparison matters |
| Pow(x, n) (50) | Fast exponentiation with negative exponent handling |

### Recommended

| Problem | Key Concept |
|---------|-------------|
| Integer to English Words (273) | Decomposition by place value (thousands, millions, billions) |

---

## Attribution

Content synthesized from labuladong's algorithmic guide, Chapter 4 — "Other Common Techniques: Math Techniques," and the Tech Interview Handbook (techinterviewhandbook.org) math cheatsheet. Reorganized and augmented with Socratic teaching prompts, cross-references, and code templates for the leetcode-teacher skill.
