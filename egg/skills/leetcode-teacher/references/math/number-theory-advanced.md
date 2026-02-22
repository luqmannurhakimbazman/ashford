# Number Theory (Advanced)

Advanced number-theoretic algorithms for coding interviews: Extended Euclidean, Fibonacci fast computation, primality testing, integer factorization, Euler's totient, and divisor functions. For basic modular arithmetic, GCD, and sieve, see `math-techniques.md`. For modular inverses with non-prime modulus and CRT, see `modular-arithmetic-advanced.md`. For Fibonacci DP, see `dynamic-programming-core.md`.

---

## Quick Reference Table

| Pattern | Recognition Signals | Key Problems | Section |
|---------|-------------------|--------------|---------|
| Extended Euclidean | "Modular inverse (non-prime mod)", "Bezout coefficients" | -- (building block) | 1 |
| Fibonacci (fast) | "N-th Fibonacci mod m", "matrix exponentiation", "Pisano period" | Fibonacci Number (509), Climbing Stairs (70) | 2 |
| Linear Sieve | "Smallest prime factor for all n", "fast range factorization" | -- (preprocessing tool) | 3 |
| Primality Testing | "Is n prime?", "large number primality" | Count Primes (204) | 4 |
| Integer Factorization | "Find all prime factors", "prime factorization" | -- (building block) | 5 |
| Euler's Totient | "Count coprimes", "a^phi(m) = 1 mod m" | -- (modular arithmetic) | 6 |
| Divisor Functions | "Count divisors", "sum of divisors", "perfect number" | -- (number theory) | 7 |

---

## Corner Cases

- **n = 0 or 1:** Most number-theoretic functions have special-case definitions (phi(1) = 1, d(1) = 1, 0 is neither prime nor composite).
- **Negative inputs:** Extended Euclidean works with negatives, but verify problem constraints.
- **Large n with mod:** When computing Fibonacci mod m, use matrix exponentiation or fast doubling — never iterate to 10^18.
- **gcd(a, m) != 1:** Modular inverse does not exist. Extended Euclidean reveals this; handle gracefully.

---

## 1. Extended Euclidean Algorithm

### Core Insight

Finds coefficients x, y such that **a*x + b*y = gcd(a, b)** (Bezout's identity). This extends the standard Euclidean algorithm by back-propagating coefficients through the recursion.

**Why it matters:** This is the general method for computing modular inverses — it works even when the modulus is **not prime** (unlike Fermat's little theorem). See `math-techniques.md` Section 1 for the Fermat-based inverse (prime modulus only).

### Template

```python
def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return (gcd, x, y) such that a*x + b*y = gcd(a, b)."""
    if b == 0:
        return a, 1, 0
    g, x1, y1 = extended_gcd(b, a % b)
    return g, y1, x1 - (a // b) * y1

def mod_inverse(a: int, m: int) -> int | None:
    """Return a^(-1) mod m, or None if inverse doesn't exist."""
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        return None  # inverse doesn't exist
    return x % m
```

**Iterative version** (avoids recursion depth issues):

```python
def extended_gcd_iter(a: int, b: int) -> tuple[int, int, int]:
    """Iterative Extended Euclidean Algorithm."""
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_r, old_s, old_t  # gcd, x, y
```

**Complexity:** O(log(min(a, b))) time — same as standard Euclidean algorithm.

*Socratic prompt: "The standard GCD algorithm gives you gcd(a, b). The extended version also gives you x and y. Why would you ever need those coefficients? Think about what happens when gcd(a, m) = 1 and you want to solve a*x = 1 (mod m)."*

### Problems

| Problem | Key Twist |
|---------|-----------|
| -- (building block) | Used to compute modular inverse when modulus is not prime |

---

## 2. Fibonacci Numbers (Fast Computation)

### Core Insight

The naive O(n) iterative approach is fine for small n, but for n up to 10^18 (with modular arithmetic), you need **O(log n)** methods. Two approaches:

1. **Matrix exponentiation:** `[[1,1],[1,0]]^n` gives F(n) in position [0][1]
2. **Fast doubling:** Uses identities F(2k) = F(k)*(2*F(k+1) - F(k)) and F(2k+1) = F(k)^2 + F(k+1)^2

### Key Properties

| Property | Formula | Interview Use |
|----------|---------|---------------|
| Cassini's identity | F(n-1)*F(n+1) - F(n)^2 = (-1)^n | Verify Fibonacci membership |
| Addition rule | F(n+k) = F(k)*F(n+1) + F(k-1)*F(n) | Derive fast doubling |
| GCD identity | gcd(F(m), F(n)) = F(gcd(m, n)) | GCD of Fibonacci numbers |
| Pisano period | F(n) mod m is periodic with period <= m^2 | Modular Fibonacci |
| Zeckendorf's theorem | Every N is a unique sum of non-consecutive Fibs | Greedy representation |

### Template: Fast Doubling (Preferred)

```python
def fib(n: int, mod: int = None) -> int:
    """Compute F(n) in O(log n) using fast doubling."""
    def _fib(n):
        if n == 0:
            return (0, 1)
        a, b = _fib(n >> 1)
        c = a * (2 * b - a)
        d = a * a + b * b
        if mod:
            c %= mod
            d %= mod
        if n & 1:
            return (d, (c + d) % mod if mod else c + d)
        return (c, d)
    return _fib(n)[0]
```

### Template: Matrix Exponentiation (Generalizes to Any Linear Recurrence)

```python
def mat_mult(A: list, B: list, mod: int) -> list:
    """Multiply two 2x2 matrices mod m."""
    return [
        [(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % mod,
         (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % mod],
        [(A[1][0]*B[0][0] + A[1][1]*B[1][0]) % mod,
         (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % mod],
    ]

def mat_pow(M: list, n: int, mod: int) -> list:
    """Compute M^n mod m using binary exponentiation."""
    result = [[1, 0], [0, 1]]  # identity
    while n:
        if n & 1:
            result = mat_mult(result, M, mod)
        M = mat_mult(M, M, mod)
        n >>= 1
    return result

def fib_matrix(n: int, mod: int = 10**9 + 7) -> int:
    """F(n) mod m via matrix exponentiation."""
    if n == 0:
        return 0
    return mat_pow([[1, 1], [1, 0]], n, mod)[0][1]
```

**Why matrix exponentiation matters beyond Fibonacci:** Any linear recurrence `f(n) = c1*f(n-1) + c2*f(n-2) + ...` can be solved in O(k^3 * log n) where k is the recurrence order.

**Complexity:** O(log n) for both methods (matrix mult is O(1) for 2x2).

*Socratic prompt: "If someone asks you to compute the 10^18-th Fibonacci number mod 10^9+7, the naive loop would take 10^18 iterations. How does fast doubling reduce this to about 60 steps? What's the analogy to fast exponentiation?"*

### Problems

| Problem | Key Twist |
|---------|-----------|
| Fibonacci Number (509) | Direct computation (O(n) suffices for small n) |
| Climbing Stairs (70) | F(n+1) in disguise — each step is 1 or 2 |
| Split Array Into Fibonacci Sequence (842) | Backtracking with Fibonacci constraint |
| Length of Longest Fibonacci Subsequence (873) | DP on Fibonacci-like subsequences |

---

## 3. Linear Sieve

### Core Insight

The standard Sieve of Eratosthenes (see `math-techniques.md` Section 4) runs in O(n log log n) and marks composites multiple times. The **linear sieve** ensures each composite is marked exactly once — by its smallest prime factor. This gives:

1. **Strict O(n) time** (vs O(n log log n))
2. **Smallest prime factor (spf)** for every number, enabling O(log n) factorization per query

**Trade-off:** Uses O(n) integers (~32x more memory than a bitset sieve). Practical for n <= 10^7.

### Template

```python
def linear_sieve(n: int) -> tuple[list[int], list[int]]:
    """Return (primes, spf) where spf[i] = smallest prime factor of i."""
    spf = [0] * (n + 1)  # smallest prime factor
    primes = []
    for i in range(2, n + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > n:
                break
            spf[i * p] = p
    return primes, spf

def factorize_with_spf(n: int, spf: list[int]) -> dict[int, int]:
    """Factorize n in O(log n) using precomputed smallest prime factors."""
    factors = {}
    while n > 1:
        p = spf[n]
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
    return factors
```

**When to use over standard sieve:**
- Need smallest prime factor for range queries
- Computing multiplicative functions (phi, d, sigma) for all numbers up to n
- Many factorizations in [2, n] after one O(n) precomputation

*Socratic prompt: "The standard sieve marks 12 as composite when processing 2 AND when processing 3. The linear sieve marks it only once. How does the 'break when p == spf[i]' condition achieve this?"*

---

## 4. Primality Testing

### Core Insight

Three tiers of primality testing, from simplest to most powerful:

| Method | Time | When to Use |
|--------|------|-------------|
| Trial division | O(sqrt(n)) | n <= 10^12, simple problems |
| Fermat test | O(k log n) | Quick probabilistic check (beware Carmichael numbers) |
| Miller-Rabin | O(k log^2 n) | Deterministic for n < 2^64 with fixed bases |

### Template: Trial Division

```python
def is_prime(n: int) -> bool:
    """O(sqrt(n)) primality test."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
```

### Template: Miller-Rabin (Deterministic for n < 2^64)

```python
def miller_rabin(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 2^64."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as 2^s * d
    s, d = 0, n - 1
    while d % 2 == 0:
        d //= 2
        s += 1

    # These bases are sufficient for all n < 2^64
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if n == a:
            return True
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False  # composite
    return True
```

**Key idea:** Write n-1 = 2^s * d. If n is prime, then for any base a, either a^d = 1 (mod n) or a^(2^r * d) = -1 (mod n) for some 0 <= r < s. If neither holds, n is composite.

*Socratic prompt: "Fermat's test says 'if a^(n-1) != 1 mod n, then n is composite.' Miller-Rabin adds a stronger check. What extra information does it extract from the squaring steps?"*

### Problems

| Problem | Key Twist |
|---------|-----------|
| Count Primes (204) | Use sieve, not per-number test (see `math-techniques.md`) |

---

## 5. Integer Factorization

### Core Insight

| Method | Time | Best For |
|--------|------|----------|
| Trial division | O(sqrt(n)) | n <= 10^12, interview default |
| Wheel factorization | O(sqrt(n)) with ~3x speedup | Optimized trial division |
| Pollard's rho | O(n^(1/4)) expected | n up to ~10^18 |

### Template: Trial Division (Interview Standard)

```python
def factorize(n: int) -> dict[int, int]:
    """Return prime factorization as {prime: exponent}."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors
```

### Template: Pollard's Rho (Competitive Programming)

```python
import math
import random

def pollard_rho(n: int) -> int:
    """Find a non-trivial factor of n (n must be composite)."""
    if n % 2 == 0:
        return 2
    while True:
        x = random.randint(2, n - 1)
        y = x
        c = random.randint(1, n - 1)
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

def full_factorize(n: int) -> dict[int, int]:
    """Complete factorization using Miller-Rabin + Pollard's rho."""
    if n <= 1:
        return {}
    factors = {}
    stack = [n]
    while stack:
        x = stack.pop()
        if x == 1:
            continue
        if miller_rabin(x):
            factors[x] = factors.get(x, 0) + 1
        else:
            d = pollard_rho(x)
            stack.extend([d, x // d])
    return factors
```

**In interviews:** Trial division is almost always sufficient. Mention Pollard's rho as a follow-up if the interviewer asks about very large numbers.

*Socratic prompt: "Trial division checks every number up to sqrt(n). Why is sqrt(n) the right stopping point? What can you conclude if no factor is found by then?"*

---

## 6. Euler's Totient Function

### Core Insight

**phi(n)** counts integers in [1, n] that are coprime to n. From the prime factorization n = p1^a1 * p2^a2 * ... * pk^ak:

```
phi(n) = n * (1 - 1/p1) * (1 - 1/p2) * ... * (1 - 1/pk)
```

### Key Properties

| Property | Formula | Use |
|----------|---------|-----|
| Prime | phi(p) = p - 1 | Base case |
| Prime power | phi(p^k) = p^k - p^(k-1) | Build from factorization |
| Multiplicative | phi(a*b) = phi(a)*phi(b) if gcd(a,b)=1 | Combine factors |
| Divisor sum | sum of phi(d) for d\|n = n | Gauss's identity |
| **Euler's theorem** | a^phi(m) = 1 (mod m) if gcd(a,m)=1 | Modular exponent reduction |
| Power reduction | a^n mod m = a^(n mod phi(m)) mod m | Large exponent problems |

### Template: Single Value

```python
def euler_phi(n: int) -> int:
    """Compute phi(n) in O(sqrt(n)) via trial division."""
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
```

### Template: Sieve for All Values

```python
def phi_sieve(n: int) -> list[int]:
    """Compute phi(i) for all i in [0, n] in O(n log log n)."""
    phi = list(range(n + 1))
    for i in range(2, n + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, n + 1, i):
                phi[j] -= phi[j] // i
    return phi
```

*Socratic prompt: "Euler's theorem says a^phi(m) = 1 (mod m). Fermat's little theorem says a^(p-1) = 1 (mod p). How is Fermat's theorem a special case of Euler's? What is phi(p) when p is prime?"*

### Problems

| Problem | Key Twist |
|---------|-----------|
| Super Pow (372) | Use Euler's theorem to reduce exponent mod phi(1337) |

---

## 7. Divisor Functions

### Core Insight

For n = p1^e1 * p2^e2 * ... * pk^ek:

| Function | Formula | Meaning |
|----------|---------|---------|
| d(n) — count | (e1+1)(e2+1)...(ek+1) | Number of divisors |
| sigma(n) — sum | product of (pi^(ei+1) - 1)/(pi - 1) | Sum of divisors |

Both are **multiplicative functions**: f(a*b) = f(a)*f(b) when gcd(a,b) = 1.

### Template

```python
def count_divisors(n: int) -> int:
    """Count divisors of n in O(sqrt(n))."""
    count = 1
    d = 2
    while d * d <= n:
        if n % d == 0:
            e = 0
            while n % d == 0:
                e += 1
                n //= d
            count *= (e + 1)
        d += 1
    if n > 1:
        count *= 2  # remaining prime with exponent 1
    return count

def sum_divisors(n: int) -> int:
    """Sum of divisors of n in O(sqrt(n))."""
    total = 1
    d = 2
    while d * d <= n:
        if n % d == 0:
            e = 0
            while n % d == 0:
                e += 1
                n //= d
            total *= (d ** (e + 1) - 1) // (d - 1)
        d += 1
    if n > 1:
        total *= (1 + n)
    return total
```

**Why the formulas work:** Each divisor picks an exponent for each prime factor (0 to ei). The count is the product of choices. The sum uses the geometric series: 1 + p + p^2 + ... + p^e = (p^(e+1) - 1)/(p - 1).

*Socratic prompt: "How many divisors does 12 = 2^2 * 3^1 have? List them. Now verify: (2+1)*(1+1) = 6. Can you see why each divisor corresponds to a choice of exponents?"*

### Problems

| Problem | Key Twist |
|---------|-----------|
| Perfect Number (507) | Check if sigma(n) = 2n (only need sum_divisors) |
| Ugly Number II (264) | Related: numbers with only 2, 3, 5 as factors (see `classic-interview-problems.md`) |

---

## Interview Tips

- **Know trial division cold.** It's the default factorization method in interviews. Everything else (linear sieve, Pollard's rho) is bonus.
- **Extended Euclidean is underrated.** It's the universal tool for modular inverse, especially when the modulus isn't prime.
- **Matrix exponentiation generalizes.** Once you know the Fibonacci matrix trick, you can solve any linear recurrence in O(k^3 log n). Interviewers love this.
- **Euler's theorem reduces exponents.** For problems like "compute a^(b^c) mod m," reduce the exponent using phi(m) before computing.
- **Mention complexity.** "I can test primality in O(sqrt(n)) with trial division, or O(log^2 n) with Miller-Rabin" shows depth.

---

## Practice Questions

### Essential

| Problem | Key Concept |
|---------|-------------|
| Climbing Stairs (70) | Fibonacci in disguise; mention O(log n) matrix method as follow-up |
| Count Primes (204) | Sieve (see `math-techniques.md`); mention linear sieve as optimization |

### Recommended

| Problem | Key Concept |
|---------|-------------|
| Super Pow (372) | Euler's theorem for exponent reduction |
| Fibonacci Number (509) | Direct computation; discuss O(log n) methods |
| Perfect Number (507) | Sum of divisors formula |

---

## Attribution

Content synthesized from cp-algorithms.com articles on Extended Euclidean Algorithm, Fibonacci Numbers, Linear Sieve, Primality Tests, Integer Factorization, Euler's Totient Function, and Divisors. Reorganized and augmented with Socratic teaching prompts, interview-focused filtering, Python templates, and cross-references for the leetcode-teacher skill.
