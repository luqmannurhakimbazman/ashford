# Modular Arithmetic (Advanced)

Advanced modular arithmetic techniques for coding interviews: modular inverse via Extended Euclidean, linear congruences, Chinese Remainder Theorem, factorial mod p, and continued fractions. For basic modular arithmetic (mod properties, Fermat-based inverse, fast exponentiation), see `math-techniques.md` Section 1. For Extended Euclidean Algorithm and Euler's totient, see `number-theory-advanced.md`.

---

## Quick Reference Table

| Pattern | Recognition Signals | Key Problems | Section |
|---------|-------------------|--------------|---------|
| Modular Inverse (General) | "Inverse mod non-prime", "gcd(a,m) = 1 but m not prime" | -- (building block) | 1 |
| Linear Congruence | "Solve a*x = b (mod n)", "modular equation" | -- (building block) | 2 |
| Chinese Remainder Theorem | "System of mod equations", "remainders with coprime moduli" | -- (building block) | 3 |
| Factorial mod p | "n! mod prime", "binomial coefficient mod p", "nCr mod p" | -- (combinatorics) | 4 |
| Continued Fractions | "Best rational approximation", "Stern-Brocot tree" | -- (rare but elegant) | 5 |

---

## Corner Cases

- **gcd(a, m) != 1:** Modular inverse does not exist. Linear congruence may still have solutions if gcd divides the RHS.
- **m = 1:** Everything is 0 mod 1. Handle as edge case.
- **Large factorials mod small p:** Factorial contains factors of p, requiring the "modified factorial" approach (Section 4).
- **Non-coprime moduli in CRT:** Solution exists only if congruences are consistent. Check before solving.

---

## 1. Modular Inverse via Extended Euclidean

### Core Insight

`math-techniques.md` Section 1 shows the Fermat-based inverse: `a^(-1) = a^(m-2) mod m` — but this **only works when m is prime**. For arbitrary modulus m where gcd(a, m) = 1, use the Extended Euclidean Algorithm instead.

**When to use which:**

| Method | Requirement | Time |
|--------|------------|------|
| Fermat's little theorem | m must be prime | O(log m) |
| Extended Euclidean | gcd(a, m) = 1 (m can be anything) | O(log m) |
| Euler's theorem | gcd(a, m) = 1, know phi(m) | O(log phi(m)) |

### Template

```python
def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return (gcd, x, y) such that a*x + b*y = gcd(a, b)."""
    if b == 0:
        return a, 1, 0
    g, x1, y1 = extended_gcd(b, a % b)
    return g, y1, x1 - (a // b) * y1

def mod_inverse(a: int, m: int) -> int | None:
    """Modular inverse of a mod m. Returns None if it doesn't exist."""
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        return None
    return x % m
```

**Python 3.8+ shortcut:** `pow(a, -1, m)` computes the modular inverse directly and raises `ValueError` if it doesn't exist.

*Socratic prompt: "You know pow(a, m-2, m) gives the inverse when m is prime. But what if m = 12 and a = 5? Fermat doesn't apply since 12 isn't prime. How does Extended Euclidean solve 5x = 1 (mod 12)?"*

---

## 2. Linear Congruence Equations

### Core Insight

Solve **a*x = b (mod n)** for x. This generalizes modular inverse (which is the special case b = 1).

**Solution existence:** Let g = gcd(a, n). A solution exists **if and only if** g divides b. When solutions exist, there are exactly g distinct solutions mod n.

### Template

```python
from math import gcd

def solve_linear_congruence(a: int, b: int, n: int) -> list[int]:
    """Solve a*x = b (mod n). Return all solutions in [0, n)."""
    g = gcd(a, n)
    if b % g != 0:
        return []  # no solution
    # Reduce to coprime case
    a2, b2, n2 = a // g, b // g, n // g
    # Now gcd(a2, n2) = 1, so inverse exists
    inv = pow(a2, -1, n2)
    x0 = (b2 * inv) % n2
    # All g solutions
    return [(x0 + i * n2) % n for i in range(g)]
```

**How it works:** Divide the equation by g to get a coprime case, solve that, then generate all g solutions by adding multiples of n/g.

*Socratic prompt: "Consider 6x = 4 (mod 10). gcd(6, 10) = 2, and 2 divides 4, so solutions exist. Dividing by 2: 3x = 2 (mod 5). The inverse of 3 mod 5 is 2 (since 3*2 = 6 = 1 mod 5), so x0 = 4. The two solutions mod 10 are 4 and 9. Verify: 6*4 = 24 = 4 mod 10, 6*9 = 54 = 4 mod 10."*

---

## 3. Chinese Remainder Theorem (CRT)

### Core Insight

Given a system of congruences with **pairwise coprime** moduli:
```
x = a1 (mod m1)
x = a2 (mod m2)
...
x = ak (mod mk)
```
CRT guarantees a **unique solution** modulo M = m1 * m2 * ... * mk.

### Template: Two-Moduli (Most Common in Interviews)

```python
def crt_two(a1: int, m1: int, a2: int, m2: int) -> tuple[int, int]:
    """Solve x = a1 (mod m1), x = a2 (mod m2). Return (x, m1*m2)."""
    g, p, q = extended_gcd(m1, m2)
    if (a2 - a1) % g != 0:
        return None  # no solution
    lcm = m1 // g * m2
    x = (a1 + m1 * ((a2 - a1) // g * p % (m2 // g))) % lcm
    return x, lcm
```

### Template: General CRT

```python
def crt(congruences: list[tuple[int, int]]) -> tuple[int, int]:
    """Solve system of x = a_i (mod m_i). Return (solution, product of moduli).
    Moduli must be pairwise coprime."""
    M = 1
    for _, m in congruences:
        M *= m
    x = 0
    for a_i, m_i in congruences:
        M_i = M // m_i
        N_i = pow(M_i, -1, m_i)
        x = (x + a_i * M_i * N_i) % M
    return x, M
```

**Garner's Algorithm** (alternative reconstruction): Converts CRT residues into mixed-radix representation. Useful when the product M would overflow, as all intermediate arithmetic stays within individual moduli. See cp-algorithms.com for details — rarely needed in interviews.

*Socratic prompt: "A number leaves remainder 2 when divided by 3, and remainder 3 when divided by 5. What is it mod 15? How does CRT systematically find this?"*

### Problems

| Problem | Key Twist |
|---------|-----------|
| -- (rare in LC) | CRT appears in scheduling/period problems and multi-modular hashing |

---

## 4. Factorial Modulo p

### Core Insight

Computing n! mod p directly fails for large n because n! contains factors of p (making it 0 mod p). The solution: compute the **modified factorial** n!_p — that is, n! with all factors of p removed, taken mod p.

**Two key tools:**

| Tool | Formula | Purpose |
|------|---------|---------|
| Wilson's theorem | (p-1)! = -1 (mod p) for prime p | Each complete block of p contributes -1 |
| Legendre's formula | v_p(n!) = sum of floor(n/p^i) for i >= 1 | Count how many times p divides n! |

### Template

```python
def factorial_mod(n: int, p: int) -> int:
    """Compute n! mod p with all factors of p removed."""
    # Precompute factorials 0! to (p-1)! mod p
    f = [1] * p
    for i in range(1, p):
        f[i] = f[i - 1] * i % p

    result = 1
    while n > 1:
        if (n // p) % 2 == 1:
            result = p - result  # multiply by -1 (Wilson's theorem)
        result = result * f[n % p] % p
        n //= p
    return result

def p_adic_valuation(n: int, p: int) -> int:
    """Count how many times p divides n! (Legendre's formula)."""
    count = 0
    pk = p
    while pk <= n:
        count += n // pk
        pk *= p
    return count
```

**Application — nCr mod p:**

```python
def ncr_mod_p(n: int, r: int, p: int) -> int:
    """Compute C(n, r) mod p for prime p."""
    if r > n or r < 0:
        return 0
    # Check if p divides C(n, r)
    vp = p_adic_valuation(n, p) - p_adic_valuation(r, p) - p_adic_valuation(n - r, p)
    if vp > 0:
        return 0
    num = factorial_mod(n, p)
    den = factorial_mod(r, p) * factorial_mod(n - r, p) % p
    return num * pow(den, -1, p) % p
```

*Socratic prompt: "10! = 3628800. What is 10! mod 7? Naively, 3628800 mod 7 = 0 because 7 divides 10!. But what if we remove all factors of 7 first? How many factors of 7 does 10! have?"*

---

## 5. Continued Fractions

### Core Insight

A continued fraction represents a rational number p/q as `[a0; a1, a2, ..., ak]` where the coefficients come from the Euclidean algorithm. The **convergents** p_k/q_k are the best rational approximations with denominators up to q_k.

**Interview relevance:** Rare as a standalone topic, but the Stern-Brocot tree (which is the convergent structure) occasionally appears, and the best-approximation property is useful for scheduling and ratio problems.

### Template

```python
def to_continued_fraction(p: int, q: int) -> list[int]:
    """Convert p/q to continued fraction [a0; a1, a2, ...]."""
    cf = []
    while q:
        cf.append(p // q)
        p, q = q, p % q
    return cf

def convergents(cf: list[int]) -> list[tuple[int, int]]:
    """Return convergents (p_k, q_k) for a continued fraction."""
    p_prev, p_curr = 0, 1
    q_prev, q_curr = 1, 0
    result = []
    for a in cf:
        p_prev, p_curr = p_curr, a * p_curr + p_prev
        q_prev, q_curr = q_curr, a * q_curr + q_prev
        result.append((p_curr, q_curr))
    return result
```

### Key Properties

- **Length:** O(log(min(p, q))) — same as Euclidean algorithm.
- **Best approximation:** |p_k/q_k - r| <= 1/q_k^2. No fraction with smaller denominator is closer.
- **Convergent identity:** p_k * q_{k-1} - p_{k-1} * q_k = (-1)^(k-1).
- **Denominators grow fast:** q_k >= F(k) (Fibonacci numbers), so convergents approach the target exponentially.

*Socratic prompt: "The Euclidean algorithm on (355, 113) gives quotients [3; 7, 16, 11]. The convergents are 3/1, 22/7, 355/113. Notice 22/7 approximates pi to 2 decimal places and 355/113 to 6. Why are these the 'best' approximations?"*

---

## Interview Tips

- **Default to Fermat for prime modulus.** Most interview problems use mod 10^9+7 (prime), so `pow(a, mod-2, mod)` suffices. Mention Extended Euclidean as the general case.
- **CRT is rare in LC but common in math interviews.** If you see multiple coprime moduli, CRT is the signal.
- **Know Legendre's formula.** "How many times does p divide n!?" comes up in trailing zeros variants (see `math-techniques.md` Section 5).
- **Wilson's theorem is a useful fact.** (p-1)! = -1 (mod p) — it's the bridge between factorials and modular arithmetic.

---

## Practice Questions

### Essential

| Problem | Key Concept |
|---------|-------------|
| Pow(x, n) (50) | Modular exponentiation (see `math-techniques.md`) |
| Factorial Trailing Zeroes (172) | Legendre's formula variant (see `math-techniques.md`) |

### Recommended

| Problem | Key Concept |
|---------|-------------|
| Super Pow (372) | Euler's theorem + modular inverse |
| Unique Paths (62) | C(m+n-2, m-1) mod p for follow-up |

---

## Attribution

Content synthesized from cp-algorithms.com articles on Linear Congruence Equations, Chinese Remainder Theorem, Garner's Algorithm, Factorial Modulo, and Continued Fractions. Reorganized and augmented with Socratic teaching prompts, interview-focused filtering, Python templates, and cross-references for the leetcode-teacher skill.
