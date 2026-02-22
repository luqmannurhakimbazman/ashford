# Numerical Search Methods

Continuous search and optimization algorithms for root-finding, unimodal optimization, and global optimization. These methods operate on **continuous functions** (real-valued domains) rather than discrete arrays — for discrete binary search on sorted arrays, see `binary-search-framework.md`. For related math foundations (modular arithmetic, exponentiation), see `math-techniques.md`.

---

## Quick Reference Table

| Pattern | Recognition Signals | Key Problems | Section |
|---------|-------------------|--------------|---------|
| Continuous Binary Search (Bisection) | "Find x where f(x) = 0", "root of equation", "precision 1e-6", "IVT" | Kth Smallest in Sorted Matrix (378), Median of Two Sorted Arrays (4) | 1 |
| Ternary Search | "Find max/min of unimodal function", "peak of curve", "minimize cost function" | Peak Index in Mountain Array (852), Pour Water (755) | 2 |
| Newton's Method | "Compute sqrt(x)", "find root fast", "quadratic convergence" | Sqrt(x) (69), Valid Perfect Square (367) | 3 |
| Simulated Annealing | "Global optimum", "TSP", "NP-hard optimization", "approximate best" | Best Position for a Service Centre (1515) | 4 |

---

## Corner Cases

- **Floating-point epsilon:** Never check `f(x) == 0`. Use `abs(f(x)) < eps` or iterate a fixed number of times (e.g., 100 iterations gives ~2^{-100} precision, far beyond `1e-9`).
- **Convergence failure:** Newton's method can diverge if the initial guess is far from the root or the derivative is near zero. Always have a fallback (bisection).
- **Multiple roots:** Bisection only finds a root in an interval where `f` changes sign. If `f` touches zero without crossing (e.g., `x^2`), bisection won't detect it.
- **Derivative undefined:** Newton's method requires `f'(x) != 0`. At inflection points or discontinuities, the tangent line is useless.
- **Unimodality assumption:** Ternary search requires exactly one peak/valley. If the function is multimodal, ternary search finds a local extremum, not the global one.
- **Integer vs float domains:** Some problems (e.g., Sqrt(x)) need integer bisection with `lo <= hi` and `mid = lo + (hi - lo) // 2`. Don't mix float and integer bisection logic.

---

## 1. Continuous Binary Search (Bisection Method)

### Core Insight

The Intermediate Value Theorem (IVT) guarantees: if `f` is continuous on `[a, b]` and `f(a)` and `f(b)` have opposite signs, then there exists a root `c` in `(a, b)`. Bisection exploits this by halving the interval each step.

**Key difference from discrete binary search:** There's no array — you're searching over a continuous range `[lo, hi]` and evaluating a function `f(x)` at the midpoint.

### Template

```python
def bisect_root(f, lo, hi, eps=1e-9):
    """Find x in [lo, hi] where f(x) ≈ 0.

    Precondition: f(lo) and f(hi) have opposite signs (IVT).
    """
    for _ in range(100):  # 100 iterations ≈ 2^-100 precision
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid       # Root is in [lo, mid]
        else:
            lo = mid       # Root is in [mid, hi]
    return (lo + hi) / 2
```

**Why fixed iterations instead of `while hi - lo > eps`?**
- Avoids floating-point subtraction issues near convergence
- 100 iterations is always enough (2^{-100} < 1e-30)
- Simpler to reason about termination

### Integer Bisection Variant

For problems like Sqrt(x) where the answer is an integer:

```python
def integer_sqrt(x):
    """Find largest n where n * n <= x."""
    lo, hi = 0, x
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if mid * mid <= x:
            lo = mid + 1   # mid might be the answer, but try larger
        else:
            hi = mid - 1   # mid^2 too big
    return hi              # hi is the last valid value
```

### "Bisect on Answer" Pattern

Many problems don't look like root-finding but become bisection when you reframe them:

> **Key question:** "Can we achieve answer = X?" If this is monotonic (if X works, all larger/smaller X also work), bisect on X.

```python
def bisect_on_answer(lo, hi, is_feasible):
    """Find the minimum X such that is_feasible(X) is True.

    Precondition: is_feasible is monotonically non-decreasing
    (False, False, ..., True, True, True).
    """
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if is_feasible(mid):
            hi = mid       # mid works, try smaller
        else:
            lo = mid + 1   # mid doesn't work, need larger
    return lo
```

**Classic applications:**
- Koko Eating Bananas (875): bisect on eating speed
- Split Array Largest Sum (410): bisect on max subarray sum
- Capacity to Ship Packages (1011): bisect on ship capacity

### Complexity

| Aspect | Value |
|--------|-------|
| Time | O(log((R - L) / eps)) per search, or O(100) with fixed iterations |
| Space | O(1) |
| Convergence | Linear — gains 1 bit of precision per iteration |

*Socratic prompt: "You're given a function f(x) that's continuous on [0, 100] with f(0) = -5 and f(100) = 12. How many bisection steps to find a root within 1e-9 accuracy? Why?"*

---

## 2. Ternary Search (Unimodal Optimization)

### Core Insight

For a **unimodal function** (one peak or one valley), you can eliminate 1/3 of the search space each step by evaluating at two interior points.

**Unimodal function:** `f` increases then decreases (single peak), or decreases then increases (single valley). Formally: there exists a point `x*` such that `f` is strictly increasing on `(-inf, x*)` and strictly decreasing on `(x*, +inf)` (for a peak).

### Template

```python
def ternary_search_max(f, lo, hi, eps=1e-9):
    """Find x in [lo, hi] that maximizes unimodal function f."""
    for _ in range(200):  # Extra iterations since we lose 1/3 not 1/2
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if f(m1) < f(m2):
            lo = m1        # Peak is in [m1, hi]
        else:
            hi = m2        # Peak is in [lo, m2]
    return (lo + hi) / 2
```

For **minimization**, flip the comparison: `if f(m1) > f(m2): lo = m1 else: hi = m2`.

### Integer Ternary Search

For discrete unimodal functions (e.g., Peak Index in Mountain Array):

```python
def ternary_search_peak(arr):
    """Find peak index in a mountain array."""
    lo, hi = 0, len(arr) - 1
    while hi - lo > 2:
        m1 = lo + (hi - lo) // 3
        m2 = hi - (hi - lo) // 3
        if arr[m1] < arr[m2]:
            lo = m1 + 1
        else:
            hi = m2 - 1
    # Check remaining 1-3 elements
    best = lo
    for i in range(lo, hi + 1):
        if arr[i] > arr[best]:
            best = i
    return best
```

> **Note:** For Peak Index in Mountain Array, regular binary search on the derivative (compare `arr[mid]` vs `arr[mid+1]`) is simpler and preferred. Ternary search shines when you can only evaluate `f(x)`, not its derivative.

### Golden Section Search (Optimization)

Ternary search wastes work: `f(m1)` and `f(m2)` are both computed but only one is reused. Golden section search picks points so that one evaluation carries over:

```python
PHI = (1 + 5**0.5) / 2  # Golden ratio ≈ 1.618
RESPHI = 2 - PHI         # ≈ 0.382

def golden_section_max(f, lo, hi, eps=1e-9):
    """Golden section search — fewer function evaluations than ternary."""
    m1 = lo + RESPHI * (hi - lo)
    m2 = hi - RESPHI * (hi - lo)
    f1, f2 = f(m1), f(m2)
    for _ in range(200):
        if f1 < f2:
            lo = m1
            m1, f1 = m2, f2
            m2 = hi - RESPHI * (hi - lo)
            f2 = f(m2)
        else:
            hi = m2
            m2, f2 = m1, f1
            m1 = lo + RESPHI * (hi - lo)
            f1 = f(m1)
    return (lo + hi) / 2
```

**Advantage:** One function evaluation per iteration (vs two for ternary search). Useful when `f` is expensive to compute.

### Complexity

| Aspect | Value |
|--------|-------|
| Ternary Search Time | O(log_{3/2}((R - L) / eps)) ≈ O(log((R - L) / eps) / log(1.5)) |
| Golden Section Time | Same convergence rate, but ~half the function evaluations |
| Space | O(1) |

*Socratic prompt: "Why can't you use ternary search on a function with two peaks? What would go wrong at the comparison step?"*

---

## 3. Newton's Method (Root-Finding)

### Core Insight

Instead of blindly halving the interval (bisection), use the **tangent line** at the current guess to leap toward the root. If `f` is smooth, this converges **quadratically** — the number of correct digits roughly doubles each iteration.

**Iteration formula:** `x_{n+1} = x_n - f(x_n) / f'(x_n)`

Geometrically: draw the tangent to `f` at `x_n`, and the new guess is where that tangent crosses the x-axis.

### Template

```python
def newton_root(f, df, x0, eps=1e-9, max_iter=100):
    """Find root of f using Newton's method.

    Args:
        f: The function.
        df: The derivative of f.
        x0: Initial guess.

    Returns:
        Approximate root x where f(x) ≈ 0.
    """
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < eps:
            return x
        dfx = df(x)
        if abs(dfx) < 1e-15:  # Derivative too small — method unstable
            break
        x = x - fx / dfx
    return x
```

### Square Root via Newton's Method

The classic application: compute `sqrt(a)` by finding the root of `f(x) = x^2 - a`.

```python
def sqrt_newton(a, eps=1e-9):
    """Compute sqrt(a) using Newton's method.

    f(x) = x^2 - a,  f'(x) = 2x
    Iteration: x = x - (x^2 - a) / (2x) = (x + a/x) / 2
    """
    if a < 0:
        raise ValueError("Cannot compute sqrt of negative number")
    if a == 0:
        return 0
    x = a  # Initial guess
    while True:
        nx = (x + a / x) / 2
        if abs(nx - x) < eps:
            return nx
        x = nx
```

### Integer Square Root

For Sqrt(x) (LeetCode 69) — find largest `n` where `n * n <= x`:

```python
def isqrt_newton(x):
    """Integer square root using Newton's method."""
    if x < 2:
        return x
    r = x
    while r * r > x:
        r = (r + x // r) // 2
    return r
```

**Why this terminates:** The sequence `r` is monotonically decreasing (after at most one step) and bounded below by `floor(sqrt(x))`. Uses integer division to stay in integers.

### Convergence Analysis

| Property | Value |
|----------|-------|
| Convergence order | Quadratic (digits of accuracy double each step) |
| Typical iterations | 5-10 for double precision (~15 decimal digits) |
| Failure modes | `f'(x) ≈ 0` (flat region), cycling, divergence from bad initial guess |

**When Newton's beats bisection:** When `f` is smooth and you have a reasonable initial guess. Newton's typically needs 5-10 iterations vs 50+ for bisection to reach the same precision.

**When to prefer bisection:** When you don't know `f'`, when `f` is non-smooth, or when you need guaranteed convergence (bisection always converges if IVT condition holds).

### Cube Root and Nth Root

Generalize: to find `a^(1/n)`, solve `f(x) = x^n - a`:

```python
def nth_root_newton(a, n, eps=1e-9):
    """Compute a^(1/n) using Newton's method.

    f(x) = x^n - a,  f'(x) = n * x^(n-1)
    Iteration: x = x - (x^n - a) / (n * x^(n-1))
             = ((n-1)*x + a / x^(n-1)) / n
    """
    x = a  # Initial guess (crude but works)
    for _ in range(200):
        xn1 = x ** (n - 1)
        nx = ((n - 1) * x + a / xn1) / n
        if abs(nx - x) < eps:
            return nx
        x = nx
    return x
```

*Socratic prompt: "Newton's method converges quadratically for sqrt. How many iterations to go from an initial error of 1.0 to an error below 1e-12? What assumption makes this estimate work?"*

---

## 4. Simulated Annealing (Global Optimization)

### Core Insight

For problems with many local optima (e.g., NP-hard combinatorial optimization), deterministic methods get stuck. Simulated annealing escapes local optima by **probabilistically accepting worse solutions**, with the probability decreasing over time as the "temperature" cools.

**Analogy:** Heating metal makes atoms move freely (explore), then slow cooling lets them settle into a low-energy (optimal) crystal structure.

### Template

```python
import random
import math

def simulated_annealing(initial_state, energy, neighbor,
                        t_start=1.0, t_end=1e-8, decay=0.9999):
    """Simulated annealing for minimization.

    Args:
        initial_state: Starting solution.
        energy: Function mapping state -> cost (lower is better).
        neighbor: Function mapping state -> random neighboring state.
        t_start: Initial temperature (high = more exploration).
        t_end: Final temperature (low = greedy).
        decay: Multiplicative cooling factor per step.

    Returns:
        Best state found.
    """
    state = initial_state
    best_state = state
    e = energy(state)
    best_e = e
    t = t_start

    while t > t_end:
        new_state = neighbor(state)
        new_e = energy(new_state)
        delta = new_e - e

        # Accept better solutions always;
        # accept worse solutions with probability exp(-delta / t)
        if delta < 0 or random.random() < math.exp(-delta / t):
            state = new_state
            e = new_e

        if e < best_e:
            best_e = e
            best_state = state

        t *= decay

    return best_state
```

### Probabilistic Acceptance Function (PAF)

The core mechanism: `P(accept) = exp(-delta_E / T)`

| Scenario | delta_E | Temperature | P(accept) |
|----------|---------|-------------|-----------|
| Better solution | < 0 | Any | 1.0 (always accept) |
| Slightly worse | Small + | High T | ~1.0 (likely accept) |
| Slightly worse | Small + | Low T | ~0 (likely reject) |
| Much worse | Large + | High T | < 1.0 (sometimes accept) |
| Much worse | Large + | Low T | ~0 (almost never accept) |

### Parameter Tuning Guide

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| `t_start` | Higher = more initial exploration | 1.0 to 1e6 (scale to problem) |
| `t_end` | Lower = more refined final search | 1e-6 to 1e-10 |
| `decay` | Closer to 1 = slower cooling, more thorough | 0.999 to 0.99999 |
| Iterations | `log(t_end / t_start) / log(decay)` | 10^4 to 10^7 |

**Rule of thumb:** Start with `decay = 0.9999` and `t_start = 1.0`. If solutions aren't good enough, increase iterations (slower decay) or widen the temperature range.

### Example: Best Position for a Service Centre (LC 1515)

Find a point that minimizes the sum of Euclidean distances to given points (geometric median):

```python
def get_min_dist_sum(positions):
    """LC 1515: Find point minimizing sum of distances to all positions."""
    def energy(point):
        x, y = point
        return sum(((x - px)**2 + (y - py)**2)**0.5
                   for px, py in positions)

    def neighbor(point, scale):
        x, y = point
        return (x + random.uniform(-scale, scale),
                y + random.uniform(-scale, scale))

    # Start at centroid
    n = len(positions)
    x = sum(p[0] for p in positions) / n
    y = sum(p[1] for p in positions) / n
    state = (x, y)
    best = state
    best_e = energy(state)
    t = 100.0  # Large initial temperature for coordinate space

    while t > 1e-6:
        new_state = neighbor(state, t)
        new_e = energy(new_state)
        delta = new_e - best_e

        if delta < 0 or random.random() < math.exp(-delta / t):
            state = new_state
            if new_e < best_e:
                best_e = new_e
                best = new_state

        t *= 0.999

    return best_e
```

### When to Use Simulated Annealing

**Good fit:**
- NP-hard optimization (TSP, graph coloring, scheduling)
- Continuous optimization with many local minima
- Contest problems with "find approximate best" and loose precision
- When exact algorithms are too slow and you need a heuristic

**Bad fit:**
- Problems with known polynomial-time exact solutions
- Problems requiring exact answers (SA is probabilistic)
- When the solution space is too large to explore meaningfully

### Complexity

| Aspect | Value |
|--------|-------|
| Time | O(iterations * cost_of_energy_and_neighbor) |
| Space | O(state_size) |
| Guarantee | None — probabilistic, no worst-case bound |

*Socratic prompt: "If you lower the decay rate from 0.9999 to 0.99, what happens to the search? Why might a very fast cooling schedule miss the global optimum?"*

---

## Method Comparison

| Method | Type | Convergence | Requires | Guaranteed | Best For |
|--------|------|-------------|----------|------------|----------|
| Bisection | Root-finding | Linear (1 bit/iter) | Sign change (IVT) | Yes (if IVT holds) | Reliable root-finding |
| Newton's | Root-finding | Quadratic | f'(x), good initial guess | No (can diverge) | Fast root-finding, sqrt |
| Ternary Search | Optimization | Linear (~log 1.5) | Unimodality | Yes (if unimodal) | Single-peak/valley optimization |
| Simulated Annealing | Optimization | N/A (stochastic) | Energy + neighbor functions | No (probabilistic) | NP-hard, multimodal optimization |

**Decision flowchart:**
1. Need a **root** of f(x) = 0?
   - Know f'? Smooth function? Good initial guess? -> **Newton's method**
   - Otherwise -> **Bisection** (always works with IVT)
2. Need **min/max** of f(x)?
   - Unimodal? -> **Ternary search** (or golden section)
   - Multimodal / NP-hard? -> **Simulated annealing**
3. Discrete "bisect on answer" pattern? -> See `binary-search-framework.md`

---

## Practice Questions

### Essential

| # | Problem | Difficulty | Key Technique |
|---|---------|-----------|---------------|
| 69 | Sqrt(x) | Easy | Integer Newton's / integer bisection |
| 367 | Valid Perfect Square | Easy | Bisection or Newton's on integers |
| 852 | Peak Index in Mountain Array | Medium | Ternary search (or binary search on derivative) |
| 378 | Kth Smallest Element in Sorted Matrix | Medium | Bisect on answer |
| 875 | Koko Eating Bananas | Medium | Bisect on answer |

### Recommended

| # | Problem | Difficulty | Key Technique |
|---|---------|-----------|---------------|
| 1515 | Best Position for a Service Centre | Hard | Simulated annealing / geometric median |
| 410 | Split Array Largest Sum | Hard | Bisect on answer |
| 1011 | Capacity to Ship Packages | Medium | Bisect on answer |
| 4 | Median of Two Sorted Arrays | Hard | Bisection on partition |
| 162 | Find Peak Element | Medium | Binary/ternary search on unimodal |

---

*Source: Algorithms adapted from [cp-algorithms.com](https://cp-algorithms.com/) (binary search, ternary search, Newton's method, simulated annealing articles). Code translated from C++ to Python.*
