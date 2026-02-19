# Numerical Integration

Methods for approximating definite integrals numerically. Used in competitive programming when an integral has no closed-form solution, or when a problem reduces to computing an area/probability over a continuous domain. For related numerical methods (root-finding, optimization), see `numerical-search.md`.

---

## Quick Reference Table

| Pattern | Recognition Signals | Key Problems | Section |
|---------|-------------------|--------------|---------|
| Trapezoidal Rule | "Approximate area under curve", "numerical integral", simple baseline | -- (building block) | 1 |
| Simpson's Rule | "High-accuracy integral", "parabolic approximation", "1e-6 precision" | -- (contest standard) | 2 |

---

## Corner Cases

- **Precision requirements:** Contest problems typically need 1e-6 accuracy. Simpson's rule with `n = 1000` subdivisions is usually sufficient. If not, double `n` and check if the answer stabilizes.
- **Singularities:** If `f(x)` blows up at an endpoint (e.g., `1/sqrt(x)` at x=0), shift the endpoint slightly: integrate `[eps, b]` instead of `[0, b]`.
- **Highly oscillating functions:** Both methods struggle with rapidly oscillating functions (e.g., `sin(1000x)`). Use more subdivisions or adaptive methods.
- **Negative function values:** Integration works fine with negative values — the integral computes signed area. If you need total area, integrate `abs(f(x))`.
- **Even number of intervals:** Simpson's rule requires an **even** number of subintervals (`n` must be even). Always enforce this.

---

## When to Use Numerical Integration

**Recognition signals in contest problems:**
- "Compute the probability that..." (integrate a PDF)
- "Find the area of a region bounded by..." (no closed-form antiderivative)
- "Expected value of a continuous random variable"
- Area under a curve defined by a black-box function
- Problems where the integrand involves `sqrt`, `exp`, `log` compositions with no elementary antiderivative

**General rule:** If you can compute `f(x)` for any `x` but can't find an antiderivative, use numerical integration.

---

## 1. Trapezoidal Rule

### Core Insight

Approximate the curve between consecutive points with a **straight line** (trapezoid). The area of each trapezoid is `h * (f(x_i) + f(x_{i+1})) / 2`, where `h` is the width of each subinterval.

The composite formula for `n` equal subintervals on `[a, b]`:

```
Integral ≈ h/2 * [f(x_0) + 2*f(x_1) + 2*f(x_2) + ... + 2*f(x_{n-1}) + f(x_n)]
```

where `h = (b - a) / n`.

### Template

```python
def trapezoidal(f, a, b, n=1000):
    """Approximate integral of f from a to b using the trapezoidal rule.

    Args:
        f: Function to integrate.
        a, b: Integration bounds.
        n: Number of subintervals (higher = more accurate).

    Returns:
        Approximate value of the definite integral.
    """
    h = (b - a) / n
    result = (f(a) + f(b)) / 2
    for i in range(1, n):
        result += f(a + i * h)
    return result * h
```

### Error Analysis

| Property | Value |
|----------|-------|
| Error order | O(h^2) = O((b-a)^2 / n^2) — "second-order" |
| Exact for | Linear functions (degree <= 1) |
| Function evals | n + 1 |

**Doubling `n` reduces the error by a factor of 4.** For 1e-6 accuracy on a well-behaved function over `[0, 1]`, `n ≈ 1000` is often sufficient.

*Socratic prompt: "Why is the trapezoidal rule exact for linear functions? Draw a picture — what does the trapezoid look like when the curve is a straight line?"*

---

## 2. Simpson's Rule

### Core Insight

Instead of fitting **lines** (trapezoidal), fit **parabolas** through every three consecutive points. A parabola captures curvature, so Simpson's rule is much more accurate for smooth functions.

The composite formula for `n` subintervals (n must be **even**) on `[a, b]`:

```
Integral ≈ h/3 * [f(x_0) + 4*f(x_1) + 2*f(x_2) + 4*f(x_3) + 2*f(x_4) + ... + 4*f(x_{n-1}) + f(x_n)]
```

The **1-4-2-4-2-...-4-1** weighting pattern comes from integrating the Lagrange interpolating polynomial over each pair of subintervals.

### Template

```python
def simpsons(f, a, b, n=1000):
    """Approximate integral of f from a to b using Simpson's rule.

    Args:
        f: Function to integrate.
        a, b: Integration bounds.
        n: Number of subintervals (MUST be even).

    Returns:
        Approximate value of the definite integral.
    """
    if n % 2 != 0:
        n += 1  # Simpson's requires even n
    h = (b - a) / n
    result = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        result += 4 * f(x) if i % 2 == 1 else 2 * f(x)
    return result * h / 3
```

### Adaptive Simpson's (Contest-Grade)

When you don't know how many subdivisions are needed, use **adaptive Simpson's**: recursively subdivide intervals where the error estimate is too large.

```python
def adaptive_simpsons(f, a, b, eps=1e-8):
    """Adaptive Simpson's rule — automatically refines where needed."""

    def _simpson(a, b):
        """Simpson's rule on [a, b] with 2 subintervals."""
        mid = (a + b) / 2
        return (b - a) / 6 * (f(a) + 4 * f(mid) + f(b))

    def _recursive(a, b, whole, eps, depth):
        mid = (a + b) / 2
        left = _simpson(a, mid)
        right = _simpson(mid, b)
        delta = left + right - whole
        if depth <= 0 or abs(delta) <= 15 * eps:
            return left + right + delta / 15  # Richardson extrapolation
        return (_recursive(a, mid, left, eps / 2, depth - 1) +
                _recursive(mid, b, right, eps / 2, depth - 1))

    whole = _simpson(a, b)
    return _recursive(a, b, whole, eps, depth=50)
```

**Why `delta / 15`?** This is Richardson extrapolation — it squeezes out extra accuracy by using the difference between the coarse and fine estimates. The factor 15 comes from Simpson's being a 4th-order method (2^4 - 1 = 15).

### Error Analysis

| Property | Value |
|----------|-------|
| Error order | O(h^4) = O((b-a)^4 / n^4) — "fourth-order" |
| Exact for | Polynomials of degree <= 3 (cubic and below) |
| Function evals | n + 1 |

**Doubling `n` reduces the error by a factor of 16.** This is why Simpson's is dramatically more accurate than trapezoidal for smooth functions.

*Socratic prompt: "Simpson's rule is exact for cubics even though it only fits parabolas. Why? Hint: think about what happens to the cubic term when you integrate symmetrically over an interval."*

---

## Method Comparison

| Property | Trapezoidal | Simpson's |
|----------|------------|-----------|
| Approximation | Linear (trapezoids) | Quadratic (parabolas) |
| Error order | O(h^2) | O(h^4) |
| Exact for | Degree <= 1 | Degree <= 3 |
| Weight pattern | 1-2-2-...-2-1 | 1-4-2-4-...-4-1 |
| Constraint | None | n must be even |
| Function evals | n + 1 | n + 1 |
| When to use | Quick estimate, non-smooth f | Default choice for smooth f |

**Bottom line:** Use Simpson's rule as your default. Use trapezoidal only when simplicity matters more than accuracy, or when `f` is non-smooth (Simpson's extra accuracy relies on smoothness).

---

## Practice Questions

### Essential

| # | Problem | Difficulty | Key Technique |
|---|---------|-----------|---------------|
| -- | Compute integral of `e^(-x^2)` on [0, 1] | -- | Simpson's (no closed form) |
| -- | Area under parametric curve | -- | Numerical integration + parametric |
| 1515 | Best Position for a Service Centre | Hard | Can also use gradient descent / SA (see `numerical-search.md`) |

### Recommended

| # | Problem | Difficulty | Key Technique |
|---|---------|-----------|---------------|
| -- | Expected value of continuous distribution | -- | Integrate x * pdf(x) |
| -- | Probability that random point in region | -- | 2D integration (nested Simpson's) |

> **Note:** Pure numerical integration problems are more common in competitive programming (Codeforces, AtCoder) than in LeetCode interviews. The technique is still valuable as a building block for probability and geometry problems.

---

*Source: Simpson's rule algorithm adapted from [cp-algorithms.com](https://cp-algorithms.com/num_methods/simpson-integration.html). Trapezoidal rule from standard numerical methods references. Code in Python.*
