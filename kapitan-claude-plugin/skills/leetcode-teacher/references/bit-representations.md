# Bit Representations

Advanced bit-level representations and enumeration techniques: Gray code, submask enumeration, and arbitrary-precision arithmetic. For core bit manipulation tricks (n & (n-1), XOR properties, shift operations), see `bit-manipulation.md`. For bitmask DP, see `brute-force-search.md`.

---

## Quick Reference Table

| Pattern | Recognition Signals | Key Problems | Section |
|---------|-------------------|--------------|---------|
| Gray Code | "Adjacent values differ by 1 bit", "Hamiltonian cycle on hypercube" | Gray Code (89), Circular Permutation (1238) | 1 |
| Submask Enumeration | "Iterate all subsets of a bitmask", "bitmask DP transitions" | -- (DP building block) | 2 |
| Arbitrary Precision | "Multiply strings", "add strings", "big number arithmetic" | Multiply Strings (43), Add Strings (415) | 3 |
| Advanced Bit Tricks | "Count total set bits 1..n", "bit-level counting" | Counting Bits (338), Total Hamming Distance (477) | 4 |

---

## Corner Cases

- **Gray code with n = 0:** Only one code: `[0]`.
- **Submask of 0:** The only submask of 0 is 0 itself. The enumeration loop must handle this to avoid infinite looping.
- **Leading zeros:** When converting between Gray code and binary, ensure you handle the correct bit width.

---

## 1. Gray Code

### Core Insight

In Gray code, consecutive values differ in **exactly one bit**. The conversion is remarkably simple:

| Direction | Formula | Time |
|-----------|---------|------|
| Binary to Gray | `G(n) = n ^ (n >> 1)` | O(1) |
| Gray to Binary | XOR-fold from MSB to LSB | O(log n) |

### Template

```python
def to_gray(n: int) -> int:
    """Convert binary number to Gray code."""
    return n ^ (n >> 1)

def from_gray(g: int) -> int:
    """Convert Gray code back to binary."""
    n = 0
    while g:
        n ^= g
        g >>= 1
    return n

def gray_code_sequence(num_bits: int) -> list[int]:
    """Generate the full n-bit Gray code sequence."""
    return [i ^ (i >> 1) for i in range(1 << num_bits)]
```

### Why It Works

The XOR with right-shift flips exactly the bits where the binary representation changes from 0 to 1 (reading left to right). This ensures adjacent codes differ by one bit.

**Connection to Towers of Hanoi:** The i-th move in the optimal Towers of Hanoi solution moves the same disk as the bit position that changes between Gray(i-1) and Gray(i). Specifically, the changed bit position tells you which disk to move.

*Socratic prompt: "Write out the 3-bit Gray code sequence: 000, 001, 011, 010, 110, 111, 101, 100. Verify that each consecutive pair differs by exactly one bit. Now notice: this sequence forms a cycle (100 -> 000 is also one bit). Why does n ^ (n >> 1) produce this?"*

### Problems

| Problem | Key Twist |
|---------|-----------|
| Gray Code (89) | Generate n-bit Gray code sequence starting from 0 |
| Circular Permutation in Binary (1238) | Gray code starting from a given value (rotate the sequence) |

---

## 2. Submask Enumeration

### Core Insight

Given a bitmask `m`, enumerate all submasks `s` (where only bits set in `m` can be set in `s`). The trick:

```
next_submask = (current_submask - 1) & mask
```

This works because subtracting 1 clears the lowest set bit and sets all bits below it, then the AND with the mask removes bits not in the original mask.

### Template

```python
def enumerate_submasks(mask: int):
    """Yield all non-zero submasks of mask in descending order."""
    s = mask
    while s > 0:
        yield s
        s = (s - 1) & mask

def enumerate_submasks_with_zero(mask: int):
    """Yield all submasks of mask, including zero."""
    s = mask
    while True:
        yield s
        if s == 0:
            break
        s = (s - 1) & mask
```

### Complexity: Why O(3^n)?

When iterating all masks (0 to 2^n - 1) and their submasks, the total work is **O(3^n)**, not O(4^n).

**Proof:** Each of the n bits has exactly 3 states:
1. Not in mask, not in submask
2. In mask, not in submask
3. In mask and in submask

By the product rule: 3^n total (mask, submask) pairs.

**Equivalently:** By the binomial theorem, sum over k of C(n,k) * 2^k = (1+2)^n = 3^n.

### Application: Bitmask DP Pattern

```python
def bitmask_dp_with_submasks(n: int) -> list[int]:
    """Template for DP where transitions iterate over submasks."""
    dp = [0] * (1 << n)
    # Base case
    dp[0] = 1  # or whatever the base is

    for mask in range(1, 1 << n):
        # Iterate all non-empty submasks of mask
        sub = mask
        while sub > 0:
            # dp[mask] = combine(dp[mask], dp[mask ^ sub], sub)
            sub = (sub - 1) & mask
    return dp
```

*Socratic prompt: "If you have a mask 1010 (bits 1 and 3 set), what are all its submasks? Trace through the (s-1) & mask trick starting from s = 1010. How many submasks does a mask with k set bits have?"*

---

## 3. Arbitrary-Precision Arithmetic

### Core Insight

**In Python, integers have arbitrary precision natively** — so big-number arithmetic is built in. However, interview problems like "Multiply Strings" (LC 43) and "Add Strings" (LC 415) ask you to implement the underlying algorithms.

The standard approach: store digits in an array (least-significant first) and implement school-method arithmetic.

### Template: String Multiplication (LC 43)

```python
def multiply_strings(num1: str, num2: str) -> str:
    """Multiply two non-negative integers represented as strings."""
    n, m = len(num1), len(num2)
    result = [0] * (n + m)

    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            product = int(num1[i]) * int(num2[j])
            p1, p2 = i + j, i + j + 1
            total = product + result[p2]
            result[p2] = total % 10
            result[p1] += total // 10

    # Remove leading zeros
    result_str = ''.join(map(str, result)).lstrip('0')
    return result_str or '0'
```

### Template: String Addition (LC 415)

```python
def add_strings(num1: str, num2: str) -> str:
    """Add two non-negative integers represented as strings."""
    i, j = len(num1) - 1, len(num2) - 1
    carry = 0
    result = []
    while i >= 0 or j >= 0 or carry:
        x = int(num1[i]) if i >= 0 else 0
        y = int(num2[j]) if j >= 0 else 0
        total = x + y + carry
        result.append(str(total % 10))
        carry = total // 10
        i -= 1
        j -= 1
    return ''.join(reversed(result))
```

*Socratic prompt: "When you multiply 123 * 456 by hand, you compute partial products and add them with carries. The string multiplication algorithm does exactly this. Where does the carry propagation happen in the code?"*

### Problems

| Problem | Key Twist |
|---------|-----------|
| Multiply Strings (43) | School multiplication without converting to int |
| Add Strings (415) | School addition digit by digit |
| Add Binary (67) | Same pattern but base 2 |
| Plus One (66) | Simplified addition (add 1 to digit array) |

---

## 4. Advanced Bit Counting

### Core Insight

Beyond single-number popcount (covered in `bit-manipulation.md` Section 1), some problems require counting set bits across a **range** of numbers.

**Count total set bits in all integers from 1 to n:** Each bit position contributes independently. For bit position k, the number of integers in [1, n] with bit k set follows a pattern based on complete groups of 2^(k+1).

### Template

```python
def count_total_set_bits(n: int) -> int:
    """Count total set bits in all integers from 1 to n."""
    if n <= 0:
        return 0
    total = 0
    bit = 1
    while bit <= n:
        # For this bit position:
        # Complete groups contribute (n + 1) // (bit << 1) * bit
        # Partial group contributes max(0, (n + 1) % (bit << 1) - bit)
        total_pairs = (n + 1) // (bit << 1)
        total += total_pairs * bit
        remainder = (n + 1) % (bit << 1)
        total += max(0, remainder - bit)
        bit <<= 1
    return total
```

**Complexity:** O(log n) — one pass per bit position.

### Application: Total Hamming Distance (LC 477)

```python
def total_hamming_distance(nums: list[int]) -> int:
    """Sum of Hamming distances between all pairs."""
    total = 0
    for bit in range(32):
        ones = sum(1 for n in nums if n & (1 << bit))
        zeros = len(nums) - ones
        total += ones * zeros  # each (0,1) pair contributes 1
    return total
```

**Key insight:** For each bit position, count how many numbers have 0 vs 1. Each (0,1) pair contributes 1 to the total distance.

*Socratic prompt: "For Total Hamming Distance, the brute force compares all O(n^2) pairs. The bit-counting approach is O(32n). Why? Think about what happens when you fix one bit position and count contributions independently."*

### Problems

| Problem | Key Twist |
|---------|-----------|
| Counting Bits (338) | DP: `dp[i] = dp[i & (i-1)] + 1` (see `bit-manipulation.md`) |
| Total Hamming Distance (477) | Count 0s and 1s per bit position |
| Hamming Distance (461) | XOR + popcount |

---

## Interview Tips

- **Gray code is a one-liner.** `n ^ (n >> 1)` — know it cold. The inverse is the only part that needs a loop.
- **Submask enumeration is the key to bitmask DP.** If the DP transition involves "try all subsets of the remaining items," this is the pattern.
- **For string arithmetic, work from least significant digit.** This naturally handles carries. Reverse at the end if needed.
- **Bit counting per position is a powerful technique.** Whenever you see "sum over all pairs" with bit operations, think about independent bit positions.

---

## Practice Questions

### Essential

| Problem | Key Concept |
|---------|-------------|
| Gray Code (89) | `n ^ (n >> 1)` formula |
| Multiply Strings (43) | School multiplication on digit arrays |

### Recommended

| Problem | Key Concept |
|---------|-------------|
| Add Strings (415) | Digit-by-digit addition with carry |
| Total Hamming Distance (477) | Per-bit-position counting |
| Circular Permutation in Binary (1238) | Gray code with offset |
| Counting Bits (338) | DP with `n & (n-1)` (see `bit-manipulation.md`) |

---

## Attribution

Content synthesized from cp-algorithms.com articles on Gray Code, Enumerating All Submasks, Arbitrary-Precision Arithmetic, and Bit Manipulation. Reorganized and augmented with Socratic teaching prompts, interview-focused filtering, Python templates, and cross-references for the leetcode-teacher skill.
