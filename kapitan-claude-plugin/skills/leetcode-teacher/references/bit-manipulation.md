# Bit Manipulation

Common bit manipulation tricks and techniques for coding interviews. These problems often have elegant O(1) or O(n) solutions that replace brute-force approaches. For related math techniques, see `math-techniques.md`.

---

## Quick Reference Table

| Pattern | Recognition Signals | Key Problems | Section |
|---------|-------------------|--------------|---------|
| n & (n-1) | "Count set bits", "power of two", "Hamming weight" | Hamming Weight (191), Power of Two (231) | 1 |
| XOR Properties | "Single number", "missing number", "find duplicate" | Single Number (136), Missing Number (268) | 2 |
| Bit Masking | "Subsets via bitmask", "state compression" | Subsets (78), Counting Bits (338) | 3 |
| Shift Operations | "Multiply/divide by 2", "extract/set specific bit" | Reverse Bits (190), Number of 1 Bits (191) | 4 |
| Useful Tricks | "Swap without temp", "check sign", "absolute value" | -- (utility techniques) | 5 |

---

## Corner Cases

- **Negative numbers:** In languages with fixed-width integers, negative numbers use two's complement. Be aware of sign bit behavior (Python integers have arbitrary precision, so this matters less in Python but is critical in Java/C++).
- **Overflow / underflow:** Shifting left can overflow in fixed-width languages. `1 << 31` in a 32-bit signed integer is negative. Use unsigned types or language-specific guards when needed.

---

## 1. The n & (n-1) Trick

### Core Insight

`n & (n-1)` removes the **lowest set bit** of `n`. This single operation unlocks several classic problems.

**Why it works:** The lowest set bit of `n` is a 1 followed by all zeros. Subtracting 1 flips that bit to 0 and all lower bits to 1. ANDing with `n` clears the lowest set bit and leaves everything else unchanged.

```
n     = 1 0 1 0 0 0   (40)
n-1   = 1 0 0 1 1 1   (39)
n&n-1 = 1 0 0 0 0 0   (32)  -- lowest set bit removed
```

### Application: Hamming Weight (LC 191)

Count the number of 1-bits in an integer.

```python
def hamming_weight(n: int) -> int:
    """Count set bits by repeatedly removing the lowest one."""
    count = 0
    while n:
        n &= n - 1  # Remove lowest set bit
        count += 1
    return count
```

**Complexity:** O(k) where k is the number of set bits, not O(32). Each iteration removes exactly one bit.

### Application: Power of Two (LC 231)

A power of two has exactly one set bit: `n > 0 and n & (n-1) == 0`.

```python
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0
```

*Socratic prompt: "If a number is a power of two, what does its binary representation look like? How many set bits does it have?"*

### Problems

| Problem | Key Twist |
|---------|-----------|
| Number of 1 Bits (191) | Direct n & (n-1) loop |
| Power of Two (231) | Single set bit check |
| Counting Bits (338) | DP: `bits[n] = bits[n & (n-1)] + 1` |

---

## 2. XOR Properties

### Core Insight

XOR has three properties that make it invaluable:

| Property | Expression | Meaning |
|----------|-----------|---------|
| Self-inverse | `a ^ a = 0` | XOR-ing a number with itself cancels it |
| Identity | `a ^ 0 = a` | XOR with zero is a no-op |
| Commutative & Associative | `a ^ b ^ c = c ^ a ^ b` | Order doesn't matter |

These mean: XOR-ing an entire collection where every element appears twice except one will cancel all pairs, leaving only the unique element.

### Application: Single Number (LC 136)

Every element appears exactly twice except one. Find it.

```python
def single_number(nums: list[int]) -> int:
    """XOR all elements -- pairs cancel, singleton remains."""
    result = 0
    for num in nums:
        result ^= num
    return result
```

**Complexity:** O(n) time, O(1) space. No hash map needed.

*Socratic prompt: "If you XOR 5 ^ 3 ^ 5, what do you get? Why? What if you had [4, 1, 2, 1, 2]?"*

### Application: Missing Number (LC 268)

Array contains n distinct numbers from 0 to n. Find the missing one.

```python
def missing_number(nums: list[int]) -> int:
    """XOR all nums with all indices 0..n. Pairs cancel, missing remains."""
    n = len(nums)
    result = n  # Start with n (the upper bound index)
    for i, num in enumerate(nums):
        result ^= i ^ num
    return result
```

**Why it works:** XOR `{0, 1, ..., n}` with the array elements. Every number that appears in both cancels out. The one that only appears in `{0, ..., n}` remains.

*Socratic prompt: "Could you also solve Missing Number with the sum formula n*(n+1)/2? What are the trade-offs between XOR and sum? (Hint: think about integer overflow in other languages.)"*

### Problems

| Problem | Key Twist |
|---------|-----------|
| Single Number (136) | Direct XOR all |
| Missing Number (268) | XOR indices with values |
| Single Number II (137) | Count bits mod 3 (not pure XOR) |
| Single Number III (260) | XOR all, split by a distinguishing bit |

---

## 3. Bit Masking

### Core Insight

An integer can represent a **set of elements**. Bit `i` is 1 if element `i` is in the set. This enables:
- Iterating all subsets of n elements: loop `mask` from `0` to `2^n - 1`
- Set operations in O(1): union = OR, intersection = AND, difference = AND NOT

### Application: Enumerate Subsets

```python
def subsets_bitmask(nums: list[int]) -> list[list[int]]:
    """Generate all subsets using bitmask enumeration."""
    n = len(nums)
    result = []
    for mask in range(1 << n):
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(nums[i])
        result.append(subset)
    return result
```

> **Submask enumeration:** Need to iterate all submasks of a given mask (not all 2^n masks)? The `(s-1) & mask` trick does this in O(2^k) where k is the popcount of the mask. Total work across all masks is O(3^n). See `bit-representations.md` Section 2 for the template and complexity proof — this is the core pattern behind bitmask DP transitions.

### Application: Counting Bits (LC 338)

For every number from 0 to n, count the number of 1-bits.

```python
def count_bits(n: int) -> list[int]:
    """DP using n & (n-1) relation: bits[i] = bits[i & (i-1)] + 1."""
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = dp[i & (i - 1)] + 1
    return dp
```

*Socratic prompt: "If you know the bit count of n & (n-1), how does that relate to the bit count of n?"*

---

## 4. Shift Operations

### Core Insight

- `x << k` multiplies x by 2^k
- `x >> k` divides x by 2^k (floor division)
- `(x >> i) & 1` extracts the i-th bit
- `x | (1 << i)` sets the i-th bit
- `x & ~(1 << i)` clears the i-th bit

### Application: Reverse Bits (LC 190)

```python
def reverse_bits(n: int) -> int:
    """Reverse all 32 bits of an unsigned integer."""
    result = 0
    for i in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result
```

### Modulo with Powers of Two

`n % (2^k)` is equivalent to `n & ((1 << k) - 1)`. This is faster than the modulo operator.

```python
# These are equivalent for power-of-two modulus:
n % 8      # Slow modulo
n & 0b111  # Fast bit mask (8 = 2^3, mask = 0b111 = 7)
```

*Socratic prompt: "Why does n & 7 give the same result as n % 8? Think about what the last 3 bits represent."*

---

## 5. Useful Bit Tricks

### Quick Reference

| Trick | Code | Notes |
|-------|------|-------|
| Check if n is even | `n & 1 == 0` | Last bit is 0 for even |
| Check sign (negative) | `n >> 31` (for 32-bit) | Sign bit is MSB |
| Swap without temp | `a ^= b; b ^= a; a ^= b` | XOR swap (mainly academic) |
| Absolute value | `(n ^ (n >> 31)) - (n >> 31)` | Uses sign extension |
| Get lowest set bit | `n & (-n)` | Isolates the rightmost 1 |
| Turn off lowest set bit | `n & (n - 1)` | See Section 1 |
| Toggle the kth bit | `n ^ (1 << k)` | Flips bit k (0→1 or 1→0) |
| Multiply by 2^k | `n << k` | Left shift = multiply by power of 2 |
| Divide by 2^k | `n >> k` | Right shift = floor divide by power of 2 |
| Check if power of 2 | `n > 0 and n & (n-1) == 0` | Exactly one set bit |

### When to Use Bit Manipulation

**Good fit:**
- Problems explicitly about binary representation
- Need O(1) space for set operations on small domains (< 32 elements)
- State compression for DP (e.g., visited states in TSP)
- Single number / missing number problems

**Not a good fit:**
- When arithmetic is clearer (don't micro-optimize for cleverness)
- Floating point operations
- When the bit-manipulation version is harder to understand and correctness matters more than performance

*Socratic prompt: "In an interview, when would you choose bit manipulation over a hash set? What are the trade-offs?"*

---

## Practice Questions

### Essential

| Problem | Key Concept |
|---------|-------------|
| Number of 1 Bits (191) | n & (n-1) loop to count set bits |
| Sum of Two Integers (371) | Add without `+`: carry = `a & b`, sum = `a ^ b` |

### Recommended

| Problem | Key Concept |
|---------|-------------|
| Single Number (136) | XOR all elements — pairs cancel |
| Reverse Bits (190) | Build result bit by bit with shifts |
| Missing Number (268) | XOR indices with values |
| Counting Bits (338) | DP: `bits[n] = bits[n & (n-1)] + 1` |

---

## Attribution

Content synthesized from labuladong's algorithmic guide, Chapter 4 — "Other Common Techniques: Bit Manipulation," and the Tech Interview Handbook (techinterviewhandbook.org) bit manipulation cheatsheet. Reorganized and augmented with Socratic teaching prompts, cross-references, and code templates for the leetcode-teacher skill.
