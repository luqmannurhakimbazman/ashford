# Probability & Randomness

Probability fundamentals, classic paradoxes, and randomized algorithms for coding interviews. For brain teasers and game theory, see `brain-teasers-games.md`. For random map generation (Minesweeper), see `brain-teasers-games.md` Section 4.

---

## Quick Reference Table

| Pattern | Recognition Signals | Key Problems | Section |
|---------|-------------------|--------------|---------|
| Probability Fundamentals | "What's the probability", "sample space", "conditional probability" | -- (foundation) | 1 |
| Classic Paradoxes | "Boy-girl problem", "birthday paradox", "Monty Hall" | -- (interview discussion) | 2 |
| Fisher-Yates Shuffle | "Shuffle an array", "random permutation", "equally likely" | Shuffle an Array (384) | 3 |
| Reservoir Sampling | "Random sample from stream", "pick random from linked list" | Linked List Random Node (382), Random Pick Index (398) | 4 |

---

## 1. Probability Fundamentals

### Key Concepts

| Concept | Definition | Example |
|---------|-----------|---------|
| Sample space (Ω) | Set of all possible outcomes | Coin flip: {H, T} |
| Event | Subset of the sample space | "Heads": {H} |
| P(A) | Number of favorable outcomes / total outcomes | P(H) = 1/2 |
| P(A∩B) | P(both A and B occur) | Independent: P(A) × P(B) |
| P(A\|B) | P(A given B) = P(A∩B) / P(B) | Bayes' theorem follows |

### Conditional Probability & Bayes' Theorem

```
P(A|B) = P(B|A) × P(A) / P(B)
```

This is the foundation for several classic puzzles. The key error people make: **confusing P(A|B) with P(B|A)**.

*Socratic prompt: "If 1% of people have a disease, and a test is 99% accurate, what's the probability someone who tests positive actually has the disease? (Hint: it's NOT 99%.)"*

---

## 2. Classic Paradoxes

### The Boy-Girl Problem

**Setup:** A family has two children. You learn that at least one is a boy. What's the probability that both are boys?

**Intuitive (wrong) answer:** 1/2 — "the other child is equally likely boy or girl."

**Correct answer:** 1/3. The sample space of "two children, at least one boy" is {BB, BG, GB}. Only one of three equally likely cases is BB.

**Variant:** "You meet one of their children and it's a boy." Now the answer IS 1/2, because you've sampled a specific child, not learned about the family as a whole.

*Socratic prompt: "Write out all four possible outcomes for two children: {BB, BG, GB, GG}. Cross out the ones eliminated by 'at least one boy.' What fraction of remaining outcomes is BB?"*

### The Birthday Paradox

**Setup:** How many people needed for a >50% chance that two share a birthday?

**Answer:** Just 23. Most people guess much higher.

**Why:** Compute the complement — probability that ALL birthdays are unique:

```
P(all unique) = 365/365 × 364/365 × 363/365 × ... × (365-n+1)/365
```

```python
def birthday_no_match_probability(n: int) -> float:
    """Probability that n people all have different birthdays."""
    p = 1.0
    for i in range(n):
        p *= (365 - i) / 365
    return p

# birthday_no_match_probability(23) ≈ 0.493 → P(match) ≈ 0.507
```

**Interview application:** This appears in hash collision analysis. With n items and k buckets, collisions become likely much sooner than intuition suggests.

### The Monty Hall Problem

**Setup:** Three doors — one has a prize. You pick door 1. The host (who knows what's behind each door) opens door 3 (a goat). Should you switch to door 2?

**Answer:** Yes! Switching wins with probability 2/3.

**Why:** Your initial pick has P = 1/3. The host's action doesn't change this — it just concentrates the remaining 2/3 probability onto the other unopened door.

| Strategy | P(win) |
|----------|--------|
| Stay | 1/3 |
| Switch | 2/3 |

**The key insight:** The host's choice is NOT random. They always open a door with a goat. This asymmetry is what makes switching beneficial.

*Socratic prompt: "Imagine 100 doors instead of 3. You pick one, the host opens 98 doors showing goats. Would you switch to the one remaining door? Does this make the 2/3 probability more intuitive?"*

---

## 3. Fisher-Yates Shuffle (LC 384)

### Problem

Shuffle an array so that every permutation is equally likely.

### Core Insight

The **Fisher-Yates (Knuth) shuffle** generates a uniformly random permutation in O(n) time. At step i, pick a random element from `arr[i..n-1]` and swap it with `arr[i]`.

### Why It Produces Uniform Permutations

There are `n!` possible permutations. The algorithm makes n random choices:
- First element: n options
- Second element: n-1 options
- ...
- Last element: 1 option

Total: `n × (n-1) × ... × 1 = n!` equally likely outcomes. Each permutation has probability `1/n!`.

### Template

```python
import random

class Solution:
    def __init__(self, nums: list[int]):
        self.original = nums[:]
        self.arr = nums[:]

    def reset(self) -> list[int]:
        self.arr = self.original[:]
        return self.arr

    def shuffle(self) -> list[int]:
        """Fisher-Yates shuffle: O(n) time, O(1) extra space."""
        for i in range(len(self.arr)):
            # Pick random index from [i, n-1]
            j = random.randint(i, len(self.arr) - 1)
            self.arr[i], self.arr[j] = self.arr[j], self.arr[i]
        return self.arr
```

### Common Mistake: The Naive Shuffle

```python
# WRONG: not uniformly random!
for i in range(n):
    j = random.randint(0, n - 1)  # Bug: should be randint(i, n-1)
    arr[i], arr[j] = arr[j], arr[i]
```

This produces `n^n` possible execution paths, which is not divisible by `n!` for most n. Some permutations are more likely than others.

*Socratic prompt: "For an array of size 3, the naive shuffle has 3^3 = 27 paths but only 3! = 6 permutations. Can 27 divide evenly among 6? What does this mean for uniformity?"*

---

## 4. Reservoir Sampling (LC 382, LC 398)

### Problem

Given a stream of unknown length, select k items uniformly at random using O(k) memory.

### Core Insight

**Reservoir sampling** maintains a "reservoir" of k items. For the i-th item in the stream (i ≥ k):
- With probability `k/i`, replace a random reservoir item with the new item
- Otherwise, discard the new item

For the common case of k=1: keep the i-th item with probability `1/i`.

### Why It's Uniform

After processing n items, each item has probability `k/n` of being in the reservoir.

**Proof for k=1:** Item i is selected with P = 1/i, then survives the next n-i steps. The probability of survival:

```
P(item i in final reservoir) = (1/i) × (i/(i+1)) × ((i+1)/(i+2)) × ... × ((n-1)/n) = 1/n
```

The fractions telescope to `1/n` for every i. Uniform!

### Template: k=1 (LC 382)

```python
import random

class Solution:
    def __init__(self, head):
        self.head = head

    def get_random(self) -> int:
        """Return a random node's value from linked list."""
        result = 0
        node = self.head
        i = 1
        while node:
            # Keep current node with probability 1/i
            if random.randint(1, i) == 1:
                result = node.val
            node = node.next
            i += 1
        return result
```

### Template: General k

```python
import random

def reservoir_sample(stream, k: int) -> list:
    """Select k items uniformly at random from a stream."""
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            # Replace a random element with probability k/(i+1)
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir
```

### Application: Random Pick Index (LC 398)

Given an array with possible duplicates, randomly pick an index of a target value. Each valid index should be equally likely.

```python
class Solution:
    def __init__(self, nums: list[int]):
        self.nums = nums

    def pick(self, target: int) -> int:
        """Reservoir sampling (k=1) over indices matching target."""
        result = -1
        count = 0
        for i, num in enumerate(self.nums):
            if num == target:
                count += 1
                if random.randint(1, count) == 1:
                    result = i
        return result
```

*Socratic prompt: "Why can't you just store all indices of the target and pick randomly? When would reservoir sampling be preferred? (Hint: think about memory constraints and streaming data.)"*

### When to Use Reservoir Sampling

| Scenario | Use Reservoir Sampling? | Alternative |
|----------|------------------------|-------------|
| Unknown stream length | Yes | -- |
| Data too large for memory | Yes | -- |
| Known size, fits in memory | No — just shuffle and take first k | Fisher-Yates |
| Need repeated sampling | No — preprocess and use random index | Hash map + random |

### Problems

| Problem | Key Twist |
|---------|-----------|
| Linked List Random Node (382) | k=1 reservoir sampling on a linked list |
| Random Pick Index (398) | Reservoir sampling filtered by target value |

---

## Attribution

Content synthesized from labuladong's algorithmic guide, Chapter 4 — "Other Common Techniques: Probability and Randomness." Reorganized and augmented with Socratic teaching prompts, cross-references, and code templates for the leetcode-teacher skill.
