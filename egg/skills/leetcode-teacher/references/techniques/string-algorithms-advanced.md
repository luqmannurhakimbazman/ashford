# Advanced String Algorithms

Twelve algorithm families covering string hashing, pattern matching automata, suffix structures, palindrome detection, and string decomposition. Builds on the interview-level basics in `string-techniques.md` and the general frameworks in `algorithm-frameworks.md`.

---

## Quick Reference Table

| Algorithm | Key Insight | When to Use | Time | Space |
|-----------|-------------|-------------|------|-------|
| String Hashing | Polynomial rolling hash maps substrings to integers | Substring comparison, duplicate detection | O(N) preprocess, O(1) query | O(N) |
| Rabin-Karp | Slide a hash window to find pattern matches | Single/multi-pattern matching, repeated substrings | O(N + M) average | O(1) |
| Prefix Function (KMP) | Longest proper prefix that is also a suffix | Exact pattern matching, period detection | O(N + M) | O(M) |
| Z-Function | Length of longest match starting at each position vs prefix | Pattern matching, string period, tandem repeats | O(N) | O(N) |
| Suffix Array | Sorted order of all suffixes + LCP array | Longest repeated/common substring, counting distinct substrings | O(N log N) build (O(N log^2 N) simple) | O(N) |
| Aho-Corasick | Trie + failure links = multi-pattern automaton | Search many patterns simultaneously in one text | O(N + M + Z) | O(M * \|Σ\|) |
| Suffix Automaton | Minimal DFA accepting all substrings | Count distinct substrings, longest common substring | O(N) build | O(N) |
| Suffix Tree (Ukkonen) | Compressed trie of all suffixes, online construction | Substring queries, longest repeated substring | O(N) build | O(N) |
| Manacher's Algorithm | Reuse palindrome expansions via rightmost boundary | All palindromic substrings/longest palindrome in O(N) | O(N) | O(N) |
| Lyndon Factorization | Duval's algorithm decomposes into non-increasing Lyndon words | Smallest cyclic rotation, lexicographic comparisons | O(N) | O(1) |
| Expression Parsing | Shunting-yard converts infix to RPN respecting precedence | Evaluate math expressions, parse nested operators | O(N) | O(N) |
| Main-Lorentz | Divide-and-conquer + Z-function finds all tandem repeats | Finding all repetitions in a string | O(N log N) | O(N) |

---

## 1. String Hashing

### Core Insight

Map a string to an integer using a polynomial hash so that substring comparisons become O(1) integer comparisons (with high probability).

### How It Works

Choose a base `p` (e.g., 31 for lowercase letters) and a large prime modulus `m` (e.g., 10^9 + 7 or 10^9 + 9). The hash of string `s[0..n-1]` is:

```
hash(s) = (s[0] * p^0 + s[1] * p^1 + ... + s[n-1] * p^(n-1)) mod m
```

Precompute prefix hashes and powers of `p` to get the hash of any substring `s[l..r]` in O(1):

```
hash(s[l..r]) = (prefix_hash[r+1] - prefix_hash[l] * p^(r-l+1)) mod m
```

### Python Template

```python
class StringHash:
    """Polynomial rolling hash for O(1) substring hash queries."""

    def __init__(self, s, base=31, mod=10**9 + 9):
        n = len(s)
        self.mod = mod
        self.pw = [1] * (n + 1)       # pw[i] = base^i mod m
        self.h = [0] * (n + 1)        # h[i] = hash(s[0..i-1])
        for i in range(n):
            self.h[i + 1] = (self.h[i] + (ord(s[i]) - ord('a') + 1) * self.pw[i]) % mod
            self.pw[i + 1] = self.pw[i] * base % mod

    def query(self, l, r):
        """Hash of s[l..r] inclusive. O(1)."""
        return (self.h[r + 1] - self.h[l] * self.pw[r - l + 1]) % self.mod
```

### Collision Avoidance — Double Hashing

A single hash has collision probability ~1/m per comparison. For problems with many comparisons, use **two independent hashes** (different base and mod):

```python
class DoubleHash:
    def __init__(self, s):
        self.h1 = StringHash(s, base=31, mod=10**9 + 7)
        self.h2 = StringHash(s, base=37, mod=10**9 + 9)

    def query(self, l, r):
        return (self.h1.query(l, r), self.h2.query(l, r))
```

### Applications

- O(1) substring equality checks after O(N) preprocessing
- Count distinct substrings (hash all substrings of each length)
- Longest common substring via binary search + hashing
- String period detection

*Socratic prompt: "If two substrings have the same hash, are they definitely equal? What's the trade-off between single and double hashing?"*

---

## 2. Rabin-Karp

### Core Insight

Slide a fixed-size hash window across the text. When the window hash matches the pattern hash, verify the match (to handle collisions).

### How It Works

1. Compute hash of the pattern `P` of length `M`.
2. Compute hash of the first `M` characters of text `T`.
3. Slide the window: remove the leftmost character's contribution, add the new rightmost character. Compare hashes at each position.

### Python Template

```python
def rabin_karp(text, pattern, base=31, mod=10**9 + 9):
    """Find all occurrences of pattern in text. O(N + M) average."""
    n, m = len(text), len(pattern)
    if m > n:
        return []

    # Precompute base^m mod
    power = pow(base, m, mod)

    # Hash function for a character
    def val(c):
        return ord(c) - ord('a') + 1

    # Compute pattern hash and initial window hash
    p_hash = 0
    t_hash = 0
    for i in range(m):
        p_hash = (p_hash * base + val(pattern[i])) % mod
        t_hash = (t_hash * base + val(text[i])) % mod

    result = []
    for i in range(n - m + 1):
        if t_hash == p_hash:
            # Verify to avoid false positives
            if text[i:i + m] == pattern:
                result.append(i)
        if i + m < n:
            # Slide window: remove text[i], add text[i + m]
            t_hash = (t_hash * base - val(text[i]) * power + val(text[i + m])) % mod
    return result
```

### Multi-Pattern Variant

To search for multiple patterns of the **same length**, hash all patterns into a set and check the sliding window hash against the set. For patterns of **different lengths**, group by length or use Aho-Corasick instead.

### Complexity

- **Average:** O(N + M) per pattern
- **Worst case:** O(N * M) if many hash collisions (rare with good hash)

*Socratic prompt: "Rabin-Karp's worst case is O(NM) — same as naive. Under what input would this happen, and how does double hashing help?"*

---

## 3. Prefix Function (KMP)

### Core Insight

For a string `s`, the prefix function `pi[i]` gives the length of the longest **proper** prefix of `s[0..i]` that is also a suffix. This encodes how much of the pattern can be reused after a mismatch.

### Key Recurrence

```
pi[0] = 0
For i = 1 to n-1:
    k = pi[i-1]
    while k > 0 and s[i] != s[k]:
        k = pi[k-1]    # Fall back through failure chain
    if s[i] == s[k]:
        k += 1
    pi[i] = k
```

### Python Template

```python
def prefix_function(s):
    """Compute KMP prefix function. O(N) time."""
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        k = pi[i - 1]
        while k > 0 and s[i] != s[k]:
            k = pi[k - 1]
        if s[i] == s[k]:
            k += 1
        pi[i] = k
    return pi


def kmp_search(text, pattern):
    """Find all occurrences of pattern in text. O(N + M)."""
    s = pattern + "#" + text  # '#' must not appear in either string
    pi = prefix_function(s)
    m = len(pattern)
    return [i - 2 * m for i in range(2 * m, len(s)) if pi[i] == m]
```

### Applications

- **Pattern matching:** Concatenate `pattern + "#" + text`, find positions where `pi[i] == len(pattern)`.
- **String period:** The smallest period of `s` is `n - pi[n-1]`. If `n % period == 0`, the string is a full repetition.
- **Counting prefix occurrences:** Walk the failure chain to count how many times each prefix appears.

### The `#` Separator Trick

When computing the prefix function on `pattern + "#" + text`, the separator `#` (a character not in either string) ensures `pi[i]` never exceeds `len(pattern)`. This prevents the prefix function from "crossing" the boundary.

*Socratic prompt: "What happens if you forget the '#' separator? Can the prefix function value exceed len(pattern)?"*

---

## 4. Z-Function

### Core Insight

For a string `s`, `z[i]` is the length of the longest substring starting at position `i` that matches a prefix of `s`. In other words, `s[0..z[i]-1] == s[i..i+z[i]-1]`.

### Algorithm (Linear Time)

Maintain a window `[l, r)` — the rightmost segment that matches a prefix of `s`:

```python
def z_function(s):
    """Compute Z-array. O(N) time."""
    n = len(s)
    z = [0] * n
    z[0] = n  # By convention (or leave as 0)
    l, r = 0, 0
    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > r:
            l, r = i, i + z[i]
    return z
```

### Applications

- **Pattern matching:** Compute Z on `pattern + "$" + text`. Positions where `z[i] == len(pattern)` are matches.
- **String period:** Smallest `i` where `z[i] + i == n` and `n % i == 0` gives the period.
- **Number of distinct substrings:** Add characters one by one, use Z-function on the reversed current string.
- **Building blocks for Main-Lorentz** (tandem repeat finding).

### Z-Function vs Prefix Function

| Property | Z-Function | Prefix Function |
|----------|-----------|-----------------|
| `z[i]` / `pi[i]` meaning | Match length at position `i` vs prefix | Longest proper prefix-suffix of `s[0..i]` |
| Pattern matching | `pattern + "$" + text` | `pattern + "#" + text` |
| Easier to understand | Often more intuitive | More commonly taught |
| Used in | Main-Lorentz, tandem repeats | KMP, Aho-Corasick failure links |

*Socratic prompt: "Both Z-function and KMP achieve O(N) pattern matching. Can you convert between z[i] and pi[i]? What information does one capture that the other doesn't directly show?"*

---

## 5. Suffix Array

### Core Insight

A suffix array is the sorted order of all suffixes of a string. Combined with the LCP (Longest Common Prefix) array, it supports powerful substring queries.

### Construction — O(N log N) Doubling

Sort suffixes by their first 1, 2, 4, 8, ... characters. Each doubling step sorts by pairs `(rank[i], rank[i+k])`. With radix/counting sort each step is O(N), giving O(N log N) total. The Python version below uses comparison sort per step — O(N log^2 N) total — which is simpler to implement and sufficient for most interview/contest problems up to N ~ 10^5.

```python
def build_suffix_array(s):
    """O(N log^2 N) suffix array via doubling + comparison sort.
    For true O(N log N), replace .sort() with two passes of counting sort."""
    s += chr(0)  # Sentinel (smallest character)
    n = len(s)
    sa = list(range(n))
    rank = [ord(c) for c in s]
    tmp = [0] * n

    k = 1
    while k < n:
        def sort_key(i):
            return (rank[i], rank[i + k] if i + k < n else -1)
        sa.sort(key=sort_key)

        # Recompute ranks
        tmp[sa[0]] = 0
        for i in range(1, n):
            tmp[sa[i]] = tmp[sa[i - 1]]
            if sort_key(sa[i]) != sort_key(sa[i - 1]):
                tmp[sa[i]] += 1
        rank = tmp[:]
        if rank[sa[-1]] == n - 1:
            break
        k *= 2

    return sa[1:]  # Remove sentinel suffix
```

### LCP Array — Kasai's Algorithm

The LCP array stores the length of the longest common prefix between consecutive suffixes in sorted order. Kasai's algorithm computes it in O(N):

```python
def build_lcp(s, sa):
    """Kasai's algorithm. O(N) LCP array from suffix array."""
    n = len(s)
    rank = [0] * n
    for i in range(n):
        rank[sa[i]] = i
    lcp = [0] * n
    k = 0
    for i in range(n):
        if rank[i] == 0:
            k = 0
            continue
        j = sa[rank[i] - 1]
        while i + k < n and j + k < n and s[i + k] == s[j + k]:
            k += 1
        lcp[rank[i]] = k
        if k > 0:
            k -= 1
    return lcp
```

### Applications

- **Longest repeated substring:** Maximum value in the LCP array.
- **Number of distinct substrings:** `n*(n+1)/2 - sum(lcp)`.
- **Longest common substring of two strings:** Concatenate with separator, build SA + LCP, find max LCP between suffixes from different strings.
- **Pattern search in text:** Binary search on the suffix array — O(M log N).

*Socratic prompt: "Why does subtracting the sum of the LCP array from the total number of substrings give the count of distinct substrings?"*

---

## 6. Aho-Corasick

### Core Insight

Build a trie of all patterns, then add **suffix links** (analogous to KMP failure links) to create a finite automaton. Process the text through this automaton in one pass to find all pattern occurrences simultaneously.

### How It Works

1. **Build trie** from all patterns.
2. **Compute suffix links** via BFS from root (like KMP failure function on a trie).
3. **Compute dictionary suffix links** — shortcuts to the nearest node that is an end of some pattern.
4. **Search:** Walk the text through the automaton. At each node, follow dictionary suffix links to collect all matching patterns.

### Python Template

```python
from collections import deque

class AhoCorasick:
    def __init__(self):
        self.goto = [{}]      # goto[state][char] -> next state
        self.fail = [0]       # suffix link
        self.output = [[]]    # pattern indices ending at this state

    def add_pattern(self, pattern, index):
        state = 0
        for c in pattern:
            if c not in self.goto[state]:
                self.goto[state][c] = len(self.goto)
                self.goto.append({})
                self.fail.append(0)
                self.output.append([])
            state = self.goto[state][c]
        self.output[state].append(index)

    def build(self):
        """BFS to compute failure links. O(sum of pattern lengths * |Sigma|)."""
        queue = deque()
        for c, s in self.goto[0].items():
            queue.append(s)
        while queue:
            u = queue.popleft()
            for c, v in self.goto[u].items():
                queue.append(v)
                f = self.fail[u]
                while f and c not in self.goto[f]:
                    f = self.fail[f]
                self.fail[v] = self.goto[f].get(c, 0)
                if self.fail[v] == v:
                    self.fail[v] = 0
                self.output[v] = self.output[v] + self.output[self.fail[v]]

    def search(self, text):
        """Find all pattern matches in text. Returns (position, pattern_index) pairs."""
        state = 0
        results = []
        for i, c in enumerate(text):
            while state and c not in self.goto[state]:
                state = self.fail[state]
            state = self.goto[state].get(c, 0)
            for pat_idx in self.output[state]:
                results.append((i, pat_idx))
        return results
```

### Complexity

- **Build:** O(sum of pattern lengths * |alphabet|)
- **Search:** O(N + Z) where Z is the total number of matches

### Applications

- Search for multiple patterns in one text simultaneously
- DNA sequence matching (multiple motifs)
- Content filtering / keyword detection
- Web crawling (match URL patterns)

*Socratic prompt: "KMP handles one pattern. Aho-Corasick handles many. What's the relationship between KMP's failure function and Aho-Corasick's suffix links?"*

---

## 7. Suffix Automaton (DAWG)

### Core Insight

The suffix automaton is the **smallest DFA** (in number of states) that accepts exactly all substrings of a string. It has at most `2N - 1` states and `3N - 4` transitions. Each state represents an **equivalence class** of substrings that always occur together (same set of ending positions).

### Key Concepts

- **endpos(t):** The set of all ending positions of substring `t` in `s`.
- **Equivalence class:** Two substrings are equivalent if they have the same `endpos` set.
- **Suffix link:** Points from a state to the state representing the longest proper suffix in a different equivalence class.
- **len:** Each state stores the length of the longest substring in its equivalence class.

### Python Template

```python
class SuffixAutomaton:
    def __init__(self):
        self.states = [{"len": 0, "link": -1, "trans": {}}]  # Initial state
        self.last = 0

    def extend(self, c):
        """Add character c to the automaton. O(1) amortized."""
        cur = len(self.states)
        self.states.append({"len": self.states[self.last]["len"] + 1,
                            "link": -1, "trans": {}})
        p = self.last
        while p != -1 and c not in self.states[p]["trans"]:
            self.states[p]["trans"][c] = cur
            p = self.states[p]["link"]

        if p == -1:
            self.states[cur]["link"] = 0
        else:
            q = self.states[p]["trans"][c]
            if self.states[p]["len"] + 1 == self.states[q]["len"]:
                self.states[cur]["link"] = q
            else:
                clone = len(self.states)
                self.states.append({
                    "len": self.states[p]["len"] + 1,
                    "link": self.states[q]["link"],
                    "trans": dict(self.states[q]["trans"])
                })
                while p != -1 and self.states[p]["trans"].get(c) == q:
                    self.states[p]["trans"][c] = clone
                    p = self.states[p]["link"]
                self.states[q]["link"] = clone
                self.states[cur]["link"] = clone
        self.last = cur

    def build(self, s):
        for c in s:
            self.extend(c)
```

### Applications

- **Count distinct substrings:** Sum `(len[v] - len[link[v]])` over all states `v != initial`.
- **Total length of all distinct substrings.**
- **Longest common substring:** Build automaton on one string, walk the other through it tracking the longest match.
- **Check if a string is a substring:** Walk the automaton — if you never fall off, it's a substring.
- **Smallest cyclic shift** (with modifications).

### Complexity

- **Construction:** O(N) amortized
- **Space:** O(N) states and O(N * |alphabet|) transitions

*Socratic prompt: "The suffix automaton has at most 2N-1 states for a string of length N. Why is this tight? What string achieves the maximum?"*

---

## 8. Suffix Tree (Ukkonen's Algorithm)

### Core Insight

A suffix tree is a **compressed trie** of all suffixes of a string. Ukkonen's algorithm builds it online in O(N) time using three key tricks: implicit extensions, the active point, and the "do nothing" trick for suffix links.

### Key Properties

- Every internal node has at least 2 children.
- Edges are labeled with substrings (stored as index ranges to save space).
- There are exactly `N` leaves (one per suffix).
- At most `N - 1` internal nodes, so `2N - 1` nodes total.

### Conceptual Overview (Ukkonen)

Ukkonen builds the tree incrementally, processing one character at a time:

1. **Phase i:** Add character `s[i]` to the tree.
2. **Rule 1 (leaf extension):** If a suffix ends at a leaf, extend the leaf edge — this happens implicitly since leaf edges extend to the end.
3. **Rule 2 (new branch):** If the suffix path leads to a mismatch at an edge, split the edge and create a new leaf.
4. **Rule 3 (do nothing):** If the suffix is already present implicitly, stop — later characters will extend it.

The **active point** (active node, active edge, active length) tracks where the next extension starts. **Suffix links** connect internal nodes to enable O(1) jumps.

### When to Use

Suffix trees and suffix arrays are largely interchangeable. In practice, suffix arrays are preferred in competitive programming due to simpler implementation and lower constant factors. Suffix trees shine for:

- Problems requiring explicit tree traversal (e.g., finding all maximal repeats)
- Online algorithms where the string grows character by character
- Theoretical proofs and teaching

### Relationship to Other Structures

| Structure | Space | Build Time | Notes |
|-----------|-------|------------|-------|
| Suffix Tree | O(N) | O(N) (Ukkonen) | Explicit tree, complex to implement |
| Suffix Array + LCP | O(N) | O(N log N) typical | Simpler, preferred in contests |
| Suffix Automaton | O(N) | O(N) | Minimal DFA, easiest to implement |

*Socratic prompt: "Suffix trees and suffix automata both process all substrings of a string. What's the conceptual difference — what does each structure organize, and when would you pick one over the other?"*

---

## 9. Manacher's Algorithm

### Core Insight

Find all palindromic substrings in O(N) by reusing previously computed palindrome expansions. Maintain the **rightmost palindrome boundary** and mirror known palindrome lengths from the left half to the right half.

### Algorithm

Transform the string to handle both odd and even length palindromes uniformly by inserting separators: `"abc"` → `"#a#b#c#"`. Then compute the palindrome radius `p[i]` at each center.

```python
def manacher(s):
    """Find all palindromic substrings in O(N).
    Returns array p where p[i] = radius of palindrome centered at i
    in the transformed string."""
    # Transform: "abc" -> "^#a#b#c#$"
    t = "^#" + "#".join(s) + "#$"
    n = len(t)
    p = [0] * n
    center = right = 0  # Center and right boundary of rightmost palindrome

    for i in range(1, n - 1):
        mirror = 2 * center - i  # Mirror of i around center

        if i < right:
            p[i] = min(right - i, p[mirror])

        # Try to expand
        while t[i + p[i] + 1] == t[i - p[i] - 1]:
            p[i] += 1

        # Update rightmost palindrome
        if i + p[i] > right:
            center, right = i, i + p[i]

    return p


def longest_palindrome_manacher(s):
    """Find the longest palindromic substring in O(N)."""
    p = manacher(s)
    max_len = max(p)
    center_idx = p.index(max_len)
    # Map back to original string
    start = (center_idx - max_len) // 2
    return s[start:start + max_len]
```

### Why It's O(N)

Each expansion either extends the rightmost boundary `right` (which can only increase N times total) or copies a value from the mirror (O(1)). The amortized work is linear.

### Applications

- Longest palindromic substring in O(N) — improves on O(N^2) expand-from-center
- Count all palindromic substrings
- Palindrome partitioning preprocessing

### Interview Relevance

Manacher's is rarely expected in interviews but is a common follow-up: *"Can you do better than O(N^2)?"* for Longest Palindromic Substring (LC 5). Knowing it exists and being able to explain the idea (without memorizing code) is sufficient for most interviews.

*Socratic prompt: "Expand-from-center is O(N^2). Manacher reuses previously computed palindromes to avoid redundant expansions. How does the 'mirror' idea prevent re-doing work?"*

---

## 10. Lyndon Factorization (Duval's Algorithm)

### Core Insight

A **Lyndon word** is a string that is strictly lexicographically smaller than all its proper rotations (e.g., `"abc"` is Lyndon but `"abab"` is not). Every string has a unique **Lyndon factorization** into non-increasing Lyndon words: `s = w1 * w2 * ... * wk` where `w1 >= w2 >= ... >= wk` lexicographically.

### Duval's Algorithm — O(N), O(1) Space

```python
def lyndon_factorization(s):
    """Decompose s into non-increasing Lyndon words. O(N) time, O(1) space."""
    n = len(s)
    factorization = []
    i = 0
    while i < n:
        j, k = i, i + 1
        while k < n and s[j] <= s[k]:
            if s[j] < s[k]:
                j = i  # Reset — still building Lyndon word
            else:
                j += 1  # s[j] == s[k], continue matching
            k += 1
        # The Lyndon word length is k - j
        word_len = k - j
        while i <= j:
            factorization.append(s[i:i + word_len])
            i += word_len
    return factorization
```

### Application: Smallest Cyclic Shift

The smallest cyclic shift of `s` starts at the beginning of the last Lyndon word in the Lyndon factorization of `s + s`:

```python
def smallest_cyclic_shift(s):
    """Find starting index of the lexicographically smallest rotation. O(N)."""
    ss = s + s
    n = len(s)
    i = 0
    while i < n:
        j, k = i, i + 1
        while k < 2 * n and ss[j] <= ss[k]:
            if ss[j] < ss[k]:
                j = i
            else:
                j += 1
            k += 1
        word_len = k - j
        while i <= j:
            if i + word_len >= n:
                return i
            i += word_len
    return 0
```

*Socratic prompt: "Why must every string have a unique Lyndon factorization? Can you prove the decomposition is unique using the definition of Lyndon words?"*

---

## 11. Expression Parsing

### Core Insight

The **Shunting-yard algorithm** (Dijkstra) converts infix expressions (e.g., `3 + 4 * 2`) to Reverse Polish Notation (RPN) or directly evaluates them, respecting operator precedence and associativity.

### Algorithm

Use two stacks: one for values, one for operators. Process tokens left to right:
- **Number:** Push to value stack.
- **`(`:** Push to operator stack.
- **`)`:** Pop and apply operators until `(` is found.
- **Operator:** Pop and apply operators with higher (or equal, for left-associative) precedence, then push current operator.

### Python Template

```python
def evaluate_expression(expr):
    """Evaluate an infix expression with +, -, *, /, and parentheses."""
    def precedence(op):
        if op in ('+', '-'):
            return 1
        if op in ('*', '/'):
            return 2
        return 0

    def apply_op(op, b, a):
        if op == '+': return a + b
        if op == '-': return a - b
        if op == '*': return a * b
        if op == '/': return int(a / b)  # Truncate toward zero

    values = []
    ops = []
    i = 0

    while i < len(expr):
        if expr[i] == ' ':
            i += 1
            continue
        if expr[i].isdigit():
            num = 0
            while i < len(expr) and expr[i].isdigit():
                num = num * 10 + int(expr[i])
                i += 1
            values.append(num)
            continue
        if expr[i] == '(':
            ops.append('(')
        elif expr[i] == ')':
            while ops[-1] != '(':
                values.append(apply_op(ops.pop(), values.pop(), values.pop()))
            ops.pop()  # Remove '('
        else:
            while ops and ops[-1] != '(' and precedence(ops[-1]) >= precedence(expr[i]):
                values.append(apply_op(ops.pop(), values.pop(), values.pop()))
            ops.append(expr[i])
        i += 1

    while ops:
        values.append(apply_op(ops.pop(), values.pop(), values.pop()))

    return values[0]
```

### Handling Unary Minus

Unary minus appears at the start of an expression or after `(`. Insert a `0` before it to convert `-x` to `0-x`, or assign unary minus a higher precedence.

### Applications

- Basic Calculator I/II/III (LC 224, 227, 772)
- Evaluate Reverse Polish Notation (LC 150)
- Building simple interpreters and calculators

*Socratic prompt: "Why does the Shunting-yard algorithm need to compare precedences when deciding whether to pop the operator stack? What goes wrong if you always push?"*

---

## 12. Main-Lorentz (Finding All Tandem Repeats)

### Core Insight

A **tandem repeat** (or square) in a string is a substring of the form `ww` (a word repeated twice). The Main-Lorentz algorithm finds **all** tandem repeats in O(N log N) time using divide-and-conquer combined with the Z-function.

### How It Works

1. **Divide** the string at the midpoint.
2. **Find tandem repeats** that cross the midpoint using Z-functions of the left half (reversed) and right half.
3. **Recurse** on both halves.

For the crossing step: for each possible repeat length `l`, use Z-function values to check if a tandem repeat of length `2l` crosses the midpoint in O(1) per length.

### Complexity

- **Time:** O(N log N) — each level of recursion does O(N) work via Z-function, with log N levels.
- **Space:** O(N)

### When to Use

This algorithm is specialized for competitive programming problems asking to:
- Count all tandem repeats (squares) in a string
- Find the shortest or longest tandem repeat
- Determine if a string is "square-free"

*Socratic prompt: "The naive approach to finding all tandem repeats is O(N^3) — check all starting positions and all lengths. How does divide-and-conquer reduce this to O(N log N)?"*

---

## Algorithm Selection Guide

**Which algorithm for which problem type?**

| Problem Type | Best Algorithm | Runner-Up |
|-------------|---------------|-----------|
| Single pattern matching | KMP (prefix function) | Z-function |
| Multiple pattern matching | Aho-Corasick | Rabin-Karp (same-length patterns) |
| Substring equality queries | String hashing | Suffix array |
| Longest repeated substring | Suffix array + LCP | Suffix automaton |
| Count distinct substrings | Suffix automaton | Suffix array + LCP |
| Longest common substring (2 strings) | Suffix automaton | Suffix array + LCP |
| Longest palindromic substring | Manacher's | Expand from center (O(N^2)) |
| All palindromic substrings | Manacher's | Palindrome DP |
| Smallest cyclic rotation | Lyndon factorization | Suffix array |
| Expression evaluation | Shunting-yard | Recursive descent |
| Find all tandem repeats | Main-Lorentz | Naive O(N^3) |
| Check if substring exists | Suffix automaton | String hashing |

**Decision tree:**
1. *How many patterns?* → 1: KMP or Z-function. Many: Aho-Corasick.
2. *Need substring queries on one text?* → Suffix array/automaton. Just equality? → Hashing.
3. *Palindrome-related?* → Manacher's for O(N), expand-from-center for simplicity.
4. *Lexicographic ordering / rotations?* → Suffix array or Lyndon factorization.

---

## Practice Problems

| Problem | Difficulty | Algorithm(s) |
|---------|-----------|-------------|
| Implement strStr (28) | Easy | KMP, Rabin-Karp |
| Repeated Substring Pattern (459) | Easy | KMP period check |
| Shortest Palindrome (214) | Hard | KMP on `s + "#" + rev(s)` |
| Longest Duplicate Substring (1044) | Hard | Binary search + Rabin-Karp or Suffix Array |
| Longest Happy Prefix (1392) | Hard | KMP prefix function |
| Longest Palindromic Substring (5) | Medium | Manacher's or expand from center |
| Palindromic Substrings (647) | Medium | Manacher's or expand from center |
| Distinct Substrings (Σ) | — | Suffix array + LCP or suffix automaton |
| Stream of Characters (1032) | Hard | Aho-Corasick (or trie of reversed patterns) |
| Basic Calculator (224) | Hard | Shunting-yard / recursive descent |
| Basic Calculator II (227) | Medium | Shunting-yard |
| Word Search II (212) | Hard | Trie (Aho-Corasick ideas) |

---

## Pattern Connections

| If You Know... | Then You Can Solve... |
|----------------|----------------------|
| KMP / prefix function (this file) | Pattern matching, string period problems, Shortest Palindrome (214) |
| String hashing (this file) | Substring equality, Longest Duplicate Substring (1044) via binary search |
| Sliding window (`algorithm-frameworks.md`) | Combine with hashing for rolling window substring problems |
| Trie (`data-structure-fundamentals.md`) | Aho-Corasick builds on tries; Word Search II uses trie pruning |
| Expand from center (`string-techniques.md`) | Manacher's is the O(N) upgrade of the same intuition |
| Binary search (`binary-search-framework.md`) | Binary search on answer + hashing = longest duplicate/common substring |
| DP string matching (`dynamic-programming-core.md`) | Edit distance, LCS complement the exact matching algorithms here |

*Socratic prompt: "Many advanced string algorithms share a common idea: avoid re-doing work by remembering what you already know. How does this manifest differently in KMP (failure links), Z-function (the [l,r) window), Manacher's (mirror trick), and suffix automaton (suffix links)?"*

---

## Attribution

The algorithms and techniques in this file are adapted from the cp-algorithms.com project (e-maxx). Templates have been converted to Python and annotated for Socratic teaching use.
