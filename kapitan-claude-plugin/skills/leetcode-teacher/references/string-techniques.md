# String Techniques

Strings are arrays of characters — most array techniques apply directly. This file covers string-specific patterns: character counting, anagram detection, palindrome techniques, and common string algorithms.

---

## Quick Reference Table

| Technique | Key Insight | When to Use | Complexity |
|-----------|-------------|-------------|------------|
| Counting Characters | Hash map or fixed-size array for frequency | "Anagram", "permutation in string", "character frequency" | O(N) |
| Bitmask Character Set | Track character presence with 26 bits | "Unique characters", "all lowercase letters" | O(N), O(1) space |
| Anagram Detection | Compare sorted strings or frequency maps | "Anagram", "group anagrams", "permutation" | O(N) with frequency map |
| Palindrome (Two Pointers) | Compare from both ends toward center | "Is palindrome", "longest palindromic substring" | O(N) check, O(N^2) search |
| Expand from Center | Expand outward from each center point | "Longest palindromic substring" | O(N^2) |
| Palindrome DP | `dp[i][j]` = whether `s[i..j]` is a palindrome | "Palindrome partitioning", "count palindromic substrings" | O(N^2) |
| Rabin-Karp | Rolling hash for substring matching | "Find pattern in string", "repeated substrings" | O(N) average |

---

## Time Complexity of Common String Operations

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Access character `s[i]` | O(1) | Strings are arrays |
| Search for character | O(N) | Linear scan |
| Search for substring | O(N*M) naive, O(N) with KMP/Rabin-Karp | M = pattern length |
| Insert / remove character | O(N) | Requires shifting or rebuilding |
| Substring / slice `s[i:j]` | O(j - i) | Creates a new string (copy) |
| Concatenation `s + t` | O(|s| + |t|) | Creates a new string |
| Compare two strings | O(min(|s|, |t|)) | Character by character |

**Interview tip:** String slicing in Python creates a copy and costs O(K) where K is the slice length. Avoid slicing inside loops — use index tracking instead.

---

## Interview Tips & Corner Cases

### Tips

- Ask about the character set — ASCII (128), extended ASCII (256), or Unicode? This determines array size for frequency counting.
- Clarify case sensitivity. Should `'A'` and `'a'` be treated the same?
- Strings are **immutable** in Python, Java, JavaScript, and Go. Concatenation in a loop is O(N^2) — use a list/StringBuilder and join at the end.
- When counting characters, use `collections.Counter` in Python or a fixed-size array of length 26/128/256.

### Corner Cases

- Empty string
- String with one or two characters
- String with all identical characters (e.g., `"aaa"`)
- Strings with only unique characters
- Strings with spaces, punctuation, or non-alphanumeric characters
- Strings with Unicode or non-ASCII characters (if applicable)

---

## 1. Counting Characters

### Hash Map Approach

The most common string technique. Count character frequencies to compare strings or detect patterns.

```python
from collections import Counter

def are_anagrams(s, t):
    """Check if two strings are anagrams. O(N) time, O(K) space."""
    return Counter(s) == Counter(t)

def first_unique_char(s):
    """Find the first non-repeating character. O(N) time."""
    count = Counter(s)
    for i, c in enumerate(s):
        if count[c] == 1:
            return i
    return -1
```

### Bitmask Trick for Unique Characters

When you only need to track **presence** (not frequency) of lowercase letters, use a 26-bit integer instead of a hash map. Each bit represents a letter.

```python
def has_all_unique_chars(s):
    """Check if all characters are unique using bitmask. O(N) time, O(1) space."""
    seen = 0
    for c in s:
        bit = 1 << (ord(c) - ord('a'))
        if seen & bit:
            return False  # Character already seen
        seen |= bit
    return True

def max_length_unique_concat(arr):
    """Maximum length of concatenation with unique characters (LC 1239)."""
    # Represent each string as a bitmask
    masks = []
    for s in arr:
        mask = 0
        for c in s:
            bit = 1 << (ord(c) - ord('a'))
            if mask & bit:  # Duplicate in s itself
                mask = 0
                break
            mask |= bit
        if mask:
            masks.append((mask, len(s)))
    # Backtrack or DP over masks
    # ...
```

*Socratic prompt: "If you need to check whether two strings share any common characters, how can you do it in O(1) with bitmasks?"*

---

## 2. Anagram Detection

Three methods, each with different trade-offs:

### Method 1: Frequency Counting (Preferred)

Compare character frequency maps. O(N) time, O(K) space where K = character set size.

```python
def find_anagrams(s, p):
    """Find all anagram start indices of p in s (LC 438). Sliding window approach."""
    from collections import Counter
    if len(p) > len(s):
        return []

    p_count = Counter(p)
    window = Counter(s[:len(p)])
    result = []

    if window == p_count:
        result.append(0)

    for i in range(len(p), len(s)):
        # Add new character to window
        window[s[i]] += 1
        # Remove oldest character from window
        old = s[i - len(p)]
        window[old] -= 1
        if window[old] == 0:
            del window[old]
        if window == p_count:
            result.append(i - len(p) + 1)

    return result
```

### Method 2: Sorting

Sort both strings and compare. O(N log N) time, O(N) space. Simple but slower.

```python
def is_anagram_sort(s, t):
    return sorted(s) == sorted(t)
```

### Method 3: Prime Mapping

Map each character to a unique prime number. The product of primes is identical for anagrams (by the fundamental theorem of arithmetic). Risk: integer overflow for long strings.

```python
def anagram_hash(s):
    """Map characters to primes. Same product = anagram."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
              47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
    product = 1
    for c in s:
        product *= primes[ord(c) - ord('a')]
    return product
```

*Socratic prompt: "The frequency counting method is O(N). The sorting method is O(N log N). When might you still prefer sorting? Hint: think about implementation simplicity and debugging."*

### Problems

| Problem | Key Twist |
|---------|-----------|
| Valid Anagram (242) | Direct frequency comparison |
| Group Anagrams (49) | Use sorted string or frequency tuple as hash key |
| Find All Anagrams in a String (438) | Sliding window + frequency map |
| Minimum Window Substring (76) | Sliding window with "need" vs "have" tracking |

---

## 3. Palindrome Techniques

### Two-Pointer Check

The simplest palindrome test: compare characters from both ends moving inward.

```python
def is_palindrome(s):
    """O(N) time, O(1) space."""
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
```

### Expand from Center

Find the longest palindromic substring by treating each character (and each gap between characters) as a potential center, then expanding outward.

```python
def longest_palindromic_substring(s):
    """O(N^2) time, O(1) space."""
    def expand(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]

    result = ""
    for i in range(len(s)):
        # Odd-length palindromes (center = single character)
        odd = expand(i, i)
        # Even-length palindromes (center = gap between two characters)
        even = expand(i, i + 1)
        result = max(result, odd, even, key=len)
    return result
```

**Why 2N - 1 centers?** N characters (odd-length centers) + N - 1 gaps (even-length centers).

*Socratic prompt: "Why do we need to check both odd-length and even-length palindromes separately? Can a single expansion handle both?"*

### Palindrome DP

For problems requiring **all** palindromic substrings (not just the longest), precompute a boolean table.

```python
def palindrome_dp(s):
    """Precompute dp[i][j] = True if s[i..j] is a palindrome. O(N^2)."""
    n = len(s)
    dp = [[False] * n for _ in range(n)]

    # Base case: single characters
    for i in range(n):
        dp[i][i] = True

    # Base case: two characters
    for i in range(n - 1):
        dp[i][i + 1] = (s[i] == s[i + 1])

    # Fill for lengths 3 to n
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = (s[i] == s[j]) and dp[i + 1][j - 1]

    return dp
```

**Use cases:** Palindrome Partitioning (LC 131, 132), Count Palindromic Substrings (LC 647).

### Longest Palindromic Subsequence

Different from substring — characters don't need to be contiguous. This is a classic DP problem on two sequences (the string and its reverse).

```python
def longest_palindrome_subseq(s):
    """O(N^2) time, O(N^2) space. Reduce to LCS(s, reverse(s))."""
    n = len(s)
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    rev = s[::-1]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if s[i - 1] == rev[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][n]
```

### Problems

| Problem | Key Technique |
|---------|--------------|
| Valid Palindrome (125) | Two pointers with alphanumeric filtering |
| Valid Palindrome II (680) | Two pointers + allow one deletion |
| Longest Palindromic Substring (5) | Expand from center |
| Palindromic Substrings (647) | Expand from center or DP |
| Palindrome Partitioning (131) | Backtracking + palindrome DP |
| Longest Palindromic Subsequence (516) | DP: LCS with reversed string |

---

## 4. Common String Algorithms

### KMP (Knuth-Morris-Pratt)

O(N + M) pattern matching using a **failure function** (partial match table) to avoid redundant comparisons. When a mismatch occurs, the failure function tells you how far back to restart the pattern comparison.

```python
def kmp_search(text, pattern):
    """O(N + M) time. Returns first match index or -1."""
    def build_lps(pattern):
        """Longest Proper Prefix which is also Suffix."""
        lps = [0] * len(pattern)
        length = 0
        i = 1
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            elif length:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
        return lps

    lps = build_lps(pattern)
    i = j = 0  # i for text, j for pattern
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == len(pattern):
                return i - j  # Match found
        elif j:
            j = lps[j - 1]  # Don't restart from beginning
        else:
            i += 1
    return -1
```

### Rabin-Karp

Rolling hash for O(N) average substring matching. See `references/array-techniques.md` for the full template and worked examples.

### When to Use Which

| Algorithm | Time | When to Prefer |
|-----------|------|---------------|
| Naive matching | O(N*M) | Short strings, simple implementation |
| KMP | O(N + M) | Guaranteed linear time, repeated searches with same pattern |
| Rabin-Karp | O(N) average | Multiple pattern matching, finding repeated substrings |

*Socratic prompt: "KMP and Rabin-Karp both achieve linear-time matching. What's the key difference in HOW they avoid redundant work?"*

---

## Essential & Recommended Practice Questions

### Essential (Do These First)

| Problem | Difficulty | Key Technique |
|---------|-----------|---------------|
| Valid Anagram (242) | Easy | Character frequency counting |
| Valid Palindrome (125) | Easy | Two pointers |
| Longest Substring Without Repeating Characters (3) | Medium | Sliding window + hash set |
| Find All Anagrams in a String (438) | Medium | Sliding window + frequency map |
| Longest Palindromic Substring (5) | Medium | Expand from center |
| String to Integer (atoi) (8) | Medium | State machine / careful parsing |
| Group Anagrams (49) | Medium | Hash map with sorted key |
| Longest Repeating Character Replacement (424) | Medium | Sliding window |

### Recommended (Build Depth)

| Problem | Difficulty | Key Technique |
|---------|-----------|---------------|
| Minimum Window Substring (76) | Hard | Sliding window with "need" tracking |
| Palindrome Partitioning (131) | Medium | Backtracking + palindrome check |
| Encode and Decode Strings (271) | Medium | Length prefix encoding |
| Palindromic Substrings (647) | Medium | Expand from center |
| Longest Palindromic Subsequence (516) | Medium | DP (LCS variant) |
| Shortest Palindrome (214) | Hard | KMP on `s + '#' + reverse(s)` |

---

## Pattern Connections

| If You Know... | Then You Can Solve... |
|----------------|----------------------|
| Sliding window (from `algorithm-frameworks.md`) | Substring problems with constraints (min window, longest without repeating) |
| Two pointers (from `algorithm-frameworks.md`) | Palindrome validation and detection |
| Hash map counting | Anagram problems, character frequency problems |
| DP subsequence template (from `dynamic-programming-core.md`) | Palindromic subsequence, edit distance with strings |
| Rabin-Karp (from `array-techniques.md`) | Pattern matching, repeated substring detection |

*Socratic prompt: "Most string problems reduce to either counting, sliding window, or two pointers. Given a new string problem, how would you decide which technique to try first?"*

---

## Attribution

The techniques and practice question recommendations in this file are adapted from the Tech Interview Handbook by Yangshun Tay (techinterviewhandbook.org). Templates have been restructured and annotated for Socratic teaching use.
