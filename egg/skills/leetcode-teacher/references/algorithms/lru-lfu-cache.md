# LRU Cache, LFU Cache & Random Set O(1)

Augmented data structure designs combining hash maps with linked lists or arrays for O(1) operations.

**Prerequisites:** Hash maps, doubly linked lists. See `references/data-structures/data-structure-fundamentals.md` for internals.

---

## LRU Cache

### Key Insight

Combine a hash map (O(1) key lookup) with a doubly linked list (O(1) insert/remove at both ends). Most recently used items go to the front; evict from the tail.

### Template

```python
class Node:
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = {}           # key -> Node
        self.head = Node()        # Dummy head (most recent)
        self.tail = Node()        # Dummy tail (least recent)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_front(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add_to_front(node)  # Mark as recently used
        return node.val

    def put(self, key, value):
        if key in self.cache:
            self._remove(self.cache[key])
        node = Node(key, value)
        self._add_to_front(node)
        self.cache[key] = node
        if len(self.cache) > self.cap:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]  # Need key stored in node for this
```

### Why Dummy Nodes?

Dummy head and tail eliminate all null-pointer checks for edge cases (empty list, single element). Every real node always has a valid `prev` and `next`.

---

## LFU Cache

### Key Insight

Maintain a frequency-to-list mapping. Each frequency has its own doubly linked list (ordered by recency). Track `min_freq` globally. Evict from the `min_freq` list's tail.

### Template

```python
from collections import defaultdict

class LFUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.min_freq = 0
        self.key_to_val = {}
        self.key_to_freq = {}
        self.freq_to_keys = defaultdict(list)  # freq -> list of keys (LRU order)
        self.key_to_pos = {}  # For O(1) removal, use OrderedDict in practice

    def _increase_freq(self, key):
        freq = self.key_to_freq[key]
        self.key_to_freq[key] = freq + 1
        # Remove from old frequency list
        self.freq_to_keys[freq].remove(key)
        if not self.freq_to_keys[freq]:
            del self.freq_to_keys[freq]
            if self.min_freq == freq:
                self.min_freq += 1
        # Add to new frequency list (most recent = end)
        self.freq_to_keys[freq + 1].append(key)

    def get(self, key):
        if key not in self.key_to_val:
            return -1
        self._increase_freq(key)
        return self.key_to_val[key]

    def put(self, key, value):
        if self.cap == 0:
            return
        if key in self.key_to_val:
            self.key_to_val[key] = value
            self._increase_freq(key)
            return
        if len(self.key_to_val) >= self.cap:
            # Evict from min_freq list (least recent = front)
            evict_key = self.freq_to_keys[self.min_freq].pop(0)
            if not self.freq_to_keys[self.min_freq]:
                del self.freq_to_keys[self.min_freq]
            del self.key_to_val[evict_key]
            del self.key_to_freq[evict_key]
        self.key_to_val[key] = value
        self.key_to_freq[key] = 1
        self.freq_to_keys[1].append(key)
        self.min_freq = 1  # New key always has freq=1
```

**Note:** For true O(1), use `OrderedDict` per frequency instead of plain lists.

### LRU vs LFU Comparison

| Property | LRU | LFU |
|----------|-----|-----|
| Eviction criterion | Least recently used | Least frequently used (ties broken by recency) |
| Data structures | HashMap + 1 DLL | HashMap + freq-to-DLL map + minFreq counter |
| `get` updates | Move to front of list | Increase frequency, move between lists |
| Implementation complexity | Moderate | High |

---

## Random Set O(1)

### Key Insight

To support `insert`, `delete`, and `getRandom` all in O(1), combine a **hash map** (key → index) with a **dynamic array** (values). The trick for O(1) deletion: swap the element to delete with the last element, then pop from the end.

### Template

```python
import random

class RandomizedSet:
    def __init__(self):
        self.val_to_idx = {}    # value -> index in self.vals
        self.vals = []           # values in arbitrary order

    def insert(self, val):
        if val in self.val_to_idx:
            return False
        self.val_to_idx[val] = len(self.vals)
        self.vals.append(val)
        return True

    def remove(self, val):
        if val not in self.val_to_idx:
            return False
        idx = self.val_to_idx[val]
        last = self.vals[-1]
        # Swap target with last element
        self.vals[idx] = last
        self.val_to_idx[last] = idx
        # Remove last element
        self.vals.pop()
        del self.val_to_idx[val]
        return True

    def getRandom(self):
        return random.choice(self.vals)
```

### Why the Swap-to-End Trick?

Arrays support O(1) random access (needed for `getRandom`) but O(N) arbitrary deletion. By swapping the target with the last element, deletion becomes O(1) — we only remove from the end.

*Socratic prompt: "Why can't you just use a hash set for getRandom? What operation does a hash set NOT support in O(1)?"*

**Example problems:** LRU Cache (146), LFU Cache (460), Insert Delete GetRandom O(1) (380), Insert Delete GetRandom O(1) - Duplicates (381)

---

## Attribution

Extracted from the LRU Cache, LFU Cache, and Random Set O(1) sections of `advanced-patterns.md`, inspired by labuladong's algorithmic guides (labuladong.online).
