# Data Structure Fundamentals

How data structures work under the hood — from first principles. Based on labuladong's "build-it-to-understand-it" approach.

---

## The Storage Duality

**Core insight:** All data structures use one of two physical storage methods at the hardware level:

1. **Sequential storage (array)** — contiguous memory, indexed by offset
2. **Linked storage (pointers)** — scattered memory, connected by references

Everything else — hash tables, trees, heaps, graphs, tries — is built on top of these two primitives.

| Storage | Strengths | Weaknesses |
|---------|-----------|------------|
| Array | O(1) random access, cache-friendly, compact | O(N) insert/delete, fixed size (or costly resize) |
| Linked | O(1) insert/delete at known position, dynamic size | O(N) access, pointer overhead, cache-unfriendly |

**Socratic prompt:** *"If you could only have arrays or linked lists — no other data structure — which problems would be easy? Which would be hard?"*

---

## Array Internals

### Why O(1) Access?

An array is a contiguous block of memory. Element `i` lives at `base_address + i * element_size`. The CPU computes this in constant time — no traversal needed.

```python
# Conceptually, array access is pointer arithmetic
def array_access(base_addr, index, elem_size):
    return memory[base_addr + index * elem_size]  # O(1)
```

### Static vs Dynamic Arrays

**Static array:** Fixed size at creation. Cannot grow.

**Dynamic array** (Python `list`, Java `ArrayList`): Starts with a capacity, doubles when full.

```python
# Simplified dynamic array resize
def append(self, val):
    if self.size == self.capacity:
        # Allocate new array with 2x capacity
        new_arr = [None] * (self.capacity * 2)
        for i in range(self.size):          # O(N) copy
            new_arr[i] = self.data[i]
        self.data = new_arr
        self.capacity *= 2
    self.data[self.size] = val              # O(1)
    self.size += 1
```

**Amortized analysis:** Most appends are O(1). Resizing happens every N inserts and costs O(N), so amortized cost per insert = O(1).

### Insert and Delete Cost

Inserting at index `i` requires shifting elements `[i, n-1]` right → O(N) worst case.
Deleting at index `i` requires shifting elements `[i+1, n-1]` left → O(N) worst case.

**Socratic prompt:** *"Why can't you just 'extend' an array when it's full? What does the memory layout prevent?"*

---

## Linked List Internals

### Node Structure

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next  # Pointer to next node

class DoublyLinkedNode:
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next
```

### Why O(1) Insert/Delete (at Known Position)?

Given a pointer to the node before the insertion point, inserting is just pointer reassignment — no shifting.

```python
# Insert new_node after prev_node — O(1)
new_node.next = prev_node.next
prev_node.next = new_node
```

But **finding** that position is O(N) — you must traverse from the head. This is why linked lists are O(N) access.

### Singly vs Doubly Linked

| Operation | Singly | Doubly |
|-----------|--------|--------|
| Access by index | O(N) | O(N) |
| Insert after known node | O(1) | O(1) |
| Delete known node | O(N) — need predecessor | O(1) — have prev pointer |
| Space per node | 1 pointer | 2 pointers |

**Socratic prompt:** *"When would you choose a linked list over an array? What if you need both fast access AND fast insertion?"*

---

## Stack & Queue

Both are abstract data types implementable with either arrays or linked lists.

### Stack (LIFO)

```python
# Array-based stack — push/pop from the end
class ArrayStack:
    def __init__(self):
        self.data = []
    def push(self, val): self.data.append(val)    # O(1) amortized
    def pop(self): return self.data.pop()          # O(1)
    def peek(self): return self.data[-1]           # O(1)

# Linked-list-based stack — push/pop from the head
class LinkedStack:
    def __init__(self):
        self.head = None
    def push(self, val):
        node = ListNode(val, self.head)
        self.head = node                           # O(1)
    def pop(self):
        val = self.head.val
        self.head = self.head.next                 # O(1)
        return val
```

### Queue (FIFO)

```python
# Array-based queue uses a circular buffer or deque
from collections import deque

class ArrayQueue:
    def __init__(self):
        self.data = deque()
    def enqueue(self, val): self.data.append(val)       # O(1) amortized
    def dequeue(self): return self.data.popleft()        # O(1)
```

**Deque (double-ended queue):** Supports push/pop from both ends in O(1). Generalizes both stack and queue. Python's `collections.deque` is a doubly-linked block list.

**Socratic prompt:** *"If a stack is LIFO and a queue is FIFO, what real-world processes behave like each?"*

---

## Hash Table Internals

### The Core Idea

Map keys to array indices via a **hash function**: `index = hash(key) % array_size`.

```python
# Simplified hash table
class SimpleHashTable:
    def __init__(self, capacity=16):
        self.capacity = capacity
        self.buckets = [[] for _ in range(capacity)]  # Chaining

    def _hash(self, key):
        return hash(key) % self.capacity

    def put(self, key, value):
        idx = self._hash(key)
        for i, (k, v) in enumerate(self.buckets[idx]):
            if k == key:
                self.buckets[idx][i] = (key, value)   # Update
                return
        self.buckets[idx].append((key, value))         # Insert

    def get(self, key):
        idx = self._hash(key)
        for k, v in self.buckets[idx]:
            if k == key:
                return v
        return None
```

### Collision Resolution

When two keys hash to the same index:

**Chaining (separate chaining):** Each bucket holds a linked list (or array) of entries. Simple but uses extra memory.

**Open addressing (linear probing):** If slot is taken, check the next slot. More cache-friendly but degrades under high load.

| Method | Pros | Cons |
|--------|------|------|
| Chaining | Simple, handles high load gracefully | Pointer overhead, cache-unfriendly |
| Open addressing | Cache-friendly, no extra pointers | Clustering, performance degrades near full |

### Load Factor and Resizing

**Load factor** = `num_entries / array_size`. When it exceeds a threshold (typically 0.75), the table resizes — allocate a new larger array and rehash all entries. This is O(N) but amortized O(1) per operation.

### Why O(1) Average but O(N) Worst Case?

With a good hash function and low load factor, each bucket has ~1 entry → O(1) lookup. With a bad hash function or high load factor, all entries could land in one bucket → O(N) linear scan.

**Why traversal order is unreliable:** Iteration order depends on hash values and table size, both of which change on resize. Never depend on dict iteration order for correctness (though Python 3.7+ guarantees insertion order as an implementation detail).

**Socratic prompt:** *"If hash tables are O(1), why don't we use them for everything? What are the hidden costs?"*

### Language Implementations

| Language | Hash Map | Hash Set | Ordered Map |
|----------|---------|---------|-------------|
| C++ | `std::unordered_map` | `std::unordered_set` | `std::map` (red-black tree) |
| Java | `HashMap` | `HashSet` | `TreeMap` (red-black tree) |
| Python | `dict` | `set` | `collections.OrderedDict` |
| JavaScript | `Map` / `Object` | `Set` | `Map` (insertion order) |
| Go | `map[K]V` | — (use `map[K]bool`) | — |

**Interview tip:** Know the difference between O(1) **average** and O(1) **amortized**. Hash table operations are O(1) on average (assuming good hash function and low load factor), but a single operation can be O(N) when rehashing occurs. Amortized O(1) means the expensive rehash is "paid for" by the many cheap operations that preceded it.

### Hash Table Interview Questions

- **Design an LRU Cache (LC 146):** Hash map (O(1) lookup) + doubly-linked list (O(1) eviction) — the canonical augmented data structure. See `references/advanced-patterns.md`.
- **Hash map with linked list buckets:** Implement `put`, `get`, `remove` with chaining collision resolution. Tests understanding of the hash table internals above.
- **Implement a hash set without using built-in hash libraries (LC 705):** Choose bucket count, hash function, and collision strategy.

### Essential & Recommended Practice Questions for Hash Tables

| Problem | Difficulty | Key Twist |
|---------|-----------|-----------|
| Two Sum (1) | Easy | Hash map for complement lookup |
| Ransom Note (383) | Easy | Character frequency counting |
| LRU Cache (146) | Medium | Hash map + doubly-linked list |
| Group Anagrams (49) | Medium | Sorted string or frequency tuple as key |
| Insert Delete GetRandom O(1) (380) | Medium | Hash map + array for random access |
| Longest Consecutive Sequence (128) | Medium | Hash set for O(1) membership check |

---

## Binary Tree Centrality

**Labuladong's thesis:** Binary trees are THE most important data structure. Not just one data structure among many — they are the mental model that unlocks everything else.

### All Advanced Data Structures Are Tree Extensions

| Data Structure | Tree Connection |
|----------------|----------------|
| BST (Binary Search Tree) | Binary tree + left < root < right ordering |
| Balanced BST (AVL, Red-Black) | BST + height-balance invariant |
| Heap / Priority Queue | Complete binary tree + heap property |
| Trie | Multi-way tree where each edge = character |
| Segment Tree | Binary tree where each node = interval aggregate |
| B-Tree / B+ Tree | Multi-way balanced tree for disk storage |
| Graph | Generalized tree (tree = connected acyclic graph) |

### All Brute-Force Algorithms Are Tree Problems

| Algorithm | Implicit Tree |
|-----------|--------------|
| Backtracking | Decision tree — each node = choice, children = options |
| BFS | Level-order traversal of the state-space tree |
| Dynamic Programming | Recursion tree with overlapping subtrees (memoize to DAG) |
| Divide and Conquer | Binary tree — split, recurse, combine |

**Key insight:** If you can solve binary tree problems fluently, you can solve anything. Binary tree traversal is the skeleton; everything else is filling in the blanks.

### Binary Tree Types

| Type | Definition | Why It Matters |
|------|-----------|---------------|
| **Perfect** (full) binary tree | Every level fully filled | Theoretical ideal, 2^h - 1 nodes |
| **Complete** binary tree | All levels filled except possibly last, filled left to right | Can be stored in array (heap), parent(i) = i//2, children(i) = 2i, 2i+1 |
| **BST** | left.val < node.val < right.val for all nodes | O(log N) search/insert/delete when balanced |
| **Height-balanced** (AVL) | |height(left) - height(right)| ≤ 1 for all nodes | Guarantees O(log N) operations |
| **Self-balancing** (Red-Black, etc.) | Maintains approximate balance with rotations | Practical balanced BST used in language libraries |

### Binary Tree Traversal: The Three Positions

Every recursive traversal visits each node exactly once. The code you place at three positions determines what the traversal does:

```python
def traverse(node):
    if not node:
        return
    # PRE-ORDER position: before visiting children
    #   → "entering" the node (process on the way down)
    traverse(node.left)
    # IN-ORDER position: between left and right children
    #   → for BSTs, this visits nodes in sorted order
    traverse(node.right)
    # POST-ORDER position: after visiting both children
    #   → "leaving" the node (process on the way up)
```

**The key insight:** The traversal ORDER is always the same (left subtree → right subtree). What changes is WHERE you place your code — pre-order, in-order, or post-order.

**Level-order traversal** uses BFS (queue) instead of recursion. Three common BFS methods:
1. Basic queue (no level separation)
2. Queue with level size tracking (`for _ in range(len(queue))`)
3. Queue with sentinel/delimiter between levels

**Socratic prompt:** *"If binary trees are so fundamental, what's the simplest binary tree problem you can think of? Can you solve it using each traversal order?"*

---

## Binary Tree Advanced Operations

### Serialize / Deserialize Binary Tree (LC 297)

**Key insight:** Serialize using preorder traversal with null markers. Deserialize by reading tokens in the same order.

#### Preorder Serialization

```python
class Codec:
    def serialize(self, root):
        """Preorder: node, left, right. Use '#' for null."""
        result = []
        def dfs(node):
            if not node:
                result.append('#')
                return
            result.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
        dfs(root)
        return ','.join(result)

    def deserialize(self, data):
        tokens = iter(data.split(','))
        def dfs():
            val = next(tokens)
            if val == '#':
                return None
            node = TreeNode(int(val))
            node.left = dfs()
            node.right = dfs()
            return node
        return dfs()
```

#### Level-Order Serialization

```python
from collections import deque

class CodecBFS:
    def serialize(self, root):
        if not root:
            return ''
        result = []
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if node:
                result.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                result.append('#')
        return ','.join(result)

    def deserialize(self, data):
        if not data:
            return None
        tokens = data.split(',')
        root = TreeNode(int(tokens[0]))
        queue = deque([root])
        i = 1
        while queue:
            node = queue.popleft()
            if tokens[i] != '#':
                node.left = TreeNode(int(tokens[i]))
                queue.append(node.left)
            i += 1
            if tokens[i] != '#':
                node.right = TreeNode(int(tokens[i]))
                queue.append(node.right)
            i += 1
        return root
```

*Socratic prompt: "Why do we need null markers in preorder serialization? Could we serialize without them?"*

### Lowest Common Ancestor (LC 236)

**Key insight:** A node is the LCA if `p` and `q` are in different subtrees, or if the node itself is `p` or `q`.

```python
def lowest_common_ancestor(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    if left and right:
        return root          # p and q are in different subtrees
    return left or right     # Both are in one subtree
```

### Count Complete Tree Nodes (LC 222) — O(log^2 N)

**Key insight:** A complete binary tree has 2^h - 1 nodes if it's perfect. Check if left and right heights are equal (perfect subtree) to skip counting.

```python
def count_nodes(root):
    if not root:
        return 0
    left_h = right_h = 0
    l, r = root, root
    while l:
        left_h += 1
        l = l.left
    while r:
        right_h += 1
        r = r.right
    if left_h == right_h:
        return 2 ** left_h - 1          # Perfect tree shortcut
    return 1 + count_nodes(root.left) + count_nodes(root.right)
```

### Flatten Nested List Iterator (LC 341)

**Stack approach:** Push elements in reverse order so the first element is on top.

```python
class NestedIterator:
    def __init__(self, nestedList):
        self.stack = list(reversed(nestedList))

    def next(self):
        return self.stack.pop().getInteger()

    def hasNext(self):
        while self.stack:
            top = self.stack[-1]
            if top.isInteger():
                return True
            self.stack.pop()
            self.stack.extend(reversed(top.getList()))
        return False
```

### Iterative Traversal (Stack-Based)

For environments where recursion depth is limited, use an explicit stack:

```python
def inorder_iterative(root):
    result = []
    stack = []
    curr = root
    while curr or stack:
        while curr:                     # Push all left children
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        result.append(curr.val)         # Visit (in-order position)
        curr = curr.right
    return result
```

**Morris traversal** achieves O(1) space by temporarily threading right pointers, but is rarely needed in interviews. Mention it for completeness.

---

## BST Operations

### BST Property

For every node: all values in the left subtree < node.val < all values in the right subtree.

### Search

```python
def search_bst(root, val):
    if not root or root.val == val:
        return root
    if val < root.val:
        return search_bst(root.left, val)
    return search_bst(root.right, val)
```

### Insert

```python
def insert_bst(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insert_bst(root.left, val)
    elif val > root.val:
        root.right = insert_bst(root.right, val)
    return root
```

### Delete

Three cases: leaf (remove), one child (replace with child), two children (replace with in-order successor).

```python
def delete_bst(root, key):
    if not root:
        return None
    if key < root.val:
        root.left = delete_bst(root.left, key)
    elif key > root.val:
        root.right = delete_bst(root.right, key)
    else:
        # Found the node to delete
        if not root.left:
            return root.right
        if not root.right:
            return root.left
        # Two children: replace with in-order successor (smallest in right subtree)
        successor = root.right
        while successor.left:
            successor = successor.left
        root.val = successor.val
        root.right = delete_bst(root.right, successor.val)
    return root
```

### Validate BST (LC 98)

Pass min/max bounds down the tree:

```python
def is_valid_bst(root, lo=float('-inf'), hi=float('inf')):
    if not root:
        return True
    if root.val <= lo or root.val >= hi:
        return False
    return (is_valid_bst(root.left, lo, root.val) and
            is_valid_bst(root.right, root.val, hi))
```

### Kth Smallest in BST (LC 230)

In-order traversal of a BST visits nodes in sorted order. The kth visited node is the answer.

```python
def kth_smallest(root, k):
    stack = []
    curr = root
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        k -= 1
        if k == 0:
            return curr.val
        curr = curr.right
```

### Convert BST to Greater Tree (LC 538)

**Reverse in-order** (right → node → left) visits nodes in descending order. Accumulate a running sum.

```python
def convert_bst(root):
    running_sum = 0
    def reverse_inorder(node):
        nonlocal running_sum
        if not node:
            return
        reverse_inorder(node.right)
        running_sum += node.val
        node.val = running_sum
        reverse_inorder(node.left)
    reverse_inorder(root)
    return root
```

*Socratic prompt: "BST in-order gives ascending order. What does reverse in-order give? How does that help compute the greater-tree sum?"*

### Construct BST from Preorder and Inorder (LC 105)

```python
def build_tree(preorder, inorder):
    if not preorder:
        return None
    root_val = preorder[0]
    root = TreeNode(root_val)
    mid = inorder.index(root_val)
    root.left = build_tree(preorder[1:mid + 1], inorder[:mid])
    root.right = build_tree(preorder[mid + 1:], inorder[mid + 1:])
    return root
```

**Example problems:** Search in BST (700), Insert into BST (701), Delete Node in BST (450), Validate BST (98), Kth Smallest (230), Convert BST to Greater Tree (538), Construct from Preorder + Inorder (105)

---

## Segment Tree

### When to Use

When you need both **range queries** (sum, min, max over a subarray) and **point/range updates**, and need both in O(log N).

| Operation | Naive Array | Prefix Sum | Segment Tree |
|-----------|-------------|------------|--------------|
| Point update | O(1) | O(N) rebuild | O(log N) |
| Range query | O(N) | O(1) | O(log N) |
| Range update | O(N) | O(N) | O(log N) with lazy propagation |

### Array-Based Implementation

Store the tree in a flat array of size 4N. Node `i` has children at `2*i` and `2*i+1`.

```python
class SegmentTree:
    def __init__(self, nums):
        self.n = len(nums)
        self.tree = [0] * (4 * self.n)
        self._build(nums, 1, 0, self.n - 1)

    def _build(self, nums, node, start, end):
        if start == end:
            self.tree[node] = nums[start]
            return
        mid = (start + end) // 2
        self._build(nums, 2 * node, start, mid)
        self._build(nums, 2 * node + 1, mid + 1, end)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def update(self, idx, val, node=1, start=0, end=None):
        """Point update: set nums[idx] = val."""
        if end is None:
            end = self.n - 1
        if start == end:
            self.tree[node] = val
            return
        mid = (start + end) // 2
        if idx <= mid:
            self.update(idx, val, 2 * node, start, mid)
        else:
            self.update(idx, val, 2 * node + 1, mid + 1, end)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, l, r, node=1, start=0, end=None):
        """Range sum query: sum of nums[l..r]."""
        if end is None:
            end = self.n - 1
        if l > end or r < start:
            return 0                    # Out of range
        if l <= start and end <= r:
            return self.tree[node]      # Fully within range
        mid = (start + end) // 2
        return (self.query(l, r, 2 * node, start, mid) +
                self.query(l, r, 2 * node + 1, mid + 1, end))
```

### Lazy Propagation (Overview)

For **range updates** (e.g., add val to all elements in `[l, r]`), defer updates to children until they're actually queried. Store a "lazy" value at each node that represents a pending operation.

*Socratic prompt: "Without lazy propagation, a range update is O(N) because we update every leaf. How does deferring the update to children save work?"*

**Example problems:** Range Sum Query - Mutable (307), Count of Smaller Numbers After Self (315)

---

## Other Specialized Structures

### Huffman Tree (Greedy Construction)

**Problem:** Assign variable-length binary codes to characters to minimize total encoding length. Frequent characters get shorter codes.

**Algorithm:** Repeatedly merge the two nodes with the smallest frequencies. This greedy approach produces an optimal prefix-free code.

```python
import heapq

def huffman_codes(freq):
    """freq: dict of char -> frequency. Returns dict of char -> binary code."""
    heap = [(f, c) for c, f in freq.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        return {heap[0][1]: '0'}

    # Build tree: merge two smallest repeatedly
    while len(heap) > 1:
        f1, left = heapq.heappop(heap)
        f2, right = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, (left, right)))

    # Traverse tree to assign codes
    codes = {}
    def traverse(node, code):
        if isinstance(node, str):
            codes[node] = code
            return
        traverse(node[0], code + '0')
        traverse(node[1], code + '1')

    traverse(heap[0][1], '')
    return codes
```

### Consistent Hashing (Hash Ring)

**Problem:** Distribute keys across servers such that adding/removing a server only remaps ~1/N of the keys (vs rehashing everything).

**Idea:** Place servers on a circular hash ring. Each key maps to the next server clockwise. Use **virtual nodes** (multiple points per server) for better balance.

This is a systems design concept rather than a LeetCode problem, but understanding it deepens your grasp of hash-based data structures.

---

## Graph Fundamentals

### Representations

**Adjacency list:** Each node stores a list of its neighbors. Space-efficient for sparse graphs.

```python
# Adjacency list — O(V + E) space
graph = {
    0: [1, 2],
    1: [0, 3],
    2: [0],
    3: [1]
}
# Or with weights:
graph = {
    0: [(1, 5), (2, 3)],   # (neighbor, weight)
}
```

**Adjacency matrix:** 2D boolean/weight matrix. O(1) edge lookup but O(V^2) space.

```python
# Adjacency matrix — O(V^2) space
matrix = [
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0]
]
```

| Representation | Space | Check Edge | Get Neighbors | Best For |
|----------------|-------|-----------|---------------|----------|
| Adjacency list | O(V + E) | O(degree) | O(1) | Sparse graphs (most real-world) |
| Adjacency matrix | O(V^2) | O(1) | O(V) | Dense graphs, edge-weight lookups |

### Three Graph Traversal Modes

Labuladong identifies three distinct traversal patterns, each using `visited` differently:

**Mode 1: Traverse Nodes** — standard DFS/BFS, visit each node once.

```python
visited = set()  # Boolean per node

def traverse_nodes(graph, node):
    if node in visited:
        return
    visited.add(node)
    for neighbor in graph[node]:
        traverse_nodes(graph, neighbor)
```

**Mode 2: Traverse Edges** — visit each edge once (useful when parallel edges matter).

```python
visited = set()  # Track (from, to) pairs

def traverse_edges(graph, node):
    for neighbor in graph[node]:
        if (node, neighbor) not in visited:
            visited.add((node, neighbor))
            traverse_edges(graph, neighbor)
```

**Mode 3: Traverse Paths** — track the current path for cycle detection or path enumeration.

```python
on_path = set()  # Nodes on the CURRENT path (backtrack when leaving)

def traverse_paths(graph, node):
    if node in on_path:
        # Cycle detected!
        return
    on_path.add(node)
    for neighbor in graph[node]:
        traverse_paths(graph, neighbor)
    on_path.remove(node)  # Backtrack — leaving this path
```

**Why the distinction matters:**
- `visited` (Mode 1) prevents re-visiting nodes — sufficient for trees and simple graph traversal
- `on_path` (Mode 3) tracks the current recursion stack — needed for **cycle detection** in directed graphs (topological sort, course schedule)
- The difference between `visited` and `on_path`: a node can be visited but not on the current path (already explored via a different branch)

### Graph = Tree + Cycles

A tree is a connected graph with no cycles. Graph traversal is tree traversal with an extra `visited` check to handle cycles. This is why mastering tree traversal first makes graph problems approachable.

**Complexity:** Graph traversal is O(V + E), not just O(V), because you examine every edge.

**Socratic prompt:** *"If you removed the `visited` check from a graph DFS, what would happen? Why doesn't tree traversal need a visited set?"*

---

## Advanced Data Structures (Conceptual)

Brief overviews of structures learners encounter in advanced problems. The goal is conceptual understanding, not implementation mastery.

### Skip List

**Problem:** Linked lists are O(N) for search. Can we do better without switching to an array?

**Idea:** Add multiple layers of "express" pointers. Each layer skips more elements. Search starts at the top layer and drops down — like binary search on a linked list.

```
Level 3: 1 ────────────────────── 9
Level 2: 1 ────── 5 ────────── 9
Level 1: 1 ── 3 ── 5 ── 7 ── 9
Level 0: 1  2  3  4  5  6  7  8  9
```

**Result:** O(log N) expected search, insert, delete. Used in Redis sorted sets.

### Bloom Filter

**Problem:** Check if an element is in a set, using minimal memory.

**Idea:** Use k hash functions, each mapping to a bit in a bit array. To insert, set all k bits. To query, check all k bits — if any is 0, definitely not in the set. If all are 1, *probably* in the set (false positives possible, false negatives impossible).

**Trade-off:** Space-efficient probabilistic membership testing. No deletions (counting bloom filter variant allows it).

### Segment Tree

**Problem:** Answer range queries (sum, min, max) over an array, with point or range updates.

**Idea:** Binary tree where each node stores the aggregate for an interval. Leaf = single element. Internal node = combination of children's intervals.

**Result:** O(log N) range query and update, vs O(N) for naive approach.

### Trie (Prefix Tree)

**Problem:** Efficiently store and search strings, especially with shared prefixes.

**Idea:** Multi-way tree where each edge represents a character. Path from root to any node = a prefix. Shared prefixes share tree nodes.

```python
class TrieNode:
    def __init__(self):
        self.children = {}      # char -> TrieNode
        self.is_end = False     # Marks end of a complete word
```

**Result:** O(L) search/insert where L = string length. Used in autocomplete, spell-check, IP routing.

**Socratic prompt:** *"For each of these structures — what problem does it solve that simpler structures can't? What does it sacrifice to solve that problem?"*

---

## Attribution

The frameworks in this file are inspired by and adapted from labuladong's algorithmic guides (labuladong.online), particularly the "Getting Started: Data Structures" curriculum and Chapter 1 "Data Structure Algorithms." The storage duality, binary tree centrality thesis, graph traversal modes, BST operations, tree serialization/deserialization, LCA, segment tree, and Huffman tree sections have been restructured and annotated for Socratic teaching use.
