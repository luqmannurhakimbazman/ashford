# Linked List Problem-Solving Techniques

Six core patterns for linked list problems. These cover the vast majority of interview linked list questions.

---

## Types of Linked Lists

- **Singly linked list:** Each node has a `val` and a `next` pointer. Traversal is one-directional (head to tail).
- **Doubly linked list:** Each node has `val`, `next`, and `prev` pointers. Traversal is bidirectional. Used in LRU Cache, browser history.
- **Circular linked list:** The tail node points back to the head (singly or doubly). Used in round-robin scheduling, circular buffers.

## Time Complexity

| Operation | Singly Linked List | Notes |
|-----------|-------------------|-------|
| Access | O(n) | Must traverse from head |
| Search | O(n) | Must traverse from head |
| Insert | O(1)* | *At a known position (given pointer to predecessor) |
| Remove | O(1)* | *At a known position (given pointer to predecessor) |

## Language Implementations

| Language | Implementation | Notes |
|----------|---------------|-------|
| C++ | `std::list` (doubly), `std::forward_list` (singly) | STL containers |
| Java | `java.util.LinkedList` (doubly) | Implements List and Deque |
| Python | No built-in linked list | Use `collections.deque` for deque behavior |

## Corner Cases

- Empty list (head is null)
- Single node
- Two nodes
- Linked list has cycles (use Floyd's cycle detection — see Pattern 1)

## Interview Tips

- Linked lists rarely have in-place modification issues like arrays. Many operations (combine two lists, swap node values, truncate) are simple and elegant with pointer manipulation.
- Interviewers prefer **in-place** linked list solutions (O(1) extra space). Avoid copying to an array and back.
- Always clarify: singly or doubly linked? Circular? Are there duplicates?

## Essential & Recommended Practice Questions

| Problem | Difficulty | Key Technique |
|---------|-----------|---------------|
| Linked List Cycle (141) | Easy | Floyd's fast-slow pointers |
| Reverse Linked List (206) | Easy | Three-pointer iterative reversal |
| Merge Two Sorted Lists (21) | Easy | Dummy node + zipper merge |
| Remove Nth Node From End (19) | Medium | Two-pointer with gap + dummy |
| Reorder List (143) | Medium | Find middle + reverse + merge |
| Merge K Sorted Lists (23) | Hard | Min-heap extension |

---

## Pattern Selection Decision Tree

```
What does the problem ask?
├── Detect or locate a cycle?
│   └── Fast-Slow Pointers (Floyd's)
├── Find the middle node?
│   └── Fast-Slow Pointers (slow = 1 step, fast = 2 steps)
├── Find kth node from end?
│   └── Two-Pointer with Gap
├── Merge sorted lists?
│   └── Dummy Node + Zipper Merge
├── Remove/delete nodes?
│   ├── Head might change? → Dummy Node
│   └── Kth from end? → Two-Pointer with Gap + Dummy
├── Reverse (all or part)?
│   └── Three-Pointer Reversal (iterative or recursive)
├── Check palindrome?
│   └── Fast-Slow (find middle) + Reverse second half
├── Find intersection of two lists?
│   └── A+B / B+A Concatenation Trick
└── Partition or reorder?
    └── Dummy Node (often two dummy nodes for two sublists)
```

---

## 1. Fast-Slow Pointers (Floyd's Algorithm)

**Core idea:** Two pointers traverse the list at different speeds. The speed difference creates useful mathematical properties.

### Cycle Detection

If a cycle exists, a fast pointer (2 steps) will eventually meet a slow pointer (1 step) inside the cycle. Why? Once both are in the cycle, the gap closes by exactly 1 node per iteration.

```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next          # 1 step
        fast = fast.next.next     # 2 steps
        if slow == fast:
            return True
    return False                  # fast hit the end — no cycle
```

*Socratic prompt: "If slow moves 1 step and fast moves 2 steps per iteration, how does the distance between them change each iteration when both are inside a cycle?"*

### Find Cycle Start

After detecting a meeting point, reset one pointer to `head`. Move both at 1 step. They meet at the cycle entrance.

**Why it works:** Let `k` = distance from head to cycle start. When slow enters the cycle, fast is `k` steps ahead inside the cycle. They meet after slow travels some distance inside the cycle. Resetting one to head and walking at equal speed means both travel exactly `k` steps to the cycle start.

```python
def detect_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            # Phase 2: find entrance
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow           # cycle start node
    return None
```

### Find Middle Node

When fast reaches the end, slow is at the middle.

```python
def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow  # For even-length lists, returns the second middle node
```

*Socratic prompt: "If the list has 6 nodes, how many steps does fast take before hitting the end? Where is slow at that point?"*

**Example problems:** Linked List Cycle (141), Linked List Cycle II (142), Middle of the Linked List (876), Happy Number (202)

---

## 2. Dummy Node Trick

**Core idea:** Create a fake head node before the real head. This eliminates all edge cases where the head itself might be removed, changed, or is unknown at the start.

```python
def remove_elements(head, val):
    dummy = ListNode(0)
    dummy.next = head             # dummy points to real head
    curr = dummy
    while curr.next:
        if curr.next.val == val:
            curr.next = curr.next.next  # skip the node
        else:
            curr = curr.next
    return dummy.next             # new head (might differ from original)
```

### When to Use

- **Removing nodes** — the head might be removed
- **Merging lists** — you don't know which list provides the first node
- **Partitioning** — splitting into two sublists (use two dummy nodes)

### Two-Dummy Pattern (Partition)

```python
def partition(head, x):
    # Two sublists: nodes < x and nodes >= x
    small_dummy = ListNode(0)
    large_dummy = ListNode(0)
    small = small_dummy
    large = large_dummy

    while head:
        if head.val < x:
            small.next = head
            small = small.next
        else:
            large.next = head
            large = large.next
        head = head.next

    large.next = None             # Terminate the large list
    small.next = large_dummy.next # Connect small → large
    return small_dummy.next
```

*Socratic prompt: "What goes wrong if you try to remove nodes without a dummy and the first node itself needs removal?"*

**Example problems:** Remove Linked List Elements (203), Merge Two Sorted Lists (21), Partition List (86)

---

## 3. Reverse Linked List

**Core idea:** Redirect each node's `next` pointer to point backward. Requires tracking three pointers: `prev`, `curr`, `next`.

### Iterative Reversal

```python
def reverse_list(head):
    prev = None
    curr = head
    while curr:
        next_node = curr.next     # Save next before overwriting
        curr.next = prev          # Reverse the pointer
        prev = curr               # Advance prev
        curr = next_node          # Advance curr
    return prev                   # prev is the new head
```

### Recursive Reversal

```python
def reverse_list_recursive(head):
    if not head or not head.next:
        return head               # Base: empty or single node
    new_head = reverse_list_recursive(head.next)
    # Post-order: head.next is now the TAIL of the reversed sublist
    head.next.next = head         # Make the next node point back to us
    head.next = None              # We become the new tail
    return new_head               # new_head stays the same throughout
```

*Socratic prompt: "In the iterative version, why do we need three pointers? What happens if we skip saving `next_node`?"*

### Reverse a Sublist (Between Positions)

```python
def reverse_between(head, left, right):
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy

    # Move prev to the node before the reversal starts
    for _ in range(left - 1):
        prev = prev.next

    # Reverse the sublist using the standard 3-pointer technique
    curr = prev.next
    for _ in range(right - left):
        next_node = curr.next
        curr.next = next_node.next
        next_node.next = prev.next
        prev.next = next_node

    return dummy.next
```

### Reverse in K-Groups

```python
def reverse_k_group(head, k):
    dummy = ListNode(0)
    dummy.next = head
    prev_group = dummy

    while True:
        # Check if k nodes remain
        kth = prev_group
        for _ in range(k):
            kth = kth.next
            if not kth:
                return dummy.next

        # Reverse k nodes
        group_start = prev_group.next
        curr = group_start
        prev = None
        for _ in range(k):
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node

        # Connect reversed group back to the list
        prev_group.next = prev           # prev is new group head
        group_start.next = curr          # group_start is now group tail
        prev_group = group_start         # move prev_group for next iteration
```

### Recursive Reversal Deep Dive

The recursive approach reveals a powerful insight: **recursion reaches the end first, then builds the reversed list on the way back** (postorder processing).

#### Recursion Tree for `reverse_list_recursive([1 → 2 → 3 → 4])`

```
reverse(1→2→3→4)
  └── reverse(2→3→4)
        └── reverse(3→4)
              └── reverse(4)        ← base case: return 4
            head=3, head.next=4
            4.next = 3, 3.next = None   → 4→3
            return 4
        head=2, head.next=3
        3.next = 2, 2.next = None      → 4→3→2
        return 4
    head=1, head.next=2
    2.next = 1, 1.next = None          → 4→3→2→1
    return 4
```

**Key insight:** `new_head` is always 4 (the original tail). It gets passed up unchanged. The actual pointer rewiring happens at the `head.next.next = head` line, which makes each node's successor point back to it.

#### Reverse First N Nodes (`reverseN`)

The **successor pointer technique**: when reversing only the first N nodes, the Nth node must point to the (N+1)th node (the "successor"), not to `None`.

```python
successor = None  # The node after the reversed portion

def reverse_n(head, n):
    global successor
    if n == 1:
        successor = head.next     # Record the (n+1)th node
        return head

    new_head = reverse_n(head.next, n - 1)
    head.next.next = head
    head.next = successor         # Connect to successor, not None
    return new_head
```

*Socratic prompt: "In full reversal, the original head becomes the tail and points to None. In reverseN, what should the original head point to instead? Why?"*

#### Reverse Sublist from Position m to n (`reverseBetween` — Recursive)

Walk to position `m` first, then apply `reverseN`:

```python
def reverse_between_recursive(head, m, n):
    if m == 1:
        return reverse_n(head, n)     # Reverse first n nodes
    # Move forward — reduce m and n (problem shrinks by 1)
    head.next = reverse_between_recursive(head.next, m - 1, n - 1)
    return head
```

**Why this works:** By recursing with `m - 1, n - 1`, we effectively "shift the window" until `m == 1`, at which point we're at the start of the sublist and can apply `reverseN`.

*Socratic prompt: "We reduce both m and n by 1 each recursion. What invariant does this maintain? What happens when m reaches 1?"*

**Example problems:** Reverse Linked List (206), Reverse Linked List II (92), Reverse Nodes in k-Group (25), Palindrome Linked List (234)

---

## 4. Merge Two Sorted Lists

**Core idea:** Use a dummy node and a "zipper" that always picks the smaller current node.

```python
def merge_two_lists(l1, l2):
    dummy = ListNode(0)
    curr = dummy

    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next

    curr.next = l1 or l2          # Attach the remaining nodes
    return dummy.next
```

### Merge K Sorted Lists (Min-Heap Extension)

```python
import heapq

def merge_k_lists(lists):
    dummy = ListNode(0)
    curr = dummy
    heap = []

    # Initialize heap with first node of each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst.val, i, lst))

    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next
```

**Why the index `i`?** Python's heapq compares tuples element by element. If two nodes have equal values, it compares `ListNode` objects (which aren't comparable). The index `i` breaks ties.

*Socratic prompt: "Merging K lists one pair at a time is O(NK). How does the heap approach reduce this? What's in the heap at any moment?"*

**Example problems:** Merge Two Sorted Lists (21), Merge K Sorted Lists (23), Sort List (148)

---

## 5. Find Intersection

**Core idea:** If list A has length `a` and list B has length `b`, a pointer walking `a + b` steps and another walking `b + a` steps will meet at the intersection (or both reach `None` if no intersection).

**Why it works:** The total distance is the same for both pointers (`a + b = b + a`). The difference in prefix lengths cancels out. After switching lists, both pointers are the same distance from the intersection node.

```python
def get_intersection_node(headA, headB):
    pA, pB = headA, headB
    while pA != pB:
        pA = pA.next if pA else headB   # Switch to other list at end
        pB = pB.next if pB else headA
    return pA  # Either intersection node or None (both hit None together)
```

*Socratic prompt: "List A has 3 unique nodes then 4 shared nodes. List B has 5 unique nodes then 4 shared nodes. If pointer A walks A then B, and pointer B walks B then A, after how many steps do they meet?"*

**Example problems:** Intersection of Two Linked Lists (160)

---

## 6. Two-Pointer Deletion (Kth from End)

**Core idea:** Advance the fast pointer `k` steps ahead. Then move both at the same speed. When fast reaches the end, slow is at the kth node from the end.

```python
def remove_nth_from_end(head, n):
    dummy = ListNode(0)
    dummy.next = head
    fast = slow = dummy

    # Advance fast by n+1 steps (so slow lands BEFORE the target)
    for _ in range(n + 1):
        fast = fast.next

    # Move both until fast hits the end
    while fast:
        fast = fast.next
        slow = slow.next

    slow.next = slow.next.next    # Skip the target node
    return dummy.next
```

**Why `n + 1` steps?** We want `slow` to stop one node BEFORE the node to remove, so we can rewire `slow.next`. The dummy node handles the edge case where the head itself is removed.

*Socratic prompt: "Why do we use a dummy node here? What happens if n equals the length of the list?"*

**Example problems:** Remove Nth Node From End of List (19), Kth Smallest Element (variant)

---

## Common Combinations

Many linked list problems combine multiple patterns:

| Problem | Patterns Used |
|---------|--------------|
| Palindrome Linked List (234) | Fast-Slow (find middle) + Reverse (second half) + Compare |
| Reorder List (143) | Fast-Slow (find middle) + Reverse (second half) + Merge (interleave) |
| Sort List (148) | Fast-Slow (find middle) + Merge Sort (recursive split + merge) |
| Add Two Numbers II (445) | Reverse + Add + Reverse (or use stack) |
| Swap Nodes in Pairs (24) | Dummy Node + iterative pointer swaps |
| Copy List with Random Pointer (138) | Hash Map (old→new mapping) or interleaving trick |

---

## Linked List vs Array: When Patterns Transfer

Several linked list patterns are variants of array techniques. Cross-reference with `problem-patterns.md`:

| Array Technique | Linked List Equivalent |
|----------------|----------------------|
| Two pointers (opposite ends) | Not directly applicable (no random access) |
| Fast-slow (same direction) | Fast-slow pointers (cycle, middle, kth from end) |
| In-place modification | Pointer rewiring (reverse, remove, partition) |
| Merge sorted arrays | Merge sorted lists (zipper technique) |
| Binary search | Not applicable (no O(1) indexing) |

The key difference: arrays support random access (O(1) indexing), linked lists support O(1) insertion/deletion at a known position. This is why linked list problems favor pointer manipulation over index arithmetic.

---

## Attribution

The patterns and techniques in this file are inspired by and adapted from labuladong's linked list problem-solving guides (labuladong.online), specifically the "linked-list-skills-summary" and "recursive-reverse-linked-list" articles. The reverseN, reverseBetween recursive approach, and successor pointer technique are from labuladong's Chapter 1 "Data Structure Algorithms." Templates have been restructured and annotated for Socratic teaching use.
