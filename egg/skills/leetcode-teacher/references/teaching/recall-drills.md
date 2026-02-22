# Recall Drills — Question Banks for Recall Mode

Structured question banks for probing depth of understanding during Recall Mode sessions. These questions are used by the interviewer persona (Section 5B) and are designed to test recall, not teach.

---

## 1. Edge Case Bank

Organized by problem type. Select 2-4 questions relevant to the problem being tested.

### Arrays

- "What happens if the array is empty?"
- "What if the array has only one element?"
- "What if all elements are identical?"
- "What about negative numbers? Zeros?"
- "What if the array is already sorted? Sorted in reverse?"
- "What happens at integer overflow boundaries (INT_MAX, INT_MIN)?"
- "What if the array length is 10^5? Does your approach still work within time limits?"
- "What about duplicate elements — does your solution handle them correctly?"

### Linked Lists

- "What if the list is empty (head is null)?"
- "What if the list has exactly one node?"
- "What if there's a cycle? How would you detect it?"
- "What happens if two lists have different lengths in a merge/intersection problem?"
- "What about a list where all values are the same?"
- "What if you need to return the head — are you handling the case where the head itself changes?"

### Trees

- "What if the tree is empty (root is null)?"
- "What if the tree has only one node?"
- "What if the tree is completely skewed (essentially a linked list)?"
- "What about a tree with all identical values?"
- "For BST problems: what if the tree isn't actually a valid BST?"
- "What's the maximum depth your recursive solution handles before stack overflow?"

### Graphs

- "What if the graph is disconnected?"
- "What if there are self-loops?"
- "What about parallel edges (multigraph)?"
- "What if the graph has no edges at all?"
- "What if the graph has a single node?"
- "For directed graphs: what about cycles?"
- "What if edge weights are negative?"

### Dynamic Programming

- "What's your base case? Walk me through why."
- "What happens when the target/capacity is 0?"
- "What if the input array is empty?"
- "Are there negative values in the input? Does your DP handle them?"
- "What if the answer overflows a 32-bit integer?"
- "What happens when your DP table dimensions are 0?"

### Binary Search

- "What if the array has one element?"
- "What if the target isn't in the array?"
- "What if there are duplicates and you need the first/last occurrence?"
- "Show me exactly what happens with your left and right pointers on a 2-element array."
- "Is your mid calculation safe from integer overflow? (left + right) vs left + (right - left) / 2?"

### Backtracking

- "What if the input is empty — what should the output be?"
- "Are you handling duplicate elements in your candidates? How do you avoid duplicate results?"
- "What's the maximum recursion depth? Could you hit a stack overflow?"
- "Are you properly undoing your choice in the backtrack step?"

### Sliding Window

- "What if the window size is larger than the array?"
- "What if the window size is 0 or 1?"
- "What happens when all elements satisfy / none satisfy the condition?"
- "For variable-width windows: what's your shrink condition exactly?"

---

## 2. Complexity Challenge Bank

Probing questions that go beyond "what's the Big O?" These test whether the user actually understands the complexity, not just memorized it.

### Time Complexity Probes

- "You said O(n). What is n exactly? Be precise."
- "Walk me through how you counted the operations. Which loop dominates?"
- "You have nested loops but claim O(n). Explain why the inner loop doesn't make it O(n^2)."
- "What's the best case? Is it different from the worst case?"
- "Your solution uses sorting. What's the overall complexity including the sort?"
- "You're using a hash map. What's the worst case for hash map operations? Does that change your analysis?"
- "If the input size doubles, how much longer does your solution take? Roughly 2x? 4x? Something else?"
- "You said O(n log n). Where does the log n come from?"
- "Is this amortized or worst case? What's the difference for your solution?"

### Space Complexity Probes

- "You said O(1) space. But you're creating a hash map — explain."
- "Does the recursion stack count as space? What's the max depth?"
- "Are you counting the output as part of your space complexity? Should you?"
- "You're modifying the input in place. Is that really O(1) space, or are you just hiding the cost?"
- "If you converted your recursive solution to iterative, would the space complexity change?"

### Amortized Analysis Probes

- "You mentioned amortized O(1). Walk me through a sequence of operations that shows this."
- "What's the worst single operation? How often does it happen?"
- "Can you give an accounting argument for why this is amortized O(1)?"
- "For your dynamic array / hash table: what happens when you resize?"

### Recurrence Relation Probes

- "Can you write the recurrence relation for your recursive solution?"
- "How do you solve T(n) = 2T(n/2) + O(n)? What's the result?"
- "Your recursion branches into k subproblems of size n/k. What's the overall complexity?"
- "Where does memoization reduce the complexity? How many unique subproblems are there?"

---

## 3. Pattern Classification Bank

Questions that test whether the user understands the problem at the pattern level, not just the specific solution.

### Pattern Recognition

- "What pattern or technique does this problem primarily use?"
- "How did you recognize this pattern? What signal in the problem statement tipped you off?"
- "Is this a pure instance of [pattern], or a hybrid? What else is involved?"
- "What's the key data structure that makes this pattern work? Why that one?"

### Transfer Questions

- "Name two other problems that use the same core technique."
- "What's a problem that looks similar but actually uses a different pattern?"
- "If I gave you [related problem], could you solve it with the same approach? What would change?"
- "What's the hardest problem you know that uses this pattern?"

### Constraint Sensitivity

- "If I changed the constraint from 'sorted array' to 'unsorted array', how would your approach change?"
- "What if I needed the k-th result instead of the optimal result?"
- "If memory was severely constrained (say, O(1) space), could you still solve this? How?"
- "What if the data was streaming in (you can't revisit elements)? Would your approach still work?"
- "If I allowed O(n) extra space, could you improve the time complexity?"
- "What if the input doesn't fit in memory? How would you adapt?"

---

## 4. Variation Bank

Structured variations organized by problem family. Present one variation during R6 to test adaptation.

### Two Sum Family

- **Original:** Two Sum (unsorted array, return indices)
- **Variation 1:** "What if the array is sorted?" → Two-pointer approach
- **Variation 2:** "What if you need all pairs, not just one?" → Handle duplicates
- **Variation 3:** "What if it's Three Sum instead?" → Reduce to Two Sum
- **Variation 4:** "What if the target changes for each query but the array stays the same?" → Preprocess with hash map
- **Variation 5:** "What if the numbers are streaming in?" → Online algorithm

### Sliding Window Family

- **Original:** Maximum sum subarray of size k
- **Variation 1:** "What if the window size isn't fixed — find the smallest window with sum >= target?"
- **Variation 2:** "What if you need the longest substring without repeating characters?"
- **Variation 3:** "What if you need the minimum window containing all characters of a target string?"
- **Variation 4:** "What if the window condition involves a frequency count (at most k distinct characters)?"

### Tree Family

- **Original:** Maximum depth of binary tree
- **Variation 1:** "What if I need the minimum depth instead?"
- **Variation 2:** "What if I need to check if the tree is balanced?"
- **Variation 3:** "What if I need the diameter (longest path between any two nodes)?"
- **Variation 4:** "What if this is an N-ary tree instead of binary?"
- **Variation 5:** "What if I need the path that gives the maximum sum?"

### Graph Family

- **Original:** Number of islands (2D grid DFS/BFS)
- **Variation 1:** "What if islands can be connected diagonally?"
- **Variation 2:** "What if I need the area of the largest island?"
- **Variation 3:** "What if I'm adding land one cell at a time and need the count after each addition?" → Union-Find
- **Variation 4:** "What if I need to find if two cells are connected?" → Union-Find / BFS
- **Variation 5:** "What if some cells are walls and I need shortest path?" → BFS with obstacles

### DP Family

- **Original:** Climbing stairs (Fibonacci-style)
- **Variation 1:** "What if you can take 1, 2, or 3 steps?"
- **Variation 2:** "What if each step has a cost (min cost climbing stairs)?"
- **Variation 3:** "What if you can also go back down?"
- **Variation 4:** "What if you need to count paths in a 2D grid instead?"
- **Variation 5:** "What if some stairs are broken (obstacles)?"

### Binary Search Family

- **Original:** Find target in sorted array
- **Variation 1:** "What if the array is rotated?"
- **Variation 2:** "What if you need to find the insertion position?"
- **Variation 3:** "What if you're searching for a range (first and last position)?"
- **Variation 4:** "What if the array has duplicates and is rotated?"
- **Variation 5:** "What if instead of an array, you're searching on an answer space (e.g., minimum capacity to ship packages in D days)?"

---

## 5. Mock Interview Simulation Patterns

Scripts for R1-R7 that maintain the interviewer persona. Use these as templates, adapting to the specific problem.

### Opening Script (R1)

> "Alright, let's do [problem name]. Here's the problem: [clean problem statement with examples]. Take a moment to think, then walk me through how you'd approach this."

**If the user asks for hints:**
> "Try to work through it on your own first. What's your initial instinct?"

**If the user asks to clarify the problem:**
> Answer factually — clarifying the problem is expected and encouraged in real interviews.

### Probing Script (R2-R5)

**Neutral acknowledgments (use instead of praise):**
- "Okay, go on."
- "Got it. What's next?"
- "I see. Continue."
- "Mm-hm."
- "Alright."

**When the user pauses:**
- "Take your time."
- "What are you thinking?"
- "Where would you go from here?"

**When the user gives a vague answer:**
- "Can you be more specific?"
- "Walk me through the exact steps."
- "Show me with the example input."
- "What data structure are you using, and why?"

**When the user gives a wrong answer (don't correct — probe):**
- "Let's trace through that with the example. What happens at step [X]?"
- "Are you sure about that? Walk me through your reasoning."
- "What would happen if the input were [edge case]?"
- "Hmm, let's verify. Can you run through [specific test case] by hand?"

### Pushback Script (R3-R6)

**For edge cases:**
- "You mentioned handling [case]. What about [case they missed]?"
- "What's the worst-case input for your solution?"
- "Would your solution handle [tricky input]? Let's trace through it."

**For complexity:**
- "You said O(n). Convince me."
- "Is there a way to do this faster?"
- "You're using extra space. Can you avoid it?"
- "What if I told you there's an O(1) space solution?"

**For pattern questions:**
- "You said this is a [pattern] problem. What specifically makes it [pattern]?"
- "Could you solve this with [different pattern]? Why or why not?"
- "If the input changed to [new constraint], would you still use [pattern]?"

### Closing Script (R7)

> "Alright, let's step out of interview mode and debrief."
>
> "**What you did well:** [specific things — algorithm choice, communication, edge case awareness]"
>
> "**Areas to strengthen:** [specific gaps with correct answers]"
>
> "**Overall:** [honest assessment — would this pass? where would they lose points?]"
>
> "**Recommendation:** [spaced repetition schedule based on performance]"

**If performance was strong:**
> "You clearly know this problem well. I'd suggest reviewing it again in about a week to keep it fresh, then move on to harder problems in the same pattern family."

**If performance had gaps:**
> "You've got the right foundation, but [specific areas] need more work. I'd recommend reviewing [concept] tomorrow, then testing yourself again in 3 days."

**If performance was weak:**
> "This one needs more practice. I'd suggest going through the full learning mode for this problem first, then coming back to quiz mode once you're more comfortable with [specific concept]."
