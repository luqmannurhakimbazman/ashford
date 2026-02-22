# Socratic Question Bank

Organized question bank for guiding learners through algorithmic and ML implementation problems.

---

## Questions by Learning Stage

### Stage 1: Understanding the Problem

- "In your own words, what is this problem asking you to do?"
- "What are the inputs? What are the outputs?"
- "Can you walk me through the example? Why does this input produce this output?"
- "What are the constraints? How do they hint at the expected complexity?"
- "What edge cases can you think of?"

### Stage 2: Brute Force Exploration

- "What's the most straightforward approach, even if it's slow?"
- "If you had to solve this by hand for the example, what steps would you take?"
- "Can you translate those hand-steps into pseudocode?"
- "What's the time complexity of this approach? How did you determine that?"
- "Why isn't this efficient enough? What's the bottleneck?"

### Stage 3: Optimization Discovery

- "Where does the brute force do redundant work?"
- "What information are we computing multiple times that we could store?"
- "What data structure would let us do [bottleneck operation] faster?"
- "Can you see a way to reduce the number of passes through the data?"
- "What if you sorted the input first? Would that help?"
- "Is there a way to avoid checking every possible pair/combination?"

### Stage 4: Implementation

- "How would you initialize your data structures?"
- "Walk me through the first few iterations of your algorithm."
- "What happens at the boundaries? First element? Last element?"
- "Are there any invariants your algorithm must maintain?"
- "How do you handle the edge case of [empty input / single element / all same values]?"

### Stage 5: Complexity Analysis

- "How many times does each element get processed?"
- "What's the worst case? Can you construct an input that triggers it?"
- "Is the space usage proportional to the input or constant?"
- "Could this be done with less space? What would you sacrifice?"
- "Is this amortized O(n) or worst-case O(n)? Does it matter here?"

### Stage 6: Pattern Recognition

- "What pattern does this problem belong to?"
- "Have you seen a similar problem before? What's the connection?"
- "If I changed [constraint], what pattern would you use instead?"
- "What's the common thread between this problem and [related problem]?"
- "How would you recognize this pattern in a new problem?"

### Stage 7: Reflection & Metacognition

- "What was the key insight that unlocked the optimal solution?"
- "If you saw this problem for the first time tomorrow, what would you look for?"
- "What was the hardest part? Why?"
- "What would you explain differently to someone else?"
- "Rate your confidence (1-5). What would move it up one point?"

---

## Questions by Problem Type

### Array / String Problems

- "Does the order of elements matter?"
- "Would sorting help? What would you lose by sorting?"
- "Can you solve it in a single pass?"
- "What if you processed from both ends simultaneously?"
- "Is there a way to represent the state with just a few variables instead of a full data structure?"

### Tree / Graph Problems

- "Is this a tree or a general graph? Does it matter?"
- "Would DFS or BFS be more natural here? Why?"
- "What information do you need to pass down from parent to child?"
- "What information needs to bubble up from children to parent?"
- "Could this be solved iteratively instead of recursively?"
- "How do you detect/handle cycles?"

### Dynamic Programming Problems

- "What decisions are being made at each step?"
- "If you solved a smaller version of this problem, how would you extend the solution?"
- "What's the state? What information do you need to make the next decision?"
- "Can you define dp[i] in plain English?"
- "What are the base cases?"
- "Does the order of filling the DP table matter?"
- "Can you optimize from 2D to 1D? Which dimension can you drop?"

### ML Implementation Problems

- "What problem was this algorithm designed to solve?"
- "What happens if you remove [specific component]? What breaks?"
- "Walk me through the shapes at each step."
- "Where could numerical instability occur?"
- "How would you verify your implementation is correct?"
- "What's the difference between training and inference behavior?"
- "Why is the gradient computed this way? Can you derive it?"

---

## Questions by Framework

These questions are designed around labuladong's meta-frameworks (see `algorithm-frameworks.md`). Use them when you recognize the framework applies but want the learner to discover it themselves.

### Sliding Window Framework Questions

- "This problem involves a contiguous subarray/substring. What three questions do you need to answer to define a sliding window?" *(Q1: when to expand, Q2: when to shrink, Q3: when to update)*
- "When should the window grow? What are you adding with each expansion?"
- "Under what condition does the window become invalid — and what do you remove to fix it?"
- "Should you update your result when the window is valid, when it becomes invalid, or both?"

### DP Framework Questions

- "What's the state — what changes between subproblems? Can you name the variables?"
- "At each state, what choices can you make? List them."
- "If I told you the answer for every smaller subproblem, could you compute the answer for the current one? How?" *(mathematical induction probe)*
- "Can you write `dp[i] = ...` in plain English before writing any code?"
- "Would top-down or bottom-up be easier to think about here? Why?"

### Backtracking Framework Questions

- "What is the decision tree for this problem? What choice do you make at each node?"
- "What three things define your state at any node: what's in your path, what's in your choice list, and what's your end condition?"
- "Can elements be reused? Are there duplicates in the input? How does that change the recursion?"
- "Where would you prune — what branches can you cut early because they can't possibly lead to a valid answer?"

### Recursion Framework Questions

- "This is a recursive problem. Are you thinking in traversal mode (walk the tree, collect state) or decomposition mode (each subtree returns a value)?"
- "If this were a tree, what work happens at each node? Is it pre-order (before recursing) or post-order (after recursing)?"
- "Can you solve this by having each recursive call return something useful to its parent?" *(nudge toward decomposition)*
- "Do you need to track state along the path, or can each subproblem be solved independently?" *(traversal vs decomposition)*

---

## Three-Tier Hint System Examples

### Example: Two Sum

**Tier 1 — High-Level Direction:**
> "Think about what lets you check if a complement exists in O(1)."

**Tier 2 — Structural Hint:**
> "What if you built a lookup as you iterate? For each element, check if its complement already exists."

**Tier 3 — Specific Guidance:**
> "Use a hash map. For each number, check if `target - nums[i]` is in the map. If not, add `nums[i]: i` to the map."

### Example: Longest Substring Without Repeating Characters

**Tier 1:**
> "Can you avoid rechecking characters you've already seen?"

**Tier 2:**
> "What if you maintained a window that always contains unique characters? When would you shrink it?"

**Tier 3:**
> "Use a sliding window with a set. Expand right. When you see a duplicate, shrink from the left until the duplicate is removed."

### Example: Adam Optimizer Implementation

**Tier 1:**
> "Adam combines two ideas from earlier optimizers. Which two?"

**Tier 2:**
> "You need a running average of gradients (like momentum) AND a running average of squared gradients (like RMSprop). But there's a correction step — why?"

**Tier 3:**
> "At t=1, m and v are initialized to 0 and heavily biased. Divide by `(1 - beta^t)` to correct. Implement the bias-corrected estimates before the parameter update."

### Example: Binary Search on Answer (Koko Eating Bananas)

**Tier 1:**
> "Instead of searching the array, what if you searched over possible answers?"

**Tier 2:**
> "The answer is a speed K. For any K, you can check if Koko can finish in time. What property does this check have as K increases?"

**Tier 3:**
> "Binary search over K from 1 to max(piles). For each K, compute total hours = sum(ceil(pile/K)). If hours <= H, K might work — search lower. Otherwise search higher."

### Example: House Robber (DP with Adjacency Constraint)

**Tier 1:**
> "You can't take two adjacent elements. If you decide to take element `i`, what constraint does that put on your previous choice?"

**Tier 2:**
> "At each house, you have exactly two choices: rob it or skip it. If you rob it, your best so far can't include the previous house. Can you express that as a recurrence?"

**Tier 3:**
> "Define `dp[i] = max(dp[i-1], dp[i-2] + nums[i])`. Either skip house `i` (take `dp[i-1]`) or rob it (add `nums[i]` to the best without the previous house, `dp[i-2]`). You only need two variables."

### Example: LRU Cache (Data Structure Design)

**Tier 1:**
> "You need O(1) for both lookup and ordering. What data structure gives O(1) lookup? What gives O(1) ordering changes?"

**Tier 2:**
> "A hash map gives O(1) lookup. A linked list lets you move elements to the front in O(1). How would you combine them so that the hash map points directly into the list?"

**Tier 3:**
> "Use a HashMap mapping key to a doubly-linked list node. On `get`: find the node via map, remove it from its position, add to front. On `put`: create node, add to front, add to map. If over capacity, remove the tail node and delete its key from the map. Use dummy head/tail to avoid null checks."

---

## Debugging Questions

When a user's solution has a bug:

- "Let's trace through your code with the example input. What happens at step [N]?"
- "What does [variable] equal at this point? Is that what you expected?"
- "What happens when the input is [edge case]?"
- "Your code handles the general case. What about [boundary condition]?"
- "Is this comparison correct? Should it be `<` or `<=`?"
- "Are you updating [variable] at the right time — before or after [operation]?"

---

## Anti-Patterns: What NOT to Ask

These questions are counterproductive to learning:

| Bad Question | Why It's Bad | Better Alternative |
|-------------|-------------|-------------------|
| "Do you know what a hash map is?" | Yes/no questions don't promote thinking | "When would you reach for a hash map?" |
| "The answer is a sliding window, right?" | Leading question gives away the answer | "What pattern do you think fits here?" |
| "Why didn't you think of that?" | Judgmental, discourages exploration | "What made this hard to see?" |
| "This is easy, just use DP" | Dismisses difficulty, not actionable | "Let's see if we can break this into subproblems." |
| "Do you understand?" | Almost everyone says yes regardless | "Can you explain this step back to me?" |

---

## Calibration: When to Ask vs When to Tell

### Ask More (increase Socratic questioning) when:
- User is making progress, even slowly
- User has the prerequisite knowledge to reason through it
- The insight is one the user can plausibly reach
- It's early in the problem (Stages 1-3)

### Tell More (reduce questioning, provide more guidance) when:
- User has been stuck on the same point for 3+ exchanges
- User is visibly frustrated ("I have no idea", "just tell me")
- The required knowledge is something the user hasn't encountered before
- It's a syntax/API detail, not a conceptual insight
- Time pressure is explicit ("I have an interview tomorrow")

### The 60/40 Rule
Aim for roughly 60% questions, 40% explanations. Adjust based on the user's responses:
- If the user answers most questions correctly → ask harder ones
- If the user struggles with most questions → shift toward guided explanation with embedded micro-questions
