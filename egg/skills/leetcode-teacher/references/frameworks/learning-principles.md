# Make It Stick: Learning Principles Applied to Coding

Evidence-based learning strategies from *Make It Stick: The Science of Successful Learning* (Brown, Roediger, McDaniel, 2014), applied to algorithmic and ML implementation learning.

---

## The 8 Principles

### 1. Retrieval Practice

**The Science:** Actively recalling information strengthens memory far more than re-reading or passive review. Testing yourself — even before you know the answer — primes your brain to learn.

**How We Apply It:**
- Every section begins with Socratic questions *before* providing explanations
- "What do you think the brute force approach is?" comes before any code
- "Can you recall a similar problem?" before showing pattern connections

**Example in Practice:**
> *Teaching Two Sum:*
> Teacher: "Before we start — what data structure gives you O(1) lookup?"
> (Student retrieves: "Hash table")
> Teacher: "Right. Now, what would you store in that hash table for this problem?"
> (Student generates the answer rather than reading it)

---

### 2. Desirable Difficulties

**The Science:** Learning that feels easy often doesn't stick. Introducing deliberate difficulty — spacing, interleaving, varied practice — produces deeper, more durable learning.

**How We Apply It:**
- Always work through brute force before optimal — the struggle with the inefficient solution makes the efficient one stick
- Don't immediately hint when the user is stuck — a productive struggle of 1-2 minutes builds stronger recall
- Present problems in mixed order (interleaving) rather than grouped by pattern

**Example in Practice:**
> *Teaching Sliding Window:*
> Student tries O(n^2) approach for "Longest Substring Without Repeating Characters"
> Teacher: "Good — you got the brute force. Now, where exactly does it waste work?"
> (The difficulty of identifying the waste makes the sliding window insight memorable)

---

### 3. Elaboration

**The Science:** Explaining new material in your own words and connecting it to what you already know creates richer mental representations.

**How We Apply It:**
- After each section: "Can you explain this approach in your own words?"
- "How does this relate to [previous problem they solved]?"
- Ask users to create their own analogies

**Example in Practice:**
> *Teaching Binary Search on Answer:*
> Teacher: "You know standard binary search on arrays. How is 'binary search on the answer' similar? How is it different?"
> (Student elaborates: "Instead of searching for an element, we search for the minimum valid answer in a range")

---

### 4. Interleaving

**The Science:** Mixing different types of problems during practice — rather than blocking (doing all of one type, then all of another) — improves the ability to discriminate between approaches and choose the right one.

**How We Apply It:**
- Section 5 always connects to related problems of *different* patterns
- "You just solved a sliding window problem. Here's a problem that *looks* like sliding window but is actually DP — what's different?"
- Pattern recognition questions after every problem

**Example in Practice:**
> *After solving "Maximum Subarray" (DP/Kadane's):*
> Teacher: "This felt like a sliding window problem at first, right? What made you realize it's DP instead?"
> (Student learns to discriminate between patterns)

---

### 5. Generation

**The Science:** Attempting to solve a problem or answer a question *before* being shown the solution — even if your attempt is wrong — dramatically improves learning.

**How We Apply It:**
- Step 5 of the workflow: "Before I show you the optimal solution, what do you think it might be?"
- "Try writing the code first, even if it's incomplete"
- Wrong answers are valuable: "Interesting approach — let's trace through it and see what happens"

**Example in Practice:**
> *Teaching Adam Optimizer:*
> Teacher: "You know momentum tracks a running average of gradients. Adam also tracks something else. What do you think it tracks, and why?"
> (Student generates: "Maybe the variance? To normalize the step size?")
> Teacher: "Exactly right — even if the details aren't perfect, you had the core insight."

---

### 6. Reflection

**The Science:** Periodically reviewing what you've learned, what strategies worked, and what you'd do differently consolidates learning and builds metacognition.

**How We Apply It:**
- Section 7 (Pattern Recognition & Reflection) is entirely dedicated to this
- "What was the key insight that cracked this problem?"
- "If you saw this problem for the first time tomorrow, what would you look for?"
- "What part was hardest? Why?"

**Example in Practice:**
> *After completing a DP problem:*
> Teacher: "Looking back, what was the hardest part — defining the state, finding the transition, or optimizing space?"
> Student: "Defining what dp[i] means."
> Teacher: "That's the crux of most DP problems. What question can you ask yourself to find the right state definition?"

---

### 7. Structure Building

**The Science:** Expert learners build mental structures — frameworks and schemas — that organize knowledge and make it retrievable. They extract rules and key ideas from examples.

**How We Apply It:**
- The entire 6-section framework is itself a structure for approaching problems
- Pattern catalog (references/frameworks/problem-patterns.md) provides scaffolding for classification
- Every problem is explicitly linked to a pattern family
- Decision trees for pattern selection

**Example in Practice:**
> *The 6-section structure becomes internalized:*
> Student (internal monologue during interview): "Okay — layman intuition first. What's the analogy? ... Now brute force. What's the O(n^2) approach? ... Where's the bottleneck? ... What data structure fixes that?"

---

### 8. Growth Mindset

**The Science:** Believing that ability is developed through effort (growth mindset) vs. fixed at birth (fixed mindset) directly impacts learning outcomes. Effort and struggle are features, not bugs.

**How We Apply It:**
- Never say "wrong" — say "that's an interesting approach, let's explore it"
- Normalize difficulty: "This pattern is hard to see the first time — most people need 3-4 problems before it clicks"
- Celebrate the process: "The fact that you identified the brute force quickly shows progress"
- Frame errors as data: "Your off-by-one error is the most common mistake with this pattern — now you'll remember"

**Example in Practice:**
> Student: "I'm terrible at DP. I can never see the state."
> Teacher: "DP state definition is genuinely one of the hardest skills in algorithms. Here's the thing — every expert you admire was bad at it once. Let's build your intuition one problem at a time."

---

## Combined Principles Example: Teaching Adam Optimizer

This walkthrough shows how all 8 principles weave together in a single teaching session.

**1. Retrieval Practice:**
> "Before we start — what do you remember about SGD with momentum? What does the momentum term track?"

**2. Generation:**
> "Momentum tracks the mean of gradients. Adam tracks something additional. What do you think it might track?"

**3. Desirable Difficulty:**
> Start with implementing SGD, then momentum, then ask: "What's still missing? When would momentum alone fail?"
> (The progression from simple to complex creates productive struggle)

**4. Elaboration:**
> "In your own words, why do we need bias correction? What would happen at t=1 without it?"

**5. Interleaving:**
> "How does Adam differ from RMSprop? They look similar — what's the key difference?"

**6. Reflection:**
> "Now that you've implemented it — what was the trickiest part? The bias correction? The epsilon placement?"

**7. Structure Building:**
> "Let's organize: SGD → SGD+Momentum → RMSprop → Adam. Each builds on the previous. What does each one add?"

**8. Growth Mindset:**
> "Getting the epsilon placement wrong is the single most common Adam implementation bug. The fact that you caught it means you're developing the numerical stability instinct."

---

## References

- Brown, P. C., Roediger, H. L., & McDaniel, M. A. (2014). *Make It Stick: The Science of Successful Learning*. Harvard University Press.
- Roediger, H. L., & Karpicke, J. D. (2006). Test-enhanced learning: Taking memory tests improves long-term retention. *Psychological Science*, 17(3), 249-255.
- Bjork, R. A. (1994). Memory and metamemory considerations in the training of human beings. In J. Metcalfe & A. Shimamura (Eds.), *Metacognition: Knowing about knowing* (pp. 185-205).
