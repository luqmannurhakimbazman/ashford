# Company Archetypes

Interview patterns by company type. Use as a proxy when company-specific research (Step 3) yields thin results. Select the best-fit archetype based on available signals.

> **Key principle:** Companies don't pick interview questions at random. Each pattern in the tables below is tested because it maps to a real engineering challenge the company's engineers face daily. The "Why They Test This" column explains the connection — use it to inform the "Why" rationale for every problem in the roadmap.

---

## FAANG / Big Tech

**Examples:** Google, Meta, Amazon, Apple, Microsoft, Netflix

**Interview Characteristics:**
- 2-3 coding rounds (45-60 min each)
- Emphasis on optimal solutions — brute force alone is insufficient
- Follow-up questions on complexity, trade-offs, and scaling
- Interviewers expect clean code and clear communication

**DSA Focus:**

| Pattern | Weight | Why They Test This |
|---------|--------|--------------------|
| Dynamic Programming | Very High | Ads ranking, content recommendation, and cost optimization all require optimal substructure reasoning |
| DFS/BFS | Very High | Social graphs, file system traversal, dependency resolution are daily engineering problems at scale |
| Hash Table | High | Caching, deduplication, and fast lookup underpin virtually every service at billion-user scale |
| Two Pointers | High | Feed merging, string processing, and diff computation in large-scale data pipelines |
| Sliding Window | High | Real-time metrics dashboards, rate limiting, and streaming analytics across services |
| Binary Search | High | Search ranking, configuration lookups, and A/B test threshold optimization |
| Backtracking | Medium | Permission systems, query planners, and configuration space exploration |
| Greedy | Medium | Task scheduling, resource allocation across data centers, interval-based capacity planning |
| Heap / Priority Queue | Medium | News feed ranking, notification prioritization, merge of distributed sorted streams |
| Union-Find | Low | Network partition detection, account merging across services |

**Difficulty Distribution:** 30% Medium, 60% Medium-Hard, 10% Hard

**Special Notes:**
- Google: Known for graph problems and DP; may ask novel problems not on LC
- Meta: Array/string heavy; practical coding focus
- Amazon: Leadership principle alignment; coding is Medium-level but tests completeness
- Apple: Emphasis on clean code and design awareness alongside algorithms

---

## Quant / HFT / Prop Trading

**Examples:** Citadel, Jane Street, Two Sigma, DE Shaw, Jump Trading, IMC, Optiver, Hudson River Trading

**Interview Characteristics:**
- 1-3 coding rounds + math/probability rounds
- Speed and optimality are paramount — O(n log n) may not be good enough
- May include live pair programming or whiteboard
- Strong emphasis on mathematical reasoning alongside DSA
- Brain teasers and probability puzzles are common in non-coding rounds

**DSA Focus:**

| Pattern | Weight | Why They Test This |
|---------|--------|--------------------|
| Dynamic Programming | Very High | Optimal execution algorithms, options pricing (binomial trees), and portfolio allocation are DP problems |
| Binary Search | Very High | Order book price lookups, bisection for numerical pricing models, binary search on answer for threshold optimization |
| Heap / Priority Queue | High | Order matching engines rank orders by price-time priority; streaming top-K for real-time P&L monitoring |
| Hash Table | High | Symbol-to-position mapping, real-time cache for market data, O(1) lookups critical at nanosecond scale |
| Greedy | High | Trade scheduling under constraints, greedy execution strategies to minimize market impact |
| Two Pointers | Medium | Time-series pair analysis, sorted order book scanning, merge operations on tick data |
| Sliding Window | Medium | Moving average computation (VWAP, TWAP), rolling volatility windows, signal detection |
| DFS/BFS | Medium | Cross-asset dependency graphs, risk exposure propagation through correlated instruments |
| Backtracking | Low | Strategy parameter space exploration |
| Union-Find | Low | Rarely tested |

**Difficulty Distribution:** 10% Medium, 50% Medium-Hard, 40% Hard

**Special Notes:**
- Jane Street: OCaml-focused; problems may have mathematical elegance requirements
- Citadel: Known for hard DP and optimization problems
- Two Sigma: Mix of practical engineering + algorithmic depth
- HFT firms (Jump, IMC, Optiver): Emphasis on speed-optimal solutions and bit manipulation

---

## AI Labs / ML-First Companies

**Examples:** Anthropic, OpenAI, DeepMind, Cohere, Hugging Face, Scale AI

**Interview Characteristics:**
- Coding rounds test general DSA + ML-specific implementation
- May include "implement X from scratch" (e.g., attention mechanism, tokenizer)
- Emphasis on understanding algorithmic complexity in ML contexts
- Some rounds may blend coding with ML knowledge

**DSA Focus:**

| Pattern | Weight | Why They Test This |
|---------|--------|--------------------|
| Dynamic Programming | High | Training loop optimization, sequence alignment (Viterbi), loss landscape navigation are DP-structured |
| DFS/BFS | High | Computation graph traversal (backprop), model architecture search, dependency ordering in training pipelines |
| Hash Table | High | Embedding table lookups, feature hashing, memoization in recursive model evaluation |
| Binary Search | Medium | Hyperparameter tuning (binary search on learning rate), threshold calibration for model confidence |
| Sliding Window | Medium | Context window processing in transformers, streaming inference, windowed feature extraction |
| Heap / Priority Queue | Medium | Beam search decoding, top-K token selection, nearest neighbor retrieval in vector DBs |
| Two Pointers | Medium | Tensor merge operations, sorted data alignment for training batches |
| Greedy | Low-Medium | Greedy decoding strategies, data selection for active learning |
| Backtracking | Low | Architecture search, feature subset selection |
| Union-Find | Low | Data clustering, deduplication in training datasets |

**Difficulty Distribution:** 20% Medium, 60% Medium-Hard, 20% Hard

**Special Notes:**
- May test ML implementation (backprop, attention, loss functions) alongside standard DSA
- Python is the dominant language; numpy-aware solutions sometimes expected
- Problems may be "applied" — framed in ML context (e.g., "implement efficient KNN")

---

## High-Growth Startup

**Examples:** Series A-D startups in any vertical

**Interview Characteristics:**
- 1-2 coding rounds (often shorter, 30-45 min)
- Emphasis on practical coding ability over algorithmic puzzle-solving
- May include take-home assignments
- Values breadth and ability to ship

**DSA Focus:**

| Pattern | Weight | Why They Test This |
|---------|--------|--------------------|
| Hash Table | Very High | Startups build CRUD apps, APIs, and caches — hash-based lookup is the bread and butter of every feature |
| DFS/BFS | High | Feature trees (navigation, categories), permission hierarchies, and workflow DAGs are common startup patterns |
| Two Pointers | Medium | String processing for user input validation, list deduplication, search result merging |
| Sliding Window | Medium | Rate limiting user requests, real-time analytics dashboards, log stream processing |
| Greedy | Medium | Feature prioritization logic, simple scheduling (notifications, batch jobs) |
| Binary Search | Medium | Paginated data lookup, configuration threshold detection |
| Dynamic Programming | Low-Medium | Rarely relevant to daily work; tested lightly to gauge problem-solving ability |
| Heap / Priority Queue | Low | Notification priority, job queue ordering when product requires it |
| Backtracking | Low | Rarely tested |
| Union-Find | Low | Rarely tested |

**Difficulty Distribution:** 40% Easy-Medium, 50% Medium, 10% Medium-Hard

**Special Notes:**
- Take-home projects may replace or supplement live coding
- System design and practical engineering skills often weighted equally with DSA
- Culture fit and "can you build things" matters more than optimal solutions

> **Caution:** Startup OAs often use third-party platforms (HackerRank, Codility, CodeSignal) with difficulty pools not calibrated to role level. Actual OA difficulty may significantly exceed this distribution — confirmed cases of early-career roles at crypto/fintech startups testing Hard-level 2D DP and Union-Find problems. Always include Hard-level preparation regardless of what the archetype distribution suggests.

---

## Unicorn / Late-Stage Tech

**Examples:** Stripe, Databricks, Cloudflare, Figma, Notion, Discord, Ramp

**Interview Characteristics:**
- 2-3 coding rounds, well-structured process
- Moderate difficulty — harder than startups, slightly easier than FAANG
- Often domain-relevant problems (e.g., Stripe asks payment-related algorithmic problems)
- Clean code and communication valued

**DSA Focus:**

| Pattern | Weight | Why They Test This |
|---------|--------|--------------------|
| Hash Table | Very High | Payment dedup (Stripe), real-time collaboration state (Figma/Notion), message routing (Discord) — all hash-based |
| DFS/BFS | High | Permission hierarchies, document trees (Notion), channel/server graphs (Discord), query plan traversal (Databricks) |
| Two Pointers | High | Transaction log merging, collaborative cursor reconciliation, sorted data alignment |
| Sliding Window | High | Rate limiting (Cloudflare), streaming data analytics (Databricks), real-time event processing |
| Binary Search | Medium | Tiered pricing lookups, capacity threshold detection, version history search |
| Dynamic Programming | Medium | Cost optimization in payment routing, query plan optimization (Databricks), cache eviction policies |
| Greedy | Medium | Request scheduling, resource allocation, notification batching |
| Heap / Priority Queue | Medium | Message delivery ordering (Discord), job scheduling, priority-based load balancing |
| Backtracking | Low | Configuration space exploration, permission rule evaluation |
| Union-Find | Low | Network partition detection (Cloudflare), account merging |

**Difficulty Distribution:** 20% Easy-Medium, 60% Medium, 20% Medium-Hard

**Special Notes:**
- Stripe: Known for practical, domain-relevant problems (payment systems, API design)
- Databricks: Heavier on data processing and optimization
- Figma: May include canvas/rendering-related algorithmic problems

> **Caution:** Unicorn/late-stage companies often use third-party OA platforms (HackerRank, Codility, CodeSignal) with difficulty pools not calibrated to role level. Actual OA difficulty may significantly exceed this distribution. Always include Hard-level preparation regardless of role level.

---

## Enterprise SaaS / Large Non-Tech

**Examples:** Salesforce, ServiceNow, JPMorgan Chase, Walmart Labs, Capital One, Intuit

**Interview Characteristics:**
- 2-3 coding rounds (45-60 min each)
- Moderate difficulty — practical engineering over algorithmic puzzles
- Emphasis on clean, maintainable code and API-aware thinking
- May include domain-specific questions (payments, workflows, enterprise integrations)

**DSA Focus:**

| Pattern | Weight | Why They Test This |
|---------|--------|--------------------|
| Hash Table | Very High | Customer record lookup, transaction caching, config mapping — hash tables power every enterprise CRUD operation |
| DFS/BFS | High | Workflow DAGs (ServiceNow), org hierarchy traversal (Salesforce), fraud detection graphs (JPMorgan/Capital One) |
| Two Pointers | High | Record deduplication, sorted ledger merging, data migration reconciliation |
| Sliding Window | Medium | Real-time transaction monitoring, rolling compliance windows, streaming inventory updates (Walmart) |
| Binary Search | Medium | Price tier lookups, regulatory threshold detection, sorted ledger queries |
| Greedy | Medium | Job scheduling, warehouse allocation (Walmart), loan approval heuristics (Capital One) |
| Dynamic Programming | Low-Medium | Tax computation optimization (Intuit), fee calculation chains — rarely multi-dimensional |
| Heap / Priority Queue | Low-Medium | Ticket priority queues (ServiceNow), transaction processing order |
| Backtracking | Low | Rarely tested |
| Union-Find | Low | Rarely tested |

**Difficulty Distribution:** 30% Easy-Medium, 55% Medium, 15% Medium-Hard

**Special Notes:**
- Salesforce/ServiceNow: API design and data modeling questions alongside DSA
- JPMorgan/Capital One: May include finance-domain problems; test practical engineering judgment
- Walmart Labs: Scale-aware questions — expect follow-ups about handling millions of records

---

## Government / Defense Tech

**Examples:** Palantir, Anduril, Raytheon, Northrop Grumman, Booz Allen Hamilton

**Interview Characteristics:**
- 2-3 coding rounds, often with systems programming emphasis
- Graph and optimization problems are common
- May include low-level implementation questions (memory, concurrency)
- Security awareness and correctness valued over speed of delivery

**DSA Focus:**

| Pattern | Weight | Why They Test This |
|---------|--------|--------------------|
| DFS/BFS | Very High | Intelligence analysts traverse entity relationship graphs; network topology analysis and threat propagation modeling are core workflows |
| Dynamic Programming | High | Resource allocation across missions, optimal path planning for autonomous systems, signal decoding |
| Binary Search | High | Sensor data thresholding, geospatial range queries, sorted intelligence record lookup |
| Hash Table | High | Entity deduplication across data sources, fast lookup in surveillance databases, classification mapping |
| Union-Find | Medium | Network segmentation analysis, identifying connected threat actor clusters, communication group detection |
| Greedy | Medium | Mission scheduling under constraints, bandwidth allocation, sensor coverage optimization |
| Two Pointers | Medium | Log correlation between sorted event streams, data reconciliation across feeds |
| Heap / Priority Queue | Medium | Threat priority triage, mission-critical task scheduling, top-K anomaly detection |
| Backtracking | Low-Medium | Configuration search for secure system parameters, constraint satisfaction in mission planning |
| Sliding Window | Low | Time-windowed anomaly detection in streaming sensor data |

**Difficulty Distribution:** 15% Easy-Medium, 50% Medium, 35% Medium-Hard

**Special Notes:**
- Palantir: Known for hard graph problems and data-infrastructure questions
- Anduril: Systems programming emphasis; may test C++ or Rust alongside algorithms
- Defense contractors: Correctness and edge-case handling valued more than raw speed

---

## How to Select an Archetype

1. **Check company directly** — if research yields clear interview data, use that over archetypes
2. **Match by company stage and domain:**
   - Public tech company with 10K+ employees → FAANG / Big Tech
   - Trading/finance with quantitative focus → Quant / HFT
   - AI/ML as core product → AI Labs
   - <500 employees, venture-backed → High-Growth Startup
   - 500-5000 employees, well-known brand → Unicorn / Late-Stage Tech
   - Large SaaS, banks, or enterprise software → Enterprise SaaS / Large Non-Tech
   - Defense, intelligence, or government-adjacent tech → Government / Defense Tech
3. **Blend when needed** — a fintech startup might combine Startup + Quant characteristics; a defense AI company might combine Government/Defense + AI Labs
4. **When in doubt, default to Unicorn / Late-Stage Tech** — it has the broadest, most balanced coverage
