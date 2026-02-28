---
name: technical-interview-roadmap
description: This skill should be used when the user wants a technical interview preparation roadmap, coding interview study plan, or DSA practice plan tailored to a specific company and role. Trigger phrases include "technical interview roadmap", "coding interview prep for", "DSA roadmap for", "DSA study plan", "leetcode prep for", "what problems should I practice for", "interview study plan", "prep me for the technical rounds", "technical prep for", "what should I study for", "coding prep plan", "roadmap from this JD", "prep me for this role [URL]", or providing a JD URL with a request for technical interview preparation.
---

# Technical Interview Roadmap

Generate a company-specific technical interview study plan from a JD URL or pasted job description. Extracts DSA-relevant signals directly from the JD, researches the company's engineering domain, consults the leetcode-teacher learner profile, and outputs a curated LeetCode problem list with phased study timeline. Output goes to `hojicha/<company>-<role>-resume/technical-roadmap.md`.

## Critical Rules

1. **Require JD input.** The user must provide a JD URL or paste JD text. The skill extracts its own DSA-focused signals directly from the JD — it does not depend on resume-builder output. If no JD is provided, prompt the user to provide a URL or paste the JD text.
2. **No paywalled sources.** Only use public engineering blogs, GitHub, official company pages, YouTube tech talks. Hard ban on Glassdoor, Blind, LeetCode Discuss company tags, or any paywalled content.
3. **Cite all research.** Every company engineering claim needs a source URL in the appendix. No unsourced assertions about tech stack or interview process.
4. **Align with leetcode-teacher taxonomy.** All pattern names, classifications, and difficulty labels must match `references/frameworks/problem-patterns.md` from the leetcode-teacher skill. Use the exact pattern names: Two Pointers, Sliding Window, Binary Search, Dynamic Programming, DFS/BFS, Backtracking, Greedy, Hash Table, Heap / Priority Queue, Union-Find.
5. **Read-only on learner profile.** Never modify `~/.local/share/claude/leetcode-teacher-profile.md`. Read it for calibration only.
6. **15-25 problems total.** The problem list must be actionable, not overwhelming. Quality over quantity.
7. **Every problem needs a "Why."** Connect each problem to the company domain, role requirements, or a learner weakness. No generic filler entries.
8. **Companies test what they build.** Interview questions are not chosen at random. Companies select problems that test the foundational concepts their engineers use daily. A payments company asks graph problems because fraud detection traverses transaction graphs. A trading firm asks DP because optimal execution is a dynamic programming problem. A search company asks trie/string problems because their core product is text retrieval. Always reason backward from "what does this company's engineering team actually build?" to "what concepts must their engineers be fluent in?" to "which problems test those concepts?" Every problem in the roadmap must trace back to the company's business, tech stack, or core engineering challenges — not just to generic pattern frequency.
9. **No system design content.** This roadmap covers DSA/coding problems only. Explicitly disclaim system design in the output.
10. **Difficulty is NEVER capped by role level.** Default to full Easy-Medium-Hard coverage for ALL roles. Companies pull from shared OA pools (HackerRank, Codility, CodeSignal) that are often not calibrated to role level — even "early career" roles at crypto/fintech/tech companies routinely test Hard-level concepts (confirmed: Crypto.com early career OA, 2026-02-28, included Hard-level 2D DP + Union-Find problems). If OA ground truth exists (user reports actual OA problems), that overrides all heuristics. Company competitiveness shifts the distribution (more Hard problems for competitive firms) but the floor for ALL companies must include at least 2-3 Hard problems. Role level is useful context for prioritization, not a difficulty ceiling — see Difficulty Calibration in Quick Reference.
11. **URL fetch tool priority.** When fetching a JD URL, use Exa `crawling_exa` as primary. Fall back to `WebFetch` if Exa is unavailable. If neither works, ask the user to paste the JD text directly.

---

## Workflow

### Step 1: Parse Inputs

The user provides a JD URL or pastes JD text. This is the only JD input — the skill does not read `notes.md` or any resume-builder output for JD analysis.

**If JD URL or pasted text is provided:** Proceed to Step 1b.

**If neither is provided:** Prompt the user to either:
1. Provide a JD URL
2. Paste the JD text directly

Also read these optional files if they exist:
```
Optional:
- ~/.local/share/claude/leetcode-teacher-profile.md (learner profile for calibration)
- hojicha/candidate-context.md (discovery interview context — technical background, project details)
```

### Step 1b: DSA-Focused JD Extraction

Runs every time — this is the core input processing step.

#### 1b.1: Fetch JD Content

- **If URL:** Use Exa `crawling_exa` (primary) or `WebFetch` (fallback) to retrieve the page content. Extract the job description text from the fetched content — strip navigation, footers, and unrelated page elements.
- **If pasted text:** Use directly.
- **If fetch fails:** Ask the user to paste the JD text instead.

#### 1b.2: Extract DSA-Relevant Signals

Parse the JD content for:

1. **Hard skills** — programming languages, frameworks, tools, platforms (e.g., Python, PyTorch, AWS, Kubernetes). These map to algorithm patterns in Step 2.
2. **Domain keywords** — industry and problem domain terms (e.g., recommendation systems, real-time ML, distributed computing). These map to problem types in Step 2.
3. **Role level** — infer junior/mid/senior from JD language using the same heuristics as Step 2 item 4 (years of experience, "lead", "architect", "entry-level", "associate").
4. **Company name and role title** — extract for output directory naming and company research (Step 3). If ambiguous, ask the user.

#### 1b.3: Pass Signals Forward

The extracted signals (hard skills, domain keywords, role level, company name, role title, raw JD text) are used directly by Step 2 and Step 8. No intermediate file is written.

### Step 2: Extract Technical Signals from JD

Use the Hard Skills and Domain Keywords extracted in Step 1b.2. If `hojicha/candidate-context.md` exists, also scan it for additional technical signals (languages, frameworks, project domains) that may inform pattern prioritization.

Focus on what JD keywords *imply for coding interviews*:

1. **Hard skills → DSA topics.** Map technical requirements to algorithm patterns using `references/jd-signal-mapping.md`. Example: "distributed systems" → Graph algorithms, BFS/DFS; "real-time processing" → Sliding Window, Heap.
2. **Domain keywords → problem types.** Extract domain-specific signals. Example: "recommendation engine" → Hash Table, Dynamic Programming; "trading systems" → Greedy, Binary Search.
3. **Role level detection.** Use the role level from Step 1b.2 for difficulty calibration.

### Step 3: Company Engineering Research

Use Exa MCP tools (`web_search_exa`, `crawling_exa`) as primary research tools. Fall back to `WebSearch`/`WebFetch` if Exa is unavailable.

**Query strategy (max 5 queries):**

1. `"<company> engineering blog"` — tech stack, problem domains, engineering culture
2. `"<company>" site:github.com` — languages, open-source projects, infrastructure choices
3. `"<company> tech stack"` — confirm and expand the technology picture
4. `"<company> <role> technical interview"` — public interview process information
5. Crawl top 2-3 engineering blog URLs from query 1 for deeper context

**Extract core engineering challenges (Critical Rule 8).** From the research, identify the 3-5 core engineering problems the company's team solves daily. Examples:
- Stripe → real-time payment routing, fraud graph traversal, idempotent transaction processing
- Spotify → recommendation ranking at scale, audio stream chunking, collaborative filtering
- Cloudflare → packet-level routing optimization, DDoS pattern detection, edge cache invalidation

These engineering challenges drive Step 5 topic prioritization and Step 6b problem selection. If research is thin, infer challenges from the company archetype and JD signals.

**Hard ban:** No Glassdoor, Blind, LeetCode Discuss company tags, or any paywalled source. If a search result comes from a banned source, skip it.

**Thin research fallback:** If fewer than 3 substantive results are found:
- Note explicitly in the output that company-specific research was limited
- Use `references/company-archetypes.md` to select the best-fit company archetype as a proxy
- Rely more heavily on JD signals from Step 2

**No web search fallback:** If no web search tools are available at all (neither Exa nor WebSearch/WebFetch), skip live research entirely. Rely on JD signals (Step 2) and company archetypes (`references/company-archetypes.md`). Note prominently in the output that no live research was performed and recommendations are based on JD analysis and archetype matching only.

### Step 4: Load Learner Profile

Read `~/.local/share/claude/leetcode-teacher-profile.md` if it exists. Extract:

- **Known Weaknesses** — recurring and improving weaknesses (these become priority study targets)
- **Session History** — which patterns and problems have been practiced (avoid duplicates)
- **About Me** — self-assessed level for difficulty calibration

**If the profile does not exist:**
- Default to Intermediate level
- Note in the output that no learner profile was found
- Recommend building a profile via leetcode-teacher sessions

**Silent calibration:** Use the profile to adjust difficulty and topic priority internally. Do not dump raw profile contents into the output. Reference specific observations only when they directly inform a recommendation (e.g., "Linked list problems are prioritized because pointer mechanics are a tracked weakness").

### Step 5: Build Topic Roadmap

Synthesize Steps 2-4 into a prioritized topic list. **Reason backward from the company's engineering challenges** (Critical Rule 8): what does the company build → what concepts do their engineers use daily → what DSA patterns test those concepts → which problems exercise those patterns?

Each topic gets:

- **Company-specific reasoning** — the concrete engineering challenge at this company that makes this topic relevant. Not "Graph algorithms are common" but "Stripe's fraud detection system traverses transaction graphs to find anomalous payment chains, so they test graph traversal to ensure engineers can reason about connected components and path finding."
- **Priority tier:**
  - **Tier 1 (Must-Know):** Directly maps to a core engineering challenge identified in Step 3, or explicitly mentioned in the JD
  - **Tier 2 (Likely):** Maps to the company archetype's engineering domain or is common for this role type
  - **Tier 3 (Stretch):** Could differentiate the candidate; covers adjacent engineering concerns
- **Difficulty calibration** — Easy/Medium/Hard distribution based on role level and company competitiveness (see Difficulty Calibration in Quick Reference). Never assume a role level caps difficulty. All roadmaps must include Hard-level problems regardless of role level. If the user provides OA ground truth (actual problems they encountered), treat that as the primary difficulty signal and override all heuristic calibration.
- **Estimated problem count** — how many problems from this topic in the final list

Use `references/domain-topic-mapping.md` for domain → topic mapping. Use the core engineering challenges from Step 3 as the primary driver for Tier 1 selection — domain-topic-mapping is supplementary context, not the sole source.

Pattern names MUST match the leetcode-teacher taxonomy. See the Quick Reference section below for the canonical list.

> **Note:** Stack / Monotonic Stack problems are also included in the curated bank. These are not a top-level pattern in the leetcode-teacher taxonomy but appear frequently in interviews. Topological Sort is classified under DFS/BFS.

### Step 6: Curate Problem List

#### 6a: Company Frequency Enrichment (Optional Bonus)

> *This step provides supplementary signal when available. The roadmap's core value comes from JD analysis (Step 2), company research (Step 3), and learner calibration (Step 4). If enrichment fails, proceed without it — the roadmap quality is not meaningfully affected.*

Before selecting from the curated bank, attempt to fetch company-specific problem frequency data:

1. **Normalize company name to GitHub slug.** Use `references/company-slug-map.md` to look up the slug (e.g., "Goldman Sachs" → `goldman-sachs`, "Meta" → `meta`). If the company isn't in the map, derive the slug by lowercasing and replacing spaces/special chars with hyphens.
2. **Fetch recent problem data.** Use `WebFetch` to retrieve:
   ```
   https://raw.githubusercontent.com/snehasishroy/leetcode-companywise-interview-questions/master/<slug>/thirty-days.csv
   ```
   If the fetch fails (404, empty, or network error), skip enrichment entirely and proceed to 6b with the curated bank only.
3. **Parse the CSV.** The CSV has columns `ID,URL,Title,Difficulty,Acceptance %,Frequency %`. Extract `(ID, Title, Difficulty, Frequency %)` from each row, skip the header. Sort by `Frequency %` descending so the most-asked problems are considered first.
4. **Cross-reference with curated bank.** For each fetched problem, check if it exists in `references/curated-problem-bank.md` by LeetCode number:
   - **Match found:** Tag the problem with its pattern from the curated bank + the company frequency%.
   - **No match:** Tag as `company-frequent (untagged)` with its difficulty and frequency%.
5. **Note enrichment status in output.** If enrichment succeeded, add a line to the Company Engineering Profile section: `Company problem frequency data: sourced from public GitHub dataset (last 30 days / last 6 months)`. If enrichment failed, note: `Company-specific frequency data: not available for <company>. This is normal — the public dataset covers ~40 major companies. Problem selection uses JD signals, company archetype, and curated bank.`.

#### 6b: Problem Selection

Select 15-25 specific LeetCode problems. Use both `references/curated-problem-bank.md` and the enrichment data (if available) from Step 6a.

**Selection priority (highest → lowest):**
1. **Company-frequent + pattern-aligned** — problems from the company's frequency list that also match a Tier 1/2 pattern from the roadmap. These are the highest-signal picks.
2. **Pattern-aligned from curated bank** — problems from the curated bank matching Tier 1/2 patterns, even if not in the company frequency list.
3. **Company-frequent but untagged** — high-frequency company problems (≥50%) not in the curated bank. Include 2-3 of these max, noting "frequently asked at <company>, verify pattern alignment."

**Additional selection criteria (applied within each priority level):**
- **Difficulty progression** — Easy → Medium → Hard within each topic area
- **Company domain relevance** — prefer problems with domain tags matching the company's engineering focus
- **Learner weakness targeting** — include problems that exercise tracked weaknesses from the learner profile
- **OA ground truth alignment** — if the user reports actual OA problems, include direct preparation paths for those problem types (prerequisite problems building to the OA difficulty level, plus the OA-equivalent problem itself)
- **No duplicates** — check Session History and exclude problems the learner has already practiced

**Every problem gets:**
- LeetCode number and name
- Difficulty (Easy/Medium/Hard)
- Primary pattern (or "untagged" for company-frequent problems not in the curated bank)
- Company frequency% (if available from enrichment)
- A 1-line "Why" rationale that traces back to a specific engineering challenge, tech stack component, or business problem at the company. Bad: "Common DP problem." Good: "Citadel's optimal execution engine uses DP to minimize market impact — this tests the same state-transition reasoning." Acceptable fallback for learner-weakness picks: "Targets tracked pointer-mechanics weakness."

### Step 7: Generate Study Plan

Organize the curated problems into 3 phases:

**Phase 1: Foundations**
- Tier 1 topics, Easy-Medium difficulty
- Goal: build fluency in must-know patterns
- leetcode-teacher integration: Use **Learning Mode** for new patterns
- Duration: ~1 week (or 40% of available time)

**Phase 2: Core Depth**
- Tier 1-2 topics, Medium difficulty
- Goal: handle standard interview-level problems in priority areas
- leetcode-teacher integration: Use **Learning Mode** for new problems, **Recall Mode** for Phase 1 revisits
- Duration: ~1 week (or 40% of available time)

**Phase 3: Edge Sharpening**
- Tier 2-3 topics, Medium-Hard difficulty + weakness retesting
- Goal: differentiate and close remaining gaps
- leetcode-teacher integration: Use **Recall Mode** for all revisits, focus on timed practice
- Duration: ~0.5-1 week (or 20% of available time)

**Timeline:** Work backward from interview date if known. If no date is provided, default to a 2-3 week plan. Adjust phase durations proportionally.

### Step 8: Output Generation

Write `technical-roadmap.md` to `hojicha/<company>-<role>-resume/`:

```
hojicha/<company>-<role>-resume/
  technical-roadmap.md   # Generated by Step 8
```

Follow the output template in `references/output-template.md`.

---

## Quick Reference

### Output Directory Convention

```
hojicha/<company>-<role>-resume/technical-roadmap.md
```

Examples:
- `hojicha/google-ml-engineer-resume/technical-roadmap.md`
- `hojicha/stripe-backend-engineer-resume/technical-roadmap.md`
- `hojicha/citadel-quantitative-developer-resume/technical-roadmap.md`

### leetcode-teacher Pattern Taxonomy

Pattern names must match exactly:

| Pattern | Key Signals |
|---------|-------------|
| Two Pointers | Sorted arrays, pair finding, palindromes |
| Sliding Window | Contiguous subarray/substring, "at most K" |
| Binary Search | Sorted input, monotonic condition, "find minimum X" |
| Dynamic Programming | Overlapping subproblems, "number of ways", "minimum cost" |
| DFS/BFS | Tree/graph traversal, connected components, shortest path |
| Backtracking | "All combinations/permutations", constraint satisfaction |
| Greedy | Local optimal → global optimal, interval scheduling |
| Hash Table | O(1) lookup, frequency counting, complement finding |
| Heap / Priority Queue | "Kth largest/smallest", merge K sorted, streaming |
| Union-Find | Connected components, cycle detection, dynamic connectivity |

> **Note:** Stack / Monotonic Stack problems (e.g., Valid Parentheses, Daily Temperatures) are also included in the curated problem bank. These are not a top-level pattern in the leetcode-teacher taxonomy but appear frequently in interviews and should be selected when JD signals or company archetypes indicate relevance. Topological Sort is classified under DFS/BFS.

### Difficulty Calibration by Role Level

| Role Level | JD Signals | Minimum Distribution (15-25 problems) | Competitive Firm Uplift |
|------------|------------|---------------------------------------|------------------------|
| Junior / Entry | "0-2 years", "new grad", "entry-level", "associate" | 5 Easy + 12 Medium + 3 Hard | +2-3 additional Hard problems |
| Mid | "3-5 years", "engineer II", "software engineer" | 3 Easy + 14 Medium + 3 Hard | +2-4 additional Hard problems |
| Senior | "5+ years", "senior", "lead", "staff", "principal" | 0 Easy + 12 Medium + 8 Hard | +2-3 additional Hard problems |

> **These are minimums.** If OA ground truth is available (user reports actual problems from the company's assessment), override with observed difficulty. Real OAs routinely exceed expected difficulty regardless of role level — companies use shared assessment platforms (HackerRank, Codility, CodeSignal) with problem pools not calibrated to role level.

**Competitive firms** are archetypes with ≤20% Easy-Medium in their difficulty distribution: FAANG / Big Tech, Quant / HFT / Prop Trading, AI Labs / ML-First Companies, Government / Defense Tech. Check `references/company-archetypes.md` for the archetype's difficulty distribution.

### Research Tool Priority

1. Exa MCP tools (`web_search_exa`, `crawling_exa`) — primary
2. `WebSearch` / `WebFetch` — fallback if Exa unavailable
3. `references/company-archetypes.md` — proxy when research is thin
