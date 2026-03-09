---
name: dln
description: This skill should be used when the user wants to learn a new domain from scratch using structured cognitive phases, or when they say "dln", "learn [domain]", "teach me [domain] from zero", "cold-start [domain]", "start learning [domain]", or reference the Dot-Linear-Network framework. It orchestrates three phase skills (dln-dot, dln-linear, dln-network) based on the learner's current phase stored in a Notion database, routing them to the appropriate learning protocol for their level of understanding.
---

# DLN Learn — Domain-Agnostic Learning Orchestrator

A meta-learning skill that accelerates cold-start learning in any new domain using the Dot-Linear-Network (DLN) cognitive topology framework. This skill routes learners to the appropriate phase skill based on their tracked progress.

---

## 1. The DLN Framework

Learning follows three cognitive phases, each requiring a different teaching strategy:

| Phase | Mental State | Teaching Ratio | Goal |
|-------|-------------|----------------|------|
| **Dot** | Isolated facts, no connections | 70% delivery / 30% elicitation | Build concept nodes and basic causal chains |
| **Linear** | Procedural chains, domain-specific | 50% delivery / 50% elicitation | Discover shared factors across chains, build transferable understanding |
| **Network** | Compressed model, cross-domain links | 20% delivery / 80% elicitation | Stress-test, refine, and compress the learner's mental model |

The framework is domain-agnostic — it works for options pricing, compiler design, immunology, or any domain where the learner starts from zero.

---

## 2. Notion Database

All learner state is persisted in the **DLN Profiles** database in Notion (under Maekar).

- **Database ID:** `1f889a62f3414c17afb1c71a883a78d3`
- **Data Source:** `collection://7d60b0fb-2a0a-473d-bd58-305e84fd0851`

### Schema

| Property | Type | Purpose |
|----------|------|---------|
| Domain | title | The learning domain (e.g., "Options Pricing") |
| Phase | select | Current phase: Dot, Linear, or Network |
| Concepts | rich text | Nodes learned (Dot phase output) |
| Chains | rich text | Procedural sequences built (Dot phase output) |
| Factors | rich text | Shared structures discovered (Linear phase output) |
| Compressed Model | rich text | Latest model statement (Network phase output) |
| Open Questions | rich text | Unresolved gaps |
| Last Session | date | Timestamp of most recent session |
| Session Count | number | Total sessions in this domain |

---

## 3. Orchestrator Flow

### Step 1: Parse Domain

Extract the domain name from the user's message. Examples:
- "dln options pricing" → `Options Pricing`
- "learn compiler design from scratch" → `Compiler Design`
- "dln list" → list command
- "dln reset options pricing" → reset command

If no domain is specified, ask: *"What domain would you like to learn? Give me a topic and I'll set up your learning path."*

### Step 2: Handle Special Commands

**`list`** — Query the DLN Profiles database and display all domains with their current phase, session count, and last session date in a table.

**`reset [domain]`** — Find the matching row and update Phase back to Dot, clear Concepts/Chains/Factors/Compressed Model/Open Questions, reset Session Count to 0. Confirm with the user before executing.

### Step 3: Query or Create Profile

Use the Notion MCP to query the DLN Profiles database for a row matching the domain name.

**If found:** Read the current Phase and all accumulated knowledge fields.

**If not found:** Create a new row with:
- Domain = parsed domain name
- Phase = Dot
- Session Count = 0
- All text fields empty

Tell the user: *"New domain detected. Starting you in the Dot phase — we'll build your foundational concepts first."*

### Step 4: Load Context and Route

Read the relevant fields based on current phase:

| Phase | Fields to Load | Route To |
|-------|---------------|----------|
| Dot | Concepts, Chains, Open Questions | `dln-dot` skill |
| Linear | Concepts, Chains, Factors, Open Questions | `dln-linear` skill |
| Network | Factors, Compressed Model, Open Questions | `dln-network` skill |

### Step 5: Invoke Phase Skill

Pass the following context to the phase skill:
1. **Domain name**
2. **Accumulated knowledge** (from the fields loaded in Step 4)
3. **Session count** (so the phase skill knows if this is the first session or a continuation)
4. **Database reference** (so the phase skill can write back updates)

Use the Skill tool to invoke the appropriate phase skill (`dln-dot`, `dln-linear`, or `dln-network`).

### Step 6: Post-Session Update

After the phase skill completes (session ends), the phase skill itself handles writing back to Notion. The orchestrator does not need to do additional write-back.

---

## 4. Phase Transition Rules

Phase transitions are handled by the phase skills themselves:

- **Dot → Linear:** When the learner passes the Dot phase gate (can name concepts, explain causal chains, trace through a scenario). The `dln-dot` skill updates Phase to Linear.
- **Linear → Network:** When the learner passes the Linear phase gate (can name shared factors, predict unseen problems, identify minimal principle set). The `dln-linear` skill updates Phase to Network.
- **Network** is terminal — no further phase transitions. The skill tracks revision count and compression quality.

---

## 5. Error Handling

- **Notion unavailable:** Tell the user Notion is unreachable and offer to run the session without persistence (phase skill still works, just no state saved).
- **Multiple domain matches:** If the query returns multiple rows, show them and ask the user to clarify.
- **Phase skill not found:** This shouldn't happen in normal operation. If it does, report the error and suggest the user check their plugin installation.
