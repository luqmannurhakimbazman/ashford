# Visual Format Constraints

Claude Code operates in a text terminal. Available visual formats:

| Format | Best For | Limitations |
|--------|----------|-------------|
| **Mermaid diagrams** | Flowcharts, sequence diagrams, concept maps | Renders in Mermaid-compatible viewers; displays as readable code in terminal |
| **ASCII box diagrams** | Simple relationships, 2-4 nodes | Universal rendering; breaks down with 5+ nodes |
| **Indented tree structures** | Hierarchies, taxonomies | Easy to read; can't show cross-links |
| **ASCII tables** | Comparisons, side-by-side analysis | Universal; limited to tabular relationships |
| **Inline notation** | Quick inline relationships | `A → B → C` is clear for simple chains |

**Default choice:** Use **Mermaid** for anything with 3+ nodes and cross-links. Use **inline notation** (`A → B → C`) for simple chains mentioned in passing. Use **ASCII tables** for side-by-side comparisons.

**When to generate visuals:**
- After building a chain (show the chain as a diagram)
- During cross-pollination (side-by-side comparison)
- When the learner's verbal model gets complex enough to benefit from spatial layout (roughly 4+ interconnected concepts)
- When the learner requests it

**When NOT to generate visuals:**
- For single-concept delivery (a diagram of one node is pointless)
- When the relationship is genuinely linear with no branches (inline notation suffices)
- When the learner is overloaded (adding a visual format on top of verbal overload makes it worse)
