# ML Implementation Special Handling

For ML implementation problems (optimizers, layers, losses, activations), augment the standard teaching flow with the following additional components.

---

## Additional Socratic Questions

- "What problem does this algorithm/layer solve? Why was it invented?"
- "What happens to training if we remove [specific component, e.g., bias correction in Adam]?"
- "Walk me through the shapes at each step. What's the input shape? Output shape?"
- "Where could numerical instability creep in? How do we guard against it?"

---

## Mathematical Foundation

- Present the key equations using clear notation
- Ask the user to explain each term before implementing
- Reference `ml-implementations.md` for standard formulations

---

## Numerical Walkthrough

For every ML implementation, walk through a tiny numerical example:
- Use small tensors (2x2 or 3x3)
- Show intermediate values at each step
- Verify gradients manually where applicable

---

## Implementation Checklist

After the user writes their implementation, verify:
- [ ] Correct state initialization (zeros, ones, time step)
- [ ] Proper gradient handling (in-place vs copy)
- [ ] Numerical stability (epsilon placement, log-sum-exp tricks)
- [ ] Shape consistency (broadcasting, transpose)
- [ ] Edge cases (first step, zero gradients, very large/small values)
