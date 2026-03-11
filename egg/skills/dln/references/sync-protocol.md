# Shared Sync Loop Protocol

This reference is shared across all DLN phase skills (dln-dot, dln-linear, dln-network).

---

## Learner-Generated Checkpoint

**On agent return** — use the re-anchor payload to prompt a **learner-generated checkpoint**. Do NOT state the summary yourself — ask the learner to produce it:

> "Quick checkpoint — before we move on, summarize where we are. What have we covered so far today, and what's the key takeaway?"

Wait for their response. Compare it against the re-anchor payload. If they miss something significant, prompt:

> "You covered the main points. One thing you didn't mention — [missed item]. Can you connect that to what you just said?"

If they nail it, confirm briefly and move on:

> "Exactly right. Let's continue."

The learner generating the summary is a retrieval event that strengthens retention. The teacher stating the summary is re-study — dramatically less effective.

---

## Plan Adjustment

If the re-anchor payload reveals drift from the original plan, include a **plan adjustment** in the next `dln-sync` dispatch to append:

```
### Plan Adjustment — [reason]
- Reordering: [what changed and why]
- Deferred: [what's pushed to next session]
```

Tell the learner what changed and why: "I'm adjusting our plan — we'll spend more time on [X] before moving to [Y]."

---

## Calibration-Driven Adjustment

When the Calibration Log shows a pattern across 2+ sessions, adjust teaching strategy:

**Overconfident learner** (mean calibration gap > +1.0):
- Increase stress-testing intensity — present harder edge cases earlier.
- Before accepting a comprehension check as "pass," ask one additional probe: "Are you sure? Walk me through your reasoning one more time."
- In the phase gate, use the harder end of the scenario spectrum.
- Never tell the learner they are overconfident. Instead, increase the difficulty until their confidence matches their ability.

**Underconfident learner** (mean calibration gap < -1.0):
- Add more reinforcement — revisit successful chains and name the learner's wins explicitly.
- After comprehension checks, say: "You got that right. That's a solid understanding."
- In worked examples, let the learner lead more — they often know more than they believe.
- Surface the pattern explicitly: "I notice you rate yourself lower than your actual performance. Your understanding is stronger than you think."

**Well-calibrated learner** (mean gap between -1.0 and +1.0):
- Proceed normally. Note in the sync payload that calibration is good.
- Periodically validate: "Your self-assessments have been accurate — that metacognitive skill will serve you well."

---

## Notion Failure Handling

If `dln-sync` returns with `Status.Write: failed`:
1. Log the intended update in-conversation as a visible checkpoint.
2. Queue the failed writes — include them in the next `dln-sync` dispatch payload. (This queue exists only in conversation context.)
3. If 3+ consecutive dispatches return failure, announce to the learner that persistence is temporarily offline. Continue with in-conversation checkpoints only. Attempt a single bulk write-back via `dln-sync` at session end.
