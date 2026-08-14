# DLN Skill — Trigger Tests

## SHOULD Trigger

### T1: Direct command
**Input:** "dln"
**Expected:** Activates `dln` and asks the learner to choose or create a domain.

### T2: Cold-start learning
**Input:** "Teach me options pricing from zero."
**Expected:** Activates `dln`, establishes the domain, and routes through the DLN setup flow.

### T3: Resume learning
**Input:** "Continue learning compiler design where I left off."
**Expected:** Activates `dln` and attempts to load the existing domain profile before routing a phase.

### T4: Exam mode
**Input:** "dln exam macroeconomics by 2026-10-20"
**Expected:** Activates `dln` and begins exam metadata capture for macroeconomics.

### T5: Mock request
**Input:** "Run a DLN mock for fixed income."
**Expected:** Activates `dln` and follows the mock workflow for the named domain.

## SHOULD NOT Trigger

### T6: One-off factual question
**Input:** "What is the capital of Portugal?"
**Expected:** Does not activate `dln`; answers directly.

### T7: Existing document summary
**Input:** "Summarize this report in five bullets."
**Expected:** Does not activate `dln` unless the user also asks to begin or resume structured learning.

### T8: Generic study logistics
**Input:** "Remind me to study tomorrow."
**Expected:** Does not activate `dln`.
