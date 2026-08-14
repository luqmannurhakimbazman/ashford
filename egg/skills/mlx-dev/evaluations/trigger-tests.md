# Trigger Tests: mlx-dev

**Test types:** `MANUAL` -- requires a live Claude Code session.

## Should Activate `MANUAL`

### 1. Write MLX code
- **Query:** "write an MLX training loop for this model"
- **Expected:** mlx-dev activates

### 2. PyTorch migration
- **Query:** "port this PyTorch model to Apple MLX"
- **Expected:** mlx-dev activates

### 3. Lazy evaluation debugging
- **Query:** "why does this MLX loop keep growing memory until mx.eval?"
- **Expected:** mlx-dev activates

### 4. MLX indexing issue
- **Query:** "fix this MLX array indexing error with a list of indices"
- **Expected:** mlx-dev activates

### 5. Apple Silicon optimization
- **Query:** "optimize this mlx-lm inference code for Apple Silicon"
- **Expected:** mlx-dev activates

## Should NOT Activate `MANUAL`

### 6. PyTorch-only request
- **Query:** "write a PyTorch CUDA training loop"
- **Expected:** Does NOT activate

### 7. General macOS development
- **Query:** "build a SwiftUI settings screen for macOS"
- **Expected:** Does NOT activate

### 8. ML paper request
- **Query:** "draft a paper about our model results"
- **Expected:** ml-paper-writing activates, NOT mlx-dev
