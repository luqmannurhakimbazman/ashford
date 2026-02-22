# ML Implementation Patterns

Reference for ML implementation problems: optimizers, layers, losses, and activations with equations, NumPy templates, Socratic questions, and numerical walkthroughs.

---

## Optimizers

### SGD (Stochastic Gradient Descent)

**Equation:**
```
w = w - lr * grad
```

**NumPy Template:**
```python
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g
```

**Socratic Questions:**
- "Why might vanilla SGD get stuck or oscillate?"
- "What happens if the learning rate is too large? Too small?"

---

### SGD with Momentum

**Equations:**
```
v_t = beta * v_{t-1} + grad_t
w = w - lr * v_t
```

**NumPy Template:**
```python
class SGDMomentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}

    def step(self, params, grads):
        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.velocities:
                self.velocities[i] = np.zeros_like(p)
            self.velocities[i] = self.momentum * self.velocities[i] + g
            p -= self.lr * self.velocities[i]
```

**Socratic Questions:**
- "What real-world analogy captures what momentum does?"
- "Why does momentum help escape shallow local minima?"
- "What happens if beta is too close to 1?"

---

### Adam (Adaptive Moment Estimation)

**Equations:**
```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          # First moment (mean)
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2         # Second moment (variance)
m_hat_t = m_t / (1 - beta1^t)                        # Bias correction
v_hat_t = v_t / (1 - beta2^t)                        # Bias correction
w = w - lr * m_hat_t / (sqrt(v_hat_t) + epsilon)     # Update
```

**NumPy Template:**
```python
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.m:
                self.m[i] = np.zeros_like(p)
                self.v[i] = np.zeros_like(p)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

**Numerical Walkthrough (1D):**
```
Given: lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, g_1=0.5

Step 1 (t=1):
  m_1 = 0.9 * 0 + 0.1 * 0.5 = 0.05
  v_1 = 0.999 * 0 + 0.001 * 0.25 = 0.00025
  m_hat_1 = 0.05 / (1 - 0.9) = 0.5
  v_hat_1 = 0.00025 / (1 - 0.999) = 0.25
  update = 0.001 * 0.5 / (sqrt(0.25) + 1e-8) = 0.001
```

**Socratic Questions:**
- "Why do we need *two* moving averages (first and second moment)?"
- "What would happen without bias correction at t=1? Why are m and v biased toward zero?"
- "Where does epsilon go — inside or outside the square root? Why does it matter?"
- "Why is beta2 typically larger than beta1?"

---

### RMSprop

**Equations:**
```
v_t = beta * v_{t-1} + (1 - beta) * g_t^2
w = w - lr * g_t / (sqrt(v_t) + epsilon)
```

**NumPy Template:**
```python
class RMSprop:
    def __init__(self, lr=0.01, beta=0.99, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.v = {}

    def step(self, params, grads):
        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.v:
                self.v[i] = np.zeros_like(p)
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * g ** 2
            p -= self.lr * g / (np.sqrt(self.v[i]) + self.eps)
```

**Socratic Questions:**
- "How does RMSprop differ from Adam? What's missing?"
- "Why divide by the square root of the second moment?"
- "What problem does RMSprop solve that vanilla SGD doesn't?"

---

### AdaGrad

**Equations:**
```
v_t = v_{t-1} + g_t^2
w = w - lr * g_t / (sqrt(v_t) + epsilon)
```

**NumPy Template:**
```python
class AdaGrad:
    def __init__(self, lr=0.01, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.v = {}

    def step(self, params, grads):
        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.v:
                self.v[i] = np.zeros_like(p)
            self.v[i] += g ** 2
            p -= self.lr * g / (np.sqrt(self.v[i]) + self.eps)
```

**Socratic Questions:**
- "What happens to the effective learning rate over time? Is this a feature or a bug?"
- "Why was RMSprop invented as an improvement over AdaGrad?"

---

### Optimizer Selection Guide

| Optimizer | Best For | Drawback |
|-----------|---------|----------|
| SGD | Simple problems, convex objectives | Slow convergence, sensitive to lr |
| SGD + Momentum | Most deep learning with tuning budget | Need to tune momentum + lr |
| Adam | Default choice, quick convergence | May not generalize as well as SGD+M for some tasks |
| RMSprop | RNNs, non-stationary objectives | No bias correction |
| AdaGrad | Sparse gradients (NLP embeddings) | Learning rate decays to zero |

---

## Layers

### Linear (Fully Connected)

**Forward:**
```
y = x @ W.T + b
```
Where x: (batch, in_features), W: (out_features, in_features), b: (out_features,)

**Backward:**
```
dW = dy.T @ x
db = dy.sum(axis=0)
dx = dy @ W
```

**NumPy Template:**
```python
class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, dy):
        self.dW = dy.T @ self.x
        self.db = dy.sum(axis=0)
        return dy @ self.W
```

**Socratic Questions:**
- "Why do we initialize weights with `sqrt(2/n)` instead of zeros?"
- "What are the shapes at each step? Trace through with batch_size=2, in=3, out=4."
- "Why is the gradient of the bias a sum over the batch dimension?"

---

### Conv2d

**Forward (naive):**
```
For each output position (i, j):
  y[b, oc, i, j] = sum over (ic, kh, kw) of x[b, ic, i+kh, j+kw] * W[oc, ic, kh, kw] + bias[oc]
```

**Shape Reasoning:**
- Input: (batch, in_channels, H, W)
- Kernel: (out_channels, in_channels, kH, kW)
- Output: (batch, out_channels, H_out, W_out)
- H_out = (H + 2*padding - kH) / stride + 1

**Socratic Questions:**
- "What does each dimension of the kernel tensor represent?"
- "How does padding affect the output spatial dimensions?"
- "Why is the naive implementation O(n^6) and how does im2col help?"

---

### BatchNorm

**Forward (training):**
```
mu = mean(x, axis=0)                    # Per-feature mean
var = var(x, axis=0)                     # Per-feature variance
x_hat = (x - mu) / sqrt(var + eps)      # Normalize
y = gamma * x_hat + beta                # Scale and shift

# Update running stats
running_mean = momentum * running_mean + (1 - momentum) * mu
running_var = momentum * running_var + (1 - momentum) * var
```

**Forward (inference):**
```
x_hat = (x - running_mean) / sqrt(running_var + eps)
y = gamma * x_hat + beta
```

**Backward:**
```
dgamma = sum(dy * x_hat, axis=0)
dbeta = sum(dy, axis=0)
dx_hat = dy * gamma
dx = (1/N) * (1/sqrt(var+eps)) * (N * dx_hat - sum(dx_hat) - x_hat * sum(dx_hat * x_hat))
```

**NumPy Template:**
```python
class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x, training=True):
        if training:
            mu = x.mean(axis=0)
            var = x.var(axis=0)
            self.x_hat = (x - mu) / np.sqrt(var + self.eps)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            # Cache for backward
            self.mu, self.var, self.x = mu, var, x
        else:
            self.x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.gamma * self.x_hat + self.beta
```

**Socratic Questions:**
- "Why normalize per-feature rather than per-sample?"
- "Why do we need gamma and beta if we just normalized?"
- "Why are running stats different from batch stats during training?"
- "What happens if batch size is 1? Why is this a problem?"

---

## Losses

### Cross-Entropy Loss

**Equation:**
```
L = -sum(y_true * log(y_pred))                    # Multi-class
L = -(1/N) * sum(log(y_pred[i, y_true[i]]))       # With class indices
```

**Numerical Stability:**
```python
def cross_entropy(logits, targets):
    # Numerically stable: subtract max before exp
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted).sum(axis=-1))
    return -(shifted[range(len(targets)), targets] - log_sum_exp).mean()
```

**Socratic Questions:**
- "Why subtract the max before taking exp? What happens without it?"
- "What's the gradient of cross-entropy with respect to logits? Why is it so clean?"

---

### MSE (Mean Squared Error)

**Equation:**
```
L = (1/N) * sum((y_pred - y_true)^2)
```

**Gradient:**
```
dL/dy_pred = (2/N) * (y_pred - y_true)
```

**Socratic Questions:**
- "When would you choose MSE over cross-entropy?"
- "Why does MSE penalize large errors more than small ones?"

---

### Binary Cross-Entropy

**Equation:**
```
L = -(1/N) * sum(y * log(p) + (1-y) * log(1-p))
```

**Numerical Stability:**
```python
def binary_cross_entropy(logits, targets):
    # Stable: use log-sigmoid trick
    return np.maximum(logits, 0) - logits * targets + np.log(1 + np.exp(-np.abs(logits)))
```

**Socratic Questions:**
- "Why not just use `log(sigmoid(x))`? What goes wrong numerically?"
- "Derive the gradient. Why is it just `sigmoid(x) - y`?"

---

## Activations

### ReLU

**Forward:** `y = max(0, x)`
**Backward:** `dy/dx = 1 if x > 0 else 0`

**Gotchas:**
- Dying ReLU problem: neurons that output 0 for all inputs get zero gradient and never recover
- Leaky ReLU fixes this: `y = max(alpha * x, x)` where alpha is small (0.01)

### Sigmoid

**Forward:** `y = 1 / (1 + exp(-x))`
**Backward:** `dy/dx = y * (1 - y)`

**Gotchas:**
- Vanishing gradients for large |x|
- Output not zero-centered (can cause zig-zag gradient updates)
- Numerical stability: `sigmoid(x) = exp(x) / (1 + exp(x))` for x < 0

### Tanh

**Forward:** `y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
**Backward:** `dy/dx = 1 - y^2`

**Gotchas:**
- Still suffers from vanishing gradients (less than sigmoid)
- Zero-centered (advantage over sigmoid)

### Softmax

**Forward:** `y_i = exp(x_i) / sum(exp(x_j))`
**Backward (Jacobian):** `J_ij = y_i * (delta_ij - y_j)`

**Gotchas:**
- Must subtract max for numerical stability: `exp(x_i - max(x))`
- The Jacobian is a full matrix, but when combined with cross-entropy, the gradient simplifies to `y - one_hot(target)`

**Socratic Questions:**
- "Why does softmax + cross-entropy have such a clean gradient?"
- "What happens to softmax outputs when inputs are very large? Very close together?"

---

## Common Implementation Gotchas

| Gotcha | Problem | Fix |
|--------|---------|-----|
| State initialization | Using random instead of zeros for optimizer moments | Always initialize moments to zeros |
| Time step tracking | Starting t at 0 instead of 1 for Adam | Initialize t=0, increment *before* use |
| Epsilon placement | `sqrt(v) + eps` vs `sqrt(v + eps)` | PyTorch uses `sqrt(v) + eps`, papers vary — be consistent |
| In-place operations | Modifying tensors that are needed for backward pass | Always use copies for cached values |
| Shape mismatches | Broadcasting errors in batch operations | Explicitly verify shapes at each step |
| Running stats | Updating running mean/var during inference | Gate with training flag |
| Weight initialization | Using zeros (symmetric neurons) or large values (exploding) | Xavier/He initialization |
| Gradient accumulation | Not zeroing gradients between batches | Zero grads before each forward pass |
