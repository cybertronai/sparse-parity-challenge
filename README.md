# Sparse Parity Challenge

Can you solve sparse parity with less energy than a neural network?

Given random {-1, +1} inputs and a label that's the product of k secret bits, find those bits. The catch: we measure not just accuracy and speed, but **data movement** -- how much energy your solution costs at the hardware level.

## Submit a Solution

[![Submit Solution](https://img.shields.io/badge/Submit-Solution-blue?style=for-the-badge)](../../issues/new?template=submission.yml)

Paste a Python function, pick a config, submit. GitHub Actions runs it automatically and posts your results.

### Function signature

```python
def solve(x, y, n_bits, k_sparse):
    """
    Args:
        x: numpy array (n_samples, n_bits), values in {-1, +1}
        y: numpy array (n_samples,), values in {-1, +1}
        n_bits: int, total number of bits
        k_sparse: int, number of secret bits
    Returns:
        list[int]: sorted indices of the k secret bits
    """
```

### Rules

- Only `numpy` allowed (imported as `np`)
- No file I/O, network, or dynamic execution
- 60 second timeout
- Must achieve ≥95% accuracy across 3 seeds
- Evaluated with [TrackedArray](https://github.com/cybertronai/SutroYaro/blob/main/src/sparse_parity/tracked_numpy.py) for automatic DMD measurement

## Leaderboard

Ranked by DMD (Data Movement Distance) -- lower is better.

<!-- LEADERBOARD_START -->
| Rank | Method | Author | DMC | Time | Accuracy | Issue |
|------|--------|--------|-----|------|----------|-------|
| 1 | GF(2) Gaussian Elimination | @SethTS | 11,164,685 | 0.1011s | 100% | [#13](../../issues/13) |
<!-- LEADERBOARD_END -->

## The metric: DMD

Your function's numpy operations are automatically tracked. Every array read has a cost based on how deep the data sits in an LRU stack (how long ago it was last written). DMD = sqrt(stack_distance) per element read. Total DMD = sum of all read DMDs.

Low DMD = data stays near the top of the stack = cache-friendly = less energy.

The neural network baseline (SGD) has DMD ~1.3M. The best known solution (KM influence estimation) has DMD ~3,600. Can you beat it?

## Research context

This challenge comes from the [Sutro Group](https://github.com/cybertronai/SutroYaro), a study group at South Park Commons exploring energy-efficient AI training. 36 experiments across algebraic, information-theoretic, and neural approaches. The full research is at [cybertronai/SutroYaro](https://github.com/cybertronai/SutroYaro).

## Local testing

```bash
git clone https://github.com/cybertronai/sparse-parity-challenge.git
cd sparse-parity-challenge
pip install numpy

# Test your solve() function locally
PYTHONPATH=src python3 -c "
import numpy as np
from sparse_parity.tracked_numpy import TrackedArray, tracking_context
from sparse_parity.lru_tracker import LRUStackTracker

# Your function here
def solve(x, y, n_bits, k_sparse):
    # example: GF(2) or whatever you want to try
    pass

# Generate data
rng = np.random.RandomState(42)
n_bits, k_sparse = 20, 3
secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())
x = rng.choice([-1.0, 1.0], size=(500, n_bits))
y = np.prod(x[:, secret], axis=1)

# Evaluate with tracking
tracker = LRUStackTracker()
with tracking_context(tracker):
    x_t = TrackedArray(x, 'x', tracker)
    y_t = TrackedArray(y, 'y', tracker)
    result = solve(x_t, y_t, n_bits, k_sparse)

print(f'Found: {result}')
print(f'Secret: {secret}')
print(f'Correct: {sorted(result) == secret}')
print(f'DMD: {tracker.summary()[\"dmd\"]:,.1f}')
"
```
