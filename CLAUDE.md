# Sparse Parity Challenge

## What this is

A submission pipeline for the sparse parity benchmark from [SutroYaro](https://github.com/cybertronai/SutroYaro). Anyone can submit a Python function that solves sparse parity. The function is automatically evaluated for accuracy, speed, and energy cost (DMD), then ranked on the leaderboard.

This repo is separate from SutroYaro to keep submission issues and CI runs out of the research repo.

## The problem

Given `x` (n random {-1,+1} inputs) and `y` (product of k secret bits), find the k secret bit indices. Default config: n=20 bits, k=3 secret, 17 noise.

## Submission flow

```
User clicks "Submit Solution" (on docs site, README, or GitHub)
  → GitHub Issue Form: paste a solve() function
  → GitHub Actions: runs harness + TrackedArray (60s timeout)
  → Bot comments: accuracy, DMD, time
  → If passes accuracy gate (≥95%): leaderboard updated
```

## Function signature

```python
def solve(x, y, n_bits, k_sparse):
    """
    Find the secret bit indices.

    Args:
        x: numpy array (n_samples, n_bits), values in {-1, +1}
        y: numpy array (n_samples,), values in {-1, +1}
        n_bits: int, total number of bits
        k_sparse: int, number of secret bits

    Returns:
        list[int]: sorted indices of the k secret bits
    """
    pass
```

## How evaluation works

1. Submitted function is wrapped with `TrackedArray` (numpy subclass that auto-tracks memory access)
2. `LRUStackTracker` records per-element LRU stack distances (Ding et al., arXiv:2312.14441)
3. Accuracy checked on held-out test set (500 samples, different seed)
4. Metrics collected: accuracy, DMD (reads-only), wall time

The submitter does NOT need to know about TrackedArray. Their code uses normal numpy. The harness wraps it automatically.

## Key files

| File | Purpose | Source |
|------|---------|--------|
| `src/harness.py` | Evaluation harness (locked, do not modify) | Copied from SutroYaro |
| `src/sparse_parity/tracker.py` | Legacy MemTracker | Copied from SutroYaro |
| `src/sparse_parity/lru_tracker.py` | LRU stack distance tracker (Ding et al.) | Copied from SutroYaro |
| `src/sparse_parity/tracked_numpy.py` | TrackedArray ndarray subclass | Copied from SutroYaro |
| `src/sparse_parity/config.py` | Experiment config | Copied from SutroYaro |
| `src/sparse_parity/data.py` | Data generation | Copied from SutroYaro |
| `src/sparse_parity/fast.py` | Fast numpy SGD | Copied from SutroYaro |
| `bin/evaluate` | Submission evaluator script | New for this repo |
| `.github/ISSUE_TEMPLATE/submission.yml` | Issue form for submissions | New for this repo |
| `.github/workflows/evaluate.yml` | CI workflow that runs on submission | New for this repo |
| `leaderboard.json` | Current rankings | Auto-updated by CI |

## Relationship to SutroYaro

- **SutroYaro** (`cybertronai/SutroYaro`): research repo. Experiments, findings, agent infrastructure, docs. The source of truth for the harness and TrackedArray.
- **This repo** (`cybertronai/sparse-parity-challenge`): submission pipeline. Issue-based submissions, automated evaluation, leaderboard. References SutroYaro for the science, keeps CI/submission noise separate.

When the harness or TrackedArray changes in SutroYaro, the files here need to be synced. This is a manual copy for now. If it becomes a pain, we can make this repo depend on SutroYaro as a git submodule or pip package.

## Current best methods (from SutroYaro)

| Method | Time (n=20/k=3) | DMD | Notes |
|--------|-----------------|-----|-------|
| KM-min (1 sample) | ~0.001s | 3,578 | DMD leader |
| GF(2) Gaussian Elimination | 509 us | ~203,000 | Fast but high DMD (row ops) |
| KM Influence Estimation | 0.001-0.006s | 20,633 | 5 influence samples per bit |
| SMT Backtracking | 0.002s | 348,336 | Constraint satisfaction |
| SGD (baseline) | 0.12s | 1,278,460 | Neural network baseline |

## For agents

If you're an AI agent working on this repo:
- Read this file first
- The harness files in `src/` are locked -- do not modify them
- Submissions come in as GitHub Issues tagged "submission"
- Your job: parse the function, run evaluation, post results
- The evaluate script at `bin/evaluate` handles the full pipeline
