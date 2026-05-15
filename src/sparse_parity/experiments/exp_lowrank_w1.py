#!/usr/bin/env python3
"""
Experiment: Low-rank W1 factorization — does rank match sparsity?

Hypothesis: Sparse parity with k=3 secret bits has a 3-dimensional signal.
A rank-r factorization W1 = U(hidden×r) @ V(r×n_bits) should reach full
accuracy at r=k=3, since the network only needs to detect 3 input directions.
If true, U and V are 6× smaller than full W1, reducing their LRU stack
distances and cutting DMC significantly.

Prediction:
- Accuracy: converges at r >= k (r=3 for k=3 parity)
- DMC:      drops roughly proportionally to r/hidden as r decreases,
            since U (hidden×r) is the dominant buffer

Sweep: r in {1, 2, 3, 4, 5, 8, 10, 20} vs full W1 baseline (r=hidden=200)

Usage:
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_lowrank_w1.py
"""

import time
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sparse_parity.config import Config
from sparse_parity.tracker import MemTracker

EXP_NAME = "exp_lowrank_w1"
RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "results" / EXP_NAME

CONFIG = Config(
    n_bits=20,
    k_sparse=3,
    hidden=200,
    lr=0.1,
    wd=0.01,
    batch_size=32,
    max_epochs=500,
    n_train=1000,
    n_test=200,
    seed=42,
)

RANKS = [1, 2, 3, 4, 5, 8, 10, 20]


# =============================================================================
# DATA
# =============================================================================

def generate_data(config):
    rng = np.random.RandomState(config.seed)
    secret = sorted(rng.choice(config.n_bits, config.k_sparse, replace=False).tolist())
    x_tr = rng.choice([-1.0, 1.0], size=(config.n_train, config.n_bits))
    y_tr = np.prod(x_tr[:, secret], axis=1)
    x_te = rng.choice([-1.0, 1.0], size=(config.n_test, config.n_bits))
    y_te = np.prod(x_te[:, secret], axis=1)
    return x_tr, y_tr, x_te, y_te, secret


# =============================================================================
# LOW-RANK MODEL
#
# Forward:  z = x @ V.T          (n_bits → r)
#           h_pre = z @ U.T + b1  (r → hidden)
#           h = relu(h_pre)
#           out = h @ W2.T + b2   (hidden → 1)
#
# Parameters: U (hidden×r), V (r×n_bits), b1 (hidden,), W2 (1×hidden), b2 (1,)
# =============================================================================

def init_params(config, rank, rng):
    std_V  = np.sqrt(2.0 / config.n_bits)
    std_U  = np.sqrt(2.0 / rank)
    std_W2 = np.sqrt(2.0 / config.hidden)
    U  = rng.randn(config.hidden, rank) * std_U
    V  = rng.randn(rank, config.n_bits) * std_V
    b1 = np.zeros(config.hidden)
    W2 = rng.randn(1, config.hidden) * std_W2
    b2 = np.zeros(1)
    return U, V, b1, W2, b2


def forward(x, U, V, b1, W2, b2):
    z     = x @ V.T           # (B, r)
    h_pre = z @ U.T + b1      # (B, hidden)
    h     = np.maximum(0.0, h_pre)
    out   = h @ W2.T + b2     # (B, 1)
    return z, h_pre, h, out


def backward(x, z, h_pre, h, out, y, U, V, b1, W2, b2, config):
    B = len(x)
    margin = y * out.flatten()
    violated = margin < 1.0
    if not violated.any():
        return U, V, b1, W2, b2

    d_out = np.zeros_like(out)
    d_out[violated, 0] = -y[violated]

    # Output layer
    dW2 = d_out.T @ h / B
    db2 = d_out.sum(axis=0) / B
    d_h = d_out @ W2                    # (B, hidden)

    # Hidden layer (ReLU)
    d_h_pre = d_h * (h_pre > 0)         # (B, hidden)
    dU  = d_h_pre.T @ z / B             # (hidden, r)
    db1 = d_h_pre.sum(axis=0) / B
    d_z = d_h_pre @ U                   # (B, r)

    # Projection layer
    dV = d_z.T @ x / B                  # (r, n_bits)

    wd = config.wd
    lr = config.lr
    W2 = W2 - lr * (dW2 + wd * W2)
    b2 = b2 - lr * db2
    U  = U  - lr * (dU  + wd * U)
    b1 = b1 - lr * db1
    V  = V  - lr * (dV  + wd * V)
    return U, V, b1, W2, b2


# =============================================================================
# DIRECT W1 BASELINE  (standard unfactored MLP, matches harness)
# =============================================================================

def run_direct_baseline(config):
    """Standard 2-layer MLP with a single W1 (hidden×n_bits) matrix."""
    rng = np.random.RandomState(config.seed)
    x_tr, y_tr, x_te, y_te, _ = generate_data(config)

    std1 = np.sqrt(2.0 / config.n_bits)
    std2 = np.sqrt(2.0 / config.hidden)
    W1 = rng.randn(config.hidden, config.n_bits) * std1
    b1 = np.zeros(config.hidden)
    W2 = rng.randn(1, config.hidden) * std2
    b2 = np.zeros(1)

    best_acc = 0.0
    converged_epoch = None
    for epoch in range(config.max_epochs):
        perm = rng.permutation(config.n_train)
        for start in range(0, config.n_train, config.batch_size):
            idx = perm[start:start + config.batch_size]
            xb, yb = x_tr[idx], y_tr[idx]
            h_pre = xb @ W1.T + b1
            h = np.maximum(0.0, h_pre)
            out = h @ W2.T + b2
            margin = yb * out.flatten()
            violated = margin < 1.0
            if violated.any():
                B = len(xb)
                d_out = np.zeros_like(out)
                d_out[violated, 0] = -yb[violated]
                dW2 = d_out.T @ h / B
                db2 = d_out.sum(0) / B
                d_h = d_out @ W2
                d_h_pre = d_h * (h_pre > 0)
                dW1 = d_h_pre.T @ xb / B
                db1_g = d_h_pre.sum(0) / B
                W2 -= config.lr * (dW2 + config.wd * W2)
                b2 -= config.lr * db2
                W1 -= config.lr * (dW1 + config.wd * W1)
                b1 -= config.lr * db1_g

        h_te = np.maximum(0.0, x_te @ W1.T + b1)
        acc = float(np.mean(np.sign((h_te @ W2.T + b2).flatten()) == y_te))
        if acc > best_acc:
            best_acc = acc
        if acc >= 1.0 and converged_epoch is None:
            converged_epoch = epoch + 1
            break

    tracker = MemTracker()
    _tracked_direct_step(x_tr[0:1], y_tr[0:1], W1, b1, W2, b2, tracker)
    s = tracker.summary()

    W1_buf = s.get("per_buffer", {}).get("W1", {})
    w1_dmc = (W1_buf.get("size", 0) *
              math.sqrt(max(W1_buf.get("avg_dist", 0), 0)) *
              W1_buf.get("read_count", 0))
    return {
        "rank": "direct",
        "best_acc": best_acc,
        "converged_epoch": converged_epoch,
        "total_epochs": epoch + 1,
        "dmc": round(s["dmc"], 1),
        "ard": round(s["weighted_ard"], 1),
        "total_floats": s["total_floats_accessed"],
        "n_params": W1.size + b1.size + W2.size + b2.size,
        "U_dmc": round(w1_dmc, 1),  # labelled U_dmc for consistent table display
        "per_buffer": s.get("per_buffer", {}),
    }


def _tracked_direct_step(x, y, W1, b1, W2, b2, tracker):
    """Mirrors harness._tracked_sgd_step exactly."""
    tracker.write("x", x.size)
    tracker.write("W1", W1.size)
    tracker.write("b1", b1.size)
    tracker.read("x"); tracker.read("W1"); tracker.read("b1")
    h_pre = x @ W1.T + b1
    tracker.write("h_pre", h_pre.size)
    tracker.read("h_pre")
    h = np.maximum(0.0, h_pre)
    tracker.write("h", h.size)
    tracker.write("W2", W2.size); tracker.write("b2", b2.size)
    tracker.read("h"); tracker.read("W2"); tracker.read("b2")
    out = h @ W2.T + b2
    tracker.write("out", out.size)
    tracker.read("out")
    d_out = out - y.reshape(-1, 1)
    tracker.write("d_out", d_out.size)
    tracker.read("d_out"); tracker.read("W2"); tracker.read("h")
    dW2 = d_out.T @ h
    tracker.write("dW2", dW2.size); tracker.write("db2", 1)
    tracker.read("d_out"); tracker.read("W2")
    d_h = d_out @ W2
    tracker.write("d_h", d_h.size)
    tracker.read("d_h"); tracker.read("h_pre")
    d_h_pre = d_h * (h_pre > 0)
    tracker.write("d_h_pre", d_h_pre.size)
    tracker.read("d_h_pre"); tracker.read("x")
    dW1 = d_h_pre.T @ x
    tracker.write("dW1", dW1.size); tracker.write("db1", b1.size)
    tracker.read("W1"); tracker.read("dW1")
    tracker.read("W2"); tracker.read("dW2")
    tracker.read("b1"); tracker.read("db1")
    tracker.read("b2"); tracker.read("db2")


# =============================================================================
# TRAINING
# =============================================================================

def run(config, rank):
    rng = np.random.RandomState(config.seed)
    x_tr, y_tr, x_te, y_te, secret = generate_data(config)
    U, V, b1, W2, b2 = init_params(config, rank, rng)

    best_acc = 0.0
    converged_epoch = None
    for epoch in range(config.max_epochs):
        perm = rng.permutation(config.n_train)
        for start in range(0, config.n_train, config.batch_size):
            idx = perm[start:start + config.batch_size]
            xb, yb = x_tr[idx], y_tr[idx]
            z, h_pre, h, out = forward(xb, U, V, b1, W2, b2)
            U, V, b1, W2, b2 = backward(xb, z, h_pre, h, out, yb, U, V, b1, W2, b2, config)

        _, _, h_te, out_te = forward(x_te, U, V, b1, W2, b2)
        acc = float(np.mean(np.sign(out_te.flatten()) == y_te))
        if acc > best_acc:
            best_acc = acc
        if acc >= 1.0 and converged_epoch is None:
            converged_epoch = epoch + 1
            break

    tracker = MemTracker()
    _tracked_step(x_tr[0:1], y_tr[0:1], U, V, b1, W2, b2, tracker)
    s = tracker.summary()

    n_params = U.size + V.size + b1.size + W2.size + b2.size
    return {
        "rank": rank,
        "best_acc": best_acc,
        "converged_epoch": converged_epoch,
        "total_epochs": epoch + 1,
        "dmc": round(s["dmc"], 1),
        "ard": round(s["weighted_ard"], 1),
        "total_floats": s["total_floats_accessed"],
        "n_params": n_params,
        "per_buffer": s.get("per_buffer", {}),
    }


# =============================================================================
# TRACKED STEP  (single sample, mirrors harness protocol)
# =============================================================================

def _tracked_step(x, y, U, V, b1, W2, b2, tracker):
    # --- Forward ---
    tracker.write("x",  x.size)
    tracker.write("V",  V.size)
    tracker.read("x")
    tracker.read("V")
    z = x @ V.T
    tracker.write("z", z.size)

    tracker.write("U",  U.size)
    tracker.write("b1", b1.size)
    tracker.read("z")
    tracker.read("U")
    tracker.read("b1")
    h_pre = z @ U.T + b1
    tracker.write("h_pre", h_pre.size)

    tracker.read("h_pre")
    h = np.maximum(0.0, h_pre)
    tracker.write("h", h.size)

    tracker.write("W2", W2.size)
    tracker.write("b2", b2.size)
    tracker.read("h")
    tracker.read("W2")
    tracker.read("b2")
    out = h @ W2.T + b2
    tracker.write("out", out.size)

    # --- Backward ---
    tracker.read("out")
    d_out = (out - y.reshape(-1, 1))
    tracker.write("d_out", d_out.size)

    tracker.read("d_out")
    tracker.read("W2")
    tracker.read("h")
    dW2 = d_out.T @ h
    tracker.write("dW2", dW2.size)
    tracker.write("db2", 1)

    tracker.read("d_out")
    tracker.read("W2")
    d_h = d_out @ W2
    tracker.write("d_h", d_h.size)

    tracker.read("d_h")
    tracker.read("h_pre")
    d_h_pre = d_h * (h_pre > 0)
    tracker.write("d_h_pre", d_h_pre.size)

    # dU = d_h_pre.T @ z
    tracker.read("d_h_pre")
    tracker.read("z")
    dU = d_h_pre.T @ z
    tracker.write("dU", dU.size)
    tracker.write("db1", b1.size)

    # d_z = d_h_pre @ U
    tracker.read("d_h_pre")
    tracker.read("U")
    d_z = d_h_pre @ U
    tracker.write("d_z", d_z.size)

    # dV = d_z.T @ x
    tracker.read("d_z")
    tracker.read("x")
    dV = d_z.T @ x
    tracker.write("dV", dV.size)

    # Update reads
    tracker.read("U");  tracker.read("dU")
    tracker.read("V");  tracker.read("dV")
    tracker.read("W2"); tracker.read("dW2")
    tracker.read("b1"); tracker.read("db1")
    tracker.read("b2"); tracker.read("db2")


# =============================================================================
# MAIN
# =============================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*72}")
    print(f"  EXPERIMENT: {EXP_NAME}")
    print(f"  n={CONFIG.n_bits}, k={CONFIG.k_sparse}, hidden={CONFIG.hidden}, "
          f"batch={CONFIG.batch_size}, max_epochs={CONFIG.max_epochs}")
    print(f"  Ranks: {RANKS}")
    print(f"{'='*72}")

    # Direct (unfactored) baseline first
    print(f"  rank=direct  ", end="", flush=True)
    t0 = time.time()
    baseline = run_direct_baseline(CONFIG)
    baseline["elapsed_s"] = round(time.time() - t0, 2)
    conv = f"ep {baseline['converged_epoch']}" if baseline["converged_epoch"] else f">{baseline['total_epochs']}"
    print(f"acc={baseline['best_acc']:.0%}  {conv:<8}  "
          f"params={baseline['n_params']:>6,}  DMC={baseline['dmc']:>12,.0f}  "
          f"W1_dmc={baseline['U_dmc']:>10,.0f}")

    results = []
    for rank in RANKS:
        t0 = time.time()
        r = run(CONFIG, rank)
        elapsed = time.time() - t0
        r["elapsed_s"] = round(elapsed, 2)

        U_buf = r["per_buffer"].get("U", {})
        u_dmc = (U_buf.get("size", 0) *
                 math.sqrt(max(U_buf.get("avg_dist", 0), 0)) *
                 U_buf.get("read_count", 0))
        r["U_dmc"] = round(u_dmc, 1)

        conv = f"ep {r['converged_epoch']}" if r["converged_epoch"] else f">{r['total_epochs']}"
        print(f"  rank={rank:>3}  acc={r['best_acc']:.0%}  {conv:<8}  "
              f"params={r['n_params']:>6,}  DMC={r['dmc']:>12,.0f}  U_dmc={r['U_dmc']:>10,.0f}")
        results.append(r)

    # Summary table vs direct baseline
    print(f"\n  {'='*74}")
    print(f"  {'rank':>6}  {'acc':>5}  {'params':>8}  {'DMC':>12}  {'vs direct':>10}  {'W1/U dmc':>10}")
    print(f"  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*12}  {'─'*10}  {'─'*10}")
    for r in [baseline] + results:
        delta = (r["dmc"] - baseline["dmc"]) / baseline["dmc"] * 100
        conv_marker = " *" if r["converged_epoch"] else "  "
        rank_str = str(r["rank"])
        print(f"  {rank_str:>6}  {r['best_acc']:>4.0%}  {r['n_params']:>8,}  "
              f"{r['dmc']:>12,.0f}  {delta:>+9.1f}%{conv_marker}  {r['U_dmc']:>10,.0f}")
    print(f"  (* = converged to 100%)")

    # Key findings
    converged = [r for r in results if r["best_acc"] >= 1.0]
    if converged:
        min_rank = min(r["rank"] for r in converged)
        min_rank_r = next(r for r in converged if r["rank"] == min_rank)
        dmc_saving = (baseline["dmc"] - min_rank_r["dmc"]) / baseline["dmc"] * 100
        print(f"\n  Minimum converging rank: {min_rank}")
        print(f"  DMC at rank {min_rank}: {min_rank_r['dmc']:,.0f}  "
              f"({dmc_saving:.1f}% reduction vs direct W1 baseline)")

    # Save
    out = {
        "experiment": EXP_NAME,
        "config": CONFIG.__dict__,
        "ranks": RANKS,
        "direct_baseline": {k: v for k, v in baseline.items() if k != "per_buffer"},
        "results": [{k: v for k, v in r.items() if k != "per_buffer"} for r in results],
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {RESULTS_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
