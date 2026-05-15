#!/usr/bin/env python3
"""
Experiment: RevNet vs baseline SGD — activation storage and DMC

Hypothesis: A 1-block reversible network eliminates h/h_pre activation storage
during backprop, reducing the LRU stack distance for W1 (the dominant cost)
and lowering total DMC. Counter-hypothesis: the extra F and G weight matrices
(W_F, W_G, each H×H) sit in the LRU stack between W_in's forward and backward
reads, increasing its stack distance and worsening DMC.

Prediction from analysis:
- For hidden=200: W_F + W_G = 20,000 floats >> h + h_pre = 400 floats.
  RevNet loses — extra weights dwarf activation savings.
- Crossover at hidden ≈ 2*batch_size (here: ~64 for batch=32).
  Below that, activation memory dominates and RevNet wins.

Answers: Does RevNet's activation-storage saving help in the ByteDMD / MemTracker
energy model for small networks? What determines the crossover point?

Usage:
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_revnet.py
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

EXP_NAME = "exp_revnet"
RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "results" / EXP_NAME

CONFIG = Config(
    n_bits=20,
    k_sparse=3,
    hidden=200,
    lr=0.1,
    wd=0.01,
    batch_size=32,
    max_epochs=200,
    n_train=1000,
    n_test=200,
    seed=42,
)


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def generate_data(config):
    rng = np.random.RandomState(config.seed)
    secret = sorted(rng.choice(config.n_bits, config.k_sparse, replace=False).tolist())
    x_tr = rng.choice([-1.0, 1.0], size=(config.n_train, config.n_bits))
    y_tr = np.prod(x_tr[:, secret], axis=1)
    x_te = rng.choice([-1.0, 1.0], size=(config.n_test, config.n_bits))
    y_te = np.prod(x_te[:, secret], axis=1)
    return x_tr, y_tr, x_te, y_te


def net_accuracy(x_te, y_te, W1, b1, W2, b2):
    h = np.maximum(0.0, x_te @ W1.T + b1)
    out = h @ W2.T + b2
    return float(np.mean(np.sign(out.flatten()) == y_te))


# =============================================================================
# BASELINE: 2-layer MLP
# =============================================================================

def baseline_init(config, rng):
    std1 = np.sqrt(2.0 / config.n_bits)
    std2 = np.sqrt(2.0 / config.hidden)
    W1 = rng.randn(config.hidden, config.n_bits) * std1
    b1 = np.zeros(config.hidden)
    W2 = rng.randn(1, config.hidden) * std2
    b2 = np.zeros(1)
    return W1, b1, W2, b2


def baseline_forward(x, W1, b1, W2, b2):
    h_pre = x @ W1.T + b1
    h = np.maximum(0.0, h_pre)
    out = h @ W2.T + b2
    return h_pre, h, out


def baseline_backward(x, h_pre, h, out, y, W1, b1, W2, b2, config):
    # Hinge loss: only update violated samples (margin < 1)
    margin = y * out.flatten()
    violated = (margin < 1.0)
    if not violated.any():
        return W1, b1, W2, b2
    d_out = np.zeros_like(out)
    d_out[violated, 0] = -y[violated]
    B = len(x)
    dW2 = d_out.T @ h / B
    db2 = d_out.sum(axis=0) / B
    d_h = d_out @ W2
    d_h_pre = d_h * (h_pre > 0)
    dW1 = d_h_pre.T @ x / B
    db1 = d_h_pre.sum(axis=0) / B
    W2 = W2 - config.lr * (dW2 + config.wd * W2)
    b2 = b2 - config.lr * db2
    W1 = W1 - config.lr * (dW1 + config.wd * W1)
    b1 = b1 - config.lr * db1
    return W1, b1, W2, b2


def run_baseline(config):
    rng = np.random.RandomState(config.seed)
    x_tr, y_tr, x_te, y_te = generate_data(config)
    W1, b1, W2, b2 = baseline_init(config, rng)

    best_acc = 0.0
    for epoch in range(config.max_epochs):
        perm = rng.permutation(config.n_train)
        for start in range(0, config.n_train, config.batch_size):
            idx = perm[start:start + config.batch_size]
            xb, yb = x_tr[idx], y_tr[idx]
            h_pre, h, out = baseline_forward(xb, W1, b1, W2, b2)
            W1, b1, W2, b2 = baseline_backward(xb, h_pre, h, out, yb, W1, b1, W2, b2, config)
        acc = net_accuracy(x_te, y_te, W1, b1, W2, b2)
        best_acc = max(best_acc, acc)
        if acc >= 1.0:
            break

    tracker = MemTracker()
    _tracked_baseline_step(x_tr[0:1], y_tr[0:1], W1, b1, W2, b2, tracker)
    s = tracker.summary()

    return {
        "best_test_acc": best_acc,
        "total_epochs": epoch + 1,
        "dmc": round(s["dmc"], 1),
        "ard": round(s["weighted_ard"], 1),
        "total_floats": s["total_floats_accessed"],
        "per_buffer": s.get("per_buffer", {}),
        "n_params": W1.size + b1.size + W2.size + b2.size,
    }


def _tracked_baseline_step(x, y, W1, b1, W2, b2, tracker):
    # Mirrors harness._tracked_sgd_step exactly
    tracker.write("x", x.size)
    tracker.write("W1", W1.size)
    tracker.write("b1", b1.size)
    tracker.read("x")
    tracker.read("W1")
    tracker.read("b1")
    h_pre = x @ W1.T + b1
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

    # Backward
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

    tracker.read("d_h_pre")
    tracker.read("x")
    dW1 = d_h_pre.T @ x
    tracker.write("dW1", dW1.size)
    tracker.write("db1", b1.size)

    # Update reads
    tracker.read("W1")
    tracker.read("dW1")
    tracker.read("W2")
    tracker.read("dW2")
    tracker.read("b1")
    tracker.read("db1")
    tracker.read("b2")
    tracker.read("db2")


# =============================================================================
# REVNET: 1-block reversible network
#
# Architecture:
#   z  = W_in (2H×n) @ x + b_in          [linear embedding, no activation]
#   z1 = z[:H],  z2 = z[H:]              [channel split]
#   y1 = z1 + relu(W_F @ z2 + b_F)       [RevNet block first half]
#   y2 = z2 + relu(W_G @ y1 + b_G)       [RevNet block second half]
#   out = W_out (1×2H) @ [y1;y2] + b_out [output]
#
# Backward:
#   Reconstruct z2 = y2 - relu(W_G @ y1 + b_G)
#   Reconstruct z1 = y1 - relu(W_F @ z2_rec + b_F)
#   No stored h_pre / h — replaced by stored y1, y2 plus a recompute.
# =============================================================================

def revnet_init(config, rng, H=None):
    if H is None:
        H = config.hidden // 2
    n = config.n_bits
    std_in = np.sqrt(2.0 / n)
    std_F = np.sqrt(2.0 / H)
    std_out = np.sqrt(2.0 / (2 * H))
    W_in  = rng.randn(2 * H, n) * std_in
    b_in  = np.zeros(2 * H)
    W_F   = rng.randn(H, H) * std_F
    b_F   = np.zeros(H)
    W_G   = rng.randn(H, H) * std_F
    b_G   = np.zeros(H)
    W_out = rng.randn(1, 2 * H) * std_out
    b_out = np.zeros(1)
    return W_in, b_in, W_F, b_F, W_G, b_G, W_out, b_out


def revnet_forward(x, W_in, b_in, W_F, b_F, W_G, b_G, W_out, b_out):
    z     = x @ W_in.T + b_in
    H     = z.shape[1] // 2
    z1, z2 = z[:, :H], z[:, H:]
    f_pre = z2 @ W_F.T + b_F
    f_act = np.maximum(0.0, f_pre)
    y1    = z1 + f_act
    g_pre = y1 @ W_G.T + b_G
    g_act = np.maximum(0.0, g_pre)
    y2    = z2 + g_act
    out   = np.concatenate([y1, y2], axis=1) @ W_out.T + b_out
    # Store only y1, y2 (not z1, z2, f_pre, g_pre)
    return y1, y2, out


def revnet_backward(x, y1, y2, out, y_true,
                    W_in, b_in, W_F, b_F, W_G, b_G, W_out, b_out, config):
    H = y1.shape[1]
    B = len(x)

    # Hinge loss gradient (d_out NOT pre-divided by B; each dW divided by B below)
    margin = y_true * out.flatten()
    violated = (margin < 1.0)
    if not violated.any():
        return W_in, b_in, W_F, b_F, W_G, b_G, W_out, b_out
    d_out = np.zeros_like(out)
    d_out[violated, 0] = -y_true[violated]

    # Output layer gradients
    y_cat = np.concatenate([y1, y2], axis=1)
    dW_out = d_out.T @ y_cat / B
    db_out = d_out.sum(axis=0) / B
    d_y_cat = d_out @ W_out          # sum gradient (not divided by B)
    d_y1 = d_y_cat[:, :H]
    d_y2 = d_y_cat[:, H:]

    # --- RevNet block backward ---
    # Reconstruct z2 from y2, W_G, y1 (no stored g_pre needed)
    g_pre_rec = y1 @ W_G.T + b_G
    g_act_rec = np.maximum(0.0, g_pre_rec)
    z2_rec    = y2 - g_act_rec

    # Reconstruct z1 from y1, W_F, z2_rec (no stored f_pre needed)
    f_pre_rec = z2_rec @ W_F.T + b_F
    f_act_rec = np.maximum(0.0, f_pre_rec)
    z1_rec    = y1 - f_act_rec

    # Backward through y2 = z2 + G(y1)
    d_z2        = d_y2
    d_g_pre     = d_y2 * (g_pre_rec > 0)
    dW_G        = d_g_pre.T @ y1 / B
    db_G        = d_g_pre.sum(axis=0) / B
    d_y1_from_G = d_g_pre @ W_G

    # Backward through y1 = z1 + F(z2)
    d_y1_total  = d_y1 + d_y1_from_G
    d_z1        = d_y1_total
    d_f_pre     = d_y1_total * (f_pre_rec > 0)
    dW_F        = d_f_pre.T @ z2_rec / B
    db_F        = d_f_pre.sum(axis=0) / B
    d_z2_from_F = d_f_pre @ W_F
    d_z2_total  = d_z2 + d_z2_from_F

    # Reassemble d_z and backward through embedding
    d_z   = np.concatenate([d_z1, d_z2_total], axis=1)
    dW_in = d_z.T @ x / B
    db_in = d_z.sum(axis=0) / B

    # Parameter updates
    wd = config.wd
    lr = config.lr
    W_out = W_out - lr * (dW_out + wd * W_out)
    b_out = b_out - lr * db_out
    W_G   = W_G   - lr * (dW_G  + wd * W_G)
    b_G   = b_G   - lr * db_G
    W_F   = W_F   - lr * (dW_F  + wd * W_F)
    b_F   = b_F   - lr * db_F
    W_in  = W_in  - lr * (dW_in + wd * W_in)
    b_in  = b_in  - lr * db_in
    return W_in, b_in, W_F, b_F, W_G, b_G, W_out, b_out


def run_revnet(config, H=None):
    if H is None:
        H = config.hidden // 2
    # Offset seed so RevNet init is independent of baseline init
    rng = np.random.RandomState(config.seed + 100)
    x_tr, y_tr, x_te, y_te = generate_data(config)
    params = revnet_init(config, rng, H)
    W_in, b_in, W_F, b_F, W_G, b_G, W_out, b_out = params

    best_acc = 0.0
    max_epochs = max(config.max_epochs, 500)  # RevNet needs more epochs than baseline
    for epoch in range(max_epochs):
        perm = rng.permutation(config.n_train)
        for start in range(0, config.n_train, config.batch_size):
            idx = perm[start:start + config.batch_size]
            xb, yb = x_tr[idx], y_tr[idx]
            y1, y2, out = revnet_forward(xb, W_in, b_in, W_F, b_F, W_G, b_G, W_out, b_out)
            W_in, b_in, W_F, b_F, W_G, b_G, W_out, b_out = revnet_backward(
                xb, y1, y2, out, yb, W_in, b_in, W_F, b_F, W_G, b_G, W_out, b_out, config)
        y1_te, y2_te, out_te = revnet_forward(x_te, W_in, b_in, W_F, b_F, W_G, b_G, W_out, b_out)
        acc = float(np.mean(np.sign(out_te.flatten()) == y_te))
        best_acc = max(best_acc, acc)
        if acc >= 1.0:
            break

    tracker = MemTracker()
    _tracked_revnet_step(x_tr[0:1], y_tr[0:1],
                         W_in, b_in, W_F, b_F, W_G, b_G, W_out, b_out, H, tracker)
    s = tracker.summary()

    n_params = (W_in.size + b_in.size + W_F.size + b_F.size +
                W_G.size + b_G.size + W_out.size + b_out.size)
    return {
        "best_test_acc": best_acc,
        "total_epochs": epoch + 1,
        "dmc": round(s["dmc"], 1),
        "ard": round(s["weighted_ard"], 1),
        "total_floats": s["total_floats_accessed"],
        "per_buffer": s.get("per_buffer", {}),
        "n_params": n_params,
        "H": H,
    }


def _tracked_revnet_step(x, y_true, W_in, b_in, W_F, b_F, W_G, b_G, W_out, b_out, H, tracker):
    """One tracked RevNet step for DMC measurement (single sample)."""
    # --- Forward ---
    tracker.write("x", x.size)
    tracker.write("W_in", W_in.size)
    tracker.write("b_in", b_in.size)
    tracker.read("x")
    tracker.read("W_in")
    tracker.read("b_in")
    z = x @ W_in.T + b_in
    tracker.write("z", z.size)

    tracker.write("W_F", W_F.size)
    tracker.write("b_F", b_F.size)
    tracker.read("z")        # read z2 portion
    tracker.read("W_F")
    tracker.read("b_F")
    z1, z2 = z[:, :H], z[:, H:]
    f_pre = z2 @ W_F.T + b_F
    f_act = np.maximum(0.0, f_pre)
    y1 = z1 + f_act
    # RevNet: store y1, NOT f_pre or z1
    tracker.write("y1", y1.size)

    tracker.write("W_G", W_G.size)
    tracker.write("b_G", b_G.size)
    tracker.read("y1")
    tracker.read("W_G")
    tracker.read("b_G")
    g_pre = y1 @ W_G.T + b_G
    g_act = np.maximum(0.0, g_pre)
    y2 = z2 + g_act
    # RevNet: store y2, NOT g_pre or z2
    tracker.write("y2", y2.size)

    tracker.write("W_out", W_out.size)
    tracker.write("b_out", b_out.size)
    tracker.read("y1")
    tracker.read("y2")
    tracker.read("W_out")
    tracker.read("b_out")
    out = np.concatenate([y1, y2], axis=1) @ W_out.T + b_out
    tracker.write("out", out.size)

    # --- Backward ---
    tracker.read("out")
    d_out = (out - y_true.reshape(-1, 1))
    tracker.write("d_out", d_out.size)

    tracker.read("d_out")
    tracker.read("W_out")
    tracker.read("y1")
    tracker.read("y2")
    y_cat = np.concatenate([y1, y2], axis=1)
    dW_out = d_out.T @ y_cat
    d_y_cat = d_out @ W_out
    d_y1 = d_y_cat[:, :H]
    d_y2 = d_y_cat[:, H:]
    tracker.write("dW_out", dW_out.size)
    tracker.write("d_y1", d_y1.size)
    tracker.write("d_y2", d_y2.size)

    # Reconstruct z2: reads W_G, y1 — NOT a stored g_pre
    tracker.read("W_G")
    tracker.read("y1")
    g_pre_rec = y1 @ W_G.T + b_G
    g_act_rec = np.maximum(0.0, g_pre_rec)
    z2_rec = y2 - g_act_rec
    tracker.write("z2_rec", z2_rec.size)

    # Reconstruct z1: reads W_F, z2_rec — NOT a stored f_pre
    tracker.read("W_F")
    tracker.read("z2_rec")
    f_pre_rec = z2_rec @ W_F.T + b_F
    f_act_rec = np.maximum(0.0, f_pre_rec)
    z1_rec = y1 - f_act_rec
    tracker.write("z1_rec", z1_rec.size)

    # Grad through G
    tracker.read("d_y2")
    tracker.read("y1")
    tracker.read("W_G")
    d_g_pre = d_y2 * (g_pre_rec > 0)
    dW_G = d_g_pre.T @ y1
    d_y1_from_G = d_g_pre @ W_G
    tracker.write("dW_G", dW_G.size)
    tracker.write("d_y1_from_G", d_y1_from_G.size)

    # Grad through F
    tracker.read("d_y1")
    tracker.read("d_y1_from_G")
    d_y1_total = d_y1 + d_y1_from_G
    tracker.read("z2_rec")
    tracker.read("W_F")
    d_f_pre = d_y1_total * (f_pre_rec > 0)
    dW_F = d_f_pre.T @ z2_rec
    d_z2_from_F = d_f_pre @ W_F
    tracker.write("dW_F", dW_F.size)
    tracker.write("d_z2_from_F", d_z2_from_F.size)
    tracker.write("d_z2_total", z2_rec.size)

    # Grad through embedding
    d_z = np.concatenate([d_y1_total, d_y2 + d_z2_from_F], axis=1)
    tracker.write("d_z", d_z.size)
    tracker.read("d_z")
    tracker.read("x")
    dW_in = d_z.T @ x
    tracker.write("dW_in", dW_in.size)
    tracker.write("db_in", b_in.size)

    # Update reads
    tracker.read("W_in");  tracker.read("dW_in")
    tracker.read("W_F");   tracker.read("dW_F")
    tracker.read("W_G");   tracker.read("dW_G")
    tracker.read("W_out"); tracker.read("dW_out")
    tracker.read("b_in");  tracker.read("db_in")
    tracker.read("b_out")


# =============================================================================
# DMC BREAKDOWN: activation buffers vs weight buffers
# =============================================================================

BASELINE_ACTIVATION_BUFS = {"h", "h_pre", "d_h", "d_h_pre"}
REVNET_ACTIVATION_BUFS   = {"y1", "y2", "d_y1", "d_y2",
                             "z2_rec", "z1_rec", "d_y1_from_G", "d_z2_from_F", "d_z2_total"}


def dmc_breakdown(per_buffer, activation_keys):
    """Split total DMC into activation vs weight fractions.

    Approximates per-buffer DMC as size * sqrt(avg_dist) * read_count.
    This is slightly conservative vs exact sum(size*sqrt(dist_i)) but
    good enough for comparing fractions across architectures.
    """
    import math
    act_dmc = 0.0
    weight_dmc = 0.0
    for name, stats in per_buffer.items():
        approx = stats["size"] * math.sqrt(max(stats["avg_dist"], 0)) * stats["read_count"]
        if name in activation_keys:
            act_dmc += approx
        else:
            weight_dmc += approx
    return act_dmc, weight_dmc


# =============================================================================
# MAIN
# =============================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'='*70}")
    print(f"  EXPERIMENT: {EXP_NAME}")
    print(f"  n={CONFIG.n_bits}, k={CONFIG.k_sparse}, hidden={CONFIG.hidden}, batch={CONFIG.batch_size}")
    print(f"{'='*70}")

    configs_to_run = [
        ("baseline",        lambda: run_baseline(CONFIG)),
        ("revnet H=100",    lambda: run_revnet(CONFIG, H=100)),
        ("revnet H=36",     lambda: run_revnet(CONFIG, H=36)),   # param-matched
    ]

    results = {}
    for label, fn in configs_to_run:
        print(f"\n  [{label}] training...")
        t0 = time.time()
        r = fn()
        elapsed = time.time() - t0
        act_keys = BASELINE_ACTIVATION_BUFS if label == "baseline" else REVNET_ACTIVATION_BUFS
        act_dmc, weight_dmc = dmc_breakdown(r["per_buffer"], act_keys)
        r["act_dmc"] = round(act_dmc, 1)
        r["weight_dmc"] = round(weight_dmc, 1)
        r["elapsed_s"] = round(elapsed, 2)
        results[label] = r
        acc_pct = f"{r['best_test_acc']:.0%}"
        print(f"    acc={acc_pct}  epochs={r['total_epochs']}  params={r['n_params']:,}")
        print(f"    DMC={r['dmc']:,.0f}  (act={act_dmc:,.0f}  weight={weight_dmc:,.0f})")
        print(f"    ARD={r['ard']:,.0f}")

    # --- Summary table ---
    print(f"\n  {'='*70}")
    print(f"  {'Method':<18} {'Acc':>5} {'Params':>8} {'DMC':>12} {'Act-DMC':>12} {'Wt-DMC':>12}")
    print(f"  {'─'*18} {'─'*5} {'─'*8} {'─'*12} {'─'*12} {'─'*12}")
    for label, r in results.items():
        print(f"  {label:<18} {r['best_test_acc']:>4.0%} {r['n_params']:>8,} "
              f"{r['dmc']:>12,.0f} {r['act_dmc']:>12,.0f} {r['weight_dmc']:>12,.0f}")

    if "baseline" in results and "revnet H=100" in results:
        b = results["baseline"]
        rv = results["revnet H=100"]
        delta_pct = (rv["dmc"] - b["dmc"]) / b["dmc"] * 100
        print(f"\n  RevNet H=100 vs baseline: {delta_pct:+.1f}% DMC")
        print(f"  Activation DMC eliminated: {b['act_dmc']:,.0f} → {rv['act_dmc']:,.0f}")
        print(f"  Weight DMC change:          {b['weight_dmc']:,.0f} → {rv['weight_dmc']:,.0f}")

    # Save
    out = {
        "experiment": EXP_NAME,
        "config": CONFIG.__dict__,
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "per_buffer"}
                    for k, v in results.items()},
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {RESULTS_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
