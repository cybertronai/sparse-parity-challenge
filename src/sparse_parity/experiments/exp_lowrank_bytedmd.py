#!/usr/bin/env python3
"""
Experiment: Low-rank W1 factorization — ByteDMD cost sweep over rank

Hypothesis: Factoring W1 = U(hidden×r) @ V(r×n_bits) reduces ByteDMD cost
proportionally to r, because U and V are smaller and stay closer to the top
of the LRU stack. The minimum rank that solves k=3 parity is r=k=3, giving
the lowest ByteDMD achievable for a gradient-based neural approach.

Comparison targets (from exp_bytedmd_floor_gap):
  KM-min:  268
  GF(2):   101,501

All implementations are pure Python (no numpy) so ByteDMD tracks every read.
Uses the same demo config as the floor-gap SGD: tiny network, short training,
does not converge at this scale. Purpose is ByteDMD profile comparison only.

Usage:
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_lowrank_bytedmd.py
"""

import math
import time
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bytedmd import bytedmd

N_BITS   = 20
K_SPARSE = 3
SEED     = 42

# Demo config — same as floor-gap SGD. Does not converge; ByteDMD profile only.
HIDDEN  = 8
BATCH   = 4
N_TRAIN = 8
EPOCHS  = 2
LR      = 0.1

RANKS = [1, 2, 3, 4, 5, 8, 10, 20]

# From exp_bytedmd_floor_gap
KNOWN = {"km_min": 268, "gf2": 101_501}

GEOMETRIC_LOWER_BOUND_FACTOR = 0.3849

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "exp_lowrank_bytedmd"


# =============================================================================
# DATA GENERATION (outside bytedmd — oracle only)
# =============================================================================

def _parity(x, secret):
    result = 1
    for i in secret:
        result *= x[i]
    return result


def make_data(n_bits, k_sparse, seed, n_train):
    rng = random.Random(seed)
    bits = list(range(n_bits)); rng.shuffle(bits)
    secret = sorted(bits[:k_sparse])
    rng2 = random.Random(seed + 4)
    flat_xs, ys = [], []
    for _ in range(n_train):
        x = [rng2.choice([-1, 1]) for _ in range(n_bits)]
        flat_xs.extend(x)
        ys.append(_parity(x, secret))
    return flat_xs, ys, secret


def make_init(n_bits, hidden, rank, seed):
    """Flat lists: v (rank×n_bits), u (hidden×rank), b1 (hidden), w2 (hidden), b2 (scalar)."""
    rng = random.Random(seed + 5)
    v  = [rng.uniform(-0.3, 0.3) for _ in range(rank * n_bits)]
    u  = [rng.uniform(-0.3, 0.3) for _ in range(hidden * rank)]
    b1 = [0.0] * hidden
    w2 = [rng.uniform(-0.3, 0.3) for _ in range(hidden)]
    b2 = 0.0
    return v, u, b1, w2, b2


# =============================================================================
# PURE-PYTHON LOW-RANK SGD  (traced by bytedmd)
#
# Architecture:  z[l]    = sum_i  V[l,i] * x[i]          (n_bits → rank)
#                h_pre[j] = b1[j] + sum_l  U[j,l] * z[l]  (rank → hidden)
#                h[j]    = relu(h_pre[j])
#                out     = b2 + sum_j  w2[j] * h[j]        (hidden → 1)
#
# Backward: hinge loss (d_out = -y if y*out < 1 else 0),
#           chain rule through w2 → h → u → v.
#
# Saliency prediction: column norms of V identify which input bits the
# network concentrated on.  Top-k columns = predicted secret.
# =============================================================================

def lowrank_solve(flat_xs, ys, v, u, b1, w2, b2,
                  n_bits, hidden, rank, n_train, batch, epochs, lr, k_sparse):
    v  = list(v)
    u  = list(u)
    b1 = list(b1)
    w2 = list(w2)

    for _epoch in range(epochs):
        for batch_start in range(0, n_train, batch):
            gv  = [0.0] * (rank * n_bits)
            gu  = [0.0] * (hidden * rank)
            gb1 = [0.0] * hidden
            gw2 = [0.0] * hidden
            gb2 = 0.0

            for s in range(batch_start, min(batch_start + batch, n_train)):
                base = s * n_bits

                # --- Forward ---
                # z = V @ x  (rank vector)
                z = []
                for l in range(rank):
                    acc = 0.0
                    for i in range(n_bits):
                        acc = acc + v[l * n_bits + i] * flat_xs[base + i]
                    z.append(acc)

                # h_pre = U @ z + b1  (hidden vector)
                h_pre = []
                for j in range(hidden):
                    acc = b1[j]
                    for l in range(rank):
                        acc = acc + u[j * rank + l] * z[l]
                    h_pre.append(acc)

                # h = relu(h_pre)
                h = [v_h if v_h > 0 else 0.0 for v_h in h_pre]

                # out = w2 · h + b2
                out = b2
                for j in range(hidden):
                    out = out + w2[j] * h[j]

                # --- Hinge loss gradient ---
                if ys[s] * out < 1.0:
                    d_out = -ys[s]
                else:
                    d_out = 0.0

                # --- Backward ---
                gb2 = gb2 + d_out
                d_h = []
                for j in range(hidden):
                    gw2[j] = gw2[j] + d_out * h[j]
                    d_h.append(d_out * w2[j])

                d_z = [0.0] * rank
                for j in range(hidden):
                    if h_pre[j] > 0:
                        d_pre = d_h[j]
                        gb1[j] = gb1[j] + d_pre
                        for l in range(rank):
                            gu[j * rank + l] = gu[j * rank + l] + d_pre * z[l]
                            d_z[l] = d_z[l] + d_pre * u[j * rank + l]

                for l in range(rank):
                    for i in range(n_bits):
                        gv[l * n_bits + i] = gv[l * n_bits + i] + d_z[l] * flat_xs[base + i]

            # --- SGD update ---
            bs = min(batch, n_train - batch_start)
            scale = lr / bs
            for k in range(rank * n_bits):
                v[k] = v[k] - scale * gv[k]
            for k in range(hidden * rank):
                u[k] = u[k] - scale * gu[k]
            for j in range(hidden):
                b1[j] = b1[j] - scale * gb1[j]
                w2[j] = w2[j] - scale * gw2[j]
            b2 = b2 - scale * gb2

    # --- Predict: top-k input bits by V column norm ---
    saliency = []
    for i in range(n_bits):
        col_sq = 0.0
        for l in range(rank):
            val = v[l * n_bits + i]
            col_sq = col_sq + val * val
        saliency.append(col_sq)

    indexed = sorted(range(n_bits), key=lambda i: -saliency[i])
    return sorted(indexed[:k_sparse])


# =============================================================================
# MAIN
# =============================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    flat_xs, ys, secret = make_data(N_BITS, K_SPARSE, SEED, N_TRAIN)

    print(f"n={N_BITS}, k={K_SPARSE}, secret={secret}")
    print(f"demo config: hidden={HIDDEN}, batch={BATCH}, "
          f"n_train={N_TRAIN}, epochs={EPOCHS}  (does not converge)\n")

    results = {}
    for rank in RANKS:
        v, u, b1, w2, b2 = make_init(N_BITS, HIDDEN, rank, SEED)
        args = (flat_xs, ys, v, u, b1, w2, b2,
                N_BITS, HIDDEN, rank, N_TRAIN, BATCH, EPOCHS, LR, K_SPARSE)

        t0 = time.time()
        cost = bytedmd(lowrank_solve, args)
        elapsed = time.time() - t0

        results[rank] = {"bytedmd": cost, "elapsed_ms": round(elapsed * 1000, 1)}
        print(f"  rank={rank:>3}  ByteDMD={cost:>10,}  ({elapsed*1000:.0f}ms)")

    # Reference: full-rank equivalent (rank = min(hidden, n_bits) = n_bits for hidden >= n_bits)
    # Use rank=n_bits as the "direct W1" proxy
    full_rank = N_BITS
    v, u, b1, w2, b2 = make_init(N_BITS, HIDDEN, full_rank, SEED)
    args_full = (flat_xs, ys, v, u, b1, w2, b2,
                 N_BITS, HIDDEN, full_rank, N_TRAIN, BATCH, EPOCHS, LR, K_SPARSE)
    t0 = time.time()
    cost_full = bytedmd(lowrank_solve, args_full)
    elapsed_full = time.time() - t0
    results[full_rank] = {"bytedmd": cost_full, "elapsed_ms": round(elapsed_full * 1000, 1)}
    print(f"  rank={full_rank:>3} (full) ByteDMD={cost_full:>10,}  ({elapsed_full*1000:.0f}ms)")

    # Summary table
    floor_n = sum(math.ceil(math.sqrt(i + 1)) for i in range(N_BITS))
    print(f"\n{'='*72}")
    print(f"  {'method':<18}  {'ByteDMD':>10}  {'vs read-n':>10}  {'Geom LB':>10}")
    print(f"  {'─'*18}  {'─'*10}  {'─'*10}  {'─'*10}")

    for name, val in KNOWN.items():
        ratio = val / floor_n
        glb   = val * GEOMETRIC_LOWER_BOUND_FACTOR
        print(f"  {name:<18}  {val:>10,}  {ratio:>9.1f}x  {glb:>10,.0f}")

    for rank in RANKS + [full_rank]:
        c = results[rank]["bytedmd"]
        ratio = c / floor_n
        glb   = c * GEOMETRIC_LOWER_BOUND_FACTOR
        label = f"lowrank r={rank}" if rank != full_rank else f"lowrank r={rank} (full)"
        print(f"  {label:<18}  {c:>10,}  {ratio:>9.1f}x  {glb:>10,.0f}")

    print(f"  {'read-n floor':<18}  {floor_n:>10,}  {'1.0x':>10}  {'─':>10}")
    print(f"{'='*72}")

    # Key finding
    if K_SPARSE in results:
        r3 = results[K_SPARSE]["bytedmd"]
        vs_full = (cost_full - r3) / cost_full * 100
        vs_km   = r3 / KNOWN["km_min"]
        vs_gf2  = r3 / KNOWN["gf2"]
        print(f"\n  rank=k={K_SPARSE} ByteDMD: {r3:,}")
        print(f"    {vs_full:.1f}% cheaper than full-rank (same demo config)")
        print(f"    {vs_km:.1f}x vs KM-min ({KNOWN['km_min']})")
        print(f"    {vs_gf2:.2f}x vs GF(2) ({KNOWN['gf2']:,})")

    out = {
        "experiment": "exp_lowrank_bytedmd",
        "config": {
            "n_bits": N_BITS, "k_sparse": K_SPARSE, "seed": SEED,
            "hidden": HIDDEN, "batch": BATCH, "n_train": N_TRAIN,
            "epochs": EPOCHS, "lr": LR,
            "note": "demo config — does not converge; ByteDMD profile only",
        },
        "ranks": RANKS,
        "results": {str(k): v for k, v in results.items()},
        "full_rank": full_rank,
        "floor_read_n": floor_n,
        "known_comparison": KNOWN,
        "geometric_lower_bound_factor": GEOMETRIC_LOWER_BOUND_FACTOR,
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {RESULTS_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
