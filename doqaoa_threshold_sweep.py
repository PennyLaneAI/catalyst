#!/usr/bin/env python3
"""
DO-QAOA Threshold Calibration Utility  (Phase 2, Task 6)
=========================================================
Sweeps the ΔB threshold on benchmark graphs and plots the fraction of
sub-problems eligible for direct-copy parameter transfer vs. the threshold.

The "Approximate Reduction Gain" (ARG) measures how many sub-problems can
reuse the representative's parameters without warm-starting:

    ARG(θ) = |{k : ΔB_k < θ}| / |total sub-problems|

Default threshold from Sang et al. 2026: θ = 0.3.

Usage
-----
    python doqaoa_threshold_sweep.py [--save]

Output
------
    doqaoa_threshold_sweep.png   (if --save)
    Printed ARG table for each graph type.
"""

import argparse
import math
import sys

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Graph generators
# ──────────────────────────────────────────────────────────────────────────────

def make_cycle(n):
    """N-node cycle graph.  J[i, (i+1)%N] = -0.5."""
    J = np.zeros((n, n))
    for i in range(n):
        J[i, (i + 1) % n] = -0.5
        J[(i + 1) % n, i] = -0.5
    h = np.zeros(n)
    return J, h


def make_complete(n):
    """Complete graph K_n with all edges J = -0.5/(n-1)."""
    J = np.full((n, n), -0.5 / max(n - 1, 1))
    np.fill_diagonal(J, 0.0)
    h = np.zeros(n)
    return J, h


def make_path(n):
    """Path graph P_n.  J[i, i+1] = -0.5."""
    J = np.zeros((n, n))
    for i in range(n - 1):
        J[i, i + 1] = -0.5
        J[i + 1, i] = -0.5
    h = np.zeros(n)
    return J, h


def make_star(n):
    """Star graph S_n: hub=0 connected to all others."""
    J = np.zeros((n, n))
    for i in range(1, n):
        J[0, i] = -0.5
        J[i, 0] = -0.5
    h = np.zeros(n)
    return J, h


def make_biased_cycle(n, h_mag=1.0):
    """4-cycle with alternating linear bias ±h_mag on nodes 0,3."""
    J, _ = make_cycle(n)
    h = np.zeros(n)
    if n >= 4:
        h[0] = h_mag
        h[n - 1] = -h_mag
    return J, h


# ──────────────────────────────────────────────────────────────────────────────
# Bias computation (mirrors EnergyEval::computeBias in C++)
# ──────────────────────────────────────────────────────────────────────────────

def compute_bias(J, h, hotspot_indices, k):
    """
    B_k = (1 / N_free) Σ|h_eff[i]|
    h_eff[i] = h[i] + Σ_{frozen j} J[i,j] · s_j
    Frozen spin: bit i of k → s_i = +1 (bit=0) or -1 (bit=1)
    """
    n = J.shape[0]
    m = len(hotspot_indices)

    is_frozen = np.zeros(n, dtype=bool)
    frozen_spin = np.zeros(n)
    for i, qi in enumerate(hotspot_indices):
        is_frozen[qi] = True
        frozen_spin[qi] = -1.0 if ((k >> i) & 1) else +1.0

    h_eff = h.copy()
    for fi in range(n):
        if is_frozen[fi]:
            continue
        for fj in range(n):
            if is_frozen[fj]:
                h_eff[fi] += J[fi, fj] * frozen_spin[fj]

    free_mask = ~is_frozen
    n_free = int(free_mask.sum())
    if n_free == 0:
        return 0.0
    return float(np.abs(h_eff[free_mask]).mean())


# ──────────────────────────────────────────────────────────────────────────────
# Sweep logic
# ──────────────────────────────────────────────────────────────────────────────

def compute_arg_curve(J, h, hotspot_indices, thresholds):
    """
    Returns ARG(θ) for each threshold in `thresholds`.
    ARG(θ) = fraction of non-representative sub-problems with ΔB < θ.
    """
    m = len(hotspot_indices)
    num_sp = 1 << m

    b_values = np.array([compute_bias(J, h, hotspot_indices, k)
                         for k in range(num_sp)])

    b_rep = b_values[0]          # representative = sub-problem 0 (k=0)
    delta_b = np.abs(b_values - b_rep)

    # Non-representative sub-problems only (k=1..num_sp-1)
    non_rep = delta_b[1:] if num_sp > 1 else delta_b
    arg = np.array([float((non_rep < thr).sum()) / max(len(non_rep), 1)
                    for thr in thresholds])
    return arg, b_values, delta_b


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark suite
# ──────────────────────────────────────────────────────────────────────────────

BENCHMARKS = [
    ("Complete K8",    make_complete(8),    [0, 1]),
    ("Cycle C6",       make_cycle(6),       [0, 3]),
    ("Path P6",        make_path(6),        [0, 3]),
    ("Star S6",        make_star(6),        [0, 2]),
    ("Biased Cycle",   make_biased_cycle(6, 1.5), [0, 3]),
]

THRESHOLDS = np.linspace(0.0, 0.8, 81)   # 0.00 to 0.80 in steps of 0.01
DEFAULT_THR = 0.3


def run_sweep():
    print(f"\n{'Graph':<22} {'ARG@0.1':>8} {'ARG@0.3*':>9} {'ARG@0.5':>8}"
          f"   (* paper default)")
    print("-" * 55)

    results = {}
    for name, (J, h), hotspots in BENCHMARKS:
        arg, bvals, dbs = compute_arg_curve(J, h, hotspots, THRESHOLDS)
        results[name] = (arg, bvals, dbs)

        def arg_at(thr):
            idx = int(round(thr / (THRESHOLDS[1] - THRESHOLDS[0])))
            return arg[min(idx, len(arg) - 1)]

        print(f"  {name:<20} {arg_at(0.1):>8.1%} {arg_at(0.3):>9.1%} "
              f"{arg_at(0.5):>8.1%}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot(results, save=False):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not found — skipping plot.  "
              "Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("DO-QAOA Bias-Threshold Calibration", fontsize=13,
                 fontweight="bold")

    # Left: ARG vs threshold for each benchmark
    ax = axes[0]
    colors = ["#2077B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD"]
    for (name, _, __), color in zip(BENCHMARKS, colors):
        arg = results[name][0]
        ax.plot(THRESHOLDS, arg * 100, label=name, color=color, linewidth=1.8)
    ax.axvline(DEFAULT_THR, color="gray", linestyle="--", linewidth=1.2,
               label=f"default θ={DEFAULT_THR}")
    ax.set_xlabel("ΔB threshold θ", fontsize=11)
    ax.set_ylabel("ARG — direct-copy eligible (%)", fontsize=11)
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("ARG vs. Threshold")
    ax.grid(alpha=0.3)

    # Right: ΔB distribution for all graphs (box plot)
    ax2 = axes[1]
    all_dbs = []
    labels = []
    for name, (J, h), hotspots in BENCHMARKS:
        _, _, dbs = results[name]
        non_rep = dbs[1:] if len(dbs) > 1 else dbs
        all_dbs.append(non_rep)
        labels.append(name.split()[-1])  # short name

    bp = ax2.boxplot(all_dbs, labels=labels, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.axhline(DEFAULT_THR, color="gray", linestyle="--", linewidth=1.2,
                label=f"θ={DEFAULT_THR}")
    ax2.set_ylabel("ΔB = |B_target − B_rep|", fontsize=11)
    ax2.set_xlabel("Graph type", fontsize=11)
    ax2.set_title("ΔB Distribution per Graph")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    if save:
        fname = "doqaoa_threshold_sweep.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"\nSaved → {fname}")
    else:
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DO-QAOA threshold sweep")
    parser.add_argument("--save", action="store_true",
                        help="Save figure to doqaoa_threshold_sweep.png")
    args = parser.parse_args()

    results = run_sweep()
    plot(results, save=args.save)
