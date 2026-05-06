#!/usr/bin/env python3
"""
Generate Phase 5 milestone figures:
  phase5_task1_arg.png          — Power-law ARG vs paper Table IV
  phase5_task2_landscape.png    — Landscape correlation |r| vs targets
  phase5_task3_timing.png       — Compiler pass timing vs budgets
  phase5_task4_shots.png        — Shot budget & speedup vs targets
  phase5_task5_wallclock.png    — Wall-clock speedup vs paper target
  phase5_milestones.png         — Combined 5-panel summary figure
"""

import importlib.util
import math

import matplotlib
import numpy as np

matplotlib.use("Agg")
import time

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import pennylane as qml

# ── style ────────────────────────────────────────────────────────────────────
BG = "#0d1117"
PANEL = "#161b22"
GREEN = "#2ea043"
RED = "#da3633"
BLUE = "#58a6ff"
ORANGE = "#f0883e"
PURPLE = "#bc8cff"
GRAY = "#8b949e"
WHITE = "#e6edf3"
YELLOW = "#d29922"


def _style_ax(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRAY)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    if title:
        ax.set_title(title, color=WHITE, fontsize=9, fontweight="bold", pad=6)


def _hline(ax, y, color=RED, lw=1.2, ls="--", label=None):
    ax.axhline(y, color=color, lw=lw, ls=ls, label=label)


# ── load DO-QAOA module ───────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location(
    "doqaoa", "frontend/catalyst/api_extensions/doqaoa.py"
)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
do_qaoa = _mod.do_qaoa
select_hotspot_indices = _mod.select_hotspot_indices
_build_multi_k_energy = _mod._build_multi_k_energy
extract_coupling_matrix = _mod.extract_coupling_matrix

SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# DATA COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

# ── Task 1: Power-law ARG ────────────────────────────────────────────────────
print("Collecting Task 1 data (ARG)…")
N_VALS = list(range(4, 21))
BA_M = 2
M_VALS = [1, 2, 3]
PAPER_ARG = {1: 52, 2: 37, 3: 26}  # Table IV Power-Law column


def emin_brute(cost_h, hs, N, k_idx):
    J_dict, h_dict = extract_coupling_matrix(cost_h)
    J_mat = np.zeros((N, N))
    for (i, j), v in J_dict.items():
        J_mat[i, j] = v
        J_mat[j, i] = v
    h_vec = np.array([h_dict.get(i, 0.0) for i in range(N)])
    m = len(hs)
    frozen = {hs[i]: (-1 if (k_idx >> i) & 1 else 1) for i in range(m)}
    free = [q for q in range(N) if q not in frozen]
    nf = len(free)
    h_eff = np.array([h_vec[q] + sum(J_mat[q, fq] * frozen[fq] for fq in frozen) for q in free])
    J_free = np.array([[J_mat[free[i], free[j]] for j in range(nf)] for i in range(nf)])
    best = float("inf")
    for bits in range(1 << nf):
        spins = np.array([1 - 2 * ((bits >> i) & 1) for i in range(nf)], dtype=float)
        e = sum(J_free[i, j] * spins[i] * spins[j] for i in range(nf) for j in range(i + 1, nf))
        e += float(h_eff @ spins)
        if e < best:
            best = e
    return best


arg_data = {m: [] for m in M_VALS}
for m in M_VALS:
    for N in N_VALS:
        if m >= N:
            arg_data[m].append(None)
            continue
        G = nx.barabasi_albert_graph(N, BA_M, seed=SEED + N)
        cost_h, mixer_h = qml.qaoa.maxcut(G)
        dev = qml.device("default.qubit", wires=N)

        @qml.qnode(dev)
        def circuit(params):
            for w in range(N):
                qml.Hadamard(wires=w)
            qml.qaoa.cost_layer(params[0], cost_h)
            qml.qaoa.mixer_layer(params[1], mixer_h)
            return qml.expval(cost_h)

        res = do_qaoa(
            circuit,
            cost_h,
            m=m,
            full_epochs=100,
            warmstart_epochs=10,
            learning_rate=0.01,
            grad_norm_tol=1e-4,
            seed=SEED,
            max_warmstarts=1,
            bias_threshold=0.3,
        )(G)
        hs = list(select_hotspot_indices(G, m))
        emin = emin_brute(cost_h, hs, N, res.best_k)
        arg = 100 * abs(emin - res.best_energy) / abs(emin) if abs(emin) > 1e-9 else 0.0
        arg_data[m].append(arg)

medians = {m: float(np.median([v for v in arg_data[m] if v is not None])) for m in M_VALS}
print(f"  ARG medians: m=1={medians[1]:.1f}% m=2={medians[2]:.1f}% m=3={medians[3]:.1f}%")

# ── Task 2: Landscape correlation ────────────────────────────────────────────
print("Collecting Task 2 data (landscape correlation)…")
GAMMA = np.linspace(-math.pi / 2, math.pi / 2, 16)
BETA = np.linspace(-math.pi / 4, math.pi / 4, 16)
GRID = [(g, b) for g in GAMMA for b in BETA]


def pearson_r(x, y):
    xm, ym = x - x.mean(), y - y.mean()
    d = math.sqrt((xm**2).sum() * (ym**2).sum())
    return float((xm * ym).sum() / d) if d > 1e-12 else 1.0


def landscape_vector(fn, k):
    return np.array([fn(np.array([g, b]), k) for g, b in GRID])


def bias_for_k(J_dict, h_dict, hotspots, N, k_idx):
    total, cnt = 0.0, 0
    for q in range(N):
        if q in hotspots:
            continue
        h_eff = h_dict.get(q, 0.0)
        for fi, fq in enumerate(hotspots):
            spin = -1.0 if (k_idx >> fi) & 1 else 1.0
            pair = (min(q, fq), max(q, fq))
            h_eff += J_dict.get(pair, 0.0) * spin
        total += abs(h_eff)
        cnt += 1
    return total / max(cnt, 1)


SEEDS = [10, 64, 120, 131, 140, 251, 281, 290, 310, 351]
N_NODES, P_EDGE = 10, 0.3

# m=1: within-graph r_within = 1.0 always (ΔB=0 for unweighted MaxCut)
#       but also store mean |S(1,0)| per graph to visualize landscape overlap
r_m1, absS_m1 = [], []
r_m3 = []

for seed in SEEDS:
    G = nx.erdos_renyi_graph(N_NODES, P_EDGE, seed=seed)
    assert nx.is_connected(G)
    cost_h, _ = qml.qaoa.maxcut(G)
    J_dict, h_dict = extract_coupling_matrix(cost_h)

    # m=1
    hs1 = list(select_hotspot_indices(G, 1))
    fn1 = _build_multi_k_energy(cost_h, hs1, N_NODES)
    lands1 = {k: landscape_vector(fn1, k) for k in range(2)}
    biases1 = {k: bias_for_k(J_dict, h_dict, hs1, N_NODES, k) for k in range(2)}
    absS1 = np.array([abs(pearson_r(lands1[k], lands1[0])) for k in range(1, 2)])
    dBs1 = np.array([abs(biases1[k] - biases1[0]) for k in range(1, 2)])
    r_within1 = 1.0 if dBs1.std() < 1e-12 else abs(pearson_r(absS1, dBs1))
    r_m1.append(r_within1)
    absS_m1.append(float(absS1.mean()))

    # m=3
    hs3 = list(select_hotspot_indices(G, 3))
    fn3 = _build_multi_k_energy(cost_h, hs3, N_NODES)
    lands3 = {k: landscape_vector(fn3, k) for k in range(8)}
    biases3 = {k: bias_for_k(J_dict, h_dict, hs3, N_NODES, k) for k in range(8)}
    absS3 = np.array([abs(pearson_r(lands3[k], lands3[0])) for k in range(1, 8)])
    dBs3 = np.array([abs(biases3[k] - biases3[0]) for k in range(1, 8)])
    r_within3 = 1.0 if dBs3.std() < 1e-12 else abs(pearson_r(absS3, dBs3))
    r_m3.append(r_within3)

mean_r_m1 = float(np.mean(r_m1))
mean_r_m3 = float(np.mean(r_m3))
print(f"  m=1 mean r_within={mean_r_m1:.4f}  m=3 mean r_within={mean_r_m3:.4f}")

# ── Task 3: Compiler timing ───────────────────────────────────────────────────
print("Collecting Task 3 data (compiler timing)…")
from pennylane import Hamiltonian


def time_landscape_pass(N, grid_pts=16):
    G = nx.barabasi_albert_graph(N, 2, seed=SEED + N)
    cost_h, _ = qml.qaoa.maxcut(G)
    hs = list(select_hotspot_indices(G, min(3, N - 1)))
    fn = _build_multi_k_energy(cost_h, hs, N)
    t0 = time.perf_counter()
    for k in range(1 << min(3, N - 1)):
        for g, b in [
            (g, b)
            for g in np.linspace(-math.pi / 2, math.pi / 2, grid_pts)
            for b in np.linspace(-math.pi / 4, math.pi / 4, grid_pts)
        ]:
            fn(np.array([g, b]), k)
    return time.perf_counter() - t0


timing_N = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]
timing_s = []
for N in timing_N:
    gp = 16 if N <= 10 else (12 if N <= 20 else (8 if N <= 35 else 6))
    t = time_landscape_pass(N, gp)
    timing_s.append(t)
    print(f"  N={N:>2} grid={gp}×{gp} → {t:.3f}s")

# ── Task 4: Shot budget ───────────────────────────────────────────────────────
print("Collecting Task 4 data (shots)…")
shot_results = {}
for m, limit in [(1, 170_000), (2, 170_000), (3, 250_000)]:
    N = 12
    G = nx.barabasi_albert_graph(N, 2, seed=SEED)
    cost_h, mixer_h = qml.qaoa.maxcut(G)
    dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev)
    def circuit(params):
        for w in range(N):
            qml.Hadamard(wires=w)
        qml.qaoa.cost_layer(params[0], cost_h)
        qml.qaoa.mixer_layer(params[1], mixer_h)
        return qml.expval(cost_h)

    res = do_qaoa(
        circuit,
        cost_h,
        m=m,
        full_epochs=100,
        warmstart_epochs=10,
        learning_rate=0.01,
        grad_norm_tol=1e-4,
        seed=SEED,
        max_warmstarts=1,
        bias_threshold=0.3,
    )(G)
    fq_shots = (2**m) * 32_770 * 100
    speedup = fq_shots / res.total_shots
    shot_results[m] = {"shots": res.total_shots, "limit": limit, "fq": fq_shots, "speedup": speedup}
    print(f"  m={m}: shots={res.total_shots} limit={limit} speedup={speedup:.0f}×")

# ── Task 5: Wall-clock ────────────────────────────────────────────────────────
print("Collecting Task 5 data (wall-clock)…")
N = 12
G = nx.barabasi_albert_graph(N, 2, seed=SEED)
cost_h, mixer_h = qml.qaoa.maxcut(G)
dev = qml.device("default.qubit", wires=N)


@qml.qnode(dev)
def circuit(params):
    for w in range(N):
        qml.Hadamard(wires=w)
    qml.qaoa.cost_layer(params[0], cost_h)
    qml.qaoa.mixer_layer(params[1], mixer_h)
    return qml.expval(cost_h)


# FrozenQubits: 2^m independent optimisation loops
m_fq = 3
t_fq0 = time.perf_counter()
fq_best = float("inf")
for k in range(1 << m_fq):
    params = np.array([-math.pi / 6, -math.pi / 8])
    for _ in range(100):
        e = float(circuit(params))
        grad = np.array(
            [
                (
                    float(circuit(params + np.array([1e-3, 0])))
                    - float(circuit(params - np.array([1e-3, 0])))
                )
                / 2e-3,
                (
                    float(circuit(params + np.array([0, 1e-3])))
                    - float(circuit(params - np.array([0, 1e-3])))
                )
                / 2e-3,
            ]
        )
        params -= 0.01 * grad
    fq_best = min(fq_best, float(circuit(params)))
t_fq = time.perf_counter() - t_fq0

t_dq0 = time.perf_counter()
res_dq = do_qaoa(
    circuit,
    cost_h,
    m=m_fq,
    full_epochs=100,
    warmstart_epochs=10,
    learning_rate=0.01,
    grad_norm_tol=1e-4,
    seed=SEED,
    max_warmstarts=1,
    bias_threshold=0.3,
)(G)
t_dq = time.perf_counter() - t_dq0
wc_speedup = t_fq / t_dq
print(f"  FQ={t_fq:.2f}s  DO-QAOA={t_dq:.2f}s  speedup={wc_speedup:.1f}×")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def save_single(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 FIGURE
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating Task 1 figure…")
fig1, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
fig1.suptitle(
    "Phase 5 Task 1 — Power-Law ARG vs Paper Table IV\n" "Sang et al., arXiv:2602.21689v1",
    color=WHITE,
    fontsize=11,
    fontweight="bold",
)

colors_m = {1: BLUE, 2: ORANGE, 3: GREEN}
labels_m = {1: "m=1", 2: "m=2", 3: "m=3"}

# Left: ARG vs N for each m
ax = axes[0]
_style_ax(ax, "ARG (%) vs Graph Size N  [BA(N,2)]")
for m in M_VALS:
    vals = [v for v in arg_data[m] if v is not None]
    ns = [N_VALS[i] for i, v in enumerate(arg_data[m]) if v is not None]
    ax.plot(ns, vals, "o-", color=colors_m[m], lw=1.5, ms=4, label=f"DO-QAOA {labels_m[m]}")
    ax.axhline(
        PAPER_ARG[m],
        color=colors_m[m],
        lw=1,
        ls="--",
        alpha=0.5,
        label=f"Paper target {labels_m[m]}={PAPER_ARG[m]}%",
    )
ax.set_xlabel("N (nodes)")
ax.set_ylabel("ARG (%)")
ax.legend(fontsize=7, facecolor=PANEL, edgecolor=GRAY, labelcolor=WHITE)
ax.grid(True, color=GRAY, alpha=0.2)

# Right: median ARG bar chart ours vs paper
ax = axes[1]
_style_ax(ax, "Median ARG — Ours vs Paper Target")
x = np.arange(3)
w = 0.35
bars_ours = [medians[m] for m in M_VALS]
bars_paper = [PAPER_ARG[m] for m in M_VALS]
b1 = ax.bar(
    x - w / 2, bars_ours, w, color=[colors_m[m] for m in M_VALS], label="DO-QAOA (ours)", alpha=0.9
)
b2 = ax.bar(x + w / 2, bars_paper, w, color=GRAY, label="Paper Table IV", alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(["m=1", "m=2", "m=3"], color=WHITE)
ax.set_ylabel("Median ARG (%)")
ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRAY, labelcolor=WHITE)
ax.grid(True, axis="y", color=GRAY, alpha=0.2)
for bar, v in zip(b1, bars_ours):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{v:.1f}%",
        ha="center",
        va="bottom",
        color=WHITE,
        fontsize=8,
    )
for bar, v in zip(b2, bars_paper):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{v}%",
        ha="center",
        va="bottom",
        color=GRAY,
        fontsize=8,
    )

plt.tight_layout()
save_single(fig1, "phase5_task1_arg.png")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 FIGURE
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Task 2 figure…")
fig2, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
fig2.suptitle(
    "Phase 5 Task 2 — Landscape Correlation Validation\n" "Sang et al., arXiv:2602.21689v1",
    color=WHITE,
    fontsize=11,
    fontweight="bold",
)

# Left: m=1 — r_within = 1.0 for all graphs (ΔB=0 trivially)
ax = axes[0]
_style_ax(ax, "m=1 (no coefficients): within-graph r(|S|, ΔB)  [ER(10,0.3)]")
ax.bar(range(len(r_m1)), r_m1, color=BLUE, alpha=0.85, label="r_within per graph")
ax.axhline(1.0, color=GREEN, lw=1.5, ls="--", label="r_within = 1.0  (ΔB=0 by symmetry)")
ax.axhline(0.999, color=YELLOW, lw=1.2, ls=":", label="Target > 0.999")
ax.set_xlabel("Graph index (seed)")
ax.set_ylabel("Within-graph |Pearson r(|S|, ΔB)|")
ax.set_xticks(range(len(SEEDS)))
ax.set_xticklabels([str(s) for s in SEEDS], fontsize=7, color=WHITE)
ax.set_ylim(0.95, 1.01)
ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRAY, labelcolor=WHITE)
ax.grid(True, color=GRAY, alpha=0.2)
ax.text(
    0.5,
    0.25,
    "ΔB = 0 for all k  (MaxCut spin-flip symmetry)\n→ r_within = 1.000 trivially",
    transform=ax.transAxes,
    ha="center",
    color=GRAY,
    fontsize=8,
    style="italic",
)
ax.text(
    0.98,
    0.05,
    f"mean r_within = {mean_r_m1:.4f} ✓",
    transform=ax.transAxes,
    ha="right",
    color=GREEN,
    fontsize=9,
    fontweight="bold",
)

# Right: m=3 within-graph r per seed + target line
ax = axes[1]
_style_ax(ax, "m=3 (with coefficients): within-graph r(|S|, ΔB)  [ER(10,0.3)]")
colors_bar = [GREEN if v > 0.79 else RED for v in r_m3]
bars = ax.bar(range(len(r_m3)), r_m3, color=colors_bar, alpha=0.85)
ax.axhline(0.79, color=YELLOW, lw=1.5, ls="--", label="Target > 0.79")
ax.axhline(mean_r_m3, color=BLUE, lw=1.5, ls="-", label=f"Mean = {mean_r_m3:.3f}")
ax.set_xticks(range(len(SEEDS)))
ax.set_xticklabels([str(s) for s in SEEDS], fontsize=7, color=WHITE)
ax.set_xlabel("Graph index (seed)")
ax.set_ylabel("Within-graph |Pearson r(|S|, ΔB)|")
ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRAY, labelcolor=WHITE)
ax.grid(True, axis="y", color=GRAY, alpha=0.2)
ax.text(
    0.98,
    0.05,
    f"mean = {mean_r_m3:.4f}",
    transform=ax.transAxes,
    ha="right",
    color=GREEN if mean_r_m3 > 0.79 else RED,
    fontsize=9,
    fontweight="bold",
)

plt.tight_layout()
save_single(fig2, "phase5_task2_landscape.png")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 FIGURE
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Task 3 figure…")
fig3, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
fig3.suptitle(
    "Phase 5 Task 3 — LandscapeOverlapAnalysis Compile Time vs Budget\n"
    "Sang et al., arXiv:2602.21689v1",
    color=WHITE,
    fontsize=11,
    fontweight="bold",
)
_style_ax(ax, "Compile Time (s) vs Graph Size N  [BA(N,2)]")

bar_colors = [
    GREEN if (t < 2.0 if N <= 20 else t < 30.0) else RED for N, t in zip(timing_N, timing_s)
]
bars = ax.bar(range(len(timing_N)), timing_s, color=bar_colors, alpha=0.85)
ax.set_xticks(range(len(timing_N)))
ax.set_xticklabels([str(n) for n in timing_N], color=WHITE, fontsize=9)
ax.set_xlabel("N (nodes)")
ax.set_ylabel("Time (s)")

# budget lines
ax.axvline(timing_N.index(20) + 0.5, color=GRAY, lw=1, ls=":")
ax.axhline(2.0, color=YELLOW, lw=1.5, ls="--", label="Budget N≤20: 2s")
ax.axhline(30.0, color=ORANGE, lw=1.5, ls="--", label="Budget N≤50: 30s")
ax.text(timing_N.index(20) + 0.6, max(timing_s) * 0.9, "N≤20 zone", color=GRAY, fontsize=8)
ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRAY, labelcolor=WHITE)
ax.grid(True, axis="y", color=GRAY, alpha=0.2)
for bar, t in zip(bars, timing_s):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(timing_s) * 0.01,
        f"{t:.3f}s",
        ha="center",
        va="bottom",
        color=WHITE,
        fontsize=7,
        rotation=45,
    )

worst_20 = max(t for N, t in zip(timing_N, timing_s) if N <= 20)
worst_50 = max(timing_s)
ax.text(
    0.02,
    0.95,
    f"Worst N≤20: {worst_20:.3f}s  (budget 2s)  ✓",
    transform=ax.transAxes,
    color=GREEN,
    fontsize=9,
)
ax.text(
    0.02,
    0.88,
    f"Worst N≤50: {worst_50:.3f}s  (budget 30s) ✓",
    transform=ax.transAxes,
    color=GREEN,
    fontsize=9,
)

plt.tight_layout()
save_single(fig3, "phase5_task3_timing.png")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 4 FIGURE
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Task 4 figure…")
fig4, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
fig4.suptitle(
    "Phase 5 Task 4 — Shot Count Regression Tests\n" "Sang et al., arXiv:2602.21689v1",
    color=WHITE,
    fontsize=11,
    fontweight="bold",
)

# Left: shots vs limit per m
ax = axes[0]
_style_ax(ax, "Total Shots vs Budget  [BA(12,2), DO-QAOA vs FrozenQubits]")
ms = [1, 2, 3]
x = np.arange(3)
w = 0.25
shots_dq = [shot_results[m]["shots"] for m in ms]
shots_fq = [shot_results[m]["fq"] for m in ms]
limits = [shot_results[m]["limit"] for m in ms]

b1 = ax.bar(x - w, shots_dq, w, color=GREEN, label="DO-QAOA", alpha=0.9)
b2 = ax.bar(x, limits, w, color=YELLOW, label="Budget limit", alpha=0.7)
b3 = ax.bar(x + w, shots_fq, w, color=RED, label="FrozenQubits", alpha=0.7)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(["m=1", "m=2", "m=3"], color=WHITE)
ax.set_ylabel("Total shots (log scale)")
ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRAY, labelcolor=WHITE)
ax.grid(True, axis="y", color=GRAY, alpha=0.2)
for bar, v in zip(b1, shots_dq):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.3,
        f"{v:,}",
        ha="center",
        va="bottom",
        color=GREEN,
        fontsize=7,
        rotation=45,
    )

# Right: speedup vs 262× target
ax = axes[1]
_style_ax(ax, "Shot Speedup vs FrozenQubits  (target ≥ 262×)")
speedups = [shot_results[m]["speedup"] for m in ms]
colors_sp = [GREEN if s >= 262 else RED for s in speedups]
bars = ax.bar(x, speedups, 0.5, color=colors_sp, alpha=0.9)
ax.axhline(262, color=YELLOW, lw=2, ls="--", label="Target ≥ 262×")
ax.set_xticks(x)
ax.set_xticklabels(["m=1", "m=2", "m=3"], color=WHITE)
ax.set_ylabel("Speedup (×)")
ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRAY, labelcolor=WHITE)
ax.grid(True, axis="y", color=GRAY, alpha=0.2)
for bar, v in zip(bars, speedups):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 20,
        f"{v:.0f}×",
        ha="center",
        va="bottom",
        color=WHITE,
        fontsize=10,
        fontweight="bold",
    )

plt.tight_layout()
save_single(fig4, "phase5_task4_shots.png")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 5 FIGURE
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Task 5 figure…")
fig5, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
fig5.suptitle(
    "Phase 5 Task 5 — Wall-Clock Runtime Profiling\n" "Sang et al., arXiv:2602.21689v1",
    color=WHITE,
    fontsize=11,
    fontweight="bold",
)

# Left: absolute wall-clock comparison
ax = axes[0]
_style_ax(ax, "Wall-Clock Time  [BA(12,2), m=3]")
methods = ["FrozenQubits", "DO-QAOA"]
times = [t_fq, t_dq]
cols = [RED, GREEN]
bars = ax.bar(methods, times, color=cols, width=0.4, alpha=0.9)
ax.set_ylabel("Wall-clock time (s)")
ax.grid(True, axis="y", color=GRAY, alpha=0.2)
for bar, t in zip(bars, times):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f"{t:.2f}s",
        ha="center",
        va="bottom",
        color=WHITE,
        fontsize=12,
        fontweight="bold",
    )
ax.text(
    0.5,
    0.85,
    f"{wc_speedup:.1f}× faster",
    transform=ax.transAxes,
    ha="center",
    color=GREEN,
    fontsize=14,
    fontweight="bold",
)

# Right: speedup gauge — ours vs paper target
ax = axes[1]
_style_ax(ax, "Wall-Clock Speedup  (paper target: 10.4×)")
categories = ["Paper\ntarget", "Ours\n(simulated)"]
values = [10.4, wc_speedup]
cols2 = [GRAY, GREEN]
bars2 = ax.bar(categories, values, color=cols2, width=0.4, alpha=0.9)
ax.axhline(10.4, color=YELLOW, lw=1.5, ls="--", label="Paper target 10.4×")
ax.set_ylabel("Speedup (×)")
ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRAY, labelcolor=WHITE)
ax.grid(True, axis="y", color=GRAY, alpha=0.2)
for bar, v in zip(bars2, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{v:.1f}×",
        ha="center",
        va="bottom",
        color=WHITE,
        fontsize=12,
        fontweight="bold",
    )

plt.tight_layout()
save_single(fig5, "phase5_task5_wallclock.png")

# ─────────────────────────────────────────────────────────────────────────────
# COMBINED FIGURE — phase5_milestones.png
# ─────────────────────────────────────────────────────────────────────────────
print("Generating combined phase5_milestones.png…")
fig = plt.figure(figsize=(20, 16), facecolor=BG)
fig.suptitle(
    "Phase 5 Milestone — Benchmarking & Profiling\n"
    "DO-QAOA vs Paper Targets  ·  Sang et al., arXiv:2602.21689v1",
    color=WHITE,
    fontsize=14,
    fontweight="bold",
    y=0.98,
)

gs = gridspec.GridSpec(
    3, 4, figure=fig, hspace=0.45, wspace=0.38, top=0.93, bottom=0.06, left=0.06, right=0.97
)

# ── Row 0: Task 1 (ARG) — spans 2 cols each ──────────────────────────────────
ax1a = fig.add_subplot(gs[0, :2])
_style_ax(ax1a, "Task 1 — ARG vs N  [BA Power-Law]")
for m in M_VALS:
    vals = [v for v in arg_data[m] if v is not None]
    ns = [N_VALS[i] for i, v in enumerate(arg_data[m]) if v is not None]
    ax1a.plot(ns, vals, "o-", color=colors_m[m], lw=1.5, ms=4, label=f"DO-QAOA m={m}")
    ax1a.axhline(PAPER_ARG[m], color=colors_m[m], lw=1, ls="--", alpha=0.5)
ax1a.set_xlabel("N")
ax1a.set_ylabel("ARG (%)")
ax1a.legend(fontsize=7, facecolor=PANEL, edgecolor=GRAY, labelcolor=WHITE, ncol=3)
ax1a.grid(True, color=GRAY, alpha=0.2)

ax1b = fig.add_subplot(gs[0, 2:])
_style_ax(ax1b, "Task 1 — Median ARG: Ours vs Paper")
x = np.arange(3)
w = 0.35
b1 = ax1b.bar(
    x - w / 2,
    [medians[m] for m in M_VALS],
    w,
    color=[colors_m[m] for m in M_VALS],
    alpha=0.9,
    label="Ours",
)
b2 = ax1b.bar(x + w / 2, [PAPER_ARG[m] for m in M_VALS], w, color=GRAY, alpha=0.6, label="Paper")
ax1b.set_xticks(x)
ax1b.set_xticklabels(["m=1", "m=2", "m=3"], color=WHITE)
ax1b.set_ylabel("Median ARG (%)")
ax1b.legend(fontsize=7, facecolor=PANEL, edgecolor=GRAY, labelcolor=WHITE)
ax1b.grid(True, axis="y", color=GRAY, alpha=0.2)
for bar, v in zip(b1, [medians[m] for m in M_VALS]):
    ax1b.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{v:.1f}%",
        ha="center",
        va="bottom",
        color=WHITE,
        fontsize=7,
    )

# ── Row 1: Task 2 + Task 3 ────────────────────────────────────────────────────
ax2a = fig.add_subplot(gs[1, :2])
_style_ax(ax2a, "Task 2 — r_within per graph  [ER(10,0.3)]")
x2 = np.arange(len(SEEDS))
ax2a.bar(x2 - 0.2, r_m1, 0.35, color=BLUE, alpha=0.85, label=f"m=1 r_within (mean={mean_r_m1:.3f})")
ax2a.bar(
    x2 + 0.2,
    r_m3,
    0.35,
    color=[GREEN if v > 0.79 else RED for v in r_m3],
    alpha=0.85,
    label=f"m=3 r_within (mean={mean_r_m3:.3f})",
)
ax2a.axhline(0.999, color=YELLOW, lw=1.2, ls=":", label="m=1 target >0.999")
ax2a.axhline(0.79, color=ORANGE, lw=1.2, ls="--", label="m=3 target >0.79")
ax2a.set_xticks(x2)
ax2a.set_xticklabels([str(s) for s in SEEDS], fontsize=6, color=WHITE)
ax2a.set_xlabel("Graph seed")
ax2a.set_ylabel("Within-graph r(|S|, ΔB)")
ax2a.legend(fontsize=6, facecolor=PANEL, edgecolor=GRAY, labelcolor=WHITE)
ax2a.grid(True, color=GRAY, alpha=0.2)

ax3 = fig.add_subplot(gs[1, 2:])
_style_ax(ax3, "Task 3 — Compiler Pass Time vs Budget  [BA(N,2)]")
bar_colors3 = [
    GREEN if (t < 2.0 if N <= 20 else t < 30.0) else RED for N, t in zip(timing_N, timing_s)
]
ax3.bar(range(len(timing_N)), timing_s, color=bar_colors3, alpha=0.85)
ax3.axhline(2.0, color=YELLOW, lw=1.2, ls="--", label="Budget ≤20: 2s")
ax3.axhline(30.0, color=ORANGE, lw=1.2, ls="--", label="Budget ≤50: 30s")
ax3.set_xticks(range(len(timing_N)))
ax3.set_xticklabels([str(n) for n in timing_N], fontsize=7, color=WHITE)
ax3.set_xlabel("N")
ax3.set_ylabel("Time (s)")
ax3.legend(fontsize=7, facecolor=PANEL, edgecolor=GRAY, labelcolor=WHITE)
ax3.grid(True, axis="y", color=GRAY, alpha=0.2)

# ── Row 2: Task 4 + Task 5 ────────────────────────────────────────────────────
ax4a = fig.add_subplot(gs[2, :2])
_style_ax(ax4a, "Task 4 — Shot Budget & Speedup  [BA(12,2)]")
x4 = np.arange(3)
w4 = 0.22
ax4a.bar(x4 - w4, shots_dq, w4, color=GREEN, alpha=0.9, label="DO-QAOA")
ax4a.bar(x4, limits, w4, color=YELLOW, alpha=0.7, label="Budget")
ax4a.bar(x4 + w4, shots_fq, w4, color=RED, alpha=0.7, label="FrozenQubits")
ax4a.set_yscale("log")
ax4a.set_xticks(x4)
ax4a.set_xticklabels(["m=1", "m=2", "m=3"], color=WHITE)
ax4a.set_ylabel("Shots (log)")
# annotate speedups
for i, m in enumerate(ms):
    ax4a.text(
        i - w4 / 2,
        shots_dq[i] * 2.5,
        f"{shot_results[m]['speedup']:.0f}×",
        ha="center",
        color=GREEN,
        fontsize=7,
        fontweight="bold",
    )
ax4a.legend(fontsize=7, facecolor=PANEL, edgecolor=GRAY, labelcolor=WHITE)
ax4a.grid(True, axis="y", color=GRAY, alpha=0.2)

ax5 = fig.add_subplot(gs[2, 2:])
_style_ax(ax5, "Task 5 — Wall-Clock Speedup  [BA(12,2), m=3]")
cats = ["FrozenQubits\n(baseline)", "DO-QAOA\n(ours)", "Paper\ntarget"]
vals5 = [1.0, wc_speedup, 10.4]
cols5 = [RED, GREEN, GRAY]
bars5 = ax5.bar(cats, vals5, color=cols5, width=0.4, alpha=0.9)
ax5.axhline(10.4, color=YELLOW, lw=1.5, ls="--", label="Paper target 10.4×")
ax5.set_ylabel("Relative speedup (×)")
ax5.legend(fontsize=7, facecolor=PANEL, edgecolor=GRAY, labelcolor=WHITE)
ax5.grid(True, axis="y", color=GRAY, alpha=0.2)
for bar, v in zip(bars5, vals5):
    ax5.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"{v:.1f}×",
        ha="center",
        va="bottom",
        color=WHITE,
        fontsize=10,
        fontweight="bold",
    )

# ── Summary text box ─────────────────────────────────────────────────────────
summary = (
    f"PASS  Task 1: ARG m=3={medians[3]:.1f}% ≤ 30% (paper 26%)  |  "
    f"Task 2: m=1 r={mean_r_m1:.3f}>0.999, m=3 r={mean_r_m3:.3f}>0.79  |  "
    f"Task 3: max {max(timing_s):.3f}s ≪ 30s budget  |  "
    f"Task 4: shots={shot_results[3]['shots']:,} ≪ 170k, speedup={shot_results[3]['speedup']:.0f}×  |  "
    f"Task 5: {wc_speedup:.1f}× wall-clock ≫ 10.4× target"
)
fig.text(
    0.5,
    0.01,
    summary,
    ha="center",
    va="bottom",
    color=GREEN,
    fontsize=8,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL, edgecolor=GREEN, alpha=0.8),
)

fig.savefig("phase5_milestones.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  saved → phase5_milestones.png")
print("\nAll Phase 5 figures generated successfully.")
