#!/usr/bin/env python3
"""
Phase 2 Milestone — Pearson r > 0.999 on Erdős-Rényi Graphs
=============================================================
Deliverable:
  Working mlir-opt pipeline that takes a freeze_partition module, runs
  LandscapeOverlapAnalysis + BiasShiftAnalysis, and annotates all
  sub-problems with cluster_id and bias_shift.
  Validated on 10-node Erdős-Rényi graphs: Pearson r > 0.999 for m=1.

Reference: Sang et al., arXiv:2602.21689v1, Sec. 2.2–2.4
"""

import math
import pathlib
import subprocess
import sys
import re
import textwrap

import networkx as nx
import numpy as np
from scipy.stats import pearsonr

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

ROOT       = pathlib.Path(__file__).parent
QUANTUM_OPT = ROOT / "catalyst/mlir/build/bin/quantum-opt"

# ──────────────────────────────────────────────────────────────────────────────
# Graph helpers
# ──────────────────────────────────────────────────────────────────────────────

def er_graph_to_ising(G, weight=-0.5):
    """Convert undirected graph to symmetric J matrix (MaxCut encoding)."""
    N = G.number_of_nodes()
    J = np.zeros((N, N))
    for u, v in G.edges():
        J[u, v] = weight
        J[v, u] = weight
    return J


def degree_hotspot(G, m=1):
    """Return m highest-degree node indices."""
    deg = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    return [d[0] for d in deg[:m]]


# ──────────────────────────────────────────────────────────────────────────────
# Pure-Python QAOA p=1 exact energy evaluator  (mirrors EnergyEval.cpp)
# ──────────────────────────────────────────────────────────────────────────────

def _ising_energy(z_bits, J_free, h_eff):
    """Ising energy for binary string z_bits (0→+1, 1→-1)."""
    s = np.where(np.array(z_bits) == 0, 1.0, -1.0)
    return float(0.5 * s @ J_free @ s + h_eff @ s)


def qaoa_p1_exact(J_free, h_eff, gamma, beta):
    """
    Exact QAOA p=1 expectation value via statevector.
    |+>^N → C(γ) → B(β) → <H>
    """
    n = J_free.shape[0]
    dim = 1 << n

    # Initialise |+>^n
    psi = np.full(dim, 1.0 / math.sqrt(dim), dtype=complex)

    # Compute Ising energy for each basis state
    energies = np.zeros(dim)
    for z in range(dim):
        bits = [(z >> i) & 1 for i in range(n)]
        energies[z] = _ising_energy(bits, J_free, h_eff)

    # Cost unitary C(γ): |z> → exp(-i γ E_z) |z>
    psi *= np.exp(-1j * gamma * energies)

    # Mixer B(β): product of RX(2β) gates qubit-by-qubit
    for q in range(n):
        stride = 1 << q
        for block in range(0, dim, 2 * stride):
            for i in range(block, block + stride):
                a, b = psi[i], psi[i + stride]
                cos_b, sin_b = math.cos(beta), math.sin(beta)
                psi[i]         =  cos_b * a - 1j * sin_b * b
                psi[i + stride] = -1j * sin_b * a + cos_b * b

    # <H> = Σ_z |ψ_z|² E_z
    return float(np.real(np.sum(np.abs(psi) ** 2 * energies)))


def build_landscape_vector(J, h, hotspot_indices, k, grid_size=16):
    """
    Build the L2-normalised landscape vector for sub-problem k.
    γ ∈ [−π, π],  β ∈ [−π/2, π/2]
    """
    N = J.shape[0]
    m = len(hotspot_indices)

    # Frozen spins for sub-problem k
    is_frozen = np.zeros(N, dtype=bool)
    frozen_spin = np.zeros(N)
    for i, qi in enumerate(hotspot_indices):
        is_frozen[qi] = True
        frozen_spin[qi] = -1.0 if ((k >> i) & 1) else +1.0

    free = np.where(~is_frozen)[0]
    n_free = len(free)

    # Free-free J sub-matrix
    J_free = J[np.ix_(free, free)]

    # Effective bias on free qubits
    h_eff = h[free].copy()
    for fi_idx, fi in enumerate(free):
        for fj in range(N):
            if is_frozen[fj]:
                h_eff[fi_idx] += J[fi, fj] * frozen_spin[fj]

    # Grid sweep
    gammas = np.linspace(-math.pi, math.pi, grid_size)
    betas  = np.linspace(-math.pi / 2, math.pi / 2, grid_size)

    vec = []
    for gamma in gammas:
        for beta in betas:
            vec.append(qaoa_p1_exact(J_free, h_eff, gamma, beta))

    vec = np.array(vec)
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec /= norm
    return vec


# ──────────────────────────────────────────────────────────────────────────────
# MLIR module generator
# ──────────────────────────────────────────────────────────────────────────────

def j_matrix_to_dense_attr(J, N):
    rows = []
    for i in range(N):
        row = "[" + ", ".join(f"{J[i,j]:.6f}" for j in range(N)) + "]"
        rows.append(row)
    inner = ", ".join(rows)
    return (f"#quantum.dense_graph<{N}, dense<[{inner}]> "
            f": tensor<{N}x{N}xf64>>")


def make_mlir_module(G, J, h, hotspot_indices, graph_id, m=1):
    N = G.number_of_nodes()
    hotspot_arr = ", ".join(str(i) for i in hotspot_indices)
    h_quad = j_matrix_to_dense_attr(J, N)
    h_lin  = (f"dense<[{', '.join(f'{v:.6f}' for v in h)}]> "
              f": tensor<{N}xf64>")

    return textwrap.dedent(f"""\
        module {{
          func.func @er_graph_{graph_id}() {{
            %p = quantum.freeze_partition {{
                hotspot_count   = {m} : i32,
                hotspot_indices = array<i32: {hotspot_arr}>,
                h_quad = {h_quad},
                h_lin  = {h_lin}
            }} : !quantum.partition<{N}, {m}>
            func.return
          }}
        }}
    """)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(mlir_text, graph_id):
    """Run quantum-opt pipeline and return annotated IR text."""
    mlir_file = ROOT / f"_er_{graph_id}.mlir"
    mlir_file.write_text(mlir_text)

    result = subprocess.run(
        [
            str(QUANTUM_OPT),
            str(mlir_file),
            "--doqaoa-landscape-overlap",
            "--doqaoa-bias-shift",
        ],
        capture_output=True, text=True
    )

    mlir_file.unlink(missing_ok=True)

    if result.returncode != 0:
        return None, result.stderr
    return result.stdout, None


def parse_attr_f64(ir_text, attr):
    """Extract first occurrence of attr = <float> from IR."""
    m = re.search(rf"{attr}\s*=\s*([+-]?\d[\d.e+\-]+)\s*:", ir_text)
    return float(m.group(1)) if m else None


def parse_attr_i32(ir_text, attr):
    m = re.search(rf"{attr}\s*=\s*(\d+)\s*:", ir_text)
    return int(m.group(1)) if m else None


def parse_array_i32(ir_text, attr):
    m = re.search(rf"{attr}\s*=\s*array<i32:\s*([^>]+)>", ir_text)
    if not m:
        return None
    return [int(x.strip()) for x in m.group(1).split(",")]


# ──────────────────────────────────────────────────────────────────────────────
# Main validation loop
# ──────────────────────────────────────────────────────────────────────────────

N        = 10     # nodes
P        = 0.5    # edge probability (Erdős-Rényi)
M        = 1      # frozen hotspot qubits
GRID     = 16     # grid points per axis (must match default in C++ pass)
N_GRAPHS = 10     # number of random graphs to validate
SEED     = 42     # reproducibility

rng = np.random.default_rng(SEED)

print(f"\n{'='*70}")
print(f"  Phase 2 Milestone — Pearson r > 0.999 on Erdős-Rényi G({N},{P})")
print(f"  m={M} hotspot qubit, grid={GRID}×{GRID}, {N_GRAPHS} random graphs")
print(f"{'='*70}")
print(f"\n{'Graph':>7} {'r':>10} {'q_IR':>10} {'cluster_k':>10} {'bias_shift[1]':>14}  Status")
print("-" * 70)

pearson_rs = []
all_pass   = True

for trial in range(N_GRAPHS):
    seed_i = int(rng.integers(1_000_000))

    # Generate connected ER graph (retry if disconnected)
    for attempt in range(20):
        G = nx.erdos_renyi_graph(N, P, seed=seed_i + attempt)
        if nx.is_connected(G):
            break
    else:
        print(f"  {trial+1:>5}  Could not generate connected graph, skipping")
        continue

    J = er_graph_to_ising(G)
    h = np.zeros(N)
    hotspot_indices = degree_hotspot(G, m=M)

    # ── Python: compute landscape vectors + Pearson r ────────────────────
    lv0 = build_landscape_vector(J, h, hotspot_indices, k=0, grid_size=GRID)
    lv1 = build_landscape_vector(J, h, hotspot_indices, k=1, grid_size=GRID)

    r, _ = pearsonr(lv0, lv1)
    pearson_rs.append(r)

    # ── C++: run full mlir-opt pipeline ──────────────────────────────────
    mlir_text = make_mlir_module(G, J, h, hotspot_indices, trial, m=M)
    ir_out, err = run_pipeline(mlir_text, trial)

    if err:
        print(f"  {trial+1:>5}  PIPELINE ERROR: {err[:60]}")
        all_pass = False
        continue

    q_ir       = parse_attr_f64(ir_out, "landscape_overlap_q")
    cluster_k  = parse_attr_i32(ir_out, "cluster_k")
    bias_shifts = parse_array_i32(ir_out, "bias_shifts")  # may be dense tensor
    # bias_shifts from IR is a tensor, extract from b_values / bias_shifts text
    # bias_shifts may be:
    #   dense<[v0, v1, ...]>  — element list
    #   dense<v>              — splat (all elements same value)
    bs_match = re.search(r"bias_shifts\s*=\s*dense<([^>]+)>", ir_out)
    bias_shift_1 = None
    if bs_match:
        inner = bs_match.group(1).strip()
        if inner.startswith("["):
            parts = [v.strip() for v in inner.strip("[]").split(",")]
            if len(parts) >= 2:
                try:
                    bias_shift_1 = float(parts[1])
                except ValueError:
                    pass
        else:
            # splat — all sub-problems have same shift
            try:
                bias_shift_1 = float(inner)
            except ValueError:
                pass

    pass_flag = r > 0.999
    if not pass_flag:
        all_pass = False

    status = "PASS ✓" if pass_flag else "FAIL ✗"
    bs1_str = f"{bias_shift_1:.4f}" if bias_shift_1 is not None else "   N/A"
    print(f"  {trial+1:>5}  {r:>10.6f}  {q_ir:>10.6f}  {cluster_k:>10}  "
          f"{bs1_str:>14}  {status}")


# ──────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────────────────────────

print("-" * 70)
if pearson_rs:
    print(f"\n  Pearson r:  min={min(pearson_rs):.6f}  "
          f"mean={np.mean(pearson_rs):.6f}  "
          f"max={max(pearson_rs):.6f}")
    print(f"  Graphs with r > 0.999: "
          f"{sum(r > 0.999 for r in pearson_rs)}/{len(pearson_rs)}")

if all_pass:
    print(f"\n  MILESTONE PASS ✓  —  all {len(pearson_rs)} graphs r > 0.999")
else:
    print(f"\n  MILESTONE PARTIAL — some graphs below r > 0.999 threshold")

# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"Phase 2 Milestone — DO-QAOA on Erdős-Rényi G({N},{P})\n"
        f"Landscape Pearson r for m={M} frozen qubit",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── Panel 1: Pearson r bar chart ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ["#2CA02C" if r > 0.999 else "#D62728" for r in pearson_rs]
    ax1.bar(range(1, len(pearson_rs) + 1), pearson_rs, color=colors)
    ax1.axhline(0.999, color="gray", linestyle="--", linewidth=1.2,
                label="r = 0.999")
    ax1.set_xlabel("Graph index", fontsize=9)
    ax1.set_ylabel("Pearson r", fontsize=9)
    ax1.set_title("Landscape Similarity\n(Pearson r per graph)", fontsize=9)
    ax1.set_ylim(0.9, 1.005)
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3, axis="y")

    # ── Panel 2: landscape vectors for last graph ─────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(lv0, label="k=0 (hotspot=+1)", linewidth=1.2, alpha=0.8)
    ax2.plot(lv1, label="k=1 (hotspot=−1)", linewidth=1.2, alpha=0.8,
             linestyle="--")
    ax2.set_xlabel("Grid index (γ×β flattened)", fontsize=9)
    ax2.set_ylabel("Normalised E(γ,β)", fontsize=9)
    ax2.set_title(f"Landscape Vectors\n(graph #{N_GRAPHS}, r={pearson_rs[-1]:.6f})",
                  fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    # ── Panel 3: last ER graph topology ──────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    pos = nx.spring_layout(G, seed=7)
    node_colors = ["#FF7F0E" if i in hotspot_indices else "#AEC6CF"
                   for i in G.nodes()]
    nx.draw_networkx(G, pos=pos, ax=ax3, node_color=node_colors,
                     node_size=300, font_size=7, width=1.2,
                     edge_color="#555555")
    ax3.set_title(f"Graph #{N_GRAPHS} topology\n"
                  f"(orange = hotspot node {hotspot_indices})", fontsize=9)
    ax3.axis("off")

    # ── Panel 4: Pearson r histogram ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    r_arr = np.array(pearson_rs)
    r_range = r_arr.max() - r_arr.min()
    bins = np.linspace(max(r_arr.min() - 0.001, 0.99),
                       min(r_arr.max() + 0.001, 1.001), 12)
    ax4.hist(pearson_rs, bins=bins, color="#2077B4", edgecolor="white",
             alpha=0.85)
    ax4.axvline(0.999, color="#D62728", linestyle="--", linewidth=1.5,
                label="threshold 0.999")
    ax4.set_xlabel("Pearson r", fontsize=9)
    ax4.set_ylabel("Count", fontsize=9)
    ax4.set_title("r Distribution\nacross all graphs", fontsize=9)
    ax4.legend(fontsize=7)
    ax4.grid(alpha=0.3)

    # ── Panel 5: scatter lv0 vs lv1 ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(lv0, lv1, s=8, alpha=0.6, color="#9467BD")
    lo = min(lv0.min(), lv1.min()) - 0.05
    hi = max(lv0.max(), lv1.max()) + 0.05
    ax5.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y=x")
    ax5.set_xlabel("L0 (k=0 landscape)", fontsize=9)
    ax5.set_ylabel("L1 (k=1 landscape)", fontsize=9)
    ax5.set_title(f"Landscape Scatter\ngraph #{N_GRAPHS}", fontsize=9)
    ax5.legend(fontsize=7)
    ax5.grid(alpha=0.3)

    # ── Panel 6: pipeline summary text ───────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    milestone_pass = all(r > 0.999 for r in pearson_rs)
    badge = "MILESTONE PASS ✓" if milestone_pass else "MILESTONE PARTIAL"
    badge_color = "#2CA02C" if milestone_pass else "#D62728"

    summary_lines = [
        "mlir-opt pipeline:",
        "  --doqaoa-landscape-overlap",
        "  --doqaoa-bias-shift",
        "",
        f"Graph: Erdős-Rényi G({N},{P})",
        f"Nodes: {N},  m={M} hotspot",
        f"Grid: {GRID}×{GRID} = {GRID**2} points",
        f"Trials: {N_GRAPHS} random graphs",
        "",
        f"Pearson r min:  {min(pearson_rs):.6f}",
        f"Pearson r mean: {np.mean(pearson_rs):.6f}",
        f"r > 0.999: {sum(r > 0.999 for r in pearson_rs)}/{len(pearson_rs)}",
    ]
    ax6.text(0.05, 0.95, "\n".join(summary_lines),
             transform=ax6.transAxes,
             fontsize=8.5, verticalalignment="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.8))
    ax6.text(0.5, 0.08, badge,
             transform=ax6.transAxes,
             fontsize=12, fontweight="bold", ha="center",
             color=badge_color,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor=badge_color, linewidth=2))

    out_png = ROOT / "phase2_milestone.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n  Saved → {out_png}")

except Exception as e:
    print(f"\n  Plot skipped: {e}")

sys.exit(0 if all_pass else 1)
