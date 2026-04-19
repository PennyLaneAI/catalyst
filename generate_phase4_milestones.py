#!/usr/bin/env python3
"""Generate phase4_milestones.png — Phase 4 summary figure."""

import math
import sys
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ── Load DO-QAOA module ──────────────────────────────────────────────────────
import importlib.util
spec = importlib.util.spec_from_file_location(
    "doqaoa", "catalyst/frontend/catalyst/api_extensions/doqaoa.py"
)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

DOQAOAConfig = _mod.DOQAOAConfig
DOQAOAResult = _mod.DOQAOAResult

# ── Reproduce milestone data ─────────────────────────────────────────────────

def build_ba_graph(n=12, m_ba=2, seed=42):
    rng = random.Random(seed)
    adj = {i: set() for i in range(n)}
    def add_edge(u, v):
        adj[u].add(v); adj[v].add(u)
    add_edge(0,1); add_edge(1,2); add_edge(0,2)
    degrees = [2,2,2]
    for new_node in range(3, n):
        total_deg = sum(degrees)
        targets = set()
        while len(targets) < m_ba:
            r = rng.random() * total_deg
            acc = 0
            for node, d in enumerate(degrees):
                acc += d
                if r <= acc:
                    targets.add(node); break
        for t in targets:
            add_edge(new_node, t)
        degrees.append(len(adj[new_node]))
        for t in targets:
            degrees[t] += 1
    return adj

def adj_to_J(adj, n):
    J = np.zeros((n, n))
    for u, nbrs in adj.items():
        for v in nbrs:
            if u < v:
                J[u, v] = J[v, u] = -0.5
    return J

def energy_cf(J_free, h_eff, gamma, beta, nf):
    E = 0.0
    for u in range(nf):
        for v in range(u+1, nf):
            Juv = J_free[u,v]
            if abs(Juv) > 1e-12:
                E += Juv/2 * math.sin(4*beta) * math.sin(2*gamma*Juv)
    for u in range(nf):
        hu = h_eff[u]
        if abs(hu) > 1e-12:
            E += hu * (-math.sin(2*beta) * math.cos(2*gamma*hu))
    return E

def make_energy(J, h, hotspots, k):
    n = len(h)
    frozen = {hotspots[i]: (-1 if (k>>i)&1 else 1) for i in range(len(hotspots))}
    free = [q for q in range(n) if q not in frozen]
    nf = len(free)
    h_eff = np.array([h[q]+sum(J[q,fq]*frozen[fq] for fq in frozen) for q in free])
    J_free = np.array([[J[free[i],free[j]] for j in range(nf)] for i in range(nf)])
    def _e(params): return energy_cf(J_free, h_eff, params[0], params[1], nf)
    return _e

def run_adam(energy_fn, g0, b0, max_ep, lr=0.01, tol=1e-4, h_fd=1e-4,
             b1=0.9, b2=0.999, eps=1e-8):
    g, b = g0, b0
    mg=mb=vg=vb=0.0
    hist = []
    shots = 0
    for ep in range(1, max_ep+1):
        E = energy_fn(np.array([g, b]));             shots+=1
        dg = (energy_fn(np.array([g+h_fd,b])) - energy_fn(np.array([g-h_fd,b])))/(2*h_fd); shots+=2
        db = (energy_fn(np.array([g,b+h_fd])) - energy_fn(np.array([g,b-h_fd])))/(2*h_fd); shots+=2
        gnorm = math.sqrt(dg**2+db**2)
        hist.append((ep, E, gnorm))
        if gnorm < tol: break
        mg=b1*mg+(1-b1)*dg; mb=b1*mb+(1-b1)*db
        vg=b2*vg+(1-b2)*dg**2; vb=b2*vb+(1-b2)*db**2
        mhg=mg/(1-b1**ep); mhb=mb/(1-b1**ep)
        vhg=vg/(1-b2**ep); vhb=vb/(1-b2**ep)
        g -= lr*mhg/(math.sqrt(vhg)+eps)
        b -= lr*mhb/(math.sqrt(vhb)+eps)
        E = energy_fn(np.array([g, b])); shots+=1   # convergence check
    return g, b, hist, shots

N=12; m=3; num_sp=1<<m
adj = build_ba_graph(N, 2, 42)
J   = adj_to_J(adj, N)
h   = np.zeros(N)
hotspots = sorted(sorted(adj, key=lambda u: len(adj[u]), reverse=True)[:m])

# Phase 1: full opt on k=0
g_rep, b_rep, hist_rep, shots_p1 = run_adam(
    make_energy(J, h, hotspots, 0), -math.pi/6, -math.pi/8, 100)

# Phase 2: warm-start on k that has max bias shift
def bias(k_idx):
    frozen = {hotspots[i]: (-1 if (k_idx>>i)&1 else 1) for i in range(m)}
    free = [q for q in range(N) if q not in frozen]
    h_eff = np.array([h[q]+sum(J[q,fq]*frozen[fq] for fq in frozen) for q in free])
    return sum(abs(v) for v in h_eff)/max(len(free),1)

b0 = bias(0)
delta_bs = {k: abs(bias(k)-b0) for k in range(1, num_sp)}
ws_k = max(delta_bs, key=delta_bs.__getitem__)   # sub-problem with largest ΔB

_, _, hist_ws, shots_p2 = run_adam(
    make_energy(J, h, hotspots, ws_k), g_rep, b_rep, 10)

# Real milestone numbers (from milestone_phase4.py run — bias_threshold=0.3 → all copies)
REAL_SHOTS      = 508
REAL_WARMSTARTS = 0
REAL_DIRECT     = 7
REAL_SPEEDUP    = 64508
REAL_ENERGY     = -4.838102

total_shots       = REAL_SHOTS
warmstart_count   = REAL_WARMSTARTS
direct_copy_count = REAL_DIRECT

epochs_rep  = [x[0] for x in hist_rep]
energy_rep  = [x[1] for x in hist_rep]
gnorm_rep   = [x[2] for x in hist_rep]
epochs_ws   = [x[0] for x in hist_ws]
energy_ws   = [x[1] for x in hist_ws]

# ── API usage comparison (new in Phase 4) ───────────────────────────────────
frozenqubits_calls = num_sp   # 8 independent calls
doqaoa_calls = 2              # 1 full + 1 warmstart

# ── Figure ───────────────────────────────────────────────────────────────────
BG   = "#0d1117"; CARD = "#161b22"; GRID = "#21262d"
C1   = "#58a6ff"; C2   = "#3fb950"; C3   = "#f78166"; C4   = "#d2a8ff"; C5 = "#ffa657"
CTXT = "#e6edf3"; CDIM = "#8b949e"

fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.44, wspace=0.36,
                       left=0.06, right=0.97, top=0.88, bottom=0.08)

ax_bar   = fig.add_subplot(gs[0, 0])   # shot-count bar chart
ax_conv  = fig.add_subplot(gs[0, 1])   # phase 1 convergence
ax_api   = fig.add_subplot(gs[0, 2])   # API complexity comparison
ax_graph = fig.add_subplot(gs[1, 0])   # BA graph topology
ax_pie   = fig.add_subplot(gs[1, 1])   # phase breakdown pie
ax_land  = fig.add_subplot(gs[1, 2])   # energy landscape

for ax in [ax_bar, ax_conv, ax_api, ax_graph, ax_pie, ax_land]:
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=CDIM, labelsize=8)
    ax.xaxis.label.set_color(CDIM)
    ax.yaxis.label.set_color(CDIM)

# ── 1. Shot-count bar chart ──────────────────────────────────────────────────
labels  = ["FrozenQubits\nbaseline", "DO-QAOA\nTarget\n(≤0.23M)", f"DO-QAOA\nActual\n({total_shots} shots)"]
values  = [32_770_000, 230_000, total_shots]
colors  = [C3, C4, C2]
bars = ax_bar.bar(labels, values, color=colors, width=0.5,
                  edgecolor=GRID, linewidth=0.8)
ax_bar.set_yscale("log")
ax_bar.set_ylabel("Quantum shots (log scale)", fontsize=8)
ax_bar.set_title("Shot Count Comparison  (m=3)", color=CTXT, fontsize=10, fontweight="bold")
for bar, val in zip(bars, values):
    ax_bar.text(bar.get_x()+bar.get_width()/2, val*1.6,
                f"{val:,}", ha="center", va="bottom", color=CTXT,
                fontsize=7.5, fontweight="bold")
ax_bar.set_ylim(100, 2e8)
ax_bar.yaxis.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
ax_bar.set_axisbelow(True)
ax_bar.text(0.97, 0.97, f"×{REAL_SPEEDUP:,} speedup",
            transform=ax_bar.transAxes, ha="right", va="top",
            color=C2, fontsize=9, fontweight="bold")

# ── 2. Phase 1 convergence ───────────────────────────────────────────────────
ax_conv.plot(epochs_rep, energy_rep, color=C1, lw=1.8, label="Rep k=0 (full opt)")
ax_conv.axhline(energy_rep[-1], color=CDIM, lw=0.8, ls=":", alpha=0.6)
ax_conv.set_xlabel("Epoch"); ax_conv.set_ylabel("⟨H⟩  (QAOA energy)")
ax_conv.set_title("Convergence Curve (Phase 1 — Rep k=0)", color=CTXT, fontsize=10, fontweight="bold")
ax_conv.legend(fontsize=7.5, facecolor=CARD, edgecolor=GRID, labelcolor=CTXT)
ax_conv.yaxis.grid(True, color=GRID, lw=0.5, alpha=0.7); ax_conv.set_axisbelow(True)
ax_conv.text(0.97, 0.97, f"7 sub-problems → direct copy\n(bias_threshold=0.3)",
             transform=ax_conv.transAxes, ha="right", va="top", color=C2, fontsize=7.5)
ax_conv.text(0.97, 0.05, f"Final ⟨H⟩={REAL_ENERGY:.4f}",
             transform=ax_conv.transAxes, ha="right", va="bottom", color=C1, fontsize=7.5)

# ── 3. API complexity comparison ─────────────────────────────────────────────
ax_api.axis("off")
ax_api.set_title("Phase 4 API — One Call Replaces Eight", color=CTXT,
                 fontsize=10, fontweight="bold")

before_code = (
    "# FrozenQubits (before)\n"
    "for k in range(2**m):\n"
    "    results[k] = qjit(circuit)(\n"
    "        params_k)"
)
after_code = (
    "# DO-QAOA (after)\n"
    "result = catalyst.do_qaoa(\n"
    "    circuit, H, m=3)(G)"
)

# Before box
bbox_b = FancyBboxPatch((0.02, 0.20), 0.44, 0.62,
                        boxstyle="round,pad=0.02", linewidth=1.2,
                        edgecolor=C3, facecolor="#1a1f2e")
ax_api.add_patch(bbox_b)
ax_api.text(0.24, 0.87, "BEFORE", ha="center", va="center",
            color=C3, fontsize=9, fontweight="bold", transform=ax_api.transAxes)
ax_api.text(0.24, 0.53, before_code, ha="center", va="center",
            color=CTXT, fontsize=7.5, fontfamily="monospace",
            transform=ax_api.transAxes)

# After box
bbox_a = FancyBboxPatch((0.54, 0.20), 0.44, 0.62,
                        boxstyle="round,pad=0.02", linewidth=1.2,
                        edgecolor=C2, facecolor="#0f1f17")
ax_api.add_patch(bbox_a)
ax_api.text(0.76, 0.87, "AFTER", ha="center", va="center",
            color=C2, fontsize=9, fontweight="bold", transform=ax_api.transAxes)
ax_api.text(0.76, 0.53, after_code, ha="center", va="center",
            color=CTXT, fontsize=7.5, fontfamily="monospace",
            transform=ax_api.transAxes)

# Arrow
ax_api.annotate("", xy=(0.54, 0.51), xytext=(0.46, 0.51),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=C4, lw=2.0))

stats = (
    f"8 calls → 1 call\n"
    f"8 opt loops → 1 full + 7 direct copies\n"
    f"32.77M shots → {total_shots} shots\n"
    f"Speedup: {REAL_SPEEDUP:,}×"
)
ax_api.text(0.5, 0.10, stats, ha="center", va="center",
            color=C4, fontsize=8, fontweight="bold",
            transform=ax_api.transAxes)

# ── 4. BA graph topology ─────────────────────────────────────────────────────
import networkx as nx
G_nx = nx.Graph()
for u, nbrs in adj.items():
    for v in nbrs:
        if u < v:
            G_nx.add_edge(u, v)
pos = nx.spring_layout(G_nx, seed=7)
node_colors = [C3 if nd in hotspots else C1 for nd in G_nx.nodes()]
nx.draw_networkx_edges(G_nx, pos, ax=ax_graph, edge_color=GRID, width=1.2, alpha=0.9)
nx.draw_networkx_nodes(G_nx, pos, ax=ax_graph, node_color=node_colors,
                       node_size=280, edgecolors=CTXT, linewidths=0.8)
nx.draw_networkx_labels(G_nx, pos, ax=ax_graph,
                        font_color=BG, font_size=7, font_weight="bold")
ax_graph.set_title(f"12-node BA Graph  (m=3 hotspots={hotspots})",
                   color=CTXT, fontsize=10, fontweight="bold")
ax_graph.axis("off"); ax_graph.set_facecolor(CARD)
from matplotlib.lines import Line2D
ax_graph.legend(handles=[
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C3, markersize=8, label=f"Hotspot {hotspots}"),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C1, markersize=8, label="Free qubit"),
], fontsize=7, facecolor=CARD, edgecolor=GRID, labelcolor=CTXT, loc="lower left")

# ── 5. Phase breakdown pie ───────────────────────────────────────────────────
phase_labels = [f"Phase 1\nFull opt\n(k=0)",
                f"Phase 3\nDirect copy\n(×{direct_copy_count})"]
phase_shots  = [shots_p1, 1]
phase_colors = [C1, C2]
wedges, texts, autotexts = ax_pie.pie(
    phase_shots, labels=phase_labels, colors=phase_colors,
    autopct="%1.1f%%", startangle=90,
    textprops={"color": CTXT, "fontsize": 7.5},
    wedgeprops={"edgecolor": CARD, "linewidth": 1.5})
for at in autotexts:
    at.set_color(BG); at.set_fontweight("bold"); at.set_fontsize(7)
ax_pie.set_title("Shot Budget by Phase", color=CTXT, fontsize=10, fontweight="bold")
ax_pie.text(0, -1.45, f"Total: {total_shots} shots  (0 warm-starts)",
            ha="center", color=C2, fontsize=9, fontweight="bold")

# ── 6. Energy landscape E(γ,β) for k=0 ──────────────────────────────────────
g_range = np.linspace(-math.pi/2, math.pi/2, 55)
b_range = np.linspace(-math.pi/4, math.pi/4, 55)
E0 = make_energy(J, h, hotspots, 0)
Z = np.array([[E0(np.array([gv, bv])) for gv in g_range] for bv in b_range])

im = ax_land.contourf(g_range, b_range, Z, levels=30, cmap="RdYlBu_r")
cb = fig.colorbar(im, ax=ax_land, pad=0.02)
cb.ax.tick_params(colors=CDIM, labelsize=7)
cb.set_label("⟨H⟩", color=CDIM, fontsize=7)
ax_land.scatter([g_rep], [b_rep], color=C2, s=120, zorder=5, marker="*",
                edgecolors=CTXT, linewidths=0.8,
                label=f"θ* = ({g_rep:.3f}, {b_rep:.3f})")
ax_land.scatter([g_rep], [b_rep], color=C5, s=40, zorder=4, marker="x",
                linewidths=1.5, label=f"warm-start init")
ax_land.set_xlabel("γ"); ax_land.set_ylabel("β")
ax_land.set_title("Energy Landscape  E(γ,β)  — k=0", color=CTXT, fontsize=10, fontweight="bold")
ax_land.legend(fontsize=7, facecolor=CARD, edgecolor=GRID, labelcolor=CTXT, loc="upper right")

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle(
    "Phase 4 Milestone — catalyst.do_qaoa()  12-node MaxCut  m=3\n"
    "Sang et al., arXiv:2602.21689v1 · Table IV",
    color=CTXT, fontsize=13, fontweight="bold", y=0.96)

out = "/Users/khounalexa/ALEXA/PKNU/lab/code/Do-QAOA-Implementation/do-qaoa/phase4_milestones.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")
