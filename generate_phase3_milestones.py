#!/usr/bin/env python3
"""Generate phase3_milestones.png — Phase 3 summary figure."""

import math
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

# ── Reproduce the milestone run data ────────────────────────────────────────

def build_ba_graph(n=10, seed=42):
    rng = random.Random(seed)
    adj = {i: set() for i in range(n)}
    def add_edge(u, v):
        adj[u].add(v); adj[v].add(u)
    add_edge(0,1); add_edge(1,2); add_edge(0,2)
    degrees = [2,2,2]
    for new_node in range(3, n):
        total_deg = sum(degrees)
        targets = set()
        while len(targets) < 2:
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
                J[u,v] = J[v,u] = -0.5
    return J

def energy_closed_form(J_free, h_eff, gamma, beta, nf):
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

def evaluate_sub(J, h, hotspots, k, gamma, beta):
    n = len(h)
    frozen = {hotspots[i]: (-1 if (k>>i)&1 else 1) for i in range(len(hotspots))}
    free = [q for q in range(n) if q not in frozen]
    nf = len(free)
    h_eff = np.array([h[q] + sum(J[q,fq]*frozen[fq] for fq in frozen) for q in free])
    J_free = np.array([[J[free[i],free[j]] for j in range(nf)] for i in range(nf)])
    return energy_closed_form(J_free, h_eff, gamma, beta, nf)

def run_adam(J, h, hotspots, k, g0, b0, max_ep, lr=0.01, tol=1e-4, h_fd=1e-4,
             b1=0.9, b2=0.999, eps=1e-8):
    g, b = g0, b0
    mg=mb=vg=vb = 0.0
    history = []
    shots = 0
    for ep in range(1, max_ep+1):
        E   = evaluate_sub(J,h,hotspots,k,g,b);      shots+=1
        Epg = evaluate_sub(J,h,hotspots,k,g+h_fd,b); shots+=1
        Emg = evaluate_sub(J,h,hotspots,k,g-h_fd,b); shots+=1
        Epb = evaluate_sub(J,h,hotspots,k,g,b+h_fd); shots+=1
        Emb = evaluate_sub(J,h,hotspots,k,g,b-h_fd); shots+=1
        dg = (Epg-Emg)/(2*h_fd)
        db = (Epb-Emb)/(2*h_fd)
        gnorm = math.sqrt(dg**2+db**2)
        history.append((ep, E, gnorm))
        if gnorm < tol:
            break
        mg = b1*mg+(1-b1)*dg; mb = b1*mb+(1-b1)*db
        vg = b2*vg+(1-b2)*dg**2; vb = b2*vb+(1-b2)*db**2
        mhg=mg/(1-b1**ep); mhb=mb/(1-b1**ep)
        vhg=vg/(1-b2**ep); vhb=vb/(1-b2**ep)
        g -= lr*mhg/(math.sqrt(vhg)+eps)
        b -= lr*mhb/(math.sqrt(vhb)+eps)
        E = evaluate_sub(J,h,hotspots,k,g,b); shots+=1
    return g, b, history, shots

N=10; m=2
adj = build_ba_graph(N, 42)
J   = adj_to_J(adj, N)
h   = np.zeros(N)
degrees = {u: len(v) for u,v in adj.items()}
hotspots = sorted(sorted(degrees, key=degrees.__getitem__, reverse=True)[:m])

g_rep, b_rep, hist_rep, shots_rep = run_adam(
    J, h, hotspots, 0, -math.pi/6, -math.pi/8, max_ep=100)

# Phase 2/3 warm-start (k=1)
_, _, hist_ws, shots_ws = run_adam(
    J, h, hotspots, 1, g_rep, b_rep, max_ep=10)

epochs_rep   = [x[0] for x in hist_rep]
energy_rep   = [x[1] for x in hist_rep]
gnorm_rep    = [x[2] for x in hist_rep]
epochs_ws    = [x[0] for x in hist_ws]
energy_ws    = [x[1] for x in hist_ws]

total_shots  = shots_rep + shots_ws + 1  # +1 final eval
total_cumulative = list(np.cumsum([1]*total_shots))

# ── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#0d1117")

gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.45, wspace=0.38,
                       left=0.07, right=0.97, top=0.88, bottom=0.09)

ax_bar   = fig.add_subplot(gs[0, 0])   # shot-count bar chart
ax_conv  = fig.add_subplot(gs[0, 1])   # convergence curve
ax_gnorm = fig.add_subplot(gs[0, 2])   # gradient norm
ax_graph = fig.add_subplot(gs[1, 0])   # BA graph topology
ax_phase = fig.add_subplot(gs[1, 1])   # phase breakdown pie
ax_land  = fig.add_subplot(gs[1, 2])   # energy landscape heatmap

BG   = "#0d1117"
CARD = "#161b22"
GRID = "#21262d"
C1   = "#58a6ff"   # blue
C2   = "#3fb950"   # green
C3   = "#f78166"   # red/orange
C4   = "#d2a8ff"   # purple
CTXT = "#e6edf3"
CDIM = "#8b949e"

for ax in [ax_bar, ax_conv, ax_gnorm, ax_graph, ax_phase, ax_land]:
    ax.set_facecolor(CARD)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=CDIM, labelsize=8)
    ax.xaxis.label.set_color(CDIM)
    ax.yaxis.label.set_color(CDIM)

# ── 1. Shot-count bar chart ──────────────────────────────────────────────────
labels  = ["FrozenQubits\nbaseline", "DO-QAOA\nTarget\n(≤0.13M)", "DO-QAOA\nActual"]
values  = [32_770_000, 130_000, total_shots]
colors  = [C3, C4, C2]
bars = ax_bar.bar(labels, values, color=colors, width=0.5,
                  edgecolor=GRID, linewidth=0.8)
ax_bar.set_yscale("log")
ax_bar.set_ylabel("Quantum shots (log scale)", fontsize=8, color=CDIM)
ax_bar.set_title("Shot Count Comparison", color=CTXT, fontsize=10, fontweight="bold")
for bar, val in zip(bars, values):
    ax_bar.text(bar.get_x()+bar.get_width()/2, val*1.6,
                f"{val:,}", ha="center", va="bottom",
                color=CTXT, fontsize=7.5, fontweight="bold")
ax_bar.set_ylim(100, 2e8)
ax_bar.yaxis.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
ax_bar.set_axisbelow(True)

speedup_txt = f"×{32_770_000//total_shots:,} speedup"
ax_bar.text(0.97, 0.97, speedup_txt, transform=ax_bar.transAxes,
            ha="right", va="top", color=C2, fontsize=9, fontweight="bold")

# ── 2. Convergence curve ─────────────────────────────────────────────────────
ax_conv.plot(epochs_rep, energy_rep, color=C1, linewidth=1.8, label="Rep k=0 (full opt)")
ax_conv.axhline(energy_rep[-1], color=CDIM, linewidth=0.8, linestyle="--", alpha=0.6)
ax_conv.set_xlabel("Epoch", fontsize=8)
ax_conv.set_ylabel("⟨H⟩  (QAOA energy)", fontsize=8)
ax_conv.set_title("Phase 1: Representative Optimisation", color=CTXT, fontsize=10, fontweight="bold")
ax_conv.legend(fontsize=7.5, facecolor=CARD, edgecolor=GRID, labelcolor=CTXT)
ax_conv.yaxis.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
ax_conv.set_axisbelow(True)
ax_conv.text(0.97, 0.05,
             f"Final ⟨H⟩ = {energy_rep[-1]:.4f}\n{len(epochs_rep)} epochs",
             transform=ax_conv.transAxes, ha="right", va="bottom",
             color=C1, fontsize=7.5)

# ── 3. Gradient norm ─────────────────────────────────────────────────────────
ax_gnorm.semilogy(epochs_rep, gnorm_rep, color=C4, linewidth=1.5, label="‖∇θ‖₂  rep")
ax_gnorm.axhline(1e-4, color=C3, linewidth=1, linestyle="--", alpha=0.8, label="tol=1e-4")
ax_gnorm.set_xlabel("Epoch", fontsize=8)
ax_gnorm.set_ylabel("Gradient norm (log)", fontsize=8)
ax_gnorm.set_title("Gradient Norm Convergence", color=CTXT, fontsize=10, fontweight="bold")
ax_gnorm.legend(fontsize=7.5, facecolor=CARD, edgecolor=GRID, labelcolor=CTXT)
ax_gnorm.yaxis.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
ax_gnorm.set_axisbelow(True)

# ── 4. BA graph topology ─────────────────────────────────────────────────────
import networkx as nx
G_nx = nx.Graph()
for u, nbrs in adj.items():
    for v in nbrs:
        if u < v:
            G_nx.add_edge(u, v)

pos = nx.spring_layout(G_nx, seed=7)
node_colors = []
for nd in G_nx.nodes():
    if nd in hotspots:
        node_colors.append(C3)
    else:
        node_colors.append(C1)

nx.draw_networkx_edges(G_nx, pos, ax=ax_graph,
                       edge_color=GRID, width=1.2, alpha=0.9)
nx.draw_networkx_nodes(G_nx, pos, ax=ax_graph,
                       node_color=node_colors, node_size=280,
                       edgecolors=CTXT, linewidths=0.8)
nx.draw_networkx_labels(G_nx, pos, ax=ax_graph,
                        font_color=BG, font_size=7, font_weight="bold")
ax_graph.set_title(f"10-node BA Graph  (m=2 hotspots={hotspots})",
                   color=CTXT, fontsize=10, fontweight="bold")
ax_graph.axis("off")
ax_graph.set_facecolor(CARD)
# Legend
from matplotlib.lines import Line2D
leg_elems = [Line2D([0],[0], marker='o', color='w', markerfacecolor=C3,
                    markersize=8, label=f"Hotspot {hotspots}"),
             Line2D([0],[0], marker='o', color='w', markerfacecolor=C1,
                    markersize=8, label="Free qubit")]
ax_graph.legend(handles=leg_elems, fontsize=7, facecolor=CARD,
                edgecolor=GRID, labelcolor=CTXT, loc="lower left")

# ── 5. Phase breakdown pie ───────────────────────────────────────────────────
phase_labels = ["Phase 1\nFull opt\n(k=0)", "Phase 2\nWarm-start\n(k=1)", "Phase 3\nDirect copy\n(k=2,3)"]
phase_shots  = [shots_rep, shots_ws, 1]
phase_colors = [C1, C4, C2]
wedges, texts, autotexts = ax_phase.pie(
    phase_shots, labels=phase_labels, colors=phase_colors,
    autopct="%1.1f%%", startangle=90,
    textprops={"color": CTXT, "fontsize": 7.5},
    wedgeprops={"edgecolor": CARD, "linewidth": 1.5})
for at in autotexts:
    at.set_color(BG)
    at.set_fontweight("bold")
    at.set_fontsize(7)
ax_phase.set_title("Shot Budget by Phase", color=CTXT, fontsize=10, fontweight="bold")
ax_phase.text(0, -1.45, f"Total: {total_shots} shots",
              ha="center", color=C2, fontsize=9, fontweight="bold")

# ── 6. Energy landscape heatmap (γ vs β) ────────────────────────────────────
g_range = np.linspace(-math.pi/2, math.pi/2, 60)
b_range = np.linspace(-math.pi/4, math.pi/4, 60)
Z = np.zeros((len(b_range), len(g_range)))
for bi, bv in enumerate(b_range):
    for gi, gv in enumerate(g_range):
        Z[bi, gi] = evaluate_sub(J, h, hotspots, 0, gv, bv)

im = ax_land.contourf(g_range, b_range, Z, levels=30, cmap="RdYlBu_r")
cb = fig.colorbar(im, ax=ax_land, pad=0.02)
cb.ax.tick_params(colors=CDIM, labelsize=7)
cb.set_label("⟨H⟩", color=CDIM, fontsize=7)
ax_land.scatter([g_rep], [b_rep], color=C2, s=80, zorder=5,
                marker="*", linewidths=0.8, edgecolors=CTXT, label=f"θ* = ({g_rep:.3f}, {b_rep:.3f})")
ax_land.set_xlabel("γ", fontsize=8)
ax_land.set_ylabel("β", fontsize=8)
ax_land.set_title("Energy Landscape  E(γ, β)  — k=0", color=CTXT, fontsize=10, fontweight="bold")
ax_land.legend(fontsize=7, facecolor=CARD, edgecolor=GRID, labelcolor=CTXT, loc="upper right")

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle(
    "Phase 3 Milestone — DO-QAOA  10-node MaxCut  m=2\n"
    "Sang et al., arXiv:2602.21689v1 · Table IV",
    color=CTXT, fontsize=13, fontweight="bold", y=0.96)

out = "/Users/khounalexa/ALEXA/PKNU/lab/code/Do-QAOA-Implementation/do-qaoa/phase3_milestones.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")
