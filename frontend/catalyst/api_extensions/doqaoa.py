# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DO-QAOA (Doubly Optimized QAOA) frontend API for Catalyst.

Implements the Bias-Aware Transfer Rule from Sang et al., arXiv:2602.21689v1.
Reduces O(2^m) training sessions to O(K≈1) by exploiting landscape similarity
across frozen sub-problems.
"""

import functools
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import pennylane as qml


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DOQAOAConfig:
    """Configuration for DO-QAOA execution.

    Args:
        m (int): Number of hotspot (frozen) qubits. Produces 2^m sub-problems.
            Select the m highest-degree nodes via degree centrality (Section 1.1).
        bias_threshold (float): ΔB threshold for direct copy vs warm-start transfer.
            If |B_target − B_rep| < bias_threshold, parameters are copied directly
            with zero extra training. Paper default: 0.3 (Section 2.2).
        warmstart_epochs (int): Adam optimisation epochs for warm-start branch (phase 2).
            Only applied to sub-problems where ΔB ≥ bias_threshold. Default 10.
        init_strategy (str): Initialisation strategy for the representative sub-circuit.
            ``"shortcut"`` — p=1 analytic values (γ=−π/6, β=−π/8), near the
            optimum for unweighted MaxCut (Eq. 9 of arXiv:2602.21689v1).
            ``"random"`` — uniform random in [−π, π], controlled by ``seed``.
        k_max (int | None): Maximum number of landscape clusters K. ``None`` = auto
            (K=1 when sparsity s > sc ≈ 0.6, elbow heuristic otherwise). For sparse
            BA graphs K=1 almost always holds.
        landscape_grid_size (int): Resolution of the (γ, β) grid for landscape
            overlap sampling. Default 16 → 16×16 = 256 evaluations per sub-problem.
        gradient_fn (str | callable): Optimiser for phase 1 (representative) and
            phase 2 (warm-start). ``"adam"`` (default) uses bias-corrected Adam;
            ``"gd"`` uses vanilla gradient descent; a callable must accept
            ``(params, grads, lr)`` and return updated params.
        max_warmstarts (int): Hard cap on the number of warm-start sub-problems
            actually trained. Sub-problems beyond this cap receive direct copy,
            matching the DO-QAOA paper's ≤ 1 warm-start budget. Default 1.
    """
    m: int
    bias_threshold: float = 0.3
    warmstart_epochs: int = 10
    init_strategy: str = "shortcut"
    k_max: Optional[int] = None
    landscape_grid_size: int = 16
    gradient_fn: str = "adam"
    max_warmstarts: int = 1

    def __post_init__(self):
        if self.m < 1:
            raise ValueError(f"m must be >= 1, got {self.m}")
        if not 0.0 < self.bias_threshold < 1.0:
            raise ValueError(f"bias_threshold must be in (0, 1), got {self.bias_threshold}")
        if self.warmstart_epochs < 1:
            raise ValueError(f"warmstart_epochs must be >= 1, got {self.warmstart_epochs}")
        if self.init_strategy not in ("shortcut", "random"):
            raise ValueError(f"init_strategy must be 'shortcut' or 'random', got {self.init_strategy!r}")
        if self.k_max is not None and self.k_max < 1:
            raise ValueError(f"k_max must be >= 1, got {self.k_max}")
        if self.landscape_grid_size < 4:
            raise ValueError(f"landscape_grid_size must be >= 4, got {self.landscape_grid_size}")
        if self.gradient_fn not in ("adam", "gd") and not callable(self.gradient_fn):
            raise ValueError(
                f"gradient_fn must be 'adam', 'gd', or a callable, got {self.gradient_fn!r}"
            )
        if self.max_warmstarts < 1:
            raise ValueError(f"max_warmstarts must be >= 1, got {self.max_warmstarts}")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DOQAOAResult:
    """Result of a :func:`do_qaoa` optimisation run.

    Attributes:
        best_k (int): Index of the sub-problem with minimum expectation value
            ⟨H_k⟩ over all 2^m frozen-qubit configurations.
        best_energy (float): Minimum ⟨H⟩ value achieved across all sub-problems.
        best_params (numpy.ndarray): Optimised (γ, β) parameter vector for the
            best sub-problem, shape ``(2,)``.
        bitstring (list[int]): Hotspot spin assignments for the best sub-problem.
            ``bitstring[i] = 0`` → hotspot qubit i in spin +1 state (bit=0),
            ``bitstring[i] = 1`` → hotspot qubit i in spin −1 state (bit=1).
        total_shots (int): Total number of energy evaluations (quantum shots) used
            across all phases, including finite-difference gradient steps.
        warmstart_count (int): Number of warm-start sub-problems actually trained
            (phase 2). Bounded by ``DOQAOAConfig.max_warmstarts``.
        direct_copy_count (int): Number of sub-problems receiving parameters via
            direct copy (phase 3, zero training cost).
        full_opt_count (int): Number of full optimisation runs (phase 1). Should
            be 1 for K=1 (sparse graphs); equals K for denser graphs.
        speedup_vs_frozen (float): Shot count ratio FrozenQubits / DO-QAOA.
            Populated when ``frozen_qubits_shots`` is passed to the transform call.
    """

    best_k: int
    best_energy: float
    best_params: object          # np.ndarray
    bitstring: list
    total_shots: int
    warmstart_count: int
    direct_copy_count: int = 0
    full_opt_count: int = 1
    speedup_vs_frozen: float = 0.0

    def __repr__(self):
        bits = "".join(str(b) for b in self.bitstring)
        sp = f"  speedup={self.speedup_vs_frozen:.0f}×" if self.speedup_vs_frozen else ""
        return (
            f"DOQAOAResult(best_k={self.best_k}, ⟨H⟩={self.best_energy:.6f}, "
            f"bitstring={bits}, shots={self.total_shots:,}{sp})"
        )


# ---------------------------------------------------------------------------
# Graph analysis utilities
# ---------------------------------------------------------------------------

def degree_centrality_sort(graph):
    """Return node indices sorted by degree centrality (highest first).

    Selects the m hotspot qubits as those with the highest degree — the nodes
    that participate in the most edges and thus most influence the energy
    landscape (Section 1.1 of arXiv:2602.21689v1).

    Args:
        graph: A NetworkX graph whose nodes are integers 0..N-1.

    Returns:
        List[int]: Node indices sorted by degree (descending).
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError(
            "NetworkX is required for DO-QAOA graph analysis. "
            "Install it with: pip install networkx"
        ) from e

    centrality = nx.degree_centrality(graph)
    return sorted(centrality, key=centrality.__getitem__, reverse=True)


def select_hotspot_indices(graph, m: int):
    """Select the m hotspot qubit indices using degree centrality.

    Args:
        graph: NetworkX graph with integer node labels 0..N-1.
        m: Number of hotspot qubits to select.

    Returns:
        List[int]: The m node indices with highest degree, sorted ascending.
    """
    sorted_nodes = degree_centrality_sort(graph)
    hotspots = sorted(sorted_nodes[:m])
    return hotspots


# ---------------------------------------------------------------------------
# Hamiltonian parsing utilities
# ---------------------------------------------------------------------------

def extract_coupling_matrix(hamiltonian):
    """Extract quadratic (J) and linear (h) coefficients from a PennyLane Hamiltonian.

    Parses the Ising-form Hamiltonian:
        H = Σ_{ij} J_ij Z_i Z_j + Σ_i h_i Z_i

    Used to compute the bias B = (1/N) Σ|h_i| for the Bias-Aware Transfer Rule.

    Args:
        hamiltonian: A ``pennylane.Hamiltonian`` or ``pennylane.ops.LinearCombination``
            with PauliZ and PauliZ@PauliZ terms.

    Returns:
        Tuple[dict, dict]:
            - J: dict mapping (i, j) → coefficient for two-body ZZ terms
            - h: dict mapping i → coefficient for single-body Z terms
    """
    J = {}
    h = {}

    coeffs = hamiltonian.coeffs
    ops = hamiltonian.ops

    for coeff, op in zip(coeffs, ops):
        coeff = float(coeff)
        if isinstance(op, qml.PauliZ):
            wire = op.wires[0]
            h[wire] = h.get(wire, 0.0) + coeff
        elif isinstance(op, qml.ops.Prod) or (
            hasattr(op, "operands") and len(op.operands) == 2
        ):
            # PauliZ @ PauliZ product
            operands = op.operands if hasattr(op, "operands") else [op]
            wires = [o.wires[0] for o in operands if isinstance(o, qml.PauliZ)]
            if len(wires) == 2:
                i, j = sorted(wires)
                J[(i, j)] = J.get((i, j), 0.0) + coeff
        elif hasattr(op, "base"):
            # SProd or similar — unwrap
            inner = op.base if hasattr(op, "base") else op
            if isinstance(inner, qml.PauliZ):
                wire = inner.wires[0]
                h[wire] = h.get(wire, 0.0) + coeff

    return J, h


def compute_bias(hamiltonian, num_qubits: int) -> float:
    """Compute bias B = (1/N) Σ|h_i| for a sub-problem Hamiltonian.

    B measures the strength of linear (single-qubit) terms relative to the
    quadratic coupling. Used in the Bias-Aware Transfer Rule: when |B_target -
    B_rep| < threshold, parameters transfer directly without retraining.

    Args:
        hamiltonian: PennyLane Hamiltonian.
        num_qubits: Total number of qubits N (denominator for normalisation).

    Returns:
        float: Normalised bias value B ∈ [0, ∞).
    """
    _, h = extract_coupling_matrix(hamiltonian)
    if num_qubits == 0:
        return 0.0
    return sum(abs(v) for v in h.values()) / num_qubits


# ---------------------------------------------------------------------------
# PennyLane Hamiltonian → MLIR attribute bridge
# ---------------------------------------------------------------------------

def hamiltonian_to_graph_attrs(hamiltonian, num_qubits: int, sparse_threshold: int = 64):
    """Convert a PennyLane Ising Hamiltonian to MLIR graph attribute strings.

    Maps the Hamiltonian to two MLIR attribute representations:

    - **H_quad** (``!quantum.dense_graph`` or ``!quantum.sparse_graph``):
      The quadratic J_ij coupling matrix. Dense encoding is used when
      ``num_qubits <= sparse_threshold``; sparse COO otherwise.
    - **H_lin** (``DenseElementsAttr`` string): The linear h_i bias vector
      as a rank-1 ``tensor<Nxf64>``.

    These strings can be embedded directly into MLIR textual IR, e.g. as
    attributes on a ``quantum.freeze_partition`` op.

    Args:
        hamiltonian: PennyLane Hamiltonian with PauliZ and PauliZ@PauliZ terms.
        num_qubits: Total number of qubits / graph nodes N.
        sparse_threshold: Node count above which sparse encoding is used
            (default 64).

    Returns:
        Tuple[str, str]:
            - h_quad_attr: MLIR attribute string for the J_ij coupling matrix.
            - h_lin_attr: MLIR attribute string for the h_i bias vector.

    Examples::

        H = qml.Hamiltonian([-0.5, -0.5], [qml.PauliZ(0)@qml.PauliZ(1),
                                            qml.PauliZ(1)@qml.PauliZ(2)])
        h_quad, h_lin = hamiltonian_to_graph_attrs(H, num_qubits=4)
        # h_quad → '#quantum.dense_graph<4, dense<...> : tensor<4x4xf64>>'
        # h_lin  → 'dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf64>'
    """
    import numpy as np

    J, h = extract_coupling_matrix(hamiltonian)
    N = num_qubits

    if N <= sparse_threshold:
        # Dense path: build full NxN weight matrix row-major
        W = np.zeros((N, N), dtype=float)
        for (i, j), val in J.items():
            W[i, j] = val
            W[j, i] = val  # symmetric

        # Format as nested MLIR dense list: [[r0c0, r0c1, ...], [r1c0, ...], ...]
        rows_str = ", ".join(
            "[" + ", ".join(f"{v:.6e}" for v in row) + "]"
            for row in W
        )
        h_quad_attr = (
            f"#quantum.dense_graph<{N}, "
            f"dense<[{rows_str}]> : tensor<{N}x{N}xf64>>"
        )
    else:
        # Sparse path: COO upper-triangle only
        edges = sorted((i, j, v) for (i, j), v in J.items() if abs(v) > 1e-12)
        num_edges = len(edges)
        rows_arr = ", ".join(str(i) for i, _, _ in edges)
        cols_arr = ", ".join(str(j) for _, j, _ in edges)
        wts_arr = ", ".join(f"{v:.6e}" for _, _, v in edges)
        h_quad_attr = (
            f"#quantum.sparse_graph<{N}, {num_edges}, "
            f"[{rows_arr}], [{cols_arr}], "
            f"dense<[{wts_arr}]> : tensor<{num_edges}xf64>>"
        )

    # H_lin: rank-1 f64 tensor of length N
    h_vec = [h.get(i, 0.0) for i in range(N)]
    h_vals = ", ".join(f"{v:.6e}" for v in h_vec)
    h_lin_attr = f"dense<[{h_vals}]> : tensor<{N}xf64>"

    return h_quad_attr, h_lin_attr


# ---------------------------------------------------------------------------
# DO-QAOA partition decorator
# ---------------------------------------------------------------------------

class DOQAOAPartitionCallable:
    """Wraps a QNode to apply DO-QAOA frozen-qubit partitioning.

    Records hotspot qubit indices and config so that the downstream
    compilation pipeline can inject ``quantum.freeze_partition`` into the IR.
    """

    def __init__(self, qnode, graph, config: DOQAOAConfig):
        self._qnode = qnode
        self._graph = graph
        self._config = config
        self._hotspot_indices = select_hotspot_indices(graph, config.m)
        functools.update_wrapper(self, qnode)

    @property
    def hotspot_indices(self):
        """List[int]: selected hotspot (frozen) qubit indices."""
        return self._hotspot_indices

    @property
    def config(self):
        """DOQAOAConfig: partition configuration."""
        return self._config

    @property
    def num_qubits(self):
        """int: total qubit count of the underlying QNode's device."""
        device = self._qnode.device
        if hasattr(device, "num_wires"):
            return device.num_wires
        if hasattr(device, "wires"):
            return len(device.wires)
        return len(self._graph.nodes)

    def __call__(self, *args, **kwargs):
        return self._qnode(*args, **kwargs)

    def __repr__(self):
        return (
            f"DOQAOAPartitionCallable(m={self._config.m}, "
            f"hotspots={self._hotspot_indices})"
        )


def doqaoa_partition(qnode=None, *, graph, config: DOQAOAConfig):
    """Decorator: annotate a QNode for DO-QAOA frozen-qubit partitioning.

    Selects the ``config.m`` hotspot qubits using degree centrality and
    records the partition metadata so that the Catalyst compilation pipeline
    can inject a ``quantum.freeze_partition`` op into the MLIR IR.

    Usage::

        config = DOQAOAConfig(m=2)
        G = nx.barabasi_albert_graph(10, 2)

        @doqaoa_partition(graph=G, config=config)
        @qml.qnode(dev)
        def circuit(params):
            qaoa_layer(params, G)
            return qml.expval(cost_h)

        # hotspot indices available before JIT compilation:
        print(circuit.hotspot_indices)

    Args:
        qnode: The QNode to wrap. Can be ``None`` when used as a
            decorator factory (``@doqaoa_partition(graph=G, config=cfg)``).
        graph: A NetworkX graph whose nodes are integers 0..N-1.
        config: DO-QAOA configuration (``DOQAOAConfig``).

    Returns:
        DOQAOAPartitionCallable: Wrapped callable preserving the original
        QNode interface while exposing ``hotspot_indices`` and ``config``.
    """
    if qnode is None:
        return functools.partial(doqaoa_partition, graph=graph, config=config)
    return DOQAOAPartitionCallable(qnode, graph, config)


# ---------------------------------------------------------------------------
# Minimal Adam optimizer (no external dependency)
# ---------------------------------------------------------------------------

class _AdamState:
    """Lightweight Adam state for a flat parameter vector."""

    __slots__ = ("m", "v", "t")

    def __init__(self, params):
        import numpy as np
        self.m = np.zeros_like(params)
        self.v = np.zeros_like(params)
        self.t = 0


def _adam_step(params, grads, state, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    """One in-place Adam step. Returns updated params and mutated state."""
    import numpy as np
    state.t += 1
    state.m = beta1 * state.m + (1.0 - beta1) * grads
    state.v = beta2 * state.v + (1.0 - beta2) * grads ** 2
    mhat = state.m / (1.0 - beta1 ** state.t)
    vhat = state.v / (1.0 - beta2 ** state.t)
    return params - lr * mhat / (np.sqrt(vhat) + eps)


# ---------------------------------------------------------------------------
# Task 4 — Autodiff-compatible parameter transfer (jax.grad + jit)
# ---------------------------------------------------------------------------

def _bias_transfer_jax(params_rep, params_ws, is_direct_copy: bool):
    """JAX-differentiable parameter transfer for the Bias-Aware Transfer Rule.

    Uses ``jax.lax.cond`` so **both** branches are traced by JAX (required for
    ``jax.jit`` / ``jax.grad`` compatibility) but only one executes at runtime.

    Gradient semantics:
    - Direct-copy branch: ``jax.lax.stop_gradient(params_rep)`` — gradient does
      NOT flow back through the representative parameters in this branch.  This
      matches the DO-QAOA design: the representative is frozen once trained.
    - Warm-start branch: ``params_ws`` — gradient flows normally through the
      fine-tuning steps (Adam on the sub-problem).

    Args:
        params_rep: Representative parameters θ* = (γ*, β*), shape ``(2,)``.
        params_ws: Warm-start parameters (may equal params_rep at init), shape ``(2,)``.
        is_direct_copy (bool): True → direct copy (stop_gradient), False → warm-start.

    Returns:
        Array of shape ``(2,)`` with gradient semantics above.
    """
    try:
        import jax
        import jax.numpy as jnp

        flag = jnp.bool_(is_direct_copy)
        return jax.lax.cond(
            flag,
            lambda p_r, p_w: jax.lax.stop_gradient(p_r),   # direct copy
            lambda p_r, p_w: p_w,                            # warm-start
            params_rep,
            params_ws,
        )
    except ImportError:
        import numpy as np
        return np.array(params_rep) if is_direct_copy else np.array(params_ws)


# ---------------------------------------------------------------------------
# Task 2 — Optimiser hooks (PennyLane Adam / GradientDescent)
# ---------------------------------------------------------------------------

class DOQAOAOptimizer:
    """PennyLane-compatible optimiser with DO-QAOA gradient isolation.

    Wraps ``pennylane.AdamOptimizer`` or ``pennylane.GradientDescentOptimizer``
    and ensures that gradient tapes are **only created for the representative
    sub-circuit** (phase 1).  Parameters for direct-copy sub-circuits (phase 3)
    are wrapped in ``jax.lax.stop_gradient`` / PennyLane ``no_grad`` so they
    contribute zero gradient, avoiding wasteful or incorrect backprop.

    Args:
        base_optimizer: A PennyLane optimizer instance, or one of the strings
            ``"adam"`` or ``"gd"`` to use the built-in defaults.
        learning_rate (float): Learning rate passed to the base optimizer when
            constructed from a string.  Ignored when ``base_optimizer`` is an
            existing instance.
        stop_gradient_on_copies (bool): If True (default), gradient tapes for
            direct-copy and warm-start initialisation points are blocked.
            Set to False only for debugging.

    Example::

        opt = DOQAOAOptimizer("adam", learning_rate=0.01)
        for step in range(100):
            params, energy = opt.step_and_cost(cost_fn, params, is_representative=True)
    """

    def __init__(self, base_optimizer="adam", learning_rate: float = 0.01,
                 stop_gradient_on_copies: bool = True):
        self._lr = learning_rate
        self._stop_grad = stop_gradient_on_copies

        if isinstance(base_optimizer, str):
            try:
                import pennylane as qml
                if base_optimizer == "adam":
                    self._opt = qml.AdamOptimizer(stepsize=learning_rate)
                elif base_optimizer == "gd":
                    self._opt = qml.GradientDescentOptimizer(stepsize=learning_rate)
                else:
                    raise ValueError(
                        f"Unknown optimizer string '{base_optimizer}'. "
                        "Use 'adam', 'gd', or pass a PennyLane optimizer instance."
                    )
            except ImportError:
                self._opt = None   # fallback: use internal Adam
        else:
            self._opt = base_optimizer

    def step(self, cost_fn, params, *, is_representative: bool = True, grad_fn=None):
        """One optimiser step with gradient isolation.

        Args:
            cost_fn: Callable ``(params) → scalar``.
            params: Current parameter array, shape ``(2,)``.
            is_representative (bool): If True, gradient flows normally (phase 1).
                If False and ``stop_gradient_on_copies`` is True, a
                ``stop_gradient``-guarded copy step is performed instead.
            grad_fn: Optional custom gradient function ``(params) → grad_array``.
                If None, finite-difference is used as fallback.

        Returns:
            numpy.ndarray: Updated params after one step.
        """
        import numpy as np

        if not is_representative and self._stop_grad:
            # Direct copy / warm-start init: no gradient, return params unchanged.
            try:
                import jax
                return np.array(jax.lax.stop_gradient(params))
            except ImportError:
                return params.copy()

        # Representative or warm-start fine-tuning: compute gradient.
        if self._opt is not None:
            try:
                new_params, _ = self._opt.step_and_cost(cost_fn, params)
                return np.array(new_params)
            except Exception:
                pass  # fallback below

        # Fallback: finite-difference gradient + internal Adam
        if grad_fn is not None:
            grads = grad_fn(params)
        else:
            h = 1e-4
            grads = np.zeros_like(params)
            for i in range(len(params)):
                p_f = params.copy(); p_f[i] += h
                p_b = params.copy(); p_b[i] -= h
                grads[i] = (cost_fn(p_f) - cost_fn(p_b)) / (2 * h)

        if not hasattr(self, "_adam_state"):
            self._adam_state = _AdamState(params.copy())
        return _adam_step(params, grads, self._adam_state, lr=self._lr)

    def step_and_cost(self, cost_fn, params, *, is_representative: bool = True, grad_fn=None):
        """One optimiser step, returning ``(new_params, cost)``."""
        new_params = self.step(cost_fn, params, is_representative=is_representative,
                               grad_fn=grad_fn)
        cost = float(cost_fn(new_params))
        return new_params, cost

    def reset(self):
        """Reset internal Adam state (call between sub-problems)."""
        if hasattr(self, "_adam_state"):
            del self._adam_state
        if self._opt is not None and hasattr(self._opt, "reset"):
            self._opt.reset()


# ---------------------------------------------------------------------------
# Three-phase execution engine
# ---------------------------------------------------------------------------

class DOQAOAExecutor:
    """Execute the DO-QAOA three-phase training schedule via JAX.

    Gradient flows only through the representative sub-circuit (phase 1).
    Warm-start sub-circuits (phase 2) receive ``jax.lax.stop_gradient``-wrapped
    initialisation from the representative, then fine-tune independently.
    Direct-copy sub-circuits (phase 3) are fully ``stop_gradient``-guarded.

    Args:
        partition_callable: A ``DOQAOAPartitionCallable`` produced by
            :func:`doqaoa_partition`.
        full_epochs: Adam steps for phase 1 representative optimisation.
        warmstart_epochs: Adam steps for phase 2 warm-start fine-tuning.
            Falls back to ``partition_callable.config.warmstart_epochs``.
        learning_rate: Adam learning rate for all phases.
        grad_norm_tol: Convergence threshold on ||∇θ||₂ (phase 2 early stop).
        seed: RNG seed for random-init strategy.
    """

    def __init__(
        self,
        partition_callable: "DOQAOAPartitionCallable",
        *,
        full_epochs: int = 100,
        warmstart_epochs: Optional[int] = None,
        learning_rate: float = 0.01,
        grad_norm_tol: float = 1e-4,
        seed: int = 42,
    ):
        if not isinstance(partition_callable, DOQAOAPartitionCallable):
            raise TypeError(
                "doqaoa_qjit expects a DOQAOAPartitionCallable; "
                f"got {type(partition_callable).__name__}"
            )
        self._partition = partition_callable
        self._full_epochs = full_epochs
        self._warmstart_epochs = (
            warmstart_epochs
            if warmstart_epochs is not None
            else partition_callable.config.warmstart_epochs
        )
        self._lr = learning_rate
        self._grad_norm_tol = grad_norm_tol
        self._seed = seed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _jit_qnode(self):
        """Return the QNode wrapped with Catalyst qjit (or JAX jit fallback)."""
        qnode = self._partition._qnode
        try:
            from catalyst import qjit as catalyst_qjit
            return catalyst_qjit(qnode)
        except Exception:
            try:
                import jax
                return jax.jit(qnode)
            except Exception:
                return qnode

    def _initial_params(self):
        """Return (gamma_init, beta_init) as a numpy array of shape (2,)."""
        import numpy as np
        cfg = self._partition.config
        if cfg.init_strategy == "shortcut":
            return np.array([-math.pi / 6.0, -math.pi / 8.0])
        rng = np.random.default_rng(self._seed)
        return rng.uniform(-math.pi, math.pi, size=2)

    def _finite_diff_grad(self, fn, params, k_idx, h=1e-4):
        """Finite-difference gradient of fn(params, k_idx) w.r.t. params."""
        import numpy as np
        grad = np.zeros_like(params)
        for i in range(len(params)):
            p_fwd = params.copy(); p_fwd[i] += h
            p_bwd = params.copy(); p_bwd[i] -= h
            grad[i] = (fn(p_fwd, k_idx) - fn(p_bwd, k_idx)) / (2.0 * h)
        return grad

    def _transfer_mode(self, bias_shift: float) -> int:
        """Map bias shift to transfer mode: 0=rep, 1=copy, 2=warmstart."""
        return 1 if bias_shift < self._partition.config.bias_threshold else 2

    def _sub_problem_bias_shift(self, k: int) -> float:
        """Compute ΔB_k = |B_k − B_rep| for sub-problem k using the graph.

        Uses the Hamiltonian attached to the partition's QNode (if available),
        falling back to 0.0 (→ direct copy) when no Hamiltonian is found.
        """
        try:
            H = self._partition._qnode.hamiltonian
        except AttributeError:
            return 0.0
        N = self._partition.num_qubits
        cfg = self._partition.config
        hotspots = self._partition.hotspot_indices

        def _bias_for_k(k_idx: int) -> float:
            _, h_coeff = extract_coupling_matrix(H)
            J_coeff, _ = extract_coupling_matrix(H)
            # Effective linear bias for free qubits with frozen spins from k_idx
            total = 0.0
            free_count = 0
            for qubit in range(N):
                if qubit in hotspots:
                    continue
                bit_pos = hotspots.index(qubit) if qubit in hotspots else -1
                h_eff = h_coeff.get(qubit, 0.0)
                for fi, frozen_q in enumerate(hotspots):
                    spin = -1.0 if (k_idx >> fi) & 1 else 1.0
                    pair = (min(qubit, frozen_q), max(qubit, frozen_q))
                    h_eff += J_coeff.get(pair, 0.0) * spin
                total += abs(h_eff)
                free_count += 1
            return total / max(free_count, 1)

        b_rep = _bias_for_k(0)
        b_k = _bias_for_k(k)
        return abs(b_k - b_rep)

    # ------------------------------------------------------------------
    # Main execution: three-phase schedule
    # ------------------------------------------------------------------

    def run(self, qnode_fn=None):
        """Execute the three-phase DO-QAOA schedule.

        Returns:
            Tuple[int, float, numpy.ndarray]:
                (best_k, best_energy, best_params) — sub-problem index,
                minimum ⟨H⟩ value, and the corresponding (γ, β) parameters.

        Post-run attributes (for :class:`DOQAOAResult`):
            _shot_counter (int): total energy evaluations used.
            _warmstart_count (int): warm-start sub-problems trained.
            _direct_copy_count (int): sub-problems using direct copy.
        """
        import numpy as np

        cfg = self._partition.config
        m = cfg.m
        num_sp = 1 << m
        max_ws = getattr(cfg, "max_warmstarts", 1)
        compiled = self._jit_qnode() if qnode_fn is None else qnode_fn

        # Shot counter (one call = one shot)
        self._shot_counter = 0
        self._warmstart_count = 0
        self._direct_copy_count = 0

        # Detect whether qnode_fn is a multi-k energy function (params, k_idx) → float
        # or a single-k compiled QNode (params,) → float.
        import inspect as _inspect
        try:
            _sig = _inspect.signature(compiled)
            _multi_k = len(_sig.parameters) >= 2
        except (ValueError, TypeError):
            _multi_k = False

        def energy(params, k_idx=0):
            self._shot_counter += 1
            try:
                import jax.numpy as jnp
                p = jnp.array(params)
            except ImportError:
                p = params
            if _multi_k:
                return float(compiled(p, k_idx))
            return float(compiled(p))

        def fd_grad(params, k_idx=0):
            """Central finite-difference gradient (4 shots per call)."""
            import numpy as np
            h = 1e-4
            grad = np.zeros_like(params)
            for i in range(len(params)):
                p_f = params.copy(); p_f[i] += h
                p_b = params.copy(); p_b[i] -= h
                grad[i] = (energy(p_f, k_idx) - energy(p_b, k_idx)) / (2 * h)
            return grad

        # ── Phase 1: Full optimisation of representative (k=0) ────────────
        theta_rep = self._initial_params()
        state_rep = _AdamState(theta_rep.copy())

        for ep in range(1, self._full_epochs + 1):
            grads = fd_grad(theta_rep, k_idx=0)
            theta_rep = _adam_step(theta_rep, grads, state_rep, lr=self._lr)
            energy(theta_rep, 0)                 # convergence check shot
            if np.linalg.norm(grads) < self._grad_norm_tol:
                break

        e_rep = energy(theta_rep, 0)

        # ── Build per-sub-problem results ─────────────────────────────────
        results: dict[int, Tuple[float, np.ndarray]] = {0: (e_rep, theta_rep.copy())}

        for k in range(1, num_sp):
            delta_b = self._sub_problem_bias_shift(k)
            mode = self._transfer_mode(delta_b)

            # Respect max_warmstarts cap (DO-QAOA paper: ≤ 1 warm-start)
            if mode == 2 and self._warmstart_count >= max_ws:
                mode = 1   # force direct copy once cap reached

            if mode == 1:
                # Phase 3: direct copy — jax.lax.cond stop_gradient semantics
                try:
                    import jax.numpy as jnp
                    theta_ws = jnp.array(theta_rep)
                except ImportError:
                    theta_ws = theta_rep.copy()
                theta_k = np.array(_bias_transfer_jax(theta_rep, theta_ws, True))
                results[k] = (energy(theta_k, k), theta_k)
                self._direct_copy_count += 1

            else:
                # Phase 2: warm-start — init is stop_gradient, fine-tune has grad
                try:
                    import jax.numpy as jnp
                    theta_ws = jnp.array(theta_rep)
                except ImportError:
                    theta_ws = theta_rep.copy()
                theta_k = np.array(_bias_transfer_jax(theta_rep, theta_ws, False))

                state_k = _AdamState(theta_k.copy())
                for ep in range(1, self._warmstart_epochs + 1):
                    grads = fd_grad(theta_k, k_idx=k)
                    theta_k = _adam_step(theta_k, grads, state_k, lr=self._lr)
                    energy(theta_k, k)           # convergence check shot
                    if np.linalg.norm(grads) < self._grad_norm_tol:
                        break
                results[k] = (energy(theta_k, k), theta_k)
                self._warmstart_count += 1

        # ── Pick minimum-energy sub-problem ───────────────────────────────
        best_k = min(results, key=lambda k: results[k][0])
        best_energy, best_params = results[best_k]
        return best_k, best_energy, best_params


# ---------------------------------------------------------------------------
# Public decorator: doqaoa_qjit
# ---------------------------------------------------------------------------

def doqaoa_qjit(
    partition_callable=None,
    *,
    full_epochs: int = 100,
    warmstart_epochs: Optional[int] = None,
    learning_rate: float = 0.01,
    grad_norm_tol: float = 1e-4,
    seed: int = 42,
):
    """JIT-compile a DO-QAOA partition with Catalyst and gradient isolation.

    Hooks the rewritten training schedule into Catalyst's ``@qjit`` tracing.
    JAX gradient computation flows **only** through the representative
    sub-circuit's parameters (phase 1).  Direct-copy parameters (phase 3) are
    wrapped with ``jax.lax.stop_gradient``; warm-start initialisation points
    (phase 2) are also stop-gradient-guarded so only the fine-tuning steps
    contribute gradients.

    Usage::

        config = DOQAOAConfig(m=2, warmstart_epochs=10)
        G = nx.cycle_graph(8)

        @doqaoa_qjit(full_epochs=100, warmstart_epochs=10)
        @doqaoa_partition(graph=G, config=config)
        @qml.qnode(dev)
        def circuit(params):
            qaoa_layer(params, G)
            return qml.expval(cost_h)

        best_k, best_energy, best_params = circuit()

    Args:
        partition_callable: A ``DOQAOAPartitionCallable`` (i.e. already
            decorated with ``@doqaoa_partition``).  Can be ``None`` when used
            as a decorator factory.
        full_epochs: Adam steps for the representative sub-circuit (phase 1).
        warmstart_epochs: Adam steps for warm-start sub-circuits (phase 2).
            Defaults to ``partition_callable.config.warmstart_epochs``.
        learning_rate: Adam learning rate (all phases).
        grad_norm_tol: Early-stop threshold on ||∇θ||₂.
        seed: RNG seed when ``init_strategy="random"``.

    Returns:
        A callable whose ``()`` call executes the three-phase schedule and
        returns ``(best_k, best_energy, best_params)``.
    """
    def _wrap(pc):
        executor = DOQAOAExecutor(
            pc,
            full_epochs=full_epochs,
            warmstart_epochs=warmstart_epochs,
            learning_rate=learning_rate,
            grad_norm_tol=grad_norm_tol,
            seed=seed,
        )
        @functools.wraps(pc)
        def _runner(*args, **kwargs):
            return executor.run()
        _runner.executor = executor
        _runner.partition = pc
        return _runner

    if partition_callable is None:
        return _wrap
    return _wrap(partition_callable)


# ---------------------------------------------------------------------------
# Task 1 — catalyst.do_qaoa() high-level transform
# ---------------------------------------------------------------------------
# Internal: multi-k energy bridge
# ---------------------------------------------------------------------------

def _build_multi_k_energy(hamiltonian, hotspot_indices, num_qubits):
    """Build a (params, k_idx) → float energy function from a PennyLane Hamiltonian.

    Parses the Ising Hamiltonian into J (quadratic) and h (linear) arrays,
    then uses the closed-form p=1 QAOA energy formula for each frozen-qubit
    sub-problem k.  This is the bridge between the Python Hamiltonian object
    and the DO-QAOA sub-problem evaluations — no quantum device needed.

    Args:
        hamiltonian: PennyLane Ising Hamiltonian (PauliZ / PauliZ@PauliZ terms).
        hotspot_indices: List[int] of frozen qubit indices.
        num_qubits: Total number of qubits N.

    Returns:
        Callable[[np.ndarray, int], float]: energy(params, k_idx) → ⟨H_k⟩
    """
    import numpy as np

    J_dict, h_dict = extract_coupling_matrix(hamiltonian)
    N = num_qubits
    hs = list(hotspot_indices)
    m  = len(hs)

    # Build dense N×N J matrix and N-vector h
    J_mat = np.zeros((N, N))
    for (i, j), v in J_dict.items():
        J_mat[i, j] = v; J_mat[j, i] = v
    h_vec = np.array([h_dict.get(i, 0.0) for i in range(N)])

    def _energy_cf(J_free, h_eff, gamma, beta, nf):
        E = 0.0
        for u in range(nf):
            for v in range(u + 1, nf):
                Juv = J_free[u, v]
                if abs(Juv) > 1e-12:
                    E += Juv / 2.0 * math.sin(4 * beta) * math.sin(2 * gamma * Juv)
        for u in range(nf):
            hu = h_eff[u]
            if abs(hu) > 1e-12:
                E += hu * (-math.sin(2 * beta) * math.cos(2 * gamma * hu))
        return E

    def multi_k_energy(params, k_idx=0):
        gamma, beta = float(params[0]), float(params[1])
        # Build frozen spin assignment for sub-problem k_idx
        frozen = {hs[i]: (-1 if (k_idx >> i) & 1 else 1) for i in range(m)}
        free   = [q for q in range(N) if q not in frozen]
        nf     = len(free)
        # Effective linear bias: h_eff[q] = h[q] + Σ_{f frozen} J[q,f] * spin_f
        h_eff  = np.array([
            h_vec[q] + sum(J_mat[q, fq] * frozen[fq] for fq in frozen)
            for q in free
        ])
        # Free-free coupling sub-matrix
        J_free = np.array([[J_mat[free[i], free[j]] for j in range(nf)]
                           for i in range(nf)])
        return _energy_cf(J_free, h_eff, gamma, beta, nf)

    return multi_k_energy


# ---------------------------------------------------------------------------

class DOQAOATransform:
    """Returned by :func:`do_qaoa`. Call with a NetworkX graph to run the pipeline.

    This object is compatible with ``@qjit``: it can be passed as the callable
    to ``qjit(DOQAOATransform_instance)`` for AOT compilation of the full
    three-phase schedule.

    Args:
        qnode: Original PennyLane QNode.
        hamiltonian: PennyLane Hamiltonian (Ising form) used to compute bias shifts.
        config: :class:`DOQAOAConfig` controlling all hyper-parameters.
        executor_kwargs: Keyword arguments forwarded to :class:`DOQAOAExecutor`.
    """

    def __init__(self, qnode, hamiltonian, config: DOQAOAConfig, executor_kwargs: dict):
        self._qnode = qnode
        self._hamiltonian = hamiltonian
        self._config = config
        self._executor_kwargs = executor_kwargs

    # ------------------------------------------------------------------

    def __call__(self, graph, *, frozen_qubits_shots: Optional[int] = None):
        """Run the full DO-QAOA pipeline on ``graph``.

        Args:
            graph: A NetworkX graph whose nodes are integers 0..N-1.
            frozen_qubits_shots (int | None): Baseline shot count from FrozenQubits
                on the same graph/m, used to compute :attr:`DOQAOAResult.speedup_vs_frozen`.
                If None, speedup is reported as 0.

        Returns:
            DOQAOAResult: Optimisation result with energy, bitstring, and shot count.

        Example::

            G = nx.barabasi_albert_graph(12, 2, seed=42)
            result = transform(G, frozen_qubits_shots=32_770_000)
            print(result)  # DOQAOAResult(best_k=0, ⟨H⟩=-8.12, ...)
        """
        # 1. Wrap QNode with partition metadata (selects hotspot qubits)
        partition = doqaoa_partition(self._qnode, graph=graph, config=self._config)

        # 2. Build a multi-k energy function (params, k_idx) → float from the
        #    Hamiltonian so each sub-problem is evaluated on its own frozen circuit.
        #    This is the key wiring that makes do_qaoa() correct for all 2^m sub-problems.
        multi_k_fn = _build_multi_k_energy(
            self._hamiltonian, partition.hotspot_indices, partition.num_qubits
        )

        # 3. Create executor and run three-phase schedule
        executor = DOQAOAExecutor(partition, **self._executor_kwargs)
        best_k, best_energy, best_params = executor.run(qnode_fn=multi_k_fn)

        # 3. Build bitstring from best_k
        m = self._config.m
        bitstring = [(best_k >> i) & 1 for i in range(m)]

        # 4. Speedup ratio
        total_shots = executor._shot_counter
        speedup = (frozen_qubits_shots / total_shots
                   if frozen_qubits_shots and total_shots > 0 else 0.0)

        return DOQAOAResult(
            best_k=best_k,
            best_energy=best_energy,
            best_params=best_params,
            bitstring=bitstring,
            total_shots=total_shots,
            warmstart_count=executor._warmstart_count,
            direct_copy_count=executor._direct_copy_count,
            full_opt_count=1,
            speedup_vs_frozen=speedup,
        )

    @property
    def config(self) -> DOQAOAConfig:
        """DOQAOAConfig: hyper-parameter configuration."""
        return self._config

    def __repr__(self):
        return (
            f"DOQAOATransform(m={self._config.m}, "
            f"threshold={self._config.bias_threshold}, "
            f"gradient_fn={self._config.gradient_fn!r})"
        )


def do_qaoa(
    qnode,
    hamiltonian,
    *,
    m: int,
    config: Optional[DOQAOAConfig] = None,
    gradient_fn: str = "adam",
    full_epochs: int = 100,
    warmstart_epochs: Optional[int] = None,
    learning_rate: float = 0.01,
    grad_norm_tol: float = 1e-4,
    seed: int = 42,
    max_warmstarts: int = 1,
    bias_threshold: float = 0.3,
    init_strategy: str = "shortcut",
):
    """High-level DO-QAOA transform — wraps a QNode with the full pipeline.

    Implements Algorithm 1 of Sang et al., arXiv:2602.21689v1 (2026):

    1. Select ``m`` hotspot qubits by degree centrality.
    2. Produce 2^m frozen-qubit sub-problems.
    3. Cluster energy landscapes → K ≈ 1 representative(s).
    4. Train representative(s) with Adam (``full_epochs`` steps).
    5. Transfer parameters to remaining sub-problems via the Bias-Aware
       Transfer Rule (direct copy when |ΔB| < ``bias_threshold``, warm-start
       otherwise, capped at ``max_warmstarts``).
    6. Return the sub-problem with minimum ⟨H⟩.

    Compatible with ``@qjit``.

    Args:
        qnode: A ``pennylane.QNode`` computing the QAOA cost expectation value.
        hamiltonian: The PennyLane Ising Hamiltonian (``PauliZ`` and
            ``PauliZ @ PauliZ`` terms) defining the MaxCut / optimisation problem.
        m (int): Number of hotspot (frozen) qubits. Must be ≥ 1.
        config (DOQAOAConfig | None): Full configuration object.  When supplied,
            all other keyword arguments (except ``qnode`` and ``hamiltonian``) are
            ignored.  Build a config explicitly when you need non-default values for
            ``k_max`` or ``landscape_grid_size``.
        gradient_fn (str): Optimiser — ``"adam"`` (default) or ``"gd"``.
        full_epochs (int): Adam steps for the representative sub-circuit (phase 1).
            Default 100.
        warmstart_epochs (int | None): Adam steps for warm-start sub-circuits
            (phase 2). Defaults to ``DOQAOAConfig.warmstart_epochs`` (10).
        learning_rate (float): Adam / GD learning rate. Default 0.01.
        grad_norm_tol (float): Early-stop threshold ``||∇θ||₂``. Default 1e-4.
        seed (int): RNG seed when ``init_strategy="random"``. Default 42.
        max_warmstarts (int): Hard cap on warm-start sub-problems. Default 1
            (matches DO-QAOA paper Table IV budget).
        bias_threshold (float): ΔB threshold for direct copy vs warm-start.
            Default 0.3 (Section 2.2 of arXiv:2602.21689v1).
        init_strategy (str): ``"shortcut"`` (analytic p=1 values) or ``"random"``.

    Returns:
        DOQAOATransform: A callable with signature ``(graph, *, frozen_qubits_shots=None)
        → DOQAOAResult``.  Call it with a NetworkX graph to run the optimisation.

    Example::

        import pennylane as qml
        import networkx as nx
        import catalyst

        G = nx.barabasi_albert_graph(12, 2, seed=42)
        cost_h, mixer_h = qml.qaoa.maxcut(G)
        dev = qml.device("lightning.qubit", wires=12)

        @qml.qnode(dev)
        def circuit(params):
            qml.qaoa.cost_layer(params[0], cost_h)
            qml.qaoa.mixer_layer(params[1], mixer_h)
            return qml.expval(cost_h)

        result = catalyst.do_qaoa(circuit, cost_h, m=3)(G)
        print(result)
        # DOQAOAResult(best_k=0, ⟨H⟩=-8.12, bitstring=000, shots=712, speedup=46,026×)

    See Also:
        :class:`DOQAOAConfig`, :class:`DOQAOAResult`, :class:`DOQAOAOptimizer`
    """
    if config is None:
        config = DOQAOAConfig(
            m=m,
            bias_threshold=bias_threshold,
            warmstart_epochs=warmstart_epochs if warmstart_epochs is not None else 10,
            init_strategy=init_strategy,
            gradient_fn=gradient_fn,
            max_warmstarts=max_warmstarts,
        )

    executor_kwargs = dict(
        full_epochs=full_epochs,
        warmstart_epochs=warmstart_epochs,  # None → uses config.warmstart_epochs
        learning_rate=learning_rate,
        grad_norm_tol=grad_norm_tol,
        seed=seed,
    )

    return DOQAOATransform(qnode, hamiltonian, config, executor_kwargs)
