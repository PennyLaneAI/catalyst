DO-QAOA API Reference
=====================

.. currentmodule:: catalyst

Overview
--------

DO-QAOA (Doubly Optimized QAOA) reduces O(2\ :sup:`m`) independent variational
training sessions to O(K≈1) by exploiting **landscape similarity** across
frozen-qubit sub-problems.  The Catalyst implementation follows Algorithm 1 of
Sang et al., `arXiv:2602.21689v1 <https://arxiv.org/abs/2602.21689>`_ (2026).

**Key reduction (Table IV):**

+---------------------+------------------+
| Method              | Shots            |
+=====================+==================+
| FrozenQubits        | 32.77 × 10⁶      |
+---------------------+------------------+
| DO-QAOA (m=2)       | ≤ 0.13 × 10⁶    |
+---------------------+------------------+
| DO-QAOA (m=3)       | ≤ 0.23 × 10⁶    |
+---------------------+------------------+


Quick Start
-----------

.. code-block:: python

    import pennylane as qml
    import networkx as nx
    import catalyst

    G = nx.barabasi_albert_graph(12, 2, seed=42)
    cost_h, mixer_h = qml.qaoa.maxcut(G)
    dev = qml.device("lightning.qubit", wires=12)

    @qml.qnode(dev)
    def circuit(params):
        for w in range(12):
            qml.Hadamard(wires=w)
        qml.qaoa.cost_layer(params[0], cost_h)
        qml.qaoa.mixer_layer(params[1], mixer_h)
        return qml.expval(cost_h)

    # One call — full DO-QAOA pipeline
    result = catalyst.do_qaoa(circuit, cost_h, m=3)(G)
    print(result)
    # DOQAOAResult(best_k=0, ⟨H⟩=-8.123, bitstring=000, shots=712, speedup=46026×)


Migration Guide: FrozenQubits → DO-QAOA
-----------------------------------------

.. list-table::
   :header-rows: 1

   * - FrozenQubits (before)
     - DO-QAOA (after)
   * - ``for k in range(2**m): qjit(circuit)(params_k)``
     - ``catalyst.do_qaoa(circuit, H, m=m)(G)``
   * - 2^m full training sessions
     - 1 full + ≤1 warm-start
   * - 32.77 × 10⁶ shots (m=2)
     - ≤ 0.13 × 10⁶ shots (m=2)
   * - Manual parameter management
     - Automatic via ``DOQAOAResult``


Public API
----------

.. autofunction:: catalyst.do_qaoa

.. autoclass:: catalyst.DOQAOAConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: catalyst.DOQAOAResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: catalyst.DOQAOAOptimizer
   :members: step, step_and_cost, reset
   :undoc-members:
   :show-inheritance:

.. autoclass:: catalyst.DOQAOATransform
   :members: __call__, config
   :show-inheritance:

.. autofunction:: catalyst.doqaoa_partition

.. autofunction:: catalyst.doqaoa_qjit

.. autofunction:: catalyst.hamiltonian_to_graph_attrs


MLIR Pass Pipeline
------------------

The ``catalyst.do_qaoa()`` transform triggers the following MLIR pass sequence
when compiled with ``@qjit``:

.. code-block:: text

    doqaoa-landscape-overlap       Phase 2, Task 1 — landscape similarity S(k,k')
    doqaoa-bias-shift              Phase 2, Task 2 — bias shift ΔB computation
    doqaoa-representative-selection Phase 3, Task 1 — mode assignment (0/1/2)
    doqaoa-training-schedule       Phase 3, Task 2 — 3-phase schedule attributes
    doqaoa-shared-buffer           Phase 3, Task 4 — thread-safe θ* LLVM global
    doqaoa-warmstart-scheduler     Phase 3, Task 5 — compile-time Adam (mode-2)
    doqaoa-direct-transfer         Phase 3, Task 6 — bias_transfer annotation
    doqaoa-aggregate-min           Phase 3, Task 7 — compile-time argmin
    doqaoa-noise-preserve          Phase 3, Task 8 — FakeBrisbane noise model
    doqaoa-depth-check             Phase 3, Task 9 — CNOT count regression

Each pass annotates ``quantum.freeze_partition`` ops with compile-time metadata.
The final LLVM lowering reads these attributes to emit the optimised binary.


Three-Phase Algorithm
---------------------

.. rubric:: Phase 1 — Representative Optimisation

Run ``full_epochs`` Adam steps on the cluster representative (k=0):

.. math::

    \theta^* = \arg\min_{\theta} \langle H_0(\theta) \rangle

Gradient computation uses central finite differences with ``h=1e-4``.

.. rubric:: Phase 2 — Warm-Start Transfer

For non-representative sub-problems with large bias shift
(|ΔB| ≥ ``bias_threshold``): run ``warmstart_epochs`` Adam steps starting from θ*,
capped at ``max_warmstarts=1``.

.. rubric:: Phase 3 — Direct Copy

For sub-problems with small bias shift (|ΔB| < ``bias_threshold``): copy θ* directly
with zero training cost.  Implemented via :func:`jax.lax.stop_gradient` to prevent
spurious gradient flow.


Bias-Aware Transfer Rule
------------------------

The transfer decision for sub-problem k is:

.. math::

    \text{mode}(k) = \begin{cases}
        0 & k = \text{cluster representative} \\
        1 & |B_k - B_0| < \theta_{\text{threshold}} \quad \text{(direct copy)} \\
        2 & \text{otherwise} \quad \text{(warm-start)}
    \end{cases}

where :math:`B_k = \frac{1}{N_f} \sum_{q \in \text{free}} |h^\text{eff}_q|` is the
normalised effective linear bias for sub-problem k (Eq. 7 of arXiv:2602.21689v1).


Autodiff Compatibility
----------------------

The parameter transfer function :func:`_bias_transfer_jax` uses ``jax.lax.cond``
for JAX-tracing compatibility:

.. code-block:: python

    import jax
    from catalyst.api_extensions.doqaoa import _bias_transfer_jax

    # Warm-start: gradient flows through params_ws
    grad_ws = jax.grad(lambda p: jnp.sum(_bias_transfer_jax(rep, p, False)))(ws)

    # Direct copy: gradient is zero (stop_gradient on rep params)
    grad_dc = jax.grad(lambda p: jnp.sum(_bias_transfer_jax(p, ws, True)))(rep)
    # grad_dc == [0., 0.]


Tutorial
--------

See :doc:`../../../tutorials/do_qaoa_tutorial` for a step-by-step walkthrough on a
12-node power-law graph demonstrating the 280× shot reduction.
