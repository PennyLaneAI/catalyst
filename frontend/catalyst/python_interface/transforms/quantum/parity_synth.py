# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains the implementation of the ``ParitySynth`` compiler pass. Given that it
operates on the phase polynomial representation of subcircuits, the implementation splits into
an xDSL-agnostic synthesis functionality and an integration thereof into xDSL."""

from dataclasses import dataclass
from itertools import product

try:
    import networkx as nx

    has_networkx = True
except ModuleNotFoundError as networkx_import_error:
    has_networkx = False

import numpy as np
from pennylane.transforms.intermediate_reps.rowcol import _rowcol_parity_matrix
from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import arith, builtin
from xdsl.ir import Operation, SSAValue
from xdsl.rewriter import InsertPoint

from catalyst.python_interface.dialects.quantum import CustomOp, InsertOp, QubitType
from catalyst.python_interface.pass_api import compiler_transform

### xDSL-agnostic part


def _apply_dfs_po_circuit(tree, source, P, inv_synth_matrix=None):
    """Traverse a tree in post-ordering (PO) depth-first search (DFS) ordering and record
    the corresponding edges into a CNOT circuit. Update the provided parity table with those CNOTs,
    and if an inverse synthesis matrix is provided, the CNOTs are applied to that matrix as well.

    Args:
        tree (nx.Graph): Tree graph to traverse. Note that this is a generic graph without
            tree-specific attributes/information.
        source (int): Root node of the tree. Only with this information can the ``tree`` be
            identified as a tree graph.
        P (np.ndarray): Parity table to which to apply the updates corresponding to the CNOTs.
        inv_synth_matrix (np.ndarray): (Inverse) parity matrix to which to apply the updates
            corresponding to the CNOTs as well. If ``None``, no action is performed.

    Returns:
        list[tuple[int]]: List of tuples corresponding to the depth-first post-order
        traversal. Each tuple contains two integers, corresponding to two nodes in ``tree``, or
        equivalently two qubit labels.
    """
    dfs_po = list(nx.dfs_postorder_nodes(tree, source=source))
    sub_circuit = []
    if inv_synth_matrix is None:
        for i, j in zip(dfs_po[:-1], dfs_po[1:]):
            sub_circuit.append((i, j))
            P[i] += P[j]
    else:
        for i, j in zip(dfs_po[:-1], dfs_po[1:]):
            sub_circuit.append((i, j))
            P[i] += P[j]
            inv_synth_matrix[:, i] += inv_synth_matrix[:, j]
    P %= 2
    return sub_circuit


def _loop_body_parity_network_synth(
    P: np.ndarray,
    inv_synth_matrix: np.ndarray,
    circuit: list[int, list[tuple[int]]],
) -> tuple[np.ndarray, list]:
    """Loop body function for ``_parity_network_synth``, the main subroutine of ``parity_synth``.
    The loop body corresponds to synthesizing one parity in the parity table ``P``, and updating
    all relevant data accordingly. It is the ``while``-loop body in Algorithm 1
    in https://arxiv.org/abs/2104.00934.

    Args:
        P (np.ndarray): (Remaining) parity table for which to synthesize the parity network.
        inv_synth_matrix (np.ndarray): Inverse of the parity _matrix_ implemented within
            the parity network that has been synthesized so far.
        circuit (list[int, list[tuple[int]]]): Circuit for the parity network that has been
            synthesized so far. Each entry of the list consists of a _relative_ index into
            the list of parities (or rotation angles) of the phase polynomial, a qubit
            index onto which the rotation should be applied, and the subcircuit that should
            be applied _before_ the rotation to achieve the respective parity.

    Returns:
        tuple[np.ndarray, list]: Same as inputs, with updates applied; ``P`` has a column less
        and has been transformed in addition. ``inv_synth_matrix`` has been transformed
        according to the newly synthesized subcircuit implementing the next parity. The
        ``circuit`` representation is grown by one entry, corresponding to that parity.

    """
    parity_idx = np.argmin(np.sum(P, axis=0))  # ┬ Line 3
    parity = P[:, parity_idx]  #                 ╯
    graph_nodes = list(map(int, np.where(parity)[0]))  # Line 5, vertices
    if len(graph_nodes) == 1:
        # The parity already has Hamming weight 1, so we don't need any modifications
        # Just slice out the parity and append the parity/angle index as well as the qubit
        # on which the parity has support
        P = np.concatenate([P[:, :parity_idx], P[:, parity_idx + 1 :]], axis=1)  # Line 4
        circuit.append((parity_idx, graph_nodes[0], []))  # Record parity index, qubit index, CNOTs
        return P, inv_synth_matrix, circuit

    # Note that there is a bug in the algorithm as written in the paper: We first want to compute
    # the edge weights for parity_graph (G_y) and _then_ slice out `parity` from `P`.
    single_weights = np.sum(P, axis=1)  #                                 ╮
    parity_graph = nx.DiGraph()  #                                        │
    parity_graph.add_weighted_edges_from(  #                              │
        [  #                                                              │
            (i, j, np.sum(np.mod(P[i] + P[j], 2)) - single_weights[j])  # ├ Line 5, edges
            for i, j in product(graph_nodes, repeat=2)  #                 │
            if i != j  #                                                  │
        ]  #                                                              │
    )  #                                                                  ╯
    arbor = nx.minimum_spanning_arborescence(parity_graph)  # Line 6

    # Find the root of the tree
    root = next(iter(node for node, degree in arbor.in_degree() if degree == 0))

    P = np.concatenate([P[:, :parity_idx], P[:, parity_idx + 1 :]], axis=1)  # Line 4
    # Lines 7-10, update P and inv_synth_matrix in place
    sub_circuit = _apply_dfs_po_circuit(arbor, root, P, inv_synth_matrix)
    circuit.append((parity_idx, root, sub_circuit))  # Record parity index, qubit index, CNOTs
    return P, inv_synth_matrix, circuit


def _parity_network_synth(P: np.ndarray) -> list[int, list[tuple[int]]]:
    """Main subroutine for the ``ParitySynth`` pass, mostly a ``for``-loop wrapper around
    ``_loop_body_parity_network_synth``. It synthesizes the parity network, as described
    in Algorithm 1 in https://arxiv.org/abs/2104.00934.

    Args:
        P (np.ndarray): Parity table to be synthesized.
            Shape should be ``(num_wires, num_parities)``

    Returns:
        tuple[list[int, list[tuple[int]]], np.ndarray]: Synthesized parity network, as a
        circuit with structure as described in ``_loop_body_parity_network_synth``. Also,
        inverse of the parity matrix implemented by the synthesized circuit.

    """
    if P.shape[-1] == 0:
        # Nothing to do if there are not parities
        return [], None

    circuit = []  # Line 1 in Alg. 1
    num_wires, num_parities = P.shape
    # Initialize an inverse parity matrix that is updated with the CNOTs that are synthesized here.
    inv_synth_mat = np.eye(num_wires, dtype=int)
    # `num_parities` loop iterations because each loop body takes care of one parity, we just
    # don't know which one. This makes the `for`-loop equivalent to line 2 in Alg. 1
    for _ in range(num_parities):
        P, inv_synth_mat, circuit = _loop_body_parity_network_synth(P, inv_synth_mat, circuit)

    return circuit, inv_synth_mat % 2


### end of xDSL-agnostic part

valid_phase_polynomial_ops = {"CNOT", "RZ"}


def make_phase_polynomial(
    ops: list[CustomOp],
    init_wire_map: dict[QubitType, int],
) -> tuple[np.ndarray]:
    r"""Compute the phase polynomial representation of a list of ``CustomOp``\ s.
    This implementation is very similar to :func:`~.transforms.intermediate_reps.phase_polynomial`
    but adjusted to work with xDSL objects."""
    wire_map = init_wire_map

    parity_matrix = np.eye(len(wire_map), dtype=int)
    parity_table = []
    angles = []
    arith_ops = []
    for op in ops:
        name = op.gate_name.data
        if name == "CNOT":
            control, target = wire_map.pop(op.in_qubits[0]), wire_map.pop(op.in_qubits[1])
            parity_matrix[target] += parity_matrix[control]
            wire_map[op.out_qubits[0]] = control
            wire_map[op.out_qubits[1]] = target
            continue

        # RZ
        angle = op.operands[0]
        if getattr(op, "adjoint", False):
            neg_op = arith.NegfOp(angle)
            arith_ops.append(neg_op)
            angle = neg_op.result
        angles.append(angle)
        wire = wire_map.pop(op.in_qubits[0])
        parity_table.append(parity_matrix[wire].copy())  # append _current_ parity (hence the copy)
        wire_map[op.out_qubits[0]] = wire

    return parity_matrix % 2, np.array(parity_table).T % 2, angles, arith_ops


def _cnot(i: int, j: int, inv_wire_map: dict[int, QubitType]):
    """Create a CNOT operator acting on the qubits that map to wires ``i`` and ``j``
    and update the wire map so that ``i`` and ``j`` point to the output qubits afterwards."""
    cnot_op = CustomOp(
        in_qubits=[inv_wire_map[i], inv_wire_map[j]],
        gate_name="CNOT",
        params=tuple(),
    )
    inv_wire_map[i] = cnot_op.out_qubits[0]
    inv_wire_map[j] = cnot_op.out_qubits[1]
    return cnot_op


def _rz(wire: int, angle: SSAValue[builtin.Float64Type], inv_wire_map: dict[int, QubitType]):
    """Create a CNOT operator acting on the qubit that maps to ``wire``
    and update the wire map so that ``wire`` points to the output qubit afterwards."""
    rz_op = CustomOp(in_qubits=[inv_wire_map[wire]], gate_name="RZ", params=(angle,))
    inv_wire_map[wire] = rz_op.out_qubits[0]
    return rz_op


class ParitySynthPattern(pattern_rewriter.RewritePattern):
    """Rewrite pattern that applies ``ParitySynth`` to subcircuits that constitute
    phase polynomials.
    """

    phase_polynomial_ops: list[CustomOp]
    init_wire_map: dict[QubitType, int]
    global_wire_map: dict[QubitType, int]
    phase_polynomial_ops: set[QubitType]
    num_phase_polynomial_qubits: int

    def __init__(self, *args, **kwargs):
        if not has_networkx:  # pragma: no cover
            raise ModuleNotFoundError(
                "The package networkx is required to run the ParitySynth pass."
                "You can install it via 'pip install networkx'."
            ) from networkx_import_error  # pylint: disable=used-before-assignment

        super().__init__(*args, **kwargs)
        self._reset_vars()

    def _reset_vars(self):
        """Initialize/reset variables that are used in ``match_and_rewrite`` as well as
        ``rewrite_phase_polynomial``."""
        self.phase_polynomial_ops = []
        self.init_wire_map = {}
        self.phase_polynomial_qubits = set()
        self.num_phase_polynomial_qubits = 0

    def _record_phase_poly_op(self, op: CustomOp):
        """Add a ``CustomOp`` to the phase polynomial ops, remove its input qubits
        from ``self.phase_polynomial_qubits`` if present or add them to ``self.init_wire_map``
        if not, and insert its output qubits in ``self.phase_polynomial_qubits``."""
        for i, q in enumerate(op.in_qubits):
            if q in self.phase_polynomial_qubits:
                self.phase_polynomial_qubits.remove(q)
            else:
                self.init_wire_map[q] = self.num_phase_polynomial_qubits
                self.num_phase_polynomial_qubits += 1
            self.phase_polynomial_qubits.add(op.out_qubits[i])
        self.phase_polynomial_ops.append(op)

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, matchedOp: Operation, rewriter: pattern_rewriter.PatternRewriter):
        r"""Implementation of rewriting ``FuncOps`` that may contain phase poynomials
        with ``ParitySynth``.

        Args:
            funcOp (func.FuncOp): function containing the operations to rewrite.
            rewriter (pattern_rewriter.PatternRewriter): Rewriter that executes operation erasure
                and insertion.

        The logic of this implementation is centered around :attr:`~.rewrite_phase_polynomial`,
        which is able to rewrite a collection of ``CustomOp``\ s that forms a phase polynomial
        (see ``valid_phase_polynomial_ops`` for the supported types) into a new collection of
        ``CustomOp``\ s that is equivalent. In addition to the operators, which are stored in
        ``self.phase_polynomial_ops``, the ``rewrite_phase_polynomial`` subroutine requires
        the initial mapping from input qubits to integer-valued wire positions, which is computed
        in ``self.init_wire_map`` using temporary variables ``self.phase_polynomial_qubits``
        and ``self.num_phase_polynomial_qubits``.

        Iterating over all operations, the collected phase polynomial ops are rewritten as soon
        as a non-phase-polynomial operation is encountered. Note that this makes the (size of the)
        rewritten phase polynomials dependent on the order in which we walk over the operations.
        """
        # The attribute is used so we don't transform the same op multiple times
        if len(matchedOp.regions) == 0 or hasattr(matchedOp, "parity_synth_done"):
            return

        for region in matchedOp.regions:
            for block in region.blocks:
                for op in block.ops:
                    # This loop body does one of three things:
                    # 1. If ``op`` is neither a ``CustomOp`` nor an InsertOp,
                    #    recurse on the regions of ``op`` but otherwise do nothing.
                    # 2. If ``op`` is a ``CustomOp`` that is not a phase polynomial operation
                    #    (RZ/CNOT), or an InsertOp, trigger rewrite_phase_polynomial
                    # 3. If ``op`` is a ``CustomOp`` that is a phase polynomial operation
                    #    (RZ/CNOT), record ``op`` to the aggregated phase polynomial subcircuit.
                    if not isinstance(op, (CustomOp, InsertOp)):
                        # Case 1: do "nothing", just recurse on op.regions
                        if len(op.regions) != 0:
                            # Do phase polynomial rewriting up to this point
                            self.rewrite_phase_polynomial(rewriter)
                            # Rewrite regions of this operation; Creating a new PatternRewriter
                            # so its matched operation is `op`, not `matchedOp`
                            self.match_and_rewrite(op, pattern_rewriter.PatternRewriter(op))
                        continue

                    if isinstance(op, CustomOp) and op.gate_name.data in valid_phase_polynomial_ops:
                        # Case 2: Include op in phase polynomial ops and track its qubits
                        self._record_phase_poly_op(op)
                        continue

                    # Case 3: not a phase polynomial op, so we trigger rewriting
                    self.rewrite_phase_polynomial(rewriter)

                # end of operations; rewrite terminal phase polynomial
                self.rewrite_phase_polynomial(rewriter)
                matchedOp.attributes["parity_synth_done"] = builtin.UnitAttr()

    def rewrite_phase_polynomial(self, rewriter: pattern_rewriter.PatternRewriter):
        """Rewrite a single region of a circuit that represents a phase polynomial."""
        if not self.phase_polynomial_ops:
            # Nothing to do
            return

        if len(self.phase_polynomial_ops) == 1:
            # Phase polynomials of length 1 are left untouched. Reset internal state
            self._reset_vars()
            return

        # Create an insertion point in the IR after the last phase polynomial op.
        # Inserting newly created ops at this point and the removing the phase polynomial ops
        # ensures that the newly synthesized phase polynomial is inserted in place of the old one
        insertion_point: InsertPoint = InsertPoint.after(self.phase_polynomial_ops[-1])

        # Mapping from integer-valued wire positions to qubits, corresponding to state before
        # phase polynomial
        inv_wire_map: dict[int, QubitType] = {val: key for key, val in self.init_wire_map.items()}

        # Calculate the new circuit by going to phase polynomial IR and back, including synthesis
        # of trailing CNOTs via rowcol
        M, P, angles, arith_ops = make_phase_polynomial(
            self.phase_polynomial_ops, self.init_wire_map
        )

        # Insert arithmetic operations produced within `make_phase_polynomial`
        for op in arith_ops:
            rewriter.insert_op(op, insertion_point)

        subcircuits, inv_network_parity_matrix = _parity_network_synth(P)
        # `inv_network_parity_matrix` might be None if the parity table was empty
        if inv_network_parity_matrix is not None:
            M = (M @ inv_network_parity_matrix) % 2
        rowcol_circuit: list[tuple[int]] = _rowcol_parity_matrix(M, connectivity=None)

        # Apply the parity network part of the new circuit
        for idx, phase_wire, subcircuit in subcircuits:
            for i, j in subcircuit:
                rewriter.insert_op(_cnot(i, j, inv_wire_map), insertion_point)

            rewriter.insert_op(_rz(phase_wire, angles.pop(idx), inv_wire_map), insertion_point)

        # Apply the remaining parity matrix part of the new circuit
        for i, j in rowcol_circuit:
            rewriter.insert_op(_cnot(i, j, inv_wire_map), insertion_point)

        # Replace the output qubits of the old phase polynomial operations by the output qubits of
        # the new circuit
        for old_qubit, int_wire in self.init_wire_map.items():
            rewriter.replace_all_uses_with(old_qubit, inv_wire_map[int_wire])

        # Erase the old phase polynomial operations.
        for op in self.phase_polynomial_ops[::-1]:
            rewriter.erase_op(op)

        # Reset internal state
        self._reset_vars()


@dataclass(frozen=True)
class ParitySynthPass(passes.ModulePass):
    """Pass for applying ParitySynth to phase polynomials in a circuit."""

    name = "parity-synth"

    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the ParitySynth pass."""
        pattern = ParitySynthPattern()
        walker = pattern_rewriter.PatternRewriteWalker(pattern, apply_recursively=False)
        walker.rewrite_module(module)


parity_synth_pass = compiler_transform(ParitySynthPass)
parity_synth_pass.__doc__ = r"""Pass for applying ParitySynth to phase polynomials in a circuit.

ParitySynth has been proposed by Vandaele et al. in `arXiv:2104.00934
<https://arxiv.org/abs/2104.00934>`__ as a technique to synthesize
`phase polynomials
<https://pennylane.ai/compilation/phase-polynomial-intermediate-representation>`__
into elementary quantum gates, namely ``CNOT`` and ``RZ``. For this, it synthesizes the
`parity table <https://pennylane.ai/compilation/parity-table>`__ of the phase polynomial,
and defers the remaining `parity matrix <https://pennylane.ai/compilation/parity-matrix>`__
synthesis to `RowCol <https://pennylane.ai/compilation/rowcol-algorithm>`__.

.. note::

    This pass requires the ``networkx`` package, which can be installed via
    ``pip install networkx``.

This pass walks over the input circuit and aggregates all ``CNOT`` and ``RZ`` operators
into a subcircuit that describes a phase polyonomial. Other gates form the boundaries of
these subcircuits, and whenever one is encountered the phase polynomial of the aggregated
subcircuit is resynthesized with the ParitySynth algorithm. This implies that while this
pass works on circuits containing any operations, it is recommended to maximize the
subcircuits that represent phase polynomials (i.e. consist of ``CNOT`` and ``RZ`` gates) to
enhance the effectiveness of the pass. This might be possible through decomposition or
re-ordering of commuting gates.
Note that nested regions, such as nested functions and control flow function bodies, are
synthesized independently, i.e., region boundaries are always treated as boundaries of phase
polynomial subcircuits. Similarly, dynamic wires create boundaries around the operations using them, causing separation
of phase polynomial operations into multiple subcircuits.

**Example**

In the following, we apply the pass to a simple quantum circuit that has optimization
potential in terms of commuting gates that can be interchanged to unlock a cancellation of
a self-inverse gate (``CNOT``) with itself. Concretely, the circuit is:

.. code-block:: python

    import pennylane as qml
    from catalyst.python_interface import Compiler
    from catalyst.python_interface.transforms import parity_synth_pass

    qml.capture.enable()
    dev = qml.device("lightning.qubit", wires=2)

    @qml.qjit(target="mlir")
    @parity_synth_pass
    @qml.qnode(dev)
    def circuit(x: float, y: float, z: float):
        qml.CNOT((0, 1))
        qml.RZ(x, 1)
        qml.CNOT((0, 1))
        qml.RX(y, 1)
        qml.CNOT((1, 0))
        qml.RZ(z, 1)
        qml.CNOT((1, 0))
        return qml.state()

We can draw the circuit and observe the last ``RZ`` gate to be wrapped in a pair of ``CNOT``
gates that commute with it:

>>> print(qml.draw(circuit)(0.52, 0.12, 0.2))
0: ─╭●───────────╭●───────────╭X───────────╭X─┤  State
1: ─╰X──RZ(0.52)─╰X──RX(0.12)─╰●──RZ(0.20)─╰●─┤  State

Now we apply the ``parity_synth_pass`` to the circuit and quantum just-in-time (qjit) compile
the circuit into a reduced MLIR module:

.. code-block:: python

    circuit_qjit = qml.qjit(parity_synth_pass(circuit), autograph=True, target="mlir")
    compiler = Compiler()
    mlir_module = compiler.run(circuit_qjit.mlir_module)

Looking at the compiled module below, we find only five gates left in the program (note that
we reduced the output for the purpose of this example); the ``CNOT``\ s
have been cancelled successfully. Note that for this circuit, ParitySynth is run twice; once
for the first three gates and once for the last three gates. This is because ``RX`` is not
a phase polynomial operation, so that it forms a boundary for the phase polynomial subcircuits
that are re-synthesized by the pass.

>>> print(mlir_module) # The following output has manually been reduced for readability
module @circuit {
  func.func public @jit_circuit([...]) -> tensor<4xcomplex<f64>> {
    %0 = "catalyst.launch_kernel"(%arg0, %arg1, %arg2) <[...]> :
        (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<4xcomplex<f64>>
    return %0 : tensor<4xcomplex<f64>>
  }
  module @module_circuit {
    func.func public @circuit(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>) ->
        tensor<4xcomplex<f64>> attributes [...] {
      %c = stablehlo.constant dense<0> : tensor<i64>
      %0 = "tensor.extract"(%c) : (tensor<i64>) -> i64
      "quantum.device"(%0) <[...]> : (i64) -> ()
      %c_0 = stablehlo.constant dense<2> : tensor<i64>
      %1 = "quantum.alloc"() <{nqubits_attr = 2 : i64}> : () -> !quantum.reg
      %2 = "tensor.extract"(%c) : (tensor<i64>) -> i64
      %3 = "quantum.extract"(%1, %2) : (!quantum.reg, i64) -> !quantum.bit
      %c_1 = stablehlo.constant dense<1> : tensor<i64>
      %4 = "tensor.extract"(%c_1) : (tensor<i64>) -> i64
      %5 = "quantum.extract"(%1, %4) : (!quantum.reg, i64) -> !quantum.bit
      %6 = "tensor.extract"(%arg0) : (tensor<f64>) -> f64
      %7:2 = "quantum.custom"(%5, %3) <{gate_name = "CNOT", [...]> :[...]
      %8 = "quantum.custom"(%6, %7#1) <{gate_name = "RZ", [...]> :[...]
      %9:2 = "quantum.custom"(%7#0, %8) <{gate_name = "CNOT", [...]> : [...]
      %10 = "tensor.extract"(%arg1) : (tensor<f64>) -> f64
      %11 = "quantum.custom"(%10, %9#0) <{gate_name = "RX", [...]> : [...]
      %12 = "tensor.extract"(%arg2) : (tensor<f64>) -> f64
      %13 = "quantum.custom"(%12, %11) <{gate_name = "RZ", [...]> : [...]
      %14 = "tensor.extract"(%c) : (tensor<i64>) -> i64
      %15 = "quantum.insert"(%1, %14, %9#1) : (!quantum.reg, i64, !quantum.bit) -> !quantum.reg
      %16 = "tensor.extract"(%c_1) : (tensor<i64>) -> i64
      %17 = "quantum.insert"(%15, %16, %13) : (!quantum.reg, i64, !quantum.bit) -> !quantum.reg
      %18 = "quantum.compbasis"(%17) <[...]> : (!quantum.reg) -> !quantum.obs
      %19 = "quantum.state"(%18) <[...]> : (!quantum.obs) -> tensor<4xcomplex<f64>>
      "quantum.dealloc"(%17) : (!quantum.reg) -> ()
      "quantum.device_release"() : () -> ()
      return %19 : tensor<4xcomplex<f64>>
    }
  }
  func.func @setup() {
    "quantum.init"() : () -> ()
    return
  }
  func.func @teardown() {
    "quantum.finalize"() : () -> ()
    return
  }
}

"""
