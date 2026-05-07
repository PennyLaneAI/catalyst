# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module exposes built-in Catalyst MLIR passes to the frontend."""

import copy
import json
from pathlib import Path
from typing import Iterable

import pennylane as qp
from pennylane.decomposition.utils import to_name

from catalyst.compiler import _options_to_cli_flags, _quantum_opt
from catalyst.utils.exceptions import CompileError
from catalyst.utils.runtime_environment import BYTECODE_FILE_PATH

# pylint: disable=line-too-long, too-many-lines


## API ##
def cancel_inverses_setup_inputs():
    """
    Specify that the ``-cancel-inverses`` MLIR compiler pass
    for cancelling two neighbouring self-inverse
    gates should be applied to the decorated QNode during :func:`~.qjit`
    compilation.

    The full list of supported gates are as follows:

    One-bit Gates:
    :class:`qp.Hadamard <pennylane.Hadamard>`,
    :class:`qp.PauliX <pennylane.PauliX>`,
    :class:`qp.PauliY <pennylane.PauliY>`,
    :class:`qp.PauliZ <pennylane.PauliZ>`

    Two-bit Gates:
    :class:`qp.CNOT <pennylane.CNOT>`,
    :class:`qp.CY <pennylane.CY>`,
    :class:`qp.CZ <pennylane.CZ>`,
    :class:`qp.SWAP <pennylane.SWAP>`

    Three-bit Gates:
    :class:`qp.Toffoli <pennylane.Toffoli>`

    .. note::

        Unlike PennyLane :doc:`circuit transformations <introduction/compiling_circuits>`,
        the QNode itself will not be changed or transformed by applying these
        decorators.

        As a result, circuit inspection tools such as :func:`~.draw` will continue
        to display the circuit as written in Python.

        To instead view the optimized circuit, the MLIR must be viewed
        after the ``"QuantumCompilationStage"`` stage via the
        :func:`~.get_compilation_stage` function.

    Args:
        fn (QNode): the QNode to apply the cancel inverses compiler pass to

    Returns:
        :class:`QNode <pennylane.QNode>`

    **Example**

    .. code-block:: python

        import pennylane as qp
        from catalyst import qjit
        from catalyst.debug import get_compilation_stage
        from catalyst.passes import cancel_inverses

        dev = qp.device("lightning.qubit", wires=1)

        @qjit(keep_intermediate=True)
        @cancel_inverses
        @qp.qnode(dev)
        def circuit(x: float):
            qp.RX(x, wires=0)
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=0)
            return qp.expval(qp.PauliZ(0))

    >>> circuit(0.54)
    Array(0.85770868, dtype=float64)

    Note that the QNode will be unchanged in Python, and will continue
    to include self-inverse gates when inspected with Python (for example,
    with :func:`~.draw`).

    To instead view the optimized circuit, the MLIR must be viewed
    after the ``"QuantumCompilationStage"`` stage:

    >>> print(get_compilation_stage(circuit, stage="QuantumCompilationStage"))
    module @circuit {
      func.func public @jit_circuit(%arg0: tensor<f64>) -> tensor<f64> attributes {llvm.emit_c_interface} {
        %0 = call @circuit(%arg0) : (tensor<f64>) -> tensor<f64>
        return %0 : tensor<f64>
      }
      func.func private @circuit(%arg0: tensor<f64>) -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
        quantum.device["catalyst/utils/../lib/librtd_lightning.dylib", "LightningSimulator", "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
        %0 = quantum.alloc( 1) : !quantum.reg
        %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
        %extracted = tensor.extract %arg0[] : tensor<f64>
        %out_qubits = quantum.custom "RX"(%extracted) %1 : !quantum.bit
        %2 = quantum.namedobs %out_qubits[ PauliZ] : !quantum.obs
        %3 = quantum.expval %2 : f64
        %from_elements = tensor.from_elements %3 : tensor<f64>
        %4 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
        quantum.dealloc %4 : !quantum.reg
        quantum.device_release
        return %from_elements : tensor<f64>
      }
      func.func @setup() {
        quantum.init
        return
      }
      func.func @teardown() {
        quantum.finalize
        return
      }
    }

    It can be seen that both Hadamards have been cancelled, and the measurement
    directly follows the ``RX`` gate:

    .. code-block:: mlir

        %out_qubits = quantum.custom "RX"(%extracted) %1 : !quantum.bit
        %2 = quantum.namedobs %out_qubits[ PauliZ] : !quantum.obs
        %3 = quantum.expval %2 : f64
    """
    return (), {}


cancel_inverses = qp.transform(
    pass_name="cancel-inverses", setup_inputs=cancel_inverses_setup_inputs
)


def diagonalize_measurements_setup_inputs(
    supported_base_obs: tuple[str, ...] = ("PauliZ", "Identity"),
    to_eigvals: bool = False,
):
    """
    Specify that the ``diagonalize-final-measurements`` compiler pass
    will be applied, which diagonalizes measurements into the standard basis.

    Args:
        qnode (QNode): The QNode to apply the ``diagonalize_final_measurement`` compiler pass to.
        supported_base_obs (tuple[str, ...]): A list of supported base observable names.
            Allowed observables are ``PauliX``, ``PauliY``, ``PauliZ``, ``Hadamard`` and ``Identity``.
            ``PauliZ`` and ``Identity`` are always treated as supported, regardless of input. Defaults to
            (``PauliZ``, ``Identity``).
        to_eigvals (bool): Whether the diagonalization should create measurements using
            eigenvalues and wires rather than observables. Defaults to ``False``.

    Returns:
        :class:`QNode <pennylane.QNode>`

    .. note::
        Unlike the PennyLane tape transform, :func:`pennylane.transforms.diagonalize_measurements`,
        the QNode itself will not be changed or transformed by applying this decorator.

        Unlike the PennyLane tape transform, ``supported_base_obs`` here only accepts a tuple of supported
        base observable names, instead of the corresponding classes. The reason is that xDSL does not accept
        class types as values of option-elements. For more details, please refer to the `xDSL repo <https://github.com/xdslproject/xdsl/blob/ba190d9ba1612807e7604374afa7eb2c1c3d2047/xdsl/utils/arg_spec.py#L315-L327>`__.

        Unlike the PennyLane tape transform, only ``to_eigvals = False`` is supported. Setting ``to_eigvals`` as ``True``
        will raise an error.

        An error will be raised if non-commuting terms are encountered.

    **Example**

    The ``diagonalize-final-measurements`` compilation pass can be applied as a decorator on a QNode:

    .. code-block:: python

        import pennylane as qp
        from catalyst import qjit
        from catalyst.passes import diagonalize_measurements

        @qjit
        @diagonalize_measurements(supported_base_obs=("PauliX",))
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit():
            qp.Hadamard(0)
            qp.RZ(1.1, 0)
            qp.PhaseShift(0.22, 0)
            return qp.expval(qp.Y(0))

        expected_substr = 'transform.apply_registered_pass "diagonalize-final-measurements" with options = {"supported-base-obs" = ["PauliX"], "to-eigvals" = false}'

    >>> expected_substr in circuit.mlir
    True
    >>> circuit()
    0.9687151001182651

    An error is raised if ``to_eigvals=True`` is passed as an option:

    .. code-block:: python

        import pennylane as qp
        from catalyst import qjit
        from catalyst.passes import diagonalize_measurements

        @diagonalize_measurements(to_eigvals=True)
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit():
            qp.Hadamard(0)
            qp.PhaseShift(0.22, 0)
            return qp.expval(qp.Y(0))

        error_msg = None

        try:
            qjit(circuit)
        except ValueError as e:
            error_msg = str(e)

    >>> print(error_msg)
    Only to_eigvals = False is supported.

    A compile error is raised if non-commuting terms are encountered:

    .. code-block:: python

        import pennylane as qp
        from pennylane.exceptions import CompileError
        from catalyst import qjit
        from catalyst.passes import diagonalize_measurements

        @diagonalize_measurements
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit():
            qp.Hadamard(0)
            return qp.expval(qp.Y(0) + qp.X(0))

        error_msg = None

        try:
            qjit(circuit)
        except CompileError as e:
            error_msg = str(e)

    >>> print(error_msg)
    Observables are not qubit-wise commuting. Please apply the `split-non-commuting` pass first.
    """
    return (), {"supported_base_obs": supported_base_obs, "to_eigvals": to_eigvals}


diagonalize_measurements = qp.transform(
    pass_name="diagonalize-final-measurements", setup_inputs=diagonalize_measurements_setup_inputs
)


def disentangle_cnot_setup_inputs():
    r"""A relaxed peephole optimization for replacing ``CNOT`` gates with single-qubit gates.

    The optimizations that this pass performs are found in
    `arXiv:2012.07711 <https://arxiv.org/pdf/2012.07711>`, specifically TABLE I. The patterns
    therein represent functional equivalencies to applying a ``CNOT`` gate on certain two-qubit
    input states.

    .. note::

        This transform requires decorating the workflow with :func:`~.qjit`.

    Args:
        fn (QNode): the QNode to apply the pass to

    Returns:
        :class:`QNode <pennylane.QNode>`

    **Example**

    In the circuit below, the ``CNOT`` gate can be simplified to just a ``PauliX`` gate since the
    control qubit is always in the :math:`|1\rangle` state.

    .. code-block:: python

        import pennylane as qp

        dev = qp.device("lightning.qubit", wires=2)

        @qp.qjit(capture=True)
        @qp.transforms.disentangle_cnot
        @qp.qnode(dev)
        def circuit():
            # first qubit in |1>
            qp.X(0)
            # second qubit in |0>
            # current state : |10>
            qp.CNOT([0, 1]) # state after CNOT : |11>
            return qp.state()

    When inspecting the circuit resources, only ``PauliX`` gates are present.

    >>> print(qp.specs(circuit, level=1)())
    Device: lightning.qubit
    Device wires: 2
    Shots: Shots(total=None)
    Level: disentangle-cnot
    <BLANKLINE>
    Wire allocations: 2
    Total gates: 2
    Gate counts:
    - PauliX: 2
    Measurements:
    - state(all wires): 1
    Depth: Not computed
    """
    return (), {}


disentangle_cnot = qp.transform(
    pass_name="disentangle-cnot", setup_inputs=disentangle_cnot_setup_inputs
)


def disentangle_swap_setup_inputs():
    r"""A relaxed peephole optimization for replacing ``SWAP`` gates with single-qubit gates.

    The optimizations that this pass performs are found in
    `arXiv:2012.07711 <https://arxiv.org/pdf/2012.07711>`, specifically TABLE VI. The patterns
    therein represent functional equivalencies to applying a ``SWAP`` gate on certain two-qubit
    input states.

    .. note::

        This transform requires decorating the workflow with :func:`~.qjit`.

    Args:
        fn (QNode): the QNode to apply the pass to

    Returns:
        :class:`QNode <pennylane.QNode>`

    **Example**

    In the circuit below, the ``SWAP`` gate can be simplified to a ``PauliX`` gate and two ``CNOT``
    gates.

    .. code-block:: python

        import pennylane as qp

        dev = qp.device("lightning.qubit", wires=2)

        @qp.qjit(keep_intermediate=True)
        @qp.transforms.disentangle_swap
        @qp.qnode(dev)
        def circuit():
            # first qubit in |1>
            qp.X(0)
            # second qubit in non-basis
            qp.RX(0.2, 1)
            qp.SWAP([0, 1])
            return qp.state()

    When inspecting the circuit resources, the ``SWAP`` gate is no longer present.

    >>> print(qp.specs(circuit, level=1)())
    Device: lightning.qubit
    Device wires: 2
    Shots: Shots(total=None)
    Level: disentangle-swap
    <BLANKLINE>
    Wire allocations: 2
    Total gates: 5
    Gate counts:
    - PauliX: 2
    - RX: 1
    - CNOT: 2
    Measurements:
    - state(all wires): 1
    Depth: Not computed
    """
    return (), {}


disentangle_swap = qp.transform(
    pass_name="disentangle-swap", setup_inputs=disentangle_swap_setup_inputs
)


def merge_rotations_setup_inputs():
    r"""Specify that the ``-merge-rotations`` MLIR compiler pass
    for merging roations (peephole) will be applied.

    The full list of supported gates are as follows:

    :class:`qp.RX <pennylane.RX>`,
    :class:`qp.CRX <pennylane.CRX>`,
    :class:`qp.RY <pennylane.RY>`,
    :class:`qp.CRY <pennylane.CRY>`,
    :class:`qp.RZ <pennylane.RZ>`,
    :class:`qp.CRZ <pennylane.CRZ>`,
    :class:`qp.PhaseShift <pennylane.PhaseShift>`,
    :class:`qp.ControlledPhaseShift <pennylane.ControlledPhaseShift>`,
    :class:`qp.Rot <pennylane.Rot>`,
    :class:`qp.CRot <pennylane.CRot>`,
    :class:`qp.MultiRZ <pennylane.MultiRZ>`.

    .. note::

        Unlike PennyLane :doc:`circuit transformations <introduction/compiling_circuits>`,
        the QNode itself will not be changed or transformed by applying these
        decorators.

        As a result, circuit inspection tools such as :func:`~.draw` will continue
        to display the circuit as written in Python.

        To instead view the optimized circuit, the MLIR must be viewed
        after the ``"QuantumCompilationStage`` stage via the
        :func:`~.get_compilation_stage` function.

    Args:
        fn (QNode): the QNode to apply the merge rotations compiler pass to

    Returns:
        :class:`QNode <pennylane.QNode>`

    **Example**

    In this example the three :class:`qp.RX <pennylane.RX>` will be merged in a single
    one with the sum of angles as parameter.

    .. code-block:: python

        from catalyst.debug import get_compilation_stage
        from catalyst.passes import merge_rotations

        dev = qp.device("lightning.qubit", wires=1)

        @qjit(keep_intermediate=True)
        @merge_rotations
        @qp.qnode(dev)
        def circuit(x: float):
            qp.RX(x, wires=0)
            qp.RX(0.1, wires=0)
            qp.RX(x**2, wires=0)
            return qp.expval(qp.PauliZ(0))

    >>> circuit(0.54)
    Array(0.5965506257017892, dtype=float64)
    """
    return (), {}


merge_rotations = qp.transform(
    pass_name="merge-rotations", setup_inputs=merge_rotations_setup_inputs
)


def parity_synth_setup_inputs():
    r"""Pass for synthesizing phase polynomials in a circuit.

    ParitySynth has been proposed by Vandaele et al. in `arXiv:2104.00934
    <https://arxiv.org/abs/2104.00934>`__ as a technique to synthesize
    `phase polynomials
    <https://pennylane.ai/compilation/phase-polynomial-intermediate-representation>`__
    into elementary quantum gates, namely ``CNOT`` and ``RZ``. For this, it synthesizes the
    `parity table <https://pennylane.ai/compilation/parity-table>`__ of the phase polynomial,
    and defers the remaining `parity matrix <https://pennylane.ai/compilation/parity-matrix>`__
    synthesis to `RowCol <https://pennylane.ai/compilation/rowcol-algorithm>`__.

    .. note::

        This transform requires decorating the workflow with :func:`~.qjit`. Additionally, this pass
        requires the ``networkx`` package, which can be installed via
        ``pip install networkx``.

    Args:
        fn (QNode): QNode to apply the pass to

    Returns:
        :class:`QNode <pennylane.QNode>`

    This pass walks over the input circuit and aggregates all ``CNOT`` and ``RZ`` operators
    into a subcircuit that describes a phase polyonomial. Other gates form the boundaries of
    these subcircuits, and whenever one is encountered the phase polynomial of the aggregated
    subcircuit is resynthesized with the ParitySynth algorithm. This implies that while this
    pass works on circuits containing any operations, it is recommended to maximize the
    subcircuits that represent phase polynomials (i.e. consist of ``CNOT`` and ``RZ`` gates) to
    enhance the effectiveness of the pass. This might be possible through decomposition or
    re-ordering of commuting gates.

    Note that higher-level program structures, such as nested functions and control flow, are
    synthesized independently. I.e., boundaries of such structures are always treated as boundaries
    of phase polynomial subcircuits as well. Similarly, dynamic wires create boundaries around the
    operations using them, potentially causing the separation of consecutive phase polynomial
    operations into multiple subcircuits.

    **Example**

    In the following, we apply the pass to a simple quantum circuit that has optimization
    potential in terms of commuting gates that can be interchanged to unlock a cancellation of
    a self-inverse gate (``CNOT``) with itself. Concretely, the circuit is:

    .. code-block:: python

        import pennylane as qp
        from catalyst.python_interface import Compiler
        import catalyst

        dev = qp.device("lightning.qubit", wires=2)

        @qp.qjit(capture=True)
        @catalyst.passes.parity_synth
        @qp.qnode(dev)
        def circuit(x: float, y: float, z: float):
            qp.CNOT((0, 1))
            qp.RZ(x, 1)
            qp.CNOT((0, 1))
            qp.RX(y, 1)
            qp.CNOT((1, 0))
            qp.RZ(z, 1)
            qp.CNOT((1, 0))
            return qp.state()

    We can draw the circuit and observe the last ``RZ`` gate to be wrapped in a pair of ``CNOT``
    gates that commute with it. Before the pass is applied:

    >>> fig, ax = catalyst.draw_graph(circuit, level=0)(0.52, 0.12, 0.2)
    >>> fig.savefig('path-to-file.png', dpi=300, bbox_inches='tight')

    .. figure:: /_static/parity-synth-example-before.png
        :width: 35%
        :alt: Example using ``parity_synth``
        :align: left

    After the pass is applied:

    >>> fig, ax = catalyst.draw_graph(circuit, level=1)(0.52, 0.12, 0.2)
    >>> fig.savefig('path-to-file.png', dpi=300, bbox_inches='tight')

    .. figure:: /_static/parity-synth-example-pass-applied.png
        :width: 35%
        :alt: Example using ``parity_synth``
        :align: left

    """
    return (), {}


parity_synth = qp.transform(pass_name="parity-synth", setup_inputs=parity_synth_setup_inputs)


def decompose_lowering_setup_inputs():  # pragma: no cover
    """
    Specify that the ``-decompose-lowering`` MLIR compiler pass
    for applying the compiled decomposition rules to the QNode
    recursively.

    Args:
        fn (QNode): the QNode to apply the decompose-lowering compiler pass to

    Returns:
        :class:`QNode <pennylane.QNode>`

    **Example**
        // TODO: add example here

    """
    return (), {}


decompose_lowering = qp.transform(
    pass_name="decompose-lowering", setup_inputs=decompose_lowering_setup_inputs
)  # pragma: no cover


def ions_decomposition_setup_inputs():  # pragma: nocover
    """
    Specify that the ``--ions-decomposition`` MLIR compiler pass should be
    applied to the decorated QNode during :func:`~.qjit` compilation.

    This compiler pass decomposes the gates from the set {T, S, PauliZ,
    Hadamard, PhaseShift, RZ, CNOT} into gates from the set {RX, RY, MS}, where
    MS is the Mølmer–Sørensen gate, commonly used by trapped-ion quantum
    devices.

    .. note::

        Unlike PennyLane :doc:`circuit transformations <introduction/compiling_circuits>`,
        the QNode itself will not be changed or transformed by applying these
        decorators.

        As a result, circuit inspection tools such as :func:`~.draw` will continue
        to display the circuit as written in Python.

        To instead view the optimized circuit, the MLIR must be viewed
        after the ``"QuantumCompilationStage"`` stage via the
        :func:`~.get_compilation_stage` function.

    Args:
        fn (QNode): the QNode to apply the ions-decomposition pass to

    Returns:
        :class:`QNode <pennylane.QNode>`

    **Example**

    .. code-block:: python

        import pennylane as qp
        from pennylane.devices import NullQubit

        import catalyst
        from catalyst import qjit
        from catalyst.debug import get_compilation_stage


        @qjit(keep_intermediate=True)
        @catalyst.passes.ions_decomposition
        @qp.qnode(NullQubit(2))
        def circuit():
            qp.Hadamard(wires=[0])
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliY(wires=0))


    >>> print(get_compilation_stage(circuit, stage="QuantumCompilationStage"))
    module @circuit {
      func.func public @jit_circuit() -> tensor<f64> attributes {llvm.emit_c_interface} {
        %0 = call @circuit_0() : () -> tensor<f64>
        return %0 : tensor<f64>
      }
      func.func public @circuit_0() -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
        %c0_i64 = arith.constant 0 : i64
        %cst = arith.constant 0.000000e+00 : f64
        %cst_0 = arith.constant 1.5707963267948966 : f64
        %cst_1 = arith.constant 3.1415926535897931 : f64
        %cst_2 = arith.constant -1.5707963267948966 : f64
        quantum.device shots(%c0_i64) ["catalyst/runtime/build/lib/librtd_null_qubit.so", "NullQubit", "{'shots': 0}"]
        %0 = quantum.alloc( 2) : !quantum.reg
        %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
        %out_qubits = quantum.custom "RX"(%cst) %1 : !quantum.bit
        %out_qubits_3 = quantum.custom "RY"(%cst_0) %out_qubits : !quantum.bit
        %out_qubits_4 = quantum.custom "RX"(%cst_1) %out_qubits_3 : !quantum.bit
        %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
        %out_qubits_5 = quantum.custom "RY"(%cst_0) %out_qubits_4 : !quantum.bit
        %out_qubits_6:2 = quantum.custom "MS"(%cst_0) %out_qubits_5, %2 : !quantum.bit, !quantum.bit
        %out_qubits_7 = quantum.custom "RX"(%cst_2) %out_qubits_6#0 : !quantum.bit
        %out_qubits_8 = quantum.custom "RY"(%cst_2) %out_qubits_6#1 : !quantum.bit
        %out_qubits_9 = quantum.custom "RY"(%cst_2) %out_qubits_7 : !quantum.bit
        %3 = quantum.namedobs %out_qubits_8[ PauliY] : !quantum.obs
        %4 = quantum.expval %3 : f64
        %from_elements = tensor.from_elements %4 : tensor<f64>
        %5 = quantum.insert %0[ 0], %out_qubits_8 : !quantum.reg, !quantum.bit
        %6 = quantum.insert %5[ 1], %out_qubits_9 : !quantum.reg, !quantum.bit
        quantum.dealloc %6 : !quantum.reg
        quantum.device_release
        return %from_elements : tensor<f64>
      }
      func.func @setup() {
        quantum.init
        return
      }
      func.func @teardown() {
        quantum.finalize
        return
      }
    }

    You can see that the Hadamard gate has been decomposed to RX(0)RY(pi/2)RX(pi):

    .. code-block:: mlir

        %cst = arith.constant 0.000000e+00 : f64
        %cst_0 = arith.constant 1.5707963267948966 : f64
        %cst_1 = arith.constant 3.1415926535897931 : f64
        ...
        %out_qubits = quantum.custom "RX"(%cst) %1 : !quantum.bit
        %out_qubits_3 = quantum.custom "RY"(%cst_0) %out_qubits : !quantum.bit
        %out_qubits_4 = quantum.custom "RX"(%cst_1) %out_qubits_3 : !quantum.bit

    and that the CNOT gate has been decomposed to its corresponding circuit
    implementation using the RX, RY and MS gates:

    .. code-block:: mlir

        %cst_0 = arith.constant 1.5707963267948966 : f64
        %cst_2 = arith.constant -1.5707963267948966 : f64
        ...
        %out_qubits_5 = quantum.custom "RY"(%cst_0) %out_qubits_4 : !quantum.bit
        %out_qubits_6:2 = quantum.custom "MS"(%cst_0) %out_qubits_5, %2 : !quantum.bit, !quantum.bit
        %out_qubits_7 = quantum.custom "RX"(%cst_2) %out_qubits_6#0 : !quantum.bit
        %out_qubits_8 = quantum.custom "RY"(%cst_2) %out_qubits_6#1 : !quantum.bit
        %out_qubits_9 = quantum.custom "RY"(%cst_2) %out_qubits_7 : !quantum.bit
    """
    return (), {}


ions_decomposition = qp.transform(
    pass_name="ions-decomposition", setup_inputs=ions_decomposition_setup_inputs
)


def combine_global_phases_setup_inputs():
    r"""A quantum compilation pass that combines global phase instructions for each region in the
    program.

    Args:
        qnode (QNode): the QNode to apply the compiler pass to

    Returns:
        :class:`QNode <pennylane.QNode>`

    **Example**

    Consider the following example:

    .. code-block:: python

        import pennylane as qp
        import catalyst
        from catalyst import qjit, for_loop
        from catalyst.passes import combine_global_phases
        from catalyst.debug import get_compilation_stage

        n = 10
        dev = qp.device('null.qubit', wires=n)

        @qjit(keep_intermediate=True, capture=True)
        @combine_global_phases
        @qp.qnode(dev)
        def circuit(n):
            qp.GlobalPhase(0.1, wires = n-1)
            qp.X(n-1)
            qp.GlobalPhase(0.1, wires = n-2)
            qp.H(n-2)

            @qp.for_loop(0, n)
            def loop(i):
                qp.GlobalPhase(0.1967, wires=i)
                qp.GlobalPhase(0.7691, wires=i)

            loop()

            qp.GlobalPhase(0.1, wires=n-3)
            qp.GlobalPhase(0.1, wires=0)

            return qp.expval(qp.Z(0))

    >>> circuit(n)
    0.0

    The ``GlobalPhase`` operations surrounding control flow operations will be merged, while those
    within the control flow are merged together separately (i.e., no formal loop-boundary
    optimizations).

    Example MLIR Representation:

    >>> print(get_compilation_stage(circuit, stage="QuantumCompilationStage"))

    .. code-block:: mlir

        . . .
        %extracted_4 = tensor.extract %3[] : tensor<i64>
        %5 = quantum.extract %4[%extracted_4] : !quantum.reg -> !quantum.bit
        %out_qubits = quantum.custom "PauliX"() %5 : !quantum.bit
        %6 = stablehlo.subtract %arg0, %c_1 : tensor<i64>
        %extracted_5 = tensor.extract %3[] : tensor<i64>
        %7 = quantum.insert %4[%extracted_5], %out_qubits : !quantum.reg, !quantum.bit
        %extracted_6 = tensor.extract %6[] : tensor<i64>
        %8 = quantum.extract %7[%extracted_6] : !quantum.reg -> !quantum.bit
        %9 = stablehlo.subtract %arg0, %c_1 : tensor<i64>
        %extracted_7 = tensor.extract %6[] : tensor<i64>
        %10 = quantum.insert %7[%extracted_7], %8 : !quantum.reg, !quantum.bit
        %extracted_8 = tensor.extract %9[] : tensor<i64>
        %11 = quantum.extract %10[%extracted_8] : !quantum.reg -> !quantum.bit
        %out_qubits_9 = quantum.custom "Hadamard"() %11 : !quantum.bit
        %extracted_10 = tensor.extract %9[] : tensor<i64>
        %12 = quantum.insert %10[%extracted_10], %out_qubits_9 : !quantum.reg, !quantum.bit
        %extracted_11 = tensor.extract %arg0[] : tensor<i64>
        %13 = arith.index_cast %extracted_11 : i64 to index
        %14 = scf.for %arg1 = %c0 to %13 step %c1 iter_args(%arg2 = %12) -> (!quantum.reg) {
        %22 = arith.index_cast %arg1 : index to i64
        %23 = quantum.extract %arg2[%22] : !quantum.reg -> !quantum.bit
        quantum.gphase(%cst_0)
        %24 = quantum.insert %arg2[%22], %23 : !quantum.reg, !quantum.bit
        scf.yield %24 : !quantum.reg
        }
        %15 = stablehlo.subtract %arg0, %c : tensor<i64>
        %extracted_12 = tensor.extract %15[] : tensor<i64>
        %16 = quantum.extract %14[%extracted_12] : !quantum.reg -> !quantum.bit
        %extracted_13 = tensor.extract %15[] : tensor<i64>
        %17 = quantum.insert %14[%extracted_13], %16 : !quantum.reg, !quantum.bit
        %18 = quantum.extract %17[ 0] : !quantum.reg -> !quantum.bit
        quantum.gphase(%cst)
        %19 = quantum.namedobs %18[ PauliZ] : !quantum.obs
        %20 = quantum.expval %19 : f64
        . . .
    """
    return (), {}


combine_global_phases = qp.transform(
    pass_name="combine-global-phases", setup_inputs=combine_global_phases_setup_inputs
)


def gridsynth_setup_inputs(epsilon=1e-4, ppr_basis=False):
    r"""A quantum compilation pass to discretize
    single-qubit RZ and PhaseShift gates into the Clifford+T basis or the PPR basis using the Ross-Selinger Gridsynth algorithm.
    Reference: https://arxiv.org/abs/1403.2975


    .. note::

        The actual discretization is only performed during execution time.

    Args:
        qnode (QNode): the QNode to apply the gridsynth compiler pass to
        epsilon (float): The maximum permissible operator norm error per rotation gate. Defaults to ``1e-4``.
        ppr_basis (bool): If true, decompose directly to Pauli Product Rotations (PPRs) in PBC dialect. Defaults to ``False``

    Returns:
        :class:`QNode <pennylane.QNode>`

    .. note::

        The circuit generated from this pass with ``ppr_basis=True`` are currently only executable on the
        ``lightning.qubit`` device with program  enabled.

    **Example**

    In this example the RZ gate will be converted into a new function, which
    calls the discretization at execution time.

    .. code-block:: python

        import pennylane as qp
        from catalyst import qjit
        from catalyst.passes import gridsynth

        pipe = [("pipe", ["quantum-compilation-stage"])]

        @qjit(pipelines=pipe, target="mlir")
        @gridsynth
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit():
            qp.RZ(x, wires=0)
            return qp.probs()

        >>> print(circuit.mlir_opt)

    Example MLIR Representation:

    .. code-block:: mlir

        . . .
        func.func private @rs_decomposition_get_phase(f64, f64, i1) -> f64
        func.func private @rs_decomposition_get_gates(memref<?xindex>, f64, f64, i1)
        func.func private @rs_decomposition_get_size(f64, f64, i1) -> index
        func.func private @__catalyst_decompose_RZ_0(%arg0: !quantum.bit, %arg1: f64) -> (!quantum.bit, f64) {
            . . .
            %2 = scf.for %arg2 = %c0 to %0 step %c1 iter_args(%arg3 = %arg0) -> (!quantum.bit) {
                %3 = memref.load %alloc[%arg2] : memref<?xindex>
                %4 = scf.index_switch %3 -> !quantum.bit
                case 0 {
                    %out_qubits = quantum.custom "T"() %arg3 : !quantum.bit
                    scf.yield %out_qubits : !quantum.bit
                }
                case 1 {
                    %out_qubits = quantum.custom "Hadamard"() %arg3 : !quantum.bit
                    %out_qubits_0 = quantum.custom "T"() %out_qubits : !quantum.bit
                    scf.yield %out_qubits_0 : !quantum.bit
                }
                case 2 {
                    %out_qubits = quantum.custom "S"() %arg3 : !quantum.bit
                    %out_qubits_0 = quantum.custom "Hadamard"() %out_qubits : !quantum.bit
                    %out_qubits_1 = quantum.custom "T"() %out_qubits_0 : !quantum.bit
                    scf.yield %out_qubits_1 : !quantum.bit
                }
                . . .
            }
        }

        func.func public @circuit_0(%arg0: tensor<f64>) -> tensor<f64> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
            . . .
            %2:2 = call @__catalyst_decompose_RZ_0(%1, %extracted) : (!quantum.bit, f64) -> (!quantum.bit, f64)
            . . .
        }



    """
    return (), {"epsilon": epsilon, "ppr_basis": ppr_basis}


gridsynth = qp.transform(pass_name="gridsynth", setup_inputs=gridsynth_setup_inputs)


def to_ppr_setup_inputs():
    r"""A quantum compilation pass that converts Clifford+T gates into Pauli Product Rotation (PPR)
    gates.

    .. note::

        This transform requires decorating the workflow with :func:`@qjit <~.qjit>`. In
        addition, the circuits generated by this pass are currently executable on
        ``lightning.qubit`` or ``null.qubit`` (for mock-execution).

    Clifford gates are defined as :math:`\exp(-{iP\tfrac{\pi}{4}})`, where :math:`P` is a Pauli word.
    Non-Clifford gates are defined as :math:`\exp(-{iP\tfrac{\pi}{8}})`.

    For more information on Pauli product measurements and Pauli product rotations, check out the
    `compilation hub <https://pennylane.ai/compilation/pauli-based-computation>`__.

    The full list of supported gates and operations are
    ``qp.H``,
    ``qp.S``,
    ``qp.T``,
    ``qp.X``,
    ``qp.Y``,
    ``qp.Z``,
    ``qp.PauliRot``,
    ``qp.adjoint(qp.PauliRot)``,
    ``qp.adjoint(qp.S)``,
    ``qp.adjoint(qp.T)``,
    ``qp.CNOT``, and
    ``qp.measure``.

    Args:
        fn (QNode): the QNode to apply the pass to

    Returns:
        :class:`QNode <pennylane.QNode>`

    .. seealso::
        :func:`pennylane.transforms.commute_ppr`, :func:`pennylane.transforms.merge_ppr_ppm`,
        :func:`pennylane.transforms.ppr_to_ppm`, :func:`pennylane.transforms.ppm_compilation`,
        :func:`pennylane.transforms.reduce_t_depth`, :func:`pennylane.transforms.decompose_arbitrary_ppr`

    .. note::

        For better compatibility with other PennyLane functionality, ensure that PennyLane program
        capture is enabled with ``@qjit(capture=True)``.

    **Example**

    The ``to_ppr`` compilation pass can be applied as a decorator on a QNode:

    .. code-block:: python

        import pennylane as qp

        @qp.qjit(capture=True)
        @qp.transforms.to_ppr
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit():
            qp.H(0)
            qp.CNOT([0, 1])
            m = qp.measure(0)
            qp.T(0)
            return qp.expval(qp.Z(0))

    >>> circuit()
    Array(-1., dtype=float64)
    >>> print(qp.specs(circuit, level=1)())
    Device: lightning.qubit
    Device wires: 2
    Shots: Shots(total=None)
    Level: to-ppr
    <BLANKLINE>
    Wire allocations: 2
    Total gates: 11
    Gate counts:
    - GlobalPhase: 3
    - PPR-pi/4-w1: 5
    - PPR-pi/4-w2: 1
    - PPM-w1: 1
    - PPR-pi/8-w1: 1
    Measurements:
    - expval(PauliZ): 1
    Depth: Not computed

    In the above output, ``PPR-theta-w<int>`` denotes the type of PPR present in the circuit, where
    ``theta`` is the PPR angle (:math:`\theta`) and ``w<int>`` denotes the PPR weight (the number of
    qubits it acts on, or the length of the Pauli word). ``PPM-w<int>`` follows the same convention.
    Note that the mid-circuit measurement (:func:`pennylane.measure`) in the circuit has been
    converted to a Pauli product measurement (PPM), as well.
    """
    return (), {}


to_ppr = qp.transform(pass_name="to-ppr", setup_inputs=to_ppr_setup_inputs)


def commute_ppr_setup_inputs(max_pauli_size=0):
    r"""A quantum compilation pass that commutes Clifford Pauli product rotation (PPR) gates,
    :math:`\exp(-{iP\tfrac{\pi}{4}})`, past non-Clifford PPRs gates,
    :math:`\exp(-{iP\tfrac{\pi}{8}})`, where :math:`P` is a Pauli word.

    .. note::

        This transform requires decorating the workflow with :func:`@qp.qjit <pennylane.qjit>`. In
        addition, the circuits generated by this pass are currently not executable on any
        backend. This pass is only for Pauli-based-computation analysis with the ``null.qubit``
        device and potential future execution when a suitable backend is available.

        Lastly, the :func:`pennylane.transforms.to_ppr` transform must be applied before
        ``commute_ppr``.

    For more information on PPRs, check out the
    `Compilation Hub <https://pennylane.ai/compilation/pauli-product-rotations>`_.

    Args:
        fn (QNode): QNode to apply the pass to
        max_pauli_size (int):
            The maximum size of Pauli strings resulting from commutation. If a commutation results
            in a PPR that acts on more than ``max_pauli_size`` qubits, that commutation will not be
            performed. Note that the default ``max_pauli_size=0`` indicates no limit.

    Returns:
        :class:`QNode <pennylane.QNode>`

    .. seealso::
        :func:`pennylane.transforms.to_ppr`, :func:`pennylane.transforms.merge_ppr_ppm`,
        :func:`pennylane.transforms.ppr_to_ppm`, :func:`pennylane.transforms.ppm_compilation`,
        :func:`pennylane.transforms.reduce_t_depth`, :func:`pennylane.transforms.decompose_arbitrary_ppr`

    .. note::

        For better compatibility with other PennyLane functionality, ensure that PennyLane program
        capture is enabled with ``@qjit(capture=True)``.

    **Example**

    The ``commute_ppr`` compilation pass can be applied as a decorator on a QNode:

    .. code-block:: python

        import pennylane as qp
        import jax.numpy as jnp

        @qp.qjit(capture=True)
        @qp.transforms.commute_ppr(max_pauli_size=2)
        @qp.transforms.to_ppr
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit():

            # equivalent to a Hadamard gate
            qp.PauliRot(jnp.pi / 2, pauli_word="Z", wires=0)
            qp.PauliRot(jnp.pi / 2, pauli_word="X", wires=0)
            qp.PauliRot(jnp.pi / 2, pauli_word="Z", wires=0)

            # equivalent to a CNOT gate
            qp.PauliRot(jnp.pi / 2, pauli_word="ZX", wires=[0, 1])
            qp.PauliRot(-jnp.pi / 2, pauli_word="Z", wires=0)
            qp.PauliRot(-jnp.pi / 2, pauli_word="X", wires=1)

            # equivalent to a T gate
            qp.PauliRot(jnp.pi / 4, pauli_word="Z", wires=0)

            return qp.expval(qp.Z(0))

    >>> circuit()
    Array(-1.11022302e-16, dtype=float64)
    >>> print(qp.specs(circuit, level=2)())
    Device: lightning.qubit
    Device wires: 2
    Shots: Shots(total=None)
    Level: commute-ppr
    <BLANKLINE>
    Wire allocations: 2
    Total gates: 7
    Gate counts:
    - PPR-pi/8-w1: 1
    - PPR-pi/4-w1: 5
    - PPR-pi/4-w2: 1
    Measurements:
    - expval(PauliZ): 1
    Depth: Not computed

    In the example above, the Clifford PPRs (:class:`~.PauliRot` instances with an angle of rotation
    of :math:`\tfrac{\pi}{2}`) will be commuted past the non-Clifford PPR (:class:`~.PauliRot`
    instances with an angle of rotation of :math:`\tfrac{\pi}{4}`). In the above output,
    ``PPR-theta-w<int>`` denotes the type of PPR present in the circuit, where ``theta`` is the PPR
    angle (:math:`\theta`) and ``w<int>`` denotes the PPR weight (the number of qubits it acts on,
    or the length of the Pauli word).

    Note that if a commutation resulted in a PPR acting on more than ``max_pauli_size`` qubits
    (here, ``max_pauli_size = 2``), that commutation would be skipped.
    """

    return (), {"max_pauli_size": max_pauli_size}


commute_ppr = qp.transform(pass_name="commute-ppr", setup_inputs=commute_ppr_setup_inputs)


def merge_ppr_ppm_setup_inputs(max_pauli_size=0):
    r"""A quantum compilation pass that absorbs Clifford Pauli product rotation (PPR) operations,
    :math:`\exp{-iP\tfrac{\pi}{4}}`, into the final Pauli product measurements (PPMs).

    .. note::

        This transform requires decorating the workflow with :func:`@qp.qjit <pennylane.qjit>`. In
        addition, the circuits generated by this pass are currently executable on
        ``lightning.qubit`` or ``null.qubit`` (for mock-execution).

        Secondly, the ``merge_ppr_ppm`` transform does not currently affect terminal measurements.
        So, for accurate results, it is recommended to return nothing (i.e., a blank ``return``
        statement) from the QNode.

        Lastly, the :func:`pennylane.transforms.to_ppr` transform must be applied before
        ``merge_ppr_ppm``.

    For more information on PPRs and PPMs, check out
    the `Compilation Hub <https://pennylane.ai/compilation/pauli-based-computation>`_.

    Args:
        fn (QNode): QNode to apply the pass to
        max_pauli_size (int):
            The maximum size of Pauli strings resulting from merging. If a merge results in a PPM
            that acts on more than ``max_pauli_size`` qubits, that merge will not be performed. The
            default value is ``0`` (no limit).

    Returns:
        :class:`QNode <pennylane.QNode>`

    .. seealso::
        :func:`pennylane.transforms.to_ppr`, :func:`pennylane.transforms.commute_ppr`,
        :func:`pennylane.transforms.ppr_to_ppm`, :func:`pennylane.transforms.ppm_compilation`,
        :func:`pennylane.transforms.reduce_t_depth`, :func:`pennylane.transforms.decompose_arbitrary_ppr`

    .. note::

        For better compatibility with other PennyLane functionality, ensure that PennyLane program
        capture is enabled with ``@qjit(capture=True)``.

    **Example**

    The ``merge_ppr_ppm`` compilation pass can be applied as a decorator on a QNode:

    .. code-block:: python

        import pennylane as qp
        import jax.numpy as jnp

        @qp.qjit(capture=True)
        @qp.transforms.merge_ppr_ppm(max_pauli_size=2)
        @qp.transforms.to_ppr
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit():
            qp.PauliRot(jnp.pi / 2, pauli_word="Z", wires=0)
            qp.PauliRot(jnp.pi / 2, pauli_word="X", wires=1)

            ppm = qp.pauli_measure(pauli_word="ZX", wires=[0, 1])

            return qp.probs()

    >>> circuit()
    Array([0.5, 0.5, 0. , 0. ], dtype=float64)
    >>> print(qp.specs(circuit, level=2)())
    Device: lightning.qubit
    Device wires: 2
    Shots: Shots(total=None)
    Level: merge-ppr-ppm
    <BLANKLINE>
    Wire allocations: 2
    Total gates: 3
    Gate counts:
    - PPM-w2: 1
    - PPR-pi/4-w1: 2
    Measurements:
    - probs(all wires): 1
    Depth: Not computed

    If a merging resulted in a PPM acting on more than ``max_pauli_size`` qubits, that merging
    operation would be skipped. In the above output, ``PPM-w<int>`` denotes the PPM weight (the
    number of qubits it acts on, or the length of the Pauli word).
    """
    return (), {"max_pauli_size": max_pauli_size}


merge_ppr_ppm = qp.transform(pass_name="merge-ppr-ppm", setup_inputs=merge_ppr_ppm_setup_inputs)


def ppr_to_ppm_setup_inputs(decompose_method="pauli-corrected", avoid_y_measure=False):
    r"""A quantum compilation pass that decomposes Pauli product rotations (PPRs),
    :math:`P(\theta) = \exp(-iP\theta)`, into Pauli product measurements (PPMs).

    .. note::

        This transform requires decorating the workflow with :func:`@qp.qjit <pennylane.qjit>`. In
        addition, the circuits generated by this pass are currently executable on
        ``lightning.qubit`` or ``null.qubit`` (for mock-execution).

        Lastly, the :func:`pennylane.transforms.to_ppr` transform must be applied before
        ``ppr_to_ppm``.

    This pass is used to decompose both non-Clifford and Clifford PPRs into PPMs. The non-Clifford
    PPRs (:math:`\theta = \tfrac{\pi}{8}`) are decomposed first, then Clifford PPRs
    (:math:`\theta = \tfrac{\pi}{4}`) are decomposed.

    For more information on PPRs and PPMs, check out
    the `Compilation Hub <https://pennylane.ai/compilation/pauli-based-computation>`_.

    Args:
        qnode (QNode): QNode to apply the pass to
        decompose_method (str): The method to use for decomposing non-Clifford PPRs.
            Options are ``"pauli-corrected"``, ``"auto-corrected"``, and ``"clifford-corrected"``.
            Defaults to ``"pauli-corrected"``.
            ``"pauli-corrected"`` uses a reactive measurement for correction that is based on Figure
            13 in `arXiv:2211.15465 <https://arxiv.org/pdf/2211.15465>`_.
            ``"auto-corrected"`` uses an additional measurement for correction that is based on
            Figure 7 in `A Game of Surface Codes <https://arxiv.org/abs/1808.02892>`__, and
            ``"clifford-corrected"`` uses a Clifford rotation for correction that is based on
            Figure 17(b) in `A Game of Surface Codes <https://arxiv.org/abs/1808.02892>`__.

        avoid_y_measure (bool): Rather than performing a Pauli-Y measurement for Clifford rotations
            (sometimes more costly), a :math:`Y` state (:math:`Y\vert 0 \rangle`) is used instead
            (requires :math:`Y`-state preparation). This is currently only supported when using the
            ``"clifford-corrected"`` and ``"pauli-corrected"`` decomposition methods. Defaults to
            ``False``.

    Returns:
        :class:`QNode <pennylane.QNode>`

    .. seealso::
        :func:`pennylane.transforms.to_ppr`, :func:`pennylane.transforms.commute_ppr`,
        :func:`pennylane.transforms.merge_ppr_ppm`, :func:`pennylane.transforms.ppm_compilation`,
        :func:`pennylane.transforms.reduce_t_depth`, :func:`pennylane.transforms.decompose_arbitrary_ppr`

    .. note::

        For better compatibility with other PennyLane functionality, ensure that PennyLane program
        capture is enabled with ``@qjit(capture=True)``.

    **Example**

    The ``ppr_to_ppm`` compilation pass can be applied as a decorator on a QNode:

    .. code-block:: python

        import pennylane as qp
        from functools import partial
        import jax.numpy as jnp

        @qp.qjit(capture=True)
        @qp.transforms.ppr_to_ppm
        @qp.transforms.to_ppr
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circuit():
            # equivalent to a Hadamard gate
            qp.PauliRot(jnp.pi / 2, pauli_word="Z", wires=0)
            qp.PauliRot(jnp.pi / 2, pauli_word="X", wires=0)
            qp.PauliRot(jnp.pi / 2, pauli_word="Z", wires=0)

            # equivalent to a CNOT gate
            qp.PauliRot(jnp.pi / 2, pauli_word="ZX", wires=[0, 1])
            qp.PauliRot(-jnp.pi / 2, pauli_word="Z", wires=[0])
            qp.PauliRot(-jnp.pi / 2, pauli_word="X", wires=[1])

            # equivalent to a T gate
            qp.PauliRot(jnp.pi / 4, pauli_word="Z", wires=0)

            return qp.expval(qp.Z(0))

    >>> print(qp.specs(circuit, level=2)())
    Device: null.qubit
    Device wires: 2
    Shots: Shots(total=None)
    Level: ppr-to-ppm
    <BLANKLINE>
    Wire allocations: 8
    Total gates: 22
    Gate counts:
    - PPM-w2: 6
    - PPM-w1: 7
    - PPM-w3: 1
    - PPR-pi/2-w1: 6
    - PPR-pi/2-w2: 1
    - pbc.fabricate: 1
    Measurements:
    - expval(PauliZ): 1
    Depth: Not computed

    In the above output, ``PPR-theta-w<int>`` denotes the type of PPR present in the circuit, where
    ``theta`` is the PPR angle (:math:`\theta`) and ``w<int>`` denotes the PPR weight (the number of
    qubits it acts on, or the length of the Pauli word). ``PPM-w<int>`` follows the same convention.

    Note that :math:`\theta = \tfrac{\pi}{2}` PPRs correspond to Pauli operators
    (:math:`P(\tfrac{\pi}{2}) = \exp(-iP\tfrac{\pi}{2}) = P`). Pauli operators can be commuted to
    the end of the circuit and absorbed into terminal measurements.
    """
    return (), {"decompose_method": decompose_method, "avoid_y_measure": avoid_y_measure}


ppr_to_ppm = qp.transform(pass_name="ppr-to-ppm", setup_inputs=ppr_to_ppm_setup_inputs)


def ppm_compilation_setup_inputs(
    decompose_method="pauli-corrected", avoid_y_measure=False, max_pauli_size=0
):
    r"""A quantum compilation pass that transforms Clifford+T gates into Pauli product measurements
    (PPMs).

    .. note::

        This transform requires decorating the workflow with :func:`@qp.qjit <pennylane.qjit>`. In
        addition, the circuits generated by this pass are currently executable on
        ``lightning.qubit`` or ``null.qubit`` (for mock-execution).

    This pass combines multiple sub-passes:

    - :func:`pennylane.transforms.to_ppr` : Converts gates into Pauli Product Rotations (PPRs)
    - :func:`pennylane.transforms.commute_ppr` : Commutes PPRs past non-Clifford PPRs
    - :func:`pennylane.transforms.merge_ppr_ppm` : Merges PPRs into Pauli Product Measurements (PPMs)
    - :func:`pennylane.transforms.ppr_to_ppm` : Decomposes PPRs into PPMs

    The ``avoid_y_measure`` and ``decompose_method`` arguments are passed to the
    :func:`pennylane.transforms.ppr_to_ppm` pass. The ``max_pauli_size`` argument is passed to the
    :func:`pennylane.transforms.commute_ppr` and :func:`pennylane.transforms.merge_ppr_ppm` passes.

    For more information on PPRs and PPMs, check out
    the `Compilation Hub <https://pennylane.ai/compilation/pauli-based-computation>`_.

    Args:
        qnode (QNode, optional): QNode to apply the pass to. If ``None``, returns a decorator.
        decompose_method (str, optional): The method to use for decomposing non-Clifford PPRs.
            Options are ``"pauli-corrected"``, ``"auto-corrected"``, and ``"clifford-corrected"``.
            Defaults to ``"pauli-corrected"``.
            ``"pauli-corrected"`` uses a reactive measurement for correction that is based on Figure
            13 in `arXiv:2211.15465 <https://arxiv.org/pdf/2211.15465>`_.
            ``"auto-corrected"`` uses an additional measurement for correction that is based on
            Figure 7 in `A Game of Surface Codes <https://arxiv.org/abs/1808.02892>`__, and
            ``"clifford-corrected"`` uses a Clifford rotation for correction that is based on
            Figure 17(b) in `A Game of Surface Codes <https://arxiv.org/abs/1808.02892>`__.

        avoid_y_measure (bool): Rather than performing a Pauli-Y measurement for Clifford rotations
            (sometimes more costly), a :math:`Y` state (:math:`Y\vert 0 \rangle`) is used instead
            (requires :math:`Y`-state preparation). This is currently only supported when using the
            ``"clifford-corrected"`` and ``"pauli-corrected"`` decomposition methods. Defaults to
            ``False``.

        max_pauli_size (int): The maximum size of the Pauli strings after commuting or merging.
            Defaults to 0 (no limit).

    Returns:
        :class:`QNode <pennylane.QNode>`

    .. note::

        For better compatibility with other PennyLane functionality, ensure that PennyLane program
        capture is enabled with ``@qjit(capture=True)``.

    **Example**

    The ``ppm_compilation`` compilation pass can be applied as a decorator on a QNode:

    .. code-block:: python

        import pennylane as qp

        @qp.qjit(capture=True)
        @qp.transforms.ppm_compilation(decompose_method="clifford-corrected", max_pauli_size=2)
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circuit():
            qp.H(0)
            qp.CNOT([0, 1])
            qp.T(0)
            return qp.expval(qp.Z(0))

    >>> print(qp.specs(circuit, level=1)())
    Device: null.qubit
    Device wires: 2
    Shots: Shots(total=None)
    Level: ppm-compilation
    <BLANKLINE>
    Wire allocations: 8
    Total gates: 25
    Gate counts:
    - GlobalPhase: 3
    - pbc.fabricate: 1
    - PPM-w2: 6
    - PPM-w1: 7
    - PPM-w3: 1
    - PPR-pi/2-w1: 6
    - PPR-pi/2-w2: 1
    Measurements:
    - expval(PauliZ): 1
    Depth: Not computed

    In the above output, ``PPR-theta-w<int>`` denotes the type of PPR present in the circuit, where
    ``theta`` is the PPR angle (:math:`\theta`) and ``w<int>`` denotes the PPR weight (the number of
    qubits it acts on, or the length of the Pauli word). ``PPM-w<int>`` follows the same convention.

    Note that :math:`\theta = \tfrac{\pi}{2}` PPRs correspond to Pauli operators
    (:math:`P(\tfrac{\pi}{2}) = \exp(-iP\tfrac{\pi}{2}) = P`). Pauli operators can be commuted to
    the end of the circuit and absorbed into terminal measurements.

    Lastly, if a commutation or merge resulted in a PPR or PPM acting on more than
    ``max_pauli_size`` qubits (here, ``max_pauli_size = 2``), that commutation or merge would be
    skipped.
    """

    return (), {
        "decompose_method": decompose_method,
        "avoid_y_measure": avoid_y_measure,
        "max_pauli_size": max_pauli_size,
    }


ppm_compilation = qp.transform(
    pass_name="ppm-compilation", setup_inputs=ppm_compilation_setup_inputs
)


def ppm_specs(fn):
    r"""This function returns following Pauli product rotation (PPR) and Pauli product measurement (PPM)
    specs in a dictionary:

    - Pi/4 PPR (count the number of clifford PPRs)
    - Pi/8 PPR (count the number of non-clifford PPRs)
    - Pi/2 PPR (count the number of classical PPRs)
    - Max weight for pi/8 PPRs
    - Max weight for pi/4 PPRs
    - Max weight for pi/2 PPRs
    - Number of logical qubits
    - Number of PPMs

    .. note::

        It is recommended to use :func:`pennylane.specs` instead of ``ppm_specs`` to retrieve
        resource counts of PPR-PPM workflows.

    When there is control flow, this function can count the above statistics inside for loops with
    a statically known number of iterations. For all other cases, including dynamically sized for
    loops, and any conditionals and while loops, this pass exits with failure.

    Args:
        fn (QJIT): qjit-decorated function for which ``ppm_specs`` need to be printed.

    Returns:
        dict: A Python dictionary containing PPM specs of all functions in ``fn``.

    **Example**

    .. code-block:: python

        import pennylane as qp
        import catalyst

        p = [("my_pipe", ["quantum-compilation-stage"])]
        device = qp.device("lightning.qubit", wires=2)

        @qp.qjit(pipelines=p, target="mlir")
        @catalyst.passes.ppm_compilation
        @qp.qnode(device)
        def circuit():
            qp.H(0)
            qp.CNOT([0,1])

            @catalyst.for_loop(0,10,1)
            def loop(i):
                qp.T(1)

            loop()
            return catalyst.measure(0), catalyst.measure(1)

        ppm_specs = catalyst.passes.ppm_specs(circuit)
        print(ppm_specs)

    Example PPM Specs:

    .. code-block:: pycon

        . . .
        {'circuit_0':
            {
                'depth_pi2_ppr': 7,
                'depth_ppm': 15,
                'logical_qubits': 2,
                'max_weight_pi2': 2,
                'num_of_ppm': 24,
                'pi2_ppr': 16
            }
        }
        . . .

    """

    if fn.mlir_module is not None:
        # aot mode
        new_options = copy.copy(fn.compile_options)
        if new_options.pipelines is None:
            raise CompileError("No pipeline found")

        # add ppm-spec pass at the end to existing pipeline
        _, pass_list = new_options.pipelines[0]  # first pipeline runs the user passes
        # check if ppm-specs is already in the pass list
        if "ppm-specs" not in pass_list:  # pragma: nocover
            pass_list.append("ppm-specs")

        new_options = _options_to_cli_flags(new_options)
        raw_result = _quantum_opt(*new_options, [], stdin=str(fn.mlir_module))

        try:
            return json.loads(
                raw_result[: raw_result.index("module")]
            )  # remove MLIR starting with substring "module..."
        except Exception as e:  # pragma: nocover
            raise CompileError(
                "Invalid json format encountered in ppm_specs. "
                f"Expected valid JSON but got {raw_result[: raw_result.index('module')]}"
            ) from e

    else:
        raise NotImplementedError("PPM passes only support AOT (Ahead-Of-Time) compilation mode.")


def reduce_t_depth_setup_inputs():
    r"""A quantum compilation pass that reduces the depth and count of non-Clifford Pauli product
    rotation (PPR, :math:`P(\theta) = \exp(-iP\theta)`) operators (e.g., ``T`` gates) by commuting
    PPRs in adjacent layers and merging compatible ones (a layer comprises PPRs that mutually
    commute). For more details, see Figure 6 of
    `A Game of Surface Codes <https://arXiv:1808.02892v3>`_.

    .. note::

        This transform requires decorating the workflow with :func:`@qp.qjit <pennylane.qjit>`. In
        addition, the circuits generated by this pass are currently executable on
        ``lightning.qubit`` or ``null.qubit`` (for mock-execution).

        Lastly, the :func:`pennylane.transforms.to_ppr` transform must be applied before
        ``reduce_t_depth``.

    Args:
        qnode (QNode): the QNode to apply the pass to

    Returns:
        :class:`QNode <pennylane.QNode>`

    .. seealso::
        :func:`pennylane.transforms.to_ppr`, :func:`pennylane.transforms.commute_ppr`,
        :func:`pennylane.transforms.merge_ppr_ppm`, :func:`pennylane.transforms.ppr_to_ppm`,
        :func:`pennylane.transforms.ppm_compilation`, :func:`pennylane.transforms.decompose_arbitrary_ppr`

    .. note::

        For better compatibility with other PennyLane functionality, ensure that PennyLane program
        capture is enabled with ``@qjit(capture=True)``.

    **Example**

    In the example below, after performing the :func:`pennylane.transforms.to_ppr` and
    :func:`pennylane.transforms.merge_ppr_ppm` passes, the circuit contains a depth of four of
    non-Clifford PPRs. Subsequently applying the ``reduce_t_depth`` pass will move PPRs around via
    commutation, resulting in a circuit with a smaller PPR depth.

    Specifically, in the circuit below, the Pauli-:math:`X` PPR (:math:`\exp(iX\tfrac{\pi}{8})`) on
    qubit Q1 will be moved to the first layer, which results in a depth of three non-Clifford PPRs.

    Consider the following example:

    .. code-block:: python

        import pennylane as qp
        import jax.numpy as jnp

        @qp.qjit(capture=True)
        @qp.transforms.reduce_t_depth
        @qp.transforms.to_ppr
        @qp.qnode(qp.device("null.qubit", wires=4))
        def circuit():
            qp.PauliRot(jnp.pi / 4, pauli_word="Z", wires=1)
            qp.PauliRot(-jnp.pi / 4, pauli_word="XYZ", wires=[0, 2, 3])
            qp.PauliRot(-jnp.pi / 2, pauli_word="XYZY", wires=[0, 1, 2, 3])
            qp.PauliRot(jnp.pi / 4, pauli_word="XZX", wires=[0, 1, 3])
            qp.PauliRot(-jnp.pi / 4, pauli_word="XZY", wires=[0, 1, 2])

            return qp.expval(qp.Z(0))

    The ``reduce_t_depth`` compilation pass will rearrange the last three PPRs in the above circuit
    to reduce the non-Clifford PPR depth. This is best seen with the :func:`catalyst.draw_graph`
    function:

    >>> import catalyst
    >>> num_passes = 2
    >>> fig1, _ = catalyst.draw_graph(circuit, level=num_passes-1)() # doctest: +SKIP
    >>> fig2, _ = catalyst.draw_graph(circuit, level=num_passes)() # doctest: +SKIP

    Without ``reduce_t_depth`` applied:

    >>> fig1.savefig('path_to_file1.png', dpi=300, bbox_inches="tight") # doctest: +SKIP

    .. figure:: /_static/reduce-t-depth-example1.png
        :width: 35%
        :alt: Graphical representation of circuit without ``reduce_t_depth``
        :align: left

    With ``reduce_t_depth`` applied:

    >>> fig2.savefig('path_to_file2.png', dpi=300, bbox_inches="tight") # doctest: +SKIP

    .. figure:: /_static/reduce-t-depth-example2.png
        :width: 35%
        :alt: Graphical representation of circuit with ``reduce_t_depth``
        :align: left
    """
    return (), {}


reduce_t_depth = qp.transform(pass_name="reduce-t-depth", setup_inputs=reduce_t_depth_setup_inputs)


def ppr_to_mbqc_setup_inputs():
    r"""Specify that the MLIR compiler pass for lowering Pauli Product Rotations (PPR)
    and Pauli Product Measurements (PPM) to a measurement-based quantum computing
    (MBQC) style circuit will be applied.

    This pass replaces PBC operations (``pbc.ppr`` and ``pbc.ppm``) with a
    gate-based sequence in the Quantum dialect using universal gates and
    measurements that supported as MBQC gate set.
    For details, see the Figure 2 of [Measurement-based Quantum Computation on cluster states](https://arxiv.org/abs/quant-ph/0301052).

    Conceptually, each Pauli product is handled by:

    - Mapping its Pauli string to the Z basis via per-qubit conjugations
      (e.g., ``H`` for ``X``; specialized ``RotXZX`` sequences for ``Y``).
    - Accumulating parity onto the first qubit with a right-to-left CNOT ladder.
    - Emitting the kernel operation:
      - **PPR**: apply an ``RZ`` with an angle derived from the rotation kind.
      - **PPM**: perform a measurement and return an ``i1`` result.
    - Uncomputing by reversing the CNOT ladder and the conjugations.
    - Conjugating the qubits back to the original basis.

    .. note::

        This pass expects PPR/PPM operations to be present. In practice, use it
        after :func:`~.passes.to_ppr`.

    Args:
        fn (QNode): the QNode to apply the pass to

    Returns:
        :class:`QNode <pennylane.QNode>`

    **Example**

    Convert a simple Clifford+T circuit to PPRs, then lower them to an
    MBQC-style circuit. Note that this pass should be applied before
    :func:`~.passes.ppr_to_ppm` since it requires the actual PPR/PPM operations.

    .. code-block:: python

        import pennylane as qp
        import catalyst

        p = [("my_pipe", ["quantum-compilation-stage"])]

        @qp.qjit(pipelines=p, target="mlir", keep_intermediate=True)
        @catalyst.passes.ppr_to_mbqc
        @catalyst.passes.to_ppr
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circuit():
            qp.H(0)
            qp.CNOT([0, 1])
            return

        print(circuit.mlir_opt)

    Example MLIR excerpt (structure only):

    .. code-block:: mlir
        ...
        %cst = arith.constant -1.5707963267948966 : f64
        %cst_0 = arith.constant 1.5707963267948966 : f64
        %0 = quantum.alloc( 2) : !quantum.reg
        %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
        %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
        %out_qubits = quantum.custom "RZ"(%cst_0) %1 : !quantum.bit
        %out_qubits_1 = quantum.custom "H"() %out_qubits : !quantum.bit
        %out_qubits_2 = quantum.custom "RZ"(%cst_0) %out_qubits_1 : !quantum.bit
        %out_qubits_3 = quantum.custom "H"() %out_qubits_2 : !quantum.bit
        %out_qubits_4 = quantum.custom "RZ"(%cst_0) %out_qubits_3 : !quantum.bit
        %out_qubits_5 = quantum.custom "H"() %2 : !quantum.bit
        %out_qubits_6:2 = quantum.custom "CNOT"() %out_qubits_5, %out_qubits_4 : !quantum.bit, !quantum.bit
        %out_qubits_7 = quantum.custom "RZ"(%cst_0) %out_qubits_6#1 : !quantum.bit
        %out_qubits_8:2 = quantum.custom "CNOT"() %out_qubits_6#0, %out_qubits_7 : !quantum.bit, !quantum.bit
        %out_qubits_9 = quantum.custom "H"() %out_qubits_8#0 : !quantum.bit
        %out_qubits_10 = quantum.custom "RZ"(%cst) %out_qubits_8#1 : !quantum.bit
        %out_qubits_11 = quantum.custom "H"() %out_qubits_9 : !quantum.bit
        %out_qubits_12 = quantum.custom "RZ"(%cst) %out_qubits_11 : !quantum.bit
        %out_qubits_13 = quantum.custom "H"() %out_qubits_12 : !quantum.bit
        %mres, %out_qubit = quantum.measure %out_qubits_13 : i1, !quantum.bit
        ...

    """

    return (), {}


ppr_to_mbqc = qp.transform(pass_name="ppr-to-mbqc", setup_inputs=ppr_to_mbqc_setup_inputs)


# This pass is already covered via applying by pass
# `qp.transform(pass_name="decompose-arbitrary-ppr")` in Pennylane.
def decompose_arbitrary_ppr_setup_inputs():  # pragma: nocover
    r"""A quantum compilation pass that decomposes arbitrary-angle Pauli product rotations (PPRs) into a
    collection of PPRs (with angles of rotation of :math:`\tfrac{\pi}{2}`, :math:`\tfrac{\pi}{4}`,
    and :math:`\tfrac{\pi}{8}`), PPMs and a single-qubit arbitrary-angle PPR in the Z basis. For
    details, see `Figure 13(d) of arXiv:2211.15465 <https://arxiv.org/abs/2211.15465>`__.

    .. note::

        This transform requires decorating the workflow with :func:`@qp.qjit <pennylane.qjit>`. In
        addition, the circuits generated by this pass are currently executable on
        ``lightning.qubit`` or ``null.qubit`` (for mock-execution).

        Lastly, the :func:`pennylane.transforms.to_ppr` transform must be applied before
        ``decompose_arbitrary_ppr``.

    Args:
        qnode (QNode): the QNode to apply the pass to

    Returns:
        :class:`QNode <pennylane.QNode>`

    .. seealso::
        :func:`pennylane.transforms.to_ppr`, :func:`pennylane.transforms.commute_ppr`,
        :func:`pennylane.transforms.merge_ppr_ppm`, :func:`pennylane.transforms.ppr_to_ppm`,
        :func:`pennylane.transforms.ppm_compilation`, :func:`pennylane.transforms.reduce_t_depth`

    .. note::

        For better compatibility with other PennyLane functionality, ensure that PennyLane program
        capture is enabled with ``@qjit(capture=True)``.

    **Example**

    In the example below, the arbitrary-angle PPR
    (``qp.PauliRot(0.1, pauli_word="XY", wires=[0, 1])``), will be decomposed into various other
    PPRs and PPMs in accordance with
    `Figure 13(d) of arXiv:2211.15465 <https://arxiv.org/abs/2211.15465>`__.

    .. code-block:: python

        import pennylane as qp

        @qp.qjit(capture=True)
        @qp.transforms.decompose_arbitrary_ppr
        @qp.transforms.to_ppr
        @qp.qnode(qp.device("null.qubit", wires=3))
        def circuit():
            qp.PauliRot(0.1, pauli_word="XY", wires=[0, 1])
            return qp.expval(qp.Z(0))

    >>> print(qp.specs(circuit, level=2)())
    Device: null.qubit
    Device wires: 3
    Shots: Shots(total=None)
    Level: decompose-arbitrary-ppr
    <BLANKLINE>
    Wire allocations: 3
    Total gates: 6
    Gate counts:
    - pbc.prepare: 1
    - PPM-w3: 1
    - PPM-w1: 1
    - PPR-pi/2-w1: 1
    - PPR-pi/2-w2: 1
    - PPR-Phi-w1: 1
    Measurements:
    - expval(PauliZ): 1
    Depth: Not computed

    In the above output, ``PPR-theta-w<int>`` denotes the type of PPR present in the circuit, where
    ``theta`` is the PPR angle (:math:`\theta`) and ``w<int>`` denotes the PPR weight (the number of
    qubits it acts on, or the length of the Pauli word). ``PPM-w<int>`` follows the same convention.
    ``PPR-Phi-w<int>`` corresponds to a PPR whose angle of rotation is not :math:`\tfrac{\pi}{2}`,
    :math:`\tfrac{\pi}{4}`, or :math:`\tfrac{\pi}{8}`.
    """
    return (), {}


decompose_arbitrary_ppr = qp.transform(
    pass_name="decompose-arbitrary-ppr", setup_inputs=decompose_arbitrary_ppr_setup_inputs
)


def graph_decomposition_setup_inputs(
    gate_set: Iterable[type | str] | dict[type | str, float],
    fixed_decomps: dict | None = None,
    alt_decomps: dict | None = None,
    bytecode_rules: str | None = None,
    _builtin_rule_path: Path = BYTECODE_FILE_PATH,
):  # pylint: disable=unused-argument
    R"""
    Specify that the ``-graph-decomposition`` MLIR compiler pass for applying the graph-based
    decomposition should be applied to the decorated QNode during :func:`~.qjit` compilation.

    The graph-based decomposition pass decomposes gates into a weighted target ``gate_set``
    by applying user-provided and built-in decomposition rules. The graph-based framework
    allows multiple decomposition rules to be defined for a quantum operation,
    and the graph solver will determine the optimal decomposition rules to apply,
    minimizing the overall gate count or the cost according to user-specified weights.

    .. note::

        The QNode itself will not be changed or transformed by applying these decorators.

        As a result, circuit inspection tools such as :func:`~.draw` will continue
        to display the circuit as written in Python.

        To instead view the optimized circuit, the MLIR must be viewed
        after the ``"QuantumCompilationStage"`` stage via the
        :func:`~.get_compilation_stage` function.

    Args:
        fn (QNode): the QNode to apply the graph decomposition compiler pass to.
        gate_set (Iterable[type | str] | dict[type | str, float]): the set of gates that are
            permissable after decomposition.
        fixed_decomps (dict | None): map ops to decomps that will be forcibly applied.
        alt_decomps (dict | None): map ops to lists of decomps that the graph system will consider.

    Returns:
        ~.QNode:

    **Example**

    .. code-block:: python

        import pennylane as qp
        import pennylane.numpy as np

        from catalyst import qjit
        from catalyst.jax_primitives import decomposition_rule
        from catalyst.passes import cancel_inverses, graph_decomposition, merge_rotations


        @decomposition_rule(op_type=qp.PauliX)
        def x_to_rx(wire: int):
            qp.RX(np.pi, wire)


        @decomposition_rule(op_type=qp.PauliY)
        def y_to_ry(wire: int):
            qp.RY(np.pi, wire)


        @decomposition_rule(op_type=qp.Hadamard)
        def h_to_rx_ry(wire: int):
            qp.RX(np.pi / 2, wire)
            qp.RY(np.pi / 2, wire)


        @qjit(capture=True)
        @graph_decomposition(gate_set={qp.Rot})
        @merge_rotations
        @graph_decomposition(
            gate_set={qp.RX, qp.RY},
            fixed_decomps={qp.PauliX: x_to_rx, qp.PauliY: y_to_ry},
            alt_decomps={qp.H: [h_to_rx_ry]},
        )
        @cancel_inverses
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit(x: float, y: float):
            qp.H(0)
            qp.H(0)
            qp.RX(x, wires=0)
            qp.PauliX(0)
            qp.RY(y, wires=0)
            qp.PauliY(0)
            qp.RY(x + y, wires=0)

            # register custom decomposition rules
            x_to_rx(int)
            y_to_ry(int)
            h_to_rx_ry(int)

            return qp.state()

    >>> qp.specs(circuit, level="device")(1.23, 4.56).resources.gate_types
    {'Rot': 2}
    """

    if not isinstance(gate_set, dict):
        gate_set = {to_name(op): 1.0 for op in gate_set}
    else:
        gate_set = {to_name(op): float(cost) for op, cost in gate_set.items()}

    options: dict[str, dict | tuple | str] = {
        "gate_set": gate_set,
        "bytecode_rules": str(_builtin_rule_path),
    }

    if fixed_decomps:
        options |= {
            "fixed_decomps": {
                to_name(op): (rule if isinstance(rule, str) else rule.__name__)
                for op, rule in fixed_decomps.items()
            }
        }

    if alt_decomps:
        options |= {
            "alt_decomps": {
                to_name(op): tuple(
                    (rule if isinstance(rule, str) else rule.__name__) for rule in rules
                )
                for op, rules in alt_decomps.items()
            }
        }

    return (), options


graph_decomposition = qp.transform(
    pass_name="graph-decomposition", setup_inputs=graph_decomposition_setup_inputs
)

__all__ = [
    "cancel_inverses",
    "combine_global_phases",
    "disentangle_cnot",
    "disentangle_swap",
    "merge_rotations",
    "parity_synth",
    "decompose_lowering",
    "ions_decomposition",
    "to_ppr",
    "gridsynth",
    "commute_ppr",
    "merge_ppr_ppm",
    "ppr_to_ppm",
    "ppm_compilation",
    "reduce_t_depth",
    "ppr_to_mbqc",
    "decompose_arbitrary_ppr",
    "graph_decomposition",
    "diagonalize_measurements",
]
