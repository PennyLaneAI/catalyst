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

from catalyst.passes.pass_api import PassPipelineWrapper


## API ##
def cancel_inverses(qnode):
    """
    Specify that the ``-removed-chained-self-inverse`` MLIR compiler pass
    for cancelling two neighbouring self-inverse
    gates should be applied to the decorated QNode during :func:`~.qjit`
    compilation.

    The full list of supported gates are as follows:

    One-bit Gates:
    :class:`qml.Hadamard <pennylane.Hadamard>`,
    :class:`qml.PauliX <pennylane.PauliX>`,
    :class:`qml.PauliY <pennylane.PauliY>`,
    :class:`qml.PauliZ <pennylane.PauliZ>`

    Two-bit Gates:
    :class:`qml.CNOT <pennylane.CNOT>`,
    :class:`qml.CY <pennylane.CY>`,
    :class:`qml.CZ <pennylane.CZ>`,
    :class:`qml.SWAP <pennylane.SWAP>`

    Three-bit Gates:
    - :class:`qml.Toffoli <pennylane.Toffoli>`

    .. note::

        Unlike PennyLane :doc:`circuit transformations <introduction/compiling_circuits>`,
        the QNode itself will not be changed or transformed by applying these
        decorators.

        As a result, circuit inspection tools such as :func:`~.draw` will continue
        to display the circuit as written in Python.

        To instead view the optimized circuit, the MLIR must be viewed
        after the ``"QuantumCompilationPass"`` stage via the
        :func:`~.get_compilation_stage` function.

    Args:
        fn (QNode): the QNode to apply the cancel inverses compiler pass to

    Returns:
        ~.QNode:

    **Example**

    .. code-block:: python

        from catalyst.debug import get_compilation_stage
        from catalyst.passes import cancel_inverses

        dev = qml.device("lightning.qubit", wires=1)

        @qjit(keep_intermediate=True)
        @cancel_inverses
        @qml.qnode(dev)
        def circuit(x: float):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

    >>> circuit(0.54)
    Array(0.85770868, dtype=float64)

    Note that the QNode will be unchanged in Python, and will continue
    to include self-inverse gates when inspected with Python (for example,
    with :func:`~.draw`).

    To instead view the optimized circuit, the MLIR must be viewed
    after the ``"QuantumCompilationPass"`` stage:

    >>> print(get_compilation_stage(circuit, stage="QuantumCompilationPass"))
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
    return PassPipelineWrapper(qnode, "remove-chained-self-inverse")


def merge_rotations(qnode):
    """
    Specify that the ``-merge-rotations`` MLIR compiler pass
    for merging roations (peephole) will be applied.

    The full list of supported gates are as follows:

    :class:`qml.RX <pennylane.RX>`,
    :class:`qml.CRX <pennylane.CRX>`,
    :class:`qml.RY <pennylane.RY>`,
    :class:`qml.CRY <pennylane.CRY>`,
    :class:`qml.RZ <pennylane.RZ>`,
    :class:`qml.CRZ <pennylane.CRZ>`,
    :class:`qml.PhaseShift <pennylane.PhaseShift>`,
    :class:`qml.ControlledPhaseShift <pennylane.ControlledPhaseShift>`,
    :class:`qml.MultiRZ <pennylane.MultiRZ>`.


    .. note::

        Unlike PennyLane :doc:`circuit transformations <introduction/compiling_circuits>`,
        the QNode itself will not be changed or transformed by applying these
        decorators.

        As a result, circuit inspection tools such as :func:`~.draw` will continue
        to display the circuit as written in Python.

        To instead view the optimized circuit, the MLIR must be viewed
        after the ``"QuantumCompilationPass"`` stage via the
        :func:`~.get_compilation_stage` function.

    Args:
        fn (QNode): the QNode to apply the cancel inverses compiler pass to

    Returns:
        ~.QNode:

    **Example**

    In this example the three :class:`qml.RX <pennylane.RX>` will be merged in a single
    one with the sum of angles as parameter.

    .. code-block:: python

        from catalyst.debug import get_compilation_stage
        from catalyst.passes import merge_rotations

        dev = qml.device("lightning.qubit", wires=1)

        @qjit(keep_intermediate=True)
        @merge_rotations
        @qml.qnode(dev)
        def circuit(x: float):
            qml.RX(x, wires=0)
            qml.RX(0.1, wires=0)
            qml.RX(x**2, wires=0)
            return qml.expval(qml.PauliZ(0))

    >>> circuit(0.54)
    Array(0.5965506257017892, dtype=float64)
    """
    return PassPipelineWrapper(qnode, "merge-rotations")


def ions_decomposition(qnode):  # pragma: nocover
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
        after the ``"QuantumCompilationPass"`` stage via the
        :func:`~.get_compilation_stage` function.

    Args:
        fn (QNode): the QNode to apply the ions-decomposition pass to

    Returns:
        ~.QNode:

    **Example**

    .. code-block:: python

        import pennylane as qml
        from pennylane.devices import NullQubit

        import catalyst
        from catalyst import qjit
        from catalyst.debug import get_compilation_stage


        @qjit(keep_intermediate=True)
        @catalyst.passes.ions_decomposition
        @qml.qnode(NullQubit(2))
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(wires=0))


    >>> print(get_compilation_stage(circuit, stage="QuantumCompilationPass"))
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
    return PassPipelineWrapper(qnode, "ions-decomposition")


def to_ppr(qnode):
    R"""
    Specify that the MLIR compiler pass for converting
    clifford+T gates into Pauli Product Rotation (PPR) gates will be applied.

    Clifford gates are defined as :math:`\exp({iP\tfrac{\pi}{4}})`,
    where :math:`P` is a Pauli word. Non-Clifford gates are defined
    as :math:`\exp({iP\tfrac{\pi}{8}})`.

    For more information on the PPM compilation pass,
    check out the `compilation hub <https://pennylane.ai/compilation/pauli-product-measurement>`__.

    .. note::

        The circuit that generated from this pass are currently
        only not executable in any backend. This pass is only for analysis
        and potential future execution when a suitable backend is available.


    The full list of supported gates are as follows:
    :class:`qml.H <pennylane.H>`,
    :class:`qml.S <pennylane.S>`,
    :class:`qml.T <pennylane.T>`,
    :class:`qml.CNOT <pennylane.CNOT>`,
    :class:`qml.measure() <pennylane.measure>`

    Args:
        fn (QNode): QNode to apply the pass to

    Returns:
        ~.QNode

    **Example**

    In this example the Clifford+T gates will be converted into PPRs.

    .. code-block:: python

        import pennylane as qml
        from catalyst import qjit, measure

        ppm_passes = [("PPM", ["to_ppr"])]

        @qjit(pipelines=ppm_passes, keep_intermediate=True, target="mlir")
        @qml.qnode(qml.device("null.qubit", wires=2))
        def circuit():
            qml.H(0)
            qml.CNOT([0, 1])
            qml.T(0)
            return measure(1)

        print(circuit.mlir_opt)

    Example MLIR Representation:

    .. code-block:: mlir

        . . .
        %2 = qec.ppr ["Z"](4) %1 : !quantum.bit
        %3 = qec.ppr ["X"](4) %2 : !quantum.bit
        %4 = qec.ppr ["Z"](4) %3 : !quantum.bit
        %c_3 = stablehlo.constant dense<1> : tensor<i64>
        %extracted_4 = tensor.extract %c_3[] : tensor<i64>
        %5 = quantum.extract %0[%extracted_4] : !quantum.reg -> !quantum.bit
        %6:2 = qec.ppr ["Z", "X"](4) %4, %5 : !quantum.bit, !quantum.bit
        %7 = qec.ppr ["Z"](-4) %6#0 : !quantum.bit
        %8 = qec.ppr ["X"](-4) %6#1 : !quantum.bit
        %9 = qec.ppr ["Z"](8) %7 : !quantum.bit
        %mres, %out_qubits = qec.ppm ["Z"] %8 : !quantum.bit
        . . .

    """
    return PassPipelineWrapper(qnode, "to_ppr")


def commute_ppr(qnode):
    R"""
    Specify that the MLIR compiler pass for commuting
    Clifford Pauli Product Rotation (PPR) gates, :math:`\exp({iP\tfrac{\pi}{4}})`,
    past non-Clifford PPRs gates, :math:`\exp({iP\tfrac{\pi}{8}})` will be applied,
    where :math:`P` is a Pauli word.

    For more information regarding to PPM,
    see here <https://pennylane.ai/compilation/pauli-product-measurement>

    .. note::

        The `commute_ppr` compilation pass requires that :func:`~.passes.to_ppr` be applied first.

    Args:
        fn (QNode): QNode to apply the pass to.

    Returns:
        ~.QNode

    **Example**

    The ``commute_ppr`` pass must be used in conjunction with :func:`~.passes.to_ppr`
    to first convert gates into PPRs. In this example, the Clifford+T gates in the
    circuit will be converted into PPRs first, then the Clifford PPRs will be
    commuted past the non-Clifford PPR.

    .. code-block:: python

        import pennylane as qml
        from catalyst import qjit, measure

        ppm_passes = [("PPM", ["to_ppr", "commute_ppr"])]

        @qjit(pipelines=ppm_passes, keep_intermediate=True, target="mlir")
        @qml.qnode(qml.device("null.qubit", wires=0))
        def circuit():
            qml.H(0)
            qml.T(0)
            return measure(0)

        print(circuit.mlir_opt)

    Example MLIR Representation:

    .. code-block:: mlir

        . . .
        %2 = qec.ppr ["X"](8) %1 : !quantum.bit
        %3 = qec.ppr ["Z"](4) %2 : !quantum.bit
        %4 = qec.ppr ["X"](4) %3 : !quantum.bit
        %5 = qec.ppr ["Z"](4) %4 : !quantum.bit
        %mres, %out_qubits = qec.ppm ["Z"] %5 : !quantum.bit
        . . .

    """
    return PassPipelineWrapper(qnode, "commute_ppr")


def ppr_to_ppm(qnode):
    R"""
    Specify that the MLIR compiler pass for absorbing Clifford Pauli
    Product Rotation (PPR) operations, :math:`\exp{iP\tfrac{\pi}{4}}`,
    into the final Pauli Product Measurement (PPM) will be applied.

    For more information regarding to PPM,
    check out the `compilation hub <https://pennylane.ai/compilation/pauli-product-measurement>`__.

    Args:
        fn (QNode): QNode to apply the pass to

    Returns:
        ~.QNode

    **Example**

    In this example, the Clifford+T gates will be converted into PPRs first,
    then the Clifford PPRs will be commuted past the non-Clifford PPR,
    and finally the Clifford PPRs will be absorbed into the Pauli Product Measurements.

    .. code-block:: python

        import pennylane as qml
        from catalyst import qjit, measure

        ppm_passes = [("PPM",["to_ppr", "commute_ppr","ppr_to_ppm",])]

        @qjit(pipelines=ppm_passes, keep_intermediate=True, target="mlir")
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit():
            qml.H(0)
            qml.T(0)
            return measure(0)

        print(circuit.mlir_opt)

    Example MLIR Representation:

    .. code-block:: mlir

        . . .
        %2 = qec.ppr ["X"](8) %1 : !quantum.bit
        %mres, %out_qubits = qec.ppm ["X"] %2 : !quantum.bit
        . . .

    """
    return PassPipelineWrapper(qnode, "ppr_to_ppm")
