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
import functools

import pennylane as qml

from catalyst.passes.pass_api import Pass


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
    if not isinstance(qnode, qml.QNode):
        raise TypeError(f"A QNode is expected, got the classical function {qnode}")

    clone = copy.copy(qnode)
    clone.__name__ += "_cancel_inverses"

    @functools.wraps(clone)
    def wrapper(*args, **kwargs):
        pass_pipeline = kwargs.pop("pass_pipeline", [])
        pass_pipeline.append(Pass("remove-chained-self-inverse"))
        kwargs["pass_pipeline"] = pass_pipeline
        return clone(*args, **kwargs)

    return wrapper


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
    if not isinstance(qnode, qml.QNode):
        raise TypeError(f"A QNode is expected, got the classical function {qnode}")

    clone = copy.copy(qnode)
    clone.__name__ += "_merge_rotations"

    @functools.wraps(clone)
    def wrapper(*args, **kwargs):
        pass_pipeline = kwargs.pop("pass_pipeline", [])
        pass_pipeline.append(Pass("merge-rotations"))
        kwargs["pass_pipeline"] = pass_pipeline
        return clone(*args, **kwargs)

    return wrapper


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

    if not isinstance(qnode, qml.QNode):
        raise TypeError(f"A QNode is expected, got the classical function {qnode}")

    @functools.wraps(qnode)
    def wrapper(*args, **kwargs):
        pass_pipeline = kwargs.pop("pass_pipeline", [])
        pass_pipeline.append(Pass("ions-decomposition"))
        kwargs["pass_pipeline"] = pass_pipeline
        return qnode(*args, **kwargs)

    return wrapper
