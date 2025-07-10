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
import json

from catalyst.compiler import _options_to_cli_flags, _quantum_opt
from catalyst.passes.pass_api import PassPipelineWrapper
from catalyst.utils.exceptions import CompileError

# pylint: disable=line-too-long, too-many-lines


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


def disentangle_cnot(qnode):
    """
    Specify that the ``-disentangle-CNOT`` MLIR compiler pass
    for simplifying CNOT gates should be applied to the decorated
    QNode during :func:`~.qjit` compilation.

    Args:
        fn (QNode): the QNode to apply the disentangle CNOT compiler pass to

    Returns:
        ~.QNode:

    **Example**

    .. code-block:: python

        import pennylane as qml
        from catalyst import qjit
        from catalyst.debug import get_compilation_stage
        from catalyst.passes import disentangle_cnot

        dev = qml.device("lightning.qubit", wires=2)

        @qjit(keep_intermediate=True)
        @disentangle_cnot
        @qml.qnode(dev)
        def circuit():
            # first qubit in |1>
            qml.X(0)
            # second qubit in |0>
            # current state : |10>
            qml.CNOT([0,1]) # state after CNOT : |11>
            return qml.state()

    >>> circuit()
    [0.+0.j  0.+0.j  0.+0.j  1.+0.j]

    Note that the QNode will be unchanged in Python, and will continue
    to include keep CNOT gates gates when inspected with Python (for example,
    with :func:`~.draw`).

    To instead view the optimized circuit, the MLIR must be viewed
    after the ``"QuantumCompilationPass"`` stage:

    >>> print(get_compilation_stage(circuit, stage="QuantumCompilationPass"))

    .. code-block:: mlir

        module @circuit {
            func.func public @jit_circuit() -> tensor<4xcomplex<f64>> attributes {llvm.emit_c_interface} {
                %0 = call @circuit_0() : () -> tensor<4xcomplex<f64>>
                return %0 : tensor<4xcomplex<f64>>
            }
            func.func public @circuit_0() -> tensor<4xcomplex<f64>> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
                %c0_i64 = arith.constant 0 : i64
                quantum.device["catalyst/utils/../lib/librtd_lightning.dylib", "LightningSimulator", "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
                %0 = quantum.alloc( 2) : !quantum.reg
                %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
                %out_qubits = quantum.custom "PauliX"() %1 : !quantum.bit
                %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
                %out_qubits_0 = quantum.custom "PauliX"() %2 : !quantum.bit
                %3 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
                %4 = quantum.insert %3[ 1], %out_qubits_0 : !quantum.reg, !quantum.bit
                %5 = quantum.compbasis qreg %4 : !quantum.obs
                %6 = quantum.state %5 : tensor<4xcomplex<f64>>
                quantum.dealloc %4 : !quantum.reg
                quantum.device_release
                return %6 : tensor<4xcomplex<f64>>
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

    It can be seen that the CNOT(0,1) has been replaced with X(1)

    .. code-block:: mlir

        %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
        %out_qubits_0 = quantum.custom "PauliX"() %2 : !quantum.bit
    """
    return PassPipelineWrapper(qnode, "disentangle-CNOT")


def disentangle_swap(qnode):
    """
    Specify that the ``-disentangle-SWAP`` MLIR compiler pass
    for simplifying SWAP gates should be applied to the decorated
    QNode during :func:`~.qjit` compilation.

    Args:
        fn (QNode): the QNode to apply the disentangle SWAP compiler pass to

    Returns:
        ~.QNode:

    **Example**

    .. code-block:: python

        import pennylane as qml
        from pennylane import numpy as np
        from catalyst import qjit
        from catalyst.debug import get_compilation_stage
        from catalyst.passes import disentangle_swap

        dev = qml.device("lightning.qubit", wires=2)

        @qjit(keep_intermediate=True)
        @disentangle_swap
        @qml.qnode(dev)
        def circuit():
            # first qubit in |1>
            qml.X(0)
            # second qubit in non-basis
            qml.RX(np.pi/4,1)
            qml.SWAP([0,1])
            return qml.state()

    >>> circuit()
    [0.+0.j  0.92387953+0.j  0.+0.j  0.-0.38268343j]

    Note that the QNode will be unchanged in Python, and will continue
    to include keep SWAP gates gates when inspected with Python (for example,
    with :func:`~.draw`).

    To instead view the optimized circuit, the MLIR must be viewed
    after the ``"QuantumCompilationPass"`` stage:

    >>> print(get_compilation_stage(circuit, stage="QuantumCompilationPass"))

    .. code-block:: mlir

        module @circuit {
            func.func public @jit_circuit() -> tensor<4xcomplex<f64>> attributes {llvm.emit_c_interface} {
                %0 = call @circuit_0() : () -> tensor<4xcomplex<f64>>
                return %0 : tensor<4xcomplex<f64>>
            }
            func.func public @circuit_0() -> tensor<4xcomplex<f64>> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
                %c0_i64 = arith.constant 0 : i64
                %cst = arith.constant 0.78539816339744828 : f64
                quantum.device["catalyst/utils/../lib/librtd_lightning.dylib", "LightningSimulator", "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
                %0 = quantum.alloc( 2) : !quantum.reg
                %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
                %out_qubits = quantum.custom "PauliX"() %1 : !quantum.bit5
                %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
                %out_qubits_0 = quantum.custom "RX"(%cst) %2 : !quantum.bit
                %out_qubits_1 = quantum.custom "PauliX"() %out_qubits_0 : !quantum.bit
                %out_qubits_2:2 = quantum.custom "CNOT"() %out_qubits_1, %out_qubits : !quantum.bit, !quantum.bit
                %out_qubits_3:2 = quantum.custom "CNOT"() %out_qubits_2#1, %out_qubits_2#0 : !quantum.bit, !quantum.bit
                %3 = quantum.insert %0[ 0], %out_qubits_3#0 : !quantum.reg, !quantum.bit
                %4 = quantum.insert %3[ 1], %out_qubits_3#1 : !quantum.reg, !quantum.bit
                %5 = quantum.compbasis qreg %4 : !quantum.obs
                %6 = quantum.state %5 : tensor<4xcomplex<f64>>
                quantum.dealloc %4 : !quantum.reg
                quantum.device_release
                return %6 : tensor<4xcomplex<f64>>
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

    It can be seen that the SWAP(0,1) has been replaced with the folliowing

    .. code-block:: mlir

        %0 = quantum.alloc( 2) : !quantum.reg
        %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
        %out_qubits = quantum.custom "PauliX"() %1 : !quantum.bit5
        %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
        %out_qubits_0 = quantum.custom "RX"(%cst) %2 : !quantum.bit
        %out_qubits_1 = quantum.custom "PauliX"() %out_qubits_0 : !quantum.bit
        %out_qubits_2:2 = quantum.custom "CNOT"() %out_qubits_1, %out_qubits : !quantum.bit, !quantum.bit
        %out_qubits_3:2 = quantum.custom "CNOT"() %out_qubits_2#1, %out_qubits_2#0 : !quantum.bit, !quantum.bit
    """
    return PassPipelineWrapper(qnode, "disentangle-SWAP")


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
    :class:`qml.X <pennylane.X>`,
    :class:`qml.Y <pennylane.Y>`,
    :class:`qml.Z <pennylane.Z>`,
    :class:`qml.adjoint(qml.S) <pennylane.adjoint(pennylane.S)>`,
    :class:`qml.adjoint(qml.T) <pennylane.adjoint(pennylane.T)>`,
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

        ppm_passes = [("PPM", ["to-ppr"])]

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
    return PassPipelineWrapper(qnode, "to-ppr")


def commute_ppr(qnode=None, *, max_pauli_size=0):
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
        max_pauli_size (int): The maximum size of the Pauli strings after commuting.

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

        ppm_passes = [("PPM", ["to-ppr", "commute-ppr"])]

        @qjit(pipelines=ppm_passes, keep_intermediate=True, target="mlir")
        @qml.qnode(qml.device("null.qubit", wires=1))
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

    If a commutation resulted in a PPR acting on more than
    `max_pauli_size` qubits (here, `max_pauli_size = 2`), that commutation would be skipped.

    .. code-block:: python

        from catalyst.passes import to_ppr, commute_ppr

        pips = [("pipe", ["enforce-runtime-invariants-pipeline"])]

        @qjit(pipelines=pips, target="mlir")
        @to_ppr
        @commute_ppr(max_pauli_size=2)
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit():
            qml.H(0)
            qml.CNOT([1, 2])
            qml.CNOT([0, 1])
            qml.CNOT([0, 2])
            for i in range(3):
                qml.T(i)
            return measure(0), measure(1), measure(2)

        print(circuit.mlir_opt)

    Example MLIR Representation:

    .. code-block:: mlir

        . . .
        %4:2 = qec.ppr ["Z", "X"](4) %2, %3 : !quantum.bit, !quantum.bit
        . . .
        %6:2 = qec.ppr ["X", "Y"](-8) %5, %4#1 : !quantum.bit, !quantum.bit
        . . .
    """

    if qnode is None:
        return functools.partial(commute_ppr, max_pauli_size=max_pauli_size)

    commute_ppr_pass = {"commute_ppr": {"max-pauli-size": max_pauli_size}}
    return PassPipelineWrapper(qnode, commute_ppr_pass)


def merge_ppr_ppm(qnode=None, *, max_pauli_size=0):
    R"""
    Specify that the MLIR compiler pass for absorbing Clifford Pauli
    Product Rotation (PPR) operations, :math:`\exp{iP\tfrac{\pi}{4}}`,
    into the final Pauli Product Measurement (PPM) will be applied.

    For more information regarding to PPM,
    check out the `compilation hub <https://pennylane.ai/compilation/pauli-product-measurement>`__.

    Args:
        fn (QNode): QNode to apply the pass to
        max_pauli_size (int): The maximum size of the Pauli strings after merging.

    Returns:
        ~.QNode

    **Example**

    In this example, the Clifford+T gates will be converted into PPRs first,
    then the Clifford PPRs will be commuted past the non-Clifford PPR,
    and finally the Clifford PPRs will be absorbed into the Pauli Product Measurements.

    .. code-block:: python

        import pennylane as qml
        from catalyst import qjit, measure

        ppm_passes = [("PPM",["to-ppr", "commute-ppr","merge-ppr-ppm",])]

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

    If a merging resulted in a PPM acting on more than
    `max_pauli_size` qubits (here, `max_pauli_size = 2`), that merging would be skipped.

    .. code-block:: python

        from catalyst import measure, qjit
        from catalyst.passes import to_ppr, merge_ppr_ppm

        pips = [("pipe", ["enforce-runtime-invariants-pipeline"])]

        @qjit(pipelines=pips, target="mlir")
        @to_ppr
        @merge_ppr_ppm(max_pauli_size=2)
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit():
            qml.CNOT([1, 2])
            qml.CNOT([0, 1])
            qml.CNOT([0, 2])
            return measure(0), measure(1), measure(2)

        print(circuit.mlir_opt)

    Example MLIR Representation:

    .. code-block:: mlir

        . . .
        %3:2 = qec.ppr ["Z", "X"](4) %1, %2 : !quantum.bit, !quantum.bit
        . . .
        %mres, %out_qubits:2 = qec.ppm ["Y", "Z"](-1) %3#1, %4 : !quantum.bit, !quantum.bit
        . . .

    """
    if qnode is None:
        return functools.partial(merge_ppr_ppm, max_pauli_size=max_pauli_size)

    merge_ppr_ppm_pass = {"merge_ppr_ppm": {"max-pauli-size": max_pauli_size}}
    return PassPipelineWrapper(qnode, merge_ppr_ppm_pass)


def ppr_to_ppm(qnode=None, *, decompose_method="auto-corrected", avoid_y_measure=False):
    R"""Specify that the MLIR compiler passes for decomposing Pauli Product rotations (PPR)
    , :math:`\exp(-iP\theta)`, into Pauli Pauli measurements (PPM) will be applied.

    This pass is used to decompose both non-Clifford and Clifford PPRs into PPMs. The non-Clifford
    PPRs (:math:`\theta = \tfrac{\pi}{8}`) are decomposed first, and then Clifford PPRs
    (:math:`\theta = \tfrac{\pi}{4}`) are decomposed.
    Non-Clifford decomposition can be performed in one of two ways:
    ``"clifford-corrected"`` or ``"auto-corrected"``, by default the latter is used.
    Both methods are based on `A Game of Surface Codes <https://arxiv.org/abs/1808.02892>`__,
    figures 7 and 17(b) respectively.

    Args:
        qnode (QNode, optional): QNode to apply the pass to. If None, returns a decorator.
        decompose_method (str, optional): The method to use for decomposing non-Clifford PPRs.
            Options are ``"auto-corrected"`` and ``"clifford-corrected"``.
            Defaults to ``"auto-corrected"``.
            ``"auto-corrected"`` uses an additional measurement for correction.
            ``"clifford-corrected"`` uses a Clifford rotation for correction.
        avoid_y_measure (bool): Rather than performing a Pauli-Y measurement for Clifford rotations
            (sometimes more costly), a :math:`Y` state (:math:`Y\vert 0 \rangle`) is used instead
            (requires :math:`Y` state preparation). Defaults to ``False``.

    Returns:
        ~.QNode or callable: Returns decorated QNode if qnode is provided,
            otherwise returns a decorator.

    **Example**

    This example shows the sequence of passes that will be applied. The last pass
    will convert the non-Clifford PPR into Pauli Product Measurements.

    .. code-block:: python

        import pennylane as qml
        from catalyst import qjit, measure
        from catalyst.passes import to_ppr, commute_ppr, merge_ppr_ppm, ppr_to_ppm

        pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

        @qjit(pipelines=pipeline, target="mlir")
        @to_ppr
        @commute_ppr
        @merge_ppr_ppm
        @ppr_to_ppm
        @qml.qnode(qml.device("null.qubit", wires=2))
        def circuit():
            qml.H(0)
            qml.T(0)
            qml.CNOT([0, 1])
            return measure(0), measure(1)

        print(circuit.mlir_opt)

    Example MLIR Representation:

    .. code-block:: mlir

        . . .
        %5 = qec.fabricate  zero : !quantum.bit
        %6 = qec.fabricate  magic : !quantum.bit
        %mres, %out_qubits:2 = qec.ppm ["X", "Z"] %1, %6 : !quantum.bit, !quantum.bit
        %mres_0, %out_qubits_1:2 = qec.ppm ["Z", "Y"] %5, %out_qubits#1 : !quantum.bit, !quantum.bit
        %mres_2, %out_qubits_3 = qec.ppm ["X"] %out_qubits_1#1 : !quantum.bit
        %mres_4, %out_qubits_5 = qec.select.ppm(%mres, ["X"], ["Z"]) %out_qubits_1#0 : !quantum.bit
        %7 = arith.xori %mres_0, %mres_2 : i1
        %8 = qec.ppr ["X"](2) %out_qubits#0 cond(%7) : !quantum.bit
        . . .

    """
    passes = {
        "decompose_non_clifford_ppr": {
            "decompose-method": decompose_method,
            "avoid-y-measure": avoid_y_measure,
        },
        "decompose_clifford_ppr": {"avoid-y-measure": avoid_y_measure},
    }

    if qnode is None:
        return functools.partial(
            ppr_to_ppm, decompose_method=decompose_method, avoid_y_measure=avoid_y_measure
        )

    return PassPipelineWrapper(qnode, passes)


def ppm_compilation(
    qnode=None, *, decompose_method="auto-corrected", avoid_y_measure=False, max_pauli_size=0
):
    R"""
    Specify that the MLIR compiler pass for transforming
    clifford+T gates into Pauli Product Measurements (PPM) will be applied.

    This pass combines multiple sub-passes:

    - :func:`~.passes.to_ppr` : Converts gates into Pauli Product Rotations (PPRs)
    - :func:`~.passes.commute_ppr` : Commutes PPRs past non-Clifford PPRs
    - :func:`~.passes.merge_ppr_ppm` : Merges PPRs into Pauli Product Measurements (PPMs)
    - :func:`~.passes.ppr_to_ppm` : Decomposes PPRs into PPMs

    The ``avoid_y_measure`` and ``decompose_method`` arguments are passed
    to the :func:`~.passes.ppr_to_ppm` pass.
    The ``max_pauli_size`` argument is passed to the :func:`~.passes.commute_ppr`
    and :func:`~.passes.merge_ppr_ppm` passes.

    Args:
        qnode (QNode, optional): QNode to apply the pass to. If None, returns a decorator.
        decompose_method (str, optional): The method to use for decomposing non-Clifford PPRs.
            Options are ``"auto-corrected"`` and ``"clifford-corrected"``. Defaults to
            ``"auto-corrected"``.
            ``"auto-corrected"`` uses an additional measurement for correction.
            ``"clifford-corrected"`` uses a Clifford rotation for correction.
        avoid_y_measure (bool): Rather than performing a Pauli-Y measurement for Clifford rotations
            (sometimes more costly), a :math:`Y` state (:math:`Y\vert 0 \rangle`) is used instead
            (requires :math:`Y` state preparation). Defaults to ``False``.
        max_pauli_size (int): The maximum size of the Pauli strings after commuting or merging.
            Defaults to 0 (no limit).

    Returns:
        ~.QNode or callable: Returns decorated QNode if qnode is provided,
        otherwise returns a decorator.

    **Example**

    If a merging resulted in a PPM acting on more than
    ``max_pauli_size`` qubits (here, ``max_pauli_size = 2``), that merging would be skipped.
    However, when decomposed into PPMs, at least one qubit will be applied, so the final
    PPMs will act on at least one additional qubit.

    .. code-block:: python

        import pennylane as qml
        from catalyst import qjit, measure
        from catalyst.passes import ppm_compilation

        pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]
        method = "clifford-corrected"

        @qjit(pipelines=pipeline, target="mlir")
        @ppm_compilation(decompose_method=method, avoid_y_measure=True, max_pauli_size=2)
        @qml.qnode(qml.device("null.qubit", wires=2))
        def circuit():
            qml.CNOT([0, 1])
            qml.CNOT([1, 0])
            qml.adjoint(qml.T)(0)
            qml.T(1)
            return measure(0), measure(1)

        print(circuit.mlir_opt)

    Example MLIR Representation:

    .. code-block:: mlir

        . . .
        %m, %out:3 = qec.ppm ["Z", "Z", "Z"] %1, %2, %4 : !quantum.bit, !quantum.bit, !quantum.bit
        %m_0, %out_1:2 = qec.ppm ["Z", "Y"] %3, %out#2 : !quantum.bit, !quantum.bit
        %m_2, %out_3 = qec.ppm ["X"] %out_1#1 : !quantum.bit
        %m_4, %out_5 = qec.select.ppm(%m, ["X"], ["Z"]) %out_1#0 : !quantum.bit
        %5 = arith.xori %m_0, %m_2 : i1
        %6:2 = qec.ppr ["Z", "Z"](2) %out#0, %out#1 cond(%5) : !quantum.bit, !quantum.bit
        quantum.dealloc_qb %out_5 : !quantum.bit
        quantum.dealloc_qb %out_3 : !quantum.bit
        %7 = quantum.alloc_qb : !quantum.bit
        %8 = qec.fabricate  magic_conj : !quantum.bit
        %m_6, %out_7:2 = qec.ppm ["Z", "Z"] %6#1, %8 : !quantum.bit, !quantum.bit
        %m_8, %out_9:2 = qec.ppm ["Z", "Y"] %7, %out_7#1 : !quantum.bit, !quantum.bit
        %m_10, %out_11 = qec.ppm ["X"] %out_9#1 : !quantum.bit
        %m_12, %out_13 = qec.select.ppm(%m_6, ["X"], ["Z"]) %out_9#0 : !quantum.bit
        %9 = arith.xori %m_8, %m_10 : i1
        %10 = qec.ppr ["Z"](2) %out_7#0 cond(%9) : !quantum.bit
        quantum.dealloc_qb %out_13 : !quantum.bit
        quantum.dealloc_qb %out_11 : !quantum.bit
        %m_14, %out_15:2 = qec.ppm ["Z", "Z"] %6#0, %10 : !quantum.bit, !quantum.bit
        %from_elements = tensor.from_elements %m_14 : tensor<i1>
        %m_16, %out_17 = qec.ppm ["Z"] %out_15#1 : !quantum.bit
        . . .

    """
    passes = {
        "ppm-compilation": {
            "decompose-method": decompose_method,
            "avoid-y-measure": avoid_y_measure,
            "max-pauli-size": max_pauli_size,
        }
    }

    if qnode is None:
        return functools.partial(
            ppm_compilation,
            decompose_method=decompose_method,
            avoid_y_measure=avoid_y_measure,
            max_pauli_size=max_pauli_size,
        )

    return PassPipelineWrapper(qnode, passes)


def get_ppm_specs(fn):
    R"""
    This function returns following PPM specs in a dictionary:
        - Pi/4 PPR (count the number of clifford PPRs)
        - Pi/8 PPR (count the number of non-clifford PPRs)
        - Pi/2 PPR (count the number of classical PPRs)
        - Max weight for pi/8 PPRs
        - Max weight for pi/4 PPRs
        - Max weight for pi/2 PPRs
        - Number of logical qubits
        - Number of PPMs

    PPM Specs are returned after the last PPM compilation pass is run.

    When there is control flow, this function can count the above statistics inside for loops with
    a statically known number of iterations. For all other cases, including dynamically sized for
    loops, and any conditionals and while loops, this pass exits with failure.

    Args:
        fn (QJIT): qjit-decorated function for which ppm_specs need to be printed

    Returns:
        dict : Returns a Python dictionary containing PPM specs of all functions in QJIT

    **Example**

    .. code-block:: python

        import pennylane as qml
        from catalyst import qjit, measure, for_loop
        from catalyst.passes import get_ppm_specs, ppm_compilation

        pipe = [("pipe", ["enforce-runtime-invariants-pipeline"])]
        device = qml.device("lightning.qubit", wires=2)

        @qjit(pipelines=pipe, target="mlir")
        @ppm_compilation
        @qml.qnode(device)
        def circuit():
            qml.H(0)
            qml.CNOT([0,1])
            @for_loop(0,10,1)
            def loop(i):
                qml.T(1)
            loop()
            return measure(0), measure(1)

        ppm_specs = get_ppm_specs(circuit)
        print(ppm_specs)

    Example PPM Specs:

    .. code-block:: pycon

        . . .
        {
            'circuit_0': {
                        'max_weight_pi2': 2,
                        'num_logical_qubits': 2,
                        'num_of_ppm': 44,
                        'num_pi2_gates': 16
                    },
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
        pass_list.append("ppm-specs")

        new_options = _options_to_cli_flags(new_options)
        raw_result = _quantum_opt(*new_options, [], stdin=str(fn.mlir_module))

        try:
            return json.loads(
                raw_result[: raw_result.index("module")]
            )  # remove MLIR starting with substring "module..."
        except Exception as e:  # pragma: nocover
            raise CompileError(
                "Invalid json format encountered in get_ppm_specs. "
                f" but got {raw_result[: raw_result.index('module')]}"
            ) from e

    else:
        raise NotImplementedError("PPM passes only support AOT (Ahead-Of-Time) compilation mode.")
