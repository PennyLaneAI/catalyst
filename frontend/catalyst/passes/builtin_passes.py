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
    Specify that the ``-cancel-inverses`` MLIR compiler pass
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
        after the ``"QuantumCompilationStage"`` stage via the
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
    return PassPipelineWrapper(qnode, "cancel-inverses")


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
    after the ``"QuantumCompilationStage"`` stage:

    >>> print(get_compilation_stage(circuit, stage="QuantumCompilationStage"))

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
    after the ``"QuantumCompilationStage"`` stage:

    >>> print(get_compilation_stage(circuit, stage="QuantumCompilationStage"))

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
    :class:`qml.Rot <pennylane.Rot>`,
    :class:`qml.CRot <pennylane.CRot>`,
    :class:`qml.MultiRZ <pennylane.MultiRZ>`.


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


def decompose_lowering(qnode):
    """
    Specify that the ``-decompose-lowering`` MLIR compiler pass
    for applying the compiled decomposition rules to the QNode
    recursively.

    Args:
        fn (QNode): the QNode to apply the cancel inverses compiler pass to

    Returns:
        ~.QNode:

    **Example**
        // TODO: add example here

    """
    return PassPipelineWrapper(qnode, "decompose-lowering")  # pragma: no cover


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
        after the ``"QuantumCompilationStage"`` stage via the
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
    return PassPipelineWrapper(qnode, "ions-decomposition")


def gridsynth(qnode=None, *, epsilon=1e-4, ppr_basis=False):
    R"""
    A quantnum compilation pass to discretize
    single-qubit RZ and PhaseShift gates into the Clifford+T basis or the PPR basis using the Ross-Selinger Gridsynth algorithm.
    Reference: https://arxiv.org/abs/1403.2975


    .. note::

        The actual discretization is only performed during execution time.

    Args:
        qnode (QNode): the QNode to apply the gridsynth compiler pass to
        epsilon (float): The maximum permissible operator norm error per rotation gate. Defaults to ``1e-4``.
        ppr_basis (bool): If true, decompose directly to Pauli Product Rotations (PPRs) in QEC dialect. Defaults to ``False``

    Returns:
        :class:`QNode <pennylane.QNode>`

    .. note::

        The circuit generated from this pass with ``ppr_basis=True`` are currently not executable on any backend.
        This is only for analysis with the ``null.qubit`` device and potential future execution
        when a suitable backend is available.

    **Example**

    In this example the RZ gate will be converted into a new function, which
    calls the discretization at execution time.

    .. code-block:: python

        import pennylane as qml
        from catalyst import qjit
        from catalyst.passes import gridsynth

        pipe = [("pipe", ["quantum-compilation-stage"])]

        @qjit(pipelines=pipe, target="mlir")
        @gridsynth
        @qml.qnode(qml.device("null.qubit", wires=1))
        def circuit():
            qml.RZ(x, wires=0)
            return qml.probs()

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
    if qnode is None:
        return functools.partial(gridsynth, epsilon=epsilon, ppr_basis=ppr_basis)

    gridsynth_pass = {"gridsynth": {"epsilon": epsilon, "ppr_basis": ppr_basis}}
    return PassPipelineWrapper(qnode, gridsynth_pass)


def to_ppr(qnode):
    R"""A quantum compilation pass that converts Clifford+T gates into Pauli Product Rotation (PPR)
    gates.

    .. note::

        For improved integration with the PennyLane frontend, including inspectability with
        :func:`pennylane.specs`, please use :func:`pennylane.transforms.to_ppr`.

    Clifford gates are defined as :math:`\exp(-{iP\tfrac{\pi}{4}})`, where :math:`P` is a Pauli word.
    Non-Clifford gates are defined as :math:`\exp(-{iP\tfrac{\pi}{8}})`.

    For more information on the PPM compilation pass, check out the
    `compilation hub <https://pennylane.ai/compilation/pauli-product-measurement>`__.

    .. note::

        The circuits that generated from this pass are currently not executable on any backend.
        This pass is only for analysis with the ``null.qubit`` device and potential future execution
        when a suitable backend is available.

    The full list of supported gates and operations are
    ``qml.H``,
    ``qml.S``,
    ``qml.T``,
    ``qml.X``,
    ``qml.Y``,
    ``qml.Z``,
    ``qml.adjoint(qml.S)``,
    ``qml.adjoint(qml.T)``,
    ``qml.CNOT``,
    ``qml.PauliRot``, and
    ``catalyst.measure``.

    Args:
        fn (QNode): QNode to apply the pass to

    Returns:
        :class:`QNode <pennylane.QNode>`

    **Example**

    In this example the Clifford+T gates will be converted into PPRs.

    .. code-block:: python

        import pennylane as qml
        import catalyst

        p = [("my_pipe", ["quantum-compilation-stage"])]

        @qml.qjit(pipelines=p, target="mlir")
        @catalyst.passes.to_ppr
        @qml.qnode(qml.device("null.qubit", wires=2))
        def circuit():
            qml.H(0)
            qml.CNOT([0, 1])
            qml.T(0)
            return

        print(circuit.mlir_opt)

    Because Catalyst does not currently support execution of Pauli-based computation operations, we
    must halt the pipeline after ``quantum-compilation-stage``. This ensures that only the quantum
    passes will be applied to the initial MLIR, without attempting to further compile for execution.

    Example MLIR Representation:

    .. code-block:: mlir

        . . .
        %2 = qec.ppr ["Z"](4) %1 : !quantum.bit
        %3 = qec.ppr ["X"](4) %2 : !quantum.bit
        %4 = qec.ppr ["Z"](4) %3 : !quantum.bit
        %5 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
        %6:2 = qec.ppr ["Z", "X"](4) %4, %5 : !quantum.bit, !quantum.bit
        %7 = qec.ppr ["Z"](-4) %6#0 : !quantum.bit
        %8 = qec.ppr ["X"](-4) %6#1 : !quantum.bit
        %9 = qec.ppr ["Z"](8) %7 : !quantum.bit
        . . .

    """
    return PassPipelineWrapper(qnode, "to-ppr")


def commute_ppr(qnode=None, *, max_pauli_size=0):
    R"""
    A quantum compilation pass that commutes Clifford Pauli product rotation (PPR) gates,
    :math:`\exp(-{iP\tfrac{\pi}{4}})`, past non-Clifford PPRs gates,
    :math:`\exp(-{iP\tfrac{\pi}{8}})`, where :math:`P` is a Pauli word.

    .. note::

        For improved integration with the PennyLane frontend, including inspectability with
        :func:`pennylane.specs`, please use :func:`pennylane.transforms.commute_ppr`.

    For more information on PPRs, check out the
    `Compilation Hub <https://pennylane.ai/compilation/pauli-product-measurement>`_.

    .. note::

        The ``commute_ppr`` compilation pass requires that :func:`~.passes.to_ppr` be applied first.

    Args:
        fn (QNode): QNode to apply the pass to.
        max_pauli_size (int): The maximum size of the Pauli strings after commuting.

    Returns:
        :class:`QNode <pennylane.QNode>`

    **Example**

    The ``commute_ppr`` pass must be used in conjunction with :func:`~.passes.to_ppr`
    to first convert gates into PPRs. In this example, the Clifford+T gates in the
    circuit will be converted into PPRs first, then the Clifford PPRs will be
    commuted past the non-Clifford PPR.

    .. code-block:: python

        import pennylane as qml
        import catalyst

        p = [("my_pipe", ["quantum-compilation-stage"])]

        @qml.qjit(pipelines=p, target="mlir")
        @catalyst.passes.commute_ppr
        @catalyst.passes.to_ppr
        @qml.qnode(qml.device("null.qubit", wires=1))
        def circuit():
            qml.H(0)
            qml.T(0)
            return

        print(circuit.mlir_opt)

    Because Catalyst does not currently support execution of Pauli-based computation operations, we
    must halt the pipeline after ``quantum-compilation-stage``. This ensures that only the quantum
    passes will be applied to the initial MLIR, without attempting to further compile for execution.

    Example MLIR Representation:

    .. code-block:: mlir

        . . .
        %2 = qec.ppr ["X"](8) %1 : !quantum.bit
        %3 = qec.ppr ["Z"](4) %2 : !quantum.bit
        %4 = qec.ppr ["X"](4) %3 : !quantum.bit
        %5 = qec.ppr ["Z"](4) %4 : !quantum.bit
        %6 = quantum.insert %0[ 0], %5 : !quantum.reg, !quantum.bit
        . . .

    If a commutation resulted in a PPR acting on more than
    ``max_pauli_size`` qubits (here, ``max_pauli_size = 2``), that commutation would be skipped.

    .. code-block:: python

        @qml.qjit(pipelines=p, target="mlir")
        @catalyst.passes.commute_ppr(max_pauli_size=2)
        @catalyst.passes.to_ppr
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit():
            qml.H(0)
            qml.CNOT([1, 2])
            qml.CNOT([0, 1])
            qml.CNOT([0, 2])
            for i in range(3):
                qml.T(i)
            return

        print(circuit.mlir_opt)

    Example MLIR Representation:

    .. code-block:: mlir

        . . .
        %4:2 = qec.ppr ["Z", "X"](4) %2, %3 : !quantum.bit, !quantum.bit
        %5 = qec.ppr ["X"](8) %1 : !quantum.bit
        %6:2 = qec.ppr ["X", "Y"](-8) %5, %4#1 : !quantum.bit, !quantum.bit
        %7 = qec.ppr ["X"](-4) %6#1 : !quantum.bit
        %8:2 = qec.ppr ["X", "Z"](8) %6#0, %4#0 : !quantum.bit, !quantum.bit
        %9 = qec.ppr ["Z"](4) %8#0 : !quantum.bit
        %10 = qec.ppr ["X"](4) %9 : !quantum.bit
        %11 = qec.ppr ["Z"](4) %10 : !quantum.bit
        %12 = qec.ppr ["Z"](-4) %8#1 : !quantum.bit
        %13:2 = qec.ppr ["Z", "X"](4) %11, %12 : !quantum.bit, !quantum.bit
        %14 = qec.ppr ["X"](-4) %13#1 : !quantum.bit
        %15 = qec.ppr ["Z"](-4) %13#0 : !quantum.bit
        %16:2 = qec.ppr ["Z", "X"](4) %15, %7 : !quantum.bit, !quantum.bit
        %17 = qec.ppr ["Z"](-4) %16#0 : !quantum.bit
        %18 = qec.ppr ["X"](-4) %16#1 : !quantum.bit
        . . .
    """

    if qnode is None:
        return functools.partial(commute_ppr, max_pauli_size=max_pauli_size)

    commute_ppr_pass = {"commute_ppr": {"max-pauli-size": max_pauli_size}}
    return PassPipelineWrapper(qnode, commute_ppr_pass)


def merge_ppr_ppm(qnode=None, *, max_pauli_size=0):
    R"""
    A quantum compilation pass that absorbs Clifford Pauli product rotation (PPR) operations,
    :math:`\exp{-iP\tfrac{\pi}{4}}`, into the final Pauli product measurements (PPMs).

    .. note::

        For improved integration with the PennyLane frontend, including inspectability with
        :func:`pennylane.specs`, please use :func:`pennylane.transforms.merge_ppr_ppm`.

    For more information on PPRs and PPMs, check out
    the `Compilation Hub <https://pennylane.ai/compilation/pauli-product-measurement>`_.

    Args:
        fn (QNode): QNode to apply the pass to
        max_pauli_size (int): The maximum size of the Pauli strings after merging.

    Returns:
        :class:`QNode <pennylane.QNode>`

    **Example**

    In this example, the Clifford+T gates will be converted into PPRs first,
    then the Clifford PPRs will be commuted past the non-Clifford PPR,
    and finally the Clifford PPRs will be absorbed into the Pauli Product Measurements.

    .. code-block:: python

        import pennylane as qml
        import catalyst

        p = [("my_pipe", ["quantum-compilation-stage"])]

        @qml.qjit(pipelines=p, target="mlir")
        @catalyst.passes.merge_ppr_ppm
        @catalyst.passes.commute_ppr
        @catalyst.passes.to_ppr
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit():
            qml.H(0)
            qml.T(0)
            return catalyst.measure(0), catalyst.measure(1)

        print(circuit.mlir_opt)

    Because Catalyst does not currently support execution of Pauli-based computation operations, we
    must halt the pipeline after ``quantum-compilation-stage``. This ensures that only the quantum
    passes will be applied to the initial MLIR, without attempting to further compile for execution.

    Example MLIR Representation:

    .. code-block:: mlir

        . . .
        %2 = qec.ppr ["X"](8) %1 : !quantum.bit
        %mres, %out_qubits = qec.ppm ["X"] %2 : i1, !quantum.bit
        %from_elements = tensor.from_elements %mres : tensor<i1>
        %3 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
        %mres_0, %out_qubits_1 = qec.ppm ["Z"] %3 : i1, !quantum.bit
        . . .

    If a merging resulted in a PPM acting on more than
    `max_pauli_size` qubits (here, `max_pauli_size = 2`), that merging would be skipped.

    .. code-block:: python

        p = [("my_pipe", ["quantum-compilation-stage"])]

        @qml.qjit(pipelines=p, target="mlir")
        @catalyst.passes.merge_ppr_ppm(max_pauli_size=2)
        @catalyst.passes.commute_ppr
        @catalyst.passes.to_ppr
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit():
            qml.CNOT([1, 2])
            qml.CNOT([0, 1])
            qml.CNOT([0, 2])
            return catalyst.measure(0), catalyst.measure(1)

        print(circuit.mlir_opt)

    Example MLIR Representation:

    .. code-block:: mlir

        . . .
        %mres, %out_qubits:2 = qec.ppm ["Z", "Z"] %1, %3 : i1, !quantum.bit, !quantum.bit
        %mres_0, %out_qubits_1 = qec.ppm ["Z"] %out_qubits#1 : i1, !quantum.bit
        . . .

    """
    if qnode is None:
        return functools.partial(merge_ppr_ppm, max_pauli_size=max_pauli_size)

    merge_ppr_ppm_pass = {"merge_ppr_ppm": {"max-pauli-size": max_pauli_size}}
    return PassPipelineWrapper(qnode, merge_ppr_ppm_pass)


def ppr_to_ppm(qnode=None, *, decompose_method="pauli-corrected", avoid_y_measure=False):
    R"""
    A quantum compilation pass that decomposes Pauli product rotations (PPRs),
    :math:`P(\theta) = \exp(-iP\theta)`, into Pauli product measurements (PPMs).

    .. note::

        For improved integration with the PennyLane frontend, including inspectability with
        :func:`pennylane.specs`, please use :func:`pennylane.transforms.ppr_to_ppm`.

    This pass is used to decompose both non-Clifford and Clifford PPRs into PPMs. The non-Clifford
    PPRs (:math:`\theta = \tfrac{\pi}{8}`) are decomposed first, then Clifford PPRs
    (:math:`\theta = \tfrac{\pi}{4}`) are decomposed.

    Args:
        qnode (QNode): QNode to apply the pass to.
        decompose_method (str, optional): The method to use for decomposing non-Clifford PPRs.
            Options are ``"pauli-corrected"``, ``"auto-corrected"``, and ``"clifford-corrected"``.
            Defaults to ``"pauli-corrected"``.
            ``"pauli-corrected"`` uses a reactive measurement for correction.
            ``"auto-corrected"`` uses an additional measurement for correction.
            ``"clifford-corrected"`` uses a Clifford rotation for correction.

        avoid_y_measure (bool): Rather than performing a Pauli-Y measurement for Clifford rotations
            (sometimes more costly), a :math:`Y` state (:math:`Y\vert 0 \rangle`) is used instead
            (requires :math:`Y` state preparation).
            This is currently only supported when using the ``"clifford-corrected"`` and ``"pauli-corrected"`` decomposition method.
            Defaults to ``False``.

    Returns:
        :class:`QNode <pennylane.QNode>`

    **Example**

    This example shows the sequence of passes that will be applied. The last pass
    will convert the non-Clifford PPR into Pauli Product Measurements.

    .. code-block:: python

        import pennylane as qml
        import catalyst

        p = [("my_pipe", ["quantum-compilation-stage"])]

        @qml.qjit(pipelines=p, target="mlir")
        @catalyst.passes.ppr_to_ppm(decompose_method="auto-corrected")
        @catalyst.passes.merge_ppr_ppm
        @catalyst.passes.commute_ppr
        @catalyst.passes.to_ppr
        @qml.qnode(qml.device("null.qubit", wires=2))
        def circuit():
            qml.H(0)
            qml.T(0)
            qml.CNOT([0, 1])
            return catalyst.measure(0), catalyst.measure(1)

        print(circuit.mlir_opt)

    Because Catalyst does not currently support execution of Pauli-based computation operations, we
    must halt the pipeline after ``quantum-compilation-stage``. This ensures that only the quantum
    passes will be applied to the initial MLIR, without attempting to further compile for execution.

    Example MLIR Representation:

    .. code-block:: mlir

        . . .
        %3 = qec.fabricate  magic : !quantum.bit
        %mres, %out_qubits:2 = qec.ppm ["X", "Z"] %1, %3 : i1, !quantum.bit, !quantum.bit
        %mres_0, %out_qubits_1:2 = qec.ppm ["Z", "Y"](-1) %out_qubits#1, %2 : i1, !quantum.bit, !quantum.bit
        %mres_2, %out_qubits_3 = qec.ppm ["X"] %out_qubits_1#0 : i1, !quantum.bit
        %mres_4, %out_qubits_5 = qec.select.ppm(%mres, ["X"], ["Z"]) %out_qubits_1#1 : i1, !quantum.bit
        %4 = arith.xori %mres_0, %mres_2 : i1
        %5 = qec.ppr ["X"](2) %out_qubits#0 cond(%4) : !quantum.bit
        . . .

    """
    passes = {
        "ppr_to_ppm": {
            "decompose-method": decompose_method,
            "avoid-y-measure": avoid_y_measure,
        },
    }

    if qnode is None:
        return functools.partial(
            ppr_to_ppm, decompose_method=decompose_method, avoid_y_measure=avoid_y_measure
        )

    return PassPipelineWrapper(qnode, passes)


def ppm_compilation(
    qnode=None, *, decompose_method="pauli-corrected", avoid_y_measure=False, max_pauli_size=0
):
    R"""
    A quantum compilation pass that transforms Clifford+T gates into Pauli product measurements
    (PPMs).

    .. note::

        For improved integration with the PennyLane frontend, including inspectability with
        :func:`pennylane.specs`, please use :func:`pennylane.transforms.ppm_compilation`.

    This pass combines multiple sub-passes:

    - :func:`~.passes.to_ppr` : Converts gates into Pauli Product Rotations (PPRs)
    - :func:`~.passes.commute_ppr` : Commutes PPRs past non-Clifford PPRs
    - :func:`~.passes.merge_ppr_ppm` : Merges PPRs into Pauli Product Measurements (PPMs)
    - :func:`~.passes.ppr_to_ppm` : Decomposes PPRs into PPMs

    The ``avoid_y_measure`` and ``decompose_method`` arguments are passed to the
    :func:`~.passes.ppr_to_ppm` pass. The ``max_pauli_size`` argument is passed to the
    :func:`~.passes.commute_ppr` and :func:`~.passes.merge_ppr_ppm` passes.

    For more information on PPRs and PPMs, check out
    the `Compilation Hub <https://pennylane.ai/compilation/pauli-product-measurement>`_.

    Args:
        qnode (QNode, optional): QNode to apply the pass to. If None, returns a decorator.
        decompose_method (str, optional): The method to use for decomposing non-Clifford PPRs.
            Options are ``"pauli-corrected"``, ``"auto-corrected"`` and ``"clifford-corrected"``. Defaults to
            ``"pauli-corrected"``.
            ``"pauli-corrected"`` uses a reactive measurement for correction.
            ``"auto-corrected"`` uses an additional measurement for correction.
            ``"clifford-corrected"`` uses a Clifford rotation for correction.
        avoid_y_measure (bool): Rather than performing a Pauli-Y measurement for Clifford rotations
            (sometimes more costly), a :math:`Y` state (:math:`Y\vert 0 \rangle`) is used instead
            (requires :math:`Y` state preparation). Defaults to ``False``.
        max_pauli_size (int): The maximum size of the Pauli strings after commuting or merging.
            Defaults to 0 (no limit).

    Returns:
        :class:`QNode <pennylane.QNode>`

    **Example**

    If a merging resulted in a PPM acting on more than
    ``max_pauli_size`` qubits (here, ``max_pauli_size = 2``), that merging would be skipped.
    However, when decomposed into PPMs, at least one qubit will be applied, so the final
    PPMs will act on at least one additional qubit.

    .. code-block:: python

        import pennylane as qml
        import catalyst

        p = [("my_pipe", ["quantum-compilation-stage"])]
        method = "clifford-corrected"

        @qml.qjit(pipelines=p, target="mlir")
        @catalyst.passes.ppm_compilation(decompose_method=method, max_pauli_size=2)
        @qml.qnode(qml.device("null.qubit", wires=2))
        def circuit():
            qml.CNOT([0, 1])
            qml.CNOT([1, 0])
            qml.adjoint(qml.T)(0)
            qml.T(1)
            return catalyst.measure(0), catalyst.measure(1)

        print(circuit.mlir_opt)

    Because Catalyst does not currently support execution of Pauli-based computation operations, we
    must halt the pipeline after ``quantum-compilation-stage``. This ensures that only the quantum
    passes will be applied to the initial MLIR, without attempting to further compile for execution.

    Example MLIR Representation:

    .. code-block:: mlir

        . . .
        %3 = qec.fabricate  magic : !quantum.bit
        %mres, %out_qubits:3 = qec.ppm ["Z", "Z", "Z"] %1, %2, %3 : i1, !quantum.bit, !quantum.bit, !quantum.bit
        %4 = quantum.alloc_qb : !quantum.bit
        %mres_0, %out_qubits_1:3 = qec.ppm ["Z", "Z", "Y"](-1) %out_qubits#0, %out_qubits#1, %4 cond(%mres) : i1, !quantum.bit, !quantum.bit, !quantum.bit
        %mres_2, %out_qubits_3 = qec.ppm ["X"] %out_qubits_1#2 cond(%mres) : i1, !quantum.bit
        %5 = arith.xori %mres_0, %mres_2 : i1
        %6:2 = qec.ppr ["Z", "Z"](2) %out_qubits_1#0, %out_qubits_1#1 cond(%5) : !quantum.bit, !quantum.bit
        quantum.dealloc_qb %out_qubits_3 : !quantum.bit
        %mres_4, %out_qubits_5 = qec.ppm ["X"] %out_qubits#2 : i1, !quantum.bit
        %7:2 = qec.ppr ["Z", "Z"](2) %6#0, %6#1 cond(%mres_4) : !quantum.bit, !quantum.bit
        quantum.dealloc_qb %out_qubits_5 : !quantum.bit
        %8 = qec.fabricate  magic_conj : !quantum.bit
        %mres_6, %out_qubits_7:2 = qec.ppm ["Z", "Z"] %7#1, %8 : i1, !quantum.bit, !quantum.bit
        %9 = quantum.alloc_qb : !quantum.bit
        %mres_8, %out_qubits_9:2 = qec.ppm ["Z", "Y"](-1) %out_qubits_7#0, %9 cond(%mres_6) : i1, !quantum.bit, !quantum.bit
        %mres_10, %out_qubits_11 = qec.ppm ["X"] %out_qubits_9#1 cond(%mres_6) : i1, !quantum.bit
        %10 = arith.xori %mres_8, %mres_10 : i1
        %11 = qec.ppr ["Z"](2) %out_qubits_9#0 cond(%10) : !quantum.bit
        quantum.dealloc_qb %out_qubits_11 : !quantum.bit
        %mres_12, %out_qubits_13 = qec.ppm ["X"] %out_qubits_7#1 : i1, !quantum.bit
        %12 = qec.ppr ["Z"](2) %11 cond(%mres_12) : !quantum.bit
        quantum.dealloc_qb %out_qubits_13 : !quantum.bit
        %mres_14, %out_qubits_15:2 = qec.ppm ["Z", "Z"] %7#0, %12 : i1, !quantum.bit, !quantum.bit
        %mres_16, %out_qubits_17 = qec.ppm ["Z"] %out_qubits_15#1 : i1, !quantum.bit
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


def ppm_specs(fn):
    R"""This function returns following Pauli product rotation (PPR) and Pauli product measurement (PPM)
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

        import pennylane as qml
        import catalyst

        p = [("my_pipe", ["quantum-compilation-stage"])]
        device = qml.device("lightning.qubit", wires=2)

        @qml.qjit(pipelines=p, target="mlir")
        @catalyst.passes.ppm_compilation
        @qml.qnode(device)
        def circuit():
            qml.H(0)
            qml.CNOT([0,1])

            @catalyst.for_loop(0,10,1)
            def loop(i):
                qml.T(1)

            loop()
            return catalyst.measure(0), catalyst.measure(1)

        ppm_specs = catalyst.passes.ppm_specs(circuit)
        print(ppm_specs)

    Because Catalyst does not currently support execution of Pauli-based computation operations, we
    must halt the pipeline after ``quantum-compilation-stage``. This ensures that only the quantum
    passes will be applied to the initial MLIR, without attempting to further compile for execution.

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


def reduce_t_depth(qnode):
    R"""
    A quantum compilation pass that reduces the depth and count of non-Clifford Pauli product
    rotation (PPR, :math:`P(\theta) = \exp(-iP\theta)`) operators (e.g., ``T`` gates) by commuting
    PPRs in adjacent layers and merging compatible ones (a layer comprises PPRs that mutually
    commute). For more details, see Figure 6 of
    `A Game of Surface Codes <https://arXiv:1808.02892v3>`_.

    .. note::

        The circuits that generated from this pass are currently not executable on any backend.
        This pass is only for analysis with the ``null.qubit`` device and potential future execution
        when a suitable backend is available.

    Args:
        qnode (QNode): QNode to apply the pass to.

    Returns:
        :class:`QNode <pennylane.QNode>`: Returns decorated QNode.

    **Example**

    In the example below, after performing the :func:`catalyst.passes.to_ppr` and
    :func:`catalyst.passes.merge_ppr_ppm` passes, the circuit contains a depth of four of
    non-Clifford PPRs. Subsequently applying the ``reduce_t_depth`` pass will move PPRs around via
    commutation, resulting in a circuit with a smaller PPR depth.

    Specifically, in the circuit below, the Pauli-:math:`X` PPR (:math:`\exp(iX\tfrac{\pi}{8})`) on
    qubit Q1 will be moved to the first layer, which results in a depth of three non-Clifford PPRs.

    .. code-block:: python

        import pennylane as qml
        import catalyst

        p = [("my_pipe", ["quantum-compilation-stage"])]

        @qml.qjit(pipelines=p, target="mlir")
        @catalyst.passes.reduce_t_depth
        @catalyst.passes.merge_ppr_ppm
        @catalyst.passes.commute_ppr
        @catalyst.passes.to_ppr
        @qml.qnode(qml.device("null.qubit", wires=3))
        def circuit():
            n = 3
            for i in range(n):
                qml.H(wires=i)
                qml.S(wires=i)
                qml.CNOT(wires=[i, (i + 1) % n])
                qml.T(wires=i)
                qml.H(wires=i)
                qml.T(wires=i)

            return catalyst.measure(0), catalyst.measure(1), catalyst.measure(2)

    Because Catalyst does not currently support execution of Pauli-based computation operations, we
    must halt the pipeline after ``quantum-compilation-stage``. This ensures that only the quantum
    passes will be applied to the initial MLIR, without attempting to further compile for execution.

    >>> print(circuit.mlir_opt)

    . . .
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    // layer 1
    %3 = qec.ppr ["X"](8) %1 : !quantum.bit
    %4 = qec.ppr ["X"](8) %2 : !quantum.bit

    // layer 2
    %5:2 = qec.ppr ["Y", "X"](8) %3, %4 : !quantum.bit, !quantum.bit
    %6 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
    %7:3 = qec.ppr ["X", "Y", "X"](8) %5#0, %5#1, %6 : !quantum.bit, !quantum.bit, !quantum.bit
    %8 = qec.ppr ["X"](8) %7#2 : !quantum.bit

    // layer 3
    %9:3 = qec.ppr ["X", "X", "Y"](8) %7#0, %7#1, %8 : !quantum.bit, !quantum.bit, !quantum.bit

    %mres, %out_qubits:3 = qec.ppm ["X", "X", "Y"] %9#0, %9#1, %9#2 : i1, !quantum.bit, !quantum.bit, !quantum.bit
    %mres_0, %out_qubits_1:3 = qec.ppm ["Y", "X", "X"] %out_qubits#0, %out_qubits#1, %out_qubits#2 : i1, !quantum.bit, !quantum.bit, !quantum.bit
    %mres_2, %out_qubits_3:3 = qec.ppm ["X", "Y", "X"] %out_qubits_1#0, %out_qubits_1#1, %out_qubits_1#2 : i1, !quantum.bit, !quantum.bit, !quantum.bit
    . . .
    """

    return PassPipelineWrapper(qnode, "reduce-t-depth")


def ppr_to_mbqc(qnode):
    R"""
    Specify that the MLIR compiler pass for lowering Pauli Product Rotations (PPR)
    and Pauli Product Measurements (PPM) to a measurement-based quantum computing
    (MBQC) style circuit will be applied.

    This pass replaces QEC operations (``qec.ppr`` and ``qec.ppm``) with a
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
        after :func:`~.passes.to_ppr` and/or :func:`~.passes.commute_ppr` and/or
        :func:`~.passes.merge_ppr_ppm`.

    Args:
        fn (QNode): QNode to apply the pass to.

    Returns:
        :class:`QNode <pennylane.QNode>`

    **Example**

    Convert a simple Clifford+T circuit to PPRs, then lower them to an
    MBQC-style circuit. Note that this pass should be applied before
    :func:`~.passes.ppr_to_ppm` since it requires the actual PPR/PPM operations.

    .. code-block:: python

        import pennylane as qml
        import catalyst

        p = [("my_pipe", ["quantum-compilation-stage"])]

        @qml.qjit(pipelines=p, target="mlir", keep_intermediate=True)
        @catalyst.passes.ppr_to_mbqc
        @catalyst.passes.to_ppr
        @qml.qnode(qml.device("null.qubit", wires=2))
        def circuit():
            qml.H(0)
            qml.CNOT([0, 1])
            return

        print(circuit.mlir_opt)

    Because Catalyst does not currently support execution of Pauli-based computation operations, we
    must halt the pipeline after ``quantum-compilation-stage``. This ensures that only the quantum
    passes will be applied to the initial MLIR, without attempting to further compile for execution.

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
    return PassPipelineWrapper(qnode, "ppr-to-mbqc")


# This pass is already covered via applying by pass
# `qml.transform(pass_name="decompose-arbitrary-ppr")` in Pennylane.
def decompose_arbitrary_ppr(qnode):  # pragma: nocover
    R"""
    Specify that the MLIR compiler pass for decomposing arbitrary Pauli product rotations (PPR)
    operations will be applied. This will decompose into a collection of PPRs, PPMs and
    a single-qubit arbitrary PPR in the Z basis. For more details, see Figure 13(d)
    in `arXiv:2211.15465 <https://arxiv.org/abs/2211.15465>`_.

    .. note::

        For improved integration with the PennyLane frontend, including inspectability with
        :func:`pennylane.specs`, please use :func:`pennylane.transforms.decompose_arbitrary_ppr`.

        The ``decompose_arbitrary_ppr`` compilation pass requires that :func:`~.passes.to_ppr` be
        applied first.

    Args:
        qnode (QNode): QNode to apply the pass to.

    Returns:
        :class:`QNode <pennylane.QNode>`

    **Example**

    .. code-block:: python

        import pennylane as qml
        import catalyst

        p = [("my_pipe", ["quantum-compilation-stage"])]

        @qml.qjit(pipelines=p, target="mlir")
        @catalyst.passes.decompose_arbitrary_ppr
        @catalyst.passes.to_ppr
        @qml.qnode(qml.device("null.qubit", wires=3))
        def circuit():
            qml.PauliRot(0.123, pauli_word="XXY", wires=[0, 1, 2])
            return catalyst.measure(0), catalyst.measure(1), catalyst.measure(2)

    Because Catalyst does not currently support execution of Pauli-based computation operations, we
    must halt the pipeline after ``quantum-compilation-stage``. This ensures that only the quantum
    passes will be applied to the initial MLIR, without attempting to further compile for execution.

    >>> print(circuit.mlir_opt)
    ...
    %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %out_qubits_2 = quantum.custom "Hadamard"() %2 : !quantum.bit
    %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
    %out_qubits_3 = quantum.custom "RX"(%cst_1) %3 : !quantum.bit
    %out_qubits_4:3 = quantum.multirz(%cst_0) %out_qubits, %out_qubits_2, %out_qubits_3 : !quantum.bit, !quantum.bit, !quantum.bit
    %out_qubits_5 = quantum.custom "Hadamard"() %out_qubits_4#0 : !quantum.bit
    %mres, %out_qubit = quantum.measure %out_qubits_5 : i1, !quantum.bit
    %from_elements = tensor.from_elements %mres : tensor<i1>
    %out_qubits_6 = quantum.custom "Hadamard"() %out_qubits_4#1 : !quantum.bit
    """
    return PassPipelineWrapper(qnode, "decompose-arbitrary-ppr")
