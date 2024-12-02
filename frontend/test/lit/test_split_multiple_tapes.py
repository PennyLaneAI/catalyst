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

"""
This file performs the frontend lit tests that multi-tape transforms are traced into a jaxpr
correctly.
"""

# RUN: %PYTHON %s | FileCheck %s

# pylint: disable=line-too-long

from typing import Callable, Sequence

import numpy as np
import pennylane as qml
from lit_util_printers import print_jaxpr, print_mlir

from catalyst import qjit


def test_multiple_tape_transforms():
    """
    Test the jaxpr of a multi-tape-transformed qnode.
    """

    def my_quantum_transform(
        tape: qml.tape.QuantumTape,
    ) -> (Sequence[qml.tape.QuantumTape], Callable):
        tape1 = tape
        tape2 = qml.tape.QuantumTape(
            [qml.RY(tape1.operations[1].parameters[0] + 0.4, wires=0)], [qml.expval(qml.X(0))]
        )

        def post_processing_fn(results):
            return results[0] + results[1]

        return [tape1, tape2], post_processing_fn

    dispatched_transform = qml.transform(my_quantum_transform)

    @qjit
    @dispatched_transform
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def circuit_twotapes(x):
        qml.StatePrep(
            np.array([complex(1, 0), complex(0, 0), complex(0, 0), complex(0, 0)]),
            wires=[0, 1],
        )
        qml.adjoint(qml.RY)(x[0], wires=0)
        qml.RX(x[1] + 0.8, wires=1)
        return qml.expval(qml.X(0))

    # CHECK: circuit_twotapes
    # CHECK: call_jaxpr={ lambda ;
    # CHECK: qdevice[
    # CHECK: ]
    # CHECK: qdealloc
    # CHECK-NEXT: qdevice[
    # CHECK: ]
    # CHECK: qdealloc
    # CHECK-NEXT: {{.+}}:f64[] = add {{.+}} {{.+}}
    print_jaxpr(circuit_twotapes, [0.1, 0.2])

    # CHECK: circuit_twotapes
    # CHECK: quantum.device
    # CHECK: quantum.dealloc
    # CHECK-NEXT: quantum.device_release
    # CHECK-NEXT: quantum.device
    # CHECK: quantum.dealloc
    # CHECK-NEXT: quantum.device_release
    # CHECK-NEXT: {{%.+}} = stablehlo.add {{%.+}}, {{%.+}} : tensor<f64>
    print_mlir(circuit_twotapes, [0.1, 0.2])
    # The above mlir is expected by --split-multiple-tapes pass.
    # See mlir/test/Catalyst/SplitMultipleTapesTest.mlir


test_multiple_tape_transforms()
