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

# RUN: %PYTHON %s | FileCheck %s

import pennylane as qml
import catalyst

def test_conditional_capture():

    qml.capture.enable()

    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def captured_circuit(x: float):
        m = qml.measure(0)
        # CHECK: [[QREG:%.+]] = quantum.insert
        # CHECK: [[QREG_3:%.+]] = scf.if
        # CHECK:    [[QREG_2:%.+]] = quantum.insert [[QREG]][ 0]
        # CHECK:    scf.yield [[QREG_2]]
        # CHECK: else
        # CHECK:     scf.yield [[QREG]]
        # CHECK: quantum.compbasis qreg [[QREG_3]] : !quantum.obs
        qml.cond(m, lambda: (qml.X(0), None)[1])()
        return qml.state()

    @qml.qjit
    def main(x: float):
        return captured_circuit(x)

    print(main.mlir)

    qml.capture.enable()

test_conditional_capture()

def test_loop_capture():

    qml.capture.enable()

    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def captured_circuit(x: float):
        m = qml.measure(0)

        # CHECK: [[QREG:%.+]] = quantum.insert
        # CHECK: [[QREG_4:%.+]] = scf.for {{.*}} iter_args([[QREG_2:%.+]] = [[QREG]]
        # CHECK:   [[QREG_3:%.+]] = quantum.insert [[QREG_2]]
        # CHECK:   scf.yield [[QREG_3]]
        # CHECK: quantum.compbasis qreg [[QREG_4]]
        @qml.for_loop(0, 2, 1)
        def loop_fn(_):
            qml.Hadamard(0)

        loop_fn()

        return qml.state()

    @qml.qjit
    def main(x: float):
        return captured_circuit(x)

    print(main.mlir)

    qml.capture.disable()

test_loop_capture()

def test_while_capture():

    qml.capture.enable()


    # CHECK: [[QREG:%.+]] = quantum.insert
    # CHECK: [[PAIR:%.+]]:2 = scf.while {{.*}} [[QREG2:%.+]] = [[QREG]]
    # CHECK:     [[QREG3:%.+]] = quantum.insert [[QREG2:%.+]]
    # CHECK:     scf.yield {{.*}} [[QREG3]]
    # CHECK: quantum.compbasis qreg [[PAIR]]#1
    @qml.qnode(qml.device("null.qubit", wires=1))
    def captured_circuit(x: float):
        m = qml.measure(0)

        def less_than_10(x):
            return x[0] < 10

        @qml.while_loop(less_than_10)
        def loop(v):
            qml.Hadamard(0)
            return v[0] + 1, v[1]

        loop((0, 1))
        return qml.state()
        

    @qml.qjit
    def main(x: float):
        return captured_circuit(x)

    print(main.mlir)

    qml.capture.disable()

test_while_capture()

