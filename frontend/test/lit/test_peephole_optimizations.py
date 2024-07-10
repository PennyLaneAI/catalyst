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
This file performs the frontend lit tests that the peephole transformations are correctly applied.
Each test has two components:
  1. A qnode with a peephole optimization applied, here we call it "f"
  2. The SAME qnode without a peephole optimization applied, here we call it "g"

We need to check that:
  1. For "f", the peephole transform is correctly applied in mlir
  2. For "g", the peephole transform is correctly not applied in mlir
  3. "f" and "g" returns the same results.
"""

# RUN: %PYTHON %s | FileCheck %s

# pylint: disable=line-too-long

import shutil

import numpy as np
import pennylane as qml

from catalyst import cancel_inverses, qjit


def flush_peephole_opted_mlir_to_iostream(filename):
    """
    The QJIT compiler does not offer an interface to access an intermediate mlir in the pipeline.
    The `QJIT.mlir` is the mlir before any passes are run, i.e. the "0_<qnode_name>.mlir".
    Since the QUANTUM_COMPILATION_PASS is located in the middle of the pipeline, we need
    to retrieve it with keep_intermediate=True and manually access the "2_QuantumCompilationPass.mlir".
    Then we delete the kept intermediates to avoid pollution of the workspace
    """
    with open(filename + "/2_QuantumCompilationPass.mlir", encoding="utf-8") as file:
        print(file.read())
    shutil.rmtree(filename)


# CHECK-LABEL: public @jit_cancel_inverses_workflow
@qjit(keep_intermediate=True)
def cancel_inverses_workflow(xx: float):
    """
    Test catalyst.cancel_inverses
    """

    @cancel_inverses
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def f(x: float):
        qml.RX(x, wires=0)
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def g(x: float):
        qml.RX(x, wires=0)
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))

    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    _ff = f(xx)

    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    _gg = g(xx)

    return _ff, _gg


ff, gg = cancel_inverses_workflow(42.42)
assert np.allclose(ff, gg)
flush_peephole_opted_mlir_to_iostream("cancel_inverses_workflow")


# CHECK-LABEL: public @jit_cancel_inverses_workflow_rerun
@qjit(keep_intermediate=True)
def cancel_inverses_workflow_rerun(xx: float):
    """
    Test catalyst.cancel_inverses but a differently-named QNode is be run.

    This test is testing the reset of the quantum pass table is functioning correctly.
    In the previous test, the line
       'func.func(remove-chained-self-inverse{func-name=f})'
    is added to the pipeline.

    If in a second qjit object some of the local names clash with previous
    local names, the previous line will run the pass on the second
    'f' by accident!

    To solve this issue, after every qjit compilation, either aot or jit,
    the pass table and pipeline are reset.

    When the mid-end supports more gates, and this file contains more tests,
    the circuit names "f" and "g" will be naturally recycled, and this
    test can be removed (although I suggest keeping this test and this docstring).
    """

    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def f(x: float):
        qml.RX(x, wires=0)
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))

    @cancel_inverses
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def g(x: float):
        qml.RX(x, wires=0)
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))

    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    _ff = f(xx)

    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    _gg = g(xx)

    return _ff, _gg


ff, gg = cancel_inverses_workflow(42.42)
assert np.allclose(ff, gg)
flush_peephole_opted_mlir_to_iostream("cancel_inverses_workflow_rerun")
