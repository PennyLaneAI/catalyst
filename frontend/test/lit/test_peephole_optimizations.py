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

In addition, we check the transform jax primitives for each pass is correctly injected
during tracing, and these transform primitives are correctly lowered to the mlir before
running -transform-interpreter. 
"""

# RUN: %PYTHON %s | FileCheck %s

# pylint: disable=line-too-long

import shutil

import numpy as np
import pennylane as qml

from catalyst import cancel_inverses, qjit
from catalyst.debug.compiler_functions import print_compilation_stage


def flush_peephole_opted_mlir_to_iostream(QJIT):
    """
    The QJIT compiler does not offer a direct interface to access an intermediate mlir in the pipeline.
    The `QJIT.mlir` is the mlir before any passes are run, i.e. the "0_<qnode_name>.mlir".
    Since the QUANTUM_COMPILATION_PASS is located in the middle of the pipeline, we need
    to retrieve it with keep_intermediate=True and manually access the "2_QuantumCompilationPass.mlir".
    Then we delete the kept intermediates to avoid pollution of the workspace
    """
    print_compilation_stage(QJIT, "QuantumCompilationPass")
    shutil.rmtree(QJIT.__name__)


def print_attr(f, attr, *args, aot: bool = False, **kwargs):
    """Print function attribute"""
    name = f"TEST {f.__name__}"
    print("\n" + "-" * len(name))
    print(f"{name}\n")
    res = None
    if not aot:
        res = f(*args, **kwargs)
    print(getattr(f, attr))
    return res


def print_jaxpr(f, *args, **kwargs):
    """Print jaxpr code of a function"""
    return print_attr(f, "jaxpr", *args, **kwargs)


def print_mlir(f, *args, **kwargs):
    """Print mlir code of a function"""
    return print_attr(f, "mlir", *args, **kwargs)


#
# cancel_inverses
#


# CHECK-LABEL: public @jit_test_peephole_workflow_cancel_inverses
@qjit(keep_intermediate=True)
def test_peephole_workflow_cancel_inverses(xx: float):
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

    @cancel_inverses
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def h(x: float):
        """
        Test that non-neighbouring self inverses are not canceled
        """
        qml.Hadamard(wires=0)
        qml.RX(x, wires=0)
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))

    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NEXT: {{%.+}} = quantum.namedobs {{%.+}}[ PauliZ] : !quantum.obs
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    _ff = f(xx)

    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK-NEXT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NEXT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    _gg = g(xx)

    # CHECK: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK-NEXT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    _hh = h(xx)

    return _ff, _gg, _hh


ff, gg, hh = test_peephole_workflow_cancel_inverses(42.42)
assert np.allclose(ff, gg)
flush_peephole_opted_mlir_to_iostream(test_peephole_workflow_cancel_inverses)


#
# General lowering tests
#


def test_transform_named_sequence_injection():
    """
    Test the transform.with_named_sequence jax primitive and mlir operation are
    always generated for qjit.
    """

    @qjit
    def func():
        return

    # CHECK: transform_named_sequence
    print_jaxpr(func)

    # CHECK: module @func attributes {transform.with_named_sequence}
    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: transform.yield
    print_mlir(func)


test_transform_named_sequence_injection()


@qjit
def test_cancel_inverses_tracing_and_lowering(xx: float):
    """
    Test catalyst.cancel_inverses during tracing and lowering
    """

    @cancel_inverses
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

    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def h(x: float):
        qml.RX(x, wires=0)
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))

    _ff = f(xx)
    _gg = g(xx)
    _hh = h(xx)


# CHECK: transform_named_sequence
# CHECK: _:AbstractTransformFunc() = apply_registered_pass[
# CHECK:   options=func-name=f
# CHECK:   pass_name=remove-chained-self-inverse
# CHECK: ]
# CHECK: _:AbstractTransformFunc() = apply_registered_pass[
# CHECK:   options=func-name=g
# CHECK:   pass_name=remove-chained-self-inverse
# CHECK: ]
# CHECK-NOT: _:AbstractTransformFunc() = apply_registered_pass[
# CHECK-NOT:   options=func-name=h
# CHECK-NOT:   pass_name=remove-chained-self-inverse
print_jaxpr(test_cancel_inverses_tracing_and_lowering, 1.1)

# CHECK: module @test_cancel_inverses_tracing_and_lowering attributes {transform.with_named_sequence}
# CHECK: transform.named_sequence @__transform_main
# CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}} {options = "func-name=f"}
# CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}} {options = "func-name=g"}
# CHECK-NOT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}} {options = "func-name=h"}
# CHECK-NEXT: transform.yield
print_mlir(test_cancel_inverses_tracing_and_lowering, 1.1)


def test_cancel_inverses_tracing_and_lowering_outside_qjit():
    """
    Test when cancel_inverses act on a qnode outside qjit
    """

    @cancel_inverses
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def f(x: float):
        qml.RX(x, wires=0)
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))

    @qjit
    def workflow(xx: float):

        @cancel_inverses
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def g(x: float):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        ff = f(xx)
        gg = g(xx)

        return ff, gg

    # CHECK: transform_named_sequence
    # CHECK: _:AbstractTransformFunc() = apply_registered_pass[
    # CHECK:   options=func-name=f
    # CHECK:   pass_name=remove-chained-self-inverse
    # CHECK: ]
    # CHECK: _:AbstractTransformFunc() = apply_registered_pass[
    # CHECK:   options=func-name=g
    # CHECK:   pass_name=remove-chained-self-inverse
    # CHECK: ]
    print_jaxpr(workflow, 1.1)

    # CHECK: module @workflow attributes {transform.with_named_sequence}
    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}} {options = "func-name=f"}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}} {options = "func-name=g"}
    # CHECK-NEXT: transform.yield
    print_mlir(workflow, 1.1)


test_cancel_inverses_tracing_and_lowering_outside_qjit()
