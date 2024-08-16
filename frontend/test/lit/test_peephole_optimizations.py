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
This file performs the frontend lit tests that the peephole transformations are correctly lowered.

We check the transform jax primitives for each pass is correctly injected
during tracing, and these transform primitives are correctly lowered to the mlir before
running -apply-transform-sequence. 
"""

# RUN: %PYTHON %s | FileCheck %s

# pylint: disable=line-too-long

import shutil

import pennylane as qml
from lit_util_printers import print_jaxpr, print_mlir

from catalyst import qjit
from catalyst.debug import get_compilation_stage
from catalyst.passes import cancel_inverses, pipeline


def flush_peephole_opted_mlir_to_iostream(QJIT):
    """
    The QJIT compiler does not offer a direct interface to access an intermediate mlir in the pipeline.
    The `QJIT.mlir` is the mlir before any passes are run, i.e. the "0_<qnode_name>.mlir".
    Since the QUANTUM_COMPILATION_PASS is located in the middle of the pipeline, we need
    to retrieve it with keep_intermediate=True and manually access the "2_QuantumCompilationPass.mlir".
    Then we delete the kept intermediates to avoid pollution of the workspace
    """
    print(get_compilation_stage(QJIT, "QuantumCompilationPass"))
    shutil.rmtree(QJIT.__name__)


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

    # CHECK: module @func {
    # CHECK: module attributes {
    # CHECK-SAME: transform.with_named_sequence
    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: transform.yield
    print_mlir(func)


test_transform_named_sequence_injection()


#
# pipeline
#


def test_pipeline_lowering():
    """
    Basic pipeline lowering on one qnode.
    """
    my_pipeline = {
        "cancel_inverses": {},
        "merge_rotations": {"my-option": "aloha"},
    }

    @qjit(keep_intermediate=True)
    @pipeline(pass_pipeline=my_pipeline)
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def test_pipeline_lowering_workflow(x):
        qml.RX(x, wires=[0])
        qml.Hadamard(wires=[1])
        qml.Hadamard(wires=[1])
        return qml.expval(qml.PauliY(wires=0))

    # CHECK: transform_named_sequence
    # CHECK: _:AbstractTransformMod() = apply_registered_pass[
    # CHECK:   options=func-name=test_pipeline_lowering_workflow_transformed0
    # CHECK:   pass_name=remove-chained-self-inverse
    # CHECK: ]
    # CHECK: _:AbstractTransformMod() = apply_registered_pass[
    # CHECK:   options=func-name=test_pipeline_lowering_workflow_transformed0 my-option=aloha
    # CHECK:   pass_name=merge-rotation
    # CHECK: ]
    print_jaxpr(test_pipeline_lowering_workflow, 1.2)

    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}} {options = "func-name=test_pipeline_lowering_workflow_transformed0"}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotation" to {{%.+}} {options = "func-name=test_pipeline_lowering_workflow_transformed0 my-option=aloha"}
    # CHECK-NEXT: transform.yield
    print_mlir(test_pipeline_lowering_workflow, 1.2)

    # CHECK: {{%.+}} = call @test_pipeline_lowering_workflow_transformed0_0(
    # CHECK: func.func private @test_pipeline_lowering_workflow_transformed0_0
    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    test_pipeline_lowering_workflow(42.42)
    flush_peephole_opted_mlir_to_iostream(test_pipeline_lowering_workflow)


test_pipeline_lowering()


def test_pipeline_lowering_keep_original():
    """
    Test when the pipelined qnode and the original qnode are both used,
    and the original is correctly kept and untransformed.
    """
    my_pipeline = {
        "cancel_inverses": {},
        "merge_rotations": {},
    }

    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def f(x):
        qml.RX(x, wires=[0])
        qml.Hadamard(wires=[1])
        qml.Hadamard(wires=[1])
        return qml.expval(qml.PauliY(wires=0))

    f_pipeline = pipeline(pass_pipeline=my_pipeline)(f)

    @qjit(keep_intermediate=True)
    def test_pipeline_lowering_keep_original_workflow(x):
        return f(x), f_pipeline(x)

    # CHECK: transform_named_sequence
    # CHECK: call_jaxpr=
    # CHECK-NOT: _:AbstractTransformMod() = apply_registered_pass[
    # CHECK: call_jaxpr=
    # CHECK: _:AbstractTransformMod() = apply_registered_pass[
    # CHECK:   options=func-name=f_transformed0
    # CHECK:   pass_name=remove-chained-self-inverse
    # CHECK: ]
    # CHECK: _:AbstractTransformMod() = apply_registered_pass[
    # CHECK:   options=func-name=f_transformed0
    # CHECK:   pass_name=merge-rotation
    # CHECK: ]
    print_jaxpr(test_pipeline_lowering_keep_original_workflow, 1.2)

    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}} {options = "func-name=f_transformed0"}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotation" to {{%.+}} {options = "func-name=f_transformed0"}
    # CHECK-NEXT: transform.yield
    print_mlir(test_pipeline_lowering_keep_original_workflow, 1.2)

    # CHECK: func.func public @jit_test_pipeline_lowering_keep_original_workflow
    # CHECK: {{%.+}} = call @f_0(
    # CHECK: {{%.+}} = call @f_transformed0_0(
    # CHECK: func.func private @f_transformed0_0
    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK: func.func private @f_0(
    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    test_pipeline_lowering_keep_original_workflow(42.42)
    flush_peephole_opted_mlir_to_iostream(test_pipeline_lowering_keep_original_workflow)


test_pipeline_lowering_keep_original()


def test_pipeline_lowering_global():
    """
    Test that the global qjit circuit_transform_pipeline option
    transforms all qnodes in the qjit.
    """
    my_pipeline = {
        "cancel_inverses": {},
        "merge_rotations": {},
    }

    @qjit(keep_intermediate=True, circuit_transform_pipeline=my_pipeline)
    def global_wf():
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def g(x):
            qml.RX(x, wires=[0])
            qml.Hadamard(wires=[1])
            qml.Hadamard(wires=[1])
            return qml.expval(qml.PauliY(wires=0))

        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def h(x):
            qml.RX(x, wires=[0])
            qml.Hadamard(wires=[1])
            qml.Hadamard(wires=[1])
            return qml.expval(qml.PauliY(wires=0))

        return g(1.2), h(1.2)

    # CHECK: transform_named_sequence
    # CHECK: _:AbstractTransformMod() = apply_registered_pass[
    # CHECK:   options=func-name=g_transformed0
    # CHECK:   pass_name=remove-chained-self-inverse
    # CHECK: ]
    # CHECK: _:AbstractTransformMod() = apply_registered_pass[
    # CHECK:   options=func-name=g_transformed0
    # CHECK:   pass_name=merge-rotation
    # CHECK: ]
    # CHECK: _:AbstractTransformMod() = apply_registered_pass[
    # CHECK:   options=func-name=h_transformed1
    # CHECK:   pass_name=remove-chained-self-inverse
    # CHECK: ]
    # CHECK: _:AbstractTransformMod() = apply_registered_pass[
    # CHECK:   options=func-name=h_transformed1
    # CHECK:   pass_name=merge-rotation
    # CHECK: ]
    print_jaxpr(global_wf)

    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}} {options = "func-name=g_transformed0"}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotation" to {{%.+}} {options = "func-name=g_transformed0"}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}} {options = "func-name=h_transformed1"}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotation" to {{%.+}} {options = "func-name=h_transformed1"}
    # CHECK-NEXT: transform.yield
    print_mlir(global_wf)

    # CHECK: func.func public @jit_global_wf()
    # CHECK {{%.+}} = call @g_transformed0(
    # CHECK {{%.+}} = call @h_transformed1(
    # CHECK: func.func private @h_transformed1
    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK: func.func private @g_transformed0
    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    global_wf()
    flush_peephole_opted_mlir_to_iostream(global_wf)


test_pipeline_lowering_global()


def test_pipeline_lowering_globloc_override():
    """
    Test that local qnode pipelines correctly overrides the global
    pipeline specified by the qjit's option.
    """
    global_pipeline = {
        "cancel_inverses": {},
        "merge_rotations": {},
    }

    local_pipeline = {
        "merge_rotations": {},
    }

    @qjit(keep_intermediate=True, circuit_transform_pipeline=global_pipeline)
    def global_wf():
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def g(x):
            qml.RX(x, wires=[0])
            qml.Hadamard(wires=[1])
            qml.Hadamard(wires=[1])
            return qml.expval(qml.PauliY(wires=0))

        @pipeline(pass_pipeline=local_pipeline)
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def h(x):
            qml.RX(x, wires=[0])
            qml.Hadamard(wires=[1])
            qml.Hadamard(wires=[1])
            return qml.expval(qml.PauliY(wires=0))

        return g(1.2), h(1.2)

    # CHECK: transform_named_sequence
    # CHECK: _:AbstractTransformMod() = apply_registered_pass[
    # CHECK:   options=func-name=g_transformed1
    # CHECK:   pass_name=remove-chained-self-inverse
    # CHECK: ]
    # CHECK: _:AbstractTransformMod() = apply_registered_pass[
    # CHECK:   options=func-name=g_transformed1
    # CHECK:   pass_name=merge-rotation
    # CHECK: ]
    # CHECK: _:AbstractTransformMod() = apply_registered_pass[
    # CHECK:   options=func-name=h_transformed0
    # CHECK-NOT:   pass_name=remove-chained-self-inverse
    # CHECK:   pass_name=merge-rotation
    # CHECK: ]
    print_jaxpr(global_wf)

    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}} {options = "func-name=g_transformed1"}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotation" to {{%.+}} {options = "func-name=g_transformed1"}
    # CHECK-NOT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}} {options = "func-name=h_transformed0"}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotation" to {{%.+}} {options = "func-name=h_transformed0"}
    # CHECK-NEXT: transform.yield
    print_mlir(global_wf)

    # CHECK: func.func public @jit_global_wf()
    # CHECK {{%.+}} = call @g_transformed1(
    # CHECK {{%.+}} = call @h_transformed0(
    # CHECK: func.func private @h_transformed0
    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK: func.func private @g_transformed1
    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    global_wf()
    flush_peephole_opted_mlir_to_iostream(global_wf)


test_pipeline_lowering_globloc_override()


#
# cancel_inverses
#


def test_cancel_inverses_tracing_and_lowering():
    """
    Test cancel_inverses during tracing and lowering
    """

    @qjit
    def test_cancel_inverses_tracing_and_lowering_workflow(xx: float):

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

        _f = f(xx)
        _g = g(xx)
        _h = h(xx)
        return _f, _g, _h

    # CHECK: transform_named_sequence
    # CHECK: _:AbstractTransformMod() = apply_registered_pass[
    # CHECK:   options=func-name=f_cancel_inverses0
    # CHECK:   pass_name=remove-chained-self-inverse
    # CHECK: ]
    # CHECK: _:AbstractTransformMod() = apply_registered_pass[
    # CHECK:   options=func-name=g_cancel_inverses1
    # CHECK:   pass_name=remove-chained-self-inverse
    # CHECK: ]
    # CHECK-NOT: _:AbstractTransformMod() = apply_registered_pass[
    # CHECK-NOT:   options=func-name=h_cancel_inverses
    # CHECK-NOT:   pass_name=remove-chained-self-inverse
    print_jaxpr(test_cancel_inverses_tracing_and_lowering_workflow, 1.1)

    # CHECK: module @test_cancel_inverses_tracing_and_lowering_workflow
    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}} {options = "func-name=f_cancel_inverses0"}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}} {options = "func-name=g_cancel_inverses1"}
    # CHECK-NOT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}} {options = "func-name=h_cancel_inverses"}
    # CHECK-NEXT: transform.yield
    print_mlir(test_cancel_inverses_tracing_and_lowering_workflow, 1.1)


test_cancel_inverses_tracing_and_lowering()


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
    def test_cancel_inverses_tracing_and_lowering_outside_qjit_workflow(xx: float):
        _f = f(xx)
        return _f

    # CHECK: transform_named_sequence
    # CHECK: _:AbstractTransformMod() = apply_registered_pass[
    # CHECK:   options=func-name=f_cancel_inverses0
    # CHECK:   pass_name=remove-chained-self-inverse
    # CHECK: ]
    print_jaxpr(test_cancel_inverses_tracing_and_lowering_outside_qjit_workflow, 1.1)

    # CHECK: module @test_cancel_inverses_tracing_and_lowering_outside_qjit_workflow
    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}} {options = "func-name=f_cancel_inverses0"}
    # CHECK-NEXT: transform.yield
    print_mlir(test_cancel_inverses_tracing_and_lowering_outside_qjit_workflow, 1.1)


test_cancel_inverses_tracing_and_lowering_outside_qjit()


def test_cancel_inverses_lowering_transform_applied():
    """
    Test cancel_inverses mlir after apply-transform-sequence
    In other words, test that the pass takes the correct effect with this frontend call.
    """

    # CHECK-LABEL: public @jit_test_cancel_inverses_lowering_transform_applied_workflow
    @qjit(keep_intermediate=True)
    def test_cancel_inverses_lowering_transform_applied_workflow(xx: float):

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

        # CHECK-LABEL: func.func private @h_cancel_inverses1_0
        # CHECK: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
        # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
        # CHECK-NEXT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit

        # CHECK-LABEL: func.func private @g_0
        # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
        # CHECK-NEXT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
        # CHECK-NEXT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit

        # CHECK-LABEL: func.func private @f_cancel_inverses0_0
        # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
        # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
        # CHECK-NEXT: {{%.+}} = quantum.namedobs {{%.+}}[ PauliZ] : !quantum.obs
        # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
        _f = f(xx)

        _g = g(xx)

        _h = h(xx)

        return _f, _g, _h

    test_cancel_inverses_lowering_transform_applied_workflow(42.42)
    flush_peephole_opted_mlir_to_iostream(test_cancel_inverses_lowering_transform_applied_workflow)


test_cancel_inverses_lowering_transform_applied()


def test_cancel_inverses_keep_original():
    """
    Test cancel_inverses does not unexpectedly mutate the original qnode.
    """

    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def f(x: float):
        qml.RX(x, wires=0)
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))

    g = cancel_inverses(f)

    # CHECK-LABEL: public @jit_test_cancel_inverses_keep_original_workflow0
    # CHECK: {{%.+}} = call @f_0({{%.+}})
    # CHECK-NOT: {{%.+}} = call @f_cancel_inverses_0({{%.+}})
    # CHECK-LABEL: private @f_0({{%.+}})
    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK-NEXT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NEXT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NOT: private @f_cancel_inverses0_0
    @qjit(keep_intermediate=True)
    def test_cancel_inverses_keep_original_workflow0():
        return f(1.0)

    test_cancel_inverses_keep_original_workflow0()
    flush_peephole_opted_mlir_to_iostream(test_cancel_inverses_keep_original_workflow0)

    # CHECK-LABEL: public @jit_test_cancel_inverses_keep_original_workflow1
    # CHECK: {{%.+}} = call @f_cancel_inverses0_0({{%.+}})
    # CHECK-NOT: {{%.+}} = call @f0_0({{%.+}})
    # CHECK-LABEL: private @f_cancel_inverses0_0({{%.+}})
    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NOT: private @f0_0
    @qjit(keep_intermediate=True)
    def test_cancel_inverses_keep_original_workflow1():
        return g(1.0)

    test_cancel_inverses_keep_original_workflow1()
    flush_peephole_opted_mlir_to_iostream(test_cancel_inverses_keep_original_workflow1)

    # CHECK-LABEL: public @jit_test_cancel_inverses_keep_original_workflow2
    # CHECK: {{%.+}} = call @f_0({{%.+}})
    # CHECK: {{%.+}} = call @f_cancel_inverses0_0({{%.+}})
    # CHECK-LABEL: private @f_cancel_inverses0_0({{%.+}})
    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-LABEL: private @f_0({{%.+}})
    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK-NEXT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NEXT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    @qjit(keep_intermediate=True)
    def test_cancel_inverses_keep_original_workflow2():
        return f(1.0), g(1.0)

    test_cancel_inverses_keep_original_workflow2()
    flush_peephole_opted_mlir_to_iostream(test_cancel_inverses_keep_original_workflow2)


test_cancel_inverses_keep_original()
