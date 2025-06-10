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

import pennylane as qml
from utils import print_jaxpr, print_mlir

from catalyst import pipeline, qjit
from catalyst.debug import get_compilation_stage
from catalyst.passes import apply_pass, cancel_inverses, merge_rotations


def flush_peephole_opted_mlir_to_iostream(QJIT):
    """
    The QJIT compiler does not offer a direct interface to access an intermediate mlir in the pipeline.
    The `QJIT.mlir` is the mlir before any passes are run, i.e. the "0_<qnode_name>.mlir".
    Since the QUANTUM_COMPILATION_PASS is located in the middle of the pipeline, we need
    to retrieve it with keep_intermediate=True and manually access the "2_QuantumCompilationPass.mlir".
    Then we delete the kept intermediates to avoid pollution of the workspace
    """
    print(get_compilation_stage(QJIT, "QuantumCompilationPass"))


#
# pipeline
#


def test_pipeline_lowering():
    """
    Basic pipeline lowering on one qnode.
    """
    my_pipeline = {
        "cancel_inverses": {},
        "merge_rotations": {},
    }

    @qjit(keep_intermediate=True)
    @pipeline(my_pipeline)
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def test_pipeline_lowering_workflow(x):
        qml.RX(x, wires=[0])
        qml.Hadamard(wires=[1])
        qml.Hadamard(wires=[1])
        return qml.expval(qml.PauliY(wires=0))

    # CHECK: pipeline=(remove-chained-self-inverse, merge-rotations)
    print_jaxpr(test_pipeline_lowering_workflow, 1.2)

    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotations" to {{%.+}}
    # CHECK-NEXT: transform.yield
    print_mlir(test_pipeline_lowering_workflow, 1.2)

    # CHECK: {{%.+}} = call @test_pipeline_lowering_workflow_0(
    # CHECK: func.func public @test_pipeline_lowering_workflow_0(
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

    f_pipeline = pipeline(my_pipeline)(f)

    @qjit(keep_intermediate=True)
    def test_pipeline_lowering_keep_original_workflow(x):
        return f(x), f_pipeline(x)

    # COM: this if for f(x)
    # CHECK: quantum_kernel
    # COM: this if for f_pipeline(x)
    # COM: Unfortunately, we don't have a nice repr for qnode
    # CHECK: quantum_kernel
    # CHECK: pipeline=(remove-chained-self-inverse, merge-rotations)
    print_jaxpr(test_pipeline_lowering_keep_original_workflow, 1.2)

    # COM: This is the one that is unchanged
    # CHECK: transform.named_sequence @__transform_main
    # COM: This is the one that is changed
    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotations" to {{%.+}}
    # CHECK-NEXT: transform.yield
    print_mlir(test_pipeline_lowering_keep_original_workflow, 1.2)

    # CHECK: func.func public @jit_test_pipeline_lowering_keep_original_workflow
    # CHECK: {{%.+}} = call @f_0(
    # CHECK: {{%.+}} = call @f_1_0(
    # CHECK: func.func public @f_0(
    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK: func.func public @f_1_0(
    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
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

    # CHECK: quantum_kernel
    # CHECK: pipeline=(remove-chained-self-inverse, merge-rotations)
    # CHECK: quantum_kernel
    # CHECK: pipeline=(remove-chained-self-inverse, merge-rotations)
    print_jaxpr(global_wf)

    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotations" to {{%.+}}
    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotations" to {{%.+}}
    # CHECK-NEXT: transform.yield
    print_mlir(global_wf)

    # CHECK: func.func public @jit_global_wf()
    # CHECK {{%.+}} = call @g_0(
    # CHECK {{%.+}} = call @h_0(
    # CHECK: func.func public @g_0(
    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK: func.func public @h_0(
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

        @pipeline(local_pipeline)
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def h(x):
            qml.RX(x, wires=[0])
            qml.Hadamard(wires=[1])
            qml.Hadamard(wires=[1])
            return qml.expval(qml.PauliY(wires=0))

        return g(1.2), h(1.2)

    # CHECK: quantum_kernel
    # CHECK: pipeline=(remove-chained-self-inverse, merge-rotations)
    # CHECK: quantum_kernel
    # CHECK: pipeline=(merge-rotations,)
    print_jaxpr(global_wf)

    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotations" to {{%.+}}
    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NOT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotations" to {{%.+}}
    # CHECK-NEXT: transform.yield
    print_mlir(global_wf)

    # CHECK: func.func public @jit_global_wf()
    # CHECK {{%.+}} = call @g_0(
    # CHECK {{%.+}} = call @h_0(
    # CHECK: func.func public @g_0(
    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK: func.func public @h_0
    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    global_wf()
    flush_peephole_opted_mlir_to_iostream(global_wf)


test_pipeline_lowering_globloc_override()


def test_chained_pipeline_lowering():
    """
    Test that chained pipelines are correctly lowered.
    """
    pipeline1 = {
        "cancel_inverses": {},
    }
    pipeline2 = {
        "cancel_inverses": {},
        "merge_rotations": {},
    }

    @qjit()
    @pipeline(pipeline1)
    @pipeline(pipeline2)
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def test_chained_pipeline_lowering_workflow(x: float):
        qml.Hadamard(wires=[1])
        qml.RX(x, wires=[0])
        qml.RX(-x, wires=[0])
        qml.Hadamard(wires=[1])
        return qml.expval(qml.PauliY(wires=0))

    print(test_chained_pipeline_lowering_workflow.mlir)

    # # CHECK: transform.named_sequence @__transform_main
    # # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
    # # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotations" to {{%.+}}
    # # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
    # # CHECK-NEXT: transform.yield


test_chained_pipeline_lowering()


def test_chained_pipeline_lowering_keep_original():
    """
    Test when the chained pipeline and the original qnode are both used,
    and the original is correctly kept and untransformed.
    """
    pipeline1 = {
        "cancel_inverses": {},
        "merge_rotations": {},
    }
    pipeline2 = {
        "cancel_inverses": {},
    }

    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def f(x: float):
        qml.RX(x, wires=[0])
        qml.Hadamard(wires=[1])
        qml.Hadamard(wires=[1])
        return qml.expval(qml.PauliY(wires=0))

    f_pipeline1 = pipeline(pipeline1)(f)
    f_pipeline2 = pipeline(pipeline2)(f_pipeline1)

    @qjit()
    def test_chained_pipeline_lowering_keep_original_workflow(x: float):
        return f(x), f_pipeline1(x), f_pipeline2(x)

    # COM: The unchanged qnode
    # CHECK: transform.named_sequence @__transform_main
    # COM: The qnode after pipeline1
    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotations" to {{%.+}}
    # COM: The qnode after pipeline2
    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotations" to {{%.+}}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
    # CHECK-NEXT: transform.yield
    print(test_chained_pipeline_lowering_keep_original_workflow.mlir)


test_chained_pipeline_lowering_keep_original()


def test_chained_apply_passes():
    """
    Test that chained passes are correctly applied in sequence using apply_pass.
    """

    @qjit()
    @apply_pass("merge-rotations")
    @apply_pass("remove-chained-self-inverse")
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def test_chained_apply_passes_workflow(x: float):
        qml.Hadamard(wires=[1])
        qml.RX(x, wires=[0])
        qml.RX(-x, wires=[0])
        qml.Hadamard(wires=[1])
        return qml.expval(qml.PauliY(wires=0))

    print(test_chained_apply_passes_workflow.mlir)

    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotations" to {{%.+}}
    # CHECK-NEXT: transform.yield


test_chained_apply_passes()


def test_chained_apply_passes_keep_original():
    """
    Test when the chained apply passes and the original qnode are both used,
    and the original is correctly kept and untransformed.
    """

    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def f(x: float):
        qml.RX(x, wires=[0])
        qml.Hadamard(wires=[1])
        qml.Hadamard(wires=[1])
        return qml.expval(qml.PauliY(wires=0))

    f_pass1 = apply_pass("remove-chained-self-inverse")(f)
    f_pass2 = apply_pass("merge-rotations")(f_pass1)

    @qjit()
    def test_chained_apply_passes_keep_original_workflow(x: float):
        return f(x), f_pass1(x), f_pass2(x)

    # COM: The unchanged qnode
    # CHECK: transform.named_sequence @__transform_main
    # COM: The qnode after pipeline1
    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
    # COM: The qnode after merge_rotations
    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotations" to {{%.+}}
    # CHECK-NEXT: transform.yield
    print(test_chained_apply_passes_keep_original_workflow.mlir)


test_chained_apply_passes_keep_original()


def test_chained_peephole_passes():
    """
    Test that chained peephole passes are correctly applied in sequence.
    """

    @qjit()
    @merge_rotations
    @cancel_inverses
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def test_chained_peephole_passes_workflow(x: float):
        qml.Hadamard(wires=[1])
        qml.RX(x, wires=[0])
        qml.RX(-x, wires=[0])
        qml.Hadamard(wires=[1])
        return qml.expval(qml.PauliY(wires=0))

    print(test_chained_peephole_passes_workflow.mlir)

    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotations" to {{%.+}}
    # CHECK-NEXT: transform.yield


test_chained_peephole_passes()


def test_chained_peephole_passes_keep_original():
    """
    Test when the chained peephole passes and the original qnode are both used,
    and the original is correctly kept and untransformed.
    """

    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def f(x: float):
        qml.RX(x, wires=[0])
        qml.Hadamard(wires=[1])
        qml.Hadamard(wires=[1])
        return qml.expval(qml.PauliY(wires=0))

    f_pass1 = cancel_inverses(f)
    f_pass2 = merge_rotations(f_pass1)

    @qjit()
    def test_chained_peephole_passes_keep_original_workflow(x: float):
        return f(x), f_pass1(x), f_pass2(x)

    # COM: The unchanged qnode
    # CHECK: transform.named_sequence @__transform_main
    # COM: The qnode after cancel_inverses
    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
    # COM: The qnode after merge_rotations
    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotations" to {{%.+}}
    # CHECK-NEXT: transform.yield
    print(test_chained_peephole_passes_keep_original_workflow.mlir)


test_chained_peephole_passes_keep_original()


#
# autograph
#


def test_single_pass_with_autograph():
    """
    Test that peephole optimization works with autograph
    """

    @qjit(autograph=True, target="mlir")
    @merge_rotations
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def f(x: float):
        qml.RX(x, wires=0)
        qml.RX(x, wires=0)
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))

    # CHECK: transform.named_sequence @__transform_main(
    # CHECK-NEXT: transform.apply_registered_pass "merge-rotations" to {{%.+}}
    # CHECK-NEXT: transform.yield
    print(f.mlir)


test_single_pass_with_autograph()


def test_pipeline_with_autograph():
    """
    Test that pipeline works with autograph
    """

    my_pipeline = {
        "cancel_inverses": {},
        "merge_rotations": {},
    }

    @qjit(autograph=True, target="mlir")
    @pipeline(my_pipeline)
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def f(x: float):
        qml.RX(x, wires=0)
        qml.RX(x, wires=0)
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))

    # CHECK: transform.named_sequence @__transform_main(
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotations" to {{%.+}}
    # CHECK-NEXT: transform.yield
    print(f.mlir)


test_pipeline_with_autograph()


def test_single_pass_for_loop_autograph():
    """
    Test a peephole optimization where the code is transformed
    """

    @qjit(autograph=True, target="mlir")
    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotations" to {{%.+}}
    # CHECK-NEXT: transform.yield
    @merge_rotations
    @qml.qnode(qml.device("null.qubit", wires=1))
    def circuit(n_iter: int):
        for _ in range(n_iter):
            qml.RX(0.1, wires=0)
            qml.T(0)
            qml.RX(0.2, wires=0)
        return qml.expval(qml.PauliZ(0))

    print(circuit.mlir)


test_single_pass_for_loop_autograph()


def test_stacked_pass_for_loop_autograph():
    """
    Test a peephole optimization where the code is transformed
    """

    @qjit(autograph=True, target="mlir")
    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotations" to {{%.+}}
    # CHECK-NEXT: transform.yield
    @merge_rotations
    @cancel_inverses
    @qml.qnode(qml.device("null.qubit", wires=1))
    def circuit(n_iter: int):
        for _ in range(n_iter):
            qml.RX(0.1, wires=0)
            qml.T(0)
            qml.RX(0.2, wires=0)
        return qml.expval(qml.PauliZ(0))

    print(circuit.mlir)


test_stacked_pass_for_loop_autograph()

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

    # CHECK: quantum_kernel
    # CHECK: pipeline=(remove-chained-self-inverse,)
    # CHECK: quantum_kernel
    # CHECK: pipeline=(remove-chained-self-inverse,)
    print_jaxpr(test_cancel_inverses_tracing_and_lowering_workflow, 1.1)

    # CHECK: module @test_cancel_inverses_tracing_and_lowering_workflow
    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
    # CHECK-NEXT: transform.yield
    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
    # CHECK-NOT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
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

    # CHECK: quantum_kernel
    # CHECK: pipeline=(remove-chained-self-inverse,)
    print_jaxpr(test_cancel_inverses_tracing_and_lowering_outside_qjit_workflow, 1.1)

    # CHECK: module @test_cancel_inverses_tracing_and_lowering_outside_qjit_workflow
    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
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

        # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
        # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
        # CHECK-NEXT: {{%.+}} = quantum.namedobs {{%.+}}[ PauliZ] : !quantum.obs
        # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
        _f = f(xx)

        # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
        # CHECK-NEXT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
        # CHECK-NEXT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
        _g = g(xx)

        # CHECK: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
        # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
        # CHECK-NEXT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
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
    # CHECK-NOT: {{%.+}} = call @f_0({{%.+}})
    # CHECK-LABEL: public @f_0({{%.+}})
    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK-NEXT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NEXT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NOT: public @f_0
    @qjit(keep_intermediate=True)
    def test_cancel_inverses_keep_original_workflow0():
        return f(1.0)

    test_cancel_inverses_keep_original_workflow0()
    flush_peephole_opted_mlir_to_iostream(test_cancel_inverses_keep_original_workflow0)

    # CHECK-LABEL: public @jit_test_cancel_inverses_keep_original_workflow1
    # CHECK: {{%.+}} = call @f_0({{%.+}})
    # CHECK-NOT: {{%.+}} = call @f_0({{%.+}})
    # CHECK-LABEL: public @f_0({{%.+}})
    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NOT: public @f_0
    @qjit(keep_intermediate=True)
    def test_cancel_inverses_keep_original_workflow1():
        return g(1.0)

    test_cancel_inverses_keep_original_workflow1()
    flush_peephole_opted_mlir_to_iostream(test_cancel_inverses_keep_original_workflow1)

    # CHECK-LABEL: public @jit_test_cancel_inverses_keep_original_workflow2
    # CHECK: {{%.+}} = call @f_0({{%.+}})
    # CHECK: {{%.+}} = call @f_1_0({{%.+}})
    # CHECK-LABEL: public @f_0({{%.+}})
    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK-NEXT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NEXT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-LABEL: public @f_1_0({{%.+}})
    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    @qjit(keep_intermediate=True)
    def test_cancel_inverses_keep_original_workflow2():
        return f(1.0), g(1.0)

    test_cancel_inverses_keep_original_workflow2()
    flush_peephole_opted_mlir_to_iostream(test_cancel_inverses_keep_original_workflow2)


test_cancel_inverses_keep_original()


#
# merge_rotations
#


def test_merge_rotations_tracing_and_lowering():
    """
    Test merge_rotations during tracing and lowering
    """

    @qjit
    def test_merge_rotations_tracing_and_lowering_workflow(xx: float):

        @merge_rotations
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def f(x: float):
            qml.RX(x, wires=0)
            qml.RX(x, wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        @merge_rotations
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def g(x: float):
            qml.RX(x, wires=0)
            qml.RX(x, wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def h(x: float):
            qml.RX(x, wires=0)
            qml.RX(x, wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        _f = f(xx)
        _g = g(xx)
        _h = h(xx)
        return _f, _g, _h

    # CHECK: quantum_kernel
    # CHECK: pipeline=(merge-rotations,)
    # CHECK: quantum_kernel
    # CHECK: pipeline=(merge-rotations,)
    # CHECK: quantum_kernel
    print_jaxpr(test_merge_rotations_tracing_and_lowering_workflow, 1.1)

    # CHECK: module @test_merge_rotations_tracing_and_lowering_workflow
    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotations" to {{%.+}}
    # CHECK-NEXT: transform.yield
    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotations" to {{%.+}}
    # CHECK-NOT: {{%.+}} = transform.apply_registered_pass "merge-rotations" to {{%.+}}
    # CHECK-NEXT: transform.yield
    print_mlir(test_merge_rotations_tracing_and_lowering_workflow, 1.1)


test_merge_rotations_tracing_and_lowering()
