# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for trace-embedded decomposition-rule definitions."""

import pennylane as qp
from pennylane.typing import Wire

from catalyst import qjit
from catalyst.jax_primitives import decomp_definition_p, decomprule_p


class RepeatedGate(qp.core.Operator2):
    """A minimal gate for testing capture-session deduplication."""

    def __init__(self, wires):
        super().__init__(wires=wires)


def test_repeated_gate_captures_one_variant_set(mocker):
    """Repeated root equations share one set of captured modifier variants."""

    import catalyst.from_plxpr.qfunc_interpreter as capture_frontend

    @qp.register_resources({})
    def empty_rule(wires):
        del wires

    spy = mocker.spy(capture_frontend, "_convert_decomp_target_spec")
    with qp.decomposition.local_decomps():
        qp.add_decomps(RepeatedGate, empty_rule)

        @qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit():
            for _ in range(20):
                RepeatedGate(0)
            return qp.state()

        mlir = str(circuit.mlir_module)
        kernel_jaxpr = circuit.jaxpr.eqns[0].params["call_jaxpr"]

    assert spy.call_count == 4
    assert mlir.count("target_gate =") == 4
    assert sum(eqn.primitive is decomp_definition_p for eqn in kernel_jaxpr.eqns) == 4
    assert all(eqn.primitive is not decomprule_p for eqn in kernel_jaxpr.eqns)


def test_shared_descendant_resources_are_collected_once(mocker):
    """Preparation drives traversal without a second resource-discovery probe."""

    import catalyst.decomposition.decomposition_rules as decomposition_rules

    class RootA(qp.core.Operator2):
        def __init__(self, wires):
            super().__init__(wires=wires)

    class RootB(qp.core.Operator2):
        def __init__(self, wires):
            super().__init__(wires=wires)

    class SharedLeaf(qp.core.Operator2):
        def __init__(self, wires):
            super().__init__(wires=wires)

    @qp.register_resources({SharedLeaf(wires=Wire[1]): 1})
    def root_rule(wires):
        SharedLeaf(wires)

    @qp.register_resources({})
    def leaf_rule(wires):
        del wires

    spy = mocker.spy(decomposition_rules, "collect_resources_for_op")
    with qp.decomposition.local_decomps():
        qp.add_decomps(RootA, root_rule)
        qp.add_decomps(RootB, root_rule)
        qp.add_decomps(SharedLeaf, leaf_rule)

        @qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit():
            RootA(0)
            RootB(0)
            return qp.state()

        assert circuit.mlir_module is not None

    # Three unique base identities, each probed once for base, adjoint, control, and
    # controlled-adjoint preparation. There is no additional discovery-only probe.
    assert spy.call_count == 12


def test_rule_uses_size_agnostic_qreg():
    """Rule templates use a register formal independent of the owning QNode width."""

    @qp.register_resources({})
    def empty_rule(wires):
        del wires

    with qp.decomposition.local_decomps():
        qp.add_decomps(RepeatedGate, empty_rule)

        @qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=5))
        def circuit():
            RepeatedGate(0)
            return qp.state()

        mlir = str(circuit.mlir_module)

    assert "!qref.reg<?>" in mlir
    assert 'target_gate = "RepeatedGate{}{wires:1}{}"' in mlir


def test_multiple_qnodes_capture_and_materialize_locally(mocker):
    """Each QNode captures and materializes its definitions in its own nested module."""

    import catalyst.from_plxpr.qfunc_interpreter as capture_frontend

    @qp.register_resources({})
    def empty_rule(wires):
        del wires

    dev1 = qp.device("null.qubit", wires=1)
    dev2 = qp.device("null.qubit", wires=4)

    @qp.qnode(dev1)
    def first():
        RepeatedGate(0)
        return qp.state()

    @qp.qnode(dev2)
    def second():
        RepeatedGate(2)
        return qp.state()

    spy = mocker.spy(capture_frontend, "_convert_decomp_target_spec")
    with qp.decomposition.local_decomps():
        qp.add_decomps(RepeatedGate, empty_rule)

        @qjit(capture=True, target="mlir")
        def workflow():
            return first(), second()

        mlir = str(workflow.mlir_module)

    assert spy.call_count == 8
    assert mlir.count("target_gate =") == 8
    assert mlir.count("!qref.reg<?>") >= 8
    assert "!qref.reg<4>" in mlir  # circuit allocation remains statically sized
