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
import pytest
from pennylane.testing import find_eqns
from pennylane.typing import Float, Wire

from catalyst import qjit
from catalyst.decomposition.capture_session import OpDecompRequest
from catalyst.decomposition.decomposition_rules import walk_reachable_decomp_rule_sets
from catalyst.decomposition.modifier_state import ModifierState
from catalyst.jax_primitives import decomp_definition_p, decomprule_p

# MLIR prints the quotes inside a string attribute as `\22`.
ZEROED = "{ctrl_work_wire_type = \\22zeroed\\22}"
BORROWED = "{ctrl_work_wire_type = \\22borrowed\\22}"


class RepeatedGate(qp.core.Operator2):
    """A minimal gate for testing capture-session deduplication."""

    def __init__(self, wires):
        super().__init__(wires=wires)


def test_repeated_gate_captures_only_plain_variant(mocker):
    """Repeated plain root equations share one captured variant."""

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

    assert spy.call_count == 1
    assert mlir.count("target_gate =") == 1
    assert len(find_eqns(kernel_jaxpr, decomp_definition_p)) == 1
    assert not find_eqns(kernel_jaxpr, decomprule_p)


def test_ambient_modifier_states_are_captured_independently(mocker):
    """Nested transforms capture only their concrete control/adjoint states."""

    import catalyst.from_plxpr.qfunc_interpreter as capture_frontend

    @qp.register_resources({})
    def empty_rule(wires):
        del wires

    spy = mocker.spy(capture_frontend, "_convert_decomp_target_spec")
    with qp.decomposition.local_decomps():
        qp.add_decomps(RepeatedGate, empty_rule)

        @qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=5))
        def circuit():
            def gate():
                RepeatedGate(0)

            qp.ctrl(gate, control=[1, 2])()
            qp.ctrl(qp.adjoint(gate), control=[3, 4])()
            return qp.state()

        mlir = str(circuit.mlir_module)

    assert spy.call_count == 2
    assert mlir.count('target_gate = "2C(RepeatedGate){}{wires:1}{}"') == 1
    assert mlir.count('target_gate = "2C(Adjoint(RepeatedGate)){}{wires:1}{}"') == 1


def test_region_adjoint_composition():
    """A standalone adjoint region is retained, while two nested adjoints cancel."""

    @qp.register_resources({})
    def empty_rule(wires):
        del wires

    with qp.decomposition.local_decomps():
        qp.add_decomps(RepeatedGate, empty_rule)

        @qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit():
            def gate():
                RepeatedGate(0)

            qp.adjoint(gate)()
            qp.adjoint(qp.adjoint(gate))()
            return qp.state()

        mlir = str(circuit.mlir_module)

    assert mlir.count('target_gate = "Adjoint(RepeatedGate){}{wires:1}{}"') == 1
    assert mlir.count('target_gate = "RepeatedGate{}{wires:1}{}"') == 1


def test_region_control_and_adjoint_orders():
    """Control and adjoint regions compose to the same canonical modifier state in either order."""

    @qp.register_resources({})
    def empty_rule(wires):
        del wires

    with qp.decomposition.local_decomps():
        qp.add_decomps(RepeatedGate, empty_rule)

        @qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=3))
        def control_adjoint():
            def gate():
                RepeatedGate(0)

            qp.ctrl(qp.adjoint(gate), control=1)()
            return qp.state()

        @qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=3))
        def adjoint_control():
            def gate():
                RepeatedGate(0)

            qp.adjoint(qp.ctrl(gate, control=2))()
            return qp.state()

        control_adjoint_mlir = str(control_adjoint.mlir_module)
        adjoint_control_mlir = str(adjoint_control.mlir_module)

    target = 'target_gate = "C(Adjoint(RepeatedGate)){}{wires:1}{}"'
    assert control_adjoint_mlir.count(target) == 1
    assert adjoint_control_mlir.count(target) == 1


def test_nested_region_controls_add():
    """Nested control regions add their control counts."""

    @qp.register_resources({})
    def empty_rule(wires):
        del wires

    with qp.decomposition.local_decomps():
        qp.add_decomps(RepeatedGate, empty_rule)

        @qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=4))
        def circuit():
            def gate():
                RepeatedGate(0)

            qp.ctrl(qp.ctrl(gate, control=1), control=[2, 3])()
            return qp.state()

        mlir = str(circuit.mlir_module)

    assert mlir.count('target_gate = "3C(RepeatedGate){}{wires:1}{}"') == 1


def test_operator_and_ambient_modifiers_compose():
    """Modifiers represented on one operator compose with surrounding transform regions."""

    @qp.register_resources({})
    def empty_rule(wires):
        del wires

    with qp.decomposition.local_decomps():
        qp.add_decomps(RepeatedGate, empty_rule)

        @qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=4))
        def circuit():
            def adjointed_gate():
                qp.adjoint(RepeatedGate(0))

            def controlled_gate():
                qp.ctrl(RepeatedGate(0), control=1)

            qp.adjoint(adjointed_gate)()
            qp.ctrl(controlled_gate, control=[2, 3])()
            return qp.state()

        mlir = str(circuit.mlir_module)

    assert mlir.count('target_gate = "RepeatedGate{}{wires:1}{}"') == 1
    assert mlir.count('target_gate = "3C(RepeatedGate){}{wires:1}{}"') == 1


def test_adjoint_context_propagates_to_descendants():
    """Distributed adjoint capture propagates its state through reachable resource operations."""

    class ContextRoot(qp.core.Operator2):
        def __init__(self, wires):
            super().__init__(wires=wires)

    class ContextLeaf(qp.core.Operator2):
        def __init__(self, wires):
            super().__init__(wires=wires)

    @qp.register_resources({ContextLeaf(wires=Wire[1]): 1})
    def root_rule(wires):
        ContextLeaf(wires)

    @qp.register_resources({})
    def leaf_rule(wires):
        del wires

    with qp.decomposition.local_decomps():
        qp.add_decomps(ContextRoot, root_rule)
        qp.add_decomps(ContextLeaf, leaf_rule)

        @qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit():
            def gate():
                ContextRoot(0)

            qp.adjoint(gate)()
            return qp.state()

        mlir = str(circuit.mlir_module)

    assert 'target_gate = "Adjoint(ContextRoot){}{wires:1}{}"' in mlir
    assert 'target_gate = "Adjoint(ContextLeaf){}{wires:1}{}"' in mlir
    assert 'target_gate = "ContextRoot{}{wires:1}{}"' not in mlir
    assert 'target_gate = "ContextLeaf{}{wires:1}{}"' not in mlir


def test_distributed_control_context_reopens_descendants():
    """A newly demanded control count propagates through distributed-rule resources."""

    class ContextRoot(qp.core.Operator2):
        def __init__(self, wires):
            super().__init__(wires=wires)

    class ContextLeaf(qp.core.Operator2):
        def __init__(self, wires):
            super().__init__(wires=wires)

    @qp.register_resources({ContextLeaf(wires=Wire[1]): 1})
    def root_rule(wires):
        ContextLeaf(wires)

    @qp.register_resources({})
    def leaf_rule(wires):
        del wires

    with qp.decomposition.local_decomps():
        qp.add_decomps(ContextRoot, root_rule)
        qp.add_decomps(ContextLeaf, leaf_rule)

        @qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=3))
        def circuit():
            def gate():
                ContextRoot(0)

            qp.ctrl(gate, control=[1, 2])()
            return qp.state()

        mlir = str(circuit.mlir_module)

    assert 'target_gate = "2C(ContextRoot){}{wires:1}{}"' in mlir
    assert 'target_gate = "2C(Adjoint(ContextRoot)){}{wires:1}{}"' not in mlir
    assert 'target_gate = "2C(ContextLeaf){}{wires:1}{}"' in mlir
    assert 'target_gate = "2C(Adjoint(ContextLeaf)){}{wires:1}{}"' not in mlir


def test_registered_symbolic_resources_keep_their_own_modifiers():
    """Direct symbolic resources use their actual GraphOpID instead of the parent's context."""

    class SymbolicRoot(qp.core.Operator2):
        def __init__(self, wires):
            super().__init__(wires=wires)

    class SymbolicLeaf(qp.core.Operator2):
        def __init__(self, wires):
            super().__init__(wires=wires)

    controlled_leaf = qp.ctrl(SymbolicLeaf(Wire[1]), control=[Wire[2], Wire[3]])

    @qp.register_resources({controlled_leaf: 1})
    def direct_control_rule(base, **_):
        del base

    @qp.register_resources({})
    def leaf_rule(wires):
        del wires

    with qp.decomposition.local_decomps():
        qp.add_decomps("C(SymbolicRoot)", direct_control_rule)
        qp.add_decomps(SymbolicLeaf, leaf_rule)
        request = OpDecompRequest.from_operation(SymbolicRoot(Wire[1]))
        target_specs = walk_reachable_decomp_rule_sets([(request, {ModifierState(False, 1)})])

    targets = {target_spec.target_id for target_spec in target_specs}
    assert "2C(SymbolicLeaf){}{wires:1}{}" in targets
    assert "2C(Adjoint(SymbolicLeaf)){}{wires:1}{}" not in targets


def test_modifier_changing_identity_cycle_is_closed():
    """A controlled self-cycle captures its first state without growing controls forever."""

    class CyclicGate(qp.core.Operator2):
        def __init__(self, wires):
            super().__init__(wires=wires)

    controlled_self = qp.ctrl(CyclicGate(Wire[1]), control=[Wire[2]])

    @qp.register_resources({controlled_self: 1})
    def cyclic_rule(wires):
        del wires

    with qp.decomposition.local_decomps():
        qp.add_decomps(CyclicGate, cyclic_rule)
        request = OpDecompRequest.from_operation(CyclicGate(Wire[1]))
        target_specs = walk_reachable_decomp_rule_sets([(request, {ModifierState()})])

    targets = {target_spec.target_id for target_spec in target_specs}
    assert targets == {
        "CyclicGate{}{wires:1}{}",
        "C(CyclicGate){}{wires:1}{}",
    }


def test_indirect_control_cycle_uses_same_ancestor_count():
    """An intervening uncontrolled operator does not hide increasing controls."""

    class IndirectCycleGate(qp.core.Operator2):
        def __init__(self, wires):
            super().__init__(wires=wires)

    class CycleBridge(qp.core.Operator2):
        def __init__(self, wires):
            super().__init__(wires=wires)

    @qp.register_resources({CycleBridge(Wire[1]): 1})
    def controlled_rule(base, **_):
        del base

    twice_controlled_root = qp.ctrl(IndirectCycleGate(Wire[1]), control=Wire[2])

    @qp.register_resources({twice_controlled_root: 1})
    def bridge_rule(wires):
        del wires

    with qp.decomposition.local_decomps():
        qp.add_decomps("C(IndirectCycleGate)", controlled_rule)
        qp.add_decomps(CycleBridge, bridge_rule)
        request = OpDecompRequest.from_operation(
            IndirectCycleGate(Wire[1]), ModifierState(False, 1)
        )
        target_specs = walk_reachable_decomp_rule_sets([(request, {request.modifier_state})])

    targets = {target_spec.target_id for target_spec in target_specs}
    assert "C(IndirectCycleGate){}{wires:1}{}" in targets
    assert "CycleBridge{}{wires:1}{}" in targets
    assert "2C(IndirectCycleGate){}{wires:1}{}" not in targets


def test_same_base_modifier_transition_reopens_closure():
    """A symbolic rule may transition back to another state of the same base identity."""

    class TransitionGate(qp.core.Operator2):
        def __init__(self, wires):
            super().__init__(wires=wires)

    @qp.register_resources({})
    def base_rule(wires):
        del wires

    @qp.register_resources({TransitionGate(wires=Wire[1]): 1})
    def adjoint_rule(base):
        del base

    with qp.decomposition.local_decomps():
        qp.add_decomps(TransitionGate, base_rule)
        qp.add_decomps("Adjoint(TransitionGate)", adjoint_rule)
        request = OpDecompRequest.from_operation(TransitionGate(Wire[1]), ModifierState(True))
        target_specs = walk_reachable_decomp_rule_sets([(request, {request.modifier_state})])

    targets = {target_spec.target_id for target_spec in target_specs}
    assert "Adjoint(TransitionGate){}{wires:1}{}" in targets
    assert "TransitionGate{}{wires:1}{}" in targets


@pytest.mark.parametrize(
    "ambient, expected",
    [
        (ModifierState(), ModifierState(False, 1, 1, "zeroed")),
        (ModifierState(True, 2), ModifierState(True, 3, 1, "zeroed")),
        (ModifierState(False, 2, 1, "zeroed"), ModifierState(False, 3, 2, "zeroed")),
        (ModifierState(False, 2, 2, "borrowed"), ModifierState(False, 3, 3, "borrowed")),
    ],
)
def test_request_composes_ambient_work_wires(ambient, expected):
    """An operator's own control work wires add to the ambient ones and merge their types."""

    op = qp.ctrl(RepeatedGate(Wire[1]), Wire[1], work_wires=Wire[1], work_wire_type="zeroed")
    request = OpDecompRequest.from_operation(op, ambient)

    assert request.base_id == "RepeatedGate{}{wires:1}{}"
    assert request.modifier_state == expected


def test_operator_work_wires_reach_captured_rules():
    """The work wires of a controlled operator reach the rule discovered for it, and through it the
    operators that rule produces.

    ``C(PauliZ)`` with three controls decomposes into a ``MultiControlledX``, whose rules depend on
    how many work wires it may use and whether they are zeroed.
    """

    @qjit(capture=True, target="mlir")
    @qp.qnode(qp.device("null.qubit", wires=5))
    def circuit():
        qp.ctrl(qp.PauliZ(0), control=[1, 2, 3], work_wires=[4], work_wire_type="zeroed")
        return qp.probs(wires=[0])

    mlir = circuit.mlir

    assert f'target_gate = "3C(PauliZ){{}}{{ctrl_work_wires:1,wires:1}}{ZEROED}"' in mlir
    assert 'target_gate = "3C(PauliZ){}{wires:1}{}"' not in mlir

    mcx_id = "MultiControlledX{control_values:[tensor<3xi1>]}{wires:4,work_wires:%d}"
    assert mcx_id % 1 + "{work_wire_type = \\22zeroed\\22}" in mlir
    assert mcx_id % 0 not in mlir


def test_work_wire_variants_are_captured_independently():
    """The same controls with and without work wires, or with work wires of another type, are
    distinct graph nodes."""

    @qp.register_resources({})
    def empty_rule(wires):
        del wires

    with qp.decomposition.local_decomps():
        qp.add_decomps(RepeatedGate, empty_rule)

        @qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=4))
        def circuit():
            qp.ctrl(RepeatedGate(0), control=[1, 2])
            qp.ctrl(RepeatedGate(0), control=[1, 2], work_wires=[3])
            qp.ctrl(RepeatedGate(0), control=[1, 2], work_wires=[3], work_wire_type="zeroed")
            qp.ctrl(RepeatedGate(0), control=[1, 2], work_wires=[3], work_wire_type="zeroed")
            return qp.state()

        mlir = str(circuit.mlir_module)

    work_wire_id = "2C(RepeatedGate){}{ctrl_work_wires:1,wires:1}"
    assert mlir.count('target_gate = "2C(RepeatedGate){}{wires:1}{}"') == 1
    assert mlir.count(f'target_gate = "{work_wire_id}{BORROWED}"') == 1
    assert mlir.count(f'target_gate = "{work_wire_id}{ZEROED}"') == 1


def test_region_work_wires_are_ambient():
    """The work wires of a control region reach the operators inside it.

    PennyLane's control transform does not record a work wire type, so they are borrowed.
    """

    @qp.register_resources({})
    def empty_rule(wires):
        del wires

    with qp.decomposition.local_decomps():
        qp.add_decomps(RepeatedGate, empty_rule)

        @qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=6))
        def circuit():
            def gate():
                RepeatedGate(0)
                qp.ctrl(RepeatedGate(0), control=4, work_wires=[5], work_wire_type="zeroed")

            qp.ctrl(gate, control=[1, 2], work_wires=[3])()
            return qp.state()

        mlir = str(circuit.mlir_module)

    assert f'target_gate = "2C(RepeatedGate){{}}{{ctrl_work_wires:1,wires:1}}{BORROWED}"' in mlir
    assert f'target_gate = "3C(RepeatedGate){{}}{{ctrl_work_wires:2,wires:1}}{BORROWED}"' in mlir
    assert 'target_gate = "2C(RepeatedGate){}{wires:1}{}"' not in mlir


def test_distributed_control_carries_work_wires():
    """A distributed control rule takes the work wires in its grouped wire operand, and its
    resources keep the work wires of the target."""

    class ContextRoot(qp.core.Operator2):
        def __init__(self, wires):
            super().__init__(wires=wires)

    class ContextLeaf(qp.core.Operator2):
        def __init__(self, wires):
            super().__init__(wires=wires)

    @qp.register_resources({ContextLeaf(wires=Wire[1]): 1})
    def root_rule(wires):
        ContextLeaf(wires)

    @qp.register_resources({})
    def leaf_rule(wires):
        del wires

    with qp.decomposition.local_decomps():
        qp.add_decomps(ContextRoot, root_rule)
        qp.add_decomps(ContextLeaf, leaf_rule)

        @qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=4))
        def circuit():
            qp.ctrl(ContextRoot(0), control=[1, 2], work_wires=[3], work_wire_type="zeroed")
            return qp.state()

        mlir = str(circuit.mlir_module)

    root_id = f"2C(ContextRoot){{}}{{ctrl_work_wires:1,wires:1}}{ZEROED}"
    leaf_id = f"2C(ContextLeaf){{}}{{ctrl_work_wires:1,wires:1}}{ZEROED}"
    (root_rule_line,) = [line for line in mlir.splitlines() if f'target_gate = "{root_id}"' in line]

    assert (
        "(%arg0: !qref.reg<?>, %arg1: tensor<2xi64>, %arg2: tensor<2xi64>, "
        "%arg3: tensor<2xi1>)" in root_rule_line
    )
    assert f'"{leaf_id}" = 1' in root_rule_line
    assert f'target_gate = "{leaf_id}"' in mlir


@pytest.mark.parametrize("work_wire_type", ["zeroed", "borrowed"])
def test_registered_callable_captures_work_wire_operands(work_wire_type):
    """Callable controls in registered rules keep work wires in the rule's JAXPR inputs."""
    from pennylane.capture.primitives import ctrl_transform_prim

    @qp.register_resources(
        {
            qp.ctrl(
                qp.RX(Float, Wire[1]),
                Wire[2],
                work_wires=Wire[1],
                work_wire_type=work_wire_type,
            ): 1
        }
    )
    def callable_ctrl_rule(base, control_wires, control_values, work_wires, work_wire_type):
        qp.ctrl(
            qp.RX,
            control=control_wires,
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )(0.37, wires=base.wires)

    with qp.decomposition.local_decomps():
        qp.add_decomps("C(RepeatedGate)", callable_ctrl_rule)

        @qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=4))
        def circuit():
            qp.ctrl(RepeatedGate(0), [1, 2], work_wires=[3], work_wire_type=work_wire_type)
            return qp.state()

    kernel_jaxpr = circuit.jaxpr.eqns[0].params["call_jaxpr"]
    (rule,) = [
        eqn
        for eqn in find_eqns(kernel_jaxpr, decomp_definition_p)
        if eqn.params["frontend_name"] == "callable_ctrl_rule"
    ]
    (control,) = find_eqns(rule.params["func_jaxpr"], ctrl_transform_prim)
    assert control.params["work_wires"] is None
    assert control.params["n_work"] == 1
    assert control.params["work_wire_type"] == work_wire_type
    assert "callable_ctrl_rule" in circuit.mlir


def test_registry_lookup_uses_operator_class():
    """An Operator2 whose graph name is shared with an Operator1 uses its own registry rules."""

    @qjit(capture=True, target="mlir")
    @qp.qnode(qp.device("null.qubit", wires=2))
    def circuit():
        qp.prod(qp.X(0), qp.Y(1))
        return qp.state()

    mlir = str(circuit.mlir)

    assert 'frontend_name = "_prod2_decomp"' in mlir
    assert 'frontend_name = "_prod_decomp"' not in mlir


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

    # Three unique plain base identities are each probed once. There is no proactive modifier
    # preparation or additional discovery-only probe.
    assert spy.call_count == 3


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

    assert spy.call_count == 2
    assert mlir.count("target_gate =") == 2
    assert mlir.count("!qref.reg<?>") >= 2
    assert "!qref.reg<4>" in mlir  # circuit allocation remains statically sized
