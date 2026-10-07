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
from pennylane.typing import Wire

from catalyst import qjit
from catalyst.decomposition.capture_session import OpDecompRequest, RuleIdentity
from catalyst.decomposition.decomposition_rules import walk_reachable_decomp_rule_sets
from catalyst.jax_primitives import decomp_definition_p, decomprule_p


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
    assert sum(eqn.primitive is decomp_definition_p for eqn in kernel_jaxpr.eqns) == 1
    assert all(eqn.primitive is not decomprule_p for eqn in kernel_jaxpr.eqns)


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


def test_precompilation_preserves_traversal(mocker):
    """The trace-time rule traversal should continue past precompiled rules."""

    import catalyst.from_plxpr.qfunc_interpreter as capture_frontend

    class CacheRoot(qp.core.Operator2):
        def __init__(self, wires):
            super().__init__(wires=wires)

    class CacheLeaf(qp.core.Operator2):
        def __init__(self, wires):
            super().__init__(wires=wires)

    @qp.register_resources({CacheLeaf(wires=Wire[1]): 1})
    def cacheroot_to_cacheleaf(wires):
        CacheLeaf(wires)

    @qp.register_resources({})
    def cacheroot_to_leaf(wires):
        del wires

    @qp.register_resources({})
    def cacheleaf_to_leaf(wires):
        del wires

    cached_rule_identity = RuleIdentity(
        "CacheRoot{}{wires:1}{}",
        "cacheroot_to_cacheleaf",
        {"CacheLeaf{}{wires:1}{}": 1},
    )

    # mock the cache to only include the test CacheRoot
    load_cache = mocker.patch.object(
        capture_frontend,
        "load_precompiled_rule_identities",
        return_value=frozenset({cached_rule_identity}),
    )

    with qp.decomposition.local_decomps():
        qp.add_decomps(CacheRoot, cacheroot_to_cacheleaf, cacheroot_to_leaf)
        qp.add_decomps(CacheLeaf, cacheleaf_to_leaf)

        @qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit():
            CacheRoot(0)
            return qp.state()

        mlir = str(circuit.mlir_module)

    # ensure cache was only loaded once
    load_cache.assert_called_once_with()

    # cached rule should be skipped
    assert 'frontend_name = "cacheroot_to_cacheleaf"' not in mlir

    # uncached rule targetting an op with some cached rules should be compiled
    assert 'frontend_name = "cacheroot_to_leaf"' in mlir

    # uncached rule targetting op that is a product of a cached rule should be compiled
    # (i.e. traversal should not stop at cached ops)
    assert 'frontend_name = "cacheleaf_to_leaf"' in mlir


def test_tracing_recovers_missed_cache(mocker):
    """Test that the trace-time traversal captures all rules if loading the cache fails."""

    import catalyst.from_plxpr.qfunc_interpreter as capture_frontend

    @qp.register_resources({})
    def first_rule(wires):
        del wires

    @qp.register_resources({})
    def second_rule(wires):
        del wires

    load_cache = mocker.patch.object(
        capture_frontend,
        "load_precompiled_rule_identities",
        side_effect=OSError("cache unavailable"),
    )

    with qp.decomposition.local_decomps(), pytest.warns(match="Failed to load precompiled rules"):
        qp.add_decomps(RepeatedGate, first_rule, second_rule)

        @qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit():
            RepeatedGate(0)
            return qp.state()

        mlir = str(circuit.mlir_module)

    # ensure cache was only loaded once
    load_cache.assert_called_once_with()

    # ensure rules were compiled
    assert 'frontend_name = "first_rule"' in mlir
    assert 'frontend_name = "second_rule"' in mlir


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
        target_specs = walk_reachable_decomp_rule_sets([(request, {(False, 1)})])

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
        target_specs = walk_reachable_decomp_rule_sets([(request, {(False, 0)})])

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
        request = OpDecompRequest.from_operation(IndirectCycleGate(Wire[1]), (False, 1))
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
        request = OpDecompRequest.from_operation(TransitionGate(Wire[1]), (True, 0))
        target_specs = walk_reachable_decomp_rule_sets([(request, {request.modifier_state})])

    targets = {target_spec.target_id for target_spec in target_specs}
    assert "Adjoint(TransitionGate){}{wires:1}{}" in targets
    assert "TransitionGate{}{wires:1}{}" in targets


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


if __name__ == "__main__":
    pytest.main(["-x", __file__])
