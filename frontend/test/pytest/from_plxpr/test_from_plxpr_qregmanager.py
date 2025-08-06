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

"""
This module tests the from_plxpr QregManager object.

Quoted from the object's docstring:

    With the above in mind, this `QregManager` class promises the following:
    1. An instance of this class will always be tied to one root qreg value.
    2. At any moment during the from_plxpr conversion:
       - `QregManager.get()` returns the current catalyst qreg SSA value for the managed
          root register on that instance;
       - `QregManager[i]` returns the current catalyst qubit SSA value for the i-th index
          on the managed root register on that instance. If none exists, a new qubit will be
          extracted.

    To achieve the above, users of this class are expected to:
    1. Initialize an instance with the qreg SSA value from a new allocation or a new block argument;
    2. Whenever a new meta-op/op is bind-ed, update the current qreg/qubit SSA value with:
       - `QubitManager.set(new_qreg_value)`
       - `QubitManager[i] = new_qubit_value`
"""
# pylint: disable=use-implicit-booleaness-not-comparison
import textwrap

import pytest
from jax._src.core import Literal
from jax.api_util import debug_info as jdb
from jax.core import set_current_trace, take_current_trace
from jax.extend.core import Primitive
from jax.interpreters.partial_eval import DynamicJaxprTrace

from catalyst.from_plxpr.qreg_manager import QregManager
from catalyst.jax_primitives import AbstractQbit, AbstractQreg, qalloc_p, qextract_p
from catalyst.utils.exceptions import CompileError


@pytest.fixture(autouse=True)
def launch_empty_jaxpr_interpreter():
    """
    Start an empty interpreter for each test.
    The interpreter we want is a DynamicJaxprTrace, which is an interpreter for creating jaxprs.
    """
    # Note that DynamicJaxprTrace requires a `debug_info` object
    # We just pass an empty one
    trace = DynamicJaxprTrace(debug_info=jdb("qreg_manager_from_plxpr_test", None, [], {}))
    with set_current_trace(trace):
        yield
    del trace


# A mock primitive that takes in a qreg and returns a qreg
qreg_mock_op_p = Primitive("qreg_mock_op")


@qreg_mock_op_p.def_abstract_eval
def _qreg_mock_op_abstract_eval(qreg):  # pylint: disable=unused-argument
    # Return a completely independent, second qreg object
    # This is to mimic real ops
    return AbstractQreg()


# A mock primitive that takes in a qubit and returns a qubit
qubit_mock_op_p = Primitive("qubit_mock_op")
qubit_mock_op_p.multiple_results = True


@qubit_mock_op_p.def_abstract_eval
def _qubit_mock_op_abstract_eval(*qubits):
    # Return a completely independent, second list of qubit objects
    # This is to mimic real ops
    return [AbstractQbit()] * len(qubits)


def _interpret_operation(wires, qreg_manager):
    """
    Convenience helper for binding a mock gate.
    Note that this helper follows plxpr semantics, so gate targets are only specified
    by a global wire index.
    """
    in_qubits = [qreg_manager[w] for w in wires]
    out_qubits = qubit_mock_op_p.bind(*in_qubits)

    # Update the qubit values.
    # This is user code. This is point 2 of the "users are expected to" in the
    # QregManager specs.
    for wire, out_qubit in zip(wires, out_qubits):
        qreg_manager[wire] = out_qubit

    return out_qubits


class TestQregGetSet:
    """Unit test for getter and setter for the managed qreg"""

    def test_getter_setter(self):
        """Test getter and setter"""
        qreg_manager = QregManager(None)
        qreg_manager.set("monkey_mock_qreg")
        assert qreg_manager.get() == "monkey_mock_qreg"
        assert qreg_manager.get() == "monkey_mock_qreg"  # test that getter does not set

        qreg_manager.set("donkey_mock_qreg")
        assert qreg_manager.get() == "donkey_mock_qreg"


class TestQregManagerInitialization:
    """Test initialization of QregManager objects."""

    def test_new_alloc(self):
        """Test qregs from new alloc"""
        qreg = qalloc_p.bind(42)
        qreg_manager = QregManager(qreg)
        assert qreg_manager.get() is qreg
        assert qreg_manager.wire_map == {}

    def test_scope_arg(self):
        """Test qregs from a scope argument"""

        def f(qreg):
            qreg_manager = QregManager(qreg)
            assert qreg_manager.get() is qreg
            assert qreg_manager.wire_map == {}

        outer_scope_qreg = qalloc_p.bind(42)
        f(outer_scope_qreg)


class TestQubitValues:
    """Test QregManager correctly updates qubit SSA values."""

    def test_auto_extract(self):
        """Test that a new qubit is extracted when indexing into a new wire"""
        qreg = qalloc_p.bind(42)
        qreg_manager = QregManager(qreg)
        new_qubit = qreg_manager[0]

        assert list(qreg_manager.wire_map.keys()) == [0]
        assert qreg_manager[0] is new_qubit
        with take_current_trace() as trace:
            # Check that an extract primitive is added
            assert trace.frame.eqns[-1].primitive is qextract_p

            # Check that the extract primitive follows the wire index in the qreg manager
            # __getitem__ method
            extract_p_index_invar = trace.frame.eqns[-1].invars[-1]
            assert isinstance(extract_p_index_invar, Literal)
            assert extract_p_index_invar.val == 0

    def test_no_overwriting_extract(self):
        """Test that no new qubit is extracted when indexing into an existing wire"""
        qreg = qalloc_p.bind(42)
        qreg_manager = QregManager(qreg)
        new_qubit = qreg_manager[0]
        not_a_new_qubit = qreg_manager[0]

        assert not_a_new_qubit is new_qubit

    def test_simple_gate(self):
        """Test a simple qubit opertaion"""
        qreg = qalloc_p.bind(42)
        qreg_manager = QregManager(qreg)

        wires = [0, 1]
        out_qubits = _interpret_operation(wires, qreg_manager)

        # Check that qubit-level ops do not affect the managed qreg
        assert qreg_manager.get() is qreg

        # First check with python handle variables
        assert qreg_manager[0] is out_qubits[0]
        assert qreg_manager[1] is out_qubits[1]

        # Also check with actual jaxpr variables
        with take_current_trace() as trace:
            gate_out_qubits = trace.frame.eqns[-1].outvars
            assert trace.frame.tracer_to_var[id(qreg_manager[0])] == gate_out_qubits[0]
            assert trace.frame.tracer_to_var[id(qreg_manager[1])] == gate_out_qubits[1]

    def test_iter(self):
        """Test __iter__ in the qreg manager"""
        qreg = qalloc_p.bind(42)
        qreg_manager = QregManager(qreg)

        target_dictionary = {}
        for x in range(3):
            target_dictionary[x] = qreg_manager[x]

        assert dict(qreg_manager) == target_dictionary

    def test_chained_gate(self):
        """Test two chained qubit opertaions"""
        qreg = qalloc_p.bind(42)
        qreg_manager = QregManager(qreg)

        wires = [0, 1]
        _ = _interpret_operation(wires, qreg_manager)
        out_qubits = _interpret_operation(wires, qreg_manager)

        # Check that qubit-level ops do not affect the managed qreg
        assert qreg_manager.get() is qreg

        # First check with python handle variables
        assert qreg_manager[0] is out_qubits[0]
        assert qreg_manager[1] is out_qubits[1]

        # Also check with actual jaxpr variables
        with take_current_trace() as trace:
            gate_out_qubits = trace.frame.eqns[-1].outvars
            assert trace.frame.tracer_to_var[id(qreg_manager[0])] == gate_out_qubits[0]
            assert trace.frame.tracer_to_var[id(qreg_manager[1])] == gate_out_qubits[1]

    def test_insert_all_dangling_qubits(self):
        """
        Test insert_all_dangling_qubits.
        Note: In the non-plxpr tracing pipeline, this is the `actualize` method from the
        QregPromise object.
        """
        qreg = qalloc_p.bind(42)
        qreg_manager = QregManager(qreg)

        # Extract some qubits
        _ = [qreg_manager[i] for i in range(3)]
        assert list(qreg_manager.wire_map.keys()) == [0, 1, 2]

        qreg_manager.insert_all_dangling_qubits()
        assert qreg_manager.wire_map == {}
        assert qreg_manager.get() is not qreg  # test that inserts update the qreg SSA value

        with take_current_trace() as trace:
            # Checking via jaxpr internals is a bit tedious here
            # So let's just check the string...
            observed_jaxpr = str(trace.to_jaxpr([], None, None)[0])
            expected = """\
            { lambda ; . let
                a:AbstractQreg() = qalloc 42:i64[]
                b:AbstractQbit() = qextract a 0:i64[]
                c:AbstractQbit() = qextract a 1:i64[]
                d:AbstractQbit() = qextract a 2:i64[]
                e:AbstractQreg() = qinsert a 0:i64[] b
                f:AbstractQreg() = qinsert e 1:i64[] c
                _:AbstractQreg() = qinsert f 2:i64[] d
              in () }"""
            expected = textwrap.dedent(expected)
            assert observed_jaxpr == expected


class TestQregValues:
    """Test QregManager correctly updates qreg SSA values."""

    def test_qreg_op(self):
        """Test that a qreg operation correctly updates the managed qreg SSA value."""
        qreg = qalloc_p.bind(42)
        qreg_manager = QregManager(qreg)
        new_qreg = qreg_mock_op_p.bind(qreg)

        # Update the qreg values.
        # This is user code. This is point 1 of the "users are expected to" in the
        # QregManager specs.
        qreg_manager.set(new_qreg)

        # Check that managed qreg value is updated
        assert qreg_manager.get() is new_qreg

    def test_qreg_op_with_dangling_qubits(self):
        """Test that dangling qubits are correctly disallowed when new qreg values appear."""
        qreg = qalloc_p.bind(42)
        qreg_manager = QregManager(qreg)
        _ = qreg_manager[0]  # extract something
        assert 0 in qreg_manager.wire_map
        new_qreg = qreg_mock_op_p.bind(qreg)  # a qreg op when the qreg still has dangling qubits

        with pytest.raises(
            CompileError,
            match="Setting new qreg value, but the previous one still has dangling qubits.",
        ):
            qreg_manager.set(new_qreg)


class TestQregAndQubit:
    """Test QregManager behaves correctly in full circuits with both qubit and qreg ops."""

    def test_qreg_and_qubit(self):
        """
        Test QregManager behaves correctly in full circuits with both qubit and qreg ops.
        This is the end-to-end system test for the QregManager.
        """

        qreg = qalloc_p.bind(42)
        qreg_manager = QregManager(qreg)
        assert qreg_manager.get() is qreg
        assert qreg_manager.wire_map == {}

        wires = [0, 1]
        out_qubits = _interpret_operation(wires, qreg_manager)
        assert qreg_manager.get() is qreg
        assert list(qreg_manager.wire_map.keys()) == [0, 1]
        assert qreg_manager[0] is out_qubits[0]
        assert qreg_manager[1] is out_qubits[1]

        # Must actualize before expiring an old qreg value as an operand
        qreg_manager.insert_all_dangling_qubits()
        assert qreg_manager.get() is not qreg  # insert creates new qreg result values
        assert qreg_manager.wire_map == {}

        new_qreg = qreg_mock_op_p.bind(qreg_manager.get())
        qreg_manager.set(new_qreg)
        assert qreg_manager.get() is new_qreg
        assert qreg_manager.wire_map == {}

        wires = [0]
        other_out_qubits = _interpret_operation(wires, qreg_manager)
        assert qreg_manager.get() is new_qreg
        assert list(qreg_manager.wire_map.keys()) == [0]
        assert qreg_manager[0] is other_out_qubits[0]

        with take_current_trace() as trace:
            # Check full jaxpr
            observed_jaxpr = str(trace.to_jaxpr([], None, None)[0])
            expected = """\
            { lambda ; . let
                a:AbstractQreg() = qalloc 42:i64[]
                b:AbstractQbit() = qextract a 0:i64[]
                c:AbstractQbit() = qextract a 1:i64[]
                d:AbstractQbit() e:AbstractQbit() = qubit_mock_op b c
                f:AbstractQreg() = qinsert a 0:i64[] d
                g:AbstractQreg() = qinsert f 1:i64[] e
                h:AbstractQreg() = qreg_mock_op g
                i:AbstractQbit() = qextract h 0:i64[]
                _:AbstractQbit() = qubit_mock_op i
              in () }"""
            expected = textwrap.dedent(expected)
            assert observed_jaxpr == expected


if __name__ == "__main__":
    pytest.main(["-x", __file__])
