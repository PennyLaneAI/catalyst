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
This module tests the from_plxpr QubitHandler.

Quoted from the object's docstring:

    With the above in mind, this `QubitHandler` class promises the following:
    1. An instance of this class will always be tied to one root qreg value.
    2. At any moment during the from_plxpr conversion:
       - `QubitHandler.get()` returns the current catalyst qreg SSA value for the managed
          root register on that instance;
       - `QubitHandler[i]` returns the current catalyst qubit SSA value for the i-th index
          on the managed root register on that instance. If none exists, a new qubit will be
          extracted.

    To achieve the above, users of this class are expected to:
    1. Initialize an instance with the qreg SSA value from a new allocation or a new block argument;
    2. Whenever a new meta-op/op is bind-ed, update the current qreg/qubit SSA value with:
       - `QubitHandler.set(new_qreg_value)`
       - `QubitHandler[i] = new_qubit_value`
"""
# pylint: disable=use-implicit-booleaness-not-comparison
import textwrap

import pytest
from jax._src.core import Literal
from jax._src.interpreters.partial_eval import DynamicJaxprTracer
from jax.api_util import debug_info as jdb
from jax.core import set_current_trace, take_current_trace
from jax.extend.core import Primitive
from jax.interpreters.partial_eval import DynamicJaxprTrace

from catalyst.from_plxpr.qubit_handler import QubitHandler, QubitIndexRecorder
from catalyst.jax_primitives import AbstractQbit, AbstractQreg, qalloc_p, qextract_p
from catalyst.jax_tracer import _get_eqn_from_tracing_eqn
from catalyst.utils.exceptions import CompileError


@pytest.fixture(autouse=True)
def launch_empty_jaxpr_interpreter():
    """
    Start an empty interpreter for each test.
    The interpreter we want is a DynamicJaxprTrace, which is an interpreter for creating jaxprs.
    """
    # Note that DynamicJaxprTrace requires a `debug_info` object
    # We just pass an empty one
    trace = DynamicJaxprTrace(debug_info=jdb("qubit_handler_from_plxpr_test", None, [], {}))
    with set_current_trace(trace):
        yield
    del trace


# A mock qubit index recorder.
# Tests in this file generally do not need it.
# Tests that actually need a meaning recorder should create their own.
mock_recorder = QubitIndexRecorder()

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


def _interpret_operation(wires, qubit_handler):
    """
    Convenience helper for binding a mock gate.
    Note that this helper follows plxpr semantics, so gate targets are only specified
    by a global wire index.
    """
    in_qubits = [qubit_handler[w] for w in wires]
    out_qubits = qubit_mock_op_p.bind(*in_qubits)

    # Update the qubit values.
    # This is user code. This is point 2 of the "users are expected to" in the
    # QubitHandler specs.
    for wire, out_qubit in zip(wires, out_qubits):
        qubit_handler[wire] = out_qubit

    return out_qubits


class TestExtractInsertWithNoQreg:
    """Test that the correct CompileErrors are raised when no qreg is set"""

    def test_errors_noqreg(self):
        """Test that extracting a qubit from a QubitHandler raises NotImplementedError"""
        qbits = [AbstractQbit(), AbstractQbit()]
        qubit_handler = QubitHandler(qbits, mock_recorder)
        assert qubit_handler.abstract_qreg_val is None

        with pytest.raises(CompileError, match="Cannot extract a qubit at index 0"):
            qubit_handler.extract(0)

        with pytest.raises(CompileError, match="Cannot insert a qubit at index 100"):
            qubit_handler.insert(100, "monkey_mock_qubit")

        with pytest.raises(CompileError, match="Cannot insert qubits back into a qreg"):
            qubit_handler.insert_all_dangling_qubits()

        with pytest.raises(CompileError, match="Cannot insert dynamic qubits back into a qreg"):
            qubit_handler.insert_dynamic_qubits(0)


class TestQubitHandlerInitGetSet:
    """Unit test for getter and setter for QubitHandler"""

    def test_getter_setter(self):
        """Test getter and setter with a qreg"""
        qreg = qalloc_p.bind(42)
        qubit_handler = QubitHandler(qreg, mock_recorder)
        qubit_handler.set("monkey_mock_qreg")
        assert qubit_handler.get() == "monkey_mock_qreg"
        assert qubit_handler.get() == "monkey_mock_qreg"  # test that getter does not set

        qubit_handler.set("donkey_mock_qreg")
        assert qubit_handler.get() == "donkey_mock_qreg"

    def test_getter_setter_no_qreg(self):
        """Test getter and setter when no qreg is set"""
        qubit_handler = QubitHandler([AbstractQbit(), AbstractQbit()], mock_recorder)

        assert isinstance(qubit_handler.get(), list)
        assert all(isinstance(q, AbstractQbit) for q in qubit_handler.get())
        assert len(qubit_handler.get()) == 2

        qubit_handler.set("monkey_mock_qubit")
        assert qubit_handler.get() == []

    def test_getitem_setitem(self):
        """Test getting and setting items"""

        q0 = AbstractQbit()
        qubit_handler = QubitHandler([q0], mock_recorder)

        assert qubit_handler[q0] == q0

        q1 = AbstractQbit()
        qubit_handler[q1] = q1
        assert qubit_handler[q1] == q1

        qubit_handler[0] = q1
        assert qubit_handler[0] == q1


class TestQubitHandlerInitialization:
    """Test initialization of QubitHandler objects."""

    def test_init_with_qreg(self):
        """Test initialization of QubitHandler with a qreg"""
        qreg = qalloc_p.bind(42)
        qubit_handler = QubitHandler(qreg, mock_recorder)

        assert not qubit_handler.is_qubit_mode()
        assert qubit_handler.get() is qreg
        assert qubit_handler.wire_map == {}

        # Check that the qreg SSA value is correctly set
        assert isinstance(qubit_handler.get(), DynamicJaxprTracer)

    def test_init_with_qubits(self):
        """Test initialization of QubitHandler with a list of qubits"""
        qubits = [AbstractQbit() for _ in range(3)]
        qubit_handler = QubitHandler(qubits, mock_recorder)

        assert qubit_handler.is_qubit_mode()
        assert isinstance(qubit_handler.get(), list)
        assert all(isinstance(q, AbstractQbit) for q in qubit_handler.get())

        assert qubit_handler.get() == qubits

    def test_init_with_empty(self):
        """Test initialization of QubitHandler with an empty list"""
        qubit_handler = QubitHandler([], mock_recorder)

        assert qubit_handler.get() == []
        assert qubit_handler.wire_map == {}

        # Check that the qreg SSA value is None
        assert qubit_handler.abstract_qreg_val is None

    def test_new_alloc(self):
        """Test qregs from new alloc"""
        qreg = qalloc_p.bind(42)
        qubit_handler = QubitHandler(qreg, mock_recorder)
        assert qubit_handler.get() is qreg
        assert qubit_handler.wire_map == {}

    def test_scope_arg(self):
        """Test qregs from a scope argument"""

        def f(qreg):
            qubit_handler = QubitHandler(qreg, mock_recorder)
            assert qubit_handler.get() is qreg
            assert qubit_handler.wire_map == {}

        outer_scope_qreg = qalloc_p.bind(42)
        f(outer_scope_qreg)


class TestQubitValues:
    """Test QubitHandler correctly updates qubit SSA values."""

    def test_auto_extract(self):
        """Test that a new qubit is extracted when indexing into a new wire"""
        qreg = qalloc_p.bind(42)
        qubit_handler = QubitHandler(qreg, mock_recorder)
        new_qubit = qubit_handler[0]

        assert list(qubit_handler.wire_map.keys()) == [0]
        assert qubit_handler[0] is new_qubit
        with take_current_trace() as trace:
            # Check that an extract primitive is added
            last_eqn = _get_eqn_from_tracing_eqn(trace.frame.tracing_eqns[-1])
            assert last_eqn.primitive is qextract_p

            # Check that the extract primitive follows the wire index in the qreg manager
            # __getitem__ method
            extract_p_index_invar = last_eqn.invars[-1]
            if isinstance(extract_p_index_invar, Literal):
                assert extract_p_index_invar.val == 0
            else:
                assert isinstance(extract_p_index_invar.val, Literal)
                assert extract_p_index_invar.val.val == 0

    def test_no_overwriting_extract(self):
        """Test that no new qubit is extracted when indexing into an existing wire"""
        qreg = qalloc_p.bind(42)
        qubit_handler = QubitHandler(qreg, mock_recorder)
        new_qubit = qubit_handler[0]
        not_a_new_qubit = qubit_handler[0]

        assert not_a_new_qubit is new_qubit

    def test_simple_gate(self):
        """Test a simple qubit opertaion"""
        qreg = qalloc_p.bind(42)
        qubit_handler = QubitHandler(qreg, mock_recorder)

        wires = [0, 1]
        out_qubits = _interpret_operation(wires, qubit_handler)

        # Check that qubit-level ops do not affect the managed qreg
        assert qubit_handler.get() is qreg

        # First check with python handle variables
        assert qubit_handler[0] is out_qubits[0]
        assert qubit_handler[1] is out_qubits[1]

        # Also check with actual jaxpr variables
        with take_current_trace() as trace:
            last_eqn = _get_eqn_from_tracing_eqn(trace.frame.tracing_eqns[-1])
            gate_out_qubits = last_eqn.outvars
            assert qubit_handler[0].val == gate_out_qubits[0]
            assert qubit_handler[1].val == gate_out_qubits[1]

    def test_iter(self):
        """Test __iter__ in the qreg manager"""
        qreg = qalloc_p.bind(42)
        qubit_handler = QubitHandler(qreg, mock_recorder)

        target_dictionary = {}
        for x in range(3):
            target_dictionary[x] = qubit_handler[x]

        assert dict(qubit_handler) == target_dictionary

    def test_iter_noqreg(self):
        """Test that iterating over a QubitHandler works as expected"""
        q0 = AbstractQbit()
        q1 = AbstractQbit()
        qubit_handler = QubitHandler([q0, q1], mock_recorder)

        target_dictionary = {q0: q0, q1: q1}
        assert dict(qubit_handler) == target_dictionary

    def test_chained_gate(self):
        """Test two chained qubit opertaions"""
        qreg = qalloc_p.bind(42)
        qubit_handler = QubitHandler(qreg, mock_recorder)

        wires = [0, 1]
        _ = _interpret_operation(wires, qubit_handler)
        out_qubits = _interpret_operation(wires, qubit_handler)

        # Check that qubit-level ops do not affect the managed qreg
        assert qubit_handler.get() is qreg

        # First check with python handle variables
        assert qubit_handler[0] is out_qubits[0]
        assert qubit_handler[1] is out_qubits[1]

        # Also check with actual jaxpr variables
        with take_current_trace() as trace:
            last_eqn = _get_eqn_from_tracing_eqn(trace.frame.tracing_eqns[-1])
            gate_out_qubits = last_eqn.outvars
            assert qubit_handler[0].val == gate_out_qubits[0]
            assert qubit_handler[1].val == gate_out_qubits[1]

    def test_insert_all_dangling_qubits(self):
        """
        Test insert_all_dangling_qubits.
        Note: In the non-plxpr tracing pipeline, this is the `actualize` method from the
        QregPromise object.
        """
        qreg = qalloc_p.bind(42)
        qubit_handler = QubitHandler(qreg, mock_recorder)

        # Extract some qubits
        _ = [qubit_handler[i] for i in range(3)]
        assert list(qubit_handler.wire_map.keys()) == [0, 1, 2]

        qubit_handler.insert_all_dangling_qubits()
        assert qubit_handler.wire_map == {}
        assert qubit_handler.get() is not qreg  # test that inserts update the qreg SSA value

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
    """Test QubitHandler correctly updates qreg SSA values."""

    def test_qreg_op(self):
        """Test that a qreg operation correctly updates the managed qreg SSA value."""
        qreg = qalloc_p.bind(42)
        qubit_handler = QubitHandler(qreg, mock_recorder)
        new_qreg = qreg_mock_op_p.bind(qreg)

        # Update the qreg values.
        # This is user code. This is point 1 of the "users are expected to" in the
        # QubitHandler specs.
        qubit_handler.set(new_qreg)

        # Check that managed qreg value is updated
        assert qubit_handler.get() is new_qreg

    def test_qreg_op_with_dangling_qubits(self):
        """Test that dangling qubits are correctly disallowed when new qreg values appear."""
        qreg = qalloc_p.bind(42)
        qubit_handler = QubitHandler(qreg, mock_recorder)
        _ = qubit_handler[0]  # extract something
        assert 0 in qubit_handler.wire_map
        new_qreg = qreg_mock_op_p.bind(qreg)  # a qreg op when the qreg still has dangling qubits

        with pytest.raises(
            CompileError,
            match="Setting new qreg value, but the previous one still has dangling qubits.",
        ):
            qubit_handler.set(new_qreg)


class TestQregAndQubit:
    """Test QubitHandler behaves correctly in full circuits with both qubit and qreg ops."""

    def test_qreg_and_qubit(self):
        """
        Test QubitHandler behaves correctly in full circuits with both qubit and qreg ops.
        This is the end-to-end system test for the QubitHandler.
        """

        qreg = qalloc_p.bind(42)
        qubit_handler = QubitHandler(qreg, mock_recorder)
        assert qubit_handler.get() is qreg
        assert qubit_handler.wire_map == {}

        wires = [0, 1]
        out_qubits = _interpret_operation(wires, qubit_handler)
        assert qubit_handler.get() is qreg
        assert list(qubit_handler.wire_map.keys()) == [0, 1]
        assert qubit_handler[0] is out_qubits[0]
        assert qubit_handler[1] is out_qubits[1]

        # Must actualize before expiring an old qreg value as an operand
        qubit_handler.insert_all_dangling_qubits()
        assert qubit_handler.get() is not qreg  # insert creates new qreg result values
        assert qubit_handler.wire_map == {}

        new_qreg = qreg_mock_op_p.bind(qubit_handler.get())
        qubit_handler.set(new_qreg)
        assert qubit_handler.get() is new_qreg
        assert qubit_handler.wire_map == {}

        wires = [0]
        other_out_qubits = _interpret_operation(wires, qubit_handler)
        assert qubit_handler.get() is new_qreg
        assert list(qubit_handler.wire_map.keys()) == [0]
        assert qubit_handler[0] is other_out_qubits[0]

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


class TestQubitIndexRecorder:
    """Test behavior of QubitIndexRecorder."""

    def test_standalone_behavior(self):
        """
        Test that methods in QubitIndexRecorder behave correctly with mock values.
        """
        recorder = QubitIndexRecorder()
        assert recorder.map == {}

        recorder["mock"] = "mockmock"
        assert recorder.contains("mock")
        assert not recorder.contains("not mock")
        assert recorder["mock"] == "mockmock"

    def test_qubit_extract_recording(self):
        """
        Test that extracting a qubit value from `QubitHandler` correctly updates the recorder.
        """
        recorder = QubitIndexRecorder()

        qreg = qalloc_p.bind(42)
        qubit_handler = QubitHandler(qreg, recorder)
        other_qreg = qalloc_p.bind(42)
        other_qubit_handler = QubitHandler(other_qreg, recorder)

        wires = [0, 1]
        _ = _interpret_operation(wires, qubit_handler)
        other_wires = [10, 20]
        _ = _interpret_operation(other_wires, other_qubit_handler)

        assert list(recorder.map.keys()) == [0, 1, 10, 20]
        assert recorder[0] is qubit_handler
        assert recorder[1] is qubit_handler
        assert recorder[10] is other_qubit_handler
        assert recorder[20] is other_qubit_handler


if __name__ == "__main__":
    pytest.main(["-x", __file__])
