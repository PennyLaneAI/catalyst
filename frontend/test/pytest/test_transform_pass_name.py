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
"""Testing use of transforms with pass name integrating with qnodes."""

from functools import partial
from typing import Type

import pennylane as qml
import pytest

from catalyst.compiler import CompileError


@pytest.mark.parametrize(
    "options, expected_strings",
    [
        ({"my_string": "blah"}, ['"my-string" = "blah"']),
        ({"on": True, "off": False}, ['"on" = true', '"off" = false']),
        ({"num_iterations": 10}, ['"num-iterations" = 10']),
        (
            {"eps_dec": 0.001, "eps_sci": 1e-3},
            ['"eps-dec" = 1.000000e-03', '"eps-sci" = 1.000000e-03'],
        ),
    ],
    ids=[
        "string",
        "bool",
        "integer",
        "float",
    ],
)
def test_pass_with_options(options, expected_strings, backend):
    """Test the integration for a circuit with a pass that takes in options."""

    my_pass = qml.transform(pass_name="my-pass")

    @qml.qjit(target="mlir")
    @partial(my_pass, **options)
    @qml.qnode(qml.device(backend, wires=1))
    def captured_circuit():
        return qml.expval(qml.PauliZ(0))

    capture_mlir = captured_circuit.mlir
    assert 'transform.apply_registered_pass "my-pass"' in capture_mlir
    for expected_str in expected_strings:
        assert expected_str in str(capture_mlir), f"Expected {expected_str} in MLIR: {capture_mlir}"


@pytest.mark.parametrize(
    "options",
    [
        {"option": [1, 2, 3]},
        {"option": {"blah": "foo"}},
        {"option": None},
    ],
    ids=["list", "dict", "None"],
)
def test_pass_with_unsupported_options(options, backend):
    """Tests that unsupported option types raise a clear error."""

    my_pass = qml.transform(pass_name="my-pass")

    @partial(my_pass, **options)
    @qml.qnode(qml.device(backend, wires=1))
    def captured_circuit():
        return qml.expval(qml.PauliZ(0))

    expected_error = CompileError if options["option"] is None else TypeError
    expected_msg = (
        r"Cannot convert Python type <class 'NoneType'> to an MLIR attribute"
        if options["option"] is None
        else "unhashable type"
    )
    with pytest.raises(expected_error, match=expected_msg):
        qml.qjit(captured_circuit)


def test_pass_before_tape_transform(backend):
    """Test that provided an mlir-only transform prior to a tape transform raises an error."""

    my_pass = qml.transform(pass_name="my-pass")

    @qml.transform
    def tape_transform(tape):
        return (tape,), lambda x: x[0]

    @qml.qjit
    @tape_transform
    @my_pass
    @qml.qnode(qml.device(backend, wires=1))
    def f(x):  # pylint: disable=unused-argument
        return qml.state()

    with pytest.raises(ValueError, match="without a tape definition occurs before tape transform"):
        f(0.5)


def test_pass_after_tape_transform(backend):
    """Test that passes can be applied after tape transforms."""

    @qml.transform
    def tape_only_cancel_inverses(tape):
        return qml.transforms.cancel_inverses(tape)

    my_pass = qml.transform(pass_name="my-pass")

    @qml.qjit(target="mlir")
    @my_pass
    @tape_only_cancel_inverses
    @qml.qnode(qml.device(backend, wires=1))
    def c():
        qml.X(0)
        qml.X(0)
        return qml.state()

    # check inverses canceled
    c_mlir = c.mlir
    assert 'quantum.custom "PauliX"()' not in c_mlir
    assert 'transform.apply_registered_pass "my-pass"' in c_mlir


def op_has_attr(op, attr):
    """Check if an MLIR operation has the specified attribute."""
    attrs = getattr(op, "attributes", [])
    return attr in attrs or attr in getattr(attrs, "keys", lambda: [])()


@pytest.mark.xdsl
@pytest.mark.usefixtures("use_both_frontend")
def test_xdsl_pass_with_qml_transform():
    """Test that applying xDSL passes using the ``qml.transform`` decorator is able to execute
    correctly."""

    @qml.qjit
    @qml.transform(pass_name="xdsl-cancel-inverses")
    @qml.qnode(qml.device("null.qubit", wires=1))
    def c():
        qml.X(0)
        qml.X(0)
        return qml.state()

    mod = c.mlir_module.operation
    named_sequence_mod = None
    for op in mod.regions[0].blocks[0].operations:
        if getattr(op, "name", "") == "builtin.module":
            for _op in op.regions[0].blocks[0].operations:
                if getattr(_op, "name", "") == "builtin.module":
                    named_sequence_mod = _op
                    break
            break

    assert named_sequence_mod is not None
    assert op_has_attr(named_sequence_mod, "catalyst.uses_xdsl_passes")
    assert op_has_attr(named_sequence_mod, "transform.with_named_sequence")

    named_sequence_op = next(iter(named_sequence_mod.regions[0].blocks[0].operations), None)
    assert named_sequence_op is not None
    first_transform_op = next(iter(named_sequence_op.regions[0].blocks[0].operations), None)
    assert first_transform_op is not None

    assert op_has_attr(first_transform_op, "catalyst.xdsl_pass")
    assert first_transform_op.attributes["pass_name"].value == "xdsl-cancel-inverses"

    assert qml.math.allclose(c(), [1, 0])


if __name__ == "__main__":
    pytest.main(["-x", __file__])
