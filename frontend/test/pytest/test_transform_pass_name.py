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

import pennylane as qp
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

    my_pass = qp.transform(pass_name="my-pass")

    @qp.qjit(target="mlir")
    @partial(my_pass, **options)
    @qp.qnode(qp.device(backend, wires=1))
    def captured_circuit():
        return qp.expval(qp.PauliZ(0))

    capture_mlir = captured_circuit.mlir
    assert 'transform.apply_registered_pass "my-pass"' in capture_mlir
    for expected_str in expected_strings:
        assert expected_str in str(capture_mlir), f"Expected {expected_str} in MLIR: {capture_mlir}"


@pytest.mark.parametrize(
    "options",
    [
        {"option": [1, 2, "blah"]},
        {"option": {"1": 2, "blah": "foo"}},
    ],
    ids=["list", "dict"],
)
def test_pass_with_complex_options(options, backend, capture_mode):
    """Tests that complex options like list, dict are supported."""

    my_pass = qp.transform(pass_name="my-pass")

    @qp.qjit(target="mlir", capture=capture_mode)
    @partial(my_pass, **options)
    @qp.qnode(qp.device(backend, wires=1))
    def captured_circuit():
        return qp.expval(qp.PauliZ(0))

    if isinstance(options["option"], list):
        assert 'with options = {"option" = [1 : i64, 2 : i64, "blah"]}' in captured_circuit.mlir

    if isinstance(options["option"], dict):
        assert 'with options = {"option" = {"1" = 2 : i64, blah = "foo"}}' in captured_circuit.mlir


def test_pass_with_none_option(backend):
    """Tests that None option values are lowered to the MLIR null attribute."""

    my_pass = qp.transform(pass_name="my-pass")

    @qp.qjit(target="mlir")
    @partial(my_pass, **{"option": None})
    @qp.qnode(qp.device(backend, wires=1))
    def captured_circuit():
        return qp.expval(qp.PauliZ(0))

    assert '"option" = none' in captured_circuit.mlir


def test_pass_with_unsupported_options(backend):
    """Tests that unsupported option types raise a clear error."""

    my_pass = qp.transform(pass_name="my-pass")

    @partial(my_pass, **{"option": object()})
    @qp.qnode(qp.device(backend, wires=1))
    def captured_circuit():
        return qp.expval(qp.PauliZ(0))

    expected_msg = r"Cannot convert Python type <class 'object'> to an MLIR attribute"
    qjc = qp.qjit(target="mlir")(captured_circuit)

    with pytest.raises(CompileError, match=expected_msg):
        qjc()


def test_pass_plugin_transform(mocker, tmp_path, backend):
    """Test that a transform from a pass plugin works with capture disabled."""

    plugin_path = tmp_path / "FakePlugin.so"
    plugin_path.touch()

    mock_plugin_module = mocker.Mock()
    mock_plugin_module.name2pass.return_value = (plugin_path, "my-custom-pass")

    mock_entry_point = mocker.Mock()
    mock_entry_point.load.return_value = mock_plugin_module
    mocker.patch(
        "catalyst.qfunc.entry_points",
        return_value={"fakeplugin.passes": mock_entry_point},
    )

    plugin_pass = qp.transform(pass_name="fakeplugin.my-custom-pass")

    @qp.qjit(target="mlir", capture=False)
    @plugin_pass
    @qp.qnode(qp.device(backend, wires=1))
    def circuit():
        return qp.expval(qp.PauliZ(0))

    assert circuit.compile_options.pass_plugins == {plugin_path}
    assert circuit.compile_options.dialect_plugins == {plugin_path}
    assert 'transform.apply_registered_pass "my-custom-pass"' in circuit.mlir
    mock_plugin_module.name2pass.assert_called_once_with("my-custom-pass")


def test_pass_plugin_transform_with_options(mocker, tmp_path, backend):
    """Test that a pass plugin transform forwards options under capture=False."""

    plugin_path = tmp_path / "FakePlugin.so"
    plugin_path.touch()

    mock_plugin_module = mocker.Mock()
    mock_plugin_module.name2pass.return_value = (plugin_path, "my-custom-pass")

    mock_entry_point = mocker.Mock()
    mock_entry_point.load.return_value = mock_plugin_module
    mocker.patch(
        "catalyst.qfunc.entry_points",
        return_value={"fakeplugin.passes": mock_entry_point},
    )

    plugin_pass = qp.transform(pass_name="fakeplugin.my-custom-pass")

    @qp.qjit(target="mlir", capture=False)
    @partial(plugin_pass, my_option="my_option_value", my_other_option=False)
    @qp.qnode(qp.device(backend, wires=1))
    def circuit():
        return qp.expval(qp.PauliZ(0))

    mlir = circuit.mlir
    assert circuit.compile_options.pass_plugins == {plugin_path}
    assert 'transform.apply_registered_pass "my-custom-pass"' in mlir
    assert 'with options = {"my-option" = "my_option_value", "my-other-option" = false}' in mlir


def test_pass_plugin_transform_with_multiple_dots(mocker, tmp_path, backend):
    """Test that a pass_name with more than one '.' splits only on the first '.'."""

    plugin_path = tmp_path / "FakePlugin.so"
    plugin_path.touch()

    mock_plugin_module = mocker.Mock()
    mock_plugin_module.name2pass.return_value = (plugin_path, "my.nested.pass")

    mock_entry_point = mocker.Mock()
    mock_entry_point.load.return_value = mock_plugin_module
    mocker.patch(
        "catalyst.qfunc.entry_points",
        return_value={"fakeplugin.passes": mock_entry_point},
    )

    plugin_pass = qp.transform(pass_name="fakeplugin.my.nested.pass")

    @qp.qjit(target="mlir", capture=False)
    @plugin_pass
    @qp.qnode(qp.device(backend, wires=1))
    def circuit():
        return qp.expval(qp.PauliZ(0))

    assert circuit.compile_options.pass_plugins == {plugin_path}
    assert 'transform.apply_registered_pass "my.nested.pass"' in circuit.mlir
    mock_plugin_module.name2pass.assert_called_once_with("my.nested.pass")


def test_pass_plugin_not_installed(mocker, backend):
    """Test that a clear error is raised when the pass plugin is not installed."""

    mocker.patch(
        "catalyst.qfunc.entry_points",
        return_value={},
    )

    missing_plugin_pass = qp.transform(pass_name="missingplugin.my-custom-pass")

    @missing_plugin_pass
    @qp.qnode(qp.device(backend, wires=1))
    def circuit():
        return qp.expval(qp.PauliZ(0))

    with pytest.raises(
        ValueError,
        match="No pass plugin registered under 'missingplugin'.*Is the plugin package installed",
    ):
        qp.qjit(circuit, capture=False)()


def test_pass_before_tape_transform(backend):
    """Test that provided an mlir-only transform prior to a tape transform raises an error."""

    my_pass = qp.transform(pass_name="my-pass")

    @qp.transform
    def tape_transform(tape):
        return (tape,), lambda x: x[0]

    @qp.qjit
    @tape_transform
    @my_pass
    @qp.qnode(qp.device(backend, wires=1))
    def f(x):  # pylint: disable=unused-argument
        return qp.state()

    with pytest.raises(ValueError, match="without a tape definition occurs before tape transform"):
        f(0.5)


def test_pass_after_tape_transform(backend):
    """Test that passes can be applied after tape transforms."""

    @qp.transform
    def tape_only_cancel_inverses(tape):
        return qp.transforms.cancel_inverses(tape)

    my_pass = qp.transform(pass_name="my-pass")

    @qp.qjit(target="mlir")
    @my_pass
    @tape_only_cancel_inverses
    @qp.qnode(qp.device(backend, wires=1))
    def c():
        qp.X(0)
        qp.X(0)
        return qp.state()

    # check inverses canceled
    c_mlir = c.mlir
    assert 'quantum.custom "PauliX"()' not in c_mlir
    assert 'transform.apply_registered_pass "my-pass"' in c_mlir


def op_has_attr(op, attr):
    """Check if an MLIR operation has the specified attribute."""
    attrs = getattr(op, "attributes", [])
    return attr in attrs or attr in getattr(attrs, "keys", lambda: [])()


@pytest.mark.xdsl
def test_xdsl_pass_with_qp_transform(capture_mode):
    """Test that applying xDSL passes using the ``qp.transform`` decorator is able to execute
    correctly."""

    @qp.qjit(capture=capture_mode)
    @qp.transform(pass_name="xdsl-cancel-inverses")
    @qp.qnode(qp.device("null.qubit", wires=1))
    def c():
        qp.X(0)
        qp.X(0)
        return qp.state()

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

    assert qp.math.allclose(c(), [1, 0])


if __name__ == "__main__":
    pytest.main(["-x", __file__])
