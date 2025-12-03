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

import pennylane as qml
import pytest


def test_pass_with_options(backend):
    """Test the integration for a circuit with a pass that takes in options."""

    my_pass = qml.transform(pass_name="my-pass")

    @qml.qjit(target="mlir")
    @partial(my_pass, my_option="my_option_value", my_other_option=False)
    @qml.qnode(qml.device(backend, wires=1))
    def captured_circuit():
        return qml.expval(qml.PauliZ(0))

    capture_mlir = captured_circuit.mlir
    assert 'transform.apply_registered_pass "my-pass"' in capture_mlir
    assert (
        'with options = {"my-option" = "my_option_value", "my-other-option" = false}'
        in capture_mlir
    )


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
