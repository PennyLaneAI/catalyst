# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test suite for the catalyst.python_interface.transforms.qecp.qec_code_lib module."""

import functools
import re

import numpy as np
import pytest
from xdsl.ir import Operation

from catalyst.python_interface.dialects import qecp
from catalyst.python_interface.transforms.qecp.qec_code_lib import (
    QecCode,
    SupportedGates,
    qecp_gate_op_from_string,
)

SUPPORTED_CODES = ["Steane"]


class TestQecCode:
    """Units tests for the QecCode class."""

    @pytest.mark.parametrize(
        "name, n, k, d", [("Steane", 7, 1, 3), ("Shor", 9, 1, 3), ("Surface_d3", 17, 1, 3)]
    )
    def test_constructor(self, name: str, n: int, k: int, d: int):
        """Test the constructor of the `QecCode` class for various QEC codes.

        Note that some parameters of the code are placeholders only and do not necessarily represent
        a true QEC code.
        """
        qec_code = QecCode(
            name,
            n,
            k,
            d,
            np.eye(n),
            np.array([1] * n),
            transversal_1q_gates={"x": ("X",) * n},
            transversal_2q_gates={"cnot": "CNOT"},
            unitary_encoding={
                "state_prep_index": 2,
                "hadamard_indices": [0, 1],
                "cnot_indices": ([0, 1], [1, 2]),
            },
        )

        assert qec_code.name == name
        assert qec_code.n == n
        assert qec_code.k == k
        assert qec_code.d == d
        assert np.all(qec_code.x_tanner == np.eye(n))
        assert np.all(qec_code.z_tanner == np.array([1] * n))
        assert qec_code.transversal_1q_gates == {"x": ("X",) * n}
        assert qec_code.transversal_2q_gates == {"cnot": "CNOT"}
        assert qec_code.unitary_encoding == {
            "state_prep_index": 2,
            "hadamard_indices": [0, 1],
            "cnot_indices": ([0, 1], [1, 2]),
        }

    @pytest.mark.parametrize(
        "inputs, expected_str",
        [
            (("Steane", 7, 1, 3), "[[7, 1, 3]] Steane"),
            (("", 7, 1, 3), "[[7, 1, 3]] <unknown>"),
            (("  ", 7, 1, 3), "[[7, 1, 3]] <unknown>"),
        ],
    )
    def test_str_representation(self, inputs: tuple[str, int, int, int], expected_str: str):
        """Test the string representation of the `QecCode` class for various inputs."""
        _, n, _, _ = inputs
        qec_code = QecCode(*inputs, np.eye(n), np.eye(n), {}, {}, {})
        assert str(qec_code) == expected_str

    @pytest.mark.parametrize(
        "data",
        [
            {
                "name": "Steane",
                "n": 7,
                "k": 1,
                "d": 3,
                "x_tanner": np.eye(7),
                "z_tanner": np.array([[0, 0, 1, 1, 0, 1, 1]]),
                "transversal_1q_gates": {"x": ("X",) * 7},
                "transversal_2q_gates": {},
                "unitary_encoding": {},
            },
            {
                "name": "Shor",
                "n": 9,
                "k": 1,
                "d": 3,
                "x_tanner": np.eye(9),
                "z_tanner": np.array([[0, 0, 1, 1, 0, 1, 1, 0, 1]]),
                "transversal_1q_gates": {
                    "y": ("Y",) * 9,
                    "hadamard": ("H",) * 9,
                },
                "transversal_2q_gates": {"cnot": "CNOT"},
                "unitary_encoding": {},
            },
            {
                "name": "Unknown",
                "n": 7,
                "k": 1,
                "d": 3,
                "x_tanner": np.eye(7),
                "z_tanner": np.array([[0, 0, 1, 1, 0, 1, 1]]),
                "extra-field": 42,
                "transversal_1q_gates": {"z": ("Z",) * 7},
                "transversal_2q_gates": {"cnot": "CNOT"},
                "unitary_encoding": {
                    "state_prep_index": 2,
                    "ops": [("CNOT", [0, 1]), ("H", [1])],
                },
            },
        ],
    )
    def test_from_dict(self, data: dict):
        """Test constructing a `QecCode` object from a dictionary using the `from_dict()` method."""
        qec_code = QecCode.from_dict(data)

        assert qec_code.name == data["name"]
        assert qec_code.n == data["n"]
        assert qec_code.k == data["k"]
        assert qec_code.d == data["d"]
        assert np.all(qec_code.x_tanner == data["x_tanner"])
        assert np.all(qec_code.z_tanner == data["z_tanner"])
        assert qec_code.transversal_1q_gates == data["transversal_1q_gates"]
        assert qec_code.transversal_2q_gates == data["transversal_2q_gates"]
        assert qec_code.unitary_encoding == data["unitary_encoding"]

    @pytest.mark.parametrize(
        "data",
        [
            {
                "name": "Steane",
                "n": 7,
                "k": 1,
                "d": 3,
                "x_tanner": np.eye(7),
                "z_tanner": np.array([[0, 0, 1, 1, 0, 1, 1]]),
                "transversal_1q_gates": {"x": ("X",) * 7},
                "transversal_2q_gates": {"cnot": "CNOT"},
                "unitary_encoding": {},
            },
            {
                "name": "Shor",
                "n": 9,
                "k": 1,
                "d": 3,
                "x_tanner": np.eye(9),
                "z_tanner": np.array([[0, 0, 1, 1, 0, 1, 1, 0, 1]]),
                "transversal_1q_gates": {"x": ("X",) * 9},
                "transversal_2q_gates": {},
                "unitary_encoding": {
                    "state_prep_index": 2,
                    "ops": [("CNOT", [0, 1]), ("H", [1])],
                },
            },
        ],
    )
    def test_constructor_with_dict_input(self, data: dict):
        """Test constructing a `QecCode` object from a dictionary using the default constructor.

        Note that constructing a `QecCode` object in this way is generally discouraged, but is
        necessary in order to parse MLIR dictionary attributes from a
        `transform.apply_registered_pass` op and construct an xDSL pass object that uses a `QecCode`
        object as a pass option.
        """
        qec_code = QecCode(**data)

        assert qec_code.name == data["name"]
        assert qec_code.n == data["n"]
        assert qec_code.k == data["k"]
        assert qec_code.d == data["d"]
        assert np.all(qec_code.x_tanner == data["x_tanner"])
        assert np.all(qec_code.z_tanner == data["z_tanner"])
        assert qec_code.transversal_1q_gates == data["transversal_1q_gates"]
        assert qec_code.transversal_2q_gates == data["transversal_2q_gates"]
        assert qec_code.unitary_encoding == data["unitary_encoding"]

    @pytest.mark.parametrize("d, expected_t", [(1, 0), (2, 0), (3, 1), (4, 1), (5, 2), (6, 2)])
    def test_correctable_errors_property(self, d: int, expected_t: int):
        """Test the `correctable_errors` property of `QecCode`, which returns the number of
        correctable errors of the code, t = floor((d - 1) / 2).
        """
        qec_code = QecCode(
            "",
            1,
            1,
            d,
            np.array([]),
            np.array([]),
            {},
            {},
            {},
        )  # only value of d matters

        assert qec_code.correctable_errors == expected_t

    @pytest.mark.parametrize("name", SUPPORTED_CODES)
    def test_get(self, name: str):
        """Test the `QecCode.get()` method for all supported QEC codes."""
        qec_code = QecCode.get(name)

        assert qec_code.name == name

    @pytest.mark.parametrize("name", ["Sgt. Pepper's Lonely Hearts Club Code", None, 1])
    def test_get_unsupported_code(self, name):
        """Test that the `QecCode.get()` method raises an error for supported QEC codes."""
        with pytest.raises(KeyError, match=re.compile(r"QEC code .* not found")):
            QecCode.get(name)

    @pytest.mark.parametrize(
        "transversal_1q_gates",
        [
            ({"x": ()}),
            ({"x": ("X",)}),
            ({"x": ("X", "Z")}),
            ({"x": ("X",), "z": ("Z",)}),
            ({"x": ("X",), "z": ("Z", "Z")}),
            ({"x": ("X", "X", "X", "X")}),
        ],
    )
    def test_invalid_transversal_1q_gate_definition(
        self, transversal_1q_gates: dict[str, tuple[str, ...]]
    ):
        """Test that defining a QEC code with invalid single-qubit gate definitions raises an error.

        The definition of a single-qubit transversal gate is invalid if the number of operators
        given is not equal to the size of the physical codeblock, n.
        """
        n = 3

        with pytest.raises(ValueError, match="Invalid single-qubit transversal gate definition"):
            _ = QecCode(
                name="TestCode",
                n=n,
                k=1,
                d=3,
                x_tanner=np.ones(n, dtype=int),
                z_tanner=np.ones(n, dtype=int),
                transversal_1q_gates=transversal_1q_gates,
                transversal_2q_gates={"cnot": "CNOT"},
                unitary_encoding={},
            )


class TestGateStringIds:
    """TODO"""

    @pytest.mark.parametrize(
        "gate_str, expected_op",
        [
            ("I", qecp.IdentityOp),
            ("X", qecp.PauliXOp),
            ("Y", qecp.PauliYOp),
            ("Z", qecp.PauliZOp),
            ("H", qecp.HadamardOp),
            ("S", qecp.SOp),
            ("CNOT", qecp.CnotOp),
        ],
    )
    def test_qecp_gate_op_from_string_valid_standard_gates(
        self, gate_str: str, expected_op: Operation
    ):
        """Test that all standard, non-partial valid gates return the exact operation class."""
        result = qecp_gate_op_from_string(gate_str)
        assert result is expected_op

    @pytest.mark.parametrize(
        "gate_str, expected_op", [("Sa", functools.partial(qecp.SOp, adjoint=True))]
    )
    def test_qecp_gate_op_from_string_adjoint_gate(
        self, gate_str: str, expected_op: functools.partial
    ):
        """Test that the 'Sa' gate returns a functools.partial object configured correctly."""
        result = qecp_gate_op_from_string(gate_str)

        assert isinstance(result, functools.partial)
        assert result.func is expected_op.func
        assert result.keywords == expected_op.keywords

    @pytest.mark.parametrize(
        "invalid_gate",
        [
            "NOT_A_GATE",
            "i",  # Test case sensitivity
            "",  # Test empty string
            "CX",  # Common alternative naming not in SupportedGates
        ],
    )
    def test_qecp_gate_op_from_string_invalid_raises_value_error(self, invalid_gate: str):
        """Test that invalid gate strings raise a ValueError with a helpful message."""
        expected_msg = (
            f"Invalid gate in QEC code definition: '{invalid_gate}'. "
            "Supported gates are: I, X, Y, Z, H, S, Sa, CNOT"
        )

        with pytest.raises(ValueError, match=expected_msg):
            qecp_gate_op_from_string(invalid_gate)
