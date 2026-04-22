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

import re

import numpy as np
import pytest

from catalyst.python_interface.transforms.qecp.qec_code_lib import QecCode

SUPPORTED_CODES = ["Steane"]


class TestQecCode:
    """Units tests for the QecCode class."""

    @pytest.mark.parametrize(
        "name, n, k, d", [("Steane", 7, 1, 3), ("Shor", 9, 1, 3), ("Surface_d3", 17, 1, 3)]
    )
    def test_constructor(self, name: str, n: int, k: int, d: int):
        """Test the constructor of the `QecCode` class for various QEC codes."""
        qec_code = QecCode(name, n, k, d, np.eye(n), np.array([1] * n))

        assert qec_code.name == name
        assert qec_code.n == n
        assert qec_code.k == k
        assert qec_code.d == d
        assert np.all(qec_code.x_tanner == np.eye(n))
        assert np.all(qec_code.z_tanner == np.array([1] * n))

    @pytest.mark.parametrize(
        "inputs, expected_str",
        [
            (("Steane", 7, 1, 3, np.eye(7), np.eye(7)), "[[7, 1, 3]] Steane"),
            (("", 7, 1, 3, np.eye(7), np.eye(7)), "[[7, 1, 3]] <unknown>"),
            (("  ", 7, 1, 3, np.eye(7), np.eye(7)), "[[7, 1, 3]] <unknown>"),
        ],
    )
    def test_str_representation(self, inputs, expected_str):
        """Test the string representation of the `QecCode` class for various inputs."""
        qec_code = QecCode(*inputs)
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
            },
            {
                "name": "Shor",
                "n": 9,
                "k": 1,
                "d": 3,
                "x_tanner": np.eye(9),
                "z_tanner": np.array([[0, 0, 1, 1, 0, 1, 1, 0, 1]]),
            },
            {
                "name": "Unknown",
                "n": 7,
                "k": 1,
                "d": 3,
                "x_tanner": np.eye(7),
                "z_tanner": np.array([[0, 0, 1, 1, 0, 1, 1]]),
                "extra-field": 42,
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
            },
            {
                "name": "Shor",
                "n": 9,
                "k": 1,
                "d": 3,
                "x_tanner": np.eye(9),
                "z_tanner": np.array([[0, 0, 1, 1, 0, 1, 1, 0, 1]]),
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
