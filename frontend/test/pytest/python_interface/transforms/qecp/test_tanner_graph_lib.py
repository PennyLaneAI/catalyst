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

"""Test suite for the catalyst.python_interface.transforms.qecp.tanner_graph_lib module."""

import numpy as np
import pytest

from catalyst.python_interface.transforms.qecp.tanner_graph_lib import dense_tanner_graph_to_csc


class TestDenseTannerGraphToCsc:
    """Unit tests for the `dense_tanner_graph_to_csc` function.

    We don't do thorough, comprehensive testing of this method because it is a wrapper around a
    scipy function, and we assume the scipy devs have already done their due diligence.
    """

    @pytest.mark.parametrize(
        "matrix, expected_row_idx, expected_col_ptr",
        [
            (
                # A simple, contrived parity-check matrix
                np.array(
                    [
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 1],
                        [1, 1, 0, 0, 0],
                        [0, 1, 1, 0, 0],
                    ]
                ),
                np.array([3, 3, 4, 4, 0, 1, 1, 2]),  # expected row_idx
                np.array([0, 1, 3, 4, 6, 8]),  # expected col_ptr
            ),
            (
                # The Steane code parity-check matrix
                np.array([[1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1]]),
                np.array([0, 0, 1, 0, 1, 2, 0, 2, 1, 1, 2, 2]),  # expected row_idx
                np.array([0, 1, 3, 6, 8, 9, 11, 12]),  # expected col_ptr
            ),
        ],
    )
    def test_dense_tanner_graph_to_csc(
        self, matrix: np.ndarray, expected_row_idx: np.ndarray, expected_col_ptr: np.ndarray
    ):
        """Test the `dense_tanner_graph_to_csc` function with simple parity-check matrix inputs."""
        row_idx, col_ptr = dense_tanner_graph_to_csc(matrix)

        assert np.array_equal(row_idx, expected_row_idx)
        assert np.array_equal(col_ptr, expected_col_ptr)
