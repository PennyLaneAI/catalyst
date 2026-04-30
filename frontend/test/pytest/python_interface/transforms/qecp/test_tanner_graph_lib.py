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
import scipy.sparse

from catalyst.python_interface.transforms.qecp.tanner_graph_lib import (
    parity_check_matrix_to_tanner_csc,
)


class TestParityCheckMatrixToTannerCsc:
    """Unit tests for the `parity_check_matrix_to_tanner_csc` function.

    We don't do thorough, comprehensive testing of this method because it is largely a wrapper
    around a scipy function, and we assume the scipy devs have already done their due diligence.
    """

    @pytest.mark.parametrize(
        "H, expected_dense_tanner",
        [
            (
                # A simple, contrived parity-check matrix
                np.array([[1, 1, 0], [0, 1, 1]]),
                np.array(
                    [
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 1],
                        [1, 1, 0, 0, 0],
                        [0, 1, 1, 0, 0],
                    ]
                ),
            ),
            (
                # The Steane code parity-check matrix
                np.array([[1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1]]),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
                    ]
                ),
            ),
        ],
    )
    def test_parity_check_matrix_to_tanner_csc(
        self, H: np.ndarray, expected_dense_tanner: np.ndarray
    ):
        """Test the `parity_check_matrix_to_tanner_csc` function with simple parity-check matrix
        inputs.

        This is a sort of round-trip test: we convert the input dense parity-check matrix, H, to a
        Tanner graph adjacency matrix in CSC form, and check the correctness of the result by
        reconstructing the dense adjacency matrix from the output `row_idx` and `col_ptr` arrays
        using the scipy sparse matrix library and check against the expected value given as input.
        """
        row_idx, col_ptr = parity_check_matrix_to_tanner_csc(H)

        # Reconstruct the dense Tanner graph adjacency matrix from row_idx and col_ptr and compare
        # against expected result.
        assert len(H.shape) == 2, "Incorrect test input: expected an m x n matrix"
        m, n = H.shape
        data = np.ones(len(row_idx), dtype=np.int32)
        tanner_adj_mat = scipy.sparse.csc_matrix((data, row_idx, col_ptr), shape=(m + n, m + n))

        assert np.array_equal(expected_dense_tanner, tanner_adj_mat.toarray())

    @pytest.mark.parametrize(
        "H",
        [
            np.zeros(shape=()),
            np.zeros(shape=(1,)),
            np.zeros(shape=(2,)),
            np.zeros(shape=(2, 2, 2)),
        ],
    )
    def test_parity_check_matrix_to_tanner_csc_invalid(self, H: np.ndarray):
        """Test the `parity_check_matrix_to_tanner_csc` function with invalid input arrays (of the
        wrong dimensions).
        """
        with pytest.raises(ValueError, match="Expected an m x n matrix"):
            _, _ = parity_check_matrix_to_tanner_csc(H)
