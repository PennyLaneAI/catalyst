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

"""This module contains utilities for managing Tanner graphs in the qecl-to-qecp dialect-conversion
pass.
"""

import numpy as np
import scipy.sparse


def parity_check_matrix_to_tanner_csc(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Converts a dense parity-check matrix `H` to a Tanner graph adjacency matrix in CSC form.

    Takes a dense 2D numpy array representing a parity-check matrix, converts it to the adjacency
    matrix for the equivalent Tanner graph, and returns the row indices and column pointers
    corresponding to the Compressed Sparse Column (CSC) format of this adjacency matrix.

    Note that this function does not return the 'data' array of the CSC matrix (the array of
    non-zero values), as these values are assumed to be all 1s.

    Args:
        matrix (numpy.ndarray): A 2D numpy array representing the dense parity-check matrix,
            typically containing 0s and 1s. The parity-check matrix is taken to have dimensions
            (n_aux x n_data), where n_aux and n_data are the number of auxiliary and data qubits,
            respectively.

    Returns:
        tuple: A tuple containing two elements representing the Tanner graph adjacency matrix:
            - row_idx (numpy.ndarray): The row indices of the non-zero elements.
            - col_ptr (numpy.ndarray): The column pointers, where indptr[i] indicates the start
              index in 'indices' for the i-th column.

    .. details::
        :title: Layout of the (dense) Tanner graph adjacency matrix

        Given an m x n parity-check matrix H, the equivalent Tanner graph adjacency matrix A has the
        form

                ┌       ┐
            A = │ 0 H^T │
                │ H  0  │
                └       ┘

        The adjacency matrix A therefore has shape (m+n, m+n). In this representation, the first n
        columns corresponding to the n data qubits of the code, from which the data qubit's
        neighbouring aux qubits in the Tanner graph can be read off from the non-zero elements in
        the column, and the last m columns correspond to the m aux qubits of the code, from which
        their neighbouring data qubits can be read off from the non-zero elements in the column.
    """
    if len(H.shape) != 2:
        raise ValueError(f"Expected an m x n matrix, but got an array with shape {H.shape}")

    m, n = H.shape  # m = n_aux, n = n_data

    H_csc = scipy.sparse.csc_matrix(H)

    # Get H^T. Note that we can't use the sparse matrix `transpose()` method because that does not
    # alter the underlying `indices` and `indptr` arrays.
    H_T = H.transpose()
    H_T_csc = scipy.sparse.csc_matrix(H_T)

    H_nnz = H_csc.nnz  # Number of non-zero elements

    # To get the CSC form of the adjacency matrix, we divide it up into two segments:
    #   - Columns [0:n-1] (inclusive)
    #   - Columns [n:n+m-1] (inclusive)
    # and get the row_idx and col_ptr values for each, then combine them together as follows:

    A_indices = np.concatenate([H_csc.indices + n, H_T_csc.indices])
    A_indptr = np.concatenate([H_csc.indptr[:-1], H_T_csc.indptr + H_nnz])

    return A_indices, A_indptr
