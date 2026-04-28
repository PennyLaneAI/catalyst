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


def dense_tanner_graph_to_csc(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Converts a dense parity-check matrix to CSC index arrays.

    Takes a dense 2D numpy array representing a Tanner graph, expressed as a parity-check matrix,
    and returns the row indices and column pointers corresponding to the Compressed Sparse Column
    (CSC) format of this matrix.

    Note that this function does not return the 'data' array of the CSC matrix (the array of
    non-zero values), as these values are assumed to be all 1s.

    Args:
        matrix (numpy.ndarray): A 2D numpy array representing the dense matrix, typically containing
            0s and 1s.

    Returns:
        tuple: A tuple containing two elements:
            - row_idx (numpy.ndarray): The row indices of the non-zero elements.
            - col_ptr (numpy.ndarray): The column pointers, where indptr[i] indicates the start
              index in 'indices' for the i-th column.
    """
    matrix_as_csc = scipy.sparse.csc_matrix(matrix)

    return matrix_as_csc.indices, matrix_as_csc.indptr
