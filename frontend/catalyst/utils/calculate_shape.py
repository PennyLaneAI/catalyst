# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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
Deduce the function signatures after taking their gradients with respect to some parameters.
"""


from jax.core import ShapedArray


class Signature:
    """Signature: class representing a python's function signature. Used for
    type deduction during abstract evaluation.

    Args:
        xs(List[Union[Any,ShapedArray]]): the domain of the function
        ys(List[Union[Any,ShapedArray]]): the range of the function
    """

    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __repr__(self):
        return f"{self.xs} -> {self.ys}"

    def get_input(self, i):
        """Get parameter at position i.

        Args:
            i (int): Integer corresponding to parameter at position i.

        Returns:
            self.xs[i]: Type corresponding to parameter at position i.
        """
        return self.xs[i]

    def get_inputs(self):
        """Get all parameters.

        Returns:
            self.xs: All parameter types.
        """
        return self.xs

    def get_result(self, i):
        """Get result values at position i.

        Args:
            i (int): Integer corresponding to return value at position i.

        Returns:
            self.ys[i]: Type corresponding to parameter at position i.
        """
        return self.ys[i]

    def get_results(self):
        """Get all result values.

        Returns:
            self.ys: All types returned.
        """
        return self.ys

    @staticmethod
    def is_tensor(x):
        """Determine whether a type ``x`` is a ``jax.core.ShapedArray``.

        Args:
            x: The type to be tested.

        Returns:
            bool: Whether the type ``x`` is a ``jax.core.ShapedArray``
        """
        return isinstance(x, ShapedArray)

    def __eq__(self, other):
        return self.xs == other.xs and self.ys == other.ys


def calculate_grad_shape(signature, indices) -> Signature:
    """calculate_grad_shape: Given a signature and a list of indices over which arguments
    to differentiate, deduce the new signature.

    Args:
        signature(Signature): a signature.
        indices(List[int]): a list of integers that correspond to parameters in signature s.
    Returns:
        A signature corresponding to the differentiation of ``signature`` with respect to
        ``indices``.
    """
    grad_result_types = []
    for index in indices:
        diff_arg_type = signature.get_input(index)
        diff_arg_shape = []

        if Signature.is_tensor(diff_arg_type):
            for axis in diff_arg_type.shape:
                diff_arg_shape.append(axis)

        for y in signature.get_results():
            grad_res_shape = diff_arg_shape.copy()
            if Signature.is_tensor(y):
                for axis in y.shape:
                    grad_res_shape.append(axis)
                element_type = y.dtype

            grad_res_type = ShapedArray(grad_res_shape, element_type) if grad_res_shape else y
            grad_result_types.append(grad_res_type)

    return Signature(signature.get_inputs(), grad_result_types)
