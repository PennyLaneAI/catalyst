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

import jax.numpy as jnp
import pennylane as qml
from catalyst import qjit
from jax.core import ShapedArray


@qjit
def f_of_empty():
    """Check empty list of arguments"""
    return True

f_of_empty.jit_compile([])
assert f_of_empty.get_arg_names() == []


@qjit
def f_of_a_b(a: float, b: float):
    """Check two float arguments"""
    return a * b

f_of_a_b.jit_compile([0.3, 0.4])
assert f_of_a_b.get_arg_names() == ["a", "b"]


@qjit(abstracted_axes={0: "n"})
def f_of_dynamic_argument(a):
    """Check dynamic argument"""
    return a

f_of_dynamic_argument.jit_compile([jnp.array([1, 2, 3])])
assert f_of_dynamic_argument.get_arg_names() == ["a", ""]


@qjit(abstracted_axes={0: "n"})
def f_of_qnode_with_dynamic_argument(a):
    """Check QNode argument with dynamic argument"""

    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def _circuit(b):
        return b

    return _circuit(a)

f_of_qnode_with_dynamic_argument.jit_compile([jnp.array([1, 2, 3])])
assert f_of_dynamic_argument.get_arg_names() == ["a", ""]


@qjit
def f_of_a_with_dynamic_result(a):
    """Check dynamic result"""
    return jnp.ones((a + 1,), dtype=float)

f_of_a_with_dynamic_result.jit_compile([3])
assert f_of_a_with_dynamic_result.get_arg_names() == ["a"]


@qjit(abstracted_axes={0: "n", 2: "m"})
def f_of_shaped_array(a: ShapedArray([1, 3, 1], dtype=float)):
    """Check ShapedArray argument"""
    return a

f_of_shaped_array.aot_compile()
assert f_of_shaped_array.get_arg_names() == ["a", "", ""]
