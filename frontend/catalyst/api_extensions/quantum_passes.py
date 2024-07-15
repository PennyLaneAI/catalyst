# Copyright 2024 Xanadu Quantum Technologies Inc.

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
This module contains public API functions that provide control for the 
user to input what MLIR compiler passes to run. 

Currently, each pass has its own user-facing decorator. In the future, 
a unified user interface for all the passes is necessary. 

Note that the decorators do not need to modify the qnode in 
any way. Its only purpose is to mark down the passes the user wants to 
run on each qnode, and then generate the corresponding 
transform.apply_apply_registered_pass in the lowered mlir.
"""

import pennylane as qml

from catalyst.jax_primitives import apply_registered_pass_p, transform_named_sequence_p


## API ##
def cancel_inverses(fn=None):
    """
    The top-level `catalyst.cancel_inverses` decorator.
    !!! TODO: add documentation here !!!
    """

    if not isinstance(fn, qml.QNode):
        raise TypeError(f"A QNode is expected, got the classical function {fn}")

    transform_named_sequence_p.bind()
    apply_registered_pass_p.bind(
        pass_name="remove-chained-self-inverse", options=f"func-name={fn.__name__}"
    )

    return fn
