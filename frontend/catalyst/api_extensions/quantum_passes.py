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


def inject_transform_named_sequence():
    """
    Inject a transform_named_sequence jax primitive.

    This must be called when preprocessing the traced function in QJIT.capture(),
    since to invoke -transform-interpreter, a transform_named_sequence primitive
    must be in the jaxpr.
    """

    transform_named_sequence_p.bind()


## API ##
def cancel_inverses(fn=None):
    """
    The top-level ``catalyst.cancel_inverses`` decorator.

    This decorator is always applied to a qnode, and it cancels two neighbouring self-inverse gates in the compiled mlir.


    .. note::

        Currently, only Hadamard gates are canceled.

    .. note::

        The qnode itself is not changed. In other words, circuit inspection tools such as ``qml.draw`` will still display the
        neighbouring self-inverse gates. However, catalyst never executes the pennylane code directly. Instead catalyst
        executes the compiled mlir, and these neighbouring self inverse gates are canceled in the compiled mlir.

        To inspect the compiled mlir from Catalyst, use ``qjit(keep_intermediate=True)`` in the top-level ``qjit`` decorator.
        This will create the intermediate mlir in a directory under the same directory where your python file is.
        The cancel inverse happens at the stage ``QuantumCompilationPass``.

    Args:
        fn (Callable): a qml.QNode to run the cancel inverses transformation on

    Returns:
        The same qml.QNode.

    **Example**

    .. code-block:: python

        @qjit(keep_intermediate=True)
        def workflow():
            @cancel_inverses
            @qml.qnode(qml.device("lightning.qubit", wires=1))
            def f(x: float):
                qml.RX(x, wires=0)
                qml.Hadamard(wires=0)
                qml.Hadamard(wires=0)
                return qml.expval(qml.PauliZ(0))

            @qml.qnode(qml.device("lightning.qubit", wires=1))
            def g(x: float):
                qml.RX(x, wires=0)
                qml.Hadamard(wires=0)
                qml.Hadamard(wires=0)
                return qml.expval(qml.PauliZ(0))

            ff = f(1.0)
            gg = g(2.0)

            return ff, gg

    >>> workflow()
    (Array(0.54030231, dtype=float64), Array(-0.41614684, dtype=float64))

    In the compiled mlir files, specifically in ``workflow/2_QuantumCompilationPass.mlir``:

    .. code-block:: mlir

          func.func private @f(%arg0: tensor<f64>) -> tensor<f64> {
            quantum.device["PATH_TO_librtd_lightning.so", "LightningSimulator",
            "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
            %0 = quantum.alloc( 1) : !quantum.reg
            %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
            %extracted = tensor.extract %arg0[] : tensor<f64>
            %out_qubits = quantum.custom "RX"(%extracted) %1 : !quantum.bit
            %2 = quantum.namedobs %out_qubits[ PauliZ] : !quantum.obs
            %3 = quantum.expval %2 : f64
            %from_elements = tensor.from_elements %3 : tensor<f64>
            %4 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
            quantum.dealloc %4 : !quantum.reg
            quantum.device_release
            return %from_elements : tensor<f64>
          }
          func.func private @g(%arg0: tensor<f64>) -> tensor<f64> {
            quantum.device["PATH_TO_librtd_lightning.so", "LightningSimulator",
            "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
            %0 = quantum.alloc( 1) : !quantum.reg
            %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
            %extracted = tensor.extract %arg0[] : tensor<f64>
            %out_qubits = quantum.custom "RX"(%extracted) %1 : !quantum.bit
            %out_qubits_0 = quantum.custom "Hadamard"() %out_qubits : !quantum.bit
            %out_qubits_1 = quantum.custom "Hadamard"() %out_qubits_0 : !quantum.bit
            %2 = quantum.namedobs %out_qubits_1[ PauliZ] : !quantum.obs
            %3 = quantum.expval %2 : f64
            %from_elements = tensor.from_elements %3 : tensor<f64>
            %4 = quantum.insert %0[ 0], %out_qubits_1 : !quantum.reg, !quantum.bit
            quantum.dealloc %4 : !quantum.reg
            quantum.device_release
            return %from_elements : tensor<f64>
          }

    We see that ``f``, decorated by ``cancel_inverses``, no longer has the neighbouring Hadamard gates.
    """
    if not isinstance(fn, qml.QNode):
        raise TypeError(f"A QNode is expected, got the classical function {fn}")

    wrapped_qnode_function = fn.func

    def wrapper(*args, **kwrags):

        apply_registered_pass_p.bind(
            pass_name="remove-chained-self-inverse", options=f"func-name={fn.__name__}"
        )

        return wrapped_qnode_function(*args, **kwrags)

    fn.func = wrapper

    return fn
