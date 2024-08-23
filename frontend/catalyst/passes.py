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
a unified user interface for all the passes will be available.

.. note::

    Unlike PennyLane :doc:`circuit transformations <introduction/compiling_circuits>`,
    the QNode itself will not be changed or transformed.

    In other words, circuit inspection tools such as
    :func:`~.draw` will still
    display the neighbouring self-inverse gates. However, Catalyst never
    executes the PennyLane code directly; instead, Catalyst captures the
    workflow from Python and lowers it into MLIR, performing compiler
    optimizations at the MLIR level.
    To inspect the compiled MLIR from Catalyst, use
    :func:`~.get_compilation_stage`,
    where ``stage="QuantumCompilationPass"``, and with ``keep_intermediate=True``
    in the ``qjit`` decorator.

"""

import copy

import pennylane as qml

from catalyst.jax_primitives import apply_registered_pass_p, transform_named_sequence_p


## API ##
def cancel_inverses(fn=None):  # pylint: disable=line-too-long
    """
    Specify that a compiler pass for cancelling two neighbouring self-inverse
    gates should be applied to the decorated QNode during qjit compilation.


    .. note::

        Currently, only Hadamard gates are canceled.


    Args:
        fn (QNode): the QNode to apply the cancel inverses compiler pass to

    Returns:
        ~.QNode:

    **Example**

    .. code-block:: python

        from catalyst.debug import get_compilation_stage
        from catalyst.passes import cancel_inverses

        dev = qml.device("lightning.qubit", wires=1)

        @qjit(keep_intermediate=True)
        def workflow():
            @cancel_inverses
            @qml.qnode(dev)
            def f(x: float):
                qml.RX(x, wires=0)
                qml.Hadamard(wires=0)
                qml.Hadamard(wires=0)
                return qml.expval(qml.PauliZ(0))

            @qml.qnode(dev)
            def g(x: float):
                qml.RX(x, wires=0)
                qml.Hadamard(wires=0)
                qml.Hadamard(wires=0)
                return qml.expval(qml.PauliZ(0))

            ff = f(1.0)
            gg = g(1.0)

            return ff, gg

    >>> workflow()
    (Array(0.54030231, dtype=float64), Array(0.54030231, dtype=float64))
    >>> print(get_compilation_stage(workflow, "QuantumCompilationPass"))

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

    We see that ``f``, decorated by ``cancel_inverses``, no longer has the neighbouring
    Hadamard gates.
    """
    if not isinstance(fn, qml.QNode):
        raise TypeError(f"A QNode is expected, got the classical function {fn}")

    wrapped_qnode_function = fn.func
    funcname = fn.__name__

    def wrapper(*args, **kwrags):
        # TODO: hint the compiler which qnodes to run the pass on via an func attribute,
        # instead of the qnode name. That way the clone can have this attribute and
        # the original can just not have it.
        # We are not doing this right now and passing by name because this would
        # be a discardable attribute (i.e. a user/developer wouldn't know that this
        # attribute exists just by looking at qnode's documentation)
        # But when we add the full peephole pipeline in the future, the attribute
        # could get properly documented.

        apply_registered_pass_p.bind(
            pass_name="remove-chained-self-inverse",
            options=f"func-name={funcname}" + "_cancel_inverses",
        )
        return wrapped_qnode_function(*args, **kwrags)

    fn_clone = copy.copy(fn)
    fn_clone.func = wrapper
    fn_clone.__name__ = funcname + "_cancel_inverses"

    return fn_clone


## IMPL and helpers ##
def _inject_transform_named_sequence():
    """
    Inject a transform_named_sequence jax primitive.

    This must be called when preprocessing the traced function in QJIT.capture(),
    since to invoke -apply-transform-sequence, a transform_named_sequence primitive
    must be in the jaxpr.
    """

    transform_named_sequence_p.bind()
