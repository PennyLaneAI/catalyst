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
This module contains which provides Python decorators
for enabling and configuring individual Catalyst MLIR compiler passes.

.. note::

    Unlike PennyLane :doc:`circuit transformations <introduction/compiling_circuits>`,
    the QNode itself will not be changed or transformed by applying these
    decorators.

    As a result, circuit inspection tools such as :func:`~.draw` will continue
    to display the circuit as written in Python.

    Instead, these compiler passes are applied at the MLIR level, which occurs
    outside of Python during compile time. To inspect the compiled MLIR from
    Catalyst, use :func:`~.get_compilation_stage` with
    ``stage="QuantumCompilationPass"``.

"""

import copy
import functools

import pennylane as qml

from catalyst.jax_primitives import apply_registered_pass_p, transform_named_sequence_p
from catalyst.tracing.contexts import EvaluationContext


## API ##
# pylint: disable=line-too-long
def pipeline(fn=None, *, pass_pipeline=None):
    """
    Here are documentation words
    """

    kwargs = copy.copy(locals())
    kwargs.pop("fn")

    if fn is None:
        return functools.partial(pipeline, **kwargs)

    if not isinstance(fn, qml.QNode):
        raise TypeError(f"A QNode is expected, got the classical function {fn}")

    if pass_pipeline is None:
        # TODO: design a default peephole pipeline
        return fn

    fn_original_name = fn.__name__
    wrapped_qnode_function = fn.func
    fn_clone = copy.copy(fn)
    fn_clone.__name__ = fn_original_name + "_transformed"

    pass_names = API_name_to_pass_name()

    def wrapper(*args, **kwrags):
        if EvaluationContext.is_tracing():
            for API_name, pass_options in pass_pipeline.items():
                apply_registered_pass_p.bind(
                    pass_name=pass_names[API_name],
                    options=f"func-name={fn_original_name}" + "_transformed",
                )
        return wrapped_qnode_function(*args, **kwrags)

    fn_clone.func = wrapper

    return fn_clone


def cancel_inverses(fn=None, keep_original=True):
    """
    Specify that the ``-removed-chained-self-inverse`` MLIR compiler pass
    for cancelling two neighbouring self-inverse
    gates should be applied to the decorated QNode during :func:`~.qjit`
    compilation.

    .. warning::

        Currently, only Hadamard gates are canceled.

    .. note::

        Unlike PennyLane :doc:`circuit transformations <introduction/compiling_circuits>`,
        the QNode itself will not be changed or transformed by applying these
        decorators.

        As a result, circuit inspection tools such as :func:`~.draw` will continue
        to display the circuit as written in Python.

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
        @cancel_inverses
        @qml.qnode(dev)
        def circuit(x: float):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

    >>> circuit(0.54)
    Array(0.85770868, dtype=float64)

    Note that the QNode will be unchanged in Python, and will continue
    to include self-inverse gates when inspected with Python (for example,
    with :func:`~.draw`).

    To instead view the optimized circuit, the MLIR must be viewed
    after the ``"QuantumCompilationPass"`` stage:

    >>> print(get_compilation_stage(circuit, stage="QuantumCompilationPass"))
    module @circuit {
      func.func public @jit_circuit(%arg0: tensor<f64>) -> tensor<f64> attributes {llvm.emit_c_interface} {
        %0 = call @circuit(%arg0) : (tensor<f64>) -> tensor<f64>
        return %0 : tensor<f64>
      }
      func.func private @circuit(%arg0: tensor<f64>) -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
        quantum.device["catalyst/utils/../lib/librtd_lightning.dylib", "LightningSimulator", "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
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
      func.func @setup() {
        quantum.init
        return
      }
      func.func @teardown() {
        quantum.finalize
        return
      }
    }

    It can be seen that both Hadamards have been cancelled, and the measurement
    directly follows the ``RX`` gate:

    .. code-block:: mlir

        %out_qubits = quantum.custom "RX"(%extracted) %1 : !quantum.bit
        %2 = quantum.namedobs %out_qubits[ PauliZ] : !quantum.obs
        %3 = quantum.expval %2 : f64
    """
    if not isinstance(fn, qml.QNode):
        raise TypeError(f"A QNode is expected, got the classical function {fn}")

    funcname = fn.__name__
    wrapped_qnode_function = fn.func

    if keep_original:

        def wrapper(*args, **kwrags):
            # TODO: hint the compiler which qnodes to run the pass on via an func attribute,
            # instead of the qnode name. That way the clone can have this attribute and
            # the original can just not have it.
            # We are not doing this right now and passing by name because this would
            # be a discardable attribute (i.e. a user/developer wouldn't know that this
            # attribute exists just by looking at qnode's documentation)
            # But when we add the full peephole pipeline in the future, the attribute
            # could get properly documented.

            if EvaluationContext.is_tracing():
                apply_registered_pass_p.bind(
                    pass_name="remove-chained-self-inverse",
                    options=f"func-name={funcname}" + "_cancel_inverses",
                )
            return wrapped_qnode_function(*args, **kwrags)

        fn_clone = copy.copy(fn)
        fn_clone.func = wrapper
        fn_clone.__name__ = funcname + "_cancel_inverses"

        return fn_clone

    else:

        def wrapper(*args, **kwrags):
            if EvaluationContext.is_tracing():
                apply_registered_pass_p.bind(
                    pass_name="remove-chained-self-inverse",
                    options=f"func-name={funcname}",
                )
            return wrapped_qnode_function(*args, **kwrags)

        fn.func = wrapper
        return fn


## IMPL and helpers ##
def API_name_to_pass_name():
    return {"cancel_inverses": "remove-chained-self-inverse", "merge_rotations": "merge-rotation"}


def _inject_transform_named_sequence():
    """
    Inject a transform_named_sequence jax primitive.

    This must be called when preprocessing the traced function in QJIT.capture(),
    since to invoke -apply-transform-sequence, a transform_named_sequence primitive
    must be in the jaxpr.
    """

    transform_named_sequence_p.bind()
