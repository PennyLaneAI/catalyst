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
This module contains Python decorators for enabling and configuring
individual Catalyst MLIR compiler passes.

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
from typing import Optional

import pennylane as qml

from catalyst.jax_primitives import apply_registered_pass_p, transform_named_sequence_p
from catalyst.tracing.contexts import EvaluationContext


## API ##
# pylint: disable=line-too-long
def pipeline(pass_pipeline: Optional[dict[str, dict[str, str]]] = None):
    """Configures the Catalyst MLIR pass pipeline for quantum circuit transformations for a QNode within a qjit-compiled program.

    Args:
        fn (QNode): The QNode to run the pass pipeline on.
        pass_pipeline (dict[str, dict[str, str]]): A dictionary that specifies the pass pipeline order, and optionally
            arguments for each pass in the pipeline. Keys of this dictionary should correspond to names of passes
            found in the `catalyst.passes <https://docs.pennylane.ai/projects/catalyst/en/stable/code
            /__init__.html#module-catalyst.passes>`_ module, values should either be empty dictionaries
            (for default pass options) or dictionaries of valid keyword arguments and values for the specific pass.
            The order of keys in this dictionary will determine the pass pipeline.
            If not specified, the default pass pipeline will be applied.

    Returns:
        ~.QNode:

    For a list of available passes, please see the :doc:`catalyst.passes module <code/passes>`.

    The default pass pipeline when used with Catalyst is currently empty.

    **Example**

    ``pipeline`` can be used to configure the pass pipeline order and options
    of a QNode within a qjit-compiled function.

    Configuration options are passed to specific passes via dictionaries:

    .. code-block:: python

        my_pass_pipeline = {
            "cancel_inverses": {},
            "my_circuit_transformation_pass": {"my-option" : "my-option-value"},
        }

        @pipeline(my_pass_pipeline)
        @qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit
        def fn(x):
            return jnp.sin(circuit(x ** 2))

    ``pipeline`` can also be used to specify different pass pipelines for different parts of the
    same qjit-compiled workflow:

    .. code-block:: python

        my_pipeline = {
            "cancel_inverses": {},
            "my_circuit_transformation_pass": {"my-option" : "my-option-value"},
        }

        my_other_pipeline = {"cancel_inverses": {}}

        @qjit
        def fn(x):
            circuit_pipeline = pipeline(my_pipeline)(circuit)
            circuit_other = pipeline(my_other_pipeline)(circuit)
            return jnp.abs(circuit_pipeline(x) - circuit_other(x))

    .. note::

        As of Python 3.7, the CPython dictionary implementation orders dictionaries based on
        insertion order. However, for an API gaurantee of dictionary order, ``collections.OrderedDict``
        may also be used.

    Note that the pass pipeline order and options can be configured *globally* for a
    qjit-compiled function, by using the ``circuit_transform_pipeline`` argument of
    the :func:`~.qjit` decorator.

    .. code-block:: python

        my_pass_pipeline = {
            "cancel_inverses": {},
            "my_circuit_transformation_pass": {"my-option" : "my-option-value"},
        }

        @qjit(circuit_transform_pipeline=my_pass_pipeline)
        def fn(x):
            return jnp.sin(circuit(x ** 2))

    Global and local (via ``@pipeline``) configurations can coexist, however local pass pipelines
    will always take precedence over global pass pipelines.
    """

    def _decorator(fn=None, **kwargs):
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
        uniquer = str(_rename_to_unique())
        fn_clone.__name__ = fn_original_name + "_transformed" + uniquer

        pass_names = _API_name_to_pass_name()

        def wrapper(*args, **kwrags):
            # TODO: we should not match pass targets by function name.
            # The quantum scope work will likely put each qnode into a module
            # instead of a `func.func ... attributes {qnode}`.
            # When that is in place, the qnode's module can have a proper attribute
            # (as opposed to discardable) that records its transform schedule, i.e.
            #    module_with_transform @name_of_module {
            #      // transform schedule
            #    } {
            #      // contents of the module
            #    }
            # This eliminates the need for matching target functions by name.

            if EvaluationContext.is_tracing():
                for API_name, pass_options in pass_pipeline.items():
                    opt = ""
                    for option, option_value in pass_options.items():
                        opt += " " + str(option) + "=" + str(option_value)
                    apply_registered_pass_p.bind(
                        pass_name=pass_names[API_name],
                        options=f"func-name={fn_original_name}" + "_transformed" + uniquer + opt,
                    )
            return wrapped_qnode_function(*args, **kwrags)

        fn_clone.func = wrapper
        fn_clone._peephole_transformed = True  # pylint: disable=protected-access

        return fn_clone

    return _decorator


def cancel_inverses(fn=None):
    """
    Specify that the ``-removed-chained-self-inverse`` MLIR compiler pass
    for cancelling two neighbouring self-inverse
    gates should be applied to the decorated QNode during :func:`~.qjit`
    compilation.

    The full list of supported gates are as follows:

    One-bit Gates:
    :class:`qml.Hadamard <pennylane.Hadamard>`,
    :class:`qml.PauliX <pennylane.PauliX>`,
    :class:`qml.PauliY <pennylane.PauliY>`,
    :class:`qml.PauliZ <pennylane.PauliZ>`

    Two-bit Gates:
    :class:`qml.CNOT <pennylane.CNOT>`,
    :class:`qml.CY <pennylane.CY>`,
    :class:`qml.CZ <pennylane.CZ>`,
    :class:`qml.SWAP <pennylane.SWAP>`

    Three-bit Gates:
    - :class:`qml.Toffoli <pennylane.Toffoli>`

    .. note::

        Unlike PennyLane :doc:`circuit transformations <introduction/compiling_circuits>`,
        the QNode itself will not be changed or transformed by applying these
        decorators.

        As a result, circuit inspection tools such as :func:`~.draw` will continue
        to display the circuit as written in Python.

        To instead view the optimized circuit, the MLIR must be viewed
        after the ``"QuantumCompilationPass"`` stage via the
        :func:`~.get_compilation_stage` function.

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
    uniquer = str(_rename_to_unique())

    def wrapper(*args, **kwrags):
        if EvaluationContext.is_tracing():
            apply_registered_pass_p.bind(
                pass_name="remove-chained-self-inverse",
                options=f"func-name={funcname}" + "_cancel_inverses" + uniquer,
            )
        return wrapped_qnode_function(*args, **kwrags)

    fn_clone = copy.copy(fn)
    fn_clone.func = wrapper
    fn_clone.__name__ = funcname + "_cancel_inverses" + uniquer

    return fn_clone

def apply_pass(pass_name, *flags, **valued_options):
    """
    """

    def decorator(fn):

        if not isinstance(fn, qml.QNode):
            # Technically, this apply pass is general enough that it can apply to
            # classical functions too. However, since we lack the current infrastructure
            # to denote a function, let's limit it to qnodes
            raise TypeError(f"A QNode is expected, got the classical function {fn}")

        funcname = fn.__name__
        wrapped_qnode_function = fn.func
        uniquer = str(_rename_to_unique())

        options = []
        for flag in flags:
            options += ["{flag}"]
        for option_name, val in valued_options.items():
            options += ["{option_name}={val}"]

        options += [f"func-name={funcname}" + f"_{pass_name}" + uniquer]
        options = ",".join(options)

        def wrapper(*args, **kwargs):
            if EvaluationContext.is_tracing():
                apply_registered_pass_p.bind(pass_name=pass_name, options = options)
            return wrapped_qnode_function(*args, **kwargs)

        fn_clone = copy.copy(fn)
        fn_clone.func = wrapper
        fn_clone.__name__ = funcname + f"_{pass_name}" + uniquer

        return fn_clone

    return decorator



def merge_rotations(fn=None):
    """
    Specify that the ``-merge-rotations`` MLIR compiler pass
    for merging roations (peephole) will be applied.

    The full list of supported gates are as follows:

    :class:`qml.RX <pennylane.RX>`,
    :class:`qml.CRX <pennylane.CRX>`,
    :class:`qml.RY <pennylane.RY>`,
    :class:`qml.CRY <pennylane.CRY>`,
    :class:`qml.RZ <pennylane.RZ>`,
    :class:`qml.CRZ <pennylane.CRZ>`,
    :class:`qml.PhaseShift <pennylane.PhaseShift>`,
    :class:`qml.ControlledPhaseShift <pennylane.ControlledPhaseShift>`,
    :class:`qml.MultiRZ <pennylane.MultiRZ>`.


    .. note::

        Unlike PennyLane :doc:`circuit transformations <introduction/compiling_circuits>`,
        the QNode itself will not be changed or transformed by applying these
        decorators.

        As a result, circuit inspection tools such as :func:`~.draw` will continue
        to display the circuit as written in Python.

        To instead view the optimized circuit, the MLIR must be viewed
        after the ``"QuantumCompilationPass"`` stage via the
        :func:`~.get_compilation_stage` function.

    Args:
        fn (QNode): the QNode to apply the cancel inverses compiler pass to

    Returns:
        ~.QNode:

    **Example**

    In this example the three :class:`qml.RX <pennylane.RX>` will be merged in a single
    one with the sum of angles as parameter.

    .. code-block:: python

        from catalyst.debug import get_compilation_stage
        from catalyst.passes import merge_rotations

        dev = qml.device("lightning.qubit", wires=1)

        @qjit(keep_intermediate=True)
        @merge_rotations
        @qml.qnode(dev)
        def circuit(x: float):
            qml.RX(x, wires=0)
            qml.RX(0.1, wires=0)
            qml.RX(x**2, wires=0)
            return qml.expval(qml.PauliZ(0))

    >>> circuit(0.54)
    Array(0.5965506257017892, dtype=float64)
    """
    if not isinstance(fn, qml.QNode):
        raise TypeError(f"A QNode is expected, got the classical function {fn}")

    funcname = fn.__name__
    wrapped_qnode_function = fn.func
    uniquer = str(_rename_to_unique())

    def wrapper(*args, **kwrags):
        if EvaluationContext.is_tracing():
            apply_registered_pass_p.bind(
                pass_name="merge-rotations",
                options=f"func-name={funcname}" + "_merge_rotations" + uniquer,
            )
        return wrapped_qnode_function(*args, **kwrags)

    fn_clone = copy.copy(fn)
    fn_clone.func = wrapper
    fn_clone.__name__ = funcname + "_merge_rotations" + uniquer

    return fn_clone


## IMPL and helpers ##
# pylint: disable=missing-function-docstring
class _PipelineNameUniquer:
    def __init__(self, i):
        self.i = i

    def get(self):
        self.i += 1
        return self.i

    def reset(self):
        self.i = -1


PipelineNameUniquer = _PipelineNameUniquer(-1)


def _rename_to_unique():
    return PipelineNameUniquer.get()


def _API_name_to_pass_name():
    return {"cancel_inverses": "remove-chained-self-inverse", "merge_rotations": "merge-rotations"}


def _inject_transform_named_sequence():
    """
    Inject a transform_named_sequence jax primitive.

    This must be called when preprocessing the traced function in QJIT.capture(),
    since to invoke -apply-transform-sequence, a transform_named_sequence primitive
    must be in the jaxpr.
    """

    transform_named_sequence_p.bind()
