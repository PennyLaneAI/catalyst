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
from pathlib import Path
from typing import TypeAlias

import pennylane as qml

from catalyst.tracing.contexts import EvaluationContext

PipelineDict: TypeAlias = dict[str, dict[str, str]]


class Pass:
    """Class intended to hold options for passes"""

    def __init__(self, name, *options, **valued_options):
        self.name = name
        self.options = options
        self.valued_options = valued_options

    def __repr__(self):
        return (
            self.name
            + " ".join(f"--{option}" for option in self.options)
            + " ".join(f"--{option}={value}" for option, value in self.valued_options)
        )


class PassPlugin(Pass):
    """Class intended to hold options for pass plugins"""

    def __init__(
        self, path: Path, name: str, *options: list[str], **valued_options: dict[str, str]
    ):
        assert EvaluationContext.is_tracing()
        EvaluationContext.add_plugin(path)
        self.path = path
        super().__init__(name, *options, **valued_options)


def dictionary_to_tuple_of_passes(pass_pipeline: PipelineDict):
    """Convert dictionary of passes into tuple of passes"""

    if type(pass_pipeline) != dict:
        return pass_pipeline

    passes = tuple()
    pass_names = _API_name_to_pass_name()
    for API_name, pass_options in pass_pipeline.items():
        name = pass_names[API_name]
        passes += (Pass(name, **pass_options),)
    return passes


## API ##
# pylint: disable=line-too-long
@functools.singledispatch
def pipeline(pass_pipeline: PipelineDict):
    """Configures the Catalyst MLIR pass pipeline for quantum circuit transformations for a QNode within a qjit-compiled program.

    Args:
        pass_pipeline (dict[str, dict[str, str]]): A dictionary that specifies the pass pipeline order, and optionally
            arguments for each pass in the pipeline. Keys of this dictionary should correspond to names of passes
            found in the `catalyst.passes <https://docs.pennylane.ai/projects/catalyst/en/stable/code
            /__init__.html#module-catalyst.passes>`_ module, values should either be empty dictionaries
            (for default pass options) or dictionaries of valid keyword arguments and values for the specific pass.
            The order of keys in this dictionary will determine the pass pipeline.
            If not specified, the default pass pipeline will be applied.

    Returns:
        callable : A decorator that can be applied to a qnode.

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

    def _decorator(qnode=None):
        if not isinstance(qnode, qml.QNode):
            raise TypeError(f"A QNode is expected, got the classical function {qnode}")

        clone = copy.copy(qnode)
        clone.__name__ += "_transformed"

        @functools.wraps(clone)
        def wrapper(*args, **kwargs):
            if EvaluationContext.is_tracing():
                passes = kwargs.pop("pass_pipeline", tuple())
                passes += dictionary_to_tuple_of_passes(pass_pipeline)
                kwargs["pass_pipeline"] = passes
            return clone(*args, **kwargs)

        return wrapper

    return _decorator


def cancel_inverses(qnode=None):
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
    if not isinstance(qnode, qml.QNode):
        raise TypeError(f"A QNode is expected, got the classical function {qnode}")

    clone = copy.copy(qnode)
    clone.__name__ += "_cancel_inverses"

    @functools.wraps(clone)
    def wrapper(*args, **kwargs):
        pass_pipeline = kwargs.pop("pass_pipeline", tuple())
        pass_pipeline += (Pass("remove-chained-self-inverse"),)
        kwargs["pass_pipeline"] = pass_pipeline
        return clone(*args, **kwargs)

    return wrapper


def apply_pass(pass_name, *flags, **valued_options):
    """Applies a single pass to the qnode"""

    def decorator(qnode):

        if not isinstance(qnode, qml.QNode):
            # Technically, this apply pass is general enough that it can apply to
            # classical functions too. However, since we lack the current infrastructure
            # to denote a function, let's limit it to qnodes
            raise TypeError(f"A QNode is expected, got the classical function {qnode}")

        def qnode_call(*args, **kwargs):
            pass_pipeline = kwargs.get("pass_pipeline", [])
            pass_pipeline.append(Pass(pass_name, *flags, **valued_options))
            kwargs["pass_pipeline"] = pass_pipeline
            return qnode(*args, **kwargs)

        return qnode_call

    return decorator


def apply_pass_plugin(plugin_name, pass_name, *flags, **valued_options):
    """Applies a pass plugin"""

    def decorator(qnode):
        if not isinstance(qnode, qml.QNode):
            # Technically, this apply pass is general enough that it can apply to
            # classical functions too. However, since we lack the current infrastructure
            # to denote a function, let's limit it to qnodes
            raise TypeError(f"A QNode is expected, got the classical function {qnode}")

        def qnode_call(*args, **kwargs):
            pass_pipeline = kwargs.get("pass_pipeline", [])
            pass_pipeline.append(PassPlugin(plugin_name, pass_name, *flags, **valued_options))
            kwargs["pass_pipeline"] = pass_pipeline
            return qnode(*args, **kwargs)

        return qnode_call

    return decorator


def merge_rotations(qnode=None):
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
    if not isinstance(qnode, qml.QNode):
        raise TypeError(f"A QNode is expected, got the classical function {qnode}")

    clone = copy.copy(qnode)
    clone.__name__ += "_merge_rotations"

    @functools.wraps(clone)
    def wrapper(*args, **kwargs):
        pass_pipeline = kwargs.pop("pass_pipeline", tuple())
        pass_pipeline += (Pass("merge-rotations"),)
        kwargs["pass_pipeline"] = pass_pipeline
        return clone(*args, **kwargs)

    return wrapper


def _API_name_to_pass_name():
    return {
        "cancel_inverses": "remove-chained-self-inverse",
        "merge_rotations": "merge-rotations",
        "ions_decomposition": "ions-decomposition",
    }


def ions_decomposition(qnode=None):  # pragma: nocover
    """Apply decomposition pass at the MLIR level"""

    if not isinstance(qnode, qml.QNode):
        raise TypeError(f"A QNode is expected, got the classical function {qnode}")

    @functools.wraps(qnode)
    def wrapper(*args, **kwargs):
        pass_pipeline = kwargs.pop("pass_pipeline", tuple())
        pass_pipeline += (Pass("ions-decomposition"),)
        kwargs["pass_pipeline"] = pass_pipeline
        return qnode(*args, **kwargs)

    return wrapper
