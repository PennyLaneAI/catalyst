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

from catalyst.jax_extras import make_jaxpr2
from catalyst.tracing.contexts import EvaluationContext

import sys
if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

discovered_plugins = entry_points(group='catalyst.passes.plugins')


class Pass:

    def __init__(self, name, *options, **valued_options):
        self.name = name
        self.options = options
        self.valued_options = valued_options

    def __repr__(self):
        options = ",".join(str(opt) for opt in self.options)
        valued_options = ",".join(f"{k}={v}" for k, v in self.valued_options.items())
        all_options = ",".join((options, valued_options))
        return self.name + f"{all_options}"

class PassPlugin(Pass):
    def __init__(self, path, name, *options, **valued_options):
        self.path = path
        return super().__init__(name, *options, **valued_options)


def pipeline(pass_pipeline=None):

    if pass_pipeline is None:
        pass_pipeline = []

    def decorator(qnode):

        if not isinstance(qnode, qml.QNode):
            raise TypeError(f"A QNode is expected, got the classical function {qnode}")

        def qnode_call(*args, **kwargs):
            pipeline = kwargs.get("pass_pipeline", pass_pipeline)
            kwargs["pass_pipeline"] = pipeline
            return qnode(*args, **kwargs)

        return qnode_call

    return decorator


def cancel_inverses(qnode=None, *pass_args, **pass_opts):
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
      // ... snip ...
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
      // ... snip ...
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

    def qnode_call(*args, **kwargs):
        pipeline = kwargs.get("pass_pipeline", [])
        pipeline.append(Pass("remove-chained-self-inverse", pass_args, pass_opts))
        kwargs["pass_pipeline"] = pipeline
        return qnode(*args, **kwargs)

    return qnode_call


def apply_pass(pass_name, *flags, **valued_options):
    """Apply pass"""

    def decorator(qnode):

        if not isinstance(qnode, qml.QNode):
            # Technically, this apply pass is general enough that it can apply to
            # classical functions too. However, since we lack the current infrastructure
            # to denote a function, let's limit it to qnodes
            raise TypeError(f"A QNode is expected, got the classical function {fn}")

        def qnode_call(*args, **kwargs):
            pipeline = kwargs.get("pass_pipeline", [])
            pipeline.append(Pass(pass_name))
            kwargs["pass_pipeline"] = pipeline
            return qnode(*args, **kwargs)

        return qnode_call

    return decorator

def pass_plugin(plugin, pass_name, *opts, **valued_options):
    """Apply pass"""
    # TODO: Register the plugin with the compiler...?
    return apply_pass(pass_name, *opts, **valued_options)

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

    def qnode_call(*args, **kwargs):
        pipeline = kwargs.get("pass_pipeline", [])
        pipeline.append(Pass("merge-rotations"))
        kwargs["pass_pipeline"] = pipeline
        return qnode(*args, **kwargs)

    return qnode_call


def ions_decomposition(qnode=None):
    if not isinstance(qnode, qml.QNode):
        raise TypeError(f"A QNode is expected, got the classical function {qnode}")

    def qnode_call(*args, **kwargs):
        pipeline = kwargs.get("pass_pipeline", [])
        pipeline.append(Pass("ions-decomposition"))
        kwargs["pass_pipeline"] = pipeline
        return qnode(*args, **kwargs)

    return qnode_call
