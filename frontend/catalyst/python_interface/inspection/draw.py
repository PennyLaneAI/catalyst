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
"""This file contains the implementation of the `draw` function for the Unified Compiler."""

from __future__ import annotations

import io
import warnings
from collections.abc import Callable
from functools import wraps
from shutil import which

from pennylane.tape import QuantumScript
from pennylane.workflow.qnode import QNode
from xdsl.dialects.builtin import ModuleOp

from catalyst.jit import QJIT
from catalyst.passes.pass_api import PassPipelineWrapper
from catalyst.python_interface.compiler import Compiler

from .collector import QMLCollector
from .specs import StopCompilation
from .xdsl_conversion import get_mlir_module

HAS_MATPLOTLIB = True
try:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False

HAS_PYDOT = True
try:
    import pydot
except ModuleNotFoundError:
    HAS_PYDOT = False

HAS_GRAPHVIZ = True
if which("dot") is None:
    HAS_GRAPHVIZ = False

# pylint: disable=wrong-import-position
from .construct_circuit_dag import ConstructCircuitDAG
from .pydot_dag_builder import PyDotDAGBuilder

# TODO: This caching mechanism should be improved,
# because now it relies on a mutable global state
_cache_store: dict[Callable, dict[int, tuple[str, str]]] = {}


def draw(qnode: QNode, *, level: int | None = None) -> Callable:
    """
    Draw the QNode at the specified level.

    This function can be used to visualize the QNode at different stages of the transformation
    pipeline when xDSL or Catalyst compilation passes are applied.
    If the specified level is not available, the highest level will be used as a fallback.

    The provided QNode is assumed to be decorated with compilation passes.
    If no passes are applied, the original QNode is visualized.

    Args:
        qnode (QNode): the input QNode that is to be visualized. The QNode is assumed to be
            compiled with ``qjit``.
        level (int | None): the level of transformation to visualize. If `None`, the final
            level is visualized.

    Returns:
        Callable:
            A wrapper function that visualizes the QNode at the specified level.

    """
    cache: dict[int, tuple[str, str]] = _cache_store.setdefault(qnode, {})

    def _draw_callback(previous_pass, module, next_pass, pass_level=0):
        """Callback function for circuit drawing."""

        pass_instance = previous_pass if previous_pass else next_pass
        collector = QMLCollector(module)
        ops, meas = collector.collect()
        tape = QuantumScript(ops, meas)
        pass_name = pass_instance.name if hasattr(pass_instance, "name") else pass_instance
        cache[pass_level] = (
            tape.draw(show_matrices=False),
            pass_name if pass_level else "No transforms",
        )

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        if args or kwargs:
            warnings.warn(
                "The `draw` function does not yet support dynamic arguments.\n"
                "To visualize the circuit with dynamic parameters or wires, please use the\n"
                "`catalyst.python_interface.inspection.generate_mlir_graph` function instead.",
                UserWarning,
            )
        mlir_module = get_mlir_module(qnode, args, kwargs)
        Compiler.run(mlir_module, callback=_draw_callback)

        if not cache:
            return None

        return cache.get(level, cache[max(cache.keys())])[0]

    return wrapper


def check_draw_imports():
    """
    Raise errors and early exit if dependencies of the draw feature are missing.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "The draw_graph functionality requires matplotlib to be installed. "
            "You can install matplotlib via 'pip install matplotlib'."
        )
    if not HAS_GRAPHVIZ:
        raise ImportError(
            "The Graphviz package is not found. Please install it for your system by "
            "following the instructions found here: https://graphviz.org/download/"
        )
    if not HAS_PYDOT:
        raise ImportError(
            "The 'pydot' package is not found. Please install with 'pip install pydot'."
        )


# pylint: disable=line-too-long
def draw_graph(qnode: QJIT, *, level: int | None = None) -> Callable:
    """
    Visualize a single QJIT compiled QNode, showing wire flow through quantum operations,
    program structure, and pass-by-pass impacts on compiled programs.

    .. note::

        The ``draw_graph`` function visualizes a QJIT-compiled QNode in a similar manner as
        `view-op-graph does in MLIR <https://mlir.llvm.org/docs/Passes/#-view-op-graph>`_,
        which leverages `Graphviz <https://graphviz.org/download/>`_ to show data-flow in the
        compiled IR.

        As such, use of ``draw_graph`` requires installation of
        `Graphviz <https://graphviz.org/download/>`_,
        `pydot <https://pypi.org/project/pydot/>`_, and `matplotlib <https://matplotlib.org/stable/install/index.html>`_ software packages.
        Please consult the links provided for installation instructions.

        Additionally, it is recommended to use ``draw_graph`` with PennyLane's program capture
        enabled (see :func:`qml.capture.enable <pennylane.capture.enable>`).

    .. warning::

        This function only visualizes quantum operations contained in workflows involving a single
        ``qjit``-compiled QNode. Workflows involving multiple QNodes or operations outside QNodes
        cannot yet be visualized.

        Only transformations found within the Catalyst compiler can be visualized. Any PennyLane
        tape transform will have already been applied before lowering to MLIR and will appear as
        the base state (``level=0``) in this visualization.

    Args:
        qnode (QJIT):
            The input qjit-compiled QNode that is to be visualized. The QNode is assumed to be
            compiled with qjit.
        level (int | None):
            The level of transformation to visualize. If ``None``, the final level is visualized.

    Returns:
        Callable:
            A function that has the same argument signature as the compiled QNode.
            When called, the function will return the graph
            as a tuple of (``matplotlib.figure.Figure``, ``matplotlib.axes._axes.Axes``) pairs.

    Raises:
        VisualizationError:
            If the circuit contains operations that cannot be converted to a graphical
            representation.
        TypeError:
            If the ``level`` argument is not of type integer or ``None``. If the input ``QNode`` is not
            qjit-compiled.
        ValueError:
            If the ``level`` argument is a negative integer.

    Warns:
        UserWarning:
            If the ``level`` argument provided is larger than the number of passes present in the
            compilation pipeline.

        Lastly, ``catalyst.draw_graph`` is currently not compatible with dynamic wire allocation.
        This includes :func:`pennylane.allocation.allocate` and dynamic wire allocation that may
        occur in MLIR directly (via ``quantum.alloc_qb`` instructions).

    **Example**

    Using ``draw_graph`` requires a ``qjit``'d QNode and a ``level`` argument, which denotes the
    cumulative set of applied compilation transforms (in the order they appear) to be applied and
    visualized.

    .. code-block::

        import pennylane as qml
        import catalyst

        qml.capture.enable()

        @qml.qjit
        @qml.transforms.merge_rotations
        @qml.transforms.cancel_inverses
        @qml.qnode(qml.device("null.qubit", wires=3))
        def circuit():
            qml.H(0)
            qml.T(1)
            qml.H(0)
            qml.RX(0.1, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.X(0))

    With ``level=0``, the graphical visualization will display the program as if no transforms are
    applied:

    >>> fig, ax = catalyst.draw_graph(circuit, level=0)()
    >>> fig.savefig('path_to_file.png', dpi=300, bbox_inches="tight")

    .. figure:: ../../../doc/_static/catalyst-draw-graph-level0-example.png
        :width: 35%
        :alt: Graphical representation of circuit with level=0
        :align: left

    Though you can ``print`` the output of ``catalyst.draw_graph``, it is recommended to use the
    ``savefig`` method of ``matplotlib.figure.Figure`` for better control over image resolution
    (DPI). Please consult the
    `matplotlib documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html>`_
    for usage details of ``savefig``.

    With ``level=2``, both :func:`~.passes.merge_rotations` and :func:`~.passes.cancel_inverses`
    will be applied, resulting in the two Hadamards cancelling and the two rotations merging:

    >>> fig, ax = catalyst.draw_graph(circuit, level=2)()
    >>> fig.savefig('path_to_file.png', dpi=300, bbox_inches="tight")

    .. figure:: ../../../doc/_static/catalyst-draw-graph-level2-example.png
        :width: 35%
        :alt: Graphical representation of circuit with level=2
        :align: left

    .. details::
        :title: Usage Details

        **Visualizing Control Flow**

        The ``draw_graph`` function can be used to visualize control flow, resulting in a scalable
        representation that preserves program structure:

        .. code-block::

            @qml.qjit(autograph=True)
            @qml.qnode(qml.device("null.qubit", wires=3))
            def circuit():
                qml.H(0)
                for i in range(3):
                    if i == 1:
                        qml.X(0)
                    elif i == 2:
                        qml.Y(0)
                    else:
                        qml.Z(0)
                return qml.probs()

        >>> fig, ax = catalyst.draw_graph(circuit)()
        >>> fig.savefig('path_to_file.png', dpi=300, bbox_inches="tight")

        .. figure:: ../../../doc/_static/catalyst-draw-graph-control-flow-example.png
            :width: 35%
            :alt: Graphical representation of circuit with control flow
            :align: left

        As one can see, the program structure is preserved in the figure.

        **Visualizing Dynamic Circuits**

        Circuits can depend on parameters that are not known at compile time, which result in
        conventional visualization tools failing. Consider the following circuit.

        .. code-block::

            @qml.qjit
            @qml.qnode(qml.device("null.qubit", wires=3))
            def circuit(x, y):
                qml.X(0)
                qml.Y(1)
                qml.Z(2)
                qml.H(x) # 'x' is a dynamic wire index
                qml.S(0)
                qml.T(2)
                qml.H(x)
                return qml.expval(qml.Z(y))

        The two ``qml.H`` gates act on wires that are dynamic. In order to preserve qubit data
        flow, each dynamic operator acts as a "choke point" to all currently active wires.
        To visualize this clearly, we use dashed lines to represent a dynamic dependency
        and solid lines for static/known values:

        >>> x, y = 1, 0
        >>> fig, ax = catalyst.draw_graph(circuit)(x, y)
        >>> fig.savefig('path_to_file.png', dpi=300, bbox_inches="tight")

        .. figure:: ../../../doc/_static/catalyst-draw-graph-dynamic-wire-example.png
            :width: 35%
            :alt: Graphical representation of circuit with control flow
            :align: left
    """

    check_draw_imports()

    if not isinstance(level, (int, type(None))):
        raise TypeError("The 'level' argument must be an integer or 'None'.")
    if isinstance(level, int) and level < 0:
        raise ValueError("The 'level' argument must be a positive integer.")

    max_level = None
    if isinstance(level, int):
        max_level = level

    if not isinstance(qnode, QJIT) or (
        not isinstance(qnode.original_function, QNode)
        and not (
            isinstance(qnode.original_function, PassPipelineWrapper)
            and isinstance(qnode.original_qnode, QNode)
        )
    ):
        raise TypeError(
            "The circuit must be a qjit-compiled qnode. "
            "Please apply the 'qml.qjit' function to your qnode."
        )

    cache: dict[int, tuple[str, str]] = {}

    def _draw_callback(previous_pass, module: ModuleOp, next_pass, pass_level: int = 0):
        """Callback function for circuit drawing."""

        pass_instance = previous_pass if previous_pass else next_pass
        utility = ConstructCircuitDAG(PyDotDAGBuilder())
        utility.construct(module)
        # Default DPI to 300 and let user fine tune control through the return MPL figure
        utility.dag_builder.graph.set_dpi(300)
        # Store DAG in cache
        # TODO: Update DAGBuilder to abstract away PyDot requirement
        dot_string = utility.dag_builder.to_string()
        pass_name = pass_instance.name if hasattr(pass_instance, "name") else pass_instance
        cache[pass_level] = (
            dot_string,
            pass_name if pass_level else "Before MLIR Passes",
        )

        if max_level is not None and pass_level >= max_level:
            raise StopCompilation("Stopping compilation after reaching max visualization level.")

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        cache.clear()
        mlir_module = get_mlir_module(qnode, args, kwargs)
        try:
            Compiler.run(mlir_module, callback=_draw_callback)
        except StopCompilation:
            # We use StopCompilation to interrupt the compilation once we reach
            # the desired level
            pass

        max_level = max(cache.keys())

        if max_level and isinstance(level, int) and level > max_level:
            warnings.warn(
                f"Level requested ({level}) is higher than the number of compilation passes present: {max_level}."
            )

        dot_string, _ = cache.get(level, cache[max_level])
        # TODO:  Remove dependency on PyDot
        (graph,) = pydot.graph_from_dot_data(dot_string)

        try:
            image_bytes = graph.create(prog="dot", format="png")
        except Exception as e:
            raise RuntimeError(
                "Failed to render graph. Ensure Graphviz is installed and 'dot' is in your PATH. "
                f"Original error: {e}"
            ) from e

        fig, ax = plt.subplots()

        img = mpimg.imread(io.BytesIO(image_bytes), format="png")
        ax.imshow(img)
        ax.set_axis_off()
        return fig, ax

    # Store cache on wrapper
    # pylint: disable = protected-access
    wrapper._cache = cache

    return wrapper
