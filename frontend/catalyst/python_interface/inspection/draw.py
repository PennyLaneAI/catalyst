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
from functools import wraps
from typing import TYPE_CHECKING

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from pennylane.tape import QuantumScript
from xdsl.dialects.builtin import ModuleOp

from catalyst.python_interface.compiler import Compiler

from ..visualization.construct_circuit_dag import ConstructCircuitDAG
from ..visualization.pydot_dag_builder import PyDotDAGBuilder
from .collector import QMLCollector
from .xdsl_conversion import get_mlir_module

if TYPE_CHECKING:
    from pennylane.typing import Callable
    from pennylane.workflow.qnode import QNode

HAS_MATPLOTLIB = True
try:
    import matplotlib
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False

# TODO: This caching mechanism should be improved,
# because now it relies on a mutable global state
_cache_store: dict[Callable, dict[int, tuple[str, str]]] = {}


def draw(qnode: QNode, *, level: None | int = None) -> Callable:
    """
    Draw the QNode at the specified level.

    This function can be used to visualize the QNode at different stages of the transformation
    pipeline when xDSL or Catalyst compilation passes are applied.
    If the specified level is not available, the highest level will be used as a fallback.

    The provided QNode is assumed to be decorated with compilation passes.
    If no passes are applied, the original QNode is visualized.

    Args:
        qnode (.QNode): the input QNode that is to be visualized. The QNode is assumed to be
            compiled with ``qjit``.
        level (None | int): the level of transformation to visualize. If `None`, the final
            level is visualized.


    Returns:
        Callable: A wrapper function that visualizes the QNode at the specified level.

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


def draw_graph(qnode: QNode, *, level: None | int = None) -> Callable:
    """
    Visualize QNodes as graphs, showing wire flow through quantum operations, program structure, and
    pass-by-pass impacts on compiled programs.

    .. note::

        The ``draw_graph`` function requires installation of
        `Graphviz <https://graphviz.org/download/>` and the
        `pydot <https://pypi.org/project/pydot/>` software package. Please consult the links
        provided for installation instructions.

    .. warning::

        This feature must be used in tandem with PennyLane's program capture, enabled with
        :func:`pennylane.capture.enable`.

    Args:
        qnode (.QNode): the input QNode that is to be visualized. The QNode is assumed to be
            compiled with ``qjit``.
        level (None | int): the level of transformation to visualize. If ``None``, the final
            level is visualized. # what are default options?

    Returns:
        Callable: A function that has the same argument signature as the compiled ``qnode``. 
            When called, the function will return the graph as a tuple of 
            (``matplotlib.figure.Figure``, ``matplotlib.axes._axes.Axes``). 

    Raises:
        VisualizationError: If the circuit contains operations that cannot be 
            converted to a graphical representation.

    **Example**

    The ``draw_graph`` function visualizes QNodes in a similar manner as
    `view-op-graph in traditional MLIR <https://mlir.llvm.org/docs/Passes/#-view-op-graph>`, which
    leverages `Graphviz <https://graphviz.org/download/>` to show data-flow in compiled programs.

    Using ``draw_graph`` requires a qjit'd QNode and a ``level`` argument, which denotes the
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
        def circuit(x, y):
            # TODO

    >>> x, y = # TODO
    >>> print(catalyst.draw_graph)(x, y)

    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "The `draw_graph` functionality requires `matplotlib` to be installed. Please install `matplotlib` with `pip install matplotlib`."
        )

    cache: dict[int, tuple[str, str]] = _cache_store.setdefault(qnode, {})

    def _draw_callback(previous_pass, module: ModuleOp, next_pass, pass_level: int = 0):
        """Callback function for circuit drawing."""

        pass_instance = previous_pass if previous_pass else next_pass
        # Process module to build DAG
        utility = ConstructCircuitDAG(PyDotDAGBuilder())
        utility.construct(module)
        # Store DAG in cache
        utility.dag_builder.graph.set_dpi(300)
        # TODO: Update DAGBuilder to abstract away extracting image bytes
        image_bytes = utility.dag_builder.graph.create_png(prog="dot")
        pass_name = pass_instance.name if hasattr(pass_instance, "name") else pass_instance
        cache[pass_level] = (
            image_bytes,
            pass_name if pass_level else "No transforms",
        )

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        mlir_module = get_mlir_module(qnode, args, kwargs)
        Compiler.run(mlir_module, callback=_draw_callback)

        if not cache:
            return None

        max_level = max(cache.keys())
        image_bytes, _ = cache.get(level, cache[max_level])

        sio = io.BytesIO()
        sio.write(image_bytes)
        sio.seek(0)

        img = mpimg.imread(sio)

        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_axis_off()
        fig.set_dpi(300)
        return fig, ax

    return wrapper
