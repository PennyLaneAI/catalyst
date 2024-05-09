# Copyright 2022-2024 Xanadu Quantum Technologies Inc.

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
This module contains public API functions which represent higher-order
transformations on functions, for example the vectorization map which adds
additional dimensions to the inputs and outputs of a function.
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.tree_util import tree_flatten, tree_leaves, tree_structure, tree_unflatten

from catalyst.api_extensions.control_flow import for_loop
from catalyst.tracing.contexts import EvaluationContext


## API ##
def vmap(
    fn: Callable,
    in_axes: Union[int, Sequence[Any]] = 0,
    out_axes: Union[int, Sequence[Any]] = 0,
    axis_size: Optional[int] = None,
) -> Callable:
    """A :func:`~.qjit` compatible vectorizing map.
    Creates a function which maps an input function over argument axes.

    Args:
        f (Callable): A Python function containing PennyLane quantum operations.
        in_axes (Union[int, Sequence[Any]]): Specifies the value(s) over which input
            array axes to map.
        out_axes (Union[int, Sequence[Any]]): Specifies where the mapped axis should appear
            in the output.
        axis_size (int): An integer can be optionally provided to indicate the size of the
            axis to be mapped. If omitted, the size of the mapped axis will be inferred from
            the provided arguments.

    Returns:
        Callable: Vectorized version of ``fn``.

    Raises:
        ValueError: Invalid ``in_axes``, ``out_axes``, and ``axis_size`` values.

    **Example**

    For example, consider the following QNode:

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(x, y):
          qml.RX(jnp.pi * x[0] + y, wires=0)
          qml.RY(x[1] ** 2, wires=0)
          qml.RX(x[1] * x[2], wires=0)
          return qml.expval(qml.PauliZ(0))

    >>> circuit(jnp.array([0.1, 0.2, 0.3]), jnp.pi)
    Array(-0.93005586, dtype=float64)

    We can use ``catalyst.vmap`` to introduce additional batch dimensions
    to our input arguments,
    without needing to use a Python for loop:

    >>> x = jnp.array([[0.1, 0.2, 0.3],
    ...                [0.4, 0.5, 0.6],
    ...                [0.7, 0.8, 0.9]])
    >>> y = jnp.array([jnp.pi, jnp.pi / 2, jnp.pi / 4])
    >>> qjit(vmap(cost))(x, y)
    array([-0.93005586, -0.97165424, -0.6987465 ])

    ``catalyst.vmap()`` has been implemented to match the same behaviour of
    ``jax.vmap``, so should be a drop-in replacement in most cases.
    Under-the-hood, it is automatically inserting Catalyst-compatible for loops,
    which will be compiled and executed outside of Python for increased performance.

    Outside of a Catalyst qjit-compiled function, ``vmap`` will simply dispatch to
    ``jax.vmap``.

    .. details::
        :title: Selecting batching axes for arguments

        The ``in_axes`` parameter provides different modes the allow large- and fine-grained
        control over which arguments to apply the batching transformation on. Enabling batching for
        a particular argument requires that the selected axis be of the same size as the determined
        batch size, which is the same for all arguments.

        The following modes are supported:

        - ``int``: Specifies the same batch axis for all arguments
        - ``Tuple[int]``: Specify a different batch axis for each argument
        - ``Tuple[int | None]``: Same as previous, but selectively disable batching for certain
          arguments with a ``None`` value
        - ``Tuple[int | PyTree[int] | None]``: Same as previous, but specify a different batch
          axis for each leaf of an argument (Note that the ``PyTreeDefs``, i.e. the container
          structure, must match between the ``in_axes`` element and the corresponding argument.)
        - ``Tuple[int | PyTree[int | None] | None]``: Same as previous, but selectively disable
          batching for individual PyTree leaves

        The ``out_axes`` parameter can be also used to specify the positions of the mapped axis
        in the output. ``out_axes`` is subject to the same modes as well.
    """

    # Check the validity of in_axes and out_axes
    if not all(isinstance(l, int) for l in tree_leaves(in_axes)):
        raise ValueError(
            "Invalid 'in_axes'; it must be an int or a tuple of PyTrees with integer leaves, "
            f"but got {in_axes}"
        )

    if not all(isinstance(l, int) for l in tree_leaves(out_axes)):
        raise ValueError(
            "Invalid 'out_axes'; it must be an int or a tuple of PyTree with integer leaves, "
            f"but got {out_axes}"
        )

    def batched_fn(*args, **kwargs):
        """Vectorization wrapper around the hybrid program using catalyst.for_loop"""

        # Dispatch to jax.vmap when it is called outside qjit.
        if not EvaluationContext.is_tracing():
            return jax.vmap(fn, in_axes, out_axes)(*args, **kwargs)

        args_flat, args_tree = tree_flatten(args)
        in_axes_flat, _ = tree_flatten(in_axes, is_leaf=lambda x: x is None)

        # Check the validity of the input arguments w.r.t. in_axes
        in_axes_deep_struct = tree_structure(in_axes, is_leaf=lambda x: x is None)
        args_deep_struct = tree_structure(args, is_leaf=lambda x: x is None)
        if not isinstance(in_axes, int) and in_axes_deep_struct != args_deep_struct:
            raise ValueError(
                "Invalid 'in_axes'; it must be an int or match the length of positional "
                f"arguments, but got {in_axes_deep_struct} axis specifiers "
                f"and {args_deep_struct} arguments."
            )
        if isinstance(in_axes, int):
            in_axes_flat = [
                in_axes,
            ] * len(args_flat)

        batch_size = _get_batch_size(args_flat, in_axes_flat, axis_size)
        batch_loc = _get_batch_loc(in_axes_flat)

        # Prepare args_flat to run 'fn' one time and get the output-shape
        fn_args_flat = args_flat.copy()
        for loc in batch_loc:
            ax = in_axes_flat[loc]
            fn_args_flat[loc] = jnp.take(args_flat[loc], 0, axis=ax)

        fn_args = tree_unflatten(args_tree, fn_args_flat)

        # Run 'fn' one time to get output-shape
        init_result = fn(*fn_args, **kwargs)

        # Check the validity of the output w.r.t. out_axes
        out_axes_deep_struct = tree_structure(out_axes, is_leaf=lambda x: x is None)
        init_result_deep_struct = tree_structure(init_result, is_leaf=lambda x: x is None)
        if not isinstance(out_axes, int) and out_axes_deep_struct != init_result_deep_struct:
            raise ValueError(
                "Invalid 'out_axes'; it must be an int or match "
                "the number of function results, but got "
                f"{out_axes_deep_struct} axis specifiers and {init_result_deep_struct} results."
            )

        init_result_flat, init_result_tree = tree_flatten(init_result)

        num_axes_out = len(init_result_flat)

        if isinstance(out_axes, int):
            out_axes_flat = [
                out_axes,
            ] * num_axes_out
        else:
            out_axes_flat, _ = tree_flatten(out_axes, is_leaf=lambda x: x is None)

        out_loc = _get_batch_loc(out_axes_flat)

        # Store batched results of all leaves
        # in the flatten format with respect to the 'init_result' shape
        batched_result_list = []
        for j in range(num_axes_out):
            out_shape = (
                (batch_size,)
                if not init_result_flat[j].shape
                else (batch_size, *init_result_flat[j].shape)
            )
            batched_result_list.append(jnp.zeros(shape=out_shape, dtype=init_result_flat[j].dtype))
            batched_result_list[j] = batched_result_list[j].at[0].set(init_result_flat[j])

        # Apply mapping batched_args[1:] ---> fn(args)
        @for_loop(1, batch_size, 1)
        def loop_fn(i, batched_result_list):
            fn_args_flat = args_flat
            for loc in batch_loc:
                ax = in_axes_flat[loc]
                fn_args_flat[loc] = jnp.take(args_flat[loc], i, axis=ax)

            fn_args = tree_unflatten(args_tree, fn_args_flat)
            res = fn(*fn_args, **kwargs)

            res_flat, _ = tree_flatten(res)

            # Update the list of results
            for j in range(num_axes_out):
                batched_result_list[j] = batched_result_list[j].at[i].set(res_flat[j])

            return batched_result_list

        batched_result_list = loop_fn(batched_result_list)

        # Support out_axes on dim > 0
        for loc in out_loc:
            if ax := out_axes_flat[loc]:
                up_axes = [*range(batched_result_list[loc].ndim)]
                up_axes[ax], up_axes[0] = up_axes[0], up_axes[ax]
                batched_result_list[loc] = jnp.transpose(batched_result_list[loc], up_axes)

        # Unflatten batched_result before return
        return tree_unflatten(init_result_tree, batched_result_list)

    return batched_fn


## PRIVATE ##
def _get_batch_loc(axes_flat):
    """
    Get the list of mapping locations in the flattened list of in-axes or out-axes.

    This function takes a flattened list of axes and identifies all elements with a
    non-None value. The resulting list contains the indices of these non-None values,
    indicating where the mapping should apply.

    Args:
        axes_flat (List): Flattened list of in-axes or out-axes including `None` elements.

    Returns:
        List: A list of indices representing the locations where the mapping should be applied.
    """

    return [i for i, d in enumerate(axes_flat) if d is not None]


def _get_batch_size(args_flat, axes_flat, axis_size):
    """Get the batch size based on the provided arguments and axes specifiers, or the manually
       specified batch size by the user request. The batch size must be the same for all arguments.

    Args:
        args_flat (List): Flatten list of arguments.
        axes_flat (List): Flatten list of `in_axes` or `our_axes` including `None` elements.
        axis_size (Optional[int]): Optional default batch size.

    Returns:
        int: Returns the batch size used as the upper bound of the QJIT-compatible for loop
            in the computation of vmap.

    Raises:
        ValueError: The batch size must be the same for all arguments.
        ValueError: The default batch is expected to be None, or less than or equal
        to the computed batch size.
    """

    batch_sizes = []
    for arg, d in zip(args_flat, axes_flat):
        shape = np.shape(arg) if arg.shape else (1,)
        if d is not None and len(shape) > d:
            batch_sizes.append(shape[d])

    if any(size != batch_sizes[0] for size in batch_sizes[1:]):
        raise ValueError(
            "Invalid batch sizes; expected the batch size to be the same for all arguments, "
            f"but got batch_sizes={batch_sizes} from args_flat={args_flat}"
        )

    batch_size = batch_sizes[0] if batch_sizes else 0

    if axis_size is not None:
        if axis_size <= batch_size:
            batch_size = axis_size
        else:
            raise ValueError(
                "Invalid 'axis_size'; the default batch is expected to be None, "
                "or less than or equal to the computed batch size, but got "
                f"axis_size={axis_size} > batch_size={batch_size}"
            )

    if not batch_size:
        raise ValueError(
            f"Invalid batch size; it must be a non-zero integer, but got {batch_size}."
        )

    return batch_size
