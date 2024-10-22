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
Utility functions for obtaining and manipulating function signatures and
arguments in the context of tracing.
"""

import enum
import inspect
from typing import Callable

import jax
from jax._src.interpreters.partial_eval import infer_lambda_input_type
from jax._src.pjit import _flat_axes_specs
from jax.api_util import shaped_abstractify
from jax.tree_util import tree_flatten, tree_unflatten

from catalyst.jax_extras import get_aval2
from catalyst.utils.exceptions import CompileError
from catalyst.utils.patching import Patcher


def get_stripped_signature(fn: Callable):
    """Return the function's signature without annotations."""

    old_params = inspect.signature(fn).parameters.values()
    new_params = [param.replace(annotation=inspect.Parameter.empty) for param in old_params]

    return inspect.Signature(new_params)


def get_param_annotations(fn: Callable):
    """Return true all parameters typed-annotations."""
    assert isinstance(fn, Callable)
    signature = inspect.signature(fn)
    parameters = signature.parameters
    return [p.annotation for p in parameters.values()]


def params_are_annotated(fn: Callable):
    """Return true if all parameters are typed-annotated, or no parameters are present."""
    assert isinstance(fn, Callable)
    annotations = get_param_annotations(fn)
    are_annotated = all(annotation is not inspect.Parameter.empty for annotation in annotations)
    if not are_annotated:
        return False
    return all(isinstance(annotation, (type, jax.core.ShapedArray)) for annotation in annotations)


def get_type_annotations(fn: Callable):
    """Get type annotations if all parameters are annotated."""
    assert isinstance(fn, Callable)
    if fn is not None and params_are_annotated(fn):
        annotations = get_param_annotations(fn)
        return tuple(annotations)

    return None


def get_abstract_signature(args):
    """Get abstract values from real arguments, preserving PyTrees.

    Args:
        args (Iterable): arguments to convert

    Returns:
        Iterable: ShapedArrays for the provided values
    """
    flat_args, treedef = tree_flatten(args)

    abstract_args = [shaped_abstractify(arg) for arg in flat_args]

    return tree_unflatten(treedef, abstract_args)


def verify_static_argnums_type(static_argnums):
    """Verify that static_argnums have correct type.

    Args:
        static_argnums (Iterable[int]): indices to verify

    Returns:
        None
    """
    is_tuple = isinstance(static_argnums, tuple)
    is_valid = is_tuple and all(isinstance(arg, int) for arg in static_argnums)
    if not is_valid:
        raise TypeError(
            "The `static_argnums` argument to `qjit` must be an int or convertable to a"
            f"tuple of ints, but got value {static_argnums}"
        )
    return None


def verify_static_argnums(args, static_argnums):
    """Verify that static_argnums have correct type and range.

    Args:
        args (Iterable): arguments to a compiled function
        static_argnums (Iterable[int]): indices to verify

    Returns:
        None
    """
    verify_static_argnums_type(static_argnums)

    for argnum in static_argnums:
        if argnum < 0 or argnum >= len(args):
            msg = f"argnum {argnum} is beyond the valid range of [0, {len(args)})."
            raise CompileError(msg)
    return None


def filter_static_args(args, static_argnums):
    """Remove static values from arguments using the provided index list.

    Args:
        args (Iterable): arguments to a compiled function
        static_argnums (Iterable[int]): indices to filter

    Returns:
        Tuple: dynamic arguments
    """
    return tuple(args[idx] for idx in range(len(args)) if idx not in static_argnums)


def split_static_args(args, static_argnums):
    """Split arguments into static and dynamic values using the provided index list.

    Args:
        args (Iterable): arguments to a compiled function
        static_argnums (Iterable[int]): indices to split on

    Returns:
        Tuple: dynamic arguments
        Tuple: static arguments
    """
    dynamic_args, static_args = [], []
    for i, arg in enumerate(args):
        if i in static_argnums:
            static_args.append(arg)
        else:
            dynamic_args.append(arg)
    return tuple(dynamic_args), tuple(static_args)


def merge_static_argname_into_argnum(fn: Callable, static_argnames, static_argnums):
    """Map static_argnames of the callable to the corresponding argument indices,
    and add them to static_argnums"""
    new_static_argnums = [] if (static_argnums is None) else list(static_argnums)
    fn_argnames = list(inspect.signature(fn).parameters.keys())

    # static_argnames can be a single str, or a list/tuple of strs
    # convert all of them to list
    if isinstance(static_argnames, str):
        static_argnames = [static_argnames]

    for static_argname in static_argnames:
        if static_argname not in fn_argnames:
            raise ValueError(
                f"""
qjitted function has invalid argname {{{static_argname}}} in static_argnames.
Function does not take these args.
                """
            )
        new_static_argnums.append(fn_argnames.index(static_argname))

    # Remove potential duplicates from static_argnums and static_argnames
    new_static_argnums = tuple(sorted(set(new_static_argnums)))

    return new_static_argnums


def merge_static_args(signature, args, static_argnums):
    """Merge static arguments back into an abstract signature, retaining the original ordering.

    Args:
        signature (Iterable[ShapedArray]): abstract values of the dynamic arguments
        args (Iterable): original argument list to draw static values from
        static_argnums (Iterable[int]): indices to merge on

    Returns:
        Tuple[ShapedArray | Any]: a mixture of ShapedArrays and static argument values
    """
    if not static_argnums:
        return signature

    merged_sig = list(args)  # mutable copy

    dynamic_indices = [idx for idx in range(len(args)) if idx not in static_argnums]
    for i, idx in enumerate(dynamic_indices):
        merged_sig[idx] = signature[i]

    return tuple(merged_sig)


def get_decomposed_signature(args, static_argnums):
    """Decompose function arguments into dynamic and static arguments, where the dynamic arguments
    are further processed into abstract values and PyTree metadata. All values returned by this
    function are hashable.

    Args:
        args (Iterable): arguments to a compiled function
        static_argnums (Iterable[int]): indices to split on

    Returns:
        Tuple[ShapedArray]: dynamic argument shape and dtype information
        PyTreeDef: dynamic argument PyTree metadata
        Tuple[Any]: static argument values
    """
    dynamic_args, static_args = split_static_args(args, static_argnums)
    flat_dynamic_args, treedef = tree_flatten(dynamic_args)
    flat_signature = get_abstract_signature(flat_dynamic_args)

    return flat_signature, treedef, static_args


class TypeCompatibility(enum.Enum):
    """Enum class to indicate result of type compatibility analysis between two signatures."""

    UNKNOWN = 0
    CAN_SKIP_PROMOTION = 1
    NEEDS_PROMOTION = 2
    NEEDS_COMPILATION = 3


def typecheck_signatures(compiled_signature, runtime_signature, abstracted_axes=None):
    """Determine whether a signature is compatible with another, possibly via promotion and
    considering dynamic axes, and return either of three states:

     - fully compatible (skip promotion)
     - conditionally compatible (requires promotion)
     - incompatible (requires re-compilation)

    Args:
        compiled_signature (Iterable[ShapedArray]): base signature to compare against, typically the
            signature of a previously compiled function or user specified type hints
        runtime_signature (Iterable[ShapedArray]): signature to examine, usually from runtime
            arguments
        abstracted_axes

    Returns:
        TypeCompatibility
    """
    if compiled_signature is None:
        return TypeCompatibility.NEEDS_COMPILATION

    flat_compiled_sig, compiled_treedef = tree_flatten(compiled_signature)
    flat_runtime_sig, runtime_treedef = tree_flatten(runtime_signature)

    if compiled_treedef != runtime_treedef:
        return TypeCompatibility.NEEDS_COMPILATION

    # We first check signature equality considering dynamic axes, allowing the shape of an array
    # to be different if it was compiled with a dynamical shape.
    # TODO: unify this with the promotion checks, allowing the dtype to change for a dynamic axis
    with Patcher(
        # pylint: disable=protected-access
        (jax._src.interpreters.partial_eval, "get_aval", get_aval2),
    ):
        # TODO: do away with private jax functions
        axes_specs_compile = _flat_axes_specs(abstracted_axes, *compiled_signature, {})
        axes_specs_runtime = _flat_axes_specs(abstracted_axes, *runtime_signature, {})
        in_type_compiled = infer_lambda_input_type(axes_specs_compile, flat_compiled_sig)
        in_type_runtime = infer_lambda_input_type(axes_specs_runtime, flat_runtime_sig)

        if in_type_compiled == in_type_runtime:
            return TypeCompatibility.CAN_SKIP_PROMOTION

    action = TypeCompatibility.CAN_SKIP_PROMOTION
    for c_param, r_param in zip(flat_compiled_sig, flat_runtime_sig):
        if c_param.dtype != r_param.dtype:
            action = TypeCompatibility.NEEDS_PROMOTION

        if c_param.shape != r_param.shape:
            return TypeCompatibility.NEEDS_COMPILATION

        promote_to = jax.numpy.promote_types(r_param.dtype, c_param.dtype)
        if c_param.dtype != promote_to:
            return TypeCompatibility.NEEDS_COMPILATION

    return action


def promote_arguments(target_signature, args):
    """Promote arguments to the provided target signature, preserving PyTrees.

    Args:
        target_signature (Iterable): target signature to promote arguments to
        args (Iterable): arguments to promote, must have matching PyTrees with target signature

    Returns:
        Iterable: arguments promoted to target signature
    """
    flat_target_sig, target_treedef = tree_flatten(target_signature)
    flat_args, treedef = tree_flatten(args)
    assert target_treedef == treedef, "Argument PyTrees did not match target signature."

    promoted_args = []
    for c_param, r_param in zip(flat_target_sig, flat_args):
        assert isinstance(c_param, jax.core.ShapedArray)
        r_param = jax.numpy.asarray(r_param)
        arg_dtype = r_param.dtype
        promote_to = jax.numpy.promote_types(arg_dtype, c_param.dtype)
        promoted_arg = jax.numpy.asarray(r_param, dtype=promote_to)
        promoted_args.append(promoted_arg)

    return tree_unflatten(treedef, promoted_args)
