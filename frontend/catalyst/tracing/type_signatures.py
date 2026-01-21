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
from jax._src.core import DShapedArray, shaped_abstractify
from jax._src.interpreters.partial_eval import infer_lambda_input_type
from jax._src.pjit import _flat_axes_specs
from jax.core import AbstractValue
from jax.tree_util import tree_flatten, tree_unflatten

from catalyst.jax_extras import get_aval2
from catalyst.utils.exceptions import CompileError
from catalyst.utils.patching import Patcher

POSITIONAL_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.VAR_POSITIONAL,
)

KEYWORD_KINDS = (
    inspect.Parameter.KEYWORD_ONLY,
    inspect.Parameter.VAR_KEYWORD,
)


def get_stripped_signature(fn: Callable) -> inspect.Signature:
    """Return the function's signature without annotations."""
    old_params = inspect.signature(fn).parameters.values()
    new_params = [param.replace(annotation=inspect.Parameter.empty) for param in old_params]

    return inspect.Signature(new_params)


def get_annotations_from_signature(sig):
    return {param.name: param.annotation for param in sig.parameters.values()}


def get_dynamic_sig(full_sig, static_argnums):
    return inspect.Signature(
        (param for i, param in enumerate(full_sig.parameters.values()) if i not in static_argnums)
    )


def get_arg_annotations_from_signature(sig):
    """
    Return arguments shape from a functions signature.
    """

    return tuple(
        param.annotation for param in sig.parameters.values() if param.kind in POSITIONAL_KINDS
    )


def get_kwarg_annotations_from_signature(sig):
    """
    Return arguments shape from a functions signature.
    """

    return {
        param.name: param.annotation
        for param in sig.parameters.values()
        if param.kind in KEYWORD_KINDS
    }


def all_params_are_annotated(sig):
    """Return true if all parameters are type-annotated, or no parameters are present."""
    assert isinstance(sig, inspect.Signature)

    params = sig.parameters.values()
    # print(f"parameters: {params}")
    are_annotated = all(param.annotation is not inspect.Parameter.empty for param in params)
    # print(f"are_annotated? {are_annotated}")

    if not are_annotated:
        return False

    # for param in params:
    #     if not isinstance(param.annotation, (type, AbstractValue)):
    #         print(f"param {param} with annotation {param.annotation} of type {type(param.annotation)} failed the check")

    are_valid_annotations = all(
        isinstance(param.annotation, (type, AbstractValue)) for param in params
    )
    # print(f"are typed? {are_typed}")
    return are_valid_annotations


def get_type_annotations_if_all_annotated(fn):
    """Get type annotations if all parameters are annotated.

    Args:
        fn (Callable): function whose annotations to retrieve.

    Returns:
        dict | None: type annotations for all parameters if available, otherwise None.
    """
    assert isinstance(fn, Callable)

    if fn is not None and all_params_are_annotated(fn):
        # print("params are annotated, getting annotations")
        return inspect.get_annotations(fn)
    # else:
    # print(f"type annotations failed, function is None? {fn is None}")

    return None


def get_abstract_args(args, kwargs):
    """Get abstract values from real arguments, preserving PyTrees.

    Args:
        args (Iterable): arguments to convert

    Returns:
        Iterable: ShapedArrays for the provided arguments
        dict: ShapedArrays for the provided keyword arguments
    """

    flat_args, in_tree = tree_flatten((args, kwargs))
    flat_abstract_args = [shaped_abstractify(arg) for arg in flat_args]
    return tree_unflatten(in_tree, flat_abstract_args)


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


def verify_static_argnums(args, kwargs, static_argnums):
    """Verify that static_argnums have correct type and range.
    Raises a `CompileError` if not.

    Args:
        args (Iterable): arguments to a compiled function
        static_argnums (Iterable[int]): indices to verify

    Returns:
        None
    """
    # check that all argnums are ints
    verify_static_argnums_type(static_argnums)
    for argnum in static_argnums:
        if argnum < 0 or argnum >= len(args) + len(kwargs):
            msg = f"argnum {argnum} is beyond the valid range of [0, {len(args) + len(kwargs)})."
            raise CompileError(msg)
    return None


def filter_static_args(args, kwargs, static_argnums):
    """Remove static values from arguments using the provided index list.

    Args:
        args (Iterable): arguments to a compiled function
        static_argnums (Iterable[int]): indices to filter

    Returns:
        Tuple: dynamic arguments
        dict: dynamic keyword arguments
    """
    # print(f"filtering with {args}, {kwargs} and {static_argnums}")
    return tuple(arg for i, arg in enumerate(args) if i not in static_argnums), {
        key: value
        for i, (key, value) in enumerate(kwargs.items())
        if len(args) + i not in static_argnums
    }


def split_static_args(args, kwargs, static_argnums):
    """Split arguments into static and dynamic values using the provided index list.

    Args:
        args (Iterable): arguments to a compiled function
        static_argnums (Iterable[int]): indices to split on

    Returns:
        Tuple: dynamic arguments
        Tuple: static arguments
    """
    split_args = ([], [])
    split_kwargs = ({}, {})

    for i in range(len(args)):
        split_args[int(i in static_argnums)].append(args[i])

    for i, argname in enumerate(kwargs):
        split_kwargs[int(len(args) + i in static_argnums)][argname] = kwargs[argname]

    return tuple(split_args[0]), split_kwargs[0], tuple(split_args[1]), split_kwargs[1]


def merge_static_argname_into_argnum(fn: Callable, static_argnames, static_argnums):
    """Map static_argnames of the callable to the corresponding argument indices,
    and add them to static_argnums"""
    new_static_argnums = [] if (static_argnums is None) else list(static_argnums)
    fn_argnames = list(inspect.signature(fn).parameters.keys())

    # static_argnames can be a single str, or a list/tuple of strs
    # convert all of them to list
    if isinstance(static_argnames, str):
        static_argnames = [static_argnames]

    non_existent_args = []
    for static_argname in static_argnames:
        if static_argname in fn_argnames:
            new_static_argnums.append(fn_argnames.index(static_argname))
            continue
        non_existent_args.append(static_argname)

    if non_existent_args:
        non_existent_args_str = "{" + ", ".join(repr(item) for item in non_existent_args) + "}"

        raise ValueError(
            f"qjitted function has invalid argname {non_existent_args_str} in static_argnames. "
            "Function does not take these args."
        )

    # Remove potential duplicates from static_argnums and static_argnames
    new_static_argnums = tuple(sorted(set(new_static_argnums)))

    return new_static_argnums


def merge_static_args(abstract_sig, static_sig, static_argnums):
    """Merge abstracted dynamic arguments back into call arguments, retaining the original ordering.

    Args:
        sig (inspect.Signature): signature of the function
        abstract_args (Iterable[ShapedArray]): abstracted values of dynamic arguments
        args (Iterable): original arguments with static values
        static_argnums (Iterable[int]): indices of static arguments

    Returns:
        Tuple[ShapedArray | Any]: a mixture of ShapedArrays and static argument values
    """
    # printf"merging static sig {static_sig} and abstract sig {abstract_sig}")
    if not static_argnums:
        return abstract_sig

    abstract_args = abstract_sig[0]
    abstracts_kwargs = abstract_sig[1]

    static_args = static_sig[0]
    static_kwargs = static_sig[1]

    merged_args = list(static_args)
    merged_kwargs = static_kwargs  # TODO make sure this doesn't mutate the original

    abstract_index = 0
    for i, arg in enumerate(static_args):
        if i not in static_argnums:
            merged_args[i] = abstract_args[abstract_index]
            abstract_index += 1

    for i, key in enumerate(static_kwargs):
        if len(static_args) + i not in static_args:
            merged_kwargs[key] = abstract_kwargs[key]
            abstract_index += 1

    return (tuple(merged_args), merged_kwargs)


def get_decomposed_signature(args, kwargs, static_argnums):
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
    dynamic_args, dynamic_kwargs, static_args, static_kwargs = split_static_args(
        args, kwargs, static_argnums
    )
    flat_dynamic_args, dynamic_in_tree = tree_flatten((dynamic_args, dynamic_kwargs))

    flat_abstract_args = tuple(shaped_abstractify(arg) for arg in flat_dynamic_args)
    return flat_abstract_args, dynamic_in_tree, static_args


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
        # print(f"compile specs from {abstracted_axes} and {flat_compiled_sig}")
        axes_specs_compile = _flat_axes_specs(abstracted_axes, *flat_compiled_sig)
        # print(f"runtime specs from {abstracted_axes} and {runtime_signature}")
        axes_specs_runtime = _flat_axes_specs(abstracted_axes, *flat_runtime_sig)
        # print(f"compile in_types from {axes_specs_compile}, {flat_compiled_sig}")
        in_type_compiled = infer_lambda_input_type(axes_specs_compile, flat_compiled_sig)
        # print(f"runtime in_types from {axes_specs_runtime}, {flat_runtime_sig}")
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


def promote_arguments(target_call_signature, args, kwargs):
    """Promote arguments to the provided target signature, preserving PyTrees.

    Args:
        target_signature (Iterable): target signature to promote arguments to
        args (Iterable): arguments to promote, must have matching PyTrees with target signature

    Returns:
        Iterable: arguments promoted to target signature
    """
    flat_target_args, target_treedef = tree_flatten(target_call_signature)
    flat_args, treedef = tree_flatten((args, kwargs))
    assert target_treedef == treedef, "Argument PyTrees did not match target structure."

    promoted_args = []
    for c_param, r_param in zip(flat_target_args, flat_args):
        assert isinstance(c_param, jax.core.ShapedArray)
        r_param = jax.numpy.asarray(r_param)
        arg_dtype = r_param.dtype
        promote_to = jax.numpy.promote_types(arg_dtype, c_param.dtype)
        promoted_arg = jax.numpy.asarray(r_param, dtype=promote_to)
        promoted_args.append(promoted_arg)

    return tree_unflatten(treedef, promoted_args)


def get_arg_names(qjit_jaxpr_in_avals: tuple[AbstractValue, ...], qjit_original_function: Callable):
    """Construct a list of argument names, with the size of qjit_jaxpr_in_avals, and fill it with
    the names of the parameters of the original function signature.
    The number of parameters of the original function could be different to the number of
    elements in qjit_jaxpr_in_avals. For example, if a function with one parameter is invoked with a
    dynamic argument, qjit_jaxpr_in_avals will contain two elements (a dynamically-shaped array, and
    its type).

    Args:
        qjit_jaxpr_in_avals: list of abstract values that represent the inputs to the QJIT's JAXPR
        qjit_original_function: QJIT's original function

    Returns:
        A list of argument names with the same number of elements than qjit_jaxpr_in_avals.
        The argument names are assigned from the list of parameters of the original function,
        in order, and until that list is empty. Then left to empty strings.
    """
    arg_names = [""] * len(qjit_jaxpr_in_avals)
    param_values = [p.name for p in inspect.signature(qjit_original_function).parameters.values()]
    for in_aval_index, in_aval in enumerate(qjit_jaxpr_in_avals):
        if len(param_values) > 0 and type(in_aval) != DShapedArray:
            arg_names[in_aval_index] = param_values.pop(0)
    return arg_names
