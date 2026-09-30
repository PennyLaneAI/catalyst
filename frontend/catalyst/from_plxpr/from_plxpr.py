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
This submodule defines a utility for converting plxpr into Catalyst jaxpr.
"""

# pylint: disable=protected-access

from copy import copy
from functools import partial
from typing import Callable

import jax
import pennylane as qp
from jax.extend.core import ClosedJaxpr, Jaxpr
from pennylane.capture import PlxprInterpreter, qnode_prim
from pennylane.capture.primitives import transform_prim
from pennylane.transforms import decompose as pl_decompose

from catalyst.backline import device_pass_pipeline, remote_device_lib
from catalyst.decomposition.capture_session import DecompositionScope
from catalyst.device import extract_backend_info
from catalyst.device.qjit_device import is_dynamic_wires
from catalyst.from_plxpr.qref_jax_primitives import (
    qref_alloc_p,
    qref_dealloc_p,
)
from catalyst.jax_extras import deduce_avals, make_jaxpr2, transient_jax_config
from catalyst.jax_extras.patches import get_jax_patches
from catalyst.jax_primitives import (
    device_init_p,
    device_release_p,
    quantum_kernel_p,
)
from catalyst.utils.patching import Patcher

from .device_utils import create_device_preprocessing_pipeline
from .qfunc_interpreter import (
    PLxPRToQuantumJaxprInterpreter,
    capture_and_bind_kernel_rules,
)

# dummy hop (higher order primitive) is used to just return a jaxpr
# produced inside of a another jaxpr
# we want to have the same tracers as inputs to plxpr capture and from_plxpr
# translation, as this tells jax which inputs match which dynamic shapes
# if we have concrete inputs to both, jax will get confused.
_dummy_hop = jax.extend.core.Primitive("dummy_hop")
_dummy_hop.multiple_results = True


# pylint: disable=unused-argument
@_dummy_hop.def_abstract_eval
def _dummy_abstract_eval(jaxpr, **kwargs):
    return jaxpr.out_avals


def _tuple_to_slice(t):
    """Convert a tuple representation of a slice back to a slice object.

    JAX converts slice objects to tuples for hashability in jaxpr parameters.
    This function converts them back to slice objects for use with indexing.

    Args:
        t: Either a slice object (returned as-is) or a tuple (start, stop, step)

    Returns:
        slice: A slice object
    """
    assert (
        isinstance(t, tuple) and len(t) == 3
    ), "Please only use _tuple_to_slice on a tuple of length 3!"
    return slice(*t)


def _is_dict_like_tuple(t):
    """Checks if a tuple t is structured like a list of (key, value) pairs."""
    return isinstance(t, tuple) and all(isinstance(item, tuple) and len(item) == 2 for item in t)


def _tuple_to_dict(t):
    """
    Recursively converts JAX-hashable tuple representations back to dicts,
    and list-like tuples back to lists.

    Args:
        t: The item to convert. Can be a dict, a tuple, or a scalar.

    Returns:
        The converted dict, list, or the original scalar value.
    """

    if not isinstance(t, (dict, tuple, list)):
        return t

    if isinstance(t, dict):  # pragma: no cover
        return {k: _tuple_to_dict(v) for k, v in t.items()}

    if isinstance(t, list):  # pragma: no cover
        return [_tuple_to_dict(item) for item in t]

    if isinstance(t, tuple):

        # A. Dict-like tuple: Convert to dict, then recurse on values
        if _is_dict_like_tuple(t):
            # This handles the main (key, value) pair structure
            return {key: _tuple_to_dict(value) for key, value in t}

        # B. List-like tuple: Convert to list, then recurse on elements
        else:
            return [_tuple_to_dict(item) for item in t]

    return t  # pragma: no cover


def _get_device_kwargs(device) -> dict:
    """Calulcate the params for a device equation."""
    info = extract_backend_info(device)
    # Note that the value of rtd_kwargs is a string version of
    # the info kwargs, not the info kwargs itself
    # this is due to ease of serialization to MLIR
    # A dispatched controller loads its runtime from its workspace, so the program names it by
    # filename. info.lpath describes this machine and is right for a controller running here.
    return {
        "rtd_kwargs": str(info.kwargs),
        "rtd_lib": remote_device_lib(device, info.lpath) or info.lpath,
        "rtd_name": info.c_interface_name,
    }


# code example has long lines
# pylint: disable=line-too-long
def from_plxpr(
    plxpr: ClosedJaxpr,
    skip_preprocess: bool = False,
    _preprocess_warn: bool = True,
    collect_decomp_rules: bool = True,
) -> Callable[..., Jaxpr]:
    """Convert PennyLane variant jaxpr to Catalyst variant jaxpr.

    Args:
        jaxpr (ClosedJaxpr): PennyLane variant jaxpr
        skip_preprocess (bool): Controls whether or not to skip quantum device preprocessing.
            If ``True``, transforms used to preprocess and validate the user program before
            executing on a quantum backend will not be used. ``False`` by default.
        _preprocess_warn (bool): Private argument to control whether a warning should be raised
            if any device preprocessing transforms in the compilation pipeline do not have
            native MLIR implementations. This argument is targeted at developers and should
            generally not be used. ``True`` by default.
        collect_decomp_rules (bool): Controls whether or not to compile the reachable
            decomposition rules from the gates in the circuit. ``True`` by default.

    Returns:
        Callable: A function that accepts the same arguments as the plxpr and returns catalyst
        variant jaxpr.

    Note that the input jaxpr should be workflow level and contain qnode primitives, rather than
    qfunc level with individual operators.

    .. code-block:: python

        from catalyst.from_plxpr import from_plxpr

        qp.capture.enable()

        @qp.qnode(qp.device('lightning.qubit', wires=2))
        def circuit(x):
            qp.RX(x, 0)
            return qp.probs(wires=(0, 1))

        def f(x):
            return circuit(2 * x) ** 2

        plxpr = jax.make_jaxpr(circuit)(0.5)

        print(from_plxpr(plxpr)(0.5))

    .. code-block:: none

        { lambda ; a:f64[]. let
            b:f64[4] = func[
            call_jaxpr={ lambda ; c:f64[]. let
                device_init[
                    rtd_kwargs={'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}
                    rtd_lib=***
                    rtd_name=LightningSimulator
                ]
                d:AbstractQreg() = qalloc 2
                e:AbstractQbit() = qextract d 0
                f:AbstractQbit() = qinst[
                    adjoint=False
                    ctrl_len=0
                    op=RX
                    params_len=1
                    qubits_len=1
                ] e c
                g:AbstractQbit() = qextract d 1
                h:AbstractObs(num_qubits=2,primitive=compbasis) = compbasis f g
                i:f64[4] = probs[shape=(4,) shots=None] h
                j:AbstractQreg() = qinsert d 0 f
                qdealloc j
                in (i,) }
            qnode=<QNode: device='<lightning.qubit device (wires=2) at 0x302761c90>', interface='auto', diff_method='best'>
            ] a
        in (b,) }

    """

    interpreter = WorkflowInterpreter(
        skip_preprocess=skip_preprocess,
        _preprocess_warn=_preprocess_warn,
        collect_decomp_rules=collect_decomp_rules,
    )
    original_fn = partial(interpreter.eval, plxpr.jaxpr, plxpr.consts)

    def wrapped_fn(*args, **kwargs):
        with Patcher(*get_jax_patches()):
            # needs a repeat of the patches in case from_plxpr used independently
            # outside of trace_From_pennylane
            return jax.make_jaxpr(original_fn)(*args, **kwargs)

    return wrapped_fn


class WorkflowInterpreter(PlxprInterpreter):
    """An interpreter that converts a qnode primitive from a plxpr variant to a catalyst jaxpr variant."""

    def __copy__(self):
        new_version = WorkflowInterpreter(
            skip_preprocess=self._skip_preprocess,
            _preprocess_warn=self._preprocess_warn,
            collect_decomp_rules=self._collect_decomp_rules,
        )
        new_version._pass_pipeline = copy(self._pass_pipeline)
        new_version.init_qreg = self.init_qreg
        new_version.requires_decompose_lowering = self.requires_decompose_lowering
        return new_version

    def __init__(self, skip_preprocess=False, _preprocess_warn=True, collect_decomp_rules=True):
        self._pass_pipeline = []
        self.init_qreg = None
        self._skip_preprocess = skip_preprocess
        self._preprocess_warn = _preprocess_warn
        self._collect_decomp_rules = collect_decomp_rules

        # Guards against applying more than one decomposition transform (not yet supported).
        self.requires_decompose_lowering = False

        super().__init__()


# pylint: disable=unused-argument, too-many-arguments
@WorkflowInterpreter.register_primitive(qnode_prim)
def handle_qnode(
    self, *args, qnode, device, shots_len, execution_config, qfunc_jaxpr, n_consts, batch_dims=None
):
    """Handle the conversion from plxpr to Catalyst jaxpr for the qnode primitive"""
    if shots_len > 1:
        raise NotImplementedError("shot vectors are not yet supported for catalyst conversion.")

    shots = args[0] if shots_len else 0
    consts = args[shots_len : n_consts + shots_len]
    non_const_args = args[shots_len + n_consts :]

    closed_jaxpr = ClosedJaxpr(qfunc_jaxpr, consts)

    decomposition_scope = DecompositionScope() if self._collect_decomp_rules else None

    def calling_convention(*args):
        device_init_p.bind(
            shots,
            auto_qubit_management=(device.wires is None),
            **_get_device_kwargs(device),
        )

        # https://github.com/PennyLaneAI/pennylane/pull/9248
        assert not is_dynamic_wires(
            device.wires
        ), "plxpr does not support dynamic number of wires on the device yet"
        qreg = qref_alloc_p.bind(static_num_qubits=len(device.wires))
        self.init_qreg = qreg

        converter = PLxPRToQuantumJaxprInterpreter(
            device,
            shots,
            self.init_qreg,
            {},
            decomposition_scope=decomposition_scope,
        )
        retvals = converter(closed_jaxpr, *args)
        # Discover and inject all relevant decomposition rules for this QNode as module functions.
        capture_and_bind_kernel_rules(converter)
        qref_dealloc_p.bind(self.init_qreg)
        device_release_p.bind()
        return retvals

    # The device may require passes of its own, e.g. a backline placement naming a QEC code implies
    # implicit encoding applied to it. Therefore we append the pass pipeline with the qec lowering
    # passes.
    pipelines = (("main", tuple(self._pass_pipeline) + device_pass_pipeline(qnode.device)),)
    if not self._skip_preprocess:
        device_preprocessing_pipeline = create_device_preprocessing_pipeline(
            qnode.device, execution_config, shots, warn=self._preprocess_warn
        )
        pipelines += (("device", device_preprocessing_pipeline),)

    # no idea what deduce_avals is doing, but this seems to make dynamic shapes work
    flattened_fn = deduce_avals(
        calling_convention, non_const_args, {}, [], debug_info=qfunc_jaxpr.debug_info
    )[0]

    return quantum_kernel_p.bind(
        flattened_fn,
        *non_const_args,
        qnode=qnode,
        pipelines=pipelines,
    )


def _guard_single_decompose(self):
    """Raise if a second decomposition transform is applied (not yet supported)."""
    if not self.requires_decompose_lowering:
        self.requires_decompose_lowering = True
    else:
        raise NotImplementedError("Multiple decomposition transforms are not yet supported.")


# qp.decompose arguments the graph-decomposition pass supports (mapped 1:1). ``tkwargs`` holds only
# the arguments the user passed, so any other key means an unsupported argument was set.
_SUPPORTED_DECOMPOSE_TKWARGS = frozenset({"gate_set", "fixed_decomps", "alt_decomps"})


def _validate_decompose_tkwargs(tkwargs):
    """Reject qp.decompose arguments the graph-decomposition pass cannot honor.
    """
    unsupported = sorted(k for k in tkwargs if k not in _SUPPORTED_DECOMPOSE_TKWARGS)
    if unsupported:
        raise NotImplementedError(
            f"qp.decompose argument(s) {unsupported} are not supported under qjit with graph-based "
            "decomposition. Supported arguments are: 'gate_set', 'fixed_decomps', 'alt_decomps'."
        )
    if tkwargs.get("gate_set") is None:
        raise ValueError(
            "qp.decompose requires an explicit 'gate_set' under qjit; the graph-based decomposition "
            "has no default target gate set."
        )


# pylint: disable=too-many-positional-arguments
def _handle_decompose_transform(self, inner_jaxpr, consts, non_const_args, tkwargs):
    """Route a captured ``qp.decompose`` onto the ``graph-decomposition`` pass.

    ``qp.decompose`` is an alias for :func:`catalyst.passes.graph_decomposition`: both build the same
    pass options and run the same C++ ``graph-decomposition`` pass, which solves and lowers the
    decomposition. Inline ``fixed_decomps``/``alt_decomps`` rule bodies are registered into a local
    decomposition scope so the trace-time rule-collection closure can capture them (the pass options
    only carry rule *names*; the bodies come from PennyLane's decomposition registry).
    """
    # Local imports avoid an import cycle (catalyst.passes imports from_plxpr indirectly).
    from pennylane.decomposition import add_decomps, enabled_graph, local_decomps

    from catalyst.passes.builtin_passes import graph_decomposition

    _guard_single_decompose(self)
    _validate_decompose_tkwargs(tkwargs)

    if not enabled_graph():
        raise RuntimeError(
            "qp.decompose under qjit requires graph-based decomposition. Call "
            "qml.decomposition.enable_graph() before compiling."
        )

    fixed_decomps = tkwargs.get("fixed_decomps") or {}
    alt_decomps = tkwargs.get("alt_decomps") or {}

    # The pass receives inline rules by name only; the bodies must be discoverable through the
    # trace-time rule closure, which reads the decomposition registry. That closure runs during the
    # eval below (inside the local_decomps scope). The compile-time on-demand loader runs later,
    # after the scope exits, so inline rules require rule collection to be enabled.
    if (fixed_decomps or alt_decomps) and not self._collect_decomp_rules:
        raise NotImplementedError(
            "Inline fixed_decomps/alt_decomps with qp.decompose require rule collection "
            "(collect_decomp_rules=True)."
        )

    next_eval = copy(self)
    with local_decomps():
        for op, rule in fixed_decomps.items():
            add_decomps(op, rule)
        for op, rules in alt_decomps.items():
            add_decomps(op, *rules)

        # Reusing the graph_decomposition transform (not a fork) guarantees identical pass options
        # to the explicit catalyst.passes.graph_decomposition entry point.
        bound_pass = graph_decomposition(
            gate_set=tkwargs["gate_set"],
            fixed_decomps=fixed_decomps or None,
            alt_decomps=alt_decomps or None,
        )
        next_eval._pass_pipeline.insert(0, bound_pass)

        return next_eval.eval(inner_jaxpr, consts, *non_const_args)


# pylint: disable=too-many-arguments
@WorkflowInterpreter.register_primitive(transform_prim)
def handle_transform(
    self,
    *args,
    args_slice,
    consts_slice,
    inner_jaxpr,
    targs_slice,
    tkwargs,
    transform,
):
    """Handle the conversion from plxpr to Catalyst jaxpr for a
    PL transform."""
    consts = args[_tuple_to_slice(consts_slice)]
    non_const_args = args[_tuple_to_slice(args_slice)]
    targs = args[_tuple_to_slice(targs_slice)]
    pl_tkwargs = _tuple_to_dict(tkwargs)

    # If the transform is a decomposition transform
    # and the graph-based decomposition is enabled
    if transform == pl_decompose:
        return _handle_decompose_transform(self, inner_jaxpr, consts, non_const_args, pl_tkwargs)

    if transform.pass_name is None:
        raise ValueError(
            f"{transform} does not have a pass_name and is not supported with the "
            "capture frontend. Set capture=False to apply tape-only transforms with qjit."
        )

    # Apply the corresponding Catalyst pass counterpart
    next_eval = copy(self)
    t = qp.transform(pass_name=transform.pass_name)
    bound_pass = qp.transforms.core.BoundTransform(t, args=targs, kwargs=pl_tkwargs)
    next_eval._pass_pipeline.insert(0, bound_pass)
    return next_eval.eval(inner_jaxpr, consts, *non_const_args)


def _extract_abstract_shapes(flat_inputs):
    abstract_shapes = []
    for a in flat_inputs:
        for s in a.shape:
            # need to us "is" for comparing tracers
            if not isinstance(s, int) and not any(s is _a for _a in abstract_shapes):
                abstract_shapes.append(s)
    return abstract_shapes


# pylint: disable=too-many-positional-arguments
def trace_from_pennylane(
    fn,
    args,
    kwargs,
    static_argnums,
    abstracted_axes,
    skip_preprocess=False,
    collect_decomp_rules=True,
    debug_info=None,
):
    """Capture the JAX program representation (JAXPR) of the wrapped function, using
    PL capure module.

    Args:
        fn(Callable): the user function to be traced
        args (tuple): the positional arguments to the user functions
        kwargs(Dict[str, Any]): keyword arguments to the function.
        static_argnums(int or Seqence[Int]): an index or a sequence of indices that specifies the
            positions of static arguments.
        abstracted_axes (Sequence[Sequence[str]] or Dict[int, str] or Sequence[Dict[int, str]]):
            An experimental option to specify dynamic tensor shapes.
            This option affects the compilation of the annotated function.
            Function arguments with ``abstracted_axes`` specified will be compiled to ranked tensors
            with dynamic shapes. For more details, please see the Dynamically-shaped Arrays section
            below.
        skip_preprocess (bool): Controls whether or not to skip quantum device preprocessing.
            If ``True``, transforms used to preprocess and validate the user program before
            executing on a quantum backend will not be used. ``False`` by default.
        collect_decomp_rules (bool): Controls whether or not to compile the reachable
            decomposition rules from the gates in the circuit. ``True`` by default.
        debug_info(jax.api_util.debug_info): a source debug information object required by jaxprs.

    Returns:
        ClosedJaxpr: captured JAXPR
        Tuple[Tuple[ShapedArray, bool]]: the return type of the captured JAXPR.
            The boolean indicates whether each result is a value returned by the user function.
        PyTreeDef: PyTree metadata of the function output
    """
    if abstracted_axes and any(isinstance(arg, jax.core.ShapedArray) for arg in args):
        # ShapedArrays incompatible with abstracted_axes, so need to create dummy arrays
        args = [jax.numpy.empty(arg.shape, dtype=arg.dtype) for arg in args]

    if isinstance(fn, qp.QNode) and static_argnums:
        # `make_jaxpr2` sees the qnode
        # The static_argnum on the wrapped function takes precedence over the
        # one in `make_jaxpr`
        # https://github.com/jax-ml/jax/blob/636691bba40b936b8b64a4792c1d2158296e9dd4/jax/_src/linear_util.py#L231
        # Therefore we need to coordinate them manually
        fn.static_argnums = static_argnums

    with transient_jax_config(
        {"jax_dynamic_shapes": True, "jax_use_shardy_partitioner": False}
    ), Patcher(*get_jax_patches()):

        make_jaxpr_kwargs = {
            "static_argnums": static_argnums,
            "abstracted_axes": abstracted_axes,
        }

        # we want to have the same tracers as inputs to plxpr capture and from_plxpr
        # translation, as this tells jax which inputs match which dynamic shapes
        # if we have concrete inputs to both, jax will get confused.
        # instead of passing in abstracted_axes, we pass in arguments with the
        # dynamic shapes in the right place matching the correct inputs.
        # really confusing, but this solution mostly seems to work
        def wrapper(*inner_args, **inner_kwargs):
            plxpr, out_type, out_treedef = make_jaxpr2(
                fn, static_argnums=static_argnums, debug_info=debug_info
            )(*inner_args, **inner_kwargs)

            flat_inputs = jax.tree.flatten((inner_args, inner_kwargs))[0]
            flat_inputs = [a for a in flat_inputs if qp.math.is_abstract(a)]
            abstract_shapes = _extract_abstract_shapes(flat_inputs)
            jaxpr = from_plxpr(
                plxpr, skip_preprocess=skip_preprocess, collect_decomp_rules=collect_decomp_rules
            )(*abstract_shapes, *flat_inputs)

            return _dummy_hop.bind(jaxpr=jaxpr, out_type=out_type, out_treedef=out_treedef)

        nested_jaxpr = jax.make_jaxpr(wrapper, **make_jaxpr_kwargs)(*args, **kwargs)
        jaxpr = nested_jaxpr.eqns[0].params["jaxpr"]
        out_type = nested_jaxpr.eqns[0].params["out_type"]
        out_treedef = nested_jaxpr.eqns[0].params["out_treedef"]

    return jaxpr, out_type, out_treedef
