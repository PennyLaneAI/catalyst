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

import warnings
from copy import copy
from functools import partial
from typing import Callable

import jax
import pennylane as qp
from jax.extend.core import ClosedJaxpr, Jaxpr
from pennylane.capture import PlxprInterpreter, qnode_prim
from pennylane.capture.expand_transforms import ExpandTransformsInterpreter
from pennylane.capture.primitives import transform_prim
from pennylane.transforms import commute_controlled as pl_commute_controlled
from pennylane.transforms import decompose as pl_decompose
from pennylane.transforms import gridsynth as pl_gridsynth
from pennylane.transforms import merge_amplitude_embedding as pl_merge_amplitude_embedding
from pennylane.transforms import single_qubit_fusion as pl_single_qubit_fusion
from pennylane.transforms import unitary_to_rot as pl_unitary_to_rot

from catalyst.device import extract_backend_info, get_device_capabilities
from catalyst.from_plxpr.decompose import COMPILER_OPS_FOR_DECOMPOSITION, DecompRuleInterpreter
from catalyst.jax_extras import deduce_avals, make_jaxpr2, transient_jax_config
from catalyst.jax_extras.patches import get_jax_patches
from catalyst.jax_primitives import (
    device_init_p,
    device_release_p,
    qalloc_p,
    qdealloc_p,
    quantum_kernel_p,
)
from catalyst.utils.patching import Patcher

from .device_utils import create_device_preprocessing_pipeline
from .qfunc_interpreter import PLxPRToQuantumJaxprInterpreter
from .qubit_handler import (
    QubitHandler,
    QubitIndexRecorder,
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
    return {
        "rtd_kwargs": str(info.kwargs),
        "rtd_lib": info.lpath,
        "rtd_name": info.c_interface_name,
    }


# code example has long lines
# pylint: disable=line-too-long
def from_plxpr(
    plxpr: ClosedJaxpr, skip_preprocess: bool = False, _preprocess_warn: bool = True
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
        skip_preprocess=skip_preprocess, _preprocess_warn=_preprocess_warn
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
            skip_preprocess=self._skip_preprocess, _preprocess_warn=self._preprocess_warn
        )
        new_version._pass_pipeline = copy(self._pass_pipeline)
        new_version.init_qreg = self.init_qreg
        new_version.requires_decompose_lowering = self.requires_decompose_lowering
        new_version.decompose_tkwargs = copy(self.decompose_tkwargs)
        return new_version

    def __init__(self, skip_preprocess=False, _preprocess_warn=True):
        self._pass_pipeline = []
        self.init_qreg = None
        self._skip_preprocess = skip_preprocess
        self._preprocess_warn = _preprocess_warn

        # Compiler options for the new decomposition system
        self.requires_decompose_lowering = False
        self.decompose_tkwargs = {}  # target gateset

        super().__init__()


def _decompose_jaxpr_to_gateset(qfunc_jaxpr, consts, device):
    gate_set = set(get_device_capabilities(device).operations)
    if get_device_capabilities(device).initial_state_prep:
        gate_set.add("StatePrep")
    targs = ()
    tkwargs = {"gate_set": gate_set}
    breakpoint()
    return qml.transforms.decompose.plxpr_transform(qfunc_jaxpr, consts, targs, tkwargs)


# pylint: disable=unused-argument, too-many-arguments
@WorkflowInterpreter.register_primitive(qnode_prim)
def handle_qnode(
    self, *args, qnode, device, shots_len, execution_config, qfunc_jaxpr, n_consts, batch_dims=None
):
    """Handle the conversion from plxpr to Catalyst jaxpr for the qnode primitive"""

    self.qubit_index_recorder = QubitIndexRecorder()

    if shots_len > 1:
        raise NotImplementedError("shot vectors are not yet supported for catalyst conversion.")

    shots = args[0] if shots_len else 0
    consts = args[shots_len : n_consts + shots_len]
    non_const_args = args[shots_len + n_consts :]

    # hopefully this patch stays patchy and doesn't become permanent
    # TODO: Too much has changed within this function, need to rework the patch
    closed_jaxpr = _decompose_jaxpr_to_gateset(qfunc_jaxpr, consts, device)

    closed_jaxpr = (
        ClosedJaxpr(qfunc_jaxpr, consts)
        if not self.requires_decompose_lowering
        else _apply_compiler_decompose_to_plxpr(
            inner_jaxpr=qfunc_jaxpr,
            consts=consts,
            ncargs=non_const_args,
            tgateset=list(self.decompose_tkwargs.get("gate_set", [])),
        )
    )

    graph_succeeded = False
    if stopping_condition := self.decompose_tkwargs.get("stopping_condition"):
        # Use the plxpr decompose transform and ignore graph decomposition
        # See https://github.com/PennyLaneAI/catalyst/pull/2472.
        closed_jaxpr = _apply_compiler_decompose_to_plxpr(
            inner_jaxpr=qfunc_jaxpr,
            consts=consts,
            ncargs=non_const_args,
            tkwargs={"gate_set": self.decompose_tkwargs.get("gate_set", [])},
            stopping_condition=stopping_condition,
        )
    elif not qp.decomposition.enabled_graph() and self.requires_decompose_lowering:
        # Use the plxpr decompose transform when graph is disabled
        closed_jaxpr = _apply_compiler_decompose_to_plxpr(
            inner_jaxpr=qfunc_jaxpr,
            consts=consts,
            ncargs=non_const_args,
            tkwargs={"gate_set": self.decompose_tkwargs.get("gate_set", [])},
        )
    elif qp.decomposition.enabled_graph() and self.requires_decompose_lowering:
        closed_jaxpr, graph_succeeded = _collect_and_compile_graph_solutions(
            inner_jaxpr=closed_jaxpr.jaxpr,
            consts=closed_jaxpr.consts,
            tkwargs=self.decompose_tkwargs,
            ncargs=non_const_args,
        )

        # Fallback to the legacy decomposition if the graph-based decomposition failed
        if not graph_succeeded:
            # Remove the decompose-lowering pass from the pipeline
            self._pass_pipeline = [
                p for p in self._pass_pipeline if p.pass_name != "decompose-lowering"
            ]
            closed_jaxpr = _apply_compiler_decompose_to_plxpr(
                inner_jaxpr=closed_jaxpr.jaxpr,
                consts=closed_jaxpr.consts,
                ncargs=non_const_args,
                tkwargs=self.decompose_tkwargs,
            )

    def calling_convention(*args):
        device_init_p.bind(
            shots,
            auto_qubit_management=(device.wires is None),
            **_get_device_kwargs(device),
        )
        qreg = qalloc_p.bind(len(device.wires))
        self.init_qreg = QubitHandler(qreg, self.qubit_index_recorder)
        converter = PLxPRToQuantumJaxprInterpreter(
            device, shots, self.init_qreg, {}, self.qubit_index_recorder
        )
        retvals = converter(closed_jaxpr, *args)
        self.init_qreg.insert_all_dangling_qubits()
        qdealloc_p.bind(self.init_qreg.get())
        device_release_p.bind()
        return retvals

    if self.requires_decompose_lowering and graph_succeeded:
        # Add gate_set attribute to the quantum kernel primitive
        # decompose_gatesets is treated as a queue of gatesets to be used
        # but we only support a single gateset for now in from_plxpr
        # as supporting multiple gatesets requires an MLIR/C++ graph-decomposition
        # implementation. The current Python implementation cannot be mixed
        # with other transforms in between.
        gateset = [_get_operator_name(op) for op in self.decompose_tkwargs.get("gate_set", [])]
        setattr(qnode, "decompose_gatesets", [gateset])
    pipelines = (("main", tuple(self._pass_pipeline)),)
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


# The map below describes the parity between PL transforms and Catalyst passes.
# PL transforms having a Catalyst pass counterpart will have a name as value,
# otherwise their value will be None. The second value indicates if the transform
# requires decomposition to be supported by Catalyst.
transforms_to_passes = {
    pl_commute_controlled: (None, False),
    pl_decompose: (None, False),
    pl_merge_amplitude_embedding: (None, True),
    pl_single_qubit_fusion: (None, False),
    pl_unitary_to_rot: (None, False),
    pl_gridsynth: ("gridsynth", False),
}


def _set_decompose_lowering_state(self):
    """Set requires_decompose_lowering and decompose_tkwargs; raise if already set."""
    if not self.requires_decompose_lowering:
        self.requires_decompose_lowering = True
    else:
        raise NotImplementedError("Multiple decomposition transforms are not yet supported.")


# pylint: disable=too-many-positional-arguments
def _handle_decompose_transform(self, inner_jaxpr, consts, non_const_args, tkwargs, use_graph=True):
    _set_decompose_lowering_state(self)

    next_eval = copy(self)
    # Update the decompose_gateset to be used by the quantum kernel primitive
    # TODO: we originally wanted to treat decompose_gateset as a queue of
    # gatesets to be used by the decompose-lowering pass at MLIR
    # but this requires a C++ implementation of the graph-based decomposition
    # which doesn't exist yet.
    next_eval.decompose_tkwargs = tkwargs

    # Note. We don't perform the compiler-specific decomposition here
    # to be able to support multiple decomposition transforms
    # and collect all the required gatesets
    # as well as being able to support other transforms in between.

    # The compiler specific transformation will be performed
    # in the qnode handler.

    # Add the decompose-lowering pass to the start of the pipeline
    if use_graph:
        t = qp.transform(pass_name="decompose-lowering")
        pass_container = qp.transforms.core.BoundTransform(t)
        next_eval._pass_pipeline.insert(0, pass_container)

    # We still need to construct and solve the graph based on
    # the current jaxpr based on the current gateset
    # but we don't rewrite the jaxpr at this stage.

    # gds_interpreter = DecompRuleInterpreter(*targs, **tkwargs)

    # def gds_wrapper(*args):
    #     return gds_interpreter.eval(inner_jaxpr, consts, *args)

    # final_jaxpr = jax.make_jaxpr(gds_wrapper)(*args)
    # return self.eval(final_jaxpr.jaxpr, consts, *non_const_args)
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
    transform_name = getattr(transform._plxpr_transform, "__name__", None)
    if transform_name == "decompose_plxpr_to_plxpr":
        use_graph = qp.decomposition.enabled_graph()
        return _handle_decompose_transform(
            self, inner_jaxpr, consts, non_const_args, pl_tkwargs, use_graph
        )

    catalyst_pass_name = transform.pass_name
    if catalyst_pass_name is None:
        catalyst_pass_name = transforms_to_passes.get(transform, (None,))[0]
    if catalyst_pass_name is None:
        # Use PL's ExpandTransformsInterpreter to expand this and any embedded
        # transform according to PL rules. It works by overriding the primitive
        # registration, making all embedded transforms follow the PL rules
        # from now on, hence ignoring the Catalyst pass conversion
        def wrapper(*args):
            return ExpandTransformsInterpreter().eval(inner_jaxpr, consts, *args)

        unravelled_jaxpr = jax.make_jaxpr(wrapper)(*non_const_args)
        final_jaxpr = transform._plxpr_transform(
            unravelled_jaxpr.jaxpr, unravelled_jaxpr.consts, targs, pl_tkwargs, *non_const_args
        )
        if transforms_to_passes[transform][1]:
            final_jaxpr = pl_decompose._plxpr_transform(
                final_jaxpr.jaxpr, final_jaxpr.consts, targs, pl_tkwargs, *non_const_args
            )

        return copy(self).eval(final_jaxpr.jaxpr, final_jaxpr.consts, *non_const_args)

    # Apply the corresponding Catalyst pass counterpart
    next_eval = copy(self)
    t = qp.transform(pass_name=catalyst_pass_name)
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
            jaxpr = from_plxpr(plxpr, skip_preprocess=skip_preprocess)(
                *abstract_shapes, *flat_inputs
            )

            return _dummy_hop.bind(jaxpr=jaxpr, out_type=out_type, out_treedef=out_treedef)

        nested_jaxpr = jax.make_jaxpr(wrapper, **make_jaxpr_kwargs)(*args, **kwargs)
        jaxpr = nested_jaxpr.eqns[0].params["jaxpr"]
        out_type = nested_jaxpr.eqns[0].params["out_type"]
        out_treedef = nested_jaxpr.eqns[0].params["out_treedef"]

    return jaxpr, out_type, out_treedef


def _apply_compiler_decompose_to_plxpr(
    inner_jaxpr, consts, ncargs, tgateset=None, tkwargs=None, stopping_condition=None
):
    """Apply the compiler-specific decomposition for a given JAXPR.

    This function first disables the graph-based decomposition optimization
    to ensure that only high-level gates and templates with a single decomposition
    are decomposed. It then performs the pre-mlir decomposition using PennyLane's
    `plxpr_transform` function.

    `tgateset` is a list of target gateset for decomposition.
    If provided, it will be combined with the default compiler ops for decomposition.
    If not provided, `tkwargs` will be used as the keyword arguments for the
    decomposition transform. This is to ensure compatibility with the existing
    PennyLane decomposition transform as well as providing a fallback mechanism.

    Args:
        inner_jaxpr (Jaxpr): The input JAXPR to be decomposed.
        consts (list): The constants used in the JAXPR.
        ncargs (list): Non-constant arguments for the JAXPR.
        tgateset (list): A list of target gateset for decomposition. Defaults to None.
        tkwargs (list): The keyword arguments of the decompose transform. Defaults to None.

    Returns:
        ClosedJaxpr: The decomposed JAXPR.
    """

    # Disable the graph decomposition optimization

    # Why? Because for the compiler-specific decomposition we want to
    # only decompose higher-level gates and templates that only have
    # a single decomposition, and not do any further optimization
    # based on the graph solution.
    # Besides, the graph-based decomposition is not supported
    # yet in from_plxpr for most gates and templates.
    # TODO: Enable the graph-based decomposition
    graph_enabled = qp.decomposition.enabled_graph()

    if graph_enabled:
        qp.decomposition.disable_graph()

    kwargs = (
        {"gate_set": set(COMPILER_OPS_FOR_DECOMPOSITION.keys()).union(tgateset)}
        if tgateset
        else tkwargs
    )

    if stopping_condition:
        kwargs["stopping_condition"] = stopping_condition

    final_jaxpr = qp.transforms.decompose.plxpr_transform(inner_jaxpr, consts, (), kwargs, *ncargs)

    if graph_enabled:
        qp.decomposition.enable_graph()

    return final_jaxpr


def _collect_and_compile_graph_solutions(inner_jaxpr, consts, tkwargs, ncargs):
    """Collect and compile graph solutions for a given JAXPR.

    This function uses the DecompRuleInterpreter to evaluate
    the input JAXPR and obtain a new JAXPR that incorporates
    the graph-based decomposition solutions.

    This function doesn't modify the underlying quantum function
    but rather constructs a new JAXPR with decomposition rules.

    Args:
        inner_jaxpr (Jaxpr): The input JAXPR to be decomposed.
        consts (list): The constants used in the JAXPR.
        tkwargs (list): The keyword arguments of the decompose transform.
        ncargs (list): Non-constant arguments for the JAXPR.

    Returns:
        ClosedJaxpr: The decomposed JAXPR.
        bool: A flag indicating whether the graph-based decomposition was successful.
    """
    gds_interpreter = DecompRuleInterpreter(**tkwargs)

    def gds_wrapper(*args):
        return gds_interpreter.eval(inner_jaxpr, consts, *args)

    graph_succeeded = True

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always", UserWarning)
        final_jaxpr = jax.make_jaxpr(gds_wrapper)(*ncargs)

    for w in captured_warnings:
        warnings.showwarning(w.message, w.category, w.filename, w.lineno)
        # TODO: use a custom warning class for this in PennyLane to remove this
        # string matching and make it more robust.
        if "The graph-based decomposition system is unable" in str(w.message):  # pragma: no cover
            graph_succeeded = False
            warnings.warn(
                "Falling back to the legacy decomposition system.",
                UserWarning,
            )

    return final_jaxpr, graph_succeeded


def _get_operator_name(op):
    """Get the name of a pennylane operator, handling wrapped operators.

    Note: Controlled and Adjoint ops aren't supported in `gate_set`
    by PennyLane's DecompositionGraph; unit tests were added in PennyLane.
    """
    if isinstance(op, str):
        return op

    # Return NoNameOp if the operator has no _primitive.name attribute.
    # This is to avoid errors when we capture the program
    # as we deal with such ops later in the decomposition graph.
    return getattr(op._primitive, "name", "NoNameOp")
