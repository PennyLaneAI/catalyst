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
from functools import partial
from typing import Callable

import jax
import pennylane as qml
from jax.extend.core import ClosedJaxpr, Jaxpr
from jax.extend.linear_util import wrap_init
from pennylane.capture import PlxprInterpreter, qnode_prim
from pennylane.capture.expand_transforms import ExpandTransformsInterpreter
from pennylane.ops.functions.map_wires import _map_wires_transform as pl_map_wires
from pennylane.transforms import cancel_inverses as pl_cancel_inverses
from pennylane.transforms import clifford_t_decomposition as pl_clifford_t_decomposition
from pennylane.transforms import commute_controlled as pl_commute_controlled
from pennylane.transforms import decompose as pl_decompose
from pennylane.transforms import merge_amplitude_embedding as pl_merge_amplitude_embedding
from pennylane.transforms import merge_rotations as pl_merge_rotations
from pennylane.transforms import single_qubit_fusion as pl_single_qubit_fusion
from pennylane.transforms import unitary_to_rot as pl_unitary_to_rot

from catalyst.device import extract_backend_info
from catalyst.from_plxpr.decompose import COMPILER_OPS_FOR_DECOMPOSITION, DecompRuleInterpreter
from catalyst.jax_extras import make_jaxpr2, transient_jax_config
from catalyst.jax_primitives import (
    device_init_p,
    device_release_p,
    qalloc_p,
    qdealloc_p,
    quantum_kernel_p,
)
from catalyst.passes.pass_api import Pass

from .qfunc_interpreter import PLxPRToQuantumJaxprInterpreter
from .qubit_handler import (
    QubitHandler,
    QubitIndexRecorder,
)


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
def from_plxpr(plxpr: ClosedJaxpr) -> Callable[..., Jaxpr]:
    """Convert PennyLane variant jaxpr to Catalyst variant jaxpr.

    Args:
        jaxpr (ClosedJaxpr): PennyLane variant jaxpr

    Returns:
        Callable: A function that accepts the same arguments as the plxpr and returns catalyst
        variant jaxpr.

    Note that the input jaxpr should be workflow level and contain qnode primitives, rather than
    qfunc level with individual operators.

    .. code-block:: python

        from catalyst.from_plxpr import from_plxpr

        qml.capture.enable()

        @qml.qnode(qml.device('lightning.qubit', wires=2))
        def circuit(x):
            qml.RX(x, 0)
            return qml.probs(wires=(0, 1))

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
    return jax.make_jaxpr(partial(WorkflowInterpreter().eval, plxpr.jaxpr, plxpr.consts))


class WorkflowInterpreter(PlxprInterpreter):
    """An interpreter that converts a qnode primitive from a plxpr variant to a catalyst jaxpr variant."""

    def __init__(self):
        self._pass_pipeline = []
        self.init_qreg = None

        # Compiler options for the new decomposition system
        self.requires_decompose_lowering = False
        self.decompose_tkwargs = {}  # target gateset

        super().__init__()


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
    if self.requires_decompose_lowering:
        closed_jaxpr, graph_succeeded = _collect_and_compile_graph_solutions(
            inner_jaxpr=closed_jaxpr.jaxpr,
            consts=closed_jaxpr.consts,
            tkwargs=self.decompose_tkwargs,
            ncargs=non_const_args,
        )

        # Fallback to the legacy decomposition if the graph-based decomposition failed
        if not graph_succeeded:
            # Remove the decompose-lowering pass from the pipeline
            self._pass_pipeline = [p for p in self._pass_pipeline if p.name != "decompose-lowering"]
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

    return quantum_kernel_p.bind(
        wrap_init(calling_convention, debug_info=qfunc_jaxpr.debug_info),
        *non_const_args,
        qnode=qnode,
        pipeline=self._pass_pipeline,
    )


# The map below describes the parity between PL transforms and Catalyst passes.
# PL transforms having a Catalyst pass counterpart will have a name as value,
# otherwise their value will be None. The second value indicates if the transform
# requires decomposition to be supported by Catalyst.
transforms_to_passes = {
    pl_cancel_inverses: ("remove-chained-self-inverse", False),
    pl_commute_controlled: (None, False),
    pl_decompose: (None, False),
    pl_map_wires: (None, False),
    pl_merge_amplitude_embedding: (None, True),
    pl_merge_rotations: ("merge-rotations", False),
    pl_single_qubit_fusion: (None, False),
    pl_unitary_to_rot: (None, False),
    pl_clifford_t_decomposition: ("rs-decomposition", False),
}


# pylint: disable-next=redefined-outer-name
def register_transform(pl_transform, pass_name, decomposition):
    """Register pennylane transforms and their conversion to Catalyst transforms"""

    # pylint: disable=too-many-arguments
    @WorkflowInterpreter.register_primitive(pl_transform._primitive)
    def handle_transform(
        self,
        *args,
        args_slice,
        consts_slice,
        inner_jaxpr,
        targs_slice,
        tkwargs,
        catalyst_pass_name=pass_name,
        requires_decomposition=decomposition,
        pl_plxpr_transform=pl_transform._plxpr_transform,
    ):
        """Handle the conversion from plxpr to Catalyst jaxpr for a
        PL transform."""
        consts = args[consts_slice]
        non_const_args = args[args_slice]
        targs = args[targs_slice]

        # If the transform is a decomposition transform
        # and the graph-based decomposition is enabled
        if (
            hasattr(pl_plxpr_transform, "__name__")
            and pl_plxpr_transform.__name__ == "decompose_plxpr_to_plxpr"
            and qml.decomposition.enabled_graph()
        ):
            if not self.requires_decompose_lowering:
                self.requires_decompose_lowering = True
            else:
                raise NotImplementedError(
                    "Multiple decomposition transforms are not yet supported."
                )

            # Update the decompose_gateset to be used by the quantum kernel primitive
            # TODO: we originally wanted to treat decompose_gateset as a queue of
            # gatesets to be used by the decompose-lowering pass at MLIR
            # but this requires a C++ implementation of the graph-based decomposition
            # which doesn't exist yet.
            self.decompose_tkwargs = tkwargs

            # Note. We don't perform the compiler-specific decomposition here
            # to be able to support multiple decomposition transforms
            # and collect all the required gatesets
            # as well as being able to support other transforms in between.

            # The compiler specific transformation will be performed
            # in the qnode handler.

            # Add the decompose-lowering pass to the start of the pipeline
            self._pass_pipeline.insert(0, Pass("decompose-lowering"))

            # We still need to construct and solve the graph based on
            # the current jaxpr based on the current gateset
            # but we don't rewrite the jaxpr at this stage.

            # gds_interpreter = DecompRuleInterpreter(*targs, **tkwargs)

            # def gds_wrapper(*args):
            #     return gds_interpreter.eval(inner_jaxpr, consts, *args)

            # final_jaxpr = jax.make_jaxpr(gds_wrapper)(*args)
            # return self.eval(final_jaxpr.jaxpr, consts, *non_const_args)
            return self.eval(inner_jaxpr, consts, *non_const_args)

        if catalyst_pass_name is None:
            # Use PL's ExpandTransformsInterpreter to expand this and any embedded
            # transform according to PL rules. It works by overriding the primitive
            # registration, making all embedded transforms follow the PL rules
            # from now on, hence ignoring the Catalyst pass conversion
            def wrapper(*args):
                return ExpandTransformsInterpreter().eval(inner_jaxpr, consts, *args)

            unravelled_jaxpr = jax.make_jaxpr(wrapper)(*non_const_args)
            final_jaxpr = pl_plxpr_transform(
                unravelled_jaxpr.jaxpr, unravelled_jaxpr.consts, targs, tkwargs, *non_const_args
            )

            if requires_decomposition:
                final_jaxpr = pl_decompose._plxpr_transform(
                    final_jaxpr.jaxpr, final_jaxpr.consts, targs, tkwargs, *non_const_args
                )

            return self.eval(final_jaxpr.jaxpr, final_jaxpr.consts, *non_const_args)

        # Apply the corresponding Catalyst pass counterpart
        self._pass_pipeline.insert(0, Pass(catalyst_pass_name))
        return self.eval(inner_jaxpr, consts, *non_const_args)


# This is our registration factory for PL transforms. The loop below iterates
# across the map above and generates a custom handler for each transform.
# In order to ensure early binding, we pass the PL plxpr transform and the
# Catalyst pass as arguments whose default values are set by the loop.
for pl_transform, (pass_name, decomposition) in transforms_to_passes.items():
    register_transform(pl_transform, pass_name, decomposition)


# pylint: disable=too-many-positional-arguments
def trace_from_pennylane(
    fn, static_argnums, dynamic_args, abstracted_axes, sig, kwargs, debug_info=None
):
    """Capture the JAX program representation (JAXPR) of the wrapped function, using
    PL capure module.

    Args:
        fn(Callable): the user function to be traced
        static_argnums(int or Seqence[Int]): an index or a sequence of indices that specifies the
            positions of static arguments.
        dynamic_args(Seqence[Any]): the abstract values of the dynamic arguments.
        abstracted_axes (Sequence[Sequence[str]] or Dict[int, str] or Sequence[Dict[int, str]]):
            An experimental option to specify dynamic tensor shapes.
            This option affects the compilation of the annotated function.
            Function arguments with ``abstracted_axes`` specified will be compiled to ranked tensors
            with dynamic shapes. For more details, please see the Dynamically-shaped Arrays section
            below.
        sig(Sequence[Any]): a tuple indicating the argument signature of the function. Static arguments
            are indicated with their literal values, and dynamic arguments are indicated by abstract
            values.
        kwargs(Dict[str, Any]): keyword argumemts to the function.
        debug_info(jax.api_util.debug_info): a source debug information object required by jaxprs.

    Returns:
        ClosedJaxpr: captured JAXPR
        Tuple[Tuple[ShapedArray, bool]]: the return type of the captured JAXPR.
            The boolean indicates whether each result is a value returned by the user function.
        PyTreeDef: PyTree metadata of the function output
        Tuple[Any]: the dynamic argument signature
    """

    with transient_jax_config({"jax_dynamic_shapes": True}):

        make_jaxpr_kwargs = {
            "static_argnums": static_argnums,
            "abstracted_axes": abstracted_axes,
            "debug_info": debug_info,
        }

        args = sig

        if isinstance(fn, qml.QNode) and static_argnums:
            # `make_jaxpr2` sees the qnode
            # The static_argnum on the wrapped function takes precedence over the
            # one in `make_jaxpr`
            # https://github.com/jax-ml/jax/blob/636691bba40b936b8b64a4792c1d2158296e9dd4/jax/_src/linear_util.py#L231
            # Therefore we need to coordinate them manually
            fn.static_argnums = static_argnums

        plxpr, out_type, out_treedef = make_jaxpr2(fn, **make_jaxpr_kwargs)(*args, **kwargs)
        jaxpr = from_plxpr(plxpr)(*dynamic_args, **kwargs)

    return jaxpr, out_type, out_treedef, sig


def _apply_compiler_decompose_to_plxpr(inner_jaxpr, consts, ncargs, tgateset=None, tkwargs=None):
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
    qml.decomposition.disable_graph()

    kwargs = (
        {"gate_set": set(COMPILER_OPS_FOR_DECOMPOSITION.keys()).union(tgateset)}
        if tgateset
        else tkwargs
    )
    final_jaxpr = qml.transforms.decompose.plxpr_transform(inner_jaxpr, consts, (), kwargs, *ncargs)

    qml.decomposition.enable_graph()

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
