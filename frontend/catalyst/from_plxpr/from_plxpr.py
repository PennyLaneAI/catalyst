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
# pylint: disable=too-many-lines

import textwrap
import warnings
from copy import copy
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import pennylane as qml
from jax._src.sharding_impls import UNSPECIFIED
from jax._src.tree_util import tree_flatten
from jax.extend.core import ClosedJaxpr, Jaxpr
from jax.extend.linear_util import wrap_init
from jax.interpreters.partial_eval import convert_constvars_jaxpr
from pennylane.capture import PlxprInterpreter, pause, qnode_prim
from pennylane.capture.expand_transforms import ExpandTransformsInterpreter
from pennylane.capture.primitives import adjoint_transform_prim as plxpr_adjoint_transform_prim
from pennylane.capture.primitives import ctrl_transform_prim as plxpr_ctrl_transform_prim
from pennylane.capture.primitives import measure_prim as plxpr_measure_prim
from pennylane.ftqc.primitives import measure_in_basis_prim as plxpr_measure_in_basis_prim
from pennylane.measurements import CountsMP
from pennylane.ops.functions.map_wires import _map_wires_transform as pl_map_wires
from pennylane.transforms import cancel_inverses as pl_cancel_inverses
from pennylane.transforms import commute_controlled as pl_commute_controlled
from pennylane.transforms import decompose as pl_decompose
from pennylane.transforms import merge_amplitude_embedding as pl_merge_amplitude_embedding
from pennylane.transforms import merge_rotations as pl_merge_rotations
from pennylane.transforms import single_qubit_fusion as pl_single_qubit_fusion
from pennylane.transforms import unitary_to_rot as pl_unitary_to_rot

from catalyst.device import extract_backend_info
from catalyst.from_plxpr.decompose import COMPILER_OPS_FOR_DECOMPOSITION, DecompRuleInterpreter
from catalyst.from_plxpr.qubit_handler import (
    QubitHandler,
    QubitIndexRecorder,
    get_in_qubit_values,
    is_dynamically_allocated_wire,
)
from catalyst.jax_extras import jaxpr_pad_consts, make_jaxpr2, transient_jax_config
from catalyst.jax_primitives import (
    AbstractQbit,
    MeasurementPlane,
    adjoint_p,
    compbasis_p,
    cond_p,
    counts_p,
    decomprule_p,
    device_init_p,
    device_release_p,
    expval_p,
    gphase_p,
    hamiltonian_p,
    hermitian_p,
    measure_in_basis_p,
    measure_p,
    namedobs_p,
    probs_p,
    qalloc_p,
    qdealloc_p,
    qinst_p,
    quantum_kernel_p,
    quantum_subroutine_p,
    sample_p,
    set_basis_state_p,
    set_state_p,
    state_p,
    tensorobs_p,
    unitary_p,
    var_p,
)
from catalyst.passes.pass_api import Pass
from catalyst.utils.exceptions import CompileError

measurement_map = {
    qml.measurements.SampleMP: sample_p,
    qml.measurements.ExpectationMP: expval_p,
    qml.measurements.VarianceMP: var_p,
    qml.measurements.ProbabilityMP: probs_p,
    qml.measurements.StateMP: state_p,
}


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


def _flat_prod_gen(op: qml.ops.Prod):
    for o in op:
        if isinstance(o, qml.ops.Prod):
            yield from _flat_prod_gen(o)
        else:
            yield o


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


# pylint: disable=too-many-instance-attributes
class PLxPRToQuantumJaxprInterpreter(PlxprInterpreter):
    """
    Unlike the previous interpreters which modified the getattr and setattr
    and maintained a stack of references to the quantum register to be used
    as an access path in stack allocated objects, this translator receives
    the qreg over which qubits will be taken and inserted into as a parameter
    during initialization.
    """

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        device,
        shots,
        init_qreg,
        cache,
        qubit_index_recorder,
        *,
        control_wires=(),
        control_values=(),
    ):
        self.device = device
        self.shots = shots
        self.init_qreg = init_qreg
        self.qubit_index_recorder = qubit_index_recorder
        self.subroutine_cache = cache
        self.control_wires = control_wires
        """Any control wires used for a subroutine."""
        self.control_values = control_values
        """Any control values for executing a subroutine."""
        self.has_dynamic_allocation = False

        super().__init__()

    def interpret_operation(self, op, is_adjoint=False, control_values=(), control_wires=()):
        """Re-bind a pennylane operation as a catalyst instruction."""
        if isinstance(op, qml.ops.Adjoint):
            return self.interpret_operation(
                op.base,
                is_adjoint=not is_adjoint,
                control_values=control_values,
                control_wires=control_wires,
            )
        if type(op) in {qml.ops.Controlled, qml.ops.ControlledOp}:
            return self.interpret_operation(
                op.base,
                is_adjoint=is_adjoint,
                control_values=control_values + tuple(op.control_values),
                control_wires=control_wires + tuple(op.control_wires),
            )

        control_wires = control_wires + self.control_wires
        control_values = control_values + self.control_values

        # Insert dynamic qubits if a qreg is available
        if not self.init_qreg.is_qubit_mode():
            self.init_qreg.insert_dynamic_qubits(op.wires + control_wires)

        in_qregs, in_qubits = get_in_qubit_values(
            op.wires, self.qubit_index_recorder, self.init_qreg
        )
        in_ctrl_qregs, in_ctrl_qubits = get_in_qubit_values(
            control_wires, self.qubit_index_recorder, self.init_qreg
        )

        if any(not qreg.is_qubit_mode() and qreg.expired for qreg in in_qregs + in_ctrl_qregs):
            raise CompileError(f"Deallocated qubits cannot be used, but used in {op.name}.")

        out_qubits = qinst_p.bind(
            *[*in_qubits, *op.data, *in_ctrl_qubits, *control_values],
            op=op.name,
            qubits_len=len(op.wires),
            params_len=len(op.data),
            ctrl_len=len(control_wires),
            adjoint=is_adjoint,
        )

        out_non_ctrl_qubits = out_qubits[: len(out_qubits) - len(control_wires)]
        out_ctrl_qubits = out_qubits[-len(control_wires) :]

        for in_qreg, w, new_wire in zip(in_qregs, op.wires, out_non_ctrl_qubits):
            in_qreg[in_qreg.global_index_to_local_index(w)] = new_wire

        for in_ctrl_qreg, w, new_ctrl_wire in zip(in_ctrl_qregs, control_wires, out_ctrl_qubits):
            in_ctrl_qreg[in_ctrl_qreg.global_index_to_local_index(w)] = new_ctrl_wire

        return out_qubits

    def _obs(self, obs):
        """Interpret the observable equation corresponding to a measurement equation's input."""
        if isinstance(obs, qml.ops.Prod):
            # catalyst cant handle product of products
            return tensorobs_p.bind(*(self._obs(t) for t in _flat_prod_gen(obs)))
        if obs.arithmetic_depth > 0:
            with pause():
                coeffs, terms = obs.terms()
            terms = [self._obs(t) for t in terms]
            return hamiltonian_p.bind(jnp.stack(coeffs), *terms)
        wires = [self.init_qreg[w] for w in obs.wires]
        if obs.name == "Hermitian":
            return hermitian_p.bind(obs.data[0], *wires)
        return namedobs_p.bind(*wires, *obs.data, kind=obs.name)

    def _compbasis_obs(self, *wires):
        """Add a computational basis sampling observable."""
        if wires:
            qubits = [self.init_qreg[w] for w in wires]
            return compbasis_p.bind(*qubits)
        else:
            self.init_qreg.insert_all_dangling_qubits()
            return compbasis_p.bind(self.init_qreg.get(), qreg_available=True)

    def interpret_measurement(self, measurement):
        """Rebind a measurement as a catalyst instruction."""
        if self.has_dynamic_allocation:
            if len(measurement.wires) == 0:
                raise CompileError(
                    textwrap.dedent(
                        """
                        Terminal measurements must take in an explicit list of wires when
                        dynamically allocated wires are present in the program.
                        """
                    )
                )
            if any(is_dynamically_allocated_wire(w) for w in measurement.wires):
                raise CompileError(
                    textwrap.dedent(
                        """
                        Terminal measurements cannot take in dynamically allocated wires
                        since they must be temporary.
                        """
                    )
                )

        if type(measurement) not in measurement_map:
            raise NotImplementedError(
                f"measurement {measurement} not yet supported for conversion."
            )

        if measurement._eigvals is not None:
            raise NotImplementedError(
                "from_plxpr does not yet support measurements with manual eigvals."
            )
        if (
            measurement.mv is not None
            or measurement.obs is not None
            and not isinstance(measurement.obs, qml.operation.Operator)
        ):
            raise NotImplementedError("Measurements of mcms are not yet supported.")

        if measurement.obs:
            obs = self._obs(measurement.obs)
        else:
            obs = self._compbasis_obs(*measurement.wires)

        shape, dtype = measurement._abstract_eval(
            n_wires=len(measurement.wires),
            shots=self.shots,
            num_device_wires=len(self.device.wires),
        )

        prim = measurement_map[type(measurement)]
        if prim is sample_p:
            num_qubits = len(measurement.wires) or len(self.device.wires)
            sample_shape = (self.shots, num_qubits)
            dyn_dims, static_shape = jax._src.lax.lax._extract_tracers_dyn_shape(sample_shape)
            mval = sample_p.bind(obs, *dyn_dims, static_shape=tuple(static_shape))
        elif prim in {expval_p, var_p}:
            mval = prim.bind(obs, shape=shape)
        else:
            dyn_dims, static_shape = jax._src.lax.lax._extract_tracers_dyn_shape(shape)
            mval = prim.bind(obs, *dyn_dims, static_shape=tuple(static_shape))

        # sample_p returns floats, so we need to converted it back to the expected integers here
        if dtype != mval.dtype:
            return jax.lax.convert_element_type(mval, dtype)
        return mval

    def _extract_shots_value(self, shots: qml.measurements.Shots | int):
        """Extract the shots value according to the type"""
        if isinstance(shots, int):
            return shots

        assert isinstance(shots, qml.measurements.Shots)

        return shots.total_shots if shots else 0

    def __call__(self, jaxpr, *args):
        """
        Execute this interpreter with this arguments.
        We expect this to be a flat function (i.e., always takes *args as inputs
        and no **kwargs) and the results is a sequence of values
        """
        return self.eval(jaxpr.jaxpr, jaxpr.consts, *args)


@PLxPRToQuantumJaxprInterpreter.register_primitive(qml.allocation.allocate_prim)
def handle_qml_alloc(self, *, num_wires, state=None, restored=False):
    """Handle the conversion from plxpr to Catalyst jaxpr for the qml.allocate primitive"""

    self.has_dynamic_allocation = True

    new_qreg = QubitHandler(
        qalloc_p.bind(num_wires), self.qubit_index_recorder, dynamically_alloced=True
    )

    # The plxpr alloc primitive returns the list of all indices available in the new qreg
    # So let's extract all qubits and return them
    for i in range(num_wires):
        new_qreg.extract(i)

    return new_qreg.get_all_current_global_indices()


@PLxPRToQuantumJaxprInterpreter.register_primitive(qml.allocation.deallocate_prim)
def handle_qml_dealloc(self, *wires):
    """Handle the conversion from plxpr to Catalyst jaxpr for the qml.deallocate primitive"""
    qreg = self.qubit_index_recorder[wires[0]]
    assert all(self.qubit_index_recorder[w] is qreg for w in wires)
    qreg.insert_all_dangling_qubits()
    qreg.expired = True
    qdealloc_p.bind(qreg.get())
    return []


@PLxPRToQuantumJaxprInterpreter.register_primitive(CountsMP._wires_primitive)
def interpret_counts(self, *wires, all_outcomes):
    """Interpret a CountsMP primitive as the catalyst version."""
    obs = self._compbasis_obs(*wires)
    num_wires = len(wires) if wires else len(self.device.wires)
    keys, vals = counts_p.bind(obs, static_shape=(2**num_wires,))
    keys = jax.lax.convert_element_type(keys, int)
    return keys, vals


@PLxPRToQuantumJaxprInterpreter.register_primitive(quantum_subroutine_p)
def handle_subroutine(self, *args, **kwargs):
    """
    Transform the subroutine from PLxPR into JAXPR with quantum primitives.
    """

    if any(is_dynamically_allocated_wire(arg) for arg in args):
        raise NotImplementedError(
            textwrap.dedent(
                """
            Dynamically allocated wires in a parent scope cannot be used in a child
            scope yet. Please consider dynamical allocation inside the child scope.
            """
            )
        )

    backup = dict(self.init_qreg)
    self.init_qreg.insert_all_dangling_qubits()

    # Make sure the quantum register is updated
    plxpr = kwargs["jaxpr"]
    transformed = self.subroutine_cache.get(plxpr)

    def wrapper(qreg, *args):
        # Launch a new interpreter for the new subroutine region
        # A new interpreter's root qreg value needs a new recorder
        converter = copy(self)
        converter.qubit_index_recorder = QubitIndexRecorder()
        init_qreg = QubitHandler(qreg, converter.qubit_index_recorder)
        converter.init_qreg = init_qreg

        retvals = converter(plxpr, *args)
        converter.init_qreg.insert_all_dangling_qubits()
        return converter.init_qreg.get(), *retvals

    if not transformed:
        converted_closed_jaxpr_branch = jax.make_jaxpr(wrapper)(self.init_qreg.get(), *args)
        self.subroutine_cache[plxpr] = converted_closed_jaxpr_branch
    else:
        converted_closed_jaxpr_branch = transformed

    # quantum_subroutine_p.bind
    # is just pjit_p with a different name.
    vals_out = quantum_subroutine_p.bind(
        self.init_qreg.get(),
        *args,
        jaxpr=converted_closed_jaxpr_branch,
        in_shardings=(UNSPECIFIED, *kwargs["in_shardings"]),
        out_shardings=(UNSPECIFIED, *kwargs["out_shardings"]),
        in_layouts=(None, *kwargs["in_layouts"]),
        out_layouts=(None, *kwargs["out_layouts"]),
        donated_invars=kwargs["donated_invars"],
        ctx_mesh=kwargs["ctx_mesh"],
        name=kwargs["name"],
        keep_unused=kwargs["keep_unused"],
        inline=kwargs["inline"],
        compiler_options_kvs=kwargs["compiler_options_kvs"],
    )

    self.init_qreg.set(vals_out[0])
    vals_out = vals_out[1:]

    for orig_wire in backup.keys():
        self.init_qreg.extract(orig_wire)

    return vals_out


@PLxPRToQuantumJaxprInterpreter.register_primitive(decomprule_p)
def handle_decomposition_rule(self, *, pyfun, func_jaxpr, is_qreg, num_params):
    """
    Transform a quantum decomposition rule from PLxPR into JAXPR with quantum primitives.
    """
    if is_qreg:
        self.init_qreg.insert_all_dangling_qubits()

        def wrapper(qreg, *args):
            # Launch a new interpreter for the new subroutine region
            # A new interpreter's root qreg value needs a new recorder
            converter = copy(self)
            converter.qubit_index_recorder = QubitIndexRecorder()
            init_qreg = QubitHandler(qreg, converter.qubit_index_recorder)
            converter.init_qreg = init_qreg

            converter(func_jaxpr, *args)
            converter.init_qreg.insert_all_dangling_qubits()
            return converter.init_qreg.get()

        converted_closed_jaxpr_branch = jax.make_jaxpr(wrapper)(
            self.init_qreg.get(), *func_jaxpr.in_avals
        )
    else:

        def wrapper(*args):
            # Launch a new interpreter for the new subroutine region
            # A new interpreter's root qreg value needs a new recorder

            # TODO: it is a bit messy that the qubit mode of decompositions,
            # which just needs to keep track of a list of explicit qubit's latest SSA values,
            # is going through the entire qreg value mapping infra.
            # Two bitter things here are that:
            #   - qubit lists do not need a recorder (they don't need to remember which qubits
            #     belong to which qregs)
            #   - the qubit list object needs to piggy-back off the `init_qreg` attribute of the
            #     interpreter, which is a wrong name for this case
            # We should refactor the QubitHandler object into a qubit mode object and a qreg
            # mode object.

            converter = copy(self)
            qubit_handler = QubitHandler(args[num_params:], recorder=None)
            converter.init_qreg = qubit_handler

            converter(func_jaxpr, *args)
            return converter.init_qreg.get()

        new_in_avals = func_jaxpr.in_avals[:num_params] + [
            AbstractQbit() for _ in func_jaxpr.in_avals[num_params:]
        ]
        converted_closed_jaxpr_branch = jax.make_jaxpr(wrapper)(*new_in_avals)

    decomprule_p.bind(pyfun=pyfun, func_jaxpr=converted_closed_jaxpr_branch)

    return ()


@PLxPRToQuantumJaxprInterpreter.register_primitive(qml.QubitUnitary._primitive)
def handle_qubit_unitary(self, *invals, n_wires):
    """Handle the conversion from plxpr to Catalyst jaxpr for the QubitUnitary primitive"""
    in_qregs, in_qubits = get_in_qubit_values(invals[1:], self.qubit_index_recorder, self.init_qreg)
    outvals = unitary_p.bind(invals[0], *in_qubits, qubits_len=n_wires, ctrl_len=0, adjoint=False)
    for in_qreg, w, new_wire in zip(in_qregs, invals[1:], outvals):
        in_qreg[in_qreg.global_index_to_local_index(w)] = new_wire


# pylint: disable=unused-argument
@PLxPRToQuantumJaxprInterpreter.register_primitive(qml.GlobalPhase._primitive)
def handle_global_phase(self, phase, *wires, n_wires):
    """Handle the conversion from plxpr to Catalyst jaxpr for the GlobalPhase primitive"""
    gphase_p.bind(phase, ctrl_len=0, adjoint=False)


@PLxPRToQuantumJaxprInterpreter.register_primitive(qml.BasisState._primitive)
def handle_basis_state(self, *invals, n_wires):
    """Handle the conversion from plxpr to Catalyst jaxpr for the BasisState primitive"""
    state_inval = invals[0]
    wires_inval = invals[1:]

    state = jax.lax.convert_element_type(state_inval, jnp.dtype(jnp.bool))
    in_qregs, in_qubits = get_in_qubit_values(
        wires_inval, self.qubit_index_recorder, self.init_qreg
    )
    out_wires = set_basis_state_p.bind(*in_qubits, state)

    for in_qreg, w, new_wire in zip(in_qregs, wires_inval, out_wires):
        in_qreg[in_qreg.global_index_to_local_index(w)] = new_wire


# pylint: disable=unused-argument
@PLxPRToQuantumJaxprInterpreter.register_primitive(qml.StatePrep._primitive)
def handle_state_prep(self, *invals, n_wires, **kwargs):
    """Handle the conversion from plxpr to Catalyst jaxpr for the StatePrep primitive"""
    state_inval = invals[0]
    wires_inval = invals[1:]

    # jnp.complex128 is the top element in the type promotion lattice so it is ok to do this:
    # https://jax.readthedocs.io/en/latest/type_promotion.html
    state = jax.lax.convert_element_type(state_inval, jnp.dtype(jnp.complex128))
    in_qregs, in_qubits = get_in_qubit_values(
        wires_inval, self.qubit_index_recorder, self.init_qreg
    )
    out_wires = set_state_p.bind(*in_qubits, state)

    for in_qreg, w, new_wire in zip(in_qregs, wires_inval, out_wires):
        in_qreg[in_qreg.global_index_to_local_index(w)] = new_wire


@PLxPRToQuantumJaxprInterpreter.register_primitive(plxpr_measure_prim)
def handle_measure(self, wire, reset, postselect):
    """Handle the conversion from plxpr to Catalyst jaxpr for the mid-circuit measure primitive."""

    in_qreg, in_wire = (
        _[0] for _ in get_in_qubit_values([wire], self.qubit_index_recorder, self.init_qreg)
    )
    result, out_wire = measure_p.bind(in_wire, postselect=postselect)

    if reset:
        # Constants need to be passed as input values for some reason I forgot about.
        correction = jaxpr_pad_consts(
            [
                jax.make_jaxpr(lambda: qinst_p.bind(out_wire, op="PauliX", qubits_len=1))().jaxpr,
                jax.make_jaxpr(lambda: out_wire)().jaxpr,
            ]
        )
        out_wire = cond_p.bind(
            result, out_wire, out_wire, branch_jaxprs=correction, nimplicit_outputs=None
        )[0]

    in_qreg[in_qreg.global_index_to_local_index(wire)] = out_wire
    return result


# pylint: disable=unused-argument, too-many-positional-arguments
@PLxPRToQuantumJaxprInterpreter.register_primitive(plxpr_measure_in_basis_prim)
def handle_measure_in_basis(self, angle, wire, plane, reset, postselect):
    """Handle the conversion from plxpr to Catalyst jaxpr for the measure_in_basis primitive"""
    _angle = jax.lax.convert_element_type(angle, jnp.dtype(jnp.float64))

    try:
        _plane = MeasurementPlane(plane)
    except ValueError as e:
        raise ValueError(
            f"Measurement plane must be one of {[plane.value for plane in MeasurementPlane]}"
        ) from e

    in_qreg, in_wire = (
        _[0] for _ in get_in_qubit_values([wire], self.qubit_index_recorder, self.init_qreg)
    )
    result, out_wire = measure_in_basis_p.bind(_angle, in_wire, plane=_plane, postselect=postselect)

    in_qreg[in_qreg.global_index_to_local_index(wire)] = out_wire

    return result


# pylint: disable=unused-argument
@PLxPRToQuantumJaxprInterpreter.register_primitive(plxpr_ctrl_transform_prim)
def handle_ctrl_transform(self, *invals, jaxpr, n_control, control_values, work_wires, n_consts):
    """Interpret a control transform primitive."""
    consts = invals[:n_consts]
    args = invals[n_consts:-n_control]
    control_wires = invals[-n_control:]

    unroller = copy(self)
    unroller.control_wires += tuple(control_wires)
    unroller.control_values += tuple(control_values)
    unroller.eval(jaxpr, consts, *args)
    return []


# pylint: disable=unused-argument
@PLxPRToQuantumJaxprInterpreter.register_primitive(plxpr_adjoint_transform_prim)
def handle_adjoint_transform(
    self,
    *plxpr_invals,
    jaxpr,
    lazy,
    n_consts,
):
    """Handle the conversion from plxpr to Catalyst jaxpr for the adjoint primitive"""
    assert jaxpr is not None
    consts = plxpr_invals[:n_consts]
    args = plxpr_invals[n_consts:]

    # Add the iteration start and the qreg to the args
    self.init_qreg.insert_all_dangling_qubits()
    qreg = self.init_qreg.get()

    jaxpr = ClosedJaxpr(jaxpr, consts)

    def calling_convention(*args_plus_qreg):
        # The last arg is the scope argument for the body jaxpr
        *args, qreg = args_plus_qreg

        # Launch a new interpreter for the body region
        # A new interpreter's root qreg value needs a new recorder
        converter = copy(self)
        converter.qubit_index_recorder = QubitIndexRecorder()
        init_qreg = QubitHandler(qreg, converter.qubit_index_recorder)
        converter.init_qreg = init_qreg

        retvals = converter(jaxpr, *args)
        init_qreg.insert_all_dangling_qubits()
        return *retvals, converter.init_qreg.get()

    _, args_tree = tree_flatten((consts, args, [qreg]))
    converted_jaxpr_branch = jax.make_jaxpr(calling_convention)(*consts, *args, qreg).jaxpr

    converted_closed_jaxpr_branch = ClosedJaxpr(convert_constvars_jaxpr(converted_jaxpr_branch), ())
    # Perform the binding
    outvals = adjoint_p.bind(
        *consts,
        *args,
        qreg,
        jaxpr=converted_closed_jaxpr_branch,
        args_tree=args_tree,
    )

    # We assume the last output value is the returned qreg.
    # Update the current qreg and remove it from the output values.
    self.init_qreg.set(outvals.pop())

    # Return only the output values that match the plxpr output values
    return outvals


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
