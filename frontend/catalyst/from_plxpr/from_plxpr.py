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
import jax.core
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
from pennylane.decomposition.collect_resource_ops import CollectResourceOps
from pennylane.ftqc.primitives import measure_in_basis_prim as plxpr_measure_in_basis_prim
from pennylane.ops.functions.map_wires import _map_wires_transform as pl_map_wires
from pennylane.transforms import cancel_inverses as pl_cancel_inverses
from pennylane.transforms import commute_controlled as pl_commute_controlled
from pennylane.transforms import decompose as pl_decompose
from pennylane.transforms import merge_amplitude_embedding as pl_merge_amplitude_embedding
from pennylane.transforms import merge_rotations as pl_merge_rotations
from pennylane.transforms import single_qubit_fusion as pl_single_qubit_fusion
from pennylane.transforms import unitary_to_rot as pl_unitary_to_rot

from catalyst.device import extract_backend_info
from catalyst.device.qjit_device import COMPILER_OPERATIONS
from catalyst.from_plxpr.decompose import (
    GraphSolutionInterpreter,
    PreMlirDecomposeInterpreter,
)
from catalyst.from_plxpr.qubit_handler import QubitHandler
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
    """An interpreter that converts a qnode primitive from a plxpr variant to a catalyst variant."""

    def __init__(self):
        self._pass_pipeline = []
        self.qubit_handler = None
        self.compiler_decompose = False
        super().__init__()


def _decompose_to_compiler_gateset(qfunc_jaxpr, consts, non_const_args):
    """First stage decomposition to compiler gate set.

    Currently, the compiler can only handle a limited set of gates and
    may not support all generic gates and templates of the original circuit.
    We perform a first stage decomposition to the compiler gate set, which includes
    only a subset of the original gates that can be represented in MLIR using the
    `quantum.custom` primitive.
    """

    # TODO: The compiler should be able to handle all gate
    # adhering to quantum.custom primitive.This includes
    # all the gates with parameters of type `TensorLike`
    # and wires of type `WiresLike` with no hyperparams.
    # Update `gate_set` to use this as the stopping condition
    # of the decomposition transform.
    gate_set = COMPILER_OPERATIONS

    decomp_args = ()
    decomp_kwargs = {"gate_set": gate_set}

    # disable the graph decomposition optimization
    graph_decomp_status = False
    if qml.decomposition.enabled_graph():
        graph_decomp_status = True
        qml.decomposition.disable_graph()

    new_jaxpr = qml.transforms.decompose.plxpr_transform(
        qfunc_jaxpr, consts, decomp_args, decomp_kwargs, *non_const_args
    )

    if graph_decomp_status:
        qml.decomposition.enable_graph()

    return new_jaxpr


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

    def calling_convention(*args):
        device_init_p.bind(
            shots,
            auto_qubit_management=(device.wires is None),
            **_get_device_kwargs(device),
        )
        qreg = qalloc_p.bind(len(device.wires))
        self.qubit_handler = QubitHandler(qreg)
        converter = PLxPRToQuantumJaxprInterpreter(device, shots, self.qubit_handler, {})
        retvals = converter(closed_jaxpr, *args)
        self.qubit_handler.insert_all_dangling_qubits()
        qdealloc_p.bind(self.qubit_handler.get())
        device_release_p.bind()
        return retvals

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


# pylint: disable=too-many-arguments
def handle_graph_decomposition(*args, inner_jaxpr, consts, non_const_args, targs, tkwargs):
    """Handle the graph decomposition for a given JAXPR."""

    gate_set = COMPILER_OPERATIONS
    decomp_kwargs = {"gate_set": gate_set}

    pmd_interpreter = PreMlirDecomposeInterpreter(*targs, **decomp_kwargs)

    def pmd_wrapper(*args):
        return pmd_interpreter.eval(inner_jaxpr, consts, *args)

    pmd_jaxpr = jax.make_jaxpr(pmd_wrapper)(*args)

    ops_collector = CollectResourceOps()
    ops_collector.eval(pmd_jaxpr.jaxpr, consts, *args)
    pl_ops = ops_collector.state["ops"]

    gds_interpreter = GraphSolutionInterpreter(*targs, **tkwargs, operations=pl_ops)

    def gds_wrapper(*args):
        return gds_interpreter.eval(pmd_jaxpr.jaxpr, consts, *args)

    return jax.make_jaxpr(gds_wrapper)(*args)


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

        # Check if the transform is a decomposition transform
        #
        # Note:
        # - The list of target gateset is always taken from the transform's attributes
        #   and passed down to the MLIR lowering as a quantum function attribute.
        if (
            hasattr(pl_plxpr_transform, "__name__")
            and pl_plxpr_transform.__name__ == "decompose_plxpr_to_plxpr"
            and qml.decomposition.enabled_graph()
        ):
            self.compiler_decompose = True

        if self.compiler_decompose:
            final_jaxpr = handle_graph_decomposition(
                *args,
                inner_jaxpr=inner_jaxpr,
                consts=consts,
                non_const_args=non_const_args,
                targs=targs,
                tkwargs=tkwargs,
            )
            return self.eval(final_jaxpr.jaxpr, final_jaxpr.consts, *non_const_args)

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
        else:
            # Apply the corresponding Catalyst pass counterpart
            self._pass_pipeline.append(Pass(catalyst_pass_name))
            return self.eval(inner_jaxpr, consts, *non_const_args)


# This is our registration factory for PL transforms. The loop below iterates
# across the map above and generates a custom handler for each transform.
# In order to ensure early binding, we pass the PL plxpr transform and the
# Catalyst pass as arguments whose default values are set by the loop.
for pl_transform, (pass_name, decomposition) in transforms_to_passes.items():
    register_transform(pl_transform, pass_name, decomposition)


class PLxPRToQuantumJaxprInterpreter(PlxprInterpreter):
    """
    Unlike the previous interpreters which modified the getattr and setattr
    and maintained a stack of references to the quantum register to be used
    as an access path in stack allocated objects, this translator receives
    the qreg over which qubits will be taken and inserted into as a parameter
    during initialization.
    """

    def __init__(self, device, shots, qubit_handler, cache, *, control_wires=(), control_values=()):
        self.device = device
        self.shots = shots
        # TODO: we assume the qreg value passed into a scope is the unique qreg in the scope
        # In other words, we assume no new qreg will be allocated in the scope
        self.qubit_handler = qubit_handler
        self.compiler_decompose = False
        self.subroutine_cache = cache
        self.control_wires = control_wires
        """Any control wires used for a subroutine."""
        self.control_values = control_values
        """Any control values for executing a subroutine."""

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
        if self.qubit_handler.abstract_qreg_val is not None:
            self.qubit_handler.insert_dynamic_qubits(op.wires + control_wires)

        in_qubits = [self.qubit_handler[w] for w in op.wires]
        control_qubits = [self.qubit_handler[w] for w in control_wires]

        out_qubits = qinst_p.bind(
            *[*in_qubits, *op.data, *control_qubits, *control_values],
            op=op.name,
            qubits_len=len(op.wires),
            params_len=len(op.data),
            ctrl_len=len(control_wires),
            adjoint=is_adjoint,
        )
        for wire_values, new_wire in zip(tuple(op.wires) + control_wires, out_qubits, strict=True):
            self.qubit_handler[wire_values] = new_wire

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
        wires = [self.qubit_handler[w] for w in obs.wires]
        if obs.name == "Hermitian":
            return hermitian_p.bind(obs.data[0], *wires)
        return namedobs_p.bind(*wires, *obs.data, kind=obs.name)

    def _compbasis_obs(self, *wires):
        """Add a computational basis sampling observable."""
        if wires:
            qubits = [self.qubit_handler[w] for w in wires]
            return compbasis_p.bind(*qubits)
        else:
            self.qubit_handler.insert_all_dangling_qubits()
            return compbasis_p.bind(self.qubit_handler.get(), qreg_available=True)

    def interpret_measurement(self, measurement):
        """Rebind a measurement as a catalyst instruction."""
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
        assert (
            prim is not counts_p
        ), "CountsMP returns a dictionary, which is not compatible with capture"
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


@PLxPRToQuantumJaxprInterpreter.register_primitive(quantum_subroutine_p)
def handle_subroutine(self, *args, **kwargs):
    """
    Transform the subroutine from PLxPR into JAXPR with quantum primitives.
    """
    backup = dict(self.qubit_handler)
    self.qubit_handler.insert_all_dangling_qubits()

    # Make sure the quantum register is updated
    plxpr = kwargs["jaxpr"]
    transformed = self.subroutine_cache.get(plxpr)

    def wrapper(qreg, *args):
        qubit_handler = QubitHandler(qreg)
        converter = copy(self)
        converter.qubit_handler = qubit_handler
        retvals = converter(plxpr, *args)
        converter.qubit_handler.insert_all_dangling_qubits()
        return converter.qubit_handler.get(), *retvals

    if not transformed:
        converted_closed_jaxpr_branch = jax.make_jaxpr(wrapper)(self.qubit_handler.get(), *args)
        self.subroutine_cache[plxpr] = converted_closed_jaxpr_branch
    else:
        converted_closed_jaxpr_branch = transformed

    # quantum_subroutine_p.bind
    # is just pjit_p with a different name.
    vals_out = quantum_subroutine_p.bind(
        self.qubit_handler.get(),
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

    self.qubit_handler.set(vals_out[0])
    vals_out = vals_out[1:]

    for orig_wire in backup.keys():
        self.qubit_handler.extract(orig_wire)

    return vals_out


@PLxPRToQuantumJaxprInterpreter.register_primitive(decomprule_p)
def handle_decomposition_rule(self, *, pyfun, func_jaxpr, is_qreg, num_params):
    """
    Transform a quantum decomposition rule from PLxPR into JAXPR with quantum primitives.
    """

    if is_qreg:
        self.qubit_handler.insert_all_dangling_qubits()

        def wrapper(qreg, *args):
            qubit_handler = QubitHandler(qreg)
            converter = copy(self)
            converter.qubit_handler = qubit_handler
            converter(func_jaxpr, *args)
            converter.qubit_handler.insert_all_dangling_qubits()
            return converter.qubit_handler.get()

        converted_closed_jaxpr_branch = jax.make_jaxpr(wrapper)(
            self.qubit_handler.get(), *func_jaxpr.in_avals
        )
    else:

        def wrapper(*args):
            qubit_handler = QubitHandler(args[num_params:])
            converter = copy(self)
            converter.qubit_handler = qubit_handler
            converter(func_jaxpr, *args)
            return converter.qubit_handler.get()

        new_in_avals = func_jaxpr.in_avals[:num_params] + [
            AbstractQbit() for _ in func_jaxpr.in_avals[num_params:]
        ]
        converted_closed_jaxpr_branch = jax.make_jaxpr(wrapper)(*new_in_avals)

    decomprule_p.bind(pyfun=pyfun, func_jaxpr=converted_closed_jaxpr_branch)

    return ()


@PLxPRToQuantumJaxprInterpreter.register_primitive(qml.QubitUnitary._primitive)
def handle_qubit_unitary(self, *invals, n_wires):
    """Handle the conversion from plxpr to Catalyst jaxpr for the QubitUnitary primitive"""
    wires = [self.qubit_handler[w] for w in invals[1:]]
    outvals = unitary_p.bind(invals[0], *wires, qubits_len=n_wires, ctrl_len=0, adjoint=False)
    for wire_values, new_wire in zip(invals[1:], outvals):
        self.qubit_handler[wire_values] = new_wire


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
    wires = [self.qubit_handler[w] for w in wires_inval]
    out_wires = set_basis_state_p.bind(*wires, state)

    for wire_values, new_wire in zip(wires_inval, out_wires):
        self.qubit_handler[wire_values] = new_wire


# pylint: disable=unused-argument
@PLxPRToQuantumJaxprInterpreter.register_primitive(qml.StatePrep._primitive)
def handle_state_prep(self, *invals, n_wires, **kwargs):
    """Handle the conversion from plxpr to Catalyst jaxpr for the StatePrep primitive"""
    state_inval = invals[0]
    wires_inval = invals[1:]

    # jnp.complex128 is the top element in the type promotion lattice so it is ok to do this:
    # https://jax.readthedocs.io/en/latest/type_promotion.html
    state = jax.lax.convert_element_type(state_inval, jnp.dtype(jnp.complex128))
    wires = [self.qubit_handler[w] for w in wires_inval]
    out_wires = set_state_p.bind(*wires, state)

    for wire_values, new_wire in zip(wires_inval, out_wires):
        self.qubit_handler[wire_values] = new_wire


@PLxPRToQuantumJaxprInterpreter.register_primitive(plxpr_measure_prim)
def handle_measure(self, wire, reset, postselect):
    """Handle the conversion from plxpr to Catalyst jaxpr for the mid-circuit measure primitive."""

    in_wire = self.qubit_handler[wire]

    result, out_wire = measure_p.bind(in_wire, postselect=postselect)

    if reset:
        # Constants need to be passed as input values for some reason I forgot about.
        correction = jaxpr_pad_consts(
            [
                jax.make_jaxpr(lambda: qinst_p.bind(in_wire, op="PauliX", qubits_len=1))().jaxpr,
                jax.make_jaxpr(lambda: out_wire)().jaxpr,
            ]
        )
        out_wire = cond_p.bind(
            result, in_wire, out_wire, branch_jaxprs=correction, nimplicit_outputs=None
        )[0]

    self.qubit_handler[wire] = out_wire
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

    in_wire = self.qubit_handler[wire]
    result, out_wire = measure_in_basis_p.bind(_angle, in_wire, plane=_plane, postselect=postselect)

    self.qubit_handler[wire] = out_wire

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
    self.qubit_handler.insert_all_dangling_qubits()
    qreg = self.qubit_handler.get()

    jaxpr = ClosedJaxpr(jaxpr, consts)

    def calling_convention(*args_plus_qreg):
        *args, qreg = args_plus_qreg
        # `qreg` is the scope argument for the body jaxpr
        qubit_handler = QubitHandler(qreg)
        converter = copy(self)
        converter.qubit_handler = qubit_handler
        retvals = converter(jaxpr, *args)
        qubit_handler.insert_all_dangling_qubits()
        return *retvals, converter.qubit_handler.get()

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
    self.qubit_handler.set(outvals.pop())

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
