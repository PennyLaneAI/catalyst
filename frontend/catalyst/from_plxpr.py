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
from functools import partial, wraps
from typing import Callable, Sequence

import jax
import jax.core
import jax.numpy as jnp
import pennylane as qml
from jax.extend.core import ClosedJaxpr, Jaxpr, jaxpr_as_fun
from jax.extend.linear_util import wrap_init
from jax.interpreters.partial_eval import convert_constvars_jaxpr
from pennylane.capture import PlxprInterpreter, qnode_prim
from pennylane.capture.expand_transforms import ExpandTransformsInterpreter
from pennylane.capture.primitives import cond_prim as plxpr_cond_prim
from pennylane.capture.primitives import for_loop_prim as plxpr_for_loop_prim
from pennylane.capture.primitives import measure_prim as plxpr_measure_prim
from pennylane.capture.primitives import while_loop_prim as plxpr_while_loop_prim
from pennylane.ftqc.primitives import measure_in_basis_prim as plxpr_measure_in_basis_prim
from pennylane.ops.functions.map_wires import _map_wires_transform as pl_map_wires
from pennylane.transforms import cancel_inverses as pl_cancel_inverses
from pennylane.transforms import commute_controlled as pl_commute_controlled
from pennylane.transforms import decompose as pl_decompose
from pennylane.transforms import merge_amplitude_embedding as pl_merge_amplitude_embedding
from pennylane.transforms import merge_rotations as pl_merge_rotations
from pennylane.transforms import single_qubit_fusion as pl_single_qubit_fusion
from pennylane.transforms import unitary_to_rot as pl_unitary_to_rot

from catalyst.device import (
    extract_backend_info,
    get_device_capabilities,
)
from catalyst.jax_extras import jaxpr_pad_consts, make_jaxpr2, transient_jax_config
from catalyst.jax_primitives import (
    AbstractQbit,
    AbstractQreg,
    MeasurementPlane,
    compbasis_p,
    cond_p,
    counts_p,
    device_init_p,
    device_release_p,
    expval_p,
    for_p,
    gphase_p,
    measure_in_basis_p,
    measure_p,
    namedobs_p,
    probs_p,
    qalloc_p,
    qdealloc_p,
    qextract_p,
    qinsert_p,
    qinst_p,
    quantum_kernel_p,
    sample_p,
    set_basis_state_p,
    set_state_p,
    state_p,
    unitary_p,
    var_p,
    while_p,
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
    capabilities = get_device_capabilities(device)
    info = extract_backend_info(device, capabilities)
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
    """An interpreter that converts a qnode primitive from a plxpr variant to a catalxpr variant."""

    def __init__(self):
        self._pass_pipeline = []
        super().__init__()


# pylint: disable=unused-argument, too-many-arguments
@WorkflowInterpreter.register_primitive(qnode_prim)
def handle_qnode(
    self, *args, qnode, shots, device, execution_config, qfunc_jaxpr, n_consts, batch_dims=None
):
    """Handle the conversion from plxpr to Catalyst jaxpr for the qnode primitive"""
    consts = args[:n_consts]
    non_const_args = args[n_consts:]

    f = partial(QFuncPlxprInterpreter(device, shots).eval, qfunc_jaxpr, consts)

    return quantum_kernel_p.bind(
        wrap_init(f, debug_info=qfunc_jaxpr.debug_info),
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

    # pylint: disable=unused-argument, too-many-arguments, cell-var-from-loop
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

    def __init__(self, device, shots, qreg):
        self.device = device
        self.shots = shots
        self.wire_map = {}
        self.qreg = qreg
        self.actualize = True
        super().__init__()

    def get_wire(self, wire_value) -> AbstractQbit:
        """Get the ``AbstractQbit`` corresponding to a wire value."""
        if wire_value in self.wire_map:
            return self.wire_map[wire_value]
        self.actualized = False
        return qextract_p.bind(self.qreg, wire_value)

    def actualize_qreg(self):
        """
        Insert all end qubits back into a qreg,
        and produce the product qreg jaxpr variable.
        """
        self.actualized = True
        for orig_wire, wire in self.wire_map.items():
            # Note: since `getattr` checks specifically for qreg, we can't
            # define qreg inside the init function.
            # pylint: disable-next=attribute-defined-outside-init
            self.qreg = qinsert_p.bind(self.qreg, orig_wire, wire)

    def interpret_operation(self, op):
        """Re-bind a pennylane operation as a catalyst instruction."""

        in_qubits = [self.get_wire(w) for w in op.wires]
        out_qubits = qinst_p.bind(
            *[*in_qubits, *op.data],
            op=op.name,
            qubits_len=len(op.wires),
            params_len=len(op.data),
            ctrl_len=0,
            adjoint=False,
        )
        for wire_values, new_wire in zip(op.wires, out_qubits):
            self.wire_map[wire_values] = new_wire

        return out_qubits

    def _obs(self, obs):
        """Interpret the observable equation corresponding to a measurement equation's input."""
        if obs.arithmetic_depth > 0:
            raise NotImplementedError("operator arithmetic not yet supported for conversion.")
        wires = [self.get_wire(w) for w in obs.wires]
        return namedobs_p.bind(*wires, *obs.data, kind=obs.name)

    def _compbasis_obs(self, *wires):
        """Add a computational basis sampling observable."""
        if wires:
            qubits = [self.get_wire(w) for w in wires]
            return compbasis_p.bind(*qubits)
        else:
            self.actualize_qreg()
            return compbasis_p.bind(self.qreg, qreg_available=True)

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
            shots=self.device.shots.total_shots,
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

    def actualize_qreg(self):
        """
        Insert all end qubits back into a qreg,
        and produce the product qreg jaxpr variable.
        """
        self.actualized = True
        for orig_wire, wire in self.wire_map.items():
            # Note: since `getattr` checks specifically for qreg, we can't
            # define qreg inside the init function.
            # pylint: disable-next=attribute-defined-outside-init
            self.qreg = qinsert_p.bind(self.qreg, orig_wire, wire)

    def __call__(self, fun, *args):
        """
        Execute this interpreter with this arguments.
        We expect this to be a flat function (i.e., always takes *args as inputs
        and no **kwargs) and the results is a sequence of values
        """
        jaxpr = jax.make_jaxpr(fun)(*args)
        return self.eval(jaxpr.jaxpr, jaxpr.consts, *args)


class SubroutineInterpreter(PlxprInterpreter):
    """Base interpreter for quantum operations.

    It is a subroutine interpreter because unlike the QFuncPlxprInterpreter it
    * does not allocate a new register upon beginning,
    * does not deallocate the quantum register upon ending,
    * and it does not release the quantum device back to the runtime.

    Args:
        device (qml.devices.Device)
        shots (qml.measurements.Shots)
    """

    def __init__(self, device, shots: qml.measurements.Shots | int):
        self.device = device
        self.shots = self._extract_shots_value(shots)
        self.stateref = None
        self.actualized = False
        super().__init__()

    def __getattr__(self, key):
        if key in {"qreg", "wire_map"}:
            return self.stateref[key]
        raise AttributeError(f"no attribute {key}")

    def __setattr__(self, __name: str, __value) -> None:
        if __name in {"qreg", "wire_map"}:
            self.stateref[__name] = __value
        else:
            super().__setattr__(__name, __value)

    def get_wire(self, wire_value) -> AbstractQbit:
        """Get the ``AbstractQbit`` corresponding to a wire value."""
        if wire_value in self.wire_map:
            return self.wire_map[wire_value]
        self.actualized = False
        return qextract_p.bind(self.qreg, wire_value)

    def actualize_qreg(self):
        """
        Insert all end qubits back into a qreg,
        and produce the product qreg jaxpr variable.
        """
        self.actualized = True
        for orig_wire, wire in self.wire_map.items():
            # Note: since `getattr` checks specifically for qreg, we can't
            # define qreg inside the init function.
            # pylint: disable-next=attribute-defined-outside-init
            self.qreg = qinsert_p.bind(self.qreg, orig_wire, wire)

    def interpret_operation(self, op):
        """Re-bind a pennylane operation as a catalyst instruction."""

        in_qubits = [self.get_wire(w) for w in op.wires]
        out_qubits = qinst_p.bind(
            *[*in_qubits, *op.data],
            op=op.name,
            qubits_len=len(op.wires),
            params_len=len(op.data),
            ctrl_len=0,
            adjoint=False,
        )
        for wire_values, new_wire in zip(op.wires, out_qubits):
            self.wire_map[wire_values] = new_wire

        return out_qubits

    def _obs(self, obs):
        """Interpret the observable equation corresponding to a measurement equation's input."""
        if obs.arithmetic_depth > 0:
            raise NotImplementedError("operator arithmetic not yet supported for conversion.")
        wires = [self.get_wire(w) for w in obs.wires]
        return namedobs_p.bind(*wires, *obs.data, kind=obs.name)

    def _compbasis_obs(self, *wires):
        """Add a computational basis sampling observable."""
        if wires:
            qubits = [self.get_wire(w) for w in wires]
            return compbasis_p.bind(*qubits)
        else:
            self.actualize_qreg()
            return compbasis_p.bind(self.qreg, qreg_available=True)

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
            shots=self.device.shots.total_shots,
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

    def eval(self, jaxpr: "jax.core.Jaxpr", consts: Sequence, *args) -> list:
        """Evaluate a jaxpr.
        Args:
            jaxpr (jax.core.Jaxpr): the jaxpr to evaluate
            consts (list[TensorLike]): the constant variables for the jaxpr
            *args (tuple[TensorLike]): The arguments for the jaxpr.
        Returns:
            list[TensorLike]: the results of the execution.

        # We assume we have at least one argument (the qreg)
        assert len(args) > 0

        self._parent_qreg = args[0]
        self.stateref = {"qreg": self._parent_qreg, "wire_map": {}}

        # Send the original args (without the qreg)
        outvals = super().eval(jaxpr, consts, *args)

        # Add the qreg to the output values
        self.qreg, retvals = outvals[0], outvals[1:]

        self.actualize_qreg()

        outvals = (self.qreg, *retvals)

        self.stateref = None

        return outvals
        """
        raise NotImplementedError("Unreachable code until we add subroutine feature")


class QFuncPlxprInterpreter(SubroutineInterpreter):
    """An interpreter that converts plxpr into catalyst-variant jaxpr.

    Args:
        device (qml.devices.Device)
        shots (qml.measurements.Shots)

    """

    def setup(self):
        """Initialize the stateref and bind the device."""
        if self.stateref is None:
            device_init_p.bind(
                self.shots,
                auto_qubit_management=(self.device.wires is None),
                **_get_device_kwargs(self.device),
            )
            self.stateref = {"qreg": qalloc_p.bind(len(self.device.wires)), "wire_map": {}}

    # pylint: disable=attribute-defined-outside-init
    def cleanup(self):
        """Perform any final steps after processing the plxpr.

        For conversion to calayst, this reinserts extracted qubits and
        deallocates the register, and releases the device.
        """
        if not self.actualized:
            self.actualize_qreg()
        qdealloc_p.bind(self.qreg)
        device_release_p.bind()
        self.stateref = None

    def eval(self, jaxpr: "jax.core.Jaxpr", consts: Sequence, *args) -> list:
        """Use the PlxprInterpreter.eval method and not the SubroutineInterpreter

        The Subroutine's eval method expects the first argument to be the qreg.
        This will not be the case when first evaluating a QFuncPlxprInterpreter
        as the qreg will be available only after the function has started running.
        It will be one of the first instructions in the function and it is
        added by the setup function.
        """
        return PlxprInterpreter.eval(self, jaxpr, consts, *args)


@PLxPRToQuantumJaxprInterpreter.register_primitive(qml.QubitUnitary._primitive)
@QFuncPlxprInterpreter.register_primitive(qml.QubitUnitary._primitive)
def handle_qubit_unitary(self, *invals, n_wires):
    """Handle the conversion from plxpr to Catalyst jaxpr for the QubitUnitary primitive"""
    wires = [self.get_wire(w) for w in invals[1:]]
    outvals = unitary_p.bind(invals[0], *wires, qubits_len=n_wires, ctrl_len=0, adjoint=False)
    for wire_values, new_wire in zip(invals[1:], outvals):
        self.wire_map[wire_values] = new_wire


# pylint: disable=unused-argument
@PLxPRToQuantumJaxprInterpreter.register_primitive(qml.GlobalPhase._primitive)
@QFuncPlxprInterpreter.register_primitive(qml.GlobalPhase._primitive)
def handle_global_phase(self, phase, *wires, n_wires):
    """Handle the conversion from plxpr to Catalyst jaxpr for the GlobalPhase primitive"""
    gphase_p.bind(phase, ctrl_len=0, adjoint=False)


@PLxPRToQuantumJaxprInterpreter.register_primitive(qml.BasisState._primitive)
@QFuncPlxprInterpreter.register_primitive(qml.BasisState._primitive)
def handle_basis_state(self, *invals, n_wires):
    """Handle the conversion from plxpr to Catalyst jaxpr for the BasisState primitive"""
    state_inval = invals[0]
    wires_inval = invals[1:]

    state = jax.lax.convert_element_type(state_inval, jnp.dtype(jnp.bool))
    wires = [self.get_wire(w) for w in wires_inval]
    out_wires = set_basis_state_p.bind(*wires, state)

    for wire_values, new_wire in zip(wires_inval, out_wires):
        self.wire_map[wire_values] = new_wire


# pylint: disable=unused-argument
@PLxPRToQuantumJaxprInterpreter.register_primitive(qml.StatePrep._primitive)
@QFuncPlxprInterpreter.register_primitive(qml.StatePrep._primitive)
def handle_state_prep(self, *invals, n_wires, **kwargs):
    """Handle the conversion from plxpr to Catalyst jaxpr for the StatePrep primitive"""
    state_inval = invals[0]
    wires_inval = invals[1:]

    # jnp.complex128 is the top element in the type promotion lattice so it is ok to do this:
    # https://jax.readthedocs.io/en/latest/type_promotion.html
    state = jax.lax.convert_element_type(state_inval, jnp.dtype(jnp.complex128))
    wires = [self.get_wire(w) for w in wires_inval]
    out_wires = set_state_p.bind(*wires, state)

    for wire_values, new_wire in zip(wires_inval, out_wires):
        self.wire_map[wire_values] = new_wire


# pylint: disable=unused-argument, too-many-arguments
@PLxPRToQuantumJaxprInterpreter.register_primitive(plxpr_cond_prim)
@QFuncPlxprInterpreter.register_primitive(plxpr_cond_prim)
def handle_cond(self, *plxpr_invals, jaxpr_branches, consts_slices, args_slice):
    """Handle the conversion from plxpr to Catalyst jaxpr for the cond primitive"""
    args = plxpr_invals[args_slice]
    args_plus_qreg = [*args, self.qreg]  # Add the qreg to the args
    converted_jaxpr_branches = []
    all_consts = []

    # Convert each branch from plxpr to jaxpr
    for const_slice, plxpr_branch in zip(consts_slices, jaxpr_branches):

        # Store all branches consts in a flat list
        branch_consts = plxpr_invals[const_slice]
        all_consts = all_consts + [*branch_consts]

        converted_jaxpr_branch = None

        if plxpr_branch is None:
            # Emit a new Catalyst jaxpr branch that simply returns a qreg
            converted_jaxpr_branch = jax.make_jaxpr(lambda x: x)(*args_plus_qreg).jaxpr
        else:

            def flat_fun(*args):
                closed_jaxpr = ClosedJaxpr(plxpr_branch, branch_consts)
                return jaxpr_as_fun(closed_jaxpr)(*args)

            def calling_convention(*args_plus_qreg):
                *args, qreg = args_plus_qreg
                device = self.device
                shots = self.shots
                converter = PLxPRToQuantumJaxprInterpreter(device, shots, qreg)
                retval = converter(flat_fun, *args)
                # We need to define a subroutine interpreter here.
                # That takes the qreg as an input.
                converter.actualize_qreg()
                return *retval, converter.qreg

            converted_jaxpr_branch = jax.make_jaxpr(calling_convention)(*args_plus_qreg).jaxpr

        converted_jaxpr_branches.append(converted_jaxpr_branch)

    # The slice [0,1) of the plxpr input values contains the true predicate of the plxpr cond,
    # whereas the slice [1,2) refers to the false predicate, which is always True.
    # We extract the true predicate and discard the false one.
    predicate_slice = slice(0, 1)
    predicate = plxpr_invals[predicate_slice]

    # Build Catalyst compatible input values
    cond_invals = [*predicate, *all_consts, *args_plus_qreg]

    # Perform the binding
    outvals = cond_p.bind(
        *cond_invals,
        branch_jaxprs=jaxpr_pad_consts(converted_jaxpr_branches),
        nimplicit_outputs=None,
    )

    # We assume the last output value is the returned qreg.
    # Update the current qreg and remove it from the output values.
    self.qreg = outvals.pop()

    # Return only the output values that match the plxpr output values
    return outvals


# pylint: disable=unused-argument, too-many-arguments
@PLxPRToQuantumJaxprInterpreter.register_primitive(plxpr_for_loop_prim)
@QFuncPlxprInterpreter.register_primitive(plxpr_for_loop_prim)
def handle_for_loop(
    self,
    start,
    stop,
    step,
    *plxpr_invals,
    jaxpr_body_fn,
    consts_slice,
    args_slice,
    abstract_shapes_slice,
):
    """Handle the conversion from plxpr to Catalyst jaxpr for the for loop primitive"""
    assert jaxpr_body_fn is not None
    args = plxpr_invals[args_slice]

    # Add the iteration start and the qreg to the args
    start_plus_args_plus_qreg = [
        start,
        *args,
        self.qreg,
    ]

    consts = plxpr_invals[consts_slice]

    def flat_fun(*args):
        jaxpr = ClosedJaxpr(jaxpr_body_fn, consts)
        return jaxpr_as_fun(jaxpr)(*args)

    def calling_convention(*args_plus_qreg):
        *args, qreg = args_plus_qreg
        device = self.device
        shots = self.shots
        converter = PLxPRToQuantumJaxprInterpreter(device, shots, qreg)
        retvals = converter(flat_fun, *args)
        converter.actualize_qreg()
        return *retvals, converter.qreg

    converted_jaxpr_branch = jax.make_jaxpr(calling_convention)(*start_plus_args_plus_qreg).jaxpr
    converted_closed_jaxpr_branch = ClosedJaxpr(convert_constvars_jaxpr(converted_jaxpr_branch), ())

    # Build Catalyst compatible input values
    for_loop_invals = [*consts, start, stop, step, *start_plus_args_plus_qreg]

    # Config additional for loop settings
    apply_reverse_transform = isinstance(step, int) and step < 0

    # Perform the binding
    outvals = for_p.bind(
        *for_loop_invals,
        body_jaxpr=converted_closed_jaxpr_branch,
        body_nconsts=len(consts),
        apply_reverse_transform=apply_reverse_transform,
        nimplicit=0,
        preserve_dimensions=True,
    )

    # We assume the last output value is the returned qreg.
    # Update the current qreg and remove it from the output values.
    self.qreg = outvals.pop()

    # Return only the output values that match the plxpr output values
    return outvals


# pylint: disable=unused-argument, too-many-arguments
@PLxPRToQuantumJaxprInterpreter.register_primitive(plxpr_while_loop_prim)
@QFuncPlxprInterpreter.register_primitive(plxpr_while_loop_prim)
def handle_while_loop(
    self,
    *plxpr_invals,
    jaxpr_body_fn,
    jaxpr_cond_fn,
    body_slice,
    cond_slice,
    args_slice,
):
    """Handle the conversion from plxpr to Catalyst jaxpr for the while loop primitive"""
    consts_body = plxpr_invals[body_slice]
    consts_cond = plxpr_invals[cond_slice]
    args = plxpr_invals[args_slice]
    args_plus_qreg = [*args, self.qreg]  # Add the qreg to the args

    def flat_fun(*args):
        jaxpr = ClosedJaxpr(jaxpr_body_fn, consts_body)
        return jaxpr_as_fun(jaxpr)(*args)

    def calling_convention(*args_plus_qreg):
        *args, qreg = args_plus_qreg
        device = self.device
        shots = self.shots
        converter = PLxPRToQuantumJaxprInterpreter(device, shots, qreg)
        retvals = converter(flat_fun, *args)
        converter.actualize_qreg()
        return *retvals, converter.qreg

    converted_body_jaxpr_branch = jax.make_jaxpr(calling_convention)(*args_plus_qreg).jaxpr
    converted_body_closed_jaxpr_branch = ClosedJaxpr(
        convert_constvars_jaxpr(converted_body_jaxpr_branch), ()
    )

    # Convert for condition from plxpr to Catalyst jaxpr
    # We need to be able to handle arbitrary plxpr here.
    # But we want to be able to create a state where:
    # * We do not pass the quantum register as an argument.

    # So let's just remove the quantum register here at the end

    def flat_fun(*args):
        jaxpr = ClosedJaxpr(jaxpr_cond_fn, consts_cond)
        return jaxpr_as_fun(jaxpr)(*args)

    def remove_qreg(*args_plus_qreg):
        *args, qreg = args_plus_qreg
        device = self.device
        shots = self.shots
        converter = PLxPRToQuantumJaxprInterpreter(device, shots, qreg)
        return converter(flat_fun, *args)

    converted_cond_jaxpr_branch = jax.make_jaxpr(remove_qreg)(*args_plus_qreg).jaxpr
    converted_cond_closed_jaxpr_branch = ClosedJaxpr(
        convert_constvars_jaxpr(converted_cond_jaxpr_branch), ()
    )

    # Build Catalyst compatible input values
    while_loop_invals = [*consts_cond, *consts_body, *args_plus_qreg]

    # Perform the binding
    outvals = while_p.bind(
        *while_loop_invals,
        cond_jaxpr=converted_cond_closed_jaxpr_branch,
        body_jaxpr=converted_body_closed_jaxpr_branch,
        cond_nconsts=len(consts_cond),
        body_nconsts=len(consts_body),
        nimplicit=0,
        preserve_dimensions=True,
    )

    # We assume the last output value is the returned qreg.
    # Update the current qreg and remove it from the output values.
    self.qreg = outvals.pop()

    # Return only the output values that match the plxpr output values
    return outvals


@PLxPRToQuantumJaxprInterpreter.register_primitive(plxpr_measure_prim)
@QFuncPlxprInterpreter.register_primitive(plxpr_measure_prim)
def handle_measure(self, wire, reset, postselect):
    """Handle the conversion from plxpr to Catalyst jaxpr for the mid-circuit measure primitive."""

    in_wire = self.get_wire(wire)

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

    self.wire_map[wire] = out_wire
    return result


# pylint: disable=unused-argument, too-many-positional-arguments
@PLxPRToQuantumJaxprInterpreter.register_primitive(plxpr_measure_in_basis_prim)
@QFuncPlxprInterpreter.register_primitive(plxpr_measure_in_basis_prim)
def handle_measure_in_basis(self, angle, wire, plane, reset, postselect):
    """Handle the conversion from plxpr to Catalyst jaxpr for the measure_in_basis primitive"""
    _angle = jax.lax.convert_element_type(angle, jnp.dtype(jnp.float64))

    try:
        _plane = MeasurementPlane(plane)
    except ValueError as e:
        raise ValueError(
            f"Measurement plane must be one of {[plane.value for plane in MeasurementPlane]}"
        ) from e

    in_wire = self.get_wire(wire)
    result, out_wire = measure_in_basis_p.bind(_angle, in_wire, plane=_plane, postselect=postselect)

    self.wire_map[wire] = out_wire

    return result


# pylint: disable=too-many-positional-arguments
def trace_from_pennylane(fn, static_argnums, abstracted_axes, sig, kwargs, debug_info=None):
    """Capture the JAX program representation (JAXPR) of the wrapped function, using
    PL capure module.

    Args:
        args (Iterable): arguments to use for program capture

    Returns:
        ClosedJaxpr: captured JAXPR
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

        plxpr, out_type, out_treedef = make_jaxpr2(fn, **make_jaxpr_kwargs)(*args, **kwargs)
        jaxpr = from_plxpr(plxpr)(*args, **kwargs)

    return jaxpr, out_type, out_treedef, sig
