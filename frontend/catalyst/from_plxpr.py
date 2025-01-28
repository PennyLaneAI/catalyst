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
from typing import Callable, Sequence

import jax
import jax.core
import pennylane as qml
from jax.extend.linear_util import wrap_init
from pennylane.capture import PlxprInterpreter, disable, enable, enabled, qnode_prim
from pennylane.capture.primitives import (
    AbstractMeasurement,
    AbstractOperator,
    cond_prim,
)

from catalyst.device import (
    extract_backend_info,
    get_device_capabilities,
    get_device_shots,
)
from catalyst.jax_extras import jaxpr_pad_consts, make_jaxpr2, transient_jax_config
from catalyst.jax_extras.tracing import bind_flexible_primitive
from catalyst.jax_primitives import (
    AbstractQbit,
    compbasis_p,
    cond_p,
    counts_p,
    expval_p,
    gphase_p,
    namedobs_p,
    probs_p,
    qalloc_p,
    qdealloc_p,
    qdevice_p,
    qextract_p,
    qinsert_p,
    qinst_p,
    quantum_kernel_p,
    qunitary_p,
    sample_p,
    state_p,
    var_p,
)

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
def from_plxpr(plxpr: jax.core.ClosedJaxpr) -> Callable[..., jax.core.Jaxpr]:
    """Convert PennyLane variant jaxpr to Catalyst variant jaxpr.

    Args:
        jaxpr (jax.core.ClosedJaxpr): PennyLane variant jaxpr

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
                qdevice[
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


# pylint: disable=unused-argument, too-many-arguments
@WorkflowInterpreter.register_primitive(qnode_prim)
def _(self, *args, qnode, shots, device, qnode_kwargs, qfunc_jaxpr, n_consts, batch_dims=None):
    consts = args[:n_consts]
    non_const_args = args[n_consts:]

    f = partial(QFuncPlxprInterpreter(device, shots).eval, qfunc_jaxpr, consts)

    return quantum_kernel_p.bind(wrap_init(f), *non_const_args, qnode=qnode)


class QFuncPlxprInterpreter(PlxprInterpreter):
    """An interpreter that converts plxpr into catalyst-variant jaxpr.

    Args:
        device (qml.devices.Device)
        shots (qml.measurements.Shots)

    """

    def __init__(self, device, shots: qml.measurements.Shots):
        self._device = device
        self._shots = shots.total_shots if shots else 0
        self.stateref = None
        super().__init__()

    def __getattr__(self, key):
        if key in {"qreg", "wire_map"}:
            if self.stateref is None:
                raise AttributeError("execution is not yet initialized.")
            return self.stateref[key]
        raise AttributeError(f"no attribute {key}")

    def __setattr__(self, __name: str, __value) -> None:
        if __name in {"qreg", "wire_map"}:
            if self.stateref is None:
                raise AttributeError("execution is not yet initialized.")
            self.stateref[__name] = __value
        else:
            super().__setattr__(__name, __value)

    def setup(self):
        """Initialize the stateref and bind the device."""
        if self.stateref is None:
            qdevice_p.bind(self._shots, **_get_device_kwargs(self._device))
            self.stateref = {"qreg": qalloc_p.bind(len(self._device.wires)), "wire_map": {}}

    # pylint: disable=attribute-defined-outside-init
    def cleanup(self):
        """Perform any final steps after processing the plxpr.

        For conversion to calayst, this reinserts extracted qubits and
        deallocates the register.
        """
        for orig_wire, wire in self.wire_map.items():
            self.qreg = qinsert_p.bind(self.qreg, orig_wire, wire)
        qdealloc_p.bind(self.qreg)
        self.stateref = None

    def get_wire(self, wire_value) -> AbstractQbit:
        """Get the ``AbstractQbit`` corresponding to a wire value."""
        if wire_value in self.wire_map:
            return self.wire_map[wire_value]
        return qextract_p.bind(self.qreg, wire_value)

    def interpret_operation(self, op):
        """Re-bind a pennylane operation as a catalyst instruction."""

        in_qubits = [self.get_wire(w) for w in op.wires]
        out_qubits = qinst_p.bind(
            *in_qubits,
            *op.data,
            op=op.name,
            ctrl_value_len=0,
            ctrl_len=0,
            qubits_len=len(op.wires),
            adjoint=False,
        )
        for wire_values, new_wire in zip(op.wires, out_qubits):
            self.wire_map[wire_values] = new_wire

    def _obs(self, obs):
        """Interpret the observable equation corresponding to a measurement equation's input."""
        if obs.arithmetic_depth > 0:
            raise NotImplementedError("operator arithmetic not yet supported for conversion.")
        wires = [self.get_wire(w) for w in obs.wires]
        return namedobs_p.bind(*wires, *obs.data, kind=obs.name)

    def _compbasis_obs(self, *wires):
        """Add a computational basis sampling observable."""
        wires = wires or self._device.wires  # broadcast across all wires
        qubits = [self.get_wire(w) for w in wires]
        return compbasis_p.bind(*qubits)

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
            shots=self._device.shots.total_shots,
            num_device_wires=len(self._device.wires),
        )

        prim = measurement_map[type(measurement)]
        if prim is sample_p:
            num_qubits = len(measurement.wires) or len(self._device.wires)
            mval = bind_flexible_primitive(
                sample_p, {"shots": self._shots}, obs, num_qubits=num_qubits
            )
        elif prim is counts_p:
            mval = bind_flexible_primitive(counts_p, {"shots": self._shots}, shape=shape)
        elif prim in {expval_p, var_p}:
            mval = prim.bind(obs, shape=shape)
        else:
            mval = prim.bind(obs, shape=shape, shots=self._shots)

        # sample_p returns floats, so we need to converted it back to the expected integers here
        if dtype != mval.dtype:
            return jax.lax.convert_element_type(mval, dtype)
        return mval

    def eval(self, jaxpr: "jax.core.Jaxpr", consts: Sequence, *args) -> list:
        """Evaluate a jaxpr.

        Args:
            jaxpr (jax.core.Jaxpr): the jaxpr to evaluate
            consts (list[TensorLike]): the constant variables for the jaxpr
            *args (tuple[TensorLike]): The arguments for the jaxpr.

        Returns:
            list[TensorLike]: the results of the execution.

        """
        self._env = {}
        self.setup()

        for arg, invar in zip(args, jaxpr.invars, strict=True):
            self._env[invar] = arg
        for const, constvar in zip(consts, jaxpr.constvars, strict=True):
            self._env[constvar] = const

        for eqn in jaxpr.eqns:

            custom_handler = self._primitive_registrations.get(eqn.primitive, None)
            if custom_handler:
                invals = [self.read(invar) for invar in eqn.invars]
                outvals = custom_handler(self, *invals, **eqn.params)
            elif isinstance(eqn.outvars[0].aval, AbstractOperator):
                outvals = self.interpret_operation_eqn(eqn)
            elif isinstance(eqn.outvars[0].aval, AbstractMeasurement):
                outvals = self.interpret_measurement_eqn(eqn)
            else:
                invals = [self.read(invar) for invar in eqn.invars]
                outvals = eqn.primitive.bind(*invals, **eqn.params)

            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            for outvar, outval in zip(eqn.outvars, outvals, strict=True):
                self._env[outvar] = outval

        # Read the final result of the Jaxpr from the environment
        outvals = []
        for var in jaxpr.outvars:
            outval = self.read(var)
            if isinstance(outval, qml.operation.Operator):
                outvals.append(self.interpret_operation(outval))
            else:
                outvals.append(outval)
        self.cleanup()
        self._env = {}
        return outvals


class BranchPlxprInterpreter(QFuncPlxprInterpreter):
    """An interpreter that converts a plxpr branch into catalyst-variant jaxpr branch.

    Args:
        parent_device (qml.devices.Device)
        parent_shots (qml.measurements.Shots)
        parent_qreg (...)

    """

    def __init__(self, parent_device, parent_shots, parent_qreg):
        self._parent_qreg = parent_qreg
        super().__init__(parent_device, parent_shots)

    def setup(self):
        """Initialize the stateref."""
        if self.stateref is None:
            self.stateref = {"qreg": self._parent_qreg, "wire_map": {}}

    # pylint: disable=attribute-defined-outside-init
    def cleanup(self):
        """Perform any final steps after processing the plxpr.

        For conversion to catalyst, this reinserts extracted qubits.
        """
        for orig_wire, wire in self.wire_map.items():
            self.qreg = qinsert_p.bind(self.qreg, orig_wire, wire)
        self.stateref = None


@QFuncPlxprInterpreter.register_primitive(qml.QubitUnitary._primitive)
def _(self, *invals, n_wires):
    wires = [self.get_wire(w) for w in invals[1:]]
    outvals = qunitary_p.bind(invals[0], *wires, qubits_len=n_wires, ctrl_len=0, adjoint=False)
    for wire_values, new_wire in zip(invals[1:], outvals):
        self.wire_map[wire_values] = new_wire


# pylint: disable=unused-argument
@QFuncPlxprInterpreter.register_primitive(qml.GlobalPhase._primitive)
def _(self, phase, *wires, n_wires):
    gphase_p.bind(phase, ctrl_len=0, adjoint=False)


# pylint: disable=unused-argument, too-many-arguments
@QFuncPlxprInterpreter.register_primitive(cond_prim)
def _(self, *plxpr_invals, jaxpr_branches, consts_slices, args_slice):
    args = plxpr_invals[args_slice]
    branches_consts = [plxpr_invals[consts_slice] for consts_slice in consts_slices]
    converted_jaxpr_branches = []
    qreg_var = None

    for branch_consts, plxpr_branch in zip(branches_consts, jaxpr_branches):

        if plxpr_branch is None:
            converted_jaxpr_branches.append(None)
        else:
            converted_func = partial(
                BranchPlxprInterpreter(self._device, self._shots, self.qreg).eval,
                plxpr_branch,
                branch_consts,
            )
            converted_jaxpr_branch = jax.make_jaxpr(converted_func)(*args).jaxpr
            qreg_var = converted_jaxpr_branch.constvars[0]
            invars = []
            invars.append(converted_jaxpr_branch.constvars[0])
            converted_jaxpr_branch = converted_jaxpr_branch.replace(invars=invars)
            outvars = []
            outvars.append(copy(converted_jaxpr_branch.constvars[0]))
            converted_jaxpr_branch = converted_jaxpr_branch.replace(outvars=outvars)
            constvars = converted_jaxpr_branch.constvars[1:]
            converted_jaxpr_branch = converted_jaxpr_branch.replace(constvars=constvars)
            converted_jaxpr_branch.eqns[len(converted_jaxpr_branch.eqns) - 1] = (
                converted_jaxpr_branch.eqns[len(converted_jaxpr_branch.eqns) - 1].replace(
                    outvars=outvars
                )
            )
            converted_jaxpr_branches.append(converted_jaxpr_branch)

    # Rebuild the invals with the Catalyst jaxpr spec:
    predicate_slice = slice(0, 1)
    predicate = plxpr_invals[predicate_slice]
    consts = [const for consts in branches_consts for const in consts]
    cond_invals = [*predicate, *consts, *args, self.qreg]

    outvars = [qreg_var]
    outvals = cond_p.bind(
        *cond_invals,
        branch_jaxprs=jaxpr_pad_consts(converted_jaxpr_branches),
        nimplicit_outputs=None,
    )

    for outvar, outval in zip(outvars, outvals, strict=True):
        self.qreg = outval
        self._env[outvar] = outval

    return ()


def trace_from_pennylane(fn, static_argnums, abstracted_axes, sig, kwargs):
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
        }

        if enabled():
            capture_on = True
        else:
            capture_on = False
            enable()

        args = sig
        try:
            plxpr, out_type, out_treedef = make_jaxpr2(fn, **make_jaxpr_kwargs)(*args, **kwargs)
            jaxpr = from_plxpr(plxpr)(*args, **kwargs)
        finally:
            if not capture_on:
                disable()

    return jaxpr, out_type, out_treedef, sig
