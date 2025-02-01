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
    AbstractQreg,
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


class BranchPlxprInterpreter(QFuncPlxprInterpreter):
    """An interpreter that converts a plxpr branch into catalyst-variant jaxpr branch.

    Args:
        parent_device (qml.devices.Device)
        parent_shots (qml.measurements.Shots)
        parent_qreg (...)

    """

    def set_qreg(self, qreg):
        """Initialize the stateref."""
        if self.stateref is None:
            self.stateref = {"qreg": qreg, "wire_map": {}}

    # pylint: disable=attribute-defined-outside-init
    def cleanup(self):
        """Perform any final steps after processing the plxpr.

        For conversion to catalyst, this reinserts extracted qubits.
        """
        for orig_wire, wire in self.wire_map.items():
            self.qreg = qinsert_p.bind(self.qreg, orig_wire, wire)
        self.stateref = None

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

        num_args = len(args)
        assert num_args > 0
        qreg = args[num_args - 1]

        self.set_qreg(qreg)

        for const, constvar in zip(consts, jaxpr.constvars, strict=True):
            self._env[constvar] = const

        for eqn in jaxpr.eqns:

            custom_handler = self._primitive_registrations.get(eqn.primitive, None)
            if custom_handler:
                invals = [self.read(invar) for invar in eqn.invars]
                custom_handler(self, *invals, **eqn.params)
            elif isinstance(eqn.outvars[0].aval, AbstractOperator):
                self.interpret_operation_eqn(eqn)
            elif isinstance(eqn.outvars[0].aval, AbstractMeasurement):
                self.interpret_measurement_eqn(eqn)
            else:
                invals = [self.read(invar) for invar in eqn.invars]
                eqn.primitive.bind(*invals, **eqn.params)

        outvals = [self.qreg]

        self.cleanup()
        self._env = {}

        return outvals


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

    # Inner functions

    def emit_new_abstract_qreg_jaxpr():
        """Emit a new Catalyst jaxpr branch that returns a qreg."""

        return jax.make_jaxpr(lambda x: x)(AbstractQreg()).jaxpr

    def convert_branch_from_plxpr_to_jaxpr():
        # Convert plxpr to Catalyst jaxpr
        converted_func = partial(
            BranchPlxprInterpreter(self._device, self._shots).eval,
            plxpr_branch,
            branch_consts,
        )
        args_plus_qreg = [*args, self.qreg]
        converted_jaxpr_branch = jax.make_jaxpr(converted_func)(*args_plus_qreg).jaxpr

        # We assume the last equation of the branch is the one returning the qreg.
        # Its output vars are outdated, so we update them with the output var qreg.
        num_eqns = len(converted_jaxpr_branch.eqns)
        if num_eqns > 1:
            last_eqn = num_eqns - 1
            converted_jaxpr_branch.eqns[last_eqn] = converted_jaxpr_branch.eqns[last_eqn].replace(
                outvars=converted_jaxpr_branch.outvars
            )

        return converted_jaxpr_branch

    def restructure_input_values():
        """The original vals do not comply as well with the Catalyst spec. We need to rebuild them accordingly."""

        # Extract the predicate
        predicate_slice = slice(0, 1)
        predicate = plxpr_invals[predicate_slice]

        # Flatten the constants of all branches
        consts = [const for consts in branches_consts for const in consts]
        return [*predicate, *consts, *args, self.qreg]

    # Logic

    args = plxpr_invals[args_slice]
    branches_consts = [plxpr_invals[consts_slice] for consts_slice in consts_slices]
    converted_jaxpr_branches = []

    # Convert each branch from plxpr to jaxpr
    for branch_consts, plxpr_branch in zip(branches_consts, jaxpr_branches):
        converted_jaxpr_branches.append(
            emit_new_abstract_qreg_jaxpr()
            if plxpr_branch is None
            else convert_branch_from_plxpr_to_jaxpr()
        )

    cond_invals = restructure_input_values()

    # Perform the binding
    outvals = cond_p.bind(
        *cond_invals,
        branch_jaxprs=jaxpr_pad_consts(converted_jaxpr_branches),
        nimplicit_outputs=None,
    )

    # We assume the first output value is the returned qreg.
    # Update the current qreg.
    assert len(outvals) > 0
    self.qreg = outvals[0]

    # Return an empty list to skip post-processing
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
