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
from functools import partial
from typing import Callable

import jax
from jax.extend.linear_util import wrap_init
from pennylane.capture import (
    AbstractMeasurement,
    AbstractOperator,
    qnode_prim,
    enable,
    enabled,
    disable,
)

from catalyst.device import extract_backend_info, get_device_capabilities
from catalyst.jax_primitives import (
    AbstractQbit,
    AbstractQreg,
    compbasis_p,
    expval_p,
    func_p,
    gphase_p,
    namedobs_p,
    probs_p,
    qalloc_p,
    qdealloc_p,
    qdevice_p,
    qextract_p,
    qinsert_p,
    qinst_p,
    qunitary_p,
    sample_p,
    state_p,
    var_p,
)
from catalyst.utils.toml import ProgramFeatures
from catalyst.jax_extras import make_jaxpr2, transient_jax_config

measurement_map = {
    "sample_wires": sample_p,
    "expval_obs": expval_p,
    "var_obs": var_p,
    "probs_wires": probs_p,
    "state_wires": state_p,
}


def _get_shapes_for(*measurements, shots=None, num_device_wires=0):
    shapes = []
    if not shots:
        shots = [None]

    for s in shots:
        for m in measurements:
            shape, dtype = m.abstract_eval(shots=s, num_device_wires=num_device_wires)
            shapes.append(jax.core.ShapedArray(shape, dtype))
    return shapes


# pylint: disable=unidiomatic-typecheck
def _read(var, env: dict):
    return var.val if type(var) is jax.core.Literal else env[var]


def _get_device_kwargs(device: "pennylane.devices.Device") -> dict:
    """Calulcate the params for a device equation."""
    features = ProgramFeatures(device.shots is not None)
    capabilities = get_device_capabilities(device, features)
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
def from_plxpr(plxpr: jax.core.Jaxpr) -> Callable[..., jax.core.Jaxpr]:
    """Convert PennyLane variant jaxpr to Catalyst variant jaxpr.

    Args:
        jaxpr (jax.core.Jaxpr): PennyLane variant jaxpr

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
            fn=<QNode: device='<lightning.qubit device (wires=2) at 0x302761c90>', interface='auto', diff_method='best'>
            ] a
        in (b,) }

    """
    return jax.make_jaxpr(partial(from_plxpr_interpreter, plxpr.jaxpr, plxpr.consts))


# docstring link too long
# pylint: disable=line-too-long
def from_plxpr_interpreter(jaxpr: jax.core.Jaxpr, consts, *args) -> list:
    """Convert PennyLane variant jaxpr to Catalyst variant jaxpr.

    See the documentation on
    `Writing custom interpreters in JAX <https://jax.readthedocs.io/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html>`_
    for a walkthrough on the general architecture and behavior of this function.

    Given that ``catalyst.jax_primitives.func_p`` does not define a concrete implementation, this
    function will fail outside of an abstract evaluation call.

    """
    env = {invar: arg for arg, invar in zip(args, jaxpr.invars)}

    for const, constvar in zip(consts, jaxpr.constvars):
        env[constvar] = const

    # Loop through equations and evaluate primitives using `bind`
    for eqn in jaxpr.eqns:
        # Read inputs to equation from environment
        invals = [_read(invar, env) for invar in eqn.invars]
        if eqn.primitive == qnode_prim:
            if eqn.params["device"].shots != eqn.params["shots"]:
                raise NotImplementedError("catalyst does not yet support dynamic shots")

            f = partial(
                QFuncPlxprInterpreter(eqn.params["device"]).convert,
                eqn.params["qfunc_jaxpr"],
                n_consts=eqn.params["n_consts"],
            )
            # func_p is a CallPrimitive, so interpreter passed as first arg
            # wrap_init turns the function into a WrappedFun, which can store
            # transformations
            outvals = func_p.bind(wrap_init(f), *invals, fn=eqn.params["qnode"])
        else:
            outvals = eqn.primitive.bind(*invals, **eqn.params)
        # Primitives may return multiple outputs or not
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        # Write the results of the primitive into the environment
        for outvar, outval in zip(eqn.outvars, outvals):
            env[outvar] = outval
    return [env[outvar] for outvar in jaxpr.outvars]


class QFuncPlxprInterpreter:
    """This dataclass stores the mutable variables modified
    over the course of interpreting the plxpr as catalxpr."""

    qreg: AbstractQreg
    """The current quantum register."""

    env: dict
    """A dictionary mapping variables to values."""

    wire_map: dict
    """A map from wire values to ``AbstractQbit`` instances.
    
    If a value is not present in this dictionary, it needs to be extracted
    from the ``qreg`` property.
    """

    op_math_cache: dict
    """A cache of operations that will be consumed by later operations.
    This is a map from the ``AbstractOperator`` variables to the corresponding
    equation. The equation will need to be interpreted when the abstract
    operator is consumed.
    """

    def __init__(self, device):
        self._device = device
        self.qreg = None
        self.env = {}
        self.wire_map = {}
        self.op_math_cache = {}

    def setup(self):
        """Perform any customized setup and processing before processing the plxpr.

        For conversion to catalyst, this allocates the device, extracts a register, and
        resets the wire map.
        """
        qdevice_p.bind(**_get_device_kwargs(self._device))
        self.qreg = qalloc_p.bind(len(self._device.wires))
        self.wire_map = {}

    def cleanup(self):
        """Perform any final steps after processing the plxpr.

        For conversion to calayst, this reinserts extracted qubits and
        deallocates the register.
        """
        for orig_wire, wire in self.wire_map.items():
            self.qreg = qinsert_p.bind(self.qreg, orig_wire, wire)
        qdealloc_p.bind(self.qreg)

    def read(self, var):
        """Extract the value corresponding to a variable."""
        return var.val if type(var) is jax.core.Literal else self.env[var]

    def _get_wire(self, wire_value) -> AbstractQbit:
        """Get the ``AbstractQbit`` corresponding to a wire value."""
        if wire_value in self.wire_map:
            return self.wire_map[wire_value]
        return qextract_p.bind(self.qreg, wire_value)

    def interpret_operator_eqn(self, eqn: jax.core.JaxprEqn) -> None:
        """Interpret a plxpr equation describing an operation as a catalxpr equation."""
        if not isinstance(eqn.outvars[0], jax.core.DropVar):
            self.op_math_cache[eqn.outvars[0]] = eqn
            return

        if "n_wires" not in eqn.params:
            raise NotImplementedError(
                f"Operator {eqn.primitive.name} not yet supported for catalyst conversion."
            )
        n_wires = eqn.params["n_wires"]
        if n_wires == 0:
            wires = []
            wire_values = []
            invals = [self.read(invar) for invar in eqn.invars]
        else:
            wire_values = [self.read(w) for w in eqn.invars[-n_wires:]]
            wires = [self._get_wire(w) for w in wire_values]
            invals = [self.read(invar) for invar in eqn.invars[:-n_wires]]

        kwargs = {
            "qubits_len": eqn.params["n_wires"],
            "ctrl_len": 0,
            "adjoint": False,
        }

        if eqn.primitive.name == "QubitUnitary":
            outvals = qunitary_p.bind(*invals, *wires, **kwargs)
        elif eqn.primitive.name == "GlobalPhase":
            outvals = gphase_p.bind(*invals, ctrl_len=0, adjoint=False)
        else:
            outvals = qinst_p.bind(
                *wires,
                *invals,
                op=eqn.primitive.name,
                params_len=len(eqn.invars) - eqn.params["n_wires"],
                **kwargs,
            )

        for wire_values, new_wire in zip(wire_values, outvals):
            self.wire_map[wire_values] = new_wire

    def _obs(self, eqn: jax.core.JaxprEqn):
        """Interpret the observable equation corresponding to a measurement equation's input."""
        obs_eqn = self.op_math_cache[eqn.invars[0]]
        if "n_wires" not in obs_eqn.params:
            raise NotImplementedError(
                f"from_plxpr can not yet interpret observables of type {obs_eqn.primitive}"
            )

        n_wires = obs_eqn.params["n_wires"]
        wires = [self._get_wire(self.read(w)) for w in obs_eqn.invars[-n_wires:]]
        invals = [self.read(invar) for invar in obs_eqn.invars[:-n_wires]]
        return namedobs_p.bind(*wires, *invals, kind=obs_eqn.primitive.name)

    def _compbasis_obs(self, eqn: jax.core.JaxprEqn):
        """Add a computational basis sampling observable."""
        if eqn.invars:
            w_vals = [self.read(w_var) for w_var in eqn.invars]
        else:
            w_vals = self._device.wires  # broadcast across all wires
        wires = [self._get_wire(w) for w in w_vals]
        return compbasis_p.bind(*wires)

    def interpret_measurement_eqn(self, eqn: jax.core.JaxprEqn):
        """Interpret a measurement equation as a catalyst equation."""
        if eqn.primitive.name not in measurement_map:
            raise NotImplementedError(
                f"measurement {eqn.primitive.name} not yet supported for conversion."
            )
        if eqn.params.get("has_eigvals", False):
            raise NotImplementedError(
                "from_plxpr does not yet support measurements with eigenvalues."
            )

        if "_wires" in eqn.primitive.name:
            obs = self._compbasis_obs(eqn)
        else:
            obs = self._obs(eqn)
        # mcm based measurements wont be in measurement map yet
        # so we can assume observable based

        shaped_array = _get_shapes_for(
            eqn.outvars[0].aval, shots=self._device.shots, num_device_wires=len(self._device.wires)
        )[0]

        primitive = measurement_map[eqn.primitive.name]
        mval = primitive.bind(obs, shape=shaped_array.shape, shots=self._device.shots.total_shots)

        # sample_p returns floats, so we need to converted it back to the expected integers here
        if shaped_array.dtype != mval.dtype:
            return jax.lax.convert_element_type(mval, shaped_array.dtype)
        return mval

    def convert(self, jaxpr: jax.core.Jaxpr, *args, n_consts=0) -> list:
        """Interpret plxpr as catalxpr."""

        consts = args[:n_consts]
        args = args[n_consts:]
        self.setup()

        self.op_math_cache = {}
        self.env = {invar: arg for arg, invar in zip(args, jaxpr.invars)}
        for const, constvar in zip(consts, jaxpr.constvars):
            self.env[constvar] = const

        for eqn in jaxpr.eqns:
            if isinstance(eqn.outvars[0].aval, AbstractOperator):
                self.interpret_operator_eqn(eqn)
            elif isinstance(eqn.outvars[0].aval, AbstractMeasurement):
                outval = self.interpret_measurement_eqn(eqn)
                self.env[eqn.outvars[0]] = outval
            else:
                invals = [self.read(invar) for invar in eqn.invars]
                outvals = eqn.primitive.bind(*invals, **eqn.params)
                if not eqn.primitive.multiple_results:
                    outvals = [outvals]
                for outvar, outval in zip(eqn.outvars, outvals):
                    self.env[outvar] = outval

        self.cleanup()
        # Read the final result of the Jaxpr from the environment
        return [self.read(outvar) for outvar in jaxpr.outvars]


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
        plxpr, out_type, out_treedef = make_jaxpr2(fn, **make_jaxpr_kwargs)(*args, **kwargs)

        if not capture_on:
            disable()

        jaxpr = from_plxpr(plxpr)(*args, **kwargs)
    return jaxpr, out_type, out_treedef, sig
