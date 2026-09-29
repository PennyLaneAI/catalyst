# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Conversion of QNodes that run on PennyLane Python devices (see
:mod:`catalyst.device.python_device`).

The outcomes of mid-circuit measurements are only known once the Python device executes the tape,
so the program cannot branch on them at runtime. Instead, branches conditioned on mid-circuit
measurements are rewritten at compile time:

* every mid-circuit measurement *site* ``k`` gets a *virtual* wire ``n + k`` beyond the ``n``
  device wires. After measuring wire ``w``, a ``CNOT(w, n + k)`` binds the virtual wire to the
  outcome (deferred-measurement style);
* ``qp.cond(m, fn)()`` becomes ``fn`` controlled on the virtual wires of the measurements that
  ``m`` depends on, once per assignment of outcomes that satisfies the predicate.

The Python side of the bridge turns controls on virtual wires back into ``Conditional``
operations tied to the ``MidMeasure`` instructions of the tape, so that the device's MCM method
(deferred, one-shot, tree-traversal, ...) handles them. Since the rewritten program is itself a
valid quantum circuit, compilation passes such as ``cancel_inverses`` remain correct.
"""

from __future__ import annotations

from copy import copy
from itertools import count

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qp
from jax.extend.core import ClosedJaxpr, Jaxpr, Literal
from pennylane.capture.primitives import cond_prim, for_loop_prim
from pennylane.capture.primitives import measure_prim as plxpr_measure_prim
from pennylane.capture.primitives import while_loop_prim
from pennylane.measurements import CountsMP
from pennylane.ops.mid_measure import MeasurementValue
from pennylane.ops.mid_measure.mid_measure import get_mcm_predicates
from pennylane.wires import is_abstract_qubit

from catalyst.from_plxpr.qfunc_interpreter import PLxPRToQuantumJaxprInterpreter
from catalyst.from_plxpr.qref_jax_primitives import qref_get_p, qref_measure_p, qref_qinst_p
from catalyst.utils.exceptions import CompileError


class _McmSite:
    """A mid-circuit measurement site of the program, bound to a virtual wire."""

    def __init__(self, index: int, virtual_wire: int):
        self.index = index
        self.virtual_wire = virtual_wire
        self.meas_uid = f"{index:08d}"
        self.postselect = None


class _Opaque:
    """A value derived from mid-circuit measurements that cannot be represented symbolically."""


class _McmState:
    """Mid-circuit measurement bookkeeping shared by all copies of an interpreter."""

    def __init__(self, num_device_wires: int):
        self.num_device_wires = num_device_wires
        self._sites = count()
        # id(tracer) -> (tracer, MeasurementValue | _Opaque). The tracer is kept alive so that its
        # id is not reused.
        self._values: dict[int, tuple] = {}

    def new_site(self) -> _McmSite:
        index = next(self._sites)
        return _McmSite(index, self.num_device_wires + index)

    def mark(self, tracer, value):
        self._values[id(tracer)] = (tracer, value)

    def lookup(self, value):
        entry = self._values.get(id(value))
        if entry is not None and entry[0] is value:
            return entry[1]
        return None


def _sub_jaxprs(params):
    for value in params.values():
        for item in value if isinstance(value, (list, tuple)) else (value,):
            if isinstance(item, ClosedJaxpr):
                yield item.jaxpr
            elif isinstance(item, Jaxpr):
                yield item


def count_mcm_sites(jaxpr: Jaxpr) -> int:
    """The number of mid-circuit measurement sites in a plxpr (including nested jaxprs)."""
    total = 0
    for eqn in jaxpr.eqns:
        if eqn.primitive is plxpr_measure_prim:
            total += 1
        total += sum(count_mcm_sites(j) for j in _sub_jaxprs(eqn.params))
    return total


class _Unknown:
    """A value that is not known at compile time."""


_UNKNOWN = _Unknown()


def _static_measurement(eqn, invals):
    """The measurement process of a measurement equation with known inputs, if possible."""
    try:
        with qp.QueuingManager.stop_recording():
            if eqn.primitive is CountsMP._wires_primitive:
                with qp.capture.pause():
                    return qp.counts(
                        wires=invals or None, all_outcomes=eqn.params["all_outcomes"]
                    )
            mp = eqn.primitive.impl(*invals, **eqn.params)
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    return mp if isinstance(mp, qp.measurements.MeasurementProcess) else None


def static_measurement_plan(jaxpr: Jaxpr) -> list | None:
    """The terminal measurements of a qfunc plxpr, if they are all known at compile time.

    Returns a list of measurement processes in program order, or ``None`` if any of them depends
    on runtime values (in which case each terminal measurement is executed separately).
    """
    env = {}

    def read(var):
        if isinstance(var, Literal):
            return var.val
        return env.get(var, _UNKNOWN)

    plan = []
    for eqn in jaxpr.eqns:
        invals = [read(v) for v in eqn.invars]
        known = not any(v is _UNKNOWN or qp.math.is_abstract(v) for v in invals)
        prim_type = getattr(eqn.primitive, "prim_type", "")
        outvals = [_UNKNOWN] * len(eqn.outvars)

        if prim_type == "measurement":
            if not known:
                return None
            mp = _static_measurement(eqn, invals)
            if mp is None:
                return None
            plan.append(mp)
        elif prim_type == "operator" and known:
            if not isinstance(eqn.outvars[0], jax.core.DropVar):
                with qp.QueuingManager.stop_recording():
                    outvals = [eqn.primitive.impl(*invals, **eqn.params)]
        elif known and not prim_type and not any(True for _ in _sub_jaxprs(eqn.params)):
            try:
                with jax.ensure_compile_time_eval():
                    res = eqn.primitive.bind(*invals, **eqn.params)
                outvals = res if eqn.primitive.multiple_results else [res]
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        for var, val in zip(eqn.outvars, outvals):
            env[var] = val
    return plan


def _lift(eqn, invals, symbols) -> list:
    """Evaluate a classical equation on values derived from mid-circuit measurements.

    Returns one ``MeasurementValue`` (or ``_Opaque``) per output of the equation.
    """
    n_out = len(eqn.outvars)
    if any(isinstance(s, _Opaque) for s in symbols) or any(
        qp.math.is_abstract(v) for v, s in zip(invals, symbols) if s is None
    ):
        return [_Opaque()] * n_out

    sites = []
    for s in symbols:
        if s is not None:
            sites.extend(m for m in s.measurements if m not in sites)
    sites.sort(key=lambda m: m.meas_uid)

    subfuns, params = eqn.primitive.get_bind_params(eqn.params)

    def evaluate(bits):
        assignment = dict(zip(sites, bits))
        args = [
            np.asarray(s.concretize(assignment)) if s is not None else v
            for v, s in zip(invals, symbols)
        ]
        with jax.ensure_compile_time_eval():
            out = eqn.primitive.bind(*subfuns, *args, **params)
        return [np.asarray(o).item() for o in (out if eqn.primitive.multiple_results else [out])]

    try:
        evaluate((0,) * len(sites))
    except Exception:  # pylint: disable=broad-exception-caught
        return [_Opaque()] * n_out

    return [
        MeasurementValue(sites, lambda *bits, _i=i: evaluate(bits)[_i]) for i in range(n_out)
    ]


def _unsupported(what):
    return CompileError(
        f"{what} depends on the outcome of a mid-circuit measurement, which is not supported on "
        "Python devices under qjit: the outcomes are only known to the device when it executes "
        "the tape. Only branches conditioned on mid-circuit measurements (qp.cond) are supported."
    )


_CONTROL_FLOW = (cond_prim, for_loop_prim, while_loop_prim, plxpr_measure_prim)


class PythonDeviceQfuncInterpreter(PLxPRToQuantumJaxprInterpreter):
    """Converts the qfunc of a QNode that runs on a Python device (see the module docstring)."""

    def __init__(self, *args, num_device_wires: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.mcm = _McmState(num_device_wires)

    # pylint: disable=too-many-branches
    def eval(self, jaxpr, consts, *args):
        self._env = {}
        self.setup()

        for arg, invar in zip(args, jaxpr.invars, strict=True):
            self._env[invar] = arg
        for const, constvar in zip(consts, jaxpr.constvars, strict=True):
            self._env[constvar] = const

        for eqn in jaxpr.eqns:
            primitive = eqn.primitive
            custom_handler = self._primitive_registrations.get(primitive, None)
            invals = [self.read(invar) for invar in eqn.invars]
            symbols = [self.mcm.lookup(v) for v in invals]
            has_symbols = any(s is not None for s in symbols)
            prim_type = getattr(primitive, "prim_type", "")

            if has_symbols and primitive in (for_loop_prim, while_loop_prim):
                args_slice = slice(*eqn.params["args_slice"])
                if any(s is not None for s in symbols[args_slice]):
                    raise _unsupported("A loop argument")
                if primitive is for_loop_prim and any(s is not None for s in symbols[:3]):
                    raise _unsupported("A loop bound")
            if has_symbols and prim_type == "operator":
                raise _unsupported("A gate parameter or wire")
            if has_symbols and prim_type == "measurement":
                raise _unsupported("A terminal measurement")

            if custom_handler:
                outvals = custom_handler(self, *invals, **eqn.params)
            elif prim_type == "operator":
                outvals = self.interpret_operation_eqn(eqn)
            elif prim_type == "measurement":
                outvals = self.interpret_measurement_eqn(eqn)
            else:
                subfuns, params = primitive.get_bind_params(eqn.params)
                outvals = primitive.bind(*subfuns, *invals, **params)

            if not primitive.multiple_results:
                outvals = [outvals]
            if has_symbols and not prim_type and primitive not in _CONTROL_FLOW:
                for outval, symbol in zip(outvals, _lift(eqn, invals, symbols)):
                    self.mcm.mark(outval, symbol)
            for outvar, outval in zip(eqn.outvars, outvals, strict=True):
                self._env[outvar] = outval

        outvals = []
        for var in jaxpr.outvars:
            outval = self.read(var)
            if self.mcm.lookup(outval) is not None:
                raise _unsupported("A value returned by the quantum function")
            if isinstance(outval, qp.operation.Operator):
                outvals.append(self.interpret_operation(outval))
            else:
                outvals.append(outval)
        self.cleanup()
        self._env = {}
        return outvals

    def qubit(self, wire):
        """The qubit of a (device or virtual) wire."""
        return wire if is_abstract_qubit(wire) else qref_get_p.bind(self.init_qreg, wire)


@PythonDeviceQfuncInterpreter.register_primitive(plxpr_measure_prim)
def _handle_measure(self, wire, reset, postselect):
    """Record the measurement and bind the outcome to the virtual wire of this site."""
    if self.control_wires:
        raise CompileError(
            "Mid-circuit measurements inside branches conditioned on mid-circuit measurements or "
            "controlled regions are not supported on Python devices."
        )
    in_qubit = self.qubit(wire)
    result = qref_measure_p.bind(in_qubit, postselect=postselect)

    site = self.mcm.new_site()
    virtual = self.qubit(site.virtual_wire)
    cnot = {"op": "CNOT", "qubits_len": 2, "params_len": 0, "ctrl_len": 0, "adjoint": False}
    qref_qinst_p.bind(in_qubit, virtual, **cnot)
    if reset:
        # ``CNOT(virtual, measured)`` right after the binding encodes the reset
        qref_qinst_p.bind(virtual, in_qubit, **cnot)

    outcome = jnp.astype(result, int)
    self.mcm.mark(outcome, MeasurementValue([site]))
    return outcome


@PythonDeviceQfuncInterpreter.register_primitive(cond_prim)
def _handle_cond(self, *invals, jaxpr_branches, consts_slices, args_slice):
    """Branches conditioned on mid-circuit measurements become controlled on virtual wires."""
    n_preds = len(jaxpr_branches) - 1
    symbols = [self.mcm.lookup(p) for p in invals[:n_preds]]
    base = PLxPRToQuantumJaxprInterpreter._primitive_registrations[cond_prim]

    if all(s is None for s in symbols):
        if any(self.mcm.lookup(v) is not None for v in invals[slice(*args_slice)]):
            raise _unsupported("A branch argument")
        return base(
            self,
            *invals,
            jaxpr_branches=jaxpr_branches,
            consts_slices=consts_slices,
            args_slice=args_slice,
        )

    if any(s is None or isinstance(s, _Opaque) for s in symbols):
        raise _unsupported("A branch predicate that mixes classical values and measurements")

    args = invals[slice(*args_slice)]
    predicates = get_mcm_predicates(tuple(symbols))
    for predicate, jaxpr, const_slice in zip(predicates, jaxpr_branches, consts_slices, strict=True):
        if jaxpr.outvars:
            raise CompileError(
                "Branches conditioned on mid-circuit measurements cannot return values on Python "
                "devices."
            )
        consts = invals[slice(*const_slice)]
        ctrl_wires = tuple(site.virtual_wire for site in predicate.measurements)
        for bits, value in predicate.items():
            if not value:
                continue
            branch = copy(self)
            branch.control_wires = self.control_wires + ctrl_wires
            branch.control_values = self.control_values + tuple(bool(b) for b in bits)
            branch.eval(jaxpr, consts, *args)
    return []
