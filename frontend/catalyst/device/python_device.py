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

"""Middle layer between PennyLane Python devices and the Catalyst runtime.

PennyLane devices that are implemented in Python (i.e. that do not provide a C interface through
``get_c_interface``) can be used with ``qjit``. The program still goes through the full
compilation pipeline (Python -> jaxpr -> MLIR -> LLVM -> binary), but its QNodes are bound to
the ``PLPythonDevice`` runtime backend (``runtime/lib/backend/pennylane_python``). At execution
time that backend streams every quantum instruction back into this module, which records it on a
:class:`~.QuantumScript`. When the first terminal measurement is requested, the tape is completed
with all terminal measurements of the QNode, handed to the device through the regular PennyLane
execution pipeline (device preprocessing, MCM method transforms, ``split_non_commuting``, ...),
and the results are streamed back to the runtime.

This module is not user facing. A Python device only needs to implement the regular PennyLane
:class:`~.devices.Device` API; it may provide a TOML file (``config_filepath``) that declares the
gates, observables, measurement processes, MCM methods and dynamic wire allocation it supports.

Developers can intercept the tapes that the runtime generates with
:func:`intercept_runtime_tape` (also available as ``catalyst.debug.intercept_runtime_tape``).
"""

from __future__ import annotations

import ast
import contextlib
import contextvars
import itertools
import json
import os
import platform
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pennylane as qp
from pennylane.devices.capabilities import DeviceCapabilities
from pennylane.ops import MidMeasure
from pennylane.ops.mid_measure import MeasurementValue
from pennylane.tape import QuantumScript

from catalyst.utils.exceptions import CompileError
from catalyst.utils.runtime_environment import get_lib_path

__all__ = ("intercept_runtime_tape", "InterceptedTape", "is_python_device")

RUNTIME_DEVICE_NAME = "PLPythonDevice"
RUNTIME_LIBRARY = "librtd_pennylane_python"
BRIDGE_TOML = "pennylane_python.toml"

# Mid-circuit measurement methods that PennyLane applies on the Python side of the bridge.
_WORKFLOW_MCM_METHODS = ("deferred", "one-shot", "tree-traversal")


######################################################
### Compile time: detection, capabilities and backend
######################################################


def is_python_device(device) -> bool:
    """Whether ``device`` is executed through the PennyLane Python device bridge.

    This is the case for devices implementing the PennyLane :class:`~.devices.Device` API that do
    not provide a C interface for the Catalyst runtime.
    """
    # pylint: disable-next=import-outside-toplevel
    from catalyst.device.qjit_device import SUPPORTED_RT_DEVICES, QJITDevice

    if isinstance(device, (QJITDevice, qp.devices.LegacyDeviceFacade)):
        return False
    if not isinstance(device, qp.devices.Device):
        return False
    if hasattr(device, "get_c_interface"):
        return False
    return device.name not in SUPPORTED_RT_DEVICES


def _bridge_toml_path() -> str:
    return os.path.join(get_lib_path("runtime", "RUNTIME_LIB_DIR"), "backend", BRIDGE_TOML)


def _library_path() -> str:
    ext = {"Linux": ".so", "Darwin": ".dylib"}.get(platform.system())
    if ext is None:  # pragma: no cover
        raise NotImplementedError(f"Platform not supported: {platform.system()}")
    return os.path.join(get_lib_path("runtime", "RUNTIME_LIB_DIR"), RUNTIME_LIBRARY + ext)


def _intersect_mps(a: dict, b: dict) -> dict:
    return {k: list(set(a[k]) | set(b[k])) for k in a.keys() & b.keys()}


def python_device_capabilities(device) -> DeviceCapabilities:
    """The capabilities of a Python device as seen by Catalyst.

    These are the capabilities of the bridge (what the runtime can stream to Python), intersected
    with the capabilities that the device declares in its own TOML file, if it has one.
    """
    bridge = DeviceCapabilities.from_toml_file(_bridge_toml_path(), "qjit")

    own = None
    if getattr(device, "config_filepath", None) is not None:
        own = DeviceCapabilities.from_toml_file(device.config_filepath, "qjit")
    elif isinstance(getattr(device, "capabilities", None), DeviceCapabilities):
        own = device.capabilities

    if own is None:
        return bridge

    return replace(
        own,
        operations={
            k: bridge.operations[k] & own.operations[k]
            for k in bridge.operations.keys() & own.operations.keys()
        },
        observables={
            k: bridge.observables[k] & own.observables[k]
            for k in bridge.observables.keys() & own.observables.keys()
        },
        measurement_processes=_intersect_mps(
            bridge.measurement_processes, own.measurement_processes
        ),
        qjit_compatible=True,
        runtime_code_generation=True,
        dynamic_qubit_management=bridge.dynamic_qubit_management and own.dynamic_qubit_management,
        supported_mcm_methods=own.supported_mcm_methods or bridge.supported_mcm_methods,
    )


@dataclass
class _BridgeEntry:
    """Compile-time information about one QNode bound to a Python device."""

    device: Any
    mcm_method: str | None = None
    postselect_mode: str | None = None
    num_device_wires: int = 0
    measurement_plan: list | None = None


_ENTRIES: dict[int, _BridgeEntry] = {}
_ENTRY_IDS = itertools.count()


def register_qnode(device, execution_config=None, measurement_plan=None) -> dict:
    """Register a QNode's device for execution through the bridge.

    The device *instance* is kept (it is not rebuilt from its name), so that its configuration is
    preserved. Returns the device kwargs for the ``PLPythonDevice`` runtime backend.
    """
    mcm_config = getattr(execution_config, "mcm_config", None)
    mcm_method = getattr(mcm_config, "mcm_method", None)
    postselect_mode = getattr(mcm_config, "postselect_mode", None)
    entry = _BridgeEntry(
        device=device,
        mcm_method=getattr(mcm_method, "value", mcm_method),
        postselect_mode=getattr(postselect_mode, "value", postselect_mode),
        num_device_wires=len(device.wires),
        measurement_plan=measurement_plan,
    )
    bridge_id = next(_ENTRY_IDS)
    _ENTRIES[bridge_id] = entry
    return {"bridge_id": bridge_id, "device_name": device.name}


def backend_info(device) -> tuple[str, str, dict]:
    """The runtime backend (C interface name, library path, kwargs) for a Python device."""
    return RUNTIME_DEVICE_NAME, _library_path(), {"device_name": device.name}


def validate_execution_config(device, execution_config) -> None:
    """Raise informative errors for execution options that the bridge does not support."""
    gradient_method = getattr(execution_config, "gradient_method", None)
    if gradient_method in ("adjoint", "device", "backprop"):
        raise CompileError(
            f"diff_method='{gradient_method}' is not supported with the Python device "
            f"'{device.name}' under qjit: the device is executed as a black box by the runtime, "
            "without access to its state vector. Use diff_method='parameter-shift' or "
            "'finite-diff' instead."
        )
    mcm_config = getattr(execution_config, "mcm_config", None)
    mcm_method = getattr(getattr(mcm_config, "mcm_method", None), "value", None)
    if mcm_method is not None and mcm_method not in (*_WORKFLOW_MCM_METHODS, "device"):
        raise CompileError(
            f"mcm_method='{mcm_method}' is not supported with the Python device '{device.name}'."
        )


_BRIDGE_REWRITES = contextvars.ContextVar("catalyst_python_device_bridge", default=True)


def bridge_rewrites_enabled() -> bool:
    """Whether QNodes on Python devices are lowered for execution through the bridge.

    ``qp.flatten`` disables this: its tape is extracted at compile time and never executed.
    """
    return _BRIDGE_REWRITES.get()


@contextlib.contextmanager
def disable_bridge_rewrites():
    """Lower QNodes on Python devices like any other QNode (e.g. for ``qp.flatten``)."""
    token = _BRIDGE_REWRITES.set(False)
    try:
        yield
    finally:
        _BRIDGE_REWRITES.reset(token)


@qp.transform
def verify_no_mid_circuit_measurements(tape):
    """Reject mid-circuit measurements on Python devices in the legacy (non-capture) pathway.

    Branches conditioned on mid-circuit measurements are only rewritten for the Python device
    bridge with ``qjit(capture=True)``.
    """
    if any(op.name in ("MidCircuitMeasure", "MidMeasureMP") for op in tape.operations):
        raise CompileError(
            "Mid-circuit measurements on Python devices are only supported with "
            "qjit(capture=True)."
        )
    return (tape,), lambda results: results[0]


######################################################
### Developer hook
######################################################


class RuntimeTapeIntercepted(Exception):
    """Raised inside the runtime to interrupt the program when a tape was intercepted."""

    def __init__(self, tape):
        super().__init__("The program was interrupted after intercepting its runtime tape.")
        self.tape = tape


@dataclass
class InterceptedTape:
    """The result of :func:`intercept_runtime_tape`.

    Attributes:
        tapes (list[QuantumScript]): the tapes generated by the runtime, before device
            preprocessing, in execution order.
        interrupted (bool): whether the program execution was interrupted.
    """

    tapes: list = field(default_factory=list)
    interrupted: bool = False

    @property
    def tape(self):
        """The first intercepted tape (``None`` if no tape was generated)."""
        return self.tapes[0] if self.tapes else None


_INTERCEPTORS: list[Callable] = []


@contextlib.contextmanager
def intercept_runtime_tape(interrupt: bool = True, callback: Callable | None = None):
    """Intercept the tapes that the runtime generates for Python devices.

    Inside the context, every tape that the Catalyst runtime builds for a Python device is
    recorded before it is handed to the device (i.e. before device preprocessing). By default the
    program execution is interrupted as soon as the first tape is complete, and the context exits
    normally; code in the ``with`` block after the interrupted call is skipped.

    Args:
        interrupt (bool): whether to interrupt the program execution once the first tape has been
            intercepted. If ``False``, the tapes are recorded and executed as usual.
        callback (Callable[[QuantumScript], None] | None): an optional function called with every
            intercepted tape. It may raise an exception to abort the execution.

    Yields:
        InterceptedTape: the intercepted tapes.

    **Example**

    .. code-block:: python

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("default.qubit", wires=2))
        def circuit(x):
            qp.RX(x, 0)
            qp.CNOT([0, 1])
            return qp.expval(qp.Z(1))

    >>> with catalyst.debug.intercept_runtime_tape() as intercepted:
    ...     circuit(0.5)
    >>> intercepted.tape.operations
    [RX(0.5, wires=[0]), CNOT(wires=[0, 1])]
    """
    record = InterceptedTape()

    def interceptor(tape):
        record.tapes.append(tape)
        if callback is not None:
            callback(tape)
        if interrupt:
            record.interrupted = True
            raise RuntimeTapeIntercepted(tape)

    _INTERCEPTORS.append(interceptor)
    try:
        yield record
    except RuntimeTapeIntercepted:
        pass
    finally:
        _INTERCEPTORS.remove(interceptor)


######################################################
### Runtime: tape recording and execution
######################################################


class _RuntimeSession:
    """The tape of one QNode execution, built from the instructions streamed by the runtime."""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, entry: _BridgeEntry):
        self.entry = entry
        self.device = entry.device
        self.shots = None
        self.num_qubits = 0
        self.operations = []
        self.observables = []
        # Virtual wires (beyond the device wires) carry the outcomes of mid-circuit measurements
        # for branches conditioned on them; see ``_PythonDeviceQfuncInterpreter``.
        self.virtual_wires: dict[int, MeasurementValue] = {}
        self.last_measurement: tuple[int, MeasurementValue] | None = None
        self.results = None
        # id(Conditional) -> (the conditioning MCMs, the assignments of outcomes it covers)
        self._assignments: dict[int, tuple] = {}

    # ---------------------------------------------------------------- wires
    def label(self, wire: int):
        if wire >= self.entry.num_device_wires:
            raise RuntimeError(f"Wire {wire} is not a device wire.")
        return self.device.wires[wire]

    def labels(self, wires):
        return [self.label(w) for w in wires]

    def is_virtual(self, wire: int) -> bool:
        return wire >= self.entry.num_device_wires

    # ---------------------------------------------------------------- instructions
    def add(self, op):
        self.operations.append(op)
        self.last_measurement = None

    def gate(self, name, params, wires, adjoint, ctrl_wires, ctrl_values, optional=()):
        """Record a gate. Controls on virtual wires make the gate a ``Conditional``."""
        if name == "CNOT" and self._binds_virtual_wire(wires):
            # ``CNOT(measured wire, virtual wire)`` right after a mid-circuit measurement binds the
            # virtual wire to the outcome of that measurement.
            self.virtual_wires[wires[1]] = self.last_measurement[1]
            return
        if name == "CNOT" and self.is_virtual(wires[0]):
            if self._is_reset(wires):
                return
            ctrl_wires, ctrl_values = [wires[0], *ctrl_wires], [True, *ctrl_values]
            name, wires = "PauliX", wires[1:]
        if any(self.is_virtual(w) for w in wires):
            raise RuntimeError(f"Gate {name} acts on a wire that is not a device wire.")
        gate = _make_gate(name, params, self.labels(wires), optional)
        self.apply(gate, adjoint, ctrl_wires, ctrl_values)

    def _binds_virtual_wire(self, wires) -> bool:
        return (
            self.last_measurement is not None
            and self.is_virtual(wires[1])
            and not self.is_virtual(wires[0])
            and wires[0] == self.last_measurement[0]
        )

    def _is_reset(self, wires) -> bool:
        """``CNOT(virtual wire, measured wire)`` right after binding the virtual wire to that
        measurement is the ``reset`` of the measured wire."""
        if not self.operations or wires[0] not in self.virtual_wires:
            return False
        mcm = self.operations[-1]
        mv = self.virtual_wires[wires[0]]
        if not isinstance(mcm, MidMeasure) or mv.measurements[0] is not mcm:
            return False
        if self.label(wires[1]) != mcm.wires[0] or mcm.reset:
            return False
        reset = MidMeasure(
            wires=mcm.wires, reset=True, postselect=mcm.postselect, meas_uid=mcm.meas_uid
        )
        self.operations[-1] = reset
        mv.measurements[0] = reset
        return True

    def matrix(self, matrix, wires, adjoint, ctrl_wires, ctrl_values):
        op = qp.QubitUnitary(matrix, wires=self.labels(wires))
        self.apply(op, adjoint, ctrl_wires, ctrl_values)

    def apply(self, op, adjoint, ctrl_wires, ctrl_values):
        """Record ``op`` with its modifiers; controls on virtual wires become conditions."""
        if adjoint:
            op = qp.adjoint(op)
        conditions = [
            (self.virtual_wires[w], v) for w, v in zip(ctrl_wires, ctrl_values) if self.is_virtual(w)
        ]
        physical = [(w, v) for w, v in zip(ctrl_wires, ctrl_values) if not self.is_virtual(w)]
        if physical:
            op = qp.ctrl(
                op,
                control=self.labels([w for w, _ in physical]),
                control_values=[bool(v) for _, v in physical],
            )
        if conditions:
            op = self._conditional(conditions, op)
        self.add(op)

    def _conditional(self, conditions, op):
        """A ``Conditional`` for ``op`` under the given assignment of MCM outcomes.

        A predicate that is satisfied by several assignments reaches the runtime as one copy of the
        branch per assignment. Consecutive copies of a single operation are merged back into one
        ``Conditional``, which is valid since the assignments are mutually exclusive.
        """
        previous = self.operations[-1] if self.operations else None
        assignments = self._assignments.get(id(previous))
        key = tuple(id(mv) for mv, _ in conditions)
        if (
            assignments is not None
            and assignments[0] == key
            and all(a != tuple(v for _, v in conditions) for a in assignments[1])
            and qp.equal(previous.base, op)
        ):
            self.operations.pop()
            del self._assignments[id(previous)]
            merged = qp.ops.Conditional(previous.meas_val | _condition(conditions), op)
            self._assignments[id(merged)] = (key, assignments[1] + [tuple(v for _, v in conditions)])
            return merged
        conditional = qp.ops.Conditional(_condition(conditions), op)
        self._assignments[id(conditional)] = (key, [tuple(v for _, v in conditions)])
        return conditional

    def measure(self, wire, postselect):
        mcm = MidMeasure(wires=self.label(wire), postselect=postselect, meas_uid=_uid())
        self.add(mcm)
        mv = MeasurementValue([mcm])
        self.last_measurement = (wire, mv)

    # ---------------------------------------------------------------- observables
    def observable(self, kind, matrix, wires):
        wires = self.labels(wires)
        if kind == "Hermitian":
            obs = qp.Hermitian(matrix, wires=wires)
        else:
            obs = _NAMED_OBSERVABLES[kind](wires=wires)
        self.observables.append(obs)
        return len(self.observables) - 1

    def tensor(self, obs_ids):
        self.observables.append(qp.prod(*[self.observables[i] for i in obs_ids]))
        return len(self.observables) - 1

    def hamiltonian(self, coeffs, obs_ids):
        self.observables.append(qp.Hamiltonian(coeffs, [self.observables[i] for i in obs_ids]))
        return len(self.observables) - 1

    # ---------------------------------------------------------------- terminal measurements
    def measurement(self, kind, obs=None, wires=None):
        """Return the result of a terminal measurement requested by the runtime."""
        mp = self._measurement_process(kind, obs, wires)

        plan = self.entry.measurement_plan
        if plan is not None and self.results is None:
            # Execute the complete tape, with all terminal measurements of the QNode, once.
            self.results = list(zip(plan, self.execute(plan), [False] * len(plan)))
        if self.results is not None:
            for i, (planned, result, consumed) in enumerate(self.results):
                if not consumed and _same_measurement(planned, mp):
                    self.results[i] = (planned, result, True)
                    return result
        # The requested measurement was not planned at compile time; execute it on its own.
        return self.execute([mp])[0]

    def _measurement_process(self, kind, obs, wires):
        wires = None if wires is None else self.labels(wires)
        if kind in ("expval", "var"):
            return getattr(qp, kind)(self.observables[obs])
        if kind == "state":
            return qp.state()
        fn = {"probs": qp.probs, "sample": qp.sample, "counts": qp.counts}[kind]
        return fn(wires=wires) if wires else fn()

    def tape(self, measurements) -> QuantumScript:
        """The tape that the runtime generated, with the given terminal measurements."""
        return QuantumScript(self.operations, list(measurements), shots=self.shots)

    def execute(self, measurements) -> list:
        """Execute the tape on the device through the PennyLane execution pipeline."""
        tape = self.tape(measurements)
        for interceptor in list(_INTERCEPTORS):
            interceptor(tape)

        kwargs = {"diff_method": None}
        if self.entry.mcm_method is not None:
            kwargs["mcm_method"] = self.entry.mcm_method
        if self.entry.postselect_mode is not None:
            kwargs["postselect_mode"] = self.entry.postselect_mode
        (result,) = qp.execute([tape], self.device, **kwargs)
        if len(tape.measurements) == 1:
            result = (result,)
        return list(result)


def _uid():
    return f"catalyst-rt-{next(_UIDS):012d}"


_UIDS = itertools.count()

_NAMED_OBSERVABLES = {
    "PauliX": qp.X,
    "PauliY": qp.Y,
    "PauliZ": qp.Z,
    "Hadamard": qp.Hadamard,
    "Identity": qp.Identity,
}


def _make_gate(name, params, wires, optional=()):
    cls = getattr(qp, name, None) or getattr(qp.ops, name, None)
    if cls is None:
        raise RuntimeError(f"The runtime received the unknown gate '{name}'.")
    if name == "PCPhase":
        return cls(params[0], dim=int(params[1]), wires=wires)
    if name == "PauliRot":
        return cls(params[0], optional[0], wires=wires)
    return cls(*params, wires=wires)


def _condition(conditions) -> MeasurementValue:
    """The condition of a gate controlled on virtual wires with the given control values."""
    result = None
    for mv, value in conditions:
        term = mv if value else ~mv
        result = term if result is None else result & term
    return result


def _same_measurement(planned, requested) -> bool:
    if type(planned) is not type(requested):
        return False
    if planned.obs is not None or requested.obs is not None:
        if planned.obs is None or requested.obs is None:
            return False
        return qp.equal(planned.obs, requested.obs)
    return planned.wires == requested.wires or (not planned.wires and not requested.wires)


def _flatten_result(kind, result, session: _RuntimeSession, wires) -> list[float]:
    """Flatten a PennyLane result into the list of floats expected by the runtime."""
    if kind == "state":
        state = np.asarray(result, dtype=complex).reshape(-1)
        return np.stack([state.real, state.imag], axis=-1).reshape(-1).tolist()
    if kind == "counts":
        n = len(wires) if wires else session.entry.num_device_wires
        counts = np.zeros(2**n)
        for outcome, count in result.items():
            counts[int(str(outcome), 2)] = count
        return list(range(2**n)) + counts.tolist()
    return np.asarray(result, dtype=float).reshape(-1).tolist()


_SESSIONS: dict[int, _RuntimeSession] = {}
_SESSION_IDS = itertools.count()
_LOCK = threading.Lock()


def _parse_kwargs(kwargs: str) -> dict:
    try:
        return ast.literal_eval(kwargs)
    except (ValueError, SyntaxError) as e:  # pragma: no cover
        raise RuntimeError(f"Invalid device kwargs for the Python device bridge: {kwargs}") from e


_PENDING_ERROR: list[BaseException] = []


def reraise_pending(runtime_error: RuntimeError) -> None:
    """Re-raise the exception that aborted a program on the Python side of the bridge.

    The runtime reports errors of Python devices as a generic ``RuntimeError``; this raises the
    original exception (e.g. a ``DeviceError`` for an unsupported instruction) instead.
    """
    if _PENDING_ERROR:
        error = _PENDING_ERROR.pop()
        _PENDING_ERROR.clear()
        raise error from runtime_error


def runtime_dispatch(method: str, payload: str) -> list:
    """Entry point for the ``PLPythonDevice`` runtime backend.

    Args:
        method (str): the name of the instruction.
        payload (str): the JSON encoded arguments of the instruction.

    Returns:
        list[float]: the flattened results of the instruction.
    """
    try:
        return _dispatch(method, json.loads(payload))
    except BaseException as e:
        _PENDING_ERROR.append(e)
        raise


def _dispatch(method: str, args: dict) -> list:
    # pylint: disable=too-many-return-statements,too-many-branches

    if method == "init":
        kwargs = _parse_kwargs(args["kwargs"])
        entry = _ENTRIES.get(int(kwargs.get("bridge_id", -1)))
        if entry is None:
            raise RuntimeError(
                "The Python device of this program is not registered in this process. Programs "
                "compiled for Python devices can only be executed by the process that compiled "
                "them."
            )
        with _LOCK:
            session_id = next(_SESSION_IDS)
            _SESSIONS[session_id] = _RuntimeSession(entry)
        return [session_id]

    session = _SESSIONS[args["session"]]

    if method == "release":
        del _SESSIONS[args["session"]]
        return []
    if method == "shots":
        session.shots = int(args["shots"]) or None
        return []
    if method == "allocate":
        session.num_qubits = int(args["num_qubits"])
        return []
    if method == "gate":
        session.gate(
            args["name"],
            args["params"],
            args["wires"],
            args["adjoint"],
            args["ctrl_wires"],
            args["ctrl_values"],
            args.get("optional_params", ()),
        )
        return []
    if method == "matrix":
        dim = 2 ** len(args["wires"])
        matrix = (np.array(args["re"]) + 1j * np.array(args["im"])).reshape(dim, dim)
        session.matrix(
            matrix, args["wires"], args["adjoint"], args["ctrl_wires"], args["ctrl_values"]
        )
        return []
    if method == "set_state":
        state = np.array(args["re"]) + 1j * np.array(args["im"])
        session.add(qp.StatePrep(state, wires=session.labels(args["wires"])))
        return []
    if method == "set_basis_state":
        session.add(qp.BasisState(np.array(args["state"]), wires=session.labels(args["wires"])))
        return []
    if method == "measure":
        session.measure(args["wire"], args.get("postselect"))
        return [0.0]
    if method == "observable":
        matrix = None
        if args["kind"] == "Hermitian":
            dim = 2 ** len(args["wires"])
            matrix = (np.array(args["re"]) + 1j * np.array(args["im"])).reshape(dim, dim)
        return [session.observable(args["kind"], matrix, args["wires"])]
    if method == "tensor":
        return [session.tensor(args["obs"])]
    if method == "hamiltonian":
        return [session.hamiltonian(args["coeffs"], args["obs"])]
    if method in ("expval", "var", "probs", "sample", "counts", "state"):
        wires = args.get("wires")
        result = session.measurement(method, obs=args.get("obs"), wires=wires)
        return _flatten_result(method, result, session, wires)

    raise RuntimeError(f"Unknown instruction '{method}' for the Python device bridge.")
