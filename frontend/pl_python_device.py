"""
PennyLane Python Device - Catalyst Frontend Integration
=========================================================

This module provides a PennyLane device wrapper that enables *any* Python-based
PennyLane device to work with Catalyst's @qml.qjit decorator.

Architecture
------------
When a user writes:

    dev = qml.device("default.qubit", wires=10)

    @qml.qjit(autograph=True)
    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.probs(wires=[0, 1])

The Catalyst frontend needs a C++ runtime backend. For `lightning.qubit` there
is a native C++ implementation. For Braket, there's the OpenQASM backend. For
everything else, there was no path.

This module provides that path: a wrapper device class `PLPythonDevice` that:
1. Exposes `get_c_interface()` -> returns ("PLPythonDevice", path_to_librtd_pennylane_python.so)
2. Carries a TOML config declaring the maximal gate set
3. Passes the underlying Python device's name + kwargs through to the C++ runtime
4. At runtime, the C++ backend accumulates gates into a JSON tape, then calls
   back into Python via nanobind to execute on the original device.

Usage
-----
There are two integration paths:

**Path A: Explicit wrapper (prototype/testing)**

    from catalyst_pennylane_compat.frontend.pl_python_device import PLPythonDevice

    dev = PLPythonDevice(wires=10, target_device="default.qubit")

    @qml.qjit
    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.probs(wires=[0, 1])


**Path B: Automatic fallback in Catalyst frontend (production)**

Add to `catalyst/device/qjit_device.py` SUPPORTED_RT_DEVICES:

    "pennylane.python": ("PLPythonDevice", "librtd_pennylane_python"),

And modify `extract_backend_info` to fall back to this backend when no native
C interface is available.
"""

import os
import pathlib
import platform
from typing import Any, Dict, Optional, Set, Tuple, Union

import pennylane as qml
from pennylane.devices import Device


class PLPythonDevice(Device):
    """Catalyst-compatible wrapper around any PennyLane Python device.

    This device class acts as a proxy: at compile-time it exposes a
    `get_c_interface()` for the Catalyst frontend, and at runtime the C++
    backend calls back into Python to execute tapes on the wrapped device.

    Parameters
    ----------
    wires : int
        Number of qubits.
    target_device : str
        Name of the PennyLane device to delegate execution to
        (e.g. "default.qubit", "default.mixed", "qiskit.aer").
    shots : int or None
        Number of shots for sampling-based measurements.
    target_device_kwargs : dict
        Additional kwargs passed to the target device constructor.
    """

    name = "pennylane.python"
    short_name = "pennylane.python"

    # Path to the TOML config file (resolved relative to this file)
    config_filepath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "runtime", "lib", "backend", "pennylane_python", "pennylane_python.toml"
    )

    def __init__(
        self,
        wires: int,
        target_device: str = "default.qubit",
        shots: Optional[int] = None,
        **target_device_kwargs: Any,
    ):
        super().__init__(wires=wires, shots=shots)
        self._target_device_name = target_device
        self._target_device_kwargs = target_device_kwargs
        self._num_wires = wires
        self._shots = shots

    @property
    def operations(self) -> Set[str]:
        """The set of operations this device supports.

        We declare the full set from the TOML. At compile time, the Catalyst
        frontend intersects this with the Catalyst runtime's supported set.
        """
        return {
            "Identity", "PauliX", "PauliY", "PauliZ", "Hadamard",
            "S", "T", "SX", "CNOT", "CY", "CZ", "SWAP", "CSWAP",
            "Toffoli", "ISWAP", "PSWAP", "ECR",
            "PhaseShift", "ControlledPhaseShift",
            "RX", "RY", "RZ", "Rot",
            "CRX", "CRY", "CRZ", "CRot",
            "IsingXX", "IsingXY", "IsingYY", "IsingZZ",
            "SingleExcitation", "DoubleExcitation",
            "QubitUnitary", "MultiRZ", "GlobalPhase",
        }

    @property
    def observables(self) -> Set[str]:
        return {
            "PauliX", "PauliY", "PauliZ", "Hadamard", "Hermitian",
            "Identity", "Projector", "SProd", "Prod", "Sum", "Hamiltonian",
        }

    def get_c_interface(self) -> Tuple[str, str]:
        """Return the C++ device factory name and shared library path.

        This is the hook that the Catalyst frontend uses to resolve the
        runtime backend.
        """
        lib_dir = self._get_lib_dir()
        sys_platform = platform.system()

        if sys_platform == "Linux":
            lib_path = os.path.join(lib_dir, "librtd_pennylane_python.so")
        elif sys_platform == "Darwin":
            lib_path = os.path.join(lib_dir, "librtd_pennylane_python.dylib")
        else:
            raise NotImplementedError(f"Platform not supported: {sys_platform}")

        return ("PLPythonDevice", lib_path)

    @property
    def device_kwargs(self) -> Dict[str, str]:
        """Kwargs passed through to the C++ runtime, which forwards them
        to the nanobind Python module for device instantiation.

        All values must be strings (the C++ parse_kwargs function expects
        string key-value pairs).
        """
        kwargs = {
            "pl_device_name": self._target_device_name,
            "wires": str(self._num_wires),
            "shots": str(self._shots or 0),
        }
        # Pass through any target device kwargs as strings
        for k, v in self._target_device_kwargs.items():
            kwargs[k] = str(v)
        return kwargs

    def execute(self, circuits, execution_config=None):
        """Fallback execution (not used when qjit'd, but needed for interface)."""
        dev = qml.device(self._target_device_name, wires=self._num_wires,
                         shots=self._shots, **self._target_device_kwargs)
        return dev.execute(circuits, execution_config)

    @staticmethod
    def _get_lib_dir() -> str:
        """Resolve the runtime library directory.

        Checks RUNTIME_LIB_DIR env var first, then falls back to the
        catalyst package's lib directory.
        """
        env_path = os.environ.get("RUNTIME_LIB_DIR")
        if env_path and os.path.isdir(env_path):
            return env_path

        try:
            import catalyst
            return str(pathlib.Path(catalyst.__file__).parent / "lib")
        except ImportError:
            # Development: look relative to this file
            return str(pathlib.Path(__file__).parent.parent /
                       "runtime" / "lib" / "backend" / "pennylane_python")
