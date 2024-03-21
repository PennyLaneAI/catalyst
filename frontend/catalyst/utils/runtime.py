# Copyright 2023 Xanadu Quantum Technologies Inc.

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
Runtime utility methods.
"""

# pylint: disable=too-many-branches

import os
import pathlib
import platform
import re
from dataclasses import dataclass
from typing import Any, Dict, Set

import pennylane as qml

from catalyst._configuration import INSTALLED
from catalyst.utils.exceptions import CompileError
from catalyst.utils.toml import (
    ProgramFeatures,
    TOMLDocument,
    check_quantum_control_flag,
    get_decomposable_gates,
    get_device_config,
    get_matrix_decomposable_gates,
    get_native_gates,
    get_observables,
    pennylane_operation_set,
    read_toml_file,
)

package_root = os.path.dirname(__file__)


# Default paths to dep libraries
DEFAULT_LIB_PATHS = {
    "llvm": os.path.join(package_root, "../../../mlir/llvm-project/build/lib"),
    "runtime": os.path.join(package_root, "../../../runtime/build/lib"),
    "enzyme": os.path.join(package_root, "../../../mlir/Enzyme/build/Enzyme"),
}


# TODO: This should be removed after implementing `get_c_interface`
# for the following backend devices:
SUPPORTED_RT_DEVICES = {
    "lightning.qubit": ("LightningSimulator", "librtd_lightning"),
    "lightning.kokkos": ("LightningKokkosSimulator", "librtd_lightning"),
    "braket.aws.qubit": ("OpenQasmDevice", "librtd_openqasm"),
    "braket.local.qubit": ("OpenQasmDevice", "librtd_openqasm"),
}


def get_lib_path(project, env_var):
    """Get the library path."""
    if INSTALLED:
        return os.path.join(package_root, "..", "lib")  # pragma: no cover
    return os.getenv(env_var, DEFAULT_LIB_PATHS.get(project, ""))


def deduce_schema1_native_controlled_gates(native_gates: Set[str]) -> Set[str]:
    """Calculate the set of controlled gates given the set of natively supported gates. This
    function is used with the toml config schema 1 which did not support per-gate control
    specifications. Later schemas provide the required information directly.
    """
    # The deduction logic is the following:
    # * Most of the gates have their `C(Gate)` controlled counterparts.
    # * Some gates have to be decomposed if controlled version is used. Typically these are gates
    #   which are already controlled but have well-known names.
    # * Few gates, like `QubitUnitary`, have separate classes for their controlled versions.
    gates_to_be_decomposed_if_controlled = [
        "Identity",
        "CNOT",
        "CY",
        "CZ",
        "CSWAP",
        "CRX",
        "CRY",
        "CRZ",
        "CRot",
        "ControlledPhaseShift",
        "QubitUnitary",
        "Toffoli",
    ]
    native_controlled_gates = set(
        [f"C({gate})" for gate in native_gates if gate not in gates_to_be_decomposed_if_controlled]
        + [f"Controlled{gate}" for gate in native_gates if gate in ["QubitUnitary"]]
    )
    return native_controlled_gates


def check_no_overlap(*args, device_name):
    """Check items in *args are mutually exclusive.

    Args:
        *args (List[Str]): List of strings.

    Raises:
        CompileError
    """
    set_of_sets = [set(arg) for arg in args]
    union = set.union(*set_of_sets)
    len_of_sets = [len(arg) for arg in args]
    if sum(len_of_sets) == len(union):
        return

    overlaps = set()
    for s in set_of_sets:
        overlaps.update(s - union)
        union = union - s

    msg = f"Device '{device_name}' has overlapping gates: {overlaps}"
    raise CompileError(msg)


def filter_out_adjoint(operations):
    """Remove Adjoint from operations.

    Args:
        operations (List[Str]): List of strings with names of supported operations

    Returns:
        List: A list of strings with names of supported operations with Adjoint and C gates
        removed.
    """
    adjoint = re.compile(r"^Adjoint\(.*\)$")

    def is_not_adj(op):
        return not re.match(adjoint, op)

    operations_no_adj = filter(is_not_adj, operations)
    return set(operations_no_adj)


def check_full_overlap(device_gates: Set[str], spec_gates: Set[str]) -> None:
    """Check that device.operations is equivalent to the union of *args

    Args:
        device_gates (Set[str]): device gates
        spec_gates (Set[str]): spec gates

    Raises: CompileError
    """
    if device_gates == spec_gates:
        return

    msg = (
        "Gates in qml.device.operations and specification file do not match.\n"
        f"Gates that present only in the device: {device_gates - spec_gates}\n"
        f"Gates that present only in spec: {spec_gates - device_gates}\n"
    )
    raise CompileError(msg)


def validate_config_with_device(device: qml.QubitDevice, config: TOMLDocument) -> None:
    """Validate configuration document against the device attributes.
    Raise CompileError in case of mismatch:
    * If device is not qjit-compatible.
    * If configuration file does not exists.
    * If decomposable, matrix, and native gates have some overlap.
    * If decomposable, matrix, and native gates do not match gates in ``device.operations`` and
      ``device.observables``.

    Args:
        device (qml.Device): An instance of a quantum device.
        config (TOMLDocument): A TOML document representation.

    Raises: CompileError
    """

    if not config["compilation"]["qjit_compatible"]:
        raise CompileError(
            f"Attempting to compile program for incompatible device '{device.name}': "
            f"Config is not marked as qjit-compatible"
        )

    device_name = device.short_name if isinstance(device, qml.Device) else device.name
    program_features = ProgramFeatures(device.shots is not None)
    device_config = get_device_config(config, program_features, device_name)

    native = pennylane_operation_set(device_config.native_gates)
    observables = pennylane_operation_set(device_config.observables)
    decomposable = pennylane_operation_set(device_config.decomp)
    matrix = pennylane_operation_set(device_config.matrix)

    check_no_overlap(native, decomposable, matrix, device_name=device_name)

    if hasattr(device, "operations") and hasattr(device, "observables"):
        device_gates = set.union(set(device.operations), set(device.observables))
        device_gates = filter_out_adjoint(device_gates)
        spec_gates = set.union(native, observables, matrix, decomposable)
        spec_gates = filter_out_adjoint(spec_gates)
        check_full_overlap(device_gates, spec_gates)


def device_get_toml_config(device) -> TOMLDocument:
    """Get the contents of the device config file."""
    if hasattr(device, "config"):
        # The expected case: device specifies its own config.
        toml_file = device.config
    else:
        # TODO: Remove this section when `qml.Device`s are guaranteed to have their own config file
        # field.
        device_lpath = pathlib.Path(get_lib_path("runtime", "RUNTIME_LIB_DIR"))

        name = device.short_name if isinstance(device, qml.Device) else device.name
        # The toml files name convention we follow is to replace
        # the dots with underscores in the device short name.
        toml_file_name = name.replace(".", "_") + ".toml"
        # And they are currently saved in the following directory.
        toml_file = device_lpath.parent / "lib" / "backend" / toml_file_name

    try:
        config = read_toml_file(toml_file)
    except FileNotFoundError as e:
        raise CompileError(
            "Attempting to compile program for incompatible device: "
            f"Config file ({toml_file}) does not exist"
        ) from e

    return config


@dataclass
class BackendInfo:
    """Backend information"""

    device_name: str
    c_interface_name: str
    lpath: str
    kwargs: Dict[str, Any]


def extract_backend_info(device: qml.QubitDevice, config: TOMLDocument) -> BackendInfo:
    """Extract the backend info from a quantum device. The device is expected to carry a reference
    to a valid TOML config file."""

    dname = device.name
    if isinstance(device, qml.Device):
        dname = device.short_name

    device_name = ""
    device_lpath = ""
    device_kwargs = {}

    if dname in SUPPORTED_RT_DEVICES:
        # Support backend devices without `get_c_interface`
        device_name = SUPPORTED_RT_DEVICES[dname][0]
        device_lpath = get_lib_path("runtime", "RUNTIME_LIB_DIR")
        sys_platform = platform.system()

        if sys_platform == "Linux":
            device_lpath = os.path.join(device_lpath, SUPPORTED_RT_DEVICES[dname][1] + ".so")
        elif sys_platform == "Darwin":  # pragma: no cover
            device_lpath = os.path.join(device_lpath, SUPPORTED_RT_DEVICES[dname][1] + ".dylib")
        else:  # pragma: no cover
            raise NotImplementedError(f"Platform not supported: {sys_platform}")
    elif hasattr(device, "get_c_interface"):
        # Support third party devices with `get_c_interface`
        device_name, device_lpath = device.get_c_interface()
    else:
        raise CompileError(f"The {dname} device does not provide C interface for compilation.")

    if not pathlib.Path(device_lpath).is_file():
        raise CompileError(f"Device at {device_lpath} cannot be found!")

    if hasattr(device, "shots"):
        if isinstance(device, qml.Device):
            device_kwargs["shots"] = device.shots if device.shots else 0
        else:
            # TODO: support shot vectors
            device_kwargs["shots"] = device.shots.total_shots if device.shots else 0

    if dname == "braket.local.qubit":  # pragma: no cover
        device_kwargs["device_type"] = dname
        device_kwargs["backend"] = (
            # pylint: disable=protected-access
            device._device._delegate.DEVICE_ID
        )
    elif dname == "braket.aws.qubit":  # pragma: no cover
        device_kwargs["device_type"] = dname
        device_kwargs["device_arn"] = device._device._arn  # pylint: disable=protected-access
        if device._s3_folder:  # pylint: disable=protected-access
            device_kwargs["s3_destination_folder"] = str(
                device._s3_folder  # pylint: disable=protected-access
            )

    options = config.get("options", {})
    for k, v in options.items():
        if hasattr(device, v):
            device_kwargs[k] = getattr(device, v)

    return BackendInfo(dname, device_name, device_lpath, device_kwargs)
