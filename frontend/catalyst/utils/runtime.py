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
from typing import Set, Dict, Any, Tuple

import pennylane as qml

from catalyst._configuration import INSTALLED
from catalyst.utils.exceptions import CompileError
from catalyst.utils.toml import toml_load, TOMLDocument

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


def check_qjit_compatibility(device, config):
    """Check that the device is qjit compatible.

    Args:
        device (qml.Device): An instance of a quantum device.
        config (Dict[Str, Any]): Configuration dictionary.

    Raises:
        CompileError
    """
    if config["compilation"]["qjit_compatible"]:
        return

    name = device.name
    msg = f"Attempting to compile program for incompatible device {name}."
    raise CompileError(msg)


def check_device_config(device):
    """Check that the device configuration exists.

    Args:
        device (qml.Device): An instance of a quantum device.

    Raises:
        CompileError
    """
    if hasattr(device, "config") and device.config.exists():
        return

    name = device.name
    msg = f"Attempting to compile program for incompatible device {name}."
    raise CompileError(msg)


def get_native_gates_PL(config) -> Set[str]:
    """Get gates that are natively supported by the device and therefore do not need to be
    decomposed.

    Args:
        config (Dict[Str, Any]): Configuration dictionary

    Returns:
        List[str]: List of gate names in the PennyLane format.
    """
    gates = config["operators"]["gates"]["named"]
    # import pdb; pdb.set_trace()
    gates_PL = set()
    for gate_name in [str(g) for g in gates]:
        gates_PL.add(f"{gate_name}")
        if gates[gate_name].get('controllable', False):
            gates_PL.add(f"C({gate_name})")
    return gates_PL


def get_decomposable_gates(config):
    """Get gates that will be decomposed according to PL's decomposition rules.

    Args:
        config (Dict[Str, Any]): Configuration dictionary
    """
    return config["operators"]["gates"]["decomp"]


def get_matrix_decomposable_gates(config):
    """Get gates that will be decomposed to QubitUnitary.

    Args:
        config (Dict[Str, Any]): Configuration dictionary
    """
    return config["operators"]["gates"]["matrix"]


def check_no_overlap(*args):
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

    msg = "Device has overlapping gates in native and decomposable sets."
    raise CompileError(msg)


def filter_out_adjoint_and_control(operations):
    """Remove Adjoint and C strings from operations.

    Args:
        operations (List[Str]): List of strings with names of supported operations

    Returns:
        List: A list of strings with names of supported operations with Adjoint and C gates
        removed.
    """
    adjoint = re.compile(r"^Adjoint\(.*\)$")
    control = re.compile(r"^C\(.*\)$")

    def is_not_adj(op):
        return not re.match(adjoint, op)

    def is_not_ctrl(op):
        return not re.match(control, op)

    operations_no_adj = filter(is_not_adj, operations)
    operations_no_adj_no_ctrl = filter(is_not_ctrl, operations_no_adj)
    return list(operations_no_adj_no_ctrl)


def check_full_overlap(device, *args):
    """Check that device.operations is equivalent to the union of *args

    Args:
        device (qml.Device): An instance of a quantum device.
        *args (List[Str]): List of strings.

    Raises: CompileError
    """
    # operations = filter_out_adjoint_and_control(device.operations)
    gates_in_device = set(device.operations)
    set_of_sets = [set(arg) for arg in args]
    gates_in_spec = set.union(*set_of_sets)
    if gates_in_device == gates_in_spec:
        return

    import pdb; pdb.set_trace()

    msg = (
        "Gates in qml.device.operations and specification file do not match.\n"
        f"Gates that present only in the device: {gates_in_device - gates_in_spec}\n"
        f"Gates that present only in spec: {gates_in_spec - gates_in_device}\n"
    )
    raise CompileError(msg)


def check_gates_are_compatible_with_device(device, config):
    """Validate configuration dictionary against device.

    Args:
        device (qml.Device): An instance of a quantum device.
        config (Dict[Str, Any]): Configuration dictionary

    Raises: CompileError
    """
    native = get_native_gates_PL(config)
    decomposable = get_decomposable_gates(config)
    matrix = get_matrix_decomposable_gates(config)
    check_no_overlap(native, decomposable, matrix)

    if hasattr(device, "operations"):  # pragma: nocover
        # The new device API has no "operations" field
        # so we cannot check that there's an overlap or not.
        check_full_overlap(device, native)


def validate_config_with_device(device):
    """Validate configuration file against device.
    Will raise a CompileError
    * if device does not contain ``config`` attribute
    * if configuration file does not exists
    * if decomposable, matrix, and native gates have some overlap
    * if decomposable, matrix, and native gates do not match gates in ``device.operations``

    Args:
        device (qml.Device): An instance of a quantum device.

    Raises: CompileError
    """
    check_device_config(device)

    with open(device.config, "rb") as f:
        config = toml_load(f)

    check_qjit_compatibility(device, config)
    check_gates_are_compatible_with_device(device, config)


def load_toml_file_into(device, toml_file_name=None):
    """Temporary function. This function adds the `config` field to devices containing the path to
    it's TOML configuration file.
    TODO: Remove this function when `qml.Device`s are guaranteed to have their own
    config file field."""
    device_lpath = pathlib.Path(get_lib_path("runtime", "RUNTIME_LIB_DIR"))
    name = device.name
    if isinstance(device, qml.Device):
        name = device.short_name

    # The toml files name convention we follow is to replace
    # the dots with underscores in the device short name.
    if toml_file_name is None:
        toml_file_name = name.replace(".", "_") + ".toml"
    # And they are currently saved in the following directory.
    toml_file = device_lpath.parent / "lib" / "backend" / toml_file_name

    with open(toml_file, "rb") as f:
        config = toml_load(f)

    toml_operations = get_native_gates_PL(config)
    device.operations = toml_operations
    # if not hasattr(device, "operations") or device.operations is None:
    # else:
    #     # TODO: make sure toml_operations matches the device operations
    #     pass
    device.config = toml_file


def extract_backend_info(device) -> Tuple[TOMLDocument, str, str, Dict[str, Any]]:
    """Extract the backend info as a tuple of (name, lib, kwargs)."""

    validate_config_with_device(device)

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
        raise CompileError(f"The {dname} device is not supported for compilation at the moment.")

    if not pathlib.Path(device_lpath).is_file():
        raise CompileError(f"Device at {device_lpath} cannot be found!")

    if hasattr(device, "shots"):
        device_kwargs["shots"] = device.shots if device.shots else 0

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

    with open(device.config, "rb") as f:
        config = toml_load(f)

    if "options" in config.keys():
        for k, v in config["options"].items():
            if hasattr(device, v):
                device_kwargs[k] = getattr(device, v)

    return config, device_name, device_lpath, device_kwargs
