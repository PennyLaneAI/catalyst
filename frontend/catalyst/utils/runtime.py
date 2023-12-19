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

import pennylane as qml

from catalyst._configuration import INSTALLED
from catalyst.utils.exceptions import CompileError
from catalyst.utils.toml import toml_load

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


def get_native_gates(config):
    """Get gates that are natively supported by the device and therefore do not need to be
    decomposed.

    Args:
        config (Dict[Str, Any]): Configuration dictionary
    """
    return config["operators"]["gates"][0]["native"]


def get_decomposable_gates(config):
    """Get gates that will be decomposed according to PL's decomposition rules.

    Args:
        config (Dict[Str, Any]): Configuration dictionary
    """
    return config["operators"]["gates"][0]["decomp"]


def get_matrix_decomposable_gates(config):
    """Get gates that will be decomposed to QubitUnitary.

    Args:
        config (Dict[Str, Any]): Configuration dictionary
    """
    return config["operators"]["gates"][0]["matrix"]


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
    operations = filter_out_adjoint_and_control(device.operations)
    gates_in_device = set(operations)
    set_of_sets = [set(arg) for arg in args]
    union = set.union(*set_of_sets)
    if gates_in_device == union:
        return

    msg = "Gates in qml.device.operations and specification file do not match"
    raise CompileError(msg)


def check_gates_are_compatible_with_device(device, config):
    """Validate configuration dictionary against device.

    Args:
        device (qml.Device): An instance of a quantum device.
        config (Dict[Str, Any]): Configuration dictionary

    Raises: CompileError
    """

    native = get_native_gates(config)
    decomposable = get_decomposable_gates(config)
    matrix = get_matrix_decomposable_gates(config)
    check_no_overlap(native, decomposable, matrix)
    if not hasattr(device, "operations"):  # pragma: nocover
        # The new device API has no "operations" field
        # so we cannot check that there's an overlap or not.
        return

    check_full_overlap(device, native, decomposable, matrix)


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


def extract_backend_info(device):
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
        device_kwargs[
            "backend"
        ] = device._device._delegate.DEVICE_ID  # pylint: disable=protected-access
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
