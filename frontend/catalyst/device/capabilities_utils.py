import pathlib

import pennylane as qml
from pennylane.devices.capabilities import DeviceCapabilities, ExecutionCondition

from catalyst.utils.exceptions import CompileError
from catalyst.utils.runtime_environment import get_lib_path


def _resolve_device_config_path(device) -> pathlib.Path:
    """Resolve the toml config path for a device."""
    if getattr(device, "config_filepath") is not None:
        return pathlib.Path(device.config_filepath)

    # TODO: Remove this section when devices are guaranteed to have their own config file
    device_lpath = pathlib.Path(get_lib_path("runtime", "RUNTIME_LIB_DIR"))
    name = device.short_name if isinstance(device, qml.devices.LegacyDevice) else device.name
    # The toml files name convention we follow is to replace
    # the dots with underscores in the device short name.
    toml_file_name = name.replace(".", "_") + ".toml"
    # And they are currently saved in the following directory.
    return device_lpath.parent / "lib" / "backend" / toml_file_name


def load_device_capabilities(device) -> DeviceCapabilities:
    """Get the contents of the device config toml file."""
    # TODO: This code exists purely for testing. Find another way to customize device
    #       capabilities as seen by Catalyst.
    if hasattr(device, "qjit_capabilities"):
        return device.qjit_capabilities

    toml_file = _resolve_device_config_path(device)

    try:
        return DeviceCapabilities.from_toml_file(toml_file, "qjit")
    except FileNotFoundError as e:
        raise CompileError(
            "Attempting to compile program for incompatible device: "
            f"Config file ({toml_file}) does not exist"
        ) from e


def requires_shots(capabilities: DeviceCapabilities) -> bool:
    """
    Checks if a device capabilities requires shots.

    A device requires shots if all of its MPs are finite shots only.
    """
    return all(
        ExecutionCondition.FINITE_SHOTS_ONLY in MP_conditions
        for _, MP_conditions in capabilities.measurement_processes.items()
    )


def filter_device_capabilities_with_shots(
    capabilities: DeviceCapabilities, shots_present: bool, unitary_support=None
) -> DeviceCapabilities:
    """
    Process the device capabilities depending on whether shots are present in the user program,
    and whether device supports QubitUnitary ops.
    """
    device_capabilities = capabilities.filter(finite_shots=shots_present)

    # TODO: This is a temporary measure to ensure consistency of behaviour. Remove this
    #       when customizable multi-pathway decomposition is implemented. (Epic 74474)
    if unitary_support is not None:
        _to_matrix_ops = unitary_support
        setattr(device_capabilities, "to_matrix_ops", _to_matrix_ops)
        if _to_matrix_ops and not device_capabilities.supports_operation("QubitUnitary"):
            raise CompileError("The device that specifies to_matrix_ops must support QubitUnitary.")

    return device_capabilities
