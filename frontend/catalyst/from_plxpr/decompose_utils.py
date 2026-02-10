import textwrap
from typing import Optional, Union
import pathlib

from catalyst.api_extensions.quantum_operators import MidCircuitMeasure
from catalyst.utils.exceptions import DifferentiableCompileError

import pennylane as qml
from pennylane.devices.capabilities import (
    DeviceCapabilities,
    ExecutionCondition,
    OperatorProperties,
)
from catalyst.utils.runtime_environment import get_lib_path
from catalyst.device.decomposition import catalyst_acceptance
from catalyst.utils.exceptions import CompileError
from catalyst.jax_primitives_utils import _calculate_diff_method


def calculate_diff_method(device, execution_config):
    """Calculate the diff method for the device."""
    return _calculate_diff_method(device, execution_config)


def load_device_capabilities(device) -> DeviceCapabilities:
    """Get the contents of the device config toml file."""

    # TODO: This code exists purely for testing. Find another way to customize device Find a
    #       better way for a device to customize its capabilities as seen by Catalyst.
    if hasattr(device, "qjit_capabilities"):
        return device.qjit_capabilities

    if getattr(device, "config_filepath") is not None:
        toml_file = device.config_filepath

    else:
        # TODO: Remove this section when devices are guaranteed to have their own config file
        device_lpath = pathlib.Path(get_lib_path("runtime", "RUNTIME_LIB_DIR"))
        name = device.short_name if isinstance(device, qml.devices.LegacyDevice) else device.name
        # The toml files name convention we follow is to replace
        # the dots with underscores in the device short name.
        toml_file_name = name.replace(".", "_") + ".toml"
        # And they are currently saved in the following directory.
        toml_file = device_lpath.parent / "lib" / "backend" / toml_file_name

    try:
        capabilities = DeviceCapabilities.from_toml_file(toml_file, "qjit")

    except FileNotFoundError as e:
        raise CompileError(
            "Attempting to compile program for incompatible device: "
            f"Config file ({toml_file}) does not exist"
        ) from e

    return capabilities


def requires_shots(capabilities):
    """
    Checks if a device capabilities requires shots.

    A device requires shots if all of its MPs are finite shots only.
    """
    return all(
        ExecutionCondition.FINITE_SHOTS_ONLY in MP_conditions
        for _, MP_conditions in capabilities.measurement_processes.items()
    )


def filter_device_capabilities_with_shots(
    capabilities, shots_present, unitary_support=None
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



def get_device_capabilities(
    device,
    execution_config,
    shots_len=None,
):

    capabilities = load_device_capabilities(device)

    if execution_config is None:
        execution_config = qml.devices.ExecutionConfig()
        execution_config = device.setup_execution_config

    if shots_len == 0 and requires_shots(capabilities):
        raise CompileError(
            textwrap.dedent(
                f"""
                {device.name} does not support analytical simulation.
                Please supply the number of shots on the qnode.
                """
            )
        )
    capabilities = filter_device_capabilities_with_shots(
        capabilities=capabilities,
        shots_present=(shots_len > 0),
        unitary_support=getattr(device, "_to_matrix_ops", None),
    )

    return capabilities
