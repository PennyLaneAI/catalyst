import textwrap

import pennylane as qml
from pennylane.devices.capabilities import DeviceCapabilities

from catalyst.device.capabilities_utils import (
    filter_device_capabilities_with_shots as _filter_device_capabilities_with_shots,
)
from catalyst.device.capabilities_utils import load_device_capabilities as _load_device_capabilities
from catalyst.device.capabilities_utils import requires_shots as _requires_shots
from catalyst.jax_primitives_utils import _calculate_diff_method
from catalyst.utils.exceptions import CompileError


def calculate_diff_method(device, execution_config):
    """Calculate the diff method for the device."""
    return _calculate_diff_method(device, execution_config)


def load_device_capabilities(device) -> DeviceCapabilities:
    """Get the contents of the device config toml file."""
    return _load_device_capabilities(device)


def requires_shots(capabilities):
    """Checks if a device capabilities requires shots."""
    return _requires_shots(capabilities)


def filter_device_capabilities_with_shots(
    capabilities, shots_present, unitary_support=None
) -> DeviceCapabilities:
    """Process device capabilities depending on shots and unitary support."""
    return _filter_device_capabilities_with_shots(capabilities, shots_present, unitary_support)


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
