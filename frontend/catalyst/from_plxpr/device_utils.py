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
"""
This submodule defines utilities for device preprocessing for from_plxpr.
"""

# pylint: disable=too-many-arguments, too-many-positional-arguments
import warnings

import pennylane as qp
from pennylane.devices.capabilities import DeviceCapabilities, ExecutionCondition
from pennylane.devices.execution_config import MCM_METHOD, POSTSELECT_MODE, ExecutionConfig
from pennylane.transforms import (
    diagonalize_measurements,
    split_non_commuting,
    split_to_single_terms,
)
from pennylane.transforms.core import BoundTransform, Transform

from catalyst.device.decomposition import (
    measurements_from_counts,
    measurements_from_samples,
)

# pylint: disable=protected-access
from catalyst.device.qjit_device import (
    _load_device_capabilities,
    filter_device_capabilities_with_shots,
    get_qjit_device_capabilities,
)
from catalyst.device.verification import (
    validate_measurements,
    validate_observables_adjoint_diff,
    validate_observables_parameter_shift,
    verify_no_state_variance_returns,
    verify_operations,
)
from catalyst.utils.exceptions import CompileError

_named_obs_dict = {
    "PauliX": qp.X,
    "PauliY": qp.Y,
    "PauliZ": qp.Z,
    "Hadamard": qp.Hadamard,
}


def create_device_preprocessing_pipeline(
    device: qp.devices.Device, execution_config: ExecutionConfig, shots: int, warn: bool = True
) -> list[BoundTransform]:
    """Create a pipeline of device preprocessing transforms for lowering QNodes."""
    shots_present = qp.math.is_abstract(shots) or shots != 0
    raw_capabilities: DeviceCapabilities = get_qjit_device_capabilities(
        _load_device_capabilities(device)
    )
    capabilities: DeviceCapabilities = filter_device_capabilities_with_shots(
        raw_capabilities,
        shots_present=shots_present,
        unitary_support=getattr(device, "_to_matrix_ops", None),
    )

    finite_shots_only = all(
        ExecutionCondition.FINITE_SHOTS_ONLY in conditions
        for conditions in capabilities.measurement_processes.values()
    )
    if not shots_present and finite_shots_only:
        raise CompileError(
            f"'{device.name}' only supports finite shot measurements, but "
            "analytic execution was requested."
        )

    # List of transforms to add to the MLIR transform sequence.
    pipeline = []
    # List of transforms that currently do not have native MLIR/xDSL implementations.
    # These will be temporarily substituted with empty xDSL passes.
    unsupported_transforms = []

    # All the below functions mutate 'pipeline' and 'unsupported_transforms' in-place.
    _mcm_preprocessing(
        pipeline, unsupported_transforms, device, execution_config, shots, capabilities
    )
    _measurements_preprocessing(
        pipeline, unsupported_transforms, device, execution_config, shots, capabilities
    )
    _operations_preprocessing(
        pipeline, unsupported_transforms, device, execution_config, shots, capabilities
    )
    _gradient_preprocessing(
        pipeline, unsupported_transforms, device, execution_config, shots, capabilities
    )

    if unsupported_transforms and warn:
        warnings.warn(
            "The following device-preprocessing transforms are currently not supported with "
            "'qp.qjit(capture=True)':\n"
            f"{unsupported_transforms}.\n"
            "They will be substituted with identity transforms.",
            UserWarning,
        )

    return tuple(pipeline)


# pylint: disable=unused-argument
def _mcm_preprocessing(
    pipeline: list[BoundTransform],
    unsupported_transforms: list[str],
    device: qp.devices.Device,
    execution_config: ExecutionConfig,
    shots: int,
    capabilities: DeviceCapabilities,
) -> None:
    """Preprocess mid-circuit measurements."""
    mcm_config = execution_config.mcm_config

    if mcm_config.postselect_mode not in (POSTSELECT_MODE.FILL_SHOTS, None):
        raise CompileError(
            f"postselect_mode='{mcm_config.postselect_mode.value} is not supported with "
            f"'qp.qjit(capture=True)'. Currently, only 'fill_shots' or None are supported."
        )

    if mcm_config.mcm_method == MCM_METHOD.ONE_SHOT:
        shots_present = qp.math.is_abstract(shots) or shots != 0
        if not shots_present:
            raise CompileError("Cannot use mcm_method='one-shot' with analytic mode.")
        pipeline.append(
            _safe_create_bound_transform(
                Transform(pass_name="dynamic-one-shot"), unsupported_transforms
            )
        )

    elif mcm_config.mcm_method not in (MCM_METHOD.SINGLE_BRANCH_STATISTICS, None):
        raise CompileError(
            f"mcm_method='{mcm_config.mcm_method.value}' is not supported with {device.name}"
            "when using 'qp.qjit(capture=True)'."
        )


# pylint: disable=unused-argument
def _measurements_preprocessing(
    pipeline: list[BoundTransform],
    unsupported_transforms: list[str],
    device: qp.devices.Device,
    execution_config: ExecutionConfig,
    shots: int,
    capabilities: DeviceCapabilities,
) -> None:
    """Preprocess terminal measurements."""
    # Check if split_non_commuting is needed
    need_split_non_commuting = False
    if not capabilities.non_commuting_observables:
        need_split_non_commuting = True
    elif not capabilities.observables:
        need_split_non_commuting = True
    elif not capabilities.measurement_processes.keys() - {"CountsMP", "SampleMP"}:
        need_split_non_commuting = True
    elif not {"PauliX", "PauliY", "PauliZ", "Hadamard"}.issubset(capabilities.observables):
        need_split_non_commuting = True

    if need_split_non_commuting:
        pipeline.append(_safe_create_bound_transform(split_non_commuting, unsupported_transforms))

    if not (need_split_non_commuting or "Sum" in capabilities.observables):
        pipeline.append(_safe_create_bound_transform(split_to_single_terms, unsupported_transforms))

    if not (
        capabilities.observables
        and capabilities.measurement_processes.keys() - {"CountsMP", "SampleMP"}
    ):
        if "SampleMP" in capabilities.measurement_processes:
            pipeline.append(
                _safe_create_bound_transform(
                    measurements_from_samples, unsupported_transforms, args=(device.wires,)
                )
            )
        elif "CountsMP" in capabilities.measurement_processes:
            pipeline.append(
                _safe_create_bound_transform(
                    measurements_from_counts, unsupported_transforms, args=(device.wires,)
                )
            )
        else:
            raise CompileError(f"{device.name} does not support observables or samples/counts.")

    else:
        supported_named_obs = set(_named_obs_dict.keys()).intersection(capabilities.observables)
        if len(supported_named_obs) != len(_named_obs_dict):
            supported_base_obs = [_named_obs_dict[obs] for obs in supported_named_obs]
            pipeline.append(
                _safe_create_bound_transform(
                    diagonalize_measurements,
                    unsupported_transforms,
                    kwargs={"supported_base_obs": supported_base_obs},
                )
            )


def _operations_preprocessing(
    pipeline: list[BoundTransform],
    unsupported_transforms: list[str],
    device: qp.devices.Device,
    execution_config: ExecutionConfig,
    shots: int,
    capabilities: DeviceCapabilities,
) -> None:
    """Preprocess operations."""
    # Currently, decomposition is not added to the pipeline. That will be addressed in the future.
    # Note that the below transforms, while not having native MLIR/xDSL implementations, will NOT
    # cause a warning to be triggered. This is because these are added unconditionally, and we do
    # not care that much about validation/verification transforms that much anyway.
    pipeline.append(
        _safe_create_bound_transform(
            verify_operations,
            unsupported_transforms,
            warn=False,
            # qjit_device should technically be an instance of QJITDevice,
            # but we'll ignore it for now.
            kwargs={"grad_method": execution_config.gradient_method, "qjit_device": device},
        )
    )
    pipeline.append(
        _safe_create_bound_transform(
            validate_measurements,
            unsupported_transforms,
            warn=False,
            kwargs={"capabilities": capabilities, "name": device.name, "shots": shots},
        )
    )


# pylint: disable=unused-argument
def _gradient_preprocessing(
    pipeline: list[BoundTransform],
    unsupported_transforms: list[str],
    device: qp.devices.Device,
    execution_config: ExecutionConfig,
    shots: int,
    capabilities: DeviceCapabilities,
) -> None:
    """Preprocess gradients."""
    if execution_config.gradient_method is not None:
        # Note that the below transform, while not having a native MLIR/xDSL implementation,
        # will NOT cause a warning to be triggered. This is because it added **most of the
        # time**, and we do not care that much about validation/verification transforms that
        # much anyway.
        pipeline.append(
            _safe_create_bound_transform(
                verify_no_state_variance_returns, unsupported_transforms, warn=False
            )
        )
    if execution_config.gradient_method == "adjoint":
        shots_present = qp.math.is_abstract(shots) or shots != 0
        if shots_present:
            raise CompileError("Cannot use diff_method='adjoint' with finite shots.")
        # qjit_device should technically be an instance of QJITDevice,
        # but we'll ignore it for now.
        pipeline.append(
            _safe_create_bound_transform(
                validate_observables_adjoint_diff,
                unsupported_transforms,
                args=(),
                kwargs={"qjit_device": device},
            )
        )
    elif execution_config.gradient_method == "parameter-shift":
        pipeline.append(
            _safe_create_bound_transform(
                validate_observables_parameter_shift, unsupported_transforms
            )
        )


def _safe_create_bound_transform(
    transform: Transform, unsupported_transforms: list[str], warn=True, args=(), kwargs=None
) -> BoundTransform:
    """Create a bound transform safely. If the transform is not supported at the MLIR/xDSL
    layer, an identity xDSL transform is inserted for it."""
    if not transform.pass_name:
        transform_name = transform.tape_transform.__name__
        if warn:
            unsupported_transforms.append(transform_name)

        empty_transform = Transform(pass_name="empty")
        return BoundTransform(empty_transform, kwargs={"key": transform_name})

    return BoundTransform(transform, args, kwargs)
