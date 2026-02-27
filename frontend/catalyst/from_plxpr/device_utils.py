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
import warnings
from functools import lru_cache

import pennylane as qml
from pennylane.devices.capabilities import DeviceCapabilities, ExecutionCondition
from pennylane.devices.execution_config import MCM_METHOD, POSTSELECT_MODE, ExecutionConfig
from pennylane.transforms import (
    diagonalize_measurements,
    split_non_commuting,
    split_to_single_terms,
)
from pennylane.transforms.core import BoundTransform, Transform
from xdsl.passes import ModulePass

from catalyst.device.decomposition import (
    measurements_from_counts,
    measurements_from_samples,
)
from catalyst.device.qjit_device import (
    get_device_capabilities,
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
    "PauliX": qml.X,
    "PauliY": qml.Y,
    "PauliZ": qml.Z,
    "Hadamard": qml.Hadamard,
}


def create_device_preprocessing_pipeline(
    device: qml.devices.Device, execution_config: ExecutionConfig, shots: int, warn: bool = True
) -> list[BoundTransform]:
    """Create a pipeline of device preprocessing transforms for lowering QNodes."""
    capabilities: DeviceCapabilities = get_qjit_device_capabilities(
        get_device_capabilities(device, shots=shots)
    )

    finite_shots_only = all(
        ExecutionCondition.FINITE_SHOTS_ONLY in conditions
        for conditions in capabilities.measurement_processes.values()
    )
    if not shots and finite_shots_only:
        raise CompileError(
            f"'{device.name}' only supports finite shot measurements, but "
            "analytic execution was requested."
        )

    pipeline = []
    unsupported_transforms = []
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
            "'qml.qjit(capture=True)':\n"
            f"{unsupported_transforms}.\n"
            "They will be substituted with identity transforms.",
            UserWarning,
        )

    return pipeline


def _mcm_preprocessing(
    pipeline: list[BoundTransform],
    unsupported_transforms: list[str],
    device: qml.devices.Device,
    execution_config: ExecutionConfig,
    shots: int,
    capabilities: DeviceCapabilities,
) -> list[BoundTransform]:
    """Preprocess mid-circuit measurements."""
    mcm_config = execution_config.mcm_config
    if mcm_config.mcm_method == MCM_METHOD.ONE_SHOT:
        if not shots:
            raise CompileError("Cannot use mcm_method='one-shot' with analytic mode.")
        if mcm_config.postselect_mode not in (POSTSELECT_MODE.FILL_SHOTS, None):
            raise CompileError(
                f"postselect_mode='{mcm_config.postselect_mode.value} is not supported with "
                f"'qml.qjit(capture=True)' and mcm_method='{mcm_config.mcm_method.value}'. "
                # "Currently, only 'fill_shots' or None are supported."
            )
        pipeline.append(
            _safe_create_bound_transform(
                Transform(pass_name="dynamic-one-shot"), unsupported_transforms
            )
        )

    elif mcm_config.mcm_method not in (
        MCM_METHOD.SINGLE_BRANCH_STATISTICS,
        *capabilities.supported_mcm_methods,
        None,
    ):
        raise CompileError(
            f"mcm_method='{mcm_config.mcm_method.value}' is not supported with {device.name}"
            "when using 'qml.qjit(capture=True)'."
        )


def _measurements_preprocessing(
    pipeline: list[BoundTransform],
    unsupported_transforms: list[str],
    device: qml.devices.Device,
    execution_config: ExecutionConfig,
    shots: int,
    capabilities: DeviceCapabilities,
) -> list[BoundTransform]:
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
                    measurements_from_samples, unsupported_transforms, (device.wires,)
                )
            )
        elif "CountsMP" in capabilities.measurement_processes:
            pipeline.append(
                _safe_create_bound_transform(
                    measurements_from_counts, unsupported_transforms, (device.wires,)
                )
            )
        else:
            raise CompileError(f"{device.name} does not support observables or sample/counts.")

    else:
        supported_named_obs = set(_named_obs_dict.keys()).intersection(capabilities.observables)
        if len(supported_named_obs) != len(_named_obs_dict):
            supported_base_obs = [_named_obs_dict[obs] for obs in supported_named_obs]
            pipeline.append(
                _safe_create_bound_transform(
                    diagonalize_measurements,
                    unsupported_transforms,
                    {"supported_base_obs": supported_base_obs},
                )
            )


def _operations_preprocessing(
    pipeline: list[BoundTransform],
    unsupported_transforms: list[str],
    device: qml.devices.Device,
    execution_config: ExecutionConfig,
    shots: int,
    capabilities: DeviceCapabilities,
) -> list[BoundTransform]:
    """Preprocess operations."""
    pipeline.append(
        _safe_create_bound_transform(
            Transform(pass_name="decompose-lowering"), unsupported_transforms
        )
    )
    pipeline.append(
        _safe_create_bound_transform(
            verify_operations,
            unsupported_transforms,
            (),
            # qjit_device should technically be an instance of QJITDevice,
            # but we'll ignore it for now.
            {"grad_method": execution_config.gradient_method, "qjit_device": device},
        )
    )
    pipeline.append(
        _safe_create_bound_transform(
            validate_measurements, unsupported_transforms, (capabilities, device.name, shots)
        )
    )


def _gradient_preprocessing(
    pipeline: list[BoundTransform],
    unsupported_transforms: list[str],
    device: qml.devices.Device,
    execution_config: ExecutionConfig,
    shots: int,
    capabilities: DeviceCapabilities,
) -> list[BoundTransform]:
    """Preprocess gradients."""
    if execution_config.gradient_method is not None:
        pipeline.append(
            _safe_create_bound_transform(verify_no_state_variance_returns, unsupported_transforms)
        )
    if execution_config.gradient_method == "adjoint":
        if shots:
            raise CompileError(f"Cannot use diff_method='adjoint' with finite shots.")
        # qjit_device should technically be an instance of QJITDevice,
        # but we'll ignore it for now.
        pipeline.append(
            _safe_create_bound_transform(
                validate_observables_adjoint_diff,
                unsupported_transforms,
                (),
                {"qjit_device": device},
            )
        )
    elif execution_config.gradient_method == "parameter-shift":
        pipeline.append(
            _safe_create_bound_transform(
                validate_observables_parameter_shift, unsupported_transforms
            )
        )


def _safe_create_bound_transform(
    transform: Transform, unsupported_transforms: list[str], args=(), kwargs=None
):
    """Create a bound transform safely. If the transform is not supported at the MLIR/xDSL
    layer, an identity xDSL transform is inserted for it."""
    if not transform.pass_name:
        identity_transform = _create_identity_xdsl_transform(transform)
        unsupported_transforms.append(transform.tape_transform.__name__)
        return BoundTransform(identity_transform, kwargs={"dummy_transform": True})

    return BoundTransform(transform, args, kwargs)


@lru_cache
def _create_identity_xdsl_transform(original_transform: Transform | BoundTransform) -> Transform:
    """Create an xDSL transform to insert into the compile pipeline."""
    from catalyst.python_interface import compiler_transform

    # Force kebab-case for the transform name
    pass_name = original_transform.tape_transform.__name__.replace("_", "-")

    class NullPass(ModulePass):
        """Empty ModulePass to handle transforms with no MLIR/xDSL implementations."""

        name = pass_name
        dummy_transform: bool

        def __init__(self, dummy_transform):
            self.dummy_transform = dummy_transform

        def apply(self, ctx, op):
            """Apply the pass (do nothing)."""

    return compiler_transform(NullPass)
