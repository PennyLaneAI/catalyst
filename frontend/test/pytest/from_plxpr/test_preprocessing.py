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
Test suite for device preprocessing with program capture.
"""

from functools import partial

import jax
import pennylane as qp
import pytest
from pennylane.devices import NullQubit
from pennylane.devices.capabilities import (
    DeviceCapabilities,
    ExecutionCondition,
    OperatorProperties,
)

from catalyst.device.decomposition import measurements_from_counts, measurements_from_samples
from catalyst.from_plxpr import from_plxpr
from catalyst.jax_primitives import quantum_kernel_p
from catalyst.utils.exceptions import CompileError

pytestmark = pytest.mark.usefixtures("use_capture")
from_plxpr_no_warn = partial(from_plxpr, _preprocess_warn=False)


class CapabilitiesDevice(NullQubit):
    """Device class that allows setting capabilities on a per-instance basis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._capabilities = super().capabilities

    @property
    def capabilities(self) -> DeviceCapabilities:
        """Capabilities."""
        return self._capabilities

    @capabilities.setter
    def capabilities(self, obj: DeviceCapabilities):
        """Capabilities setter."""
        self._capabilities = obj
        self._capabilities.qjit_compatible = True

    @property
    def qjit_capabilities(self):
        """qjit_capabilities alias to capabilities. Used to override Catalyst behaviour
        where it prioritizes the TOML file over device.capabilities."""
        return self._capabilities


def get_pipelines(f, skip_preprocess=False):
    """Utility to get the transform pipelines for a QNode from jaxpr."""
    jaxpr = jax.make_jaxpr(f)()
    cjaxpr = from_plxpr_no_warn(jaxpr, skip_preprocess=skip_preprocess)()
    pipelines = None

    for eqn in cjaxpr.eqns:
        if eqn.primitive == quantum_kernel_p:
            pipelines = eqn.params.get("pipelines", None)
            break

    assert pipelines is not None
    return pipelines


class TestGeneralPreprocessing:
    """Tests for general preprocessing."""

    @pytest.mark.parametrize("skip_preprocess", [True, False])
    def test_skip_preprocess(self, skip_preprocess):
        """Test that skip_preprocess=True causes device preprocessing to be skipped."""
        dev = qp.device("null.qubit", wires=4)

        # mcm_method="one-shot" should trigger at least one device preprocessing transform,
        # which is qp.transform(pass_name="dynamic-one-shot")
        @qp.qnode(dev, shots=1, mcm_method="one-shot")
        def f():
            _ = qp.measure(0)
            return qp.expval(qp.Z(0))

        pipelines = get_pipelines(f, skip_preprocess=skip_preprocess)

        if skip_preprocess:
            assert len(pipelines) == 1
            assert pipelines[0][0] == "main"
        else:
            assert len(pipelines) == 2
            assert pipelines[0][0] == "main"
            assert pipelines[1][0] == "device"

    def test_finite_shots_only(self):
        """Test that an error is raised if trying to do an analytic execution with
        a finite-shots-only device."""
        dev = CapabilitiesDevice(wires=4)
        dev.capabilities = DeviceCapabilities(
            observables={"PauliZ": OperatorProperties()},
            measurement_processes={
                "ExpectationMP": [ExecutionCondition.FINITE_SHOTS_ONLY],
                "SampleMP": [ExecutionCondition.FINITE_SHOTS_ONLY],
                "CountsMP": [ExecutionCondition.FINITE_SHOTS_ONLY],
            },
        )

        @qp.qnode(dev, shots=None)
        def f():
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        interpreter = from_plxpr_no_warn(jaxpr, skip_preprocess=False)

        with pytest.raises(CompileError, match="only supports finite shot measurements"):
            _ = interpreter()

    def test_unimplemented_transforms_warning(self):
        """Test that a warning is raised if device preprocessing requires transforms that
        are not yet available as MLIR/xDSL transforms."""
        dev = qp.device("null.qubit", wires=4)

        # Adjoint will cause an unsupported transform to be added to device preprocessing
        @qp.qnode(dev, diff_method="adjoint")
        def f():
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        interpreter = from_plxpr(jaxpr, skip_preprocess=False)

        with pytest.warns(
            UserWarning,
            match="The following device-preprocessing transforms are currently not supported",
        ):
            _ = interpreter()


class TestMCMPreprocessing:
    """Tests for preprocessing related to mid-circuit measurements."""

    def test_one_shot_analytic_error(self):
        """Test than an error is raised if trying to use mcm_method="one-shot"
        with shots=None."""

        dev = qp.device("null.qubit", wires=4)

        @qp.qnode(dev, shots=None, mcm_method="one-shot")
        def f():
            _ = qp.measure(0)
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        interpreter = from_plxpr_no_warn(jaxpr, skip_preprocess=False)

        with pytest.raises(
            CompileError, match="Cannot use mcm_method='one-shot' with analytic mode"
        ):
            _ = interpreter()

    @pytest.mark.parametrize("mcm_method", ["deferred", "tree-traversal"])
    def test_unusupported_mcm_method_error(self, mcm_method):
        """Test that an error is raised if an unsupported mcm_method is used."""

        dev = qp.device("null.qubit", wires=4)

        @qp.qnode(dev, shots=None, mcm_method=mcm_method)
        def f():
            _ = qp.measure(0)
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        interpreter = from_plxpr_no_warn(jaxpr, skip_preprocess=False)

        with pytest.raises(CompileError, match=f"mcm_method='{mcm_method}' is not supported"):
            _ = interpreter()

    @pytest.mark.parametrize("postselect_mode", ["hw-like", "pad-invalid-samples"])
    @pytest.mark.parametrize("mcm_method", ["one-shot", "single-branch-statistics", None])
    def test_invalid_postselect_mode_error(self, postselect_mode, mcm_method):
        """Test that an error is raised if postselect_mode other than "fill-shots" or
        None is used."""

        dev = qp.device("null.qubit", wires=4)

        @qp.qnode(dev, shots=1, mcm_method=mcm_method, postselect_mode=postselect_mode)
        def f():
            _ = qp.measure(0)
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        interpreter = from_plxpr_no_warn(jaxpr, skip_preprocess=False)

        with pytest.raises(
            CompileError, match=f"postselect_mode='{postselect_mode} is not supported"
        ):
            _ = interpreter()

    def test_dynamic_one_shot(self):
        """Test that the MLIR dynamic-one-shot transform is added to the pipeline if
        mcm_method="one-shot"."""

        dev = qp.device("null.qubit", wires=4)

        @qp.qnode(dev, shots=1, mcm_method="one-shot")
        def f():
            _ = qp.measure(0)
            return qp.expval(qp.Z(0))

        pipelines = get_pipelines(f, skip_preprocess=False)
        device_pipeline = pipelines[1][1]
        assert any(t.pass_name == "dynamic-one-shot" for t in device_pipeline)


class TestMeasurementPreprocessing:
    """Tests for preprocessing related to terminal measurements."""

    @staticmethod
    def assert_transform_presence(pipeline, transform, transform_needed, is_empty_transform):
        """Utility function to assert whether a transform is present in the pipeline
        only when necessary."""
        if is_empty_transform:

            def cond_fn(t):
                return t.pass_name == "empty" and t.kwargs["key"] == transform.__name__

        else:

            def cond_fn(t):
                return t.pass_name == transform.pass_name

        has_transform = any(cond_fn(t) for t in pipeline)

        if transform_needed:
            assert has_transform
        else:
            assert not has_transform

    @pytest.mark.parametrize(
        "capabilities,needs_split_non_commuting",
        [
            (
                # Non-commuting observables unsupported
                DeviceCapabilities(
                    non_commuting_observables=False,
                    measurement_processes={"SampleMP": [], "CountsMP": []},
                ),
                True,
            ),
            (  # No observables supported
                DeviceCapabilities(
                    non_commuting_observables=True,
                    observables={},
                    measurement_processes={"SampleMP": [], "CountsMP": []},
                ),
                True,
            ),
            (  # No non-sample/counts observables supported
                DeviceCapabilities(
                    non_commuting_observables=True,
                    observables={
                        "PauliX": OperatorProperties(),
                        "PauliY": OperatorProperties(),
                        "PauliZ": OperatorProperties(),
                        "Hadamard": OperatorProperties(),
                    },
                    measurement_processes={"SampleMP": [], "CountsMP": []},
                ),
                True,
            ),
            (  # Not all named observables supported
                DeviceCapabilities(
                    non_commuting_observables=True,
                    observables={"PauliZ": OperatorProperties()},
                    measurement_processes={"ExpectationMP": [], "SampleMP": [], "CountsMP": []},
                ),
                True,
            ),
            (
                DeviceCapabilities(
                    non_commuting_observables=True,
                    observables={
                        "PauliX": OperatorProperties(),
                        "PauliY": OperatorProperties(),
                        "PauliZ": OperatorProperties(),
                        "Hadamard": OperatorProperties(),
                    },
                    measurement_processes={"ExpectationMP": [], "SampleMP": []},
                ),
                False,
            ),
        ],
    )
    def test_split_non_commuting(self, capabilities, needs_split_non_commuting):
        """Test that split_non_commuting is added to the pipeline when needed."""
        dev = CapabilitiesDevice(wires=4)
        dev.capabilities = capabilities

        @qp.qnode(dev, shots=1)
        def f():
            qp.expval(qp.Z(0))

        pipelines = get_pipelines(f, skip_preprocess=False)
        device_pipeline = pipelines[1][1]
        self.assert_transform_presence(
            device_pipeline,
            transform=qp.transforms.split_non_commuting,
            transform_needed=needs_split_non_commuting,
            is_empty_transform=True,
        )

    @pytest.mark.parametrize(
        "split_non_commuting_present,sum_supported,needs_split_to_single_terms",
        [(True, False, False), (True, True, False), (False, True, False), (False, False, True)],
    )
    def test_split_to_single_terms(
        self, split_non_commuting_present, sum_supported, needs_split_to_single_terms
    ):
        """Test that split_to_single_terms is added to the pipeline when needed."""
        observables = {
            "PauliX": OperatorProperties(),
            "PauliY": OperatorProperties(),
            "PauliZ": OperatorProperties(),
            "Hadamard": OperatorProperties(),
        }
        if sum_supported:
            observables["Sum"] = OperatorProperties()

        dev = CapabilitiesDevice(wires=4)
        dev.capabilities = DeviceCapabilities(
            non_commuting_observables=not split_non_commuting_present,
            observables=observables,
            measurement_processes={"ExpectationMP": [], "SampleMP": [], "CountsMP": []},
        )

        @qp.qnode(dev, shots=1)
        def f():
            qp.expval(qp.Z(0))

        pipelines = get_pipelines(f, skip_preprocess=False)
        device_pipeline = pipelines[1][1]
        self.assert_transform_presence(
            device_pipeline,
            transform=qp.transforms.split_to_single_terms,
            transform_needed=needs_split_to_single_terms,
            is_empty_transform=False,
        )

    @pytest.mark.parametrize("only_samples_supported", [True, False])
    def test_measurements_from_samples(self, only_samples_supported):
        """Test that measurements_from_samples is added to the pipeline when needed."""
        measurements = {"SampleMP": []}
        if not only_samples_supported:
            measurements["ExpectationMP"] = []

        dev = CapabilitiesDevice(wires=4)
        dev.capabilities = DeviceCapabilities(
            observables={
                "PauliX": OperatorProperties(),
                "PauliY": OperatorProperties(),
                "PauliZ": OperatorProperties(),
                "Hadamard": OperatorProperties(),
            },
            measurement_processes=measurements,
        )

        @qp.qnode(dev, shots=1)
        def f():
            qp.expval(qp.Z(0))

        pipelines = get_pipelines(f, skip_preprocess=False)
        device_pipeline = pipelines[1][1]
        self.assert_transform_presence(
            device_pipeline,
            transform=measurements_from_samples,
            transform_needed=only_samples_supported,
            is_empty_transform=True,
        )

    @pytest.mark.parametrize("only_counts_supported", [True, False])
    def test_measurements_from_counts(self, only_counts_supported):
        """Test that measurements_from_counts is added to the pipeline when needed."""
        measurements = {"CountsMP": []}
        if not only_counts_supported:
            measurements["SampleMP"] = []

        dev = CapabilitiesDevice(wires=4)
        dev.capabilities = DeviceCapabilities(
            observables={
                "PauliX": OperatorProperties(),
                "PauliY": OperatorProperties(),
                "PauliZ": OperatorProperties(),
                "Hadamard": OperatorProperties(),
            },
            measurement_processes=measurements,
        )

        @qp.qnode(dev, shots=1)
        def f():
            qp.expval(qp.Z(0))

        pipelines = get_pipelines(f, skip_preprocess=False)
        device_pipeline = pipelines[1][1]
        self.assert_transform_presence(
            device_pipeline,
            transform=measurements_from_counts,
            transform_needed=only_counts_supported,
            is_empty_transform=True,
        )

    def test_unsupported_samples_counts_observables_error(self):
        """Test that an error is raised if a device doesn't support any observables,
        samples, or counts."""

        dev = CapabilitiesDevice(wires=4)
        dev.capabilities = DeviceCapabilities(observables={}, measurement_processes={})

        @qp.qnode(dev, shots=1)
        def f():
            qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        interpreter = from_plxpr_no_warn(jaxpr, skip_preprocess=False)

        with pytest.raises(CompileError, match="does not support observables or samples/counts"):
            _ = interpreter()

    @pytest.mark.parametrize(
        "supported_obs,needs_diagonalize_measurements",
        [
            (["PauliZ", "Prod"], True),
            (["PauliX", "PauliY", "Hadamard"], True),
            (["PauliX", "PauliY", "PauliZ", "Hadamard"], False),
            (["PauliX", "PauliY", "PauliZ", "Hadamard", "Sum"], False),
        ],
    )
    def test_diagonalize_measurements(self, supported_obs, needs_diagonalize_measurements):
        """Test that diagonalize_measurements is added to the pipeline when needed."""

        dev = CapabilitiesDevice(wires=4)
        dev.capabilities = DeviceCapabilities(
            observables={obs: OperatorProperties() for obs in supported_obs},
            measurement_processes={"ExpectationMP": [], "SampleMP": []},
        )

        @qp.qnode(dev, shots=1)
        def f():
            qp.expval(qp.Z(0))

        pipelines = get_pipelines(f, skip_preprocess=False)
        device_pipeline = pipelines[1][1]
        self.assert_transform_presence(
            device_pipeline,
            transform=qp.transforms.diagonalize_measurements,
            transform_needed=needs_diagonalize_measurements,
            is_empty_transform=True,
        )


class TestOperationPreprocessing:
    """Tests for preprocessing related to operations."""

    def test_validation_transforms(self):
        """Test that transforms for validating operations and measurements are
        added to the pipeline."""
        dev = qp.device("null.qubit", wires=4)

        @qp.qnode(dev)
        def f():
            return qp.expval(qp.Z(0))

        pipelines = get_pipelines(f, skip_preprocess=False)
        device_pipeline = pipelines[1][1]
        assert any(
            t.pass_name == "empty" and t.kwargs["key"] == "verify_operations"
            for t in device_pipeline
        )
        assert any(
            t.pass_name == "empty" and t.kwargs["key"] == "validate_measurements"
            for t in device_pipeline
        )


class TestGradientPreprocessing:
    """Tests for preprocessing related to gradients."""

    @pytest.mark.parametrize("diff_method", ["adjoint", "parameter-shift", None])
    def test_gradient_obs_validation(self, diff_method):
        """Test that a transform for validating return types is added to the pipeline
        if gradients are requested."""
        dev = qp.device("lightning.qubit", wires=4)

        @qp.qnode(dev, diff_method=diff_method)
        def f():
            return qp.expval(qp.Z(0))

        pipelines = get_pipelines(f, skip_preprocess=False)
        device_pipeline = pipelines[1][1]

        if diff_method is not None:
            assert any(
                t.pass_name == "empty" and t.kwargs["key"] == "verify_no_state_variance_returns"
                for t in device_pipeline
            )
        else:
            assert not any(
                t.pass_name == "empty" and t.kwargs["key"] == "verify_no_state_variance_returns"
                for t in device_pipeline
            )

    @pytest.mark.parametrize("diff_method", ["adjoint", "parameter-shift", None])
    def test_adjoint_with_shots_error(self, diff_method):
        """Test that an error is raised if diff_method="adjoint" with finite shots."""
        dev = qp.device("lightning.qubit", wires=4)

        @qp.qnode(dev, diff_method=diff_method, shots=1)
        def f():
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        interpreter = from_plxpr_no_warn(jaxpr, skip_preprocess=False)

        if diff_method == "adjoint":
            with pytest.raises(
                CompileError, match="Cannot use diff_method='adjoint' with finite shots"
            ):
                _ = interpreter()
        else:
            _ = interpreter()

    @pytest.mark.parametrize("diff_method", ["adjoint", "parameter-shift", None])
    def test_parameter_shift_validation(self, diff_method):
        """Test that a transform for validating observables is added to the pipeline
        if diff_method="parameter-shift"."""
        dev = qp.device("lightning.qubit", wires=4)

        @qp.qnode(dev, diff_method=diff_method)
        def f():
            return qp.expval(qp.Z(0))

        pipelines = get_pipelines(f, skip_preprocess=False)
        device_pipeline = pipelines[1][1]

        if diff_method == "adjoint":
            assert any(
                t.pass_name == "empty" and t.kwargs["key"] == "validate_observables_adjoint_diff"
                for t in device_pipeline
            )
        else:
            assert not any(
                t.pass_name == "empty" and t.kwargs["key"] == "validate_observables_adjoint_diff"
                for t in device_pipeline
            )

    @pytest.mark.parametrize("diff_method", ["adjoint", "parameter-shift", None])
    def test_adjoint_validation(self, diff_method):
        """Test that a transform for validating observables is added to the pipeline
        if diff_method="adjoint"."""
        dev = qp.device("lightning.qubit", wires=4)

        @qp.qnode(dev, diff_method=diff_method)
        def f():
            return qp.expval(qp.Z(0))

        pipelines = get_pipelines(f, skip_preprocess=False)
        device_pipeline = pipelines[1][1]

        if diff_method == "parameter-shift":
            assert any(
                t.pass_name == "empty" and t.kwargs["key"] == "validate_observables_parameter_shift"
                for t in device_pipeline
            )
        else:
            assert not any(
                t.pass_name == "empty" and t.kwargs["key"] == "validate_observables_parameter_shift"
                for t in device_pipeline
            )


class TestIntegration:
    """Integration tests for device preprocessing with program capture."""

    @pytest.mark.parametrize("skip_preprocess", [True, False, None])
    def test_qjit_skip_preprocess(self, skip_preprocess, recwarn):
        """Test that device preprocessing is added to the pass pipeline only if requested."""
        qjit_args = {"target": "mlir", "capture": True}
        if skip_preprocess is not None:
            qjit_args["skip_preprocess"] = skip_preprocess

        dev = qp.device("null.qubit", wires=4)

        @qp.transforms.merge_rotations
        @qp.qnode(dev, shots=1, mcm_method="one-shot", diff_method="parameter-shift")
        def f1():
            _ = qp.measure(0)
            return qp.expval(qp.Z(0))

        @qp.transforms.cancel_inverses
        @qp.qnode(dev, shots=None, diff_method="adjoint")
        def f2():
            return qp.expval(qp.Z(0))

        @qp.qjit(**qjit_args)
        def workflow():
            return f1() + f2()

        cjaxpr = workflow.jaxpr
        f1_pipelines = None
        f2_pipelines = None
        for eqn in cjaxpr.eqns:
            if eqn.primitive == quantum_kernel_p:
                if eqn.params["qnode"].__name__ == "f1":
                    f1_pipelines = eqn.params["pipelines"]
                elif eqn.params["qnode"].__name__ == "f2":
                    f2_pipelines = eqn.params["pipelines"]

        assert f1_pipelines
        assert f2_pipelines

        assert f1_pipelines[0][1][0].pass_name == "merge-rotations"
        assert f2_pipelines[0][1][0].pass_name == "cancel-inverses"

        if skip_preprocess:
            assert len(f1_pipelines) == 1
            assert len(f2_pipelines) == 1
            assert len(recwarn) == 0
        else:
            assert len(f1_pipelines) > 1
            assert len(f2_pipelines) > 1

            f1_device_pipeline = f1_pipelines[1][1]
            f2_device_pipeline = f2_pipelines[1][1]

            assert any(t.pass_name == "dynamic-one-shot" for t in f1_device_pipeline)
            assert any(
                t.pass_name == "empty" and t.kwargs["key"] == "verify_no_state_variance_returns"
                for t in f1_device_pipeline
            )
            assert any(
                t.pass_name == "empty" and t.kwargs["key"] == "validate_observables_parameter_shift"
                for t in f1_device_pipeline
            )

            assert any(
                t.pass_name == "empty" and t.kwargs["key"] == "verify_no_state_variance_returns"
                for t in f2_device_pipeline
            )
            assert any(
                t.pass_name == "empty" and t.kwargs["key"] == "validate_observables_adjoint_diff"
                for t in f2_device_pipeline
            )

            # 2 warnings for 2 QNodes
            assert len(recwarn) == 2
            for _ in range(2):
                w = recwarn.pop()
                assert w.category == UserWarning
                assert (
                    "The following device-preprocessing transforms are currently not supported"
                    in str(w.message)
                )


if __name__ == "__main__":
    pytest.main(["-x", __file__])
