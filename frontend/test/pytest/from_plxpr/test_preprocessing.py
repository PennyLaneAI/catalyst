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
import pennylane as qml
import pytest
from pennylane.devices import NullQubit
from pennylane.devices.capabilities import (
    DeviceCapabilities,
    ExecutionCondition,
    OperatorProperties,
)

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


class TestGeneralPreprocessing:
    """Tests for general preprocessing."""

    def test_skip_preprocess(self):
        """Test that skip_preprocess=True causes device preprocessing to be skipped."""
        dev = qml.device("null.qubit", wires=4)

        # mcm_method="one-shot" should trigger at least one device preprocessing transform,
        # which is qml.transform(pass_name="dynamic-one-shot")
        @qml.qnode(dev, shots=1, mcm_method="one-shot")
        def f():
            _ = qml.measure(0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        cjaxpr = from_plxpr_no_warn(jaxpr, skip_preprocess=True)()

        pipeline = None
        for eqn in cjaxpr.eqns:
            if eqn.primitive == quantum_kernel_p:
                pipeline = eqn.params.get("pipeline", None)
                break

        assert pipeline is not None
        assert len(pipeline) == 0

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

        @qml.qnode(dev, shots=None)
        def f():
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        interpreter = from_plxpr_no_warn(jaxpr, skip_preprocess=False)

        with pytest.raises(CompileError, match="only supports finite shot measurements"):
            _ = interpreter()

    def test_unimplemented_transforms_warning(self):
        """Test that a warning is raised if device preprocessing requires transforms that
        are not yet available as MLIR/xDSL transforms."""
        dev = qml.device("null.qubit", wires=4)

        @qml.qnode(dev)
        def f():
            return qml.expval(qml.Z(0))

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

        dev = qml.device("null.qubit", wires=4)

        @qml.qnode(dev, shots=None, mcm_method="one-shot")
        def f():
            _ = qml.measure(0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        interpreter = from_plxpr_no_warn(jaxpr, skip_preprocess=False)

        with pytest.raises(
            CompileError, match="Cannot use mcm_method='one-shot' with analytic mode"
        ):
            _ = interpreter()

    @pytest.mark.parametrize("mcm_method", ["deferred", "tree-traversal"])
    def test_unusupported_mcm_method_error(self, mcm_method):
        """Test that an error is raised if an unsupported mcm_method is used."""

        dev = qml.device("null.qubit", wires=4)

        @qml.qnode(dev, shots=None, mcm_method=mcm_method)
        def f():
            _ = qml.measure(0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        interpreter = from_plxpr_no_warn(jaxpr, skip_preprocess=False)

        with pytest.raises(CompileError, match=f"mcm_method='{mcm_method}' is not supported"):
            _ = interpreter()

    @pytest.mark.parametrize("postselect_mode", ["hw-like", "pad-invalid-samples"])
    @pytest.mark.parametrize("mcm_method", ["one-shot", "single-branch-statistics", None])
    def test_invalid_postselect_mode_error(self, postselect_mode, mcm_method):
        """Test that an error is raised if postselect_mode other than "fill-shots" or
        None is used."""

        dev = qml.device("null.qubit", wires=4)

        @qml.qnode(dev, shots=1, mcm_method=mcm_method, postselect_mode=postselect_mode)
        def f():
            _ = qml.measure(0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        interpreter = from_plxpr_no_warn(jaxpr, skip_preprocess=False)

        with pytest.raises(
            CompileError, match=f"postselect_mode='{postselect_mode} is not supported"
        ):
            _ = interpreter()

    def test_dynamic_one_shot(self):
        """Test that the MLIR dynamic-one-shot transform is added to the pipeline if
        mcm_method="one-shot"."""

        dev = qml.device("null.qubit", wires=4)

        @qml.qnode(dev, shots=1, mcm_method="one-shot")
        def f():
            _ = qml.measure(0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        cjaxpr = from_plxpr_no_warn(jaxpr, skip_preprocess=False)()

        pipeline = None
        for eqn in cjaxpr.eqns:
            if eqn.primitive == quantum_kernel_p:
                pipeline = eqn.params.get("pipeline", None)
                break

        assert pipeline
        assert any(t.pass_name == "dynamic-one-shot" for t in pipeline)


class TestMeasurementPreprocessing:
    """Tests for preprocessing related to terminal measurements."""

    @pytest.mark.parametrize(
        "capabilities,transform_needed",
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
    def test_split_non_commuting(self, capabilities, transform_needed):
        """Test that split_non_commuting is added to the pipeline when needed."""
        dev = CapabilitiesDevice(wires=4)
        dev.capabilities = capabilities

        @qml.qnode(dev, shots=1)
        def f():
            qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        cjaxpr = from_plxpr_no_warn(jaxpr, skip_preprocess=False)()

        pipeline = None
        for eqn in cjaxpr.eqns:
            if eqn.primitive == quantum_kernel_p:
                pipeline = eqn.params.get("pipeline", None)
                break

        assert pipeline
        if transform_needed:
            assert any(t.pass_name == "split-non-commuting" for t in pipeline)
        else:
            assert not any(t.pass_name == "split-non-commuting" for t in pipeline)

    @pytest.mark.parametrize(
        "split_non_commuting_needed,sum_supported,transform_needed",
        [(True, False, False), (True, True, False), (False, True, False), (False, False, True)],
    )
    def test_split_to_single_terms(
        self, split_non_commuting_needed, sum_supported, transform_needed
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
            non_commuting_observables=not split_non_commuting_needed,
            observables=observables,
            measurement_processes={"ExpectationMP": [], "SampleMP": [], "CountsMP": []},
        )

        @qml.qnode(dev, shots=1)
        def f():
            qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        cjaxpr = from_plxpr_no_warn(jaxpr, skip_preprocess=False)()

        pipeline = None
        for eqn in cjaxpr.eqns:
            if eqn.primitive == quantum_kernel_p:
                pipeline = eqn.params.get("pipeline", None)
                break

        assert pipeline
        if transform_needed:
            assert any(t.pass_name == "split-to-single-terms" for t in pipeline)
        else:
            assert not any(t.pass_name == "split-to-single-terms" for t in pipeline)

    @pytest.mark.parametrize(
        "non_sample_supported,transform_needed", [(True, False), (False, True)]
    )
    def test_measurements_from_samples(self, non_sample_supported, transform_needed):
        """Test that measurements_from_samples is added to the pipeline when needed."""
        measurements = {"SampleMP": []}
        if non_sample_supported:
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

        @qml.qnode(dev, shots=1)
        def f():
            qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        cjaxpr = from_plxpr_no_warn(jaxpr, skip_preprocess=False)()

        pipeline = None
        for eqn in cjaxpr.eqns:
            if eqn.primitive == quantum_kernel_p:
                pipeline = eqn.params.get("pipeline", None)
                break

        assert pipeline
        if transform_needed:
            assert any(t.pass_name == "measurements-from-samples" for t in pipeline)
        else:
            assert not any(t.pass_name == "measurements-from-samples" for t in pipeline)

    @pytest.mark.parametrize(
        "non_counts_supported,transform_needed", [(True, False), (False, True)]
    )
    def test_measurments_from_counts(self, non_counts_supported, transform_needed):
        """Test that measurements_from_counts is added to the pipeline when needed."""
        measurements = {"CountsMP": []}
        if non_counts_supported:
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

        @qml.qnode(dev, shots=1)
        def f():
            qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        cjaxpr = from_plxpr_no_warn(jaxpr, skip_preprocess=False)()

        pipeline = None
        for eqn in cjaxpr.eqns:
            if eqn.primitive == quantum_kernel_p:
                pipeline = eqn.params.get("pipeline", None)
                break

        assert pipeline
        if transform_needed:
            assert any(t.pass_name == "measurements-from-counts" for t in pipeline)
        else:
            assert not any(t.pass_name == "measurements-from-counts" for t in pipeline)

    def test_unsupported_samples_counts_observables_error(self):
        """Test that an error is raised if a device doesn't support any observables,
        samples, or counts."""

        dev = CapabilitiesDevice(wires=4)
        dev.capabilities = DeviceCapabilities(observables={}, measurement_processes={})

        @qml.qnode(dev, shots=1)
        def f():
            qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        interpreter = from_plxpr_no_warn(jaxpr, skip_preprocess=False)

        with pytest.raises(CompileError, match="does not support observables or samples/counts"):
            _ = interpreter()

    @pytest.mark.parametrize(
        "supported_obs,transform_needed",
        [
            (["PauliZ", "Prod"], True),
            (["PauliX", "PauliY", "Hadamard"], True),
            (["PauliX", "PauliY", "PauliZ", "Hadamard"], False),
            (["PauliX", "PauliY", "PauliZ", "Hadamard", "Sum"], False),
        ],
    )
    def test_diagonalize_measurements(self, supported_obs, transform_needed):
        """Test that diagonalize_measurements is added to the pipeline when needed."""

        dev = CapabilitiesDevice(wires=4)
        dev.capabilities = DeviceCapabilities(
            observables={obs: OperatorProperties() for obs in supported_obs},
            measurement_processes={"ExpectationMP": [], "SampleMP": []},
        )

        @qml.qnode(dev, shots=1)
        def f():
            qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        cjaxpr = from_plxpr_no_warn(jaxpr, skip_preprocess=False)()

        pipeline = None
        for eqn in cjaxpr.eqns:
            if eqn.primitive == quantum_kernel_p:
                pipeline = eqn.params.get("pipeline", None)
                break

        assert pipeline
        if transform_needed:
            assert any(t.pass_name == "diagonalize-measurements" for t in pipeline)
        else:
            assert not any(t.pass_name == "diagonalize-measurements" for t in pipeline)


class TestOperationPreprocessing:
    """Tests for preprocessing related to operations."""

    def test_validation_transforms(self):
        """Test that transforms for validating operations and measurements are
        added to the pipeline."""
        dev = qml.device("null.qubit", wires=4)

        @qml.qnode(dev)
        def f():
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        cjaxpr = from_plxpr_no_warn(jaxpr, skip_preprocess=False)()

        pipeline = None
        for eqn in cjaxpr.eqns:
            if eqn.primitive == quantum_kernel_p:
                pipeline = eqn.params.get("pipeline", None)
                break

        assert pipeline
        assert any(t.pass_name == "verify-operations" for t in pipeline)
        assert any(t.pass_name == "validate-measurements" for t in pipeline)


class TestGradientPreprocessing:
    """Tests for preprocessing related to gradients."""

    @pytest.mark.parametrize("diff_method", ["adjoint", "parameter-shift", None])
    def test_gradient_obs_validation(self, diff_method):
        """Test that a transform for validating return types is added to the pipeline
        if gradients are requested."""
        dev = qml.device("lightning.qubit", wires=4)

        @qml.qnode(dev, diff_method=diff_method)
        def f():
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        cjaxpr = from_plxpr_no_warn(jaxpr, skip_preprocess=False)()

        pipeline = None
        for eqn in cjaxpr.eqns:
            if eqn.primitive == quantum_kernel_p:
                pipeline = eqn.params.get("pipeline", None)
                break

        assert pipeline
        if diff_method is not None:
            assert any(t.pass_name == "verify-no-state-variance-returns" for t in pipeline)
        else:
            assert not any(t.pass_name == "verify-no-state-variance-returns" for t in pipeline)

    @pytest.mark.parametrize("diff_method", ["adjoint", "parameter-shift", None])
    def test_adjoint_with_shots_error(self, diff_method):
        """Test that an error is raised if diff_method="adjoint" with finite shots."""
        dev = qml.device("lightning.qubit", wires=4)

        @qml.qnode(dev, diff_method=diff_method, shots=1)
        def f():
            return qml.expval(qml.Z(0))

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
        dev = qml.device("lightning.qubit", wires=4)

        @qml.qnode(dev, diff_method=diff_method)
        def f():
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        cjaxpr = from_plxpr_no_warn(jaxpr, skip_preprocess=False)()

        pipeline = None
        for eqn in cjaxpr.eqns:
            if eqn.primitive == quantum_kernel_p:
                pipeline = eqn.params.get("pipeline", None)
                break

        assert pipeline
        if diff_method == "adjoint":
            assert any(t.pass_name == "validate-observables-adjoint-diff" for t in pipeline)
        else:
            assert not any(t.pass_name == "validate-observables-adjoint-diff" for t in pipeline)

    @pytest.mark.parametrize("diff_method", ["adjoint", "parameter-shift", None])
    def test_adjoint_validation(self, diff_method):
        """Test that a transform for validating observables is added to the pipeline
        if diff_method="adjoint"."""
        dev = qml.device("lightning.qubit", wires=4)

        @qml.qnode(dev, diff_method=diff_method)
        def f():
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()
        cjaxpr = from_plxpr_no_warn(jaxpr, skip_preprocess=False)()

        pipeline = None
        for eqn in cjaxpr.eqns:
            if eqn.primitive == quantum_kernel_p:
                pipeline = eqn.params.get("pipeline", None)
                break

        assert pipeline
        if diff_method == "parameter-shift":
            assert any(t.pass_name == "validate-observables-parameter-shift" for t in pipeline)
        else:
            assert not any(t.pass_name == "validate-observables-parameter-shift" for t in pipeline)


class TestIntegration:
    """Integration tests for device preprocessing with program capture."""

    @pytest.mark.parametrize("skip_preprocess", [True, False, None])
    def test_qjit_skip_preprocess(self, skip_preprocess, recwarn):
        """Test that device preprocessing is added to the pass pipeline only if requested."""
        qjit_args = {"target": "mlir", "capture": True}
        if skip_preprocess is not None:
            qjit_args["skip_preprocess"] = skip_preprocess

        dev = qml.device("null.qubit", wires=4)

        @qml.transforms.merge_rotations
        @qml.qnode(dev, shots=1, mcm_method="one-shot", diff_method="parameter-shift")
        def f1():
            _ = qml.measure(0)
            return qml.expval(qml.Z(0))

        @qml.transforms.cancel_inverses
        @qml.qnode(dev, shots=None, diff_method="adjoint")
        def f2():
            return qml.expval(qml.Z(0))

        @qml.qjit(**qjit_args)
        def workflow():
            return f1() + f2()

        cjaxpr = workflow.jaxpr
        f1_pipeline = None
        f2_pipeline = None
        for eqn in cjaxpr.eqns:
            if eqn.primitive == quantum_kernel_p:
                if eqn.params["qnode"].__name__ == "f1":
                    f1_pipeline = eqn.params["pipeline"]
                elif eqn.params["qnode"].__name__ == "f2":
                    f2_pipeline = eqn.params["pipeline"]

        assert f1_pipeline
        assert f2_pipeline

        assert f1_pipeline[0].pass_name == "merge-rotations"
        assert f2_pipeline[0].pass_name == "cancel-inverses"

        if skip_preprocess:
            assert len(f1_pipeline) == 1
            assert len(f2_pipeline) == 1
            assert len(recwarn) == 0
        else:
            assert len(f1_pipeline) > 1
            assert len(f2_pipeline) > 1

            assert any(t.pass_name == "dynamic-one-shot" for t in f1_pipeline)
            assert any(t.pass_name == "verify-no-state-variance-returns" for t in f1_pipeline)
            assert any(t.pass_name == "validate-observables-parameter-shift" for t in f1_pipeline)

            assert any(t.pass_name == "verify-no-state-variance-returns" for t in f2_pipeline)
            assert any(t.pass_name == "validate-observables-adjoint-diff" for t in f2_pipeline)

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
