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
from pennylane.devices.capabilities import DeviceCapabilities

from catalyst.from_plxpr import from_plxpr
from catalyst.jax_primitives import quantum_kernel_p
from catalyst.utils.exceptions import CompileError

pytestmark = pytest.mark.usefixtures("use_capture")
from_plxpr_no_warn = partial(from_plxpr, _preprocess_warn=False)


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

    def test_split_non_commuting(self):
        """Test that split_non_commuting is added to the pipeline when needed."""

    def test_split_to_single_terms(self):
        """Test that split_to_single_terms is added to the pipeline when needed."""

    def test_measurements_from_samples(self):
        """Test that measurements_from_samples is added to the pipeline when needed."""

    def test_measurments_from_counts(self):
        """Test that measurements_from_counts is added to the pipeline when needed."""

    def test_unsupported_samples_counts_observables_error(self):
        """Test that an error is raised if a device doesn't support any observables,
        samples, or counts."""

    def test_diagonalize_measurements(self):
        """Test that diagonalize_measurements is added to the pipeline when needed."""


class TestOperationPreprocessing:
    """Tests for preprocessing related to operations."""

    def test_validation_transforms(self):
        """Test that transforms for validating operations and measurements are
        added to the pipeline."""


class TestGradientPreprocessing:
    """Tests for preprocessing related to gradients."""

    def test_gradient_obs_validation(self):
        """Test that a transform for validating return types is added to the pipeline
        if gradients are requested."""

    def test_adjoint_with_shots_error(self):
        """Test that an error is raised if diff_method="adjoint" with finite shots."""

    def test_parameter_shift_validation(self):
        """Test that a transform for validating observables is added to the pipeline
        if diff_method="parameter-shift"."""


class TestIntegration:
    """Integration tests for device preprocessing with program capture."""

    @pytest.mark.parametrize("skip_preprocess", [True, False])
    def test_from_plxpr_preprocessing(self, skip_preprocess):
        """Device preprocessing is added to the pass pipeline if requested."""

    @pytest.mark.parametrize("skip_preprocess", [True, False])
    def test_qjit_preprocessing(self, skip_preprocess):
        """Device preprocessing is added to the pass pipeline if requested."""
