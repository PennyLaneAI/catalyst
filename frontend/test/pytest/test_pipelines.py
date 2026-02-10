# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Unit tests for pipeline options and utility functions.
"""

import numpy as np
import pennylane as qml
import pytest

from catalyst import qjit
from catalyst.pipelines import (
    CompileOptions,
    default_pipeline,
    insert_pass_after,
    insert_pass_before,
    insert_stage_after,
    insert_stage_before,
)


class TestDefaultPipeline:
    """Test the default pipeline."""

    def test_default_pipeline(self):
        """Trivial test that checks if the pipeline from ``default_pipeline()`` is identical to the
        one from ``CompileOptions.get_stages()``."""
        pipeline = default_pipeline()

        options = CompileOptions()
        pipeline_expected = options.get_stages()
        assert pipeline == pipeline_expected

    def test_default_pipeline_compilation(self):
        """Test that the compilation of a qjit-compiled circuit with all default compiler options
        and the compilation of the same circuit with the compilation pipeline explicitly set using
        ``default_pipeline()`` yield the same IR representation both before and after optimization.
        """
        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(angle: float):
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            qml.RX(angle, wires=0)
            qml.RX(angle / 2, wires=0)
            return qml.state()

        circuit_ref = qjit(circuit, target="mlir")
        circuit_default_pipeline = qjit(circuit, target="mlir", pipelines=default_pipeline())

        assert circuit_ref.mlir == circuit_default_pipeline.mlir
        assert circuit_ref.mlir_opt == circuit_default_pipeline.mlir_opt

    def test_default_pipeline_execution(self):
        """Test that the execution of a qjit-compiled circuit with all default compiler options and
        the execution of the same circuit with the compilation pipeline explicitly set using
        ``default_pipeline()`` yield the same results."""
        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(angle: float):
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            qml.RX(angle, wires=0)
            qml.RX(angle / 2, wires=0)
            return qml.state()

        circuit_ref = qjit(circuit)
        circuit_default_pipeline = qjit(circuit, pipelines=default_pipeline())

        angle = 0.5
        assert np.allclose(circuit_ref(angle), circuit_default_pipeline(angle))


class TestPassInsertion:
    """Test insertion of a pass into an existing pass pipeline."""

    # Tests for `insert_pass_after`
    def test_insert_pass_after(self):
        """Test the ``insert_pass_after`` function with valid inputs, inserting into the middle of
        the pipeline.
        """
        pipeline = ["pass1", "pass2", "pass3"]
        new_pass = "new_pass"
        insert_pass_after(pipeline, new_pass, ref_pass="pass1")

        pipeline_expected = ["pass1", new_pass, "pass2", "pass3"]
        assert pipeline == pipeline_expected

    def test_insert_pass_after_at_end(self):
        """Test the ``insert_pass_after`` function with valid inputs, inserting at the end of the
        pipeline.
        """
        pipeline = ["pass1", "pass2", "pass3"]
        new_pass = "new_pass"
        insert_pass_after(pipeline, new_pass, ref_pass="pass3")

        pipeline_expected = ["pass1", "pass2", "pass3", new_pass]
        assert pipeline == pipeline_expected

    def test_insert_pass_after_with_multiple_occurrences(self):
        """Test the ``insert_pass_after`` function where the reference pass appears more than once
        in the pipeline. In this case, the new pass should be inserted only once after the first
        occurrence of the reference pass.
        """
        pipeline = ["pass1", "pass2", "pass3", "pass2", "pass4"]
        new_pass = "new_pass"
        insert_pass_after(pipeline, new_pass, ref_pass="pass2")

        pipeline_expected = ["pass1", "pass2", "new_pass", "pass3", "pass2", "pass4"]
        assert pipeline == pipeline_expected

    @pytest.mark.parametrize("ref_pass", ["not_a_pass", 1, [], None])
    def test_insert_pass_after_invalid_ref_pass(self, ref_pass):
        """Test the ``insert_pass_after`` function with an invalid reference pass name."""
        pipeline = ["pass1", "pass2", "pass3"]
        new_pass = "new_pass"

        with pytest.raises(ValueError, match=f"Cannot insert pass '{new_pass}' into pipeline"):
            insert_pass_after(pipeline, new_pass, ref_pass=ref_pass)

    # Tests for `insert_pass_before`
    def test_insert_pass_before(self):
        """Test the ``insert_pass_before`` function with valid inputs, inserting into the middle of
        the pipeline.
        """
        pipeline = ["pass1", "pass2", "pass3"]
        new_pass = "new_pass"
        insert_pass_before(pipeline, new_pass, ref_pass="pass2")

        pipeline_expected = ["pass1", new_pass, "pass2", "pass3"]
        assert pipeline == pipeline_expected

    def test_insert_pass_before_at_beginning(self):
        """Test the ``insert_pass_before`` function with valid inputs, inserting at the beginning of
        the pipeline.
        """
        pipeline = ["pass1", "pass2", "pass3"]
        new_pass = "new_pass"
        insert_pass_before(pipeline, new_pass, ref_pass="pass1")

        pipeline_expected = [new_pass, "pass1", "pass2", "pass3"]
        assert pipeline == pipeline_expected

    def test_insert_pass_before_with_multiple_occurrences(self):
        """Test the ``insert_pass_before`` function where the reference pass appears more than once
        in the pipeline. In this case, the new pass should be inserted only once before the first
        occurrence of the reference pass.
        """
        pipeline = ["pass1", "pass2", "pass3", "pass2", "pass4"]
        new_pass = "new_pass"
        insert_pass_before(pipeline, new_pass, ref_pass="pass2")

        pipeline_expected = ["pass1", "new_pass", "pass2", "pass3", "pass2", "pass4"]
        assert pipeline == pipeline_expected

    @pytest.mark.parametrize("ref_pass", ["not_a_pass", 1, [], None])
    def test_insert_pass_before_invalid_ref_pass(self, ref_pass):
        """Test the ``insert_pass_before`` function with an invalid reference pass name."""
        pipeline = ["pass1", "pass2", "pass3"]
        new_pass = "new_pass"

        with pytest.raises(ValueError, match=f"Cannot insert pass '{new_pass}' into pipeline"):
            insert_pass_before(pipeline, new_pass, ref_pass=ref_pass)


class TestStageInsertion:
    """Test insertion of a compilation stage into an existing sequence of stages."""

    # Tests for `insert_stage_after`
    def test_insert_stage_after(self):
        """Test the ``insert_stage_after`` function with valid inputs, inserting into the middle of
        the sequence of stages.
        """
        stages = [("stage1", [""]), ("stage2", [""])]
        new_stage = ("new_stage", ["new_pass"])
        insert_stage_after(stages, new_stage, ref_stage="stage1")

        stages_expected = [("stage1", [""]), ("new_stage", ["new_pass"]), ("stage2", [""])]
        assert stages == stages_expected

    @pytest.mark.parametrize("ref_stage", ["not_a_stage", 1, [], None])
    def test_insert_stage_after_invalid_ref_pass(self, ref_stage):
        """Test the ``insert_stage_after`` function with an invalid reference stage name."""
        stages = [("stage1", [""]), ("stage2", [""])]
        new_stage = ("new_stage", ["new_pass"])

        with pytest.raises(ValueError, match=f"Cannot insert stage '{new_stage[0]}' into sequence"):
            insert_stage_after(stages, new_stage, ref_stage=ref_stage)

    @pytest.mark.parametrize("new_stage", ["", 1, [], None, ("",)])
    def test_insert_stage_after_invalid_new_stage(self, new_stage):
        """Test the ``insert_stage_after`` function with an invalid new stage type."""
        stages = [("stage1", [""]), ("stage2", [""])]
        ref_stage = "stage1"

        with pytest.raises(TypeError, match="The stage to insert must be a tuple"):
            insert_stage_after(stages, new_stage, ref_stage=ref_stage)

    # Tests for `insert_stage_before`
    def test_insert_stage_before(self):
        """Test the ``insert_stage_before`` function with valid inputs, inserting into the middle of
        the sequence of stages.
        """
        stages = [("stage1", [""]), ("stage2", [""])]
        new_stage = ("new_stage", ["new_pass"])
        insert_stage_before(stages, new_stage, ref_stage="stage2")

        stages_expected = [("stage1", [""]), ("new_stage", ["new_pass"]), ("stage2", [""])]
        assert stages == stages_expected

    @pytest.mark.parametrize("ref_stage", ["not_a_stage", 1, [], None])
    def test_insert_stage_before_invalid_ref_pass(self, ref_stage):
        """Test the ``insert_stage_before`` function with an invalid reference stage name."""
        stages = [("stage1", [""]), ("stage2", [""])]
        new_stage = ("new_stage", ["new_pass"])

        with pytest.raises(ValueError, match=f"Cannot insert stage '{new_stage[0]}' into sequence"):
            insert_stage_before(stages, new_stage, ref_stage=ref_stage)

    @pytest.mark.parametrize("new_stage", ["", 1, [], None, ("",), ("", [], [])])
    def test_insert_stage_before_invalid_new_stage(self, new_stage):
        """Test the ``insert_stage_before`` function with an invalid new stage type."""
        stages = [("stage1", [""]), ("stage2", [""])]
        ref_stage = "stage1"

        with pytest.raises(TypeError, match="The stage to insert must be a tuple"):
            insert_stage_before(stages, new_stage, ref_stage=ref_stage)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
