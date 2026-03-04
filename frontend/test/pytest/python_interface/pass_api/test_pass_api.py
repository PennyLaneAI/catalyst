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
"""Unit tests for the 'pass_api.py' file."""

import pennylane as qml
import pytest
from pennylane.transforms.core import BoundTransform, CompilePipeline

from catalyst.passes.pass_api import apply_pass, dict_to_compile_pipeline, pipeline
from catalyst.python_interface.transforms.quantum import cancel_inverses


class TestDictToCompilePipeline:
    """Tests the 'dict_to_compile_pipeline' helper function."""

    def test_input_is_none(self):
        """Tests that an empty pipeline is returned."""

        flags = ()
        valued_options = {}
        pipeline = dict_to_compile_pipeline(None, *flags, **valued_options)

        assert pipeline == CompilePipeline()

    @pytest.mark.parametrize("pass_name", ["merge_rotations", "merge-rotations"])
    def test_input_is_str(self, pass_name):
        """Tests that a str is properly processed."""

        flags = ()
        valued_options = {}
        pipeline = dict_to_compile_pipeline(pass_name, *flags, **valued_options)

        assert pipeline == CompilePipeline(qml.transform(pass_name="merge-rotations"))

    def test_input_is_dict(self):
        """Tests that a dict of passes gets processed properly."""

        flags = ()
        valued_options = {}
        pass_pipeline_dict = {
            "cancel_inverses": {},
            "gridsynth": {"epsilon": 42},
        }
        pipeline = dict_to_compile_pipeline(pass_pipeline_dict, *flags, **valued_options)
        t1 = BoundTransform(qml.transform(pass_name="cancel-inverses"))
        t2 = BoundTransform(qml.transform(pass_name="gridsynth"), kwargs={"epsilon": 42})
        exp_pipeline = CompilePipeline(t1, t2)

        assert pipeline == exp_pipeline

    @pytest.mark.parametrize("pass_name", ["disentangle_cnot", "disentangle_swap"])
    def test_mixed_case_passes(self, pass_name):
        """Test that the passes with mixed cases are handled in a special way."""

        flags = ()
        valued_options = {}
        pipeline = dict_to_compile_pipeline(pass_name, *flags, **valued_options)

        test_map = {"disentangle_cnot": "disentangle-CNOT", "disentangle_swap": "disentangle-SWAP"}

        assert pipeline == CompilePipeline(qml.transform(pass_name=test_map[pass_name]))


class TestPipeline:
    """Tests the 'pipeline' function."""

    def test_qnode_is_not_mutated(self):
        """Ensures the QNode is not mutated."""

        pass_pipeline = "merge-rotations"

        @qml.qnode(qml.device("null.qubit"))
        def circ():
            return qml.expval(qml.Z(0))

        assert circ.compile_pipeline == CompilePipeline()

        new_qn = pipeline(pass_pipeline)(circ)

        assert new_qn is not circ

    def test_appends_to_qnode_pipeline(self):
        """Tests that the functions correctly appends to the existing pipeline."""

        @qml.qnode(qml.device("null.qubit"))
        def circ():
            return qml.expval(qml.Z(0))

        assert circ.compile_pipeline == CompilePipeline()

        new_qn = pipeline("merge_rotations")(circ)
        assert len(new_qn.compile_pipeline) == 1
        assert new_qn.compile_pipeline == CompilePipeline(
            qml.transform(pass_name="merge-rotations")
        )

        new_new_qn = pipeline("cancel_inverses")(new_qn)
        assert len(new_qn.compile_pipeline) == 2
        assert new_qn.compile_pipeline == CompilePipeline(
            qml.transform(pass_name="merge-rotations"), qml.transform(pass_name="cancel-inverses")
        )


def test_apply_pass():
    """Test that 'apply_pass' correctly adds the transform to the transform sequence."""

    @apply_pass("cancel-inverses")
    @apply_pass("gridsynth", epsilon=42)
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def qnode():
        qml.X(0)
        qml.X(0)
        return qml.state()

    @qml.qjit(target="mlir")
    def module():
        return qnode()

    assert "cancel-inverses" in module.mlir
    assert "gridsynth" in module.mlir
    assert '"epsilon" = 42' in module.mlir
