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

import pennylane as qp
import pytest
from pennylane.transforms.core import BoundTransform, CompilePipeline

from catalyst.passes.pass_api import (
    apply_pass,
    apply_pass_plugin,
    dict_to_compile_pipeline,
    pipeline,
)


class TestDictToCompilePipeline:
    """Tests the 'dict_to_compile_pipeline' helper function."""

    def test_input_is_none(self):
        """Tests that an empty pipeline is returned."""

        flags = ()
        valued_options = {}
        cp = dict_to_compile_pipeline(None, *flags, **valued_options)

        assert cp == CompilePipeline()

    @pytest.mark.parametrize("pass_name", ["merge_rotations", "merge-rotations"])
    def test_input_is_str(self, pass_name):
        """Tests that a str is properly processed."""

        flags = ()
        valued_options = {}
        cp = dict_to_compile_pipeline(pass_name, *flags, **valued_options)

        assert cp == CompilePipeline(qp.transform(pass_name="merge-rotations"))

    def test_input_is_dict(self):
        """Tests that a dict of passes gets processed properly."""

        flags = ()
        valued_options = {}
        pass_pipeline_dict = {
            "cancel_inverses": {},
            "gridsynth": {"epsilon": 42},
            "diagonalize_final_measurements": {
                "supported_obs": ("PauliX", "Hadamard"),
                "to_eigvals": False,
            },
        }
        cp = dict_to_compile_pipeline(pass_pipeline_dict, *flags, **valued_options)
        t1 = BoundTransform(qp.transform(pass_name="cancel-inverses"))
        t2 = BoundTransform(qp.transform(pass_name="gridsynth"), kwargs={"epsilon": 42})
        t3 = BoundTransform(
            qp.transform(pass_name="diagonalize-final-measurements"),
            kwargs={"supported_obs": ("PauliX", "Hadamard"), "to_eigvals": False},
        )
        exp_pipeline = CompilePipeline(t1, t2, t3)

        assert cp == exp_pipeline

    @pytest.mark.parametrize("pass_name", ["disentangle_cnot", "disentangle_swap"])
    def test_mixed_case_passes(self, pass_name):
        """Test that the passes with mixed cases are handled in a special way."""

        flags = ()
        valued_options = {}
        cp = dict_to_compile_pipeline(pass_name, *flags, **valued_options)

        test_map = {"disentangle_cnot": "disentangle-cnot", "disentangle_swap": "disentangle-swap"}

        assert cp == CompilePipeline(qp.transform(pass_name=test_map[pass_name]))


class TestPipeline:
    """Tests the 'pipeline' function."""

    def test_qnode_is_not_mutated(self):
        """Ensures the QNode is not mutated."""

        pass_pipeline = "merge-rotations"

        @qp.qnode(qp.device("null.qubit"))
        def circ():
            return qp.expval(qp.Z(0))

        assert circ.compile_pipeline == CompilePipeline()

        new_qn = pipeline(pass_pipeline)(circ)

        assert new_qn is not circ

    def test_appends_to_qnode_pipeline(self):
        """Tests that the functions correctly appends to the existing pipeline."""

        @qp.qnode(qp.device("null.qubit"))
        def circ():
            return qp.expval(qp.Z(0))

        assert circ.compile_pipeline == CompilePipeline()

        new_qn = pipeline("merge_rotations")(circ)
        assert len(new_qn.compile_pipeline) == 1
        assert new_qn.compile_pipeline == CompilePipeline(qp.transform(pass_name="merge-rotations"))

        new_new_qn = pipeline("cancel_inverses")(new_qn)
        assert len(new_new_qn.compile_pipeline) == 2
        assert new_new_qn.compile_pipeline == CompilePipeline(
            qp.transform(pass_name="merge-rotations"), qp.transform(pass_name="cancel-inverses")
        )


def test_apply_pass():
    """Test that 'apply_pass' correctly adds the transform to the transform sequence."""

    @apply_pass("diagonalize-final-measurements", supported_obs=("PauliX",))
    @apply_pass("cancel-inverses")
    @apply_pass("gridsynth", epsilon=42)
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    def qnode():
        qp.X(0)
        qp.X(0)
        return qp.state()

    @qp.qjit(target="mlir")
    def module():
        return qnode()

    assert "diagonalize-final-measurements" in module.mlir
    assert "cancel-inverses" in module.mlir
    assert "gridsynth" in module.mlir
    assert '"epsilon" = 42' in module.mlir


@pytest.mark.xfail(reason="to_eigvals = True is not supported yet")
def test_apply_pass_unsupported_inputs():
    """Test if the pipeline would break for an unsupported input for the
    diagonalize-final-measurements pass"""

    @qp.qjit(target="mlir")
    @apply_pass("diagonalize-final-measurements", to_eigvals=True)
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    def qnode():
        qp.X(0)
        qp.X(0)
        return qp.state()

    qnode()


def test_apply_pass_raise_error():
    """Test if an error would be raised for a non-commuting circuit."""

    @apply_pass("diagonalize-final-measurements")
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    def qnode():
        qp.X(0)
        qp.X(0)
        return qp.expval(qp.X(0) + qp.Z(0))

    with pytest.raises(
        qp.exceptions.CompileError,
        match="Observables are not qubit-wise commuting. Please apply the "
        "`split-non-commuting` pass first.",
    ):

        @qp.qjit(target="mlir")
        def module():
            return qnode()

        module()


def test_apply_pass_plugin(tmp_path):
    """Tests that a pass plugin can be used."""

    # Use pytest's https://docs.pytest.org/en/stable/how-to/tmp_path.html to
    # by pass the check for existence
    fake_plugin = tmp_path / "fake_plugin.so"
    fake_plugin.touch()

    @qp.qjit(target="mlir")
    @apply_pass_plugin(str(fake_plugin), "my-custom-pass")
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    def circuit():
        return qp.state()

    mlir = circuit.mlir
    assert 'transform.apply_registered_pass "my-custom-pass"' in mlir
