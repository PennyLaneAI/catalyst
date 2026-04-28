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
"""Unit test module for the MLIR graph generation in the Unified Compiler visualization module."""

from pathlib import Path
from shutil import which

import pennylane as qp
import pytest

from catalyst.python_interface.inspection import generate_mlir_graph
from catalyst.python_interface.transforms import (
    iterative_cancel_inverses_pass,
    merge_rotations_pass,
)

pytestmark = pytest.mark.xdsl
graphviz = pytest.importorskip("graphviz")

if which("dot") is None:
    pytest.skip(reason="Graphviz isn't installed.", allow_module_level=True)


@pytest.fixture(autouse=True)
def _chdir_tmp(monkeypatch, tmp_path: Path):
    """Ensure all tests run inside a temp directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def collect_files(tmp_path: Path) -> list[str]:
    """Return the set of generated SVG files."""
    out_dir = tmp_path / "mlir_generated_graphs"
    return sorted([f.name for f in out_dir.glob("*.svg")])


class TestMLIRGraph:
    """Test the MLIR graph generation"""

    def test_no_qjit_error(self):
        """Test that an error is raised if trying to use anything other than QJIT as
        an input."""

        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def f():
            qp.RX(0.1, 0)
            qp.RX(2.0, 0)
            qp.CNOT([0, 2])
            qp.CNOT([0, 2])
            return qp.state()

        gen = generate_mlir_graph(f)
        with pytest.raises(TypeError, match="Cannot generate MLIR module"):
            gen()

    @pytest.mark.parametrize("skip_preprocess", [True, False])
    def test_no_transforms(self, tmp_path: Path, skip_preprocess, capture_mode):
        """Test the MLIR graph is still generated when no transforms are applied"""
        if not capture_mode and skip_preprocess:
            pytest.skip("skip_preprocess only used when program capture is enabled.")

        @qp.qjit(skip_preprocess=skip_preprocess, capture=capture_mode)
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def f():
            qp.RX(0.1, 0)
            qp.RX(2.0, 0)
            qp.CNOT([0, 2])
            qp.CNOT([0, 2])
            return qp.state()

        generate_mlir_graph(f)()

        collected_files = collect_files(tmp_path)
        assert collected_files[0] == "QNode_level_0_no_transforms.svg"
        if not capture_mode or skip_preprocess:
            assert len(collected_files) == 1
        else:
            assert len(collected_files) > 1

    @pytest.mark.parametrize("skip_preprocess", [True, False])
    def test_xdsl_transforms_no_args(self, tmp_path: Path, skip_preprocess, capture_mode):
        """Test the MLIR graph generation with no arguments to the QNode with and without qjit"""
        if not capture_mode and skip_preprocess:
            pytest.skip("skip_preprocess only used when program capture is enabled.")

        @qp.qjit(skip_preprocess=skip_preprocess, capture=capture_mode)
        @merge_rotations_pass
        @iterative_cancel_inverses_pass
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def f():
            qp.RX(0.1, 0)
            qp.RX(2.0, 0)
            qp.CNOT([0, 2])
            qp.CNOT([0, 2])
            return qp.state()

        generate_mlir_graph(f)()

        collected_files = collect_files(tmp_path)
        expected_user = [
            "QNode_level_0_no_transforms.svg",
            "QNode_level_1_after_xdsl-cancel-inverses.svg",
            "QNode_level_2_after_xdsl-merge-rotations.svg",
        ]
        if not capture_mode or skip_preprocess:
            assert collected_files == expected_user
        else:
            assert collected_files[0 : len(expected_user)] == expected_user
            assert len(collected_files) > len(expected_user)

    @pytest.mark.parametrize("skip_preprocess", [True, False])
    def test_xdsl_transforms_args(self, tmp_path: Path, skip_preprocess, capture_mode):
        """Test the MLIR graph generation with arguments to the QNode for xDSL transforms"""
        if not capture_mode and skip_preprocess:
            pytest.skip("skip_preprocess only used when program capture is enabled.")

        @qp.qjit(skip_preprocess=skip_preprocess, capture=capture_mode)
        @merge_rotations_pass
        @iterative_cancel_inverses_pass
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def f(x, y, w1, w2):
            qp.RX(x, w1)
            qp.RX(y, w2)
            return qp.state()

        generate_mlir_graph(f)(0.1, 0.2, 0, 1)

        collected_files = collect_files(tmp_path)
        expected_user = [
            "QNode_level_0_no_transforms.svg",
            "QNode_level_1_after_xdsl-cancel-inverses.svg",
            "QNode_level_2_after_xdsl-merge-rotations.svg",
        ]
        if not capture_mode or skip_preprocess:
            assert collected_files == expected_user
        else:
            assert collected_files[0 : len(expected_user)] == expected_user
            assert len(collected_files) > len(expected_user)

    @pytest.mark.parametrize("skip_preprocess", [True, False])
    def test_catalyst_transforms_args(self, tmp_path: Path, skip_preprocess, capture_mode):
        """Test the MLIR graph generation with arguments to the QNode for catalyst transforms"""
        if not capture_mode and skip_preprocess:
            pytest.skip("skip_preprocess only used when program capture is enabled.")

        @qp.qjit(skip_preprocess=skip_preprocess, capture=capture_mode)
        @qp.transforms.merge_rotations
        @qp.transforms.cancel_inverses
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def f(x, y, w1, w2):
            qp.RX(x, w1)
            qp.RX(y, w2)
            return qp.state()

        generate_mlir_graph(f)(0.1, 0.2, 0, 1)

        collected_files = collect_files(tmp_path)
        expected_user = [
            "QNode_level_0_no_transforms.svg",
            "QNode_level_1_after_cancel-inverses.svg",
            "QNode_level_2_after_merge-rotations.svg",
        ]
        if not capture_mode or skip_preprocess:
            assert collected_files == expected_user
        else:
            assert collected_files[0 : len(expected_user)] == expected_user
            assert len(collected_files) > len(expected_user)

    @pytest.mark.parametrize("skip_preprocess", [True, False])
    def test_catalyst_xdsl_transforms_args(self, tmp_path: Path, skip_preprocess, capture_mode):
        """Test the MLIR graph generation with arguments to the QNode for catalyst and xDSL
        transforms"""
        if not capture_mode and skip_preprocess:
            pytest.skip("skip_preprocess only used when program capture is enabled.")

        @qp.qjit(skip_preprocess=skip_preprocess, capture=capture_mode)
        @qp.transforms.merge_rotations
        @iterative_cancel_inverses_pass
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def f(x, y, w1, w2):
            qp.RX(x, w1)
            qp.RX(y, w2)
            return qp.state()

        generate_mlir_graph(f)(0.1, 0.2, 0, 1)

        collected_files = collect_files(tmp_path)
        expected_user = [
            "QNode_level_0_no_transforms.svg",
            "QNode_level_1_after_xdsl-cancel-inverses.svg",
            "QNode_level_2_after_merge-rotations.svg",
        ]
        if not capture_mode or skip_preprocess:
            assert collected_files == expected_user
        else:
            assert collected_files[0 : len(expected_user)] == expected_user
            assert len(collected_files) > len(expected_user)

    @pytest.mark.parametrize("skip_preprocess", [True, False])
    def test_cond(self, tmp_path: Path, skip_preprocess, capture_mode):
        """Test the MLIR graph generation for a conditional"""
        if not capture_mode and skip_preprocess:
            pytest.skip("skip_preprocess only used when program capture is enabled.")

        @qp.qjit(skip_preprocess=skip_preprocess, capture=capture_mode)
        @merge_rotations_pass
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def f(pred, arg1, arg2):
            """Quantum circuit with conditional branches."""

            qp.RX(0.10, wires=0)

            def true_fn(arg1, arg2):
                qp.RY(arg1, wires=0)
                qp.RX(arg2, wires=0)
                qp.RZ(arg1, wires=0)

            def false_fn(arg1, arg2):
                qp.RX(arg1, wires=0)
                qp.RX(arg2, wires=0)

            qp.cond(pred > 0, true_fn, false_fn)(arg1, arg2)
            qp.RX(0.10, wires=0)
            return qp.expval(qp.Z(wires=0))

        generate_mlir_graph(f)(0.5, 0.1, 0.2)

        collected_files = collect_files(tmp_path)
        expected_user = [
            "QNode_level_0_no_transforms.svg",
            "QNode_level_1_after_xdsl-merge-rotations.svg",
        ]
        if not capture_mode or skip_preprocess:
            assert collected_files == expected_user
        else:
            assert collected_files[0 : len(expected_user)] == expected_user
            assert len(collected_files) > len(expected_user)

    @pytest.mark.parametrize("skip_preprocess", [True, False])
    def test_cond_with_mcm(self, tmp_path: Path, skip_preprocess, capture_mode):
        """Test the MLIR graph generation for a conditional with MCM"""
        if not capture_mode and skip_preprocess:
            pytest.skip("skip_preprocess only used when program capture is enabled.")

        def true_fn(arg):
            qp.RX(arg, 0)

        def false_fn(arg):
            qp.RY(3 * arg, 0)

        @qp.qjit(skip_preprocess=skip_preprocess, capture=capture_mode)
        @merge_rotations_pass
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def f(x, y):
            """Quantum circuit with conditional branches."""

            qp.RX(x, 0)
            m = qp.measure(0)

            qp.cond(m, true_fn, false_fn)(y)
            return qp.expval(qp.Z(0))

        generate_mlir_graph(f)(0.5, 0.1)

        collected_files = collect_files(tmp_path)
        expected_user = [
            "QNode_level_0_no_transforms.svg",
            "QNode_level_1_after_xdsl-merge-rotations.svg",
        ]
        if not capture_mode or skip_preprocess:
            assert collected_files == expected_user
        else:
            assert collected_files[0 : len(expected_user)] == expected_user
            assert len(collected_files) > len(expected_user)

    @pytest.mark.parametrize("skip_preprocess", [True, False])
    def test_for_loop(self, tmp_path: Path, skip_preprocess, capture_mode):
        """Test the MLIR graph generation for a for loop"""
        if not capture_mode and skip_preprocess:
            pytest.skip("skip_preprocess only used when program capture is enabled.")

        @qp.qjit(skip_preprocess=skip_preprocess, capture=capture_mode)
        @merge_rotations_pass
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def f():
            @qp.for_loop(0, 100)
            def loop(_):
                qp.RX(0.1, 0)
                qp.RX(0.1, 0)

            # pylint: disable=no-value-for-parameter
            loop()
            return qp.state()

        generate_mlir_graph(f)()

        collected_files = collect_files(tmp_path)
        expected_user = [
            "QNode_level_0_no_transforms.svg",
            "QNode_level_1_after_xdsl-merge-rotations.svg",
        ]
        if not capture_mode or skip_preprocess:
            assert collected_files == expected_user
        else:
            assert collected_files[0 : len(expected_user)] == expected_user
            assert len(collected_files) > len(expected_user)

    @pytest.mark.parametrize("skip_preprocess", [True, False])
    def test_while_loop(self, tmp_path: Path, skip_preprocess, capture_mode):
        """Test the MLIR graph generation for a while loop"""
        if not capture_mode and skip_preprocess:
            pytest.skip("skip_preprocess only used when program capture is enabled.")

        @qp.qjit(skip_preprocess=skip_preprocess, capture=capture_mode)
        @merge_rotations_pass
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def f(x):
            def cond_fn(x):
                return x < 2

            @qp.while_loop(cond_fn)
            def loop(x):
                return x**2

            loop(x)
            return qp.expval(qp.PauliZ(0))

        generate_mlir_graph(f)(0.5)

        collected_files = collect_files(tmp_path)
        expected_user = [
            "QNode_level_0_no_transforms.svg",
            "QNode_level_1_after_xdsl-merge-rotations.svg",
        ]
        if not capture_mode or skip_preprocess:
            assert collected_files == expected_user
        else:
            assert collected_files[0 : len(expected_user)] == expected_user
            assert len(collected_files) > len(expected_user)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
