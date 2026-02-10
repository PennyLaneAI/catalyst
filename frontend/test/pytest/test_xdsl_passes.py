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
Unit tests for xDSL pass-related functionality in the Compiler class.
"""
# pylint: disable=protected-access

import os
import pathlib
import tempfile
from unittest.mock import Mock

import pennylane as qml
import pytest

from catalyst import qjit
from catalyst.compiler import CompileOptions, Compiler
from catalyst.pipelines import KeepIntermediateLevel
from catalyst.utils.filesystem import Directory


@pytest.mark.xdsl
class TestHasXDSLPassesInTransformModules:
    """Test the has_xdsl_passes_in_transform_modules method and its exception handling."""

    @staticmethod
    def _create_mock_module(attrs):
        """Helper to create a mock module with nested operations."""
        nested_op = Mock()
        nested_op.attributes = attrs
        op = Mock()
        op.regions = [Mock()]
        op.regions[0].blocks = [Mock()]
        op.regions[0].blocks[0].operations = [nested_op]
        module = Mock()
        module.operation.regions = [Mock()]
        module.operation.regions[0].blocks = [Mock()]
        module.operation.regions[0].blocks[0].operations = [op]
        return module

    @pytest.mark.parametrize("exception", [AttributeError, KeyError, TypeError])
    def test_exception_handling(self, exception):
        """Test that exceptions are handled correctly."""

        class MockAttrs:
            """Mock attrs class."""

            def __contains__(self, key):
                """Mock contains method."""
                raise exception("Error")

            def keys(self):
                """Mock keys method."""
                raise exception("Error")

        compiler = Compiler()
        module = self._create_mock_module(MockAttrs())
        assert compiler.has_xdsl_passes_in_transform_modules(module) is False


@pytest.mark.xdsl
class TestCreatePassSaveCallback:
    """Test the _create_xdsl_pass_save_callback method."""

    def test_save_callback_workspace_none(self):
        """Test that callback returns None when workspace is None."""
        options = CompileOptions(keep_intermediate=KeepIntermediateLevel.CHANGED)
        compiler = Compiler(options=options)

        callback = compiler._create_xdsl_pass_save_callback(None)
        assert callback is None

    def test_save_callback_keep_intermediate_pipeline(self):
        """Test that callback returns None when keep_intermediate is PIPELINE."""
        options = CompileOptions(keep_intermediate=KeepIntermediateLevel.PIPELINE)
        compiler = Compiler(options=options)

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Directory(pathlib.Path(tmpdir))
            callback = compiler._create_xdsl_pass_save_callback(workspace)
            assert callback is None

    def test_save_callback_returns_callback(self):
        """Test that callback is returned when conditions are met."""
        options = CompileOptions(keep_intermediate=KeepIntermediateLevel.CHANGED)
        compiler = Compiler(options=options)

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Directory(pathlib.Path(tmpdir))
            callback = compiler._create_xdsl_pass_save_callback(workspace)
            assert callback is not None
            assert callable(callback)

    def test_pass_save_callback_skips_when_previous_pass_none(self):
        """Test that callback skips saving when previous_pass is None."""
        options = CompileOptions(keep_intermediate=KeepIntermediateLevel.CHANGED)
        compiler = Compiler(options=options)

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Directory(pathlib.Path(tmpdir))
            callback = compiler._create_xdsl_pass_save_callback(workspace)

            mock_module = Mock()
            callback(None, mock_module)

            expected_dir = os.path.join(str(workspace), "0_QuantumCompilationStage")
            assert os.path.exists(expected_dir)
            files = os.listdir(expected_dir)
            assert len(files) == 0

    def test_pass_save_callback_saves_ir_correctly(self):
        """Test that callback saves the IR correctly."""
        try:
            # pylint: disable-next=import-outside-toplevel
            from xdsl.dialects.builtin import ModuleOp
        except ImportError:
            pytest.skip("xdsl not available")

        options = CompileOptions(keep_intermediate=KeepIntermediateLevel.CHANGED)
        compiler = Compiler(options=options)

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Directory(pathlib.Path(tmpdir))
            callback = compiler._create_xdsl_pass_save_callback(workspace)

            module = ModuleOp([])

            # Create multiple passes
            pass1 = Mock()
            pass1.name = "Pass1"
            pass2 = Mock()
            pass2.name = "Pass2"
            pass3 = Mock()
            pass3.name = "Pass3"

            # Call callback multiple times
            callback(pass1, module)
            callback(pass2, module)
            callback(pass3, module)

            expected_dir = os.path.join(str(workspace), "0_QuantumCompilationStage")
            assert os.path.exists(expected_dir)
            assert os.path.isdir(expected_dir)
            files = sorted(os.listdir(expected_dir))
            assert len(files) == 3
            assert files[0] == "1_Pass1.mlir"
            assert files[1] == "2_Pass2.mlir"
            assert files[2] == "3_Pass3.mlir"


@pytest.mark.xdsl
class TestXDSLPassesIntegration:
    """Test the xDSL passes integration."""

    @pytest.mark.usefixtures("use_capture")
    def test_xdsl_passes_integration(self):
        """Test the xDSL passes integration."""
        # pylint: disable-next=import-outside-toplevel
        from catalyst.python_interface.transforms import merge_rotations_pass

        @qjit(keep_intermediate="changed", verbose=True)
        def workflow(x):
            @merge_rotations_pass
            @qml.transforms.cancel_inverses
            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def f(_x):
                qml.RX(_x, 0)
                qml.RX(1.6, 0)
                qml.Hadamard(1)
                qml.Hadamard(1)
                return qml.expval(qml.Z(0))

            return f(x)

        # Create tmp workspaces for intermediates to avoid CI race conditions
        workflow.use_cwd_for_workspace = False

        workflow.jit_compile((1.2,))
        workspace_path = str(workflow.workspace)
        assert os.path.exists(
            os.path.join(workspace_path, "0_QuantumCompilationStage", "1_cancel-inverses.mlir")
        )
        assert os.path.exists(
            os.path.join(workspace_path, "0_QuantumCompilationStage", "2_xdsl-merge-rotations.mlir")
        )
        workflow.workspace.cleanup()
