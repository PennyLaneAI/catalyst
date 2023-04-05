"""
Unit tests for CompilerDriver class
"""

import os
import sys
import warnings
import pytest

import pennylane as qml
from catalyst import qjit
from catalyst.compiler import Compiler
from catalyst.compiler import CompilerDriver
from catalyst.compiler import MHLOPass
from catalyst.compiler import QuantumCompilationPass
from catalyst.compiler import BufferizationPass
from catalyst.compiler import MLIRToLLVMDialect
from catalyst.compiler import LLVMDialectToLLVMIR
from catalyst.compiler import LLVMIRToObjectFile
from catalyst.jax_tracer import get_mlir


class TestCompilerOptions:
    """Unit test for Compiler class."""

    def test_catalyst_cc_available(self, monkeypatch):
        """Test that the compiler resolution order contains the preferred compiler and no warnings
        are emitted"""
        compiler = "c99"
        monkeypatch.setenv("CATALYST_CC", compiler)
        # If a warning is emitted, raise an error.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # pylint: disable=protected-access
            compilers = CompilerDriver._get_compiler_fallback_order([])
            assert compiler in compilers

    @pytest.mark.parametrize("logfile", [("stdout"), ("stderr"), (None)])
    def test_verbose_compilation(self, logfile, capsys):
        """Test verbose compilation mode"""

        if logfile is not None:
            logfile = getattr(sys, logfile)

        verbose = logfile is not None

        @qjit(verbose=verbose, logfile=logfile)
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def workflow():
            qml.X(wires=1)
            return qml.state()

        workflow()
        capture_result = capsys.readouterr()
        capture = capture_result.out + capture_result.err
        assert ("[RUNNING]" in capture) if verbose else ("[RUNNING]" not in capture)


class TestCompilerWarnings:
    """Test compiler's warning messages."""

    def test_catalyst_cc_unavailable_warning(self, monkeypatch):
        """Test that a warning is emitted when the preferred compiler is not in PATH."""
        monkeypatch.setenv("CATALYST_CC", "this-binary-does-not-exist")
        with pytest.warns(UserWarning, match="User defined compiler.* is not in PATH."):
            # pylint: disable=protected-access
            CompilerDriver._get_compiler_fallback_order([])

    def test_compiler_failed_warning(self):
        """Test that a warning is emitted when a compiler failed."""
        with pytest.warns(UserWarning, match="Compiler .* failed .*"):
            # pylint: disable=protected-access
            CompilerDriver._attempt_link("cc", [""], "in.o", "out.so", None)


class TestCompilerErrors:
    """Test compiler's error messages."""

    def test_link_fail_exception(self):
        """Test that an exception is raised when all compiler possibilities are exhausted."""
        with pytest.raises(EnvironmentError, match="Unable to link .*"):
            CompilerDriver.run("in.o", fallback_compilers=["this-binary-does-not-exist"])

    def test_lower_mhlo_input_validation(self):
        """Test if the function detects wrong extensions"""
        with pytest.raises(ValueError, match="is not an MLIR file"):
            MHLOPass.run("file-name.nomlir")

    def test_quantum_compilation_input_validation(self):
        """Test if the function detects wrong extensions"""
        with pytest.raises(ValueError, match="is not an MLIR file"):
            QuantumCompilationPass.run("file-name.nomlir")

    def test_bufferize_tensors_input_validation(self):
        """Test if the function detects wrong extensions"""
        with pytest.raises(ValueError, match="is not an MLIR file"):
            BufferizationPass.run("file-name.nomlir")

    def test_lower_all_to_llvm_input_validation(self):
        """Test if the function detects wrong extensions"""
        with pytest.raises(ValueError, match="is not a bufferized MLIR file"):
            MLIRToLLVMDialect.run("file-name.nobuff.mlir")

    def test_convert_mlir_to_llvmir_input_validation(self):
        """Test if the function detects wrong extensions"""
        with pytest.raises(ValueError, match="is not an LLVM dialect MLIR file"):
            LLVMDialectToLLVMIR.run("file-name.nollvm.mlir")

    def test_compile_llvmir_input_validation(self):
        """Test if the function detects wrong extensions"""
        with pytest.raises(ValueError, match="is not an LLVMIR file"):
            LLVMIRToObjectFile.run("file-name.noll")

    def test_link_lightning_runtime_input_validation(self):
        """Test if the function detects wrong extensions"""
        with pytest.raises(ValueError, match="is not an object file"):
            CompilerDriver.run("file-name.noo")

    def test_attempts_to_get_inexistent_intermediate_file(self):
        """Test for error raised if user request intermediate file that doesn't exist."""
        compiler = Compiler()
        with pytest.raises(ValueError, match="pass .* not found."):
            compiler.get_output_of("inexistent-file")


# pylint: disable=too-few-public-methods
class TestCompilerState:
    """Test states that the compiler can reach."""

    def test_print_stages(self):
        """Test that after compiling the intermediate files exist."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def workflow():
            qml.X(wires=1)
            return qml.state()

        mlir_module, _ = get_mlir(workflow)
        compiler = Compiler()
        compiler.run(mlir_module)
        compiler.get_output_of("MHLOPass")
        compiler.get_output_of("QuantumCompilationPass")
        compiler.get_output_of("BufferizationPass")
        compiler.get_output_of("MLIRToLLVMDialect")
        compiler.get_output_of("LLVMDialectToLLVMIR")

    def test_workspace_keep_intermediate(self):
        """Test cwd's has been modified with folder containing intermediate results"""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def workflow():
            qml.X(wires=1)
            return qml.state()

        mlir_module, _ = get_mlir(workflow)
        # This means that we are not running any pass.
        passes = []
        identity_compiler = Compiler()
        identity_compiler.run(mlir_module, keep_intermediate=True, passes=passes)
        directory = os.path.join(os.getcwd(), workflow.__name__)
        assert os.path.exists(directory)
        files = os.listdir(directory)
        # The directory is non-empty. Should at least contain the original .mlir file
        assert files

    def test_workspace_temporary(self):
        """Test temporary directory has been modified with folder containing intermediate results"""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def workflow():
            qml.X(wires=1)
            return qml.state()

        mlir_module, _ = get_mlir(workflow)
        # This means that we are not running any pass.
        passes = []
        identity_compiler = Compiler()
        identity_compiler.run(mlir_module, passes=passes)
        files = os.listdir(identity_compiler.workspace.name)
        # The directory is non-empty. Should at least contain the original .mlir file
        assert files
