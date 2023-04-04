"""
Unit tests for CompilerDriver class
"""

import sys
import warnings
import subprocess
import pytest

import pennylane as qml
from catalyst import qjit
from catalyst.compiler import CompileOptions
from catalyst.compiler import CompilerDriver
from catalyst.compiler import MHLOPass
from catalyst.compiler import QuantumCompilationPass
from catalyst.compiler import BufferizationPass
from catalyst.compiler import MLIRToLLVMDialect
from catalyst.compiler import LLVMDialectToLLVMIR
from catalyst.compiler import LLVMIRToObjectFile


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
    def test_catalyst_cc_unavailable_warning(self, monkeypatch):
        """Test that a warning is emitted when the preferred compiler is not in PATH."""
        monkeypatch.setenv("CATALYST_CC", "this-binary-does-not-exist")
        with pytest.warns(UserWarning, match="User defined compiler.* is not in PATH."):
            # pylint: disable=protected-access
            CompilerDriver._get_compiler_fallback_order([])

    @pytest.mark.parametrize("verbose", [True, False])
    def test_compiler_failed_warning(self, verbose):
        """Test that a warning is emitted when a compiler failed."""
        with pytest.warns(UserWarning, match="Compiler .* failed .*"):
            CompilerDriver._attempt_link("cc", [""], "in.o", "out.so", None)


class TestCompilerErrors:
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


class TestCompilerState:
    def test_print_stages(self):
        @qjit(keep_intermediate=True)
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def workflow():
            qml.X(wires=1)
            return qml.state()

        # The test here is just that these files exist
        # and can therefore be printed.
        workflow.print_stage("MHLOPass")
        workflow.print_stage("QuantumCompilationPass")
        workflow.print_stage("BufferizationPass")
        workflow.print_stage("MLIRToLLVMDialect")
        workflow.print_stage("LLVMDialectToLLVMIR")
