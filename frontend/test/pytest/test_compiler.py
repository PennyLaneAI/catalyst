"""
Unit tests for CompilerDriver class
"""

import sys
import warnings

import pytest

import pennylane as qml
from catalyst import qjit
from catalyst.compiler import (
    CompilerDriver,
    CompileOptions,
    bufferize_tensors,
    compile_llvmir,
    convert_mlir_to_llvmir,
    link_lightning_runtime,
    lower_all_to_llvm,
    lower_mhlo_to_linalg,
    transform_quantum_ir,
)
from catalyst.compiler import MHLOPass
from catalyst.compiler import CompilerDriver
import subprocess


class TestMHLOPass:
    """Unit test for MHLOPass class."""

    def test_run_fails(self):
        with pytest.raises(subprocess.CalledProcessError):
            MHLOPass.run("non-existing.mlir")


class TestCompilerDriver:
    """Unit test for CompilerDriver class."""

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

    def test_catalyst_cc_unavailable_warning(self, monkeypatch):
        """Test that a warning is emitted when the preferred compiler is not in PATH."""
        monkeypatch.setenv("CATALYST_CC", "this-binary-does-not-exist")
        with pytest.warns(UserWarning, match="User defined compiler.* is not in PATH."):
            # pylint: disable=protected-access
            CompilerDriver._get_compiler_fallback_order([])

    @pytest.mark.parametrize("verbose", [True, False])
    def test_compiler_failed_warning(self, verbose):
        """Test that a warning is emitted when a compiler failed."""
        compiler = "cc"
        with pytest.warns(UserWarning, match="Compiler .* failed .*"):
            # pylint: disable=protected-access
            CompilerDriver._attempt_link(compiler, [""], "in.o", "out.so", CompileOptions(verbose))

    def test_link_fail_exception(self):
        """Test that an exception is raised when all compiler possibilities are exhausted."""
        with pytest.raises(EnvironmentError, match="Unable to link .*"):
            CompilerDriver.link("in.o", "out.so", fallback_compilers=["this-binary-does-not-exist"])

    def test_lower_mhlo_input_validation(self):
        """Test if the function detects wrong extensions"""
        with pytest.raises(ValueError, match="is not an MLIR file"):
            lower_mhlo_to_linalg("file-name.nomlir")

    def test_quantum_compilation_input_validation(self):
        """Test if the function detects wrong extensions"""
        with pytest.raises(ValueError, match="is not an MLIR file"):
            transform_quantum_ir("file-name.nomlir")

    def test_bufferize_tensors_input_validation(self):
        """Test if the function detects wrong extensions"""
        with pytest.raises(ValueError, match="is not an MLIR file"):
            bufferize_tensors("file-name.nomlir")

    def test_lower_all_to_llvm_input_validation(self):
        """Test if the function detects wrong extensions"""
        with pytest.raises(ValueError, match="is not a bufferized MLIR file"):
            lower_all_to_llvm("file-name.nobuff.mlir")

    def test_convert_mlir_to_llvmir_input_validation(self):
        """Test if the function detects wrong extensions"""
        with pytest.raises(ValueError, match="is not an LLVM dialect MLIR file"):
            convert_mlir_to_llvmir("file-name.nollvm.mlir")

    def test_compile_llvmir_input_validation(self):
        """Test if the function detects wrong extensions"""
        with pytest.raises(ValueError, match="is not an LLVMIR file"):
            compile_llvmir("file-name.noll")

    def test_link_lightning_runtime_input_validation(self):
        """Test if the function detects wrong extensions"""
        with pytest.raises(ValueError, match="is not an object file"):
            link_lightning_runtime("file-name.noo")

    @pytest.mark.parametrize("verbose,logfile", [(True, "stdout"), (True, "stderr"), (False, None)])
    def test_verbose_compilation(self, verbose, logfile, capsys):
        """Test verbose compilation mode"""

        @qjit(verbose=verbose, logfile=getattr(sys, logfile) if logfile else None)
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def workflow():
            qml.X(wires=1)
            return qml.state()

        workflow()
        capture_result = capsys.readouterr()
        capture = capture_result.out + capture_result.err
        assert ("[RUNNING]" in capture) if verbose else ("[RUNNING]" not in capture)
