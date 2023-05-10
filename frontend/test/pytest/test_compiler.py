"""
Unit tests for CompilerDriver class
"""

import os
import shutil
import subprocess
import sys
import tempfile
import warnings

import pennylane as qml
import pytest

from catalyst import qjit
from catalyst.compiler import (
    BufferizationPass,
    CompileOptions,
    Compiler,
    CompilerDriver,
    LLVMDialectToLLVMIR,
    LLVMIRToObjectFile,
    MHLOPass,
    MLIRToLLVMDialect,
    PassPipeline,
    QuantumCompilationPass,
    JVPLoweringPass,
)
from catalyst.jax_tracer import get_mlir

# pylint: disable=missing-function-docstring


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
    def test_verbose_compilation(self, logfile, capsys, backend):
        """Test verbose compilation mode"""

        if logfile is not None:
            logfile = getattr(sys, logfile)

        verbose = logfile is not None

        @qjit(verbose=verbose, logfile=logfile)
        @qml.qnode(qml.device(backend, wires=1))
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

    def test_no_executable(self):
        """Test that executable was set from a custom PassPipeline."""

        class CustomClassWithNoExecutable(PassPipeline):
            """Custom pipeline with missing executable."""

            _default_flags = ["some-command-but-it-is-actually-a-flag"]

        with pytest.raises(ValueError, match="Executable not specified."):
            CustomClassWithNoExecutable.run("some-filename")

    def test_link_fail_exception(self):
        """Test that an exception is raised when all compiler possibilities are exhausted."""
        with pytest.raises(EnvironmentError, match="Unable to link .*"):
            with pytest.warns(UserWarning, match="Compiler c99"):
                CompilerDriver.run("in.o", fallback_compilers=["c99"])

    @pytest.mark.parametrize(
        "p", [MHLOPass, JVPLoweringPass, QuantumCompilationPass, BufferizationPass]
    )
    def test_mlir_input_validation(self, p):
        """Test if the function detects wrong extensions"""
        with pytest.raises(ValueError, match="is not an MLIR file"):
            p.run("file-name.nomlir")

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
        """Test return value if user request intermediate file that doesn't exist."""
        compiler = Compiler()
        result = compiler.get_output_of("inexistent-file")
        assert result is None

    def test_runtime_error(self):
        """Test that an exception is emitted when the runtime raises a C++ exception."""

        class CompileCXXException:
            """Class that overrides the program to be compiled."""

            _executable = "cc"
            _default_flags = ["-shared", "-fPIC", "-x", "c++"]

            @staticmethod
            def get_output_filename(infile):
                """Get the name of the output file based on the input file."""
                return infile.replace(".mlir", ".o")

            @staticmethod
            def run(infile, **_kwargs):
                """Run the compilation step."""
                contents = """
#include <stdexcept>
extern "C" {
  void _catalyst_pyface_jit_cpp_exception_test(void*, void*);
  void setup(int, char**);
  void teardown();
}
void setup(int argc, char** argv) {}
void teardown() {}
void _catalyst_pyface_jit_cpp_exception_test(void*, void*) {
  throw std::runtime_error("Hello world");
}
                """
                exe = CompileCXXException._executable
                flags = CompileCXXException._default_flags
                outfile = CompileCXXException.get_output_filename(infile)
                command = [exe] + flags + ["-o", outfile, "-"]
                with subprocess.Popen(command, stdin=subprocess.PIPE) as pipe:
                    pipe.communicate(input=bytes(contents, "UTF-8"))
                return outfile

        @qjit(
            pipelines=[CompileCXXException, CompilerDriver],
        )
        def cpp_exception_test():
            """A function that will be overwritten by CompileCXXException."""
            return None

        with pytest.raises(RuntimeError, match="Hello world"):
            cpp_exception_test()


class TestCompilerState:
    """Test states that the compiler can reach."""

    def test_print_stages(self, backend):
        """Test that after compiling the intermediate files exist."""

        @qml.qnode(qml.device(backend, wires=1))
        def workflow():
            qml.X(wires=1)
            return qml.state()

        mlir_module, _, _ = get_mlir(workflow)
        compiler = Compiler()
        compiler.run(mlir_module, CompileOptions())
        compiler.get_output_of("MHLOPass")
        compiler.get_output_of("QuantumCompilationPass")
        compiler.get_output_of("BufferizationPass")
        compiler.get_output_of("MLIRToLLVMDialect")
        compiler.get_output_of("LLVMDialectToLLVMIR")

    def test_workspace_keep_intermediate(self, backend):
        """Test cwd's has been modified with folder containing intermediate results"""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def workflow():
            qml.X(wires=1)
            return qml.state()

        mlir_module, _, _ = get_mlir(workflow)
        # This means that we are not running any pass.
        pipelines = []
        identity_compiler = Compiler()
        options = CompileOptions(keep_intermediate=True, pipelines=pipelines)
        identity_compiler.run(mlir_module, options)
        directory = os.path.join(os.getcwd(), workflow.__name__)
        assert os.path.exists(directory)
        files = os.listdir(directory)
        # The directory is non-empty. Should at least contain the original .mlir file
        assert files
        shutil.rmtree(directory)

    def test_workspace_temporary(self):
        """Test temporary directory has been modified with folder containing intermediate results"""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def workflow():
            qml.X(wires=1)
            return qml.state()

        mlir_module, _, _ = get_mlir(workflow)
        # This means that we are not running any pass.
        pipelines = []
        identity_compiler = Compiler()
        options = CompileOptions(pipelines=pipelines)
        identity_compiler.run(mlir_module, options)
        files = os.listdir(identity_compiler.workspace.name)
        # The directory is non-empty. Should at least contain the original .mlir file
        assert files

    def test_pass_with_output_name(self):
        """Test for making sure that outfile in arguments works"""

        class PassWithNoFlags(PassPipeline):
            """Pass pipeline without any flags."""

            _executable = "c99"
            _default_flags = []

        with tempfile.TemporaryDirectory() as workspace:
            filename = workspace + "a.c"
            outfilename = workspace + "a.out"
            with open(filename, "w", encoding="utf-8") as f:
                print("int main() {}", file=f)

            PassWithNoFlags.run(filename, outfile=outfilename)

            assert os.path.exists(outfilename)

    def test_pass_with_different_executable(self):
        """Test for making sure different executable works.

        It might be best in the future to remove this functionality and instead
        guarantee it from the start."""

        class C99(PassPipeline):
            """Pass pipeline using custom executable."""

            _executable = "c99"
            _default_flags = []

            @staticmethod
            def get_output_filename(infile):
                return infile.replace(".c", ".out")

        with tempfile.TemporaryDirectory() as workspace:
            filename = workspace + "a.c"
            expected_outfilename = workspace + "a.out"
            with open(filename, "w", encoding="utf-8") as f:
                print("int main() {}", file=f)

            observed_outfilename = C99.run(filename, executable="c89")

            assert observed_outfilename == expected_outfilename
            assert os.path.exists(observed_outfilename)

    def test_pass_with_flags(self):
        """Test with non-default flags."""

        class C99(PassPipeline):
            """Simple pass pipeline."""

            _executable = "c99"
            _default_flags = []

            @staticmethod
            def get_output_filename(infile):
                return infile.replace(".c", ".o")

        with tempfile.TemporaryDirectory() as workspace:
            filename = workspace + "a.c"
            expected_outfilename = workspace + "a.o"
            with open(filename, "w", encoding="utf-8") as f:
                print("int main() {}", file=f)

            observed_outfilename = C99.run(filename, flags=["-c"])

            assert observed_outfilename == expected_outfilename
            assert os.path.exists(observed_outfilename)

    def test_custom_compiler_pass_output(self):
        """Test that the output of a custom compiler pass is accessible."""

        class MyPass(PassPipeline):
            """Simple pass pipeline."""

            _executable = "echo"
            _default_flags = []

            @staticmethod
            def get_output_filename(infile):
                return infile.replace(".mlir", ".txt")

            @staticmethod
            def _run(_infile, outfile, executable, _flags, _options):
                cmd = [executable, "hi"]
                with open(outfile, "w", encoding="UTF-8") as f:
                    subprocess.run(cmd, stdout=f, check=True)

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def workflow():
            qml.X(wires=1)
            return qml.state()

        mlir_module, _, _ = get_mlir(workflow)
        compiler = Compiler()
        compiler.run(mlir_module, CompileOptions(pipelines=[MyPass]))
        result = compiler.get_output_of("MyPass")
        assert result == "hi\n"

    def test_compiler_driver_with_output_name(self):
        """Test with non-default output name."""
        with tempfile.TemporaryDirectory() as workspace:
            filename = workspace + "a.c"
            outfilename = workspace + "a.out"
            with open(filename, "w", encoding="utf-8") as f:
                print("int main() {}", file=f)

            CompilerDriver.run(filename, outfile=outfilename)

            assert os.path.exists(outfilename)

    def test_compiler_driver_with_flags(self):
        """Test with non-default flags."""

        class C99(PassPipeline):
            """Pass pipeline with custom flags."""

            _executable = "c99"
            _default_flags = ["-c"]

            @staticmethod
            def get_output_filename(infile):
                return infile.replace(".c", ".o")

        with tempfile.TemporaryDirectory() as workspace:
            filename = workspace + "a.c"
            with open(filename, "w", encoding="utf-8") as f:
                print("int main() {}", file=f)

            object_file = C99.run(filename)
            expected_outfilename = workspace + "a.so"
            observed_outfilename = CompilerDriver.run(object_file, flags=[])

            assert observed_outfilename == expected_outfilename
            assert os.path.exists(observed_outfilename)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
