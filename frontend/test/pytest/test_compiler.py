# Copyright 2023 Xanadu Quantum Technologies Inc.

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
Unit tests for LinkerDriver class
"""

import os
import pathlib
import platform
import subprocess
import sys
import tempfile
import warnings
from os.path import isfile

import pennylane as qml
import pytest

from catalyst import qjit
from catalyst.compilation_pipelines import WorkspaceManager
from catalyst.compiler import DEFAULT_PIPELINES, CompileOptions, Compiler, LinkerDriver
from catalyst.jax_tracer import trace_to_mlir
from catalyst.pennylane_extensions import measure, qfunc
from catalyst.utils.exceptions import CompileError
from catalyst.utils.filesystem import Directory

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
            compilers = LinkerDriver._get_compiler_fallback_order([])
            assert compiler in compilers

    @pytest.mark.parametrize(
        "logfile,keep_intermediate", [("stdout", True), ("stderr", False), (None, False)]
    )
    def test_verbose_compilation(self, logfile, keep_intermediate, capsys, backend):
        """Test verbose compilation mode"""

        if logfile is not None:
            logfile = getattr(sys, logfile)

        verbose = logfile is not None

        @qjit(verbose=verbose, logfile=logfile, keep_intermediate=keep_intermediate)
        @qml.qnode(qml.device(backend, wires=1))
        def workflow():
            qml.PauliX(wires=0)
            return qml.state()

        workflow()
        capture_result = capsys.readouterr()
        capture = capture_result.out + capture_result.err
        assert ("[SYSTEM]" in capture) if verbose else ("[SYSTEM]" not in capture)
        assert ("[LIB]" in capture) if verbose else ("[LIB]" not in capture)
        assert ("Dumping" in capture) if (verbose and keep_intermediate) else True
        workflow.workspace.cleanup()


class TestCompilerWarnings:
    """Test compiler's warning messages."""

    def test_catalyst_cc_unavailable_warning(self, monkeypatch):
        """Test that a warning is emitted when the preferred compiler is not in PATH."""
        monkeypatch.setenv("CATALYST_CC", "this-binary-does-not-exist")
        with pytest.warns(UserWarning, match="User defined compiler.* is not in PATH."):
            # pylint: disable=protected-access
            LinkerDriver._get_compiler_fallback_order([])

    def test_compiler_failed_warning(self):
        """Test that a warning is emitted when a compiler failed."""
        with pytest.warns(UserWarning, match="Compiler .* failed .*"):
            # pylint: disable=protected-access
            LinkerDriver._attempt_link("cc", [""], "in.o", "out.so", CompileOptions(verbose=True))


class TestCompilerErrors:
    """Test compiler's error messages."""

    def test_link_failure(self):
        """Test that an exception is raised when all compiler possibilities are exhausted."""
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", suffix=".o") as invalid_file:
            invalid_file.write("These are invalid contents.")
            invalid_file.flush()
            with pytest.raises(CompileError, match="Unable to link .*"):
                LinkerDriver.run(invalid_file.name, fallback_compilers=["cc"])

    def test_attempts_to_get_file_on_invalid_dir(self):
        """Test return value if user request intermediate file on a dir that doesn't exist.
        Or doesn't make sense.
        """
        compiler = Compiler()
        with pytest.raises(AssertionError, match="expects a Directory type"):
            compiler.run_from_ir("ir-placeholder", "ir-module-name", "inexistent-file")
        with pytest.raises(AssertionError, match="expects an existing directory"):
            compiler.run_from_ir(
                "ir-placeholder", "ir-module-name", Directory(pathlib.Path("a-name"))
            )

    def test_attempts_to_get_inexistent_intermediate_file(self):
        """Test return value if user request intermediate file that doesn't exist."""
        compiler = Compiler()
        result = compiler.get_output_of("inexistent-file")
        assert result is None

    def test_runtime_error(self, backend):
        """Test with non-default flags."""
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

        class MockCompiler(Compiler):
            """Mock compiler class"""

            def run_from_ir(self, *_args, **_kwargs):
                with tempfile.TemporaryDirectory() as workspace:
                    filename = workspace + "a.cpp"
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(contents)

                    object_file = filename.replace(".c", ".o")
                    # libstdc++ has been deprecated on macOS in favour of libc++
                    libcpp = "-lstdc++" if platform.system() == "Linux" else "-lc++"
                    subprocess.run(
                        f"cc -shared {libcpp} -fPIC -x c++ {filename} -o {object_file}".split(),
                        check=True,
                    )
                    output = LinkerDriver.run(object_file, options=self.options)
                    filename = str(pathlib.Path(output).absolute())
                    return filename, "<FAKE_IR>", ["<FAKE_FN>", "<FAKE_TYPE>"]

        @qjit(target="fake_binary")
        @qml.qnode(qml.device(backend, wires=1))
        def cpp_exception_test():
            return None

        cpp_exception_test.compiler = MockCompiler(cpp_exception_test.compiler.options)
        compiled_function = cpp_exception_test.compile()

        with pytest.raises(RuntimeError, match="Hello world"):
            compiled_function()

    def test_linker_driver_invalid_file(self):
        """Test with the invalid input name."""
        with pytest.raises(FileNotFoundError):
            LinkerDriver.get_output_filename("fooo.cpp")


class TestCompilerState:
    """Test states that the compiler can reach."""

    def test_print_stages(self, backend):
        """Test that after compiling the intermediate files exist."""

        @qjit(
            keep_intermediate=True,
            pipelines=[("EmptyPipeline1", [])] + DEFAULT_PIPELINES + [("EmptyPipeline2", [])],
        )
        @qml.qnode(qml.device(backend, wires=1))
        def workflow():
            qml.PauliX(wires=0)
            return qml.state()

        compiler = workflow.compiler
        assert compiler.get_output_of("EmptyPipeline1") is None
        assert compiler.get_output_of("HLOLoweringPass")
        assert compiler.get_output_of("QuantumCompilationPass")
        assert compiler.get_output_of("BufferizationPass")
        assert compiler.get_output_of("MLIRToLLVMDialect")
        assert compiler.get_output_of("EmptyPipeline2") is None
        assert compiler.get_output_of("PreEnzymeOpt")
        assert compiler.get_output_of("Enzyme")
        assert compiler.get_output_of("None-existing-pipeline") is None
        workflow.workspace.cleanup()

    def test_print_nonexistent_stages(self, backend):
        """What happens if we attempt to print something that doesn't exist?"""

        @qjit(keep_intermediate=True)
        @qml.qnode(qml.device(backend, wires=1))
        def workflow():
            qml.PauliX(wires=0)
            return qml.state()

        assert workflow.compiler.get_output_of("None-existing-pipeline") is None
        workflow.workspace.cleanup()

    def test_workspace(self):
        """Test directory has been modified with folder containing intermediate results"""

        @qjit(keep_intermediate=True, target="mlir")
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def workflow():
            qml.PauliX(wires=0)
            return qml.state()

        directory = os.path.join(os.getcwd(), workflow.__name__)
        files = os.listdir(directory)
        # The directory is non-empty. Should at least contain the original .mlir file
        assert files
        workflow.workspace.cleanup()

    def test_compiler_driver_with_output_name(self):
        """Test with non-default output name."""
        with tempfile.TemporaryDirectory() as workspace:
            filename = workspace + "a.c"
            outfilename = workspace + "a.out"
            with open(filename, "w", encoding="utf-8") as f:
                print("int main() {}", file=f)

            LinkerDriver.run(filename, outfile=outfilename)

            assert os.path.exists(outfilename)

    def test_compiler_driver_with_flags(self):
        """Test with non-default flags."""

        with tempfile.TemporaryDirectory() as workspace:
            filename = workspace + "a.c"
            with open(filename, "w", encoding="utf-8") as f:
                print("int main() {}", file=f)

            object_file = filename.replace(".c", ".o")
            libcpp = "-lstdc++" if platform.system() == "Linux" else "-lc++"
            subprocess.run(f"c99 {libcpp} -c {filename} -o {object_file}".split(), check=True)
            expected_outfilename = workspace + "a.so"
            observed_outfilename = LinkerDriver.run(object_file, flags=[])

            assert observed_outfilename == expected_outfilename
            assert os.path.exists(observed_outfilename)

    def test_compiler_from_textual_ir(self):
        """Test the textual IR compilation."""

        ir = r"""
module @workflow {
  func.func public @catalyst.entry_point(%arg0: tensor<f64>) -> tensor<f64> attributes {llvm.emit_c_interface} {
    %0 = call @workflow(%arg0) : (tensor<f64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
  func.func private @workflow(%arg0: tensor<f64>) -> tensor<f64> attributes {diff_method = "finite-diff", llvm.linkage = #llvm.linkage<internal>, qnode} {
    quantum.device ["kwargs", "{'shots': 0}"]
    quantum.device ["backend", "lightning.qubit"]
    %0 = stablehlo.constant dense<4> : tensor<i64>
    %1 = quantum.alloc( 4) : !quantum.reg
    %2 = stablehlo.constant dense<0> : tensor<i64>
    %extracted = tensor.extract %2[] : tensor<i64>
    %3 = quantum.extract %1[%extracted] : !quantum.reg -> !quantum.bit
    %4 = quantum.custom "PauliX"() %3 : !quantum.bit
    %5 = stablehlo.constant dense<1> : tensor<i64>
    %extracted_0 = tensor.extract %5[] : tensor<i64>
    %6 = quantum.extract %1[%extracted_0] : !quantum.reg -> !quantum.bit
    %extracted_1 = tensor.extract %arg0[] : tensor<f64>
    %7 = quantum.custom "RX"(%extracted_1) %6 : !quantum.bit
    %8 = quantum.namedobs %4[ PauliZ] : !quantum.obs
    %9 = quantum.expval %8 : f64
    %from_elements = tensor.from_elements %9 : tensor<f64>
    quantum.dealloc %1 : !quantum.reg
    return %from_elements : tensor<f64>
  }
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
  }
}
"""
        compiled_function = qjit(ir)
        assert compiled_function(0.1) == -1

    def test_parsing_errors(self):
        """Test parsing error handling."""

        ir = r"""
module @workflow {
  func.func public @catalyst.entry_point(%arg0: tensor<f64>) -> tensor<f64> attributes {llvm.emit_c_interface} {
    %c = stablehlo.constant dense<4.0> : tensor<i64>
    return %c : tensor<f64> // Invalid type
  }
}
"""
        with pytest.raises(CompileError) as e:
            qjit(ir)(0.1)

        assert "Failed to parse module as MLIR source" in e.value.args[0]
        assert "Failed to parse module as LLVM source" in e.value.args[0]

    def test_pipeline_error(self):
        """Test pipeline error handling."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit():
            return qml.state()

        test_pipelines = [("PipelineA", ["canonicalize"]), ("PipelineB", ["test"])]
        with pytest.raises(CompileError) as e:
            compiled = qjit(
                circuit, pipelines=test_pipelines, target="mlir", keep_intermediate=True
            )
            compiled.compile()

        assert "Failed to lower MLIR module" in e.value.args[0]
        assert "While processing 'TestPass' pass of the 'PipelineB' pipeline" in e.value.args[0]
        assert "PipelineA" not in e.value.args[0]
        assert "Trace" not in e.value.args[0]
        assert isfile(os.path.join(str(compiled.workspace), "2_PipelineB_FAILED.mlir"))
        compiled.workspace.cleanup()

        with pytest.raises(CompileError) as e:
            qjit(circuit, pipelines=test_pipelines, verbose=True)()

        assert "Trace" in e.value.args[0]


if __name__ == "__main__":
    pytest.main(["-x", __file__])
