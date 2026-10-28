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

import io
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
import warnings
from os.path import isfile

import numpy as np
import pennylane as qp
import pytest

from catalyst import QJIT, qjit
from catalyst.compiler import (
    CompileOptions,
    Compiler,
    LinkerDriver,
    _catalyst,
    _options_to_cli_flags,
)
from catalyst.debug import instrumentation
from catalyst.pipelines import KeepIntermediateLevel
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
        @qp.qnode(qp.device(backend, wires=1))
        def workflow(x):
            qp.RX(x, wires=0)
            return qp.state()

        # Create tmp workspaces for intermediates to avoid CI race conditions
        workflow.use_cwd_for_workspace = False
        workflow.jit_compile((1.2,))

        capture_result = capsys.readouterr()
        capture = capture_result.out + capture_result.err
        assert ("[SYSTEM]" in capture) if verbose else ("[SYSTEM]" not in capture)
        assert ("[LIB]" in capture) if verbose else ("[LIB]" not in capture)
        assert ("Dumping" in capture) if (verbose and keep_intermediate) else True
        workflow.workspace.cleanup()

    def test_compilation_with_instrumentation(self, capsys, backend):
        """Test compilation with instrumentation"""

        @qp.qnode(qp.device(backend, wires=1))
        def circuit():
            return qp.state()

        with instrumentation(circuit.__name__, filename=None, detailed=True):
            qjit(circuit)()

        capture_result = capsys.readouterr()
        capture = capture_result.out + capture_result.err
        assert "[DIAGNOSTICS]" in capture

    @pytest.mark.parametrize(
        "input_value, expected_level",
        [
            (None, KeepIntermediateLevel.NONE),
            (False, KeepIntermediateLevel.NONE),
            (True, KeepIntermediateLevel.PIPELINE),
            (0, KeepIntermediateLevel.NONE),
            (1, KeepIntermediateLevel.PIPELINE),
            (2, KeepIntermediateLevel.CHANGED),
            (3, KeepIntermediateLevel.PASS),
            ("none", KeepIntermediateLevel.NONE),
            ("pipeline", KeepIntermediateLevel.PIPELINE),
            ("changed", KeepIntermediateLevel.CHANGED),
            ("pass", KeepIntermediateLevel.PASS),
        ],
    )
    def test_keep_intermediate_levels_conversion(self, input_value, expected_level):
        """Test that various inputs for keep_intermediate are correctly converted to Enum."""
        options = CompileOptions(keep_intermediate=input_value)
        assert options.keep_intermediate == expected_level

    @pytest.mark.parametrize("invalid_input", [4, -1, "invalid_string", 4.0, []])
    def test_keep_intermediate_invalid_inputs(self, invalid_input):
        """Test that invalid inputs for keep_intermediate raise appropriate errors."""
        with pytest.raises(ValueError, match="Invalid value for keep_intermediate:"):
            CompileOptions(keep_intermediate=invalid_input)

    def test_options_to_cli_flags_keep_intermediate_none(self):
        """Test _options_to_cli_flags with KeepIntermediateLevel.NONE."""
        options = CompileOptions(keep_intermediate=KeepIntermediateLevel.NONE)
        flags = _options_to_cli_flags(options)
        assert "--keep-intermediate" not in flags
        assert "--save-ir-after-each=pass" not in flags

    def test_options_to_cli_flags_keep_intermediate_basic(self):
        """Test _options_to_cli_flags with KeepIntermediateLevel.PIPELINE."""
        options = CompileOptions(keep_intermediate=KeepIntermediateLevel.PIPELINE)
        flags = _options_to_cli_flags(options)
        assert "--keep-intermediate" in flags
        assert "--save-ir-after-each=pass" not in flags

    def test_options_to_cli_flags_keep_intermediate_changed(self):
        """Test _options_to_cli_flags with KeepIntermediateLevel.CHANGED."""
        options = CompileOptions(keep_intermediate=KeepIntermediateLevel.CHANGED)
        flags = _options_to_cli_flags(options)
        assert "--keep-intermediate" in flags
        assert "--save-ir-after-each=changed" in flags
        assert "--dump-module-scope" in flags

    def test_options_to_cli_flags_keep_intermediate_pass(self):
        """Test _options_to_cli_flags with KeepIntermediateLevel.PASS."""
        options = CompileOptions(keep_intermediate=KeepIntermediateLevel.PASS)
        flags = _options_to_cli_flags(options)
        assert "--keep-intermediate" in flags
        assert "--save-ir-after-each=pass" in flags


class TestCompilerOutput:
    """Tools write their final artifacts to the requested destination."""

    @pytest.mark.parametrize("destination", [None, "-", "output"])
    @pytest.mark.parametrize(
        "tool,source,expected",
        [
            ("opt", "func.func @foo() { return }", b"llvm.func @foo"),
            ("translate", "module { llvm.func @foo() { llvm.return } }", b"define void @foo"),
            ("llvm-opt", "define void @foo() { ret void }", b"define void @foo"),
            ("opt-translate", "func.func @foo() { return }", b"define void @foo"),
            ("opt-translate-llvm", "func.func @foo() { return }", b"define void @foo"),
        ],
    )
    def test_final_output(self, tool, source, expected, destination, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        flags = [("--tool", tool), ("--workspace", str(workspace)), ("--module-name", "foo")]
        if destination is not None:
            flags.append(("-o", str(tmp_path / destination) if destination != "-" else "-"))

        stdout = _catalyst(*flags, stdin=source.encode(), text=False)
        if destination == "output":
            assert stdout == b""
            result = (tmp_path / destination).read_bytes()
        else:
            result = stdout

        assert expected in result
        assert not list(workspace.glob("*.o"))
        assert not list(workspace.iterdir())

    def test_intermediate_files(self, tmp_path):
        """Intermediate saving writes snapshots into the requested workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        output = tmp_path / "output.ll"
        _catalyst(
            ("--tool", "opt-translate"),
            ("--workspace", workspace),
            "--keep-intermediate",
            ("-o", output),
            stdin="func.func @foo() { return }",
        )

        assert "define void @foo" in output.read_text()
        assert list(workspace.rglob("*.mlir"))
        assert list(workspace.rglob("*.ll"))

    def test_bytecode_file(self, tmp_path):
        """The compiler can read back MLIR bytecode written to a file."""
        output = tmp_path / "output.mlirbc"
        assert (
            _catalyst(("--tool", "opt"), "--emit-bytecode", ("-o", output), stdin="module {}") == ""
        )
        assert "module" in _catalyst(("--tool", "opt"), str(output))

    @pytest.mark.parametrize("destination", [None, "-", "output.o"])
    @pytest.mark.parametrize(
        "tool,source",
        [
            (None, "func.func @foo() { return }"),
            ("all", "func.func @foo() { return }"),
            ("llc", "define void @foo() { ret void }"),
        ],
    )
    def test_object_output(self, tool, source, destination, tmp_path, monkeypatch):
        """Object-generating tools use file destinations and reject stdout."""
        monkeypatch.chdir(tmp_path)
        flags = [("--tool", tool)] if tool is not None else []
        if destination is not None:
            flags.append(("-o", destination))
        if destination == "-":
            with pytest.raises(CompileError, match="Object code cannot be written to stdout"):
                _catalyst(*flags, stdin=source)
            assert not list(tmp_path.iterdir())
            return

        assert _catalyst(*flags, stdin=source) == ""
        output = tmp_path / (destination or "catalyst_module.o")
        # ELF and Mach-O object headers on the supported Linux and macOS hosts.
        assert output.read_bytes().startswith(
            (b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe")
        )

    def test_default_object_filename(self, tmp_path, monkeypatch):
        """File inputs use the default object name in the current directory."""
        monkeypatch.chdir(tmp_path)
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        source = input_dir / "program.mlir"
        source.write_text("func.func @foo() { return }")

        assert _catalyst(str(source)) == ""
        assert (tmp_path / "catalyst_module.o").stat().st_size > 0
        assert not (input_dir / "catalyst_module.o").exists()

    @pytest.mark.parametrize("destination", [None, "output.o"])
    def test_module_name_object_output(self, destination, tmp_path, monkeypatch):
        """The module name supplies the object filename unless an output path is given."""
        monkeypatch.chdir(tmp_path)
        flags = [("--module-name", "custom")]
        if destination is not None:
            flags.append(("-o", destination))

        assert _catalyst(*flags, stdin="func.func @foo() { return }") == ""
        output = tmp_path / (destination or "custom.o")
        assert output.stat().st_size > 0
        assert list(tmp_path.iterdir()) == [output]

    def test_verbose_object_output(self, tmp_path, capsys):
        """Verbose object compilation reports progress on stderr."""
        output = tmp_path / "output.o"
        assert (
            _catalyst(
                "--verbose", ("-o", output), stdin="func.func @foo() { return }", stderr_return=True
            )
            == ""
        )
        assert output.stat().st_size > 0
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "LLC object code generation successful" in captured.err

    def test_frontend_object_output(self, tmp_path):
        """Unlinked frontend compilation produces an object in its workspace."""
        compiler = Compiler(CompileOptions(link=False))
        compiler.run_from_ir("func.func @foo() { return }", "foo", Directory(tmp_path))

        assert (tmp_path / "foo.o").stat().st_size > 0


class TestCompilerDiagnostics:
    """Unit tests for diagnostics emitted by the Catalyst subprocess."""

    def test_verbose_logging_preserves_ir(self, capsys):
        """Verbose progress goes to stderr without contaminating the returned LLVM IR."""
        tool = "opt-translate-llvm"
        source = "func.func @foo() { return }"
        quiet_ir = _catalyst(("--tool", tool), stdin=source)
        verbose_ir = _catalyst(("--tool", tool), "--verbose", stdin=source, stderr_return=True)

        assert "define void @foo" in quiet_ir
        assert verbose_ir == quiet_ir
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "MLIR parsing successful" in captured.err
        assert "Compilation successful:" in captured.err

    @pytest.mark.parametrize(
        "text,stdout,stderr",
        [
            (True, "output", "compiler warning\n"),
            (False, b"output", b"compiler warning\n"),
        ],
    )
    def test_forward_stderr(self, monkeypatch, text, stdout, stderr):
        stderr_stream = io.StringIO()
        result = subprocess.CompletedProcess([], 0, stdout=stdout, stderr=stderr)
        monkeypatch.setattr(sys, "stderr", stderr_stream)
        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: result)

        assert _catalyst(text=text, stderr_return=True) == stdout
        assert stderr_stream.getvalue() == "compiler warning\n"

    def test_stderr_not_forwarded_by_default(self, monkeypatch):
        stderr_stream = io.StringIO()
        result = subprocess.CompletedProcess([], 0, stdout="output", stderr="compiler warning\n")
        monkeypatch.setattr(sys, "stderr", stderr_stream)
        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: result)

        assert _catalyst() == "output"
        assert stderr_stream.getvalue() == ""

    def test_color_diagnostics_in_terminal(self, monkeypatch):
        class Terminal(io.StringIO):
            def isatty(self):
                return True

        command = []

        def run(cmd, **_kwargs):
            command.extend(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(sys, "stderr", Terminal())
        monkeypatch.setattr(subprocess, "run", run)

        _catalyst()

        assert command[1] == "--color"


class TestDefaultFlags:
    """Unit tests for optional libraries discovered in LinkerDriver.get_default_flags."""

    @staticmethod
    def _lib_name(stem):
        """Build the platform-specific shared library file name for a library stem."""
        ext = ".so" if platform.system() == "Linux" else ".dylib"
        return stem + ext

    def _patch_isfile(self, monkeypatch, overrides):
        """Patch isfile to force given library file names present/absent."""
        real_isfile = os.path.isfile
        overrides = {os.path.basename(name): value for name, value in overrides.items()}

        def fake_isfile(p):
            name = os.path.basename(p)
            if name in overrides:
                return overrides[name]
            return real_isfile(p)

        monkeypatch.setattr("catalyst.compiler.os.path.isfile", fake_isfile)

    def test_rt_executor_linked_when_present(self, monkeypatch):
        """-lrt_executor is added when librt_executor exists in the runtime lib dir."""
        self._patch_isfile(monkeypatch, overrides={self._lib_name("librt_executor"): True})
        flags = LinkerDriver.get_default_flags(CompileOptions())
        assert "-lrt_executor" in flags

    def test_rt_executor_not_linked_when_absent(self, monkeypatch):
        """-lrt_executor is omitted when librt_executor is missing."""
        self._patch_isfile(monkeypatch, overrides={self._lib_name("librt_executor"): False})
        flags = LinkerDriver.get_default_flags(CompileOptions())
        assert "-lrt_executor" not in flags


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
        """Test the return value if a user requests an intermediate file that doesn't exist."""
        compiler = Compiler()
        with pytest.raises(CompileError, match="Attempting to get output for pipeline"):
            compiler.get_output_of("inexistent-file", ".")

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
                    return (
                        filename,
                        "<FAKE_IR>",
                    )

        @qjit(target="mlir")
        @qp.qnode(qp.device(backend, wires=1))
        def cpp_exception_test():
            return None

        cpp_exception_test.compiler = MockCompiler(cpp_exception_test.compiler.options)
        compiled_function, _ = cpp_exception_test.compile()

        with pytest.raises(RuntimeError, match="Hello world"):
            compiled_function()

    def test_linker_driver_invalid_file(self):
        """Test with the invalid input name."""
        with pytest.raises(FileNotFoundError):
            LinkerDriver.get_output_filename("fooo.cpp")


class TestCompilerState:
    """Test states that the compiler can reach."""

    def test_invalid_target(self):
        """Test that nothing happens in AOT compilation when an invalid target is provided."""

        @qjit(target="hello")
        def f():
            return 0

        assert f.jaxpr is None
        assert f.mlir is None
        assert f.llvmir is None
        assert f.compiled_function is None

    def test_mlir_only_qjit_compile(self):
        """Test that the QJIT compilation stage can be run without LLVM lowering or linking."""

        test_pipes = [("test_pipe", ["canonicalize"])]

        def f(x):
            return x + 1.0

        log = io.StringIO()  # for inspection
        options = CompileOptions(
            lower_to_llvm=False, link=False, pipelines=test_pipes, verbose=True, logfile=log
        )

        compiled = QJIT(f, options)
        compiled.workspace = compiled._get_workspace()  # pylint: disable=protected-access

        # Call individual QJIT stages
        compiled.jaxpr, *_ = compiled.capture((0.5,))
        compiled.mlir_module = compiled.generate_ir()
        compiled.compile()

        logtext = log.getvalue()
        assert "MLIR parsing successful" in logtext
        assert "opt transformations successful" in logtext
        assert "LLVMIR translation successful" not in logtext
        assert "object code generation successful" not in logtext
        assert "Compilation successful" in logtext
        assert "object linking successful" not in logtext

        # since these are not advertised options, an assertion should be okay
        with pytest.raises(AssertionError, match="invalid options for jit_compile"):
            compiled(0.5)

    @pytest.mark.parametrize(
        "target, num_stages_run", [("mlir_opt", 1), ("llvmir", 3), ("binary", 4)]
    )
    def test_aot_compile_up_to_target(self, target, num_stages_run):
        """Test that ahead-of-time compilation stops at the target, and that a later call still
        compiles the function to a shared library and runs it."""

        stages = [
            "opt transformations successful",
            "LLVMIR translation successful",
            "object code generation successful",
            "object linking successful",
        ]
        log = io.StringIO()

        @qjit(target=target, verbose=True, logfile=log)
        def f(x):
            return x + 1.0

        f.aot_compile(0.5)

        logtext = log.getvalue()
        for stage in stages[:num_stages_run]:
            assert stage in logtext
        for stage in stages[num_stages_run:]:
            assert stage not in logtext
        assert f(0.5) == 1.5

    def test_mlir_opt_target_without_llvmir(self):
        """Test that the MLIR returned by the compiler for the ``mlir_opt`` target, when
        intermediate files are kept, is not stored as LLVM IR."""

        @qjit(target="mlir_opt", keep_intermediate=True)
        def f(x):
            return x + 1.0

        # Create tmp workspaces for intermediates to avoid CI race conditions
        f.use_cwd_for_workspace = False
        f.aot_compile(0.5)

        assert f.llvm_ir is None
        f.workspace.cleanup()

    @pytest.mark.parametrize("target", ["mlir_opt", "llvmir"])
    def test_compile_options_restored_on_error(self, target):
        """Test that the compile options set for the target are restored when the ahead-of-time
        compilation fails."""

        @qjit(target=target, pipelines=[("PipelineA", ["test"])])
        def f(x):
            return x + 1.0

        with pytest.raises(CompileError, match="Failed to lower MLIR module"):
            f.aot_compile(0.5)

        assert f.compile_options.lower_to_llvm
        assert f.compile_options.link

    def test_callable_without_name(self):
        """Test that a callable without __name__ property can be compiled, if it is otherwise
        traceable."""

        class NoNameClass:
            def __call__(self, x):
                return x

        f = qjit(NoNameClass())

        assert f(3) == 3
        assert f.__name__ == "unknown"

    def test_print_stages(self, backend):
        """Test that after compiling the intermediate files exist."""

        options = CompileOptions()
        pipelines = options.get_stages()

        @qjit(
            keep_intermediate=True,
            pipelines=[("EmptyPipeline1", [])] + pipelines + [("EmptyPipeline2", [])],
        )
        @qp.qnode(qp.device(backend, wires=1))
        def workflow(x):
            qp.RX(x, wires=0)
            return qp.state()

        # Create tmp workspaces for intermediates to avoid CI race conditions
        workflow.use_cwd_for_workspace = False
        workflow.jit_compile((1.2,))

        compiler = workflow.compiler
        with pytest.raises(CompileError, match="Attempting to get output for pipeline"):
            compiler.get_output_of("EmptyPipeline1", workflow.workspace)
        assert compiler.get_output_of("HLOLoweringStage", workflow.workspace)
        assert compiler.get_output_of("QuantumCompilationStage", workflow.workspace)
        with pytest.raises(CompileError, match="Attempting to get output for pipeline"):
            compiler.get_output_of("EmptyPipeline2", workflow.workspace)
        assert compiler.get_output_of("BufferizationStage", workflow.workspace)
        assert compiler.get_output_of("MLIRToLLVMDialectConversion", workflow.workspace)
        with pytest.raises(CompileError, match="Attempting to get output for pipeline"):
            compiler.get_output_of("None-existing-pipeline", workflow.workspace)
        workflow.workspace.cleanup()

    def test_print_nonexistent_stages(self, backend):
        """What happens if we attempt to print something that doesn't exist?"""

        @qjit(keep_intermediate=True)
        @qp.qnode(qp.device(backend, wires=1))
        def workflow(x):
            qp.RX(x, wires=0)
            return qp.state()

        # Create tmp workspaces for intermediates to avoid CI race conditions
        workflow.use_cwd_for_workspace = False
        workflow.jit_compile((1.2,))

        with pytest.raises(CompileError, match="Attempting to get output for pipeline"):
            workflow.compiler.get_output_of("None-existing-pipeline", workflow.workspace)
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

    def test_pipeline_error(self):
        """Test pipeline error handling."""

        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit():
            return qp.state()

        test_pipelines = [("PipelineA", ["canonicalize"]), ("PipelineB", ["test"])]
        with pytest.raises(CompileError) as e:
            compiled = qjit(
                circuit, pipelines=test_pipelines, target="mlir", keep_intermediate=True
            )
            compiled.compile()

        stack_trace_pattern = "diagnostic emitted with trace"

        assert "Failed to lower MLIR module" in e.value.args[0]
        assert "While processing 'TestPass' pass " in e.value.args[0]
        assert stack_trace_pattern not in e.value.args[0]
        assert isfile(os.path.join(str(compiled.workspace), "2_AfterTestPass_FAILED.mlir"))
        compiled.workspace.cleanup()

        with pytest.raises(CompileError) as e:
            qjit(circuit, pipelines=test_pipelines, verbose=True)()

        assert stack_trace_pattern in e.value.args[0]


class TestCustomCall:
    """Test compilation of `lapack_dsyevd` via lowering to `stablehlo.custom_call`."""

    def test_dsyevd(self):
        """Test the support of `lapack_dsyevd` in one single `stablehlo.custom_call` call."""

        A = np.array([[4, 9], [9, 4]])

        def workflow(A):
            B = qp.math.sqrt_matrix(A)
            return B @ A

        qjit_result = qjit(workflow)(A)
        pl_result = workflow(A)
        assert np.allclose(qjit_result, pl_result)

    def test_dsyevd_multiple_calls(self):
        """Test the support of `lapack_dsyevd` in multiple `stablehlo.custom_call` calls."""

        A = np.array([[4, 9], [9, 4]])

        def workflow(A):
            B = qp.math.sqrt_matrix(A) @ qp.math.sqrt_matrix(A)
            return B @ A

        qjit_result = qjit(workflow)(A)
        pl_result = workflow(A)
        assert np.allclose(qjit_result, pl_result)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
