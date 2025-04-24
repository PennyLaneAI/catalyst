# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains functions for lowering, compiling, and linking
MLIR/LLVM representations.
"""
import glob
import importlib
import logging
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
import warnings
from os import path
from typing import List, Optional

from catalyst.logging import debug_logger, debug_logger_init
from catalyst.pipelines import CompileOptions
from catalyst.utils.exceptions import CompileError
from catalyst.utils.filesystem import Directory
from catalyst.utils.runtime_environment import get_cli_path, get_lib_path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

package_root = os.path.dirname(__file__)

DEFAULT_CUSTOM_CALLS_LIB_PATH = path.join(package_root, "utils")


@debug_logger
def run_writing_command(command: List[str], compile_options: Optional[CompileOptions]) -> None:
    """Run the command after optionally announcing this fact to the user.

    Args:
        command (List[str]): command to be sent to a subprocess.
        compile_options (Optional[CompileOptions]): compile options.
    """

    if compile_options.verbose:
        print(f"[SYSTEM] {' '.join(command)}", file=compile_options.logfile)
    subprocess.run(command, check=True)


class LinkerDriver:
    """Compiler used to drive the linking stage.
    In order to avoid relying on a single linker at run time and allow the user some flexibility,
    this class defines a compiler resolution order where multiple known compilers are attempted.
    The order is defined as follows:
    1. A user specified compiler via the environment variable CATALYST_CC. It is expected that the
        user provided compiler is flag compatible with GCC/Clang.
    2. clang: Priority is given to clang to maintain an LLVM toolchain through most of the process.
    3. gcc: Usually configured to link with LD.
    4. c99: Usually defaults to gcc, but no linker interface is specified.
    5. c89: Usually defaults to gcc, but no linker interface is specified.
    6. cc: Usually defaults to gcc, however POSIX states that it is deprecated.
    """

    _default_fallback_compilers = ["clang", "gcc", "c99", "c89", "cc"]

    @staticmethod
    @debug_logger
    def get_default_flags(options):
        """Re-compute the path where the libraries exist.

        The use case for this is if someone is in a python jupyter notebook and
        needs to change the environment mid computation.
        Returns
            (List[str]): The default flag list.
        """
        mlir_lib_path = get_lib_path("llvm", "MLIR_LIB_DIR")
        rt_lib_path = get_lib_path("runtime", "RUNTIME_LIB_DIR")

        # Adds RUNTIME_LIB_DIR to the Python system path to allow the catalyst_callback_registry
        # to be importable.
        sys.path.append(get_lib_path("runtime", "RUNTIME_LIB_DIR"))
        import catalyst_callback_registry as registry  # pylint: disable=import-outside-toplevel

        # We use MLIR's C runner utils library in the registry.
        # In order to be able to dlopen that library we need to know the path
        # So we set the path here.
        registry.set_mlir_lib_path(mlir_lib_path)

        lib_path_flags = [
            f"-Wl,-rpath,{mlir_lib_path}",
            f"-L{mlir_lib_path}",
        ]

        if rt_lib_path != mlir_lib_path:
            lib_path_flags += [
                f"-Wl,-rpath,{rt_lib_path}",
                f"-L{rt_lib_path}",
            ]
        else:
            pass  # pragma: nocover

        # Discover the custom call library provided by the frontend & add it to the rpath and -L.
        lib_path_flags += [
            f"-Wl,-rpath,{DEFAULT_CUSTOM_CALLS_LIB_PATH}",
            f"-L{DEFAULT_CUSTOM_CALLS_LIB_PATH}",
        ]

        # Discover the LAPACK library provided by scipy & add link against it.
        # Doing this here ensures we will always have the correct library name.
        lib_name = "openblas"
        package_name = "scipy_openblas32"
        path_within_package = "lib"
        file_extension = ".so" if platform.system() == "Linux" else ".dylib"  # pragma: no branch

        if platform.system() == "Darwin" and platform.machine() == "arm64":  # pragma: nocover
            # use our own build of LAPACKe to interface with Accelerate
            lapack_lib_name = "lapacke.3"
        else:
            package_spec = importlib.util.find_spec(package_name)
            package_directory = path.dirname(package_spec.origin)
            lapack_lib_path = path.join(package_directory, path_within_package)

            search_pattern = path.join(lapack_lib_path, f"lib*{lib_name}*{file_extension}")
            search_result = glob.glob(search_pattern)
            if not search_result:  # pragma: nocover
                raise CompileError(
                    f'Unable to find OpenBLAS library at "{search_pattern}". '
                    "Please ensure that scipy is installed and available via pip."
                )

            lib_path_flags += [f"-Wl,-rpath,{lapack_lib_path}", f"-L{lapack_lib_path}"]
            lapack_lib_name = path.basename(search_result[0])[3 : -len(file_extension)]

        system_flags = []
        if platform.system() == "Linux":
            # --disable-new-dtags makes the linker use RPATH instead of RUNPATH.
            # RPATH influences search paths globally while RUNPATH only works for
            # a single file, but not its dependencies.
            system_flags += ["-Wl,-no-as-needed", "-Wl,--disable-new-dtags"]
        else:  # pragma: nocover
            assert platform.system() == "Darwin", f"Unsupported OS {platform.system()}"
            system_flags += ["-Wl,-arch_errors_fatal"]

        # The exception handling mechanism requires linking against
        # __gxx_personality_v0 which is either on -lstdc++ in
        # or -lc++. We choose based on the operating system.
        if options.async_qnodes and platform.system() == "Linux":  # pragma: nocover
            system_flags += ["-lstdc++"]
        elif options.async_qnodes and platform.system() == "Darwin":  # pragma: nocover
            system_flags += ["-lc++"]

        default_flags = [
            "-shared",
            "-rdynamic",
            *system_flags,
            *lib_path_flags,
            "-lrt_capi",
            "-lpthread",
            "-lmlir_c_runner_utils",  # required for memref.copy
            f"-l{lapack_lib_name}",  # required for custom_calls lib
            "-lcustom_calls",
            "-lmlir_async_runtime",
        ]

        # If OQD runtime capi is built, link to it as well
        # TODO: This is not ideal and should be replaced when the compiler is device aware
        if os.path.isfile(os.path.join(rt_lib_path, "librt_OQD_capi" + file_extension)):
            default_flags.append("-lrt_OQD_capi")

        return default_flags

    @staticmethod
    def _get_compiler_fallback_order(fallback_compilers):
        """Compiler fallback order"""
        preferred_compiler = os.environ.get("CATALYST_CC", None)
        preferred_compiler_exists = LinkerDriver._exists(preferred_compiler)
        compilers = fallback_compilers
        emit_warning = preferred_compiler and not preferred_compiler_exists
        if emit_warning:
            msg = f"User defined compiler {preferred_compiler} is not in PATH. Using fallback ..."
            warnings.warn(msg, UserWarning)
        else:
            compilers = [preferred_compiler] + fallback_compilers
        return compilers

    @staticmethod
    def _exists(compiler):
        if compiler is None:
            return None
        return shutil.which(compiler)

    @staticmethod
    def _available_compilers(fallback_compilers):
        for compiler in LinkerDriver._get_compiler_fallback_order(fallback_compilers):
            if LinkerDriver._exists(compiler):
                yield compiler

    @staticmethod
    def _attempt_link(compiler, flags, infile, outfile, options):
        try:
            command = [compiler] + flags + [infile, "-o", outfile]
            run_writing_command(command, options)
            return True
        except subprocess.CalledProcessError as e:
            # Only warn in verbose mode, as users might see it otherwise in regular use.
            if options.verbose:
                msg = f"Compiler {compiler} failed to link executable and returned with exit code "
                msg += f"{e.returncode}. Output was: {e.output}.\nCommand: {command}"
                warnings.warn(msg, UserWarning)
            return False

    @staticmethod
    @debug_logger
    def get_output_filename(infile):
        """Rename object file to shared object

        Args:
            infile (str): input file name
            outfile (str): output file name
        """
        infile_path = pathlib.Path(infile)
        if not infile_path.exists():
            raise FileNotFoundError(f"Cannot find {infile}.")
        return str(infile_path.with_suffix(".so"))

    @staticmethod
    @debug_logger
    def run(infile, outfile=None, flags=None, fallback_compilers=None, options=None):
        """
        Link the infile against the necessary libraries and produce the outfile.

        Args:
            infile (str): input file
            outfile (Optional[str]): output file
            flags (Optional[List[str]]): flags to be passed down to the compiler
            fallback_compilers (Optional[List[str]]): name of executables to be looked for in PATH
            compile_options (Optional[CompileOptions]): generic compilation options.
        Raises:
            EnvironmentError: The exception is raised when no compiler succeeded.
        """
        if outfile is None:
            outfile = LinkerDriver.get_output_filename(infile)
        if options is None:
            options = CompileOptions()
        if flags is None:
            flags = LinkerDriver.get_default_flags(options)
        if fallback_compilers is None:
            fallback_compilers = LinkerDriver._default_fallback_compilers
        for compiler in LinkerDriver._available_compilers(fallback_compilers):
            success = LinkerDriver._attempt_link(compiler, flags, infile, outfile, options)
            if success:
                return outfile
        msg = f"Unable to link {infile}. Please check the output for any error messages. If no "
        msg += "compiler was found by Catalyst, please specify a compatible one via $CATALYST_CC."
        raise CompileError(msg)


def _get_catalyst_cli_cmd(*args, stdin=None):
    """Just get the command, do not run it"""
    cli_path = get_cli_path()
    if not path.isfile(cli_path):
        raise FileNotFoundError("catalyst executable was not found.")  # pragma: nocover

    cmd = [cli_path]
    for arg in args:
        if not isinstance(arg, str):
            cmd += [str(x) for x in arg]
        else:
            cmd += [str(arg)]

    if stdin:
        cmd += ["-"]

    return cmd


def _catalyst(*args, stdin=None):
    """Raw interface to catalyst

    echo ${stdin} | catalyst *args -
    catalyst *args
    """
    cmd = _get_catalyst_cli_cmd(*args, stdin=stdin)
    try:
        result = subprocess.run(cmd, input=stdin, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise CompileError(f"catalyst failed with error code {e.returncode}: {e.stderr}") from e


def _quantum_opt(*args, stdin=None):
    """Raw interface to quantum-opt

    echo ${stdin} | catalyst --tool=opt *args -
    catalyst --tool=opt *args
    """
    return _catalyst(("--tool", "opt"), *args, stdin=stdin)


def canonicalize(*args, stdin=None):
    """Run opt with canonicalization

    echo ${stdin} | catalyst --tool=opt \
            --catalyst-pipeline='builtin.module(canonicalize)' *args -
    catalyst --tool=opt \
            --catalyst-pipeline='builtin.module(canonicalize)' *args

    Returns stdout string
    """
    return _quantum_opt(("--pass-pipeline", "builtin.module(canonicalize)"), *args, stdin=stdin)


def _options_to_cli_flags(options):
    """CompileOptions -> list[str|Tuple[str, str]]"""

    extra_args = []
    pipelines = options.get_pipelines()
    pipeline_str = ""

    for pipeline in pipelines:
        pipeline_name, passes = pipeline
        passes_str = ";".join(passes)
        pipeline_str += f"{pipeline_name}({passes_str}),"
    extra_args += ["--catalyst-pipeline", pipeline_str]

    for plugin in options.pass_plugins:
        extra_args += [("--load-pass-plugin", plugin)]
    for plugin in options.dialect_plugins:
        extra_args += [("--load-dialect-plugin", plugin)]
    if options.checkpoint_stage:
        extra_args += [("--checkpoint-stage", options.checkpoint_stage)]

    if not options.lower_to_llvm:
        extra_args += [("--tool", "opt")]

    if options.keep_intermediate:
        extra_args += ["--keep-intermediate"]

    if options.verbose:
        extra_args += ["--verbose"]

    if options.async_qnodes:  # pragma: nocover
        extra_args += ["--async-qnodes"]

    return extra_args


def to_llvmir(*args, stdin=None, options: Optional[CompileOptions] = None):
    """echo ${input} | catalyst *args -"""
    # These are the options that may affect compilation
    if not options:
        return _catalyst(*args, stdin=stdin)

    opts = _options_to_cli_flags(options)
    return _catalyst(*opts, *args, stdin=stdin)


def to_mlir_opt(*args, stdin=None, options: Optional[CompileOptions] = None):
    """echo ${input} | catalyst --tool=opt *args *opts -"""
    # These are the options that may affect compilation
    if not options:
        return _quantum_opt(*args, stdin=stdin)

    opts = _options_to_cli_flags(options)
    return _quantum_opt(*opts, *args, stdin=stdin)


class Compiler:
    """Compiles MLIR modules to shared objects by executing the Catalyst compiler driver library."""

    @debug_logger_init
    def __init__(self, options: Optional[CompileOptions] = None):
        self.options = options if options is not None else CompileOptions()

    @debug_logger
    def get_cli_command(self, tmp_infile_name, output_ir_name, module_name, workspace):
        """Prepare the command to run the Catalyst CLI to compile the file.

        Args:
            module_name (str): Module name to use for naming
            workspace (Directory): directory that holds output files and/or debug dumps.
        Returns:
            cmd (str): The command to be executed.
        """
        opts = _options_to_cli_flags(self.options)
        cmd = _get_catalyst_cli_cmd(
            ("-o", output_ir_name),
            ("--module-name", module_name),
            ("--workspace", str(workspace)),
            "-verify-each=false",
            *opts,
            tmp_infile_name,
        )
        return cmd

    @debug_logger
    def run_from_ir(self, ir: str, module_name: str, workspace: Directory):
        """Compile a shared object from a textual IR (MLIR or LLVM).

        Args:
            ir (str): Textual MLIR to be compiled
            module_name (str): Module name to use for naming
            workspace (Directory): directory that holds output files and/or debug dumps.

        Returns:
            output_filename (str): Output file name. For the default pipeline this would be the
                                   shared object library path.
            out_IR (str): Output IR in textual form. For the default pipeline this would be the
                          LLVM IR.
        """
        assert isinstance(
            workspace, Directory
        ), f"Compiler expects a Directory type, got {type(workspace)}."
        assert workspace.is_dir(), f"Compiler expects an existing directory, got {workspace}."
        assert (
            self.options.lower_to_llvm
        ), "lower_to_llvm must be set to True in order to compile to a shared object"

        if self.options.verbose:
            print(f"[LIB] Running compiler driver in {workspace}", file=self.options.logfile)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", dir=str(workspace), delete=False
        ) as tmp_infile:
            tmp_infile_name = tmp_infile.name
            tmp_infile.write(ir)

        output_object_name = os.path.join(str(workspace), f"{module_name}.o")
        output_ir_name = os.path.join(str(workspace), f"{module_name}.ll")

        cmd = self.get_cli_command(tmp_infile_name, output_ir_name, module_name, workspace)
        try:
            if self.options.verbose:
                print(f"[SYSTEM] {' '.join(cmd)}", file=self.options.logfile)
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if self.options.verbose or os.getenv("ENABLE_DIAGNOSTICS"):
                if result.stdout:
                    print(result.stdout.strip(), file=self.options.logfile)
                if result.stderr:
                    print(result.stderr.strip(), file=self.options.logfile)
        except subprocess.CalledProcessError as e:  # pragma: nocover
            raise CompileError(f"catalyst failed with error code {e.returncode}: {e.stderr}") from e

        if os.path.exists(output_ir_name):
            with open(output_ir_name, "r", encoding="utf-8") as f:
                out_IR = f.read()
        else:
            out_IR = None

        output = LinkerDriver.run(output_object_name, options=self.options)
        output_object_name = str(pathlib.Path(output).absolute())

        # Clean up temporary files
        if os.path.exists(tmp_infile_name):
            os.remove(tmp_infile_name)
        if os.path.exists(output_ir_name):
            os.remove(output_ir_name)

        return output_object_name, out_IR

    @debug_logger
    def run(self, mlir_module, *args, **kwargs):
        """Compile an MLIR module to a shared object.

        .. note::

            For compilation of hybrid quantum-classical PennyLane programs,
            please see the :func:`~.qjit` decorator.

        Args:
            mlir_module: The MLIR module to be compiled

        Returns:
            (str): filename of shared object
        """

        return self.run_from_ir(
            mlir_module.operation.get_asm(
                binary=False, print_generic_op_form=False, assume_verified=True
            ),
            str(mlir_module.operation.attributes["sym_name"]).replace('"', ""),
            *args,
            **kwargs,
        )

    @debug_logger
    def get_output_of(self, pipeline, workspace) -> Optional[str]:
        """Get the output IR of a pipeline.
        Args:
            pipeline (str): name of pass class

        Returns
            (Optional[str]): output IR
        """
        file_content = None
        for dirpath, _, filenames in os.walk(str(workspace)):
            filenames = [f for f in filenames if f.endswith(".mlir") or f.endswith(".ll")]
            if not filenames:
                break
            filenames_no_ext = [os.path.splitext(f)[0] for f in filenames]
            if pipeline == "mlir":
                # Sort files and pick the first one
                selected_file = [
                    sorted(filenames)[0],
                ]
            elif pipeline == "last":
                # Sort files and pick the last one
                selected_file = [
                    sorted(filenames)[-1],
                ]
            else:
                selected_file = [
                    f
                    for f, name_no_ext in zip(filenames, filenames_no_ext)
                    if pipeline in name_no_ext
                ]
            if len(selected_file) != 1:
                msg = f"Attempting to get output for pipeline: {pipeline},"
                msg += " but no or more than one file was found.\n"
                raise CompileError(msg)
            filename = selected_file[0]

            full_path = os.path.join(dirpath, filename)
            with open(full_path, "r", encoding="utf-8") as file:
                file_content = file.read()

        if file_content is None:
            msg = f"Attempting to get output for pipeline: {pipeline},"
            msg += " but no file was found.\n"
            msg += "Are you sure the file exists?"
            raise CompileError(msg)
        return file_content
