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

import abc
import os
import sys
import shutil
import subprocess
import tempfile
import warnings
from io import TextIOWrapper
from typing import Optional, List
from dataclasses import dataclass

from catalyst._configuration import INSTALLED

package_root = os.path.dirname(__file__)


@dataclass
class CompileOptions:
    """Generic compilation options"""

    verbose: bool = False
    logfile: Optional[TextIOWrapper] = sys.stderr

    def get_logfile(self) -> TextIOWrapper:
        """Get the effective file object, as configured"""
        return self.logfile


default_compile_options = CompileOptions()


def run_writing_command(
    command: List[str], compile_options: Optional[CompileOptions] = None
) -> None:
    """Run the command after optionally announcing this fact to the user"""
    if compile_options is None:
        compile_options = default_compile_options

    if compile_options.verbose:
        print(f"[RUNNING] {' '.join(command)}", file=compile_options.get_logfile())
    subprocess.run(command, check=True)


default_bin_paths = {
    "llvm": os.path.join(package_root, "../../mlir/llvm-project/build/bin"),
    "mhlo": os.path.join(package_root, "../../mlir/mlir-hlo/build/bin"),
    "quantum": os.path.join(package_root, "../../mlir/build/bin"),
}

default_lib_paths = {
    "llvm": os.path.join(package_root, "../../mlir/llvm-project/build/lib"),
    "runtime": os.path.join(package_root, "../../runtime/build/lib"),
}


def get_executable_path(project, tool):
    """Get path to executable."""
    path = os.path.join(package_root, "bin") if INSTALLED else default_bin_paths.get(project, "")
    executable_path = os.path.join(path, tool)
    return executable_path if os.path.exists(executable_path) else tool


def get_lib_path(project, env_var):
    """Get the library path."""
    if INSTALLED:
        return os.path.join(package_root, "lib")  # pragma: no cover
    return os.getenv(env_var, default_lib_paths.get(project, ""))


class PassPipeline(abc.ABC):
    """Abstract PassPipeline class."""

    _executable = None
    _default_flags = None

    @staticmethod
    @abc.abstractmethod
    def get_output_filename(infile):
        """Compute the output filename from the input filename.

        Args:
            infile (str): input file
            outfile (str): output file
        """

    @staticmethod
    def _run(infile, outfile, executable, flags, options):
        command = [executable] + flags + [infile, "-o", outfile]
        run_writing_command(command, options)

    @classmethod
    # pylint: disable=too-many-arguments
    def run(cls, infile, outfile=None, executable=None, flags=None, options=None):
        """Run the pass.

        Args:
            infile (str): path to MLIR file to be compiled
            outfile (str): path to output file, defaults to replacing extension in infile to .nohlo
            executable (str): path to executable, defaults to mlir-hlo-opt
            flags (List[str]): flags to mlir-hlo-opt, defaults to _default_flags
            options (CompileOptions): compile options
        """
        if outfile is None:
            outfile = cls.get_output_filename(infile)
        if executable is None:
            executable = cls._executable
        if executable is None:
            raise ValueError("Executable not specified.")
        if flags is None:
            flags = cls._default_flags
        cls._run(infile, outfile, executable, flags, options)
        return outfile


# pylint: disable=too-few-public-methods
class MHLOPass(PassPipeline):
    """Pass pipeline to convert (M)HLO dialects to standard MLIR dialects."""

    _executable = get_executable_path("mhlo", "mlir-hlo-opt")
    _default_flags = [
        "--allow-unregistered-dialect",
        "--canonicalize",
        "--chlo-legalize-to-hlo",
        "--mhlo-legalize-control-flow",
        "--hlo-legalize-to-linalg",
        "--mhlo-legalize-to-std",
        "--convert-to-signless",
        "--canonicalize",
    ]

    @staticmethod
    def get_output_filename(infile):
        if not infile.endswith(".mlir"):
            raise ValueError(f"Input file ({infile}) for MHLO is not an MLIR file")
        return infile.replace(".mlir", ".nohlo.mlir")


class BufferizationPass(PassPipeline):
    """Pass pipeline that bufferizes MLIR dialects."""

    _executable = get_executable_path("quantum", "quantum-opt")
    _default_flags = [
        "--inline",
        "--gradient-bufferize",
        "--scf-bufferize",
        "--convert-tensor-to-linalg",  # tensor.pad
        "--convert-elementwise-to-linalg",  # Must be run before --arith-bufferize
        "--arith-bufferize",
        "--empty-tensor-to-alloc-tensor",
        "--bufferization-bufferize",
        "--tensor-bufferize",
        "--linalg-bufferize",
        "--tensor-bufferize",
        "--quantum-bufferize",
        "--func-bufferize",
        "--finalizing-bufferize",
        "--buffer-hoisting",
        "--buffer-loop-hoisting",
        "--buffer-deallocation",
        "--convert-bufferization-to-memref",
        "--canonicalize",
        "--cse",
    ]

    @staticmethod
    def get_output_filename(infile):
        if not infile.endswith(".opt.mlir"):
            raise ValueError(f"Input file ({infile}) for bufferization is not an MLIR file")

        return infile.replace(".opt.mlir", ".buff.mlir")


class MLIRToLLVMDialect(PassPipeline):
    """Pass pipeline to lower MLIR dialects to LLVM dialect."""

    _executable = get_executable_path("quantum", "quantum-opt")
    _default_flags = [
        "--convert-linalg-to-loops",
        "--convert-scf-to-cf",
        # This pass expands memref operations that modify the metadata of a memref (sizes, offsets,
        # stdies) into a sequence of easier to analyze constructs. In particular, this pass transforms
        # operations into explicit sequence of operations that model the effect of this operation on the
        # different metadata. This pass uses affine constructs to materialize these effects.
        # Concretely, expanded-strided-metadata is used to decompose memref.subview as it has no
        # lowering in -convert-memref-to-llvm.
        "--expand-strided-metadata",
        "--lower-affine",
        "--convert-complex-to-standard",  # added for complex.exp lowering
        "--convert-complex-to-llvm",
        "--convert-math-to-llvm",
        # Must be run after -convert-math-to-llvm as it marks math::powf illegal but doesn't convert it.
        "--convert-math-to-libm",
        "--convert-arith-to-llvm",
        "--convert-memref-to-llvm",
        "--convert-index-to-llvm",
        "--convert-gradient-to-llvm",
        "--convert-quantum-to-llvm",
        # Remove any dead casts as the final pass expects to remove all existing casts,
        # but only those that form a loop back to the original type.
        "--canonicalize",
        "--reconcile-unrealized-casts",
    ]

    @staticmethod
    def get_output_filename(infile):
        if not infile.endswith(".buff.mlir"):
            raise ValueError(f"Input file ({infile}) is not a bufferized MLIR file")
        return infile.replace(".buff.mlir", ".llvm.mlir")


class QuantumCompilationPass(PassPipeline):
    """Pass pipeline to lower gradients."""

    _executable = get_executable_path("quantum", "quantum-opt")
    _default_flags = ["--lower-gradients"]

    @staticmethod
    def get_output_filename(infile):
        if not infile.endswith(".mlir"):
            raise ValueError(f"Input file ({infile}) for quantum transforms is not an MLIR file")
        return infile.replace(".mlir", ".opt.mlir")


class LLVMDialectToLLVMIR(PassPipeline):
    """Convert LLVM Dialect to LLVM-IR."""

    _executable = get_executable_path("llvm", "mlir-translate")
    _default_flags = ["--mlir-to-llvmir"]

    @staticmethod
    def get_output_filename(infile):
        if not infile.endswith(".llvm.mlir"):
            raise ValueError(f"Input file ({infile}) is not an LLVM dialect MLIR file")
        return infile.replace(".llvm.mlir", ".ll")


class LLVMIRToObjectFile(PassPipeline):
    """LLVMIR To Object File."""

    _executable = get_executable_path("llvm", "llc")
    _default_flags = [
        "--filetype=obj",
        "--relocation-model=pic",
    ]

    @staticmethod
    def get_output_filename(infile):
        if not infile.endswith(".ll"):
            raise ValueError(f"Input file ({infile}) for compilation is not an LLVMIR file")
        return infile.replace(".ll", ".o")


# pylint: disable=too-few-public-methods
class CompilerDriver:
    """Compiler Driver Interface
    In order to avoid relying on a single compiler at run time and allow the user some flexibility,
    this class defines a compiler resolution order where multiple known compilers are attempted.
    The order is defined as follows:
    1. A user specified compiler via the environment variable CATALYST_CC. It is expected that the
        user provided compiler is flag compatilble with GCC/Clang.
    2. clang: Priority is given to clang to maintain an LLVM toolchain through most of the process.
    3. gcc: Usually configured to link with LD.
    4. c99: Usually defaults to gcc, but no linker interface is specified.
    5. c89: Usually defaults to gcc, but no linker interface is specified.
    6. cc: Usually defaults to gcc, however POSIX states that it is deprecated.
    """

    _default_fallback_compilers = ["clang", "gcc", "c99", "c89", "cc"]

    @staticmethod
    def get_default_flags():
        """Re-compute the path where the libraries exist.

        The use case for this is if someone is in a python jupyter notebook and
        needs to change the environment mid computation.
        Returns
            (List[str]): The default flag list.
        """
        mlir_lib_path = get_lib_path("llvm", "MLIR_LIB_DIR")
        lrt_lib_path = get_lib_path("runtime", "RUNTIME_LIB_DIR")
        lrt_capi_path = os.path.join(lrt_lib_path, "capi")
        lrt_backend_path = os.path.join(lrt_lib_path, "backend")

        default_flags = [
            "-shared",
            "-rdynamic",
            "-Wl,-no-as-needed",
            f"-Wl,-rpath,{lrt_capi_path}:{lrt_backend_path}:{mlir_lib_path}",
            f"-L{mlir_lib_path}",
            f"-L{lrt_capi_path}",
            f"-L{lrt_backend_path}",
            "-lrt_backend",
            "-lrt_capi",
            "-lpthread",
            "-lmlir_c_runner_utils",  # required for memref.copy
        ]

        return default_flags

    @staticmethod
    def _get_compiler_fallback_order(fallback_compilers):
        """Compiler fallback order"""
        preferred_compiler = os.environ.get("CATALYST_CC", None)
        preferred_compiler_exists = CompilerDriver._exists(preferred_compiler)
        compilers = fallback_compilers
        emit_warning = preferred_compiler and not preferred_compiler_exists
        if emit_warning:
            msg = f"User defined compiler {preferred_compiler} is not in PATH. Will attempt fallback on available compilers."
            warnings.warn(msg, UserWarning)
        else:
            compilers = [preferred_compiler] + fallback_compilers
        return compilers

    @staticmethod
    # pylint: disable=redefined-outer-name
    def _exists(compiler):
        if compiler is None:
            return None
        return shutil.which(compiler)

    @staticmethod
    def _available_compilers(fallback_compilers):
        # pylint: disable=redefined-outer-name
        for compiler in CompilerDriver._get_compiler_fallback_order(fallback_compilers):
            if CompilerDriver._exists(compiler):
                yield compiler

    @staticmethod
    # pylint: disable=redefined-outer-name
    def _attempt_link(compiler, flags, infile, outfile, options):
        try:
            command = [compiler] + flags + [infile, "-o", outfile]
            run_writing_command(command, options)
            return True
        except subprocess.CalledProcessError:
            msg = (
                f"Compiler {compiler} failed during execution of command {command}. "
                "Will attempt fallback on available compilers."
            )
            warnings.warn(msg, UserWarning)
            return False

    @staticmethod
    def get_output_filename(infile):
        """Rename object file to shared object

        Args:
            infile (str): input file name
            outfile (str): output file name
        """
        if not infile.endswith(".o"):
            raise ValueError(f"Input file ({infile}) is not an object file")
        return infile.replace(".o", ".so")

    @staticmethod
    def run(infile, outfile=None, flags=None, fallback_compilers=None, options=None):
        """
        Link the infile against the necessary libraries and produce the outfile.

        Args:
            infile (str): input file
            outfile (str): output file
            Optional flags (List[str]): flags to be passed down to the compiler
            Optional fallback_compilers (List[str]): name of executables to be looked for in PATH
            Optional compile_options (CompileOptions): generic compilation options.
        Raises:
            EnvironmentError: The exception is raised when no compiler succeeded.
        """
        if outfile is None:
            outfile = CompilerDriver.get_output_filename(infile)
        if flags is None:
            flags = CompilerDriver.get_default_flags()
        if fallback_compilers is None:
            fallback_compilers = CompilerDriver._default_fallback_compilers
        for compiler in CompilerDriver._available_compilers(fallback_compilers):
            success = CompilerDriver._attempt_link(compiler, flags, infile, outfile, options)
            if success:
                return outfile
        msg = f"Unable to link {infile}. All available compiler options exhausted. Please provide a compatible compiler via $CATALYST_CC."
        raise EnvironmentError(msg)


class Compiler:
    """Compiles MLIR modules to shared objects."""

    def __init__(self):
        self.pass_pipeline_output = None
        # The temporary directory must be referenced by the wrapper class
        # in order to avoid being garbage collected
        # pylint: disable=consider-using-with
        self.workspace = tempfile.TemporaryDirectory()

    def run(self, mlir_module, keep_intermediate=False, passes=None, options=None):
        """Compile an MLIR module to a shared object.

        .. note::

            For compilation of hybrid quantum-classical PennyLane programs,
            please see the :func:`~.qjit` decorator.

        Args:
            mlir_module (Module): the MLIR module
            passes (List[Any]): the list of compilation passes

        Returns:
            Shared object
        """
        module_name = mlir_module.operation.attributes["sym_name"]
        # Convert MLIR string to Python string
        module_name = str(module_name)
        # Remove quotations
        module_name = module_name.replace('"', "")

        if keep_intermediate:
            parent_dir = os.getcwd()
            path = os.path.join(parent_dir, module_name)
            os.makedirs(path, exist_ok=True)
            workspace_name = os.path.abspath(path)
        else:
            workspace_name = self.workspace.name

        if passes is None:
            passes = [
                MHLOPass,
                QuantumCompilationPass,
                BufferizationPass,
                MLIRToLLVMDialect,
                LLVMDialectToLLVMIR,
                LLVMIRToObjectFile,
                CompilerDriver,
            ]

        self.pass_pipeline_output = {}

        filename = f"{workspace_name}/{module_name}.mlir"
        with open(filename, "w", encoding="utf-8") as f:
            mlir_module.operation.print(f, print_generic_op_form=False, assume_verified=True)

        for _pass in passes:
            output = _pass.run(filename, options=options)
            self.pass_pipeline_output[_pass] = output
            filename = os.path.abspath(output)

        return filename

    @staticmethod
    def _get_class_from_string(_pass):
        return getattr(sys.modules[__name__], _pass)

    def _get_output_file_of(self, _pass):
        cls = Compiler._get_class_from_string(_pass)
        return self.pass_pipeline_output.get(cls)

    def get_output_of(self, _pass):
        """Get the output IR of a pass.
        Args:
            _pass (str): name of pass class

        Returns
            (str): output IR
        """
        try:
            fname = self._get_output_file_of(_pass)
        except AttributeError as e:
            raise ValueError(f"Output for pass {_pass} not found.") from e
        with open(fname, "r", encoding="utf-8") as f:
            txt = f.read()
        return txt

    def print(self, _pass):
        """Print the output IR of pass.
        Args:
            _pass (str): name of pass class
        """
        txt = self.get_output_of(_pass)
        print(txt)
