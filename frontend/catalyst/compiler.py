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
import shutil
import subprocess
import sys
import warnings

from catalyst._configuration import INSTALLED

package_root = os.path.dirname(__file__)

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


def get_lib_path(project, env):
    """Get the library path."""
    return (
        os.path.join(package_root, "lib")
        if INSTALLED
        else os.getenv(env, default_lib_paths.get(project, ""))
    )


class Pass(abc.ABC):
    """Abstract Pass class."""

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
    def _run(infile, outfile, executable, flags):
        command = [executable] + flags + [infile, "-o", outfile]
        subprocess.run(command, check=True)

    @classmethod
    def run(cls, infile, outfile=None, executable=None, flags=None):
        """Run the MHLO pass.

        Args:
            infile (str): path to MLIR file to be compiled
            outfile (str): path to output file, defaults to replacing extension in infile to .nohlo.
            executable (str): path to executable, defaults to mlir-hlo-opt
            flags (List[str]): flags to mlir-hlo-opt, defaults to _default_flags.
        """
        if outfile is None:
            outfile = cls.get_output_filename(infile)
        if executable is None:
            executable = cls._executable
        if flags is None:
            flags = cls._default_flags
        cls._run(infile, outfile, executable, flags)
        return outfile


# pylint: disable=too-few-public-methods
class MHLOPass(Pass):
    """MHLO Pass."""

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
        return infile.replace(".mlir", ".nohlo.mlir")


class BufferizationPass(Pass):
    """Bufferization Pass."""

    _executable = get_executable_path("quantum", "quantum-opt")
    _default_flags = [
        "--lower-gradients",
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
        # "--buffer-deallocation",
        "--convert-bufferization-to-memref",
        "--canonicalize",
        "--cse",
    ]

    @staticmethod
    def get_output_filename(infile):
        return infile.replace(".nohlo.mlir", ".buff.mlir")


class MLIRToLLVMDialect(Pass):
    """MLIR To LLVM"""

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
        return infile.replace(".buff.mlir", ".llvm.mlir")


class LLVMDialectToLLVMIR(Pass):
    """Convert LLVM Dialect to LLVM-IR."""

    _executable = get_executable_path("llvm", "mlir-translate")
    _default_flags = ["--mlir-to-llvmir"]

    @staticmethod
    def get_output_filename(infile):
        return infile.replace(".llvm.mlir", ".ll")


class LLVMIRToObjectFile(Pass):
    """LLVMIR To Object File."""

    _executable = get_executable_path("llvm", "llc")
    _default_flags = [
        "--filetype=obj",
        "--relocation-model=pic",
    ]

    @staticmethod
    def get_output_filename(infile):
        return infile.replace(".ll", ".o")


mlir_lib_path = get_lib_path("llvm", "MLIR_LIB_DIR")
lrt_lib_path = get_lib_path("runtime", "RUNTIME_LIB_DIR")
lrt_capi_path = os.path.join(lrt_lib_path, "capi")
lrt_backend_path = os.path.join(lrt_lib_path, "backend")


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

    _default_flags = [
        "-shared",
        "-rdynamic",
        f"-L{mlir_lib_path}",
        "-Wl,-no-as-needed",
        f"-Wl,-rpath,{mlir_lib_path}",
        f"-L{lrt_capi_path}",
        f"-L{lrt_backend_path}",
        f"-Wl,-rpath,{lrt_capi_path}:{lrt_backend_path}",
        f"-L{lrt_capi_path}",
        f"-L{lrt_backend_path}",
        f"-Wl,-rpath,{lrt_capi_path}:{lrt_backend_path}",
        "-lrt_backend",
        "-lrt_capi",
        "-lpthread",
        "-lmlir_c_runner_utils",  # required for memref.copy
    ]

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
    def _attempt_link(compiler, flags, infile, outfile):
        try:
            command = [compiler] + flags + [infile, "-o", outfile]
            subprocess.run(command, check=True)
            return True
        except subprocess.CalledProcessError:
            msg = f"Compiler {compiler} failed during execution of command {command}. Will attempt fallback on available compilers."
            warnings.warn(msg, UserWarning)
            return False

    @staticmethod
    def get_output_filename(infile):
        """Rename object file to shared object

        Args:
            infile (str): input file name
            outfile (str): output file name
        """
        return infile.replace(".o", ".so")

    @staticmethod
    def run(infile, outfile=None, flags=None, fallback_compilers=None):
        """
        Link the infile against the necessary libraries and produce the outfile.

        Args:
            infile (str): input file
            outfile (str): output file
            Optional flags (List[str]): flags to be passed down to the compiler
            Optional fallback_compilers (List[str]): name of executables to be looked for in PATH
        Raises:
            EnvironmentError: The exception is raised when no compiler succeeded.
        """
        if outfile is None:
            outfile = CompilerDriver.get_output_filename(infile)
        if flags is None:
            flags = CompilerDriver._default_flags
        if fallback_compilers is None:
            fallback_compilers = CompilerDriver._default_fallback_compilers
        # pylint: disable=redefined-outer-name
        for compiler in CompilerDriver._available_compilers(fallback_compilers):
            success = CompilerDriver._attempt_link(compiler, flags, infile, outfile)
            if success:
                return outfile
        msg = f"Unable to link {infile}. All available compiler options exhausted. Please provide a compatible compiler via $CATALYST_CC."
        raise EnvironmentError(msg)


class Compiler:
    """Compiles MLIR modules to shared objects."""

    def __init__(self):
        self.order = None

    def run(self, mlir_module, workspace, passes=None):
        """Compile an MLIR module to a shared object.

        .. note::

            For compilation of hybrid quantum-classical PennyLane programs,
            please see the :func:`~.qjit` decorator.

        Args:
            mlir_module (Module): the MLIR module
            workspace (str): the absolute path to the MLIR module
            passes (List[Any]): the list of compilation passes

        Returns:
            Shared object
        """

        module_name = mlir_module.operation.attributes["sym_name"]
        # Convert MLIR string to Python string
        module_name = str(module_name)
        # Remove quotations
        module_name = module_name.replace('"', "")
        # need to create a temporary file with the string contents
        filename = workspace + f"/{module_name}.mlir"
        with open(filename, "w", encoding="utf-8") as f:
            mlir_module.operation.print(f, print_generic_op_form=False, assume_verified=True)

        if passes is None:
            passes = [
                MHLOPass,
                BufferizationPass,
                MLIRToLLVMDialect,
                LLVMDialectToLLVMIR,
                LLVMIRToObjectFile,
                CompilerDriver,
            ]

        self.order = {}
        for _pass in passes:
            filename = _pass.run(filename)
            self.order[_pass] = filename

        return filename

    @staticmethod
    def _get_class_from_string(_pass):
        return getattr(sys.modules[__name__], _pass)

    def _get_output_file_of(self, _pass):
        cls = Compiler._get_class_from_string(_pass)
        return self.order[cls]

    def get_output_of(self, _pass):
        """Get the output IR of a pass.

        Args:
            _pass (str): name of pass class

        Returns
            (str): output IR
        """
        fname = self._get_output_file_of(_pass)
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
