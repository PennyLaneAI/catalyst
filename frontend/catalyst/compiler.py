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

import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from io import TextIOWrapper
from typing import List, Optional, Any

from mlir_quantum.compiler_driver._catalystDriver import run_compiler_driver

from catalyst._configuration import INSTALLED

package_root = os.path.dirname(__file__)


@dataclass
class CompileOptions:
    """Generic compilation options, for which reasonable default values exist.

    Args:
        verbose (bool, optional): flag indicating whether to enable verbose output.
            Default is ``False``
        logfile (TextIOWrapper, optional): the logfile to write output to.
            Default is ``sys.stderr``
        keep_intermediate (bool, optional): flag indicating whether to keep intermediate results.
            Default is ``False``
    """

    verbose: Optional[bool] = False
    logfile: Optional[TextIOWrapper] = sys.stderr
    target: Optional[str] = "binary"
    keep_intermediate: Optional[bool] = False
    pipelines: Optional[List[Any]] = None
    autograph: Optional[bool] = False


def run_writing_command(
    command: List[str], compile_options: Optional[CompileOptions] = None
) -> None:
    """Run the command after optionally announcing this fact to the user"""
    if compile_options is None:
        compile_options = CompileOptions()

    if compile_options.verbose:
        print(f"[SYSTEM] {' '.join(command)}", file=compile_options.logfile)
    subprocess.run(command, check=True)


default_bin_paths = {
    "llvm": os.path.join(package_root, "../../mlir/llvm-project/build/bin"),
    "mhlo": os.path.join(package_root, "../../mlir/mlir-hlo/build/bin"),
    "quantum": os.path.join(package_root, "../../mlir/build/bin"),
}

default_lib_paths = {
    "llvm": os.path.join(package_root, "../../mlir/llvm-project/build/lib"),
    "runtime": os.path.join(package_root, "../../runtime/build/lib"),
    "enzyme": os.path.join(package_root, "../../mlir/Enzyme/build/Enzyme"),
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


DEFAULT_PIPELINES = [
    (
        "HLOLoweringPass",
        [
            "canonicalize",
            "func.func(chlo-legalize-to-hlo)",
            "stablehlo-legalize-to-hlo",
            "func.func(mhlo-legalize-control-flow)",
            "func.func(hlo-legalize-to-linalg)",
            "func.func(mhlo-legalize-to-std)",
            "convert-to-signless",
            "func.func(scalarize)",
            "canonicalize",
        ],
    ),
    (
        "QuantumCompilationPass",
        [
            "lower-gradients",
            "adjoint-lowering",
        ],
    ),
    (
        "BufferizationPass",
        [
            "one-shot-bufferize{dialect-filter=memref}",
            "inline",
            "gradient-bufferize",
            "scf-bufferize",
            "convert-tensor-to-linalg",  # tensor.pad
            "convert-elementwise-to-linalg",  # Must be run before --arith-bufferize
            "arith-bufferize",
            "empty-tensor-to-alloc-tensor",
            "func.func(bufferization-bufferize)",
            "func.func(tensor-bufferize)",
            "func.func(linalg-bufferize)",
            "func.func(tensor-bufferize)",
            "quantum-bufferize",
            "func-bufferize",
            "func.func(finalizing-bufferize)",
            "func.func(buffer-hoisting)",
            "func.func(buffer-loop-hoisting)",
            "func.func(buffer-deallocation)",
            "convert-arraylist-to-memref",
            "convert-bufferization-to-memref",
            "canonicalize",
            # "cse",
            "cp-global-memref",
        ],
    ),
    (
        "MLIRToLLVMDialect",
        [
            "convert-gradient-to-llvm",
            "func.func(convert-linalg-to-loops)",
            "convert-scf-to-cf",
            # This pass expands memref ops that modify the metadata of a memref (sizes, offsets,
            # strides) into a sequence of easier to analyze constructs. In particular, this pass
            # transforms ops into explicit sequence of operations that model the effect of this
            # operation on the different metadata. This pass uses affine constructs to materialize
            # these effects. Concretely, expanded-strided-metadata is used to decompose
            # memref.subview as it has no lowering in -finalize-memref-to-llvm.
            "expand-strided-metadata",
            "lower-affine",
            "arith-expand",  # some arith ops (ceildivsi) require expansion to be lowered to llvm
            "convert-complex-to-standard",  # added for complex.exp lowering
            "convert-complex-to-llvm",
            "convert-math-to-llvm",
            # Run after -convert-math-to-llvm as it marks math::powf illegal without converting it.
            "convert-math-to-libm",
            "convert-arith-to-llvm",
            "finalize-memref-to-llvm{use-generic-functions}",
            "convert-index-to-llvm",
            "convert-quantum-to-llvm",
            "emit-catalyst-py-interface",
            # Remove any dead casts as the final pass expects to remove all existing casts,
            # but only those that form a loop back to the original type.
            "canonicalize",
            "reconcile-unrealized-casts",
        ],
    ),
]


class LinkerDriver:
    """Compiler used to drive the linking stage.
    In order to avoid relying on a single linker at run time and allow the user some flexibility,
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
        rt_lib_path = get_lib_path("runtime", "RUNTIME_LIB_DIR")
        error_flag_apple = "-Wl,-arch_errors_fatal"
        error_flag_linux = ""
        error_flag = error_flag_linux if platform.system() == "Linux" else error_flag_apple

        default_flags = [
            "-shared",
            "-rdynamic",
            f"-Wl,-rpath,{rt_lib_path}",
            f"-Wl,-rpath,{mlir_lib_path}",
            f"-L{mlir_lib_path}",
            f"-L{rt_lib_path}",
            "-lrt_backend",
            "-lrt_capi",
            "-lpthread",
            f"{error_flag}",
            "-lmlir_c_runner_utils",  # required for memref.copy
        ]

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
        path = pathlib.Path(infile)
        if not path.exists():
            raise FileNotFoundError(f"Cannot find {infile}.")
        return str(path.with_suffix(".so"))

    @staticmethod
    def run(infile, outfile=None, flags=None, fallback_compilers=None, options=None):
        """
        Link the infile against the necessary libraries and produce the outfile.

        Args:
            infile (str): input file
            outfile (str): output file
            flags (List[str], optional): flags to be passed down to the compiler
            fallback_compilers (List[str], optional): name of executables to be looked for in PATH
            compile_options (CompileOptions, optional): generic compilation options.
        Raises:
            EnvironmentError: The exception is raised when no compiler succeeded.
        """
        if outfile is None:
            outfile = LinkerDriver.get_output_filename(infile)
        if flags is None:
            flags = LinkerDriver.get_default_flags()
        if fallback_compilers is None:
            fallback_compilers = LinkerDriver._default_fallback_compilers
        for compiler in LinkerDriver._available_compilers(fallback_compilers):
            success = LinkerDriver._attempt_link(compiler, flags, infile, outfile, options)
            if success:
                return outfile
        msg = f"Unable to link {infile}. All available compiler options exhausted. "
        msg += "Please provide a compatible compiler via $CATALYST_CC."
        raise EnvironmentError(msg)


class Compiler:
    """Compiles MLIR modules to shared objects by executing the Catalyst compiler driver library."""

    def __init__(self, options: Optional[CompileOptions] = None):
        self.options = options if options is not None else CompileOptions
        self.last_compiler_output = None
        self.last_workspace = None
        self.last_tmpdir = None

    def run_from_ir(
        self,
        ir: str,
        module_name: str,
        pipelines=None,
        lower_to_llvm=True,
    ):
        """Compile a shared object from a textual IR (MLIR or LLVM).

        Args:
            ir (str): Textual MLIR to be compiled
            module_name (str): Module name to use for naming
            pipelines (list, optional): Custom compilation pipelines configuration. The default is
                                        None which means to use the default pipelines config.
            lower_to_llvm (bool, optional): Whether to lower to LLVM after finishing processing of
                                            the pipelines. Defaults to True.

        Returns:
            output_filename (str): Output file name. For the default pipeline this would be the
                                   shard object library path.
            out_IR (str): Output IR in textual form. For the default pipeline this would be the
                          LLVM IR.
            A list of:
               func_name (str) Inferred name of the main function
               ret_type_name (str) Inferred main function result type name
        """
        pipelines = pipelines if pipelines is not None else DEFAULT_PIPELINES
        if self.options.keep_intermediate:
            workspace = os.path.abspath(os.path.join(os.getcwd(), module_name))
            os.makedirs(workspace, exist_ok=True)
        else:
            # pylint: disable=consider-using-with
            self.last_tmpdir = tempfile.TemporaryDirectory()
            workspace = self.last_tmpdir.name

        self.last_workspace = workspace

        if self.options.verbose:
            print(f"[LIB] Running compiler driver in {workspace}", file=self.options.logfile)

        compiler_output = run_compiler_driver(
            ir,
            workspace,
            module_name,
            keep_intermediate=self.options.keep_intermediate,
            verbose=self.options.verbose,
            pipelines=pipelines,
            lower_to_llvm=lower_to_llvm,
        )

        if self.options.verbose:
            for line in compiler_output.get_diagnostic_messages().strip().split("\n"):
                print(f"[LIB] {line}", file=self.options.logfile)

        filename = compiler_output.get_object_filename()
        out_IR = compiler_output.get_output_ir()
        func_name = compiler_output.get_function_attributes().get_function_name()
        ret_type_name = compiler_output.get_function_attributes().get_return_type()

        if lower_to_llvm:
            output = LinkerDriver.run(filename, options=self.options)
            output_filename = str(pathlib.Path(output).absolute())
        else:
            output_filename = filename

        self.last_compiler_output = compiler_output
        return output_filename, out_IR, [func_name, ret_type_name]

    def run(self, mlir_module, *args, **kwargs):
        """Compile an MLIR module to a shared object.

        .. note::

            For compilation of hybrid quantum-classical PennyLane programs,
            please see the :func:`~.qjit` decorator.

        Returns:
            (str): filename of shared object
        """

        return self.run_from_ir(
            mlir_module.operation.get_asm(
                binary=False, print_generic_op_form=False, assume_verified=True
            ),
            *args,
            module_name=str(mlir_module.operation.attributes["sym_name"]).replace('"', ""),
            **kwargs,
        )

    def get_output_of(self, pipeline) -> Optional[str]:
        """Get the output IR of a pipeline.
        Args:
            pipeline (str): name of pass class

        Returns
            (Optional[str]): output IR
        """
        return (
            self.last_compiler_output.get_pipeline_output(pipeline)
            if self.last_compiler_output
            else None
        )

    def print(self, pipeline):
        """Print the output IR of pass.
        Args:
            pipeline (str): name of pass class
        """
        print(self.get_output_of(pipeline))  # pragma: no cover
