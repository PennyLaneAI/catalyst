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
import warnings
from copy import deepcopy
from dataclasses import dataclass
from io import TextIOWrapper
from typing import Any, List, Optional, Tuple

from mlir_quantum.compiler_driver import run_compiler_driver

from catalyst._configuration import INSTALLED
from catalyst.utils.exceptions import CompileError
from catalyst.utils.filesystem import Directory

package_root = os.path.dirname(__file__)


@dataclass
class CompileOptions:
    """Generic compilation options, for which reasonable default values exist.

    Args:
        verbose (Optional[bool]): flag indicating whether to enable verbose output.
            Default is ``False``
        logfile (Optional[TextIOWrapper]): the logfile to write output to.
            Default is ``sys.stderr``
        keep_intermediate (Optional[bool]): flag indicating whether to keep intermediate results.
            Default is ``False``
        pipelines (Optional[List[Tuple[str,List[str]]]]): A list of tuples. The first entry of the
            tuple corresponds to the name of a pipeline. The second entry of the tuple corresponds
            to a list of MLIR passes.
        autograph (Optional[bool]): flag indicating whether experimental autograph support is to
            be enabled.
        lower_to_llvm (Optional[bool]): flag indicating whether to attempt the LLVM lowering after
            the main compilation pipeline is complete. Default is ``True``.
        abstracted_axes (Optional[Any]): TODO(@erick-xanadu): Add documentation
    """

    verbose: Optional[bool] = False
    logfile: Optional[TextIOWrapper] = sys.stderr
    target: Optional[str] = "binary"
    keep_intermediate: Optional[bool] = False
    pipelines: Optional[List[Any]] = None
    autograph: Optional[bool] = False
    lower_to_llvm: Optional[bool] = True
    abstracted_axes: Optional[Any] = None

    def __deepcopy__(self, memo):
        """Make a deep copy of all fields of a CompileOptions object except the logfile, which is
        copied directly"""
        return CompileOptions(
            **{
                k: (deepcopy(v) if k != "logfile" else self.logfile)
                for k, v in self.__dict__.items()
                if k != "logfile"
            }
        )

    def get_pipelines(self) -> List[Tuple[str, List[str]]]:
        """Get effective pipelines"""
        return self.pipelines if self.pipelines is not None else DEFAULT_PIPELINES


def run_writing_command(command: List[str], compile_options: Optional[CompileOptions]) -> None:
    """Run the command after optionally announcing this fact to the user.

    Args:
        command (List[str]): command to be sent to a subprocess.
        compile_options (Optional[CompileOptions]): compile options.
    """

    if compile_options.verbose:
        print(f"[SYSTEM] {' '.join(command)}", file=compile_options.logfile)
    subprocess.run(command, check=True)


default_lib_paths = {
    "llvm": os.path.join(package_root, "../../mlir/llvm-project/build/lib"),
    "runtime": os.path.join(package_root, "../../runtime/build/lib"),
    "enzyme": os.path.join(package_root, "../../mlir/Enzyme/build/Enzyme"),
}


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
            "scatter-lowering",
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
            "catalyst-bufferize",
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
            "convert-catalyst-to-llvm",
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

        system_flags = []
        if platform.system() == "Linux":
            system_flags += ["-Wl,-no-as-needed"]
        elif platform.system() == "Darwin":  # pragma: nocover
            system_flags += ["-Wl,-arch_errors_fatal"]

        default_flags = [
            "-shared",
            "-rdynamic",
            *system_flags,
            *lib_path_flags,
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
            outfile (Optional[str]): output file
            flags (Optional[List[str]]): flags to be passed down to the compiler
            fallback_compilers (Optional[List[str]]): name of executables to be looked for in PATH
            compile_options (Optional[CompileOptions]): generic compilation options.
        Raises:
            EnvironmentError: The exception is raised when no compiler succeeded.
        """
        if outfile is None:
            outfile = LinkerDriver.get_output_filename(infile)
        if flags is None:
            flags = LinkerDriver.get_default_flags()
        if fallback_compilers is None:
            fallback_compilers = LinkerDriver._default_fallback_compilers
        if options is None:
            options = CompileOptions()
        for compiler in LinkerDriver._available_compilers(fallback_compilers):
            success = LinkerDriver._attempt_link(compiler, flags, infile, outfile, options)
            if success:
                return outfile
        msg = f"Unable to link {infile}. Please check the output for any error messages. If no "
        msg += "compiler was found by Catalyst, please specify a compatible one via $CATALYST_CC."
        raise CompileError(msg)


class Compiler:
    """Compiles MLIR modules to shared objects by executing the Catalyst compiler driver library."""

    def __init__(self, options: Optional[CompileOptions] = None):
        self.options = options if options is not None else CompileOptions()
        self.last_compiler_output = None

    def run_from_ir(self, ir: str, module_name: str, workspace: Directory):
        """Compile a shared object from a textual IR (MLIR or LLVM).

        Args:
            ir (str): Textual MLIR to be compiled
            module_name (str): Module name to use for naming
            workspace (Directory): directory that holds output files and/or debug dumps.

        Returns:
            output_filename (str): Output file name. For the default pipeline this would be the
                                   shard object library path.
            out_IR (str): Output IR in textual form. For the default pipeline this would be the
                          LLVM IR.
            A list of:
               func_name (str) Inferred name of the main function
               ret_type_name (str) Inferred main function result type name
        """
        assert isinstance(
            workspace, Directory
        ), f"Compiler expects a Directory type, got {type(workspace)}."
        assert workspace.is_dir(), f"Compiler expects an existing directory, got {workspace}."

        lower_to_llvm = (
            self.options.lower_to_llvm if self.options.lower_to_llvm is not None else False
        )

        if self.options.verbose:
            print(f"[LIB] Running compiler driver in {workspace}", file=self.options.logfile)

        try:
            compiler_output = run_compiler_driver(
                ir,
                str(workspace),
                module_name,
                keep_intermediate=self.options.keep_intermediate,
                verbose=self.options.verbose,
                pipelines=self.options.get_pipelines(),
                lower_to_llvm=lower_to_llvm,
            )
        except RuntimeError as e:
            raise CompileError(*e.args) from e

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

    def get_output_of(self, pipeline) -> Optional[str]:
        """Get the output IR of a pipeline.
        Args:
            pipeline (str): name of pass class

        Returns
            (Optional[str]): output IR
        """
        if len(dict(self.options.get_pipelines()).get(pipeline, [])) == 0:
            warnings.warn("Requesting an output of an empty pipeline")  # pragma: no cover

        if not self.last_compiler_output:
            return None

        return self.last_compiler_output.get_pipeline_output(pipeline)

    def print(self, pipeline):
        """Print the output IR of pass.
        Args:
            pipeline (str): name of pass class
        """
        print(self.get_output_of(pipeline))  # pragma: no cover
