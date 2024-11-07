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
import warnings
from copy import deepcopy
from dataclasses import dataclass
from io import TextIOWrapper
from os import path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from mlir_quantum.compiler_driver import run_compiler_driver

from catalyst.logging import debug_logger, debug_logger_init
from catalyst.utils.exceptions import CompileError
from catalyst.utils.filesystem import Directory
from catalyst.utils.runtime_environment import get_lib_path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

package_root = os.path.dirname(__file__)

DEFAULT_CUSTOM_CALLS_LIB_PATH = path.join(package_root, "utils")


# pylint: disable=too-many-instance-attributes
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
        autograph_include (Optional[Iterable[str]]): A list of (sub)modules to be allow-listed
        for autograph conversion.
        async_qnodes (Optional[bool]): flag indicating whether experimental asynchronous execution
            of QNodes support is to be enabled.
        lower_to_llvm (Optional[bool]): flag indicating whether to attempt the LLVM lowering after
            the main compilation pipeline is complete. Default is ``True``.
        static_argnums (Optional[Union[int, Iterable[int]]]): indices of static arguments.
            Default is ``None``.
        static_argnames (Optional[Union[str, Iterable[str]]]): names of static arguments.
            Default is ``None``.
        abstracted_axes (Optional[Any]): store the abstracted_axes value. Defaults to ``None``.
        disable_assertions (Optional[bool]): disables all assertions. Default is ``False``.
        seed (Optional[int]) : the seed for random operations in a qjit call.
            Default is None.
        experimental_capture (bool): If set to ``True``,
            use PennyLane's experimental program capture capabilities
            to capture the function for compilation.
        circuit_transform_pipeline (Optional[dict[str, dict[str, str]]]):
            A dictionary that specifies the quantum circuit transformation pass pipeline order,
            and optionally arguments for each pass in the pipeline.
            Default is None.
    """

    verbose: Optional[bool] = False
    logfile: Optional[TextIOWrapper] = sys.stderr
    target: Optional[str] = "binary"
    keep_intermediate: Optional[bool] = False
    pipelines: Optional[List[Any]] = None
    autograph: Optional[bool] = False
    autograph_include: Optional[Iterable[str]] = ()
    async_qnodes: Optional[bool] = False
    static_argnums: Optional[Union[int, Iterable[int]]] = None
    static_argnames: Optional[Union[str, Iterable[str]]] = None
    abstracted_axes: Optional[Union[Iterable[Iterable[str]], Dict[int, str]]] = None
    lower_to_llvm: Optional[bool] = True
    checkpoint_stage: Optional[str] = ""
    disable_assertions: Optional[bool] = False
    seed: Optional[int] = None
    experimental_capture: Optional[bool] = False
    circuit_transform_pipeline: Optional[dict[str, dict[str, str]]] = None

    def __post_init__(self):
        # Check that async runs must not be seeded
        if self.async_qnodes and self.seed != None:
            raise CompileError(
                """
                Seeding has no effect on asyncronous qnodes,
                as the execution order of parallel runs is not guaranteed.
                As such, seeding an asynchronous run is not supported.
                """
            )

        # Check that seed is 32-bit unsigned int
        if (self.seed != None) and (self.seed < 0 or self.seed > 2**32 - 1):
            raise ValueError(
                """
                Seed must be an unsigned 32-bit integer!
                """
            )

        # Make the format of static_argnums easier to handle.
        static_argnums = self.static_argnums
        if static_argnums is None:
            self.static_argnums = ()
        elif isinstance(static_argnums, int):
            self.static_argnums = (static_argnums,)
        elif isinstance(static_argnums, Iterable):
            self.static_argnums = tuple(static_argnums)

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
        if self.pipelines:
            return self.pipelines
        elif self.async_qnodes:
            return DEFAULT_ASYNC_PIPELINES  # pragma: nocover
        if self.disable_assertions:
            if "disable-assertion" not in QUANTUM_COMPILATION_PASS[1]:
                QUANTUM_COMPILATION_PASS[1].append("disable-assertion")
        else:
            if "disable-assertion" in QUANTUM_COMPILATION_PASS[1]:
                QUANTUM_COMPILATION_PASS[1].remove("disable-assertion")
        return DEFAULT_PIPELINES


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


ENFORCE_RUNTIME_INVARIANTS_PASS = (
    "EnforeRuntimeInvariantsPass",
    [
        # We want the invariant that transforms that generate multiple
        # tapes will generate multiple qnodes. One for each tape.
        # Split multiple tapes enforces that invariant.
        "split-multiple-tapes",
        # Run the transform sequence defined in the MLIR module
        "apply-transform-sequence",
        # Nested modules are something that will be used in the future
        # for making device specific transformations.
        # Since at the moment, nothing in the runtime is using them
        # and there is no lowering for them,
        # we inline them to preserve the semantics. We may choose to
        # keep inlining modules targetting the Catalyst runtime.
        # But qnodes targetting other backends may choose to lower
        # this into something else.
        "inline-nested-module",
    ],
)

HLO_LOWERING_PASS = (
    "HLOLoweringPass",
    [
        "canonicalize",
        "func.func(chlo-legalize-to-hlo)",
        "stablehlo-legalize-to-hlo",
        "func.func(mhlo-legalize-control-flow)",
        "func.func(hlo-legalize-to-linalg)",
        "func.func(mhlo-legalize-to-std)",
        "func.func(hlo-legalize-sort)",
        "convert-to-signless",
        "canonicalize",
        "scatter-lowering",
        "hlo-custom-call-lowering",
        "cse",
        "func.func(linalg-detensorize{aggressive-mode})",
        "detensorize-scf",
        "canonicalize",
    ],
)

QUANTUM_COMPILATION_PASS = (
    "QuantumCompilationPass",
    [
        "annotate-function",
        "lower-mitigation",
        "lower-gradients",
        "adjoint-lowering",
        "disable-assertion",
    ],
)

# From: https://mlir.llvm.org/docs/Bufferization/#overview
#
# Preprocessing
#     |               rewrite_in_destination_passing_style
#     |               -eliminate-empty-tensors
# Bufferization
#     |               -one-shot-bufferize
# Buffer-Level
# Optimizations
#     |               -buffer-hoisting
#     |               -buffer-loop-hoisting
#     |               -buffer-results-to-out-params
#     |               -drop-equivalent-buffer-results
#     |               -promote-buffers-to-stack
# Deallocation
#     |               -buffer-deallocation-pipeline

BUFFERIZATION_PASS = (
    "BufferizationPass",
    [
        "inline",
        "gradient-preprocess",
        "convert-elementwise-to-linalg",
        "canonicalize",
        # Preprocessing:
        # rewrite_in_destination_passing_style
        #
        # We are not rewriting everything in DPS before -one-shot-bufferize
        # This was discussed with the main author of the -one-shot-bufferize
        # pass and he stated the following:
        #
        #     One-Shot Bufferize was designed for ops that are in DPS (destination-passing style).
        #     Ops that are not in DPS can still be bufferized,
        #     but a new buffer will be allocated for every tensor result.
        #     That’s functionally correct but inefficient.
        #
        #     I’m not sure whether it’s better to first migrate to the new bufferization,
        #     then turn the ops into DPS ops, or do it the other way around.
        #     One benefit of implementing the bufferization first is that
        #     it’s a smaller step that you can already run end-to-end.
        #     And you can think of the DPS of a performance improvement on top of it.
        #
        # https://discourse.llvm.org/t/steps-of-migrating-to-one-shot-bufferization/81062/2
        #
        # Here, please note that gradient-preprocessing is different than rewriting in DPS.
        # So, overall, we are skipping this section while we first focus on migrating to the
        # new -one-shot-bufferize
        "eliminate-empty-tensors",
        (
            # Before we enter one-shot-bufferize, here is what we expect:
            # * Given
            #
            #     One-Shot Bufferize was designed for ops that are in DPS
            #     (destination-passing style).
            #     Ops that are not in DPS can still be bufferized,
            #     but a new buffer will be allocated for every tensor result.
            #     That’s functionally correct but inefficient.
            #
            #   https://discourse.llvm.org/t/steps-of-migrating-to-one-shot-bufferization/81062/2
            #
            #   we expect that results will be (automatically?) converted into new buffers. And it
            #   is up to us to just define the bufferization for the operands.
            #
            # So what is the state of the catalyst, gradient, quantum dialects at this point?
            #
            # Let's start with quantum:
            #
            # |-------------------------|--------------------|
            # |      operation          |  has result tensor |
            # |-------------------------|--------------------|
            # | quantum.set_state       |                    |
            # | quantum.set_basis_state |                    |
            # | quantum.unitary         |                    |
            # | quantum.hermitian       |                    |
            # | quantum.hamiltonian     |                    |
            # | quantum.sample_op       |     YES            |
            # | quantum.counts_op       |     YES            |
            # | quantum.probs_op        |     YES            |
            # | quantum.state_op        |     YES            |
            # |-------------------------|--------------------|
            # | catalyst.print_op       |                    |
            # | catalyst.custom_call    |     YES            |
            # | catalyst.callback       |                    |
            # | catalyst.callback_call  |     YES            |
            # | catalyst.launch_kernel  |     YES            |
            # |-------------------------|--------------------|
            # | gradient.grad           |     YES            |
            # | gradient.value_and_grad |     YES            |
            # | gradient.adjoint        |     YES            |
            # | gradient.backprop       |     YES            |
            # | gradient.jvp            |     YES            |
            # | gradient.vjp            |     YES            |
            # | gradient.forward        |     YES            |
            # | gradient.reverse        |     YES            |
            # |-------------------------|--------------------|
            #
            # So what this means is that for the operands, all the ones that have the YES
            # means that no operands are written to. They are only read.
            "one-shot-bufferize"
            "{"
            "bufferize-function-boundaries "
            # - Bufferize function boundaries (experimental).
            #
            #     By default, function boundaries are not bufferized.
            #     This is because there are currently limitations around function graph
            #     bufferization:
            #     recursive calls are not supported.
            #     As long as there are no recursive calls, function boundary bufferization can be
            #     enabled with bufferize-function-boundaries.
            #     Each tensor function argument and tensor function result is then turned into a memref.
            #     The layout map of the memref type can be controlled with function-boundary-type-conversion.
            #
            # https://mlir.llvm.org/docs/Bufferization/#using-one-shot-bufferize
            "allow-return-allocs-from-loops "
            # - Allows returning/yielding new allocations from a loop.
            # https://github.com/llvm/llvm-project/pull/83964
            # https://github.com/llvm/llvm-project/pull/87594
            "function-boundary-type-conversion=identity-layout-map"
            # - Controls layout maps when bufferizing function signatures.
            #     You can control the memref types at the function boundary with
            #     function-boundary-type-conversion. E.g., if you set it to identity-layout-map,
            #     you should get the same type as with --func-bufferize.
            #     By default, we put a fully dynamic layout map strided<[?, ?], offset: ?>
            #     because that works best if you don't know what layout map the buffers at
            #     the call site have -- you can always cast a buffer to a type with
            #     fully dynamic layout map. (But not the other way around. That may require a
            #     reallocation.)
            #
            #  https://discord.com/channels/636084430946959380/642426447167881246/1212338527824515102
            "}"
        ),
        # Remove dead memrefToTensorOp's
        # introduced during gradient-bufferize of callbacks
        # TODO: Figure out how to remove this.
        "gradient-postprocess",
        "func.func(buffer-hoisting)",
        "func.func(buffer-loop-hoisting)",
        # TODO: Figure out how to include the other buffer-level optimizations.
        # -buffer-results-to-out-params,
        # -drop-equivalent-buffer-results,
        # -promote-buffers-to-stack
        # Deallocation
        # The buffer deallocation pass has been deprecated in favor of the
        # ownership-based buffer deallocation pipeline.
        # The deprecated pass has some limitations that may cause memory leaks in the resulting IR.
        # TODO: Switch to one-shot-bufferization once it is merged.
        "func.func(buffer-deallocation)",
        # catalyst.list_* operations are not bufferized through
        # the bufferization interface
        # This is because they store a memref inside of a memref
        # which is incompatible with the bufferization pipeline.
        "convert-arraylist-to-memref",
        "convert-bufferization-to-memref",
        # Must be after convert-bufferization-to-memref
        # otherwise there are issues in lowering of dynamic tensors.
        "canonicalize",
        # "cse",
        "cp-global-memref",
    ],
)

BUFFERIZATION_ASYNC_PASS = (
    "BufferizationPass",
    [
        # TODO: Can we remove copy-before-write?
        # copy-before-write:
        # Skip the analysis. Make a buffer copy on every write.
        s.replace("}", " copy-before-write}") if s.startswith("one-shot-bufferize") else s
        for s in BUFFERIZATION_PASS[1]
    ],
)

MLIR_TO_LLVM_PASS = (
    "MLIRToLLVMDialect",
    [
        "expand-realloc",
        "convert-gradient-to-llvm",
        "memrefcpy-to-linalgcpy",
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
        "memref-to-llvm-tbaa",  # load and store are converted to llvm with tbaa tags
        "finalize-memref-to-llvm{use-generic-functions}",
        "convert-index-to-llvm",
        "convert-catalyst-to-llvm",
        "convert-quantum-to-llvm",
        # There should be no identical code folding
        # (`mergeIdenticalBlocks` in the MLIR source code)
        # between convert-async-to-llvm and
        # add-exception-handling.
        # So, if there's a pass from the beginning
        # of this list to here that does folding
        # add-exception-handling will fail to add async.drop_ref
        # correctly. See https://github.com/PennyLaneAI/catalyst/pull/995
        "add-exception-handling",
        "emit-catalyst-py-interface",
        # Remove any dead casts as the final pass expects to remove all existing casts,
        # but only those that form a loop back to the original type.
        "canonicalize",
        "reconcile-unrealized-casts",
        "gep-inbounds",
        "register-inactive-callback",
    ],
)


DEFAULT_PIPELINES = [
    ENFORCE_RUNTIME_INVARIANTS_PASS,
    HLO_LOWERING_PASS,
    QUANTUM_COMPILATION_PASS,
    BUFFERIZATION_PASS,
    MLIR_TO_LLVM_PASS,
]

MLIR_TO_LLVM_ASYNC_PASS = deepcopy(MLIR_TO_LLVM_PASS)
MLIR_TO_LLVM_ASYNC_PASS[1][:0] = [
    "qnode-to-async-lowering",
    "async-func-to-async-runtime",
    "async-to-async-runtime",
    "convert-async-to-llvm",
]

DEFAULT_ASYNC_PIPELINES = [
    ENFORCE_RUNTIME_INVARIANTS_PASS,
    HLO_LOWERING_PASS,
    QUANTUM_COMPILATION_PASS,
    BUFFERIZATION_ASYNC_PASS,
    MLIR_TO_LLVM_ASYNC_PASS,
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

        if platform.system() == "Linux":
            file_path_within_package = "../scipy.libs/"
            file_extension = ".so"
        else:  # pragma: nocover
            msg = "Attempting to use catalyst on an unsupported system"
            assert platform.system() == "Darwin", msg
            file_path_within_package = ".dylibs/"
            file_extension = ".dylib"

        package_name = "scipy"
        scipy_package = importlib.util.find_spec(package_name)
        package_directory = path.dirname(scipy_package.origin)
        scipy_lib_path = path.join(package_directory, file_path_within_package)

        file_prefix = "libopenblas"
        search_pattern = path.join(scipy_lib_path, f"{file_prefix}*{file_extension}")
        search_result = glob.glob(search_pattern)
        if not search_result:
            raise CompileError(
                f'Unable to find OpenBLAS library at "{search_pattern}". '
                "Please ensure that SciPy is installed and available via pip."
            )
        openblas_so_file = search_result[0]
        openblas_lib_name = path.basename(openblas_so_file)[3 : -len(file_extension)]

        lib_path_flags += [
            f"-Wl,-rpath,{scipy_lib_path}",
            f"-L{scipy_lib_path}",
        ]

        system_flags = []
        if platform.system() == "Linux":
            # --disable-new-dtags makes the linker use RPATH instead of RUNPATH.
            # RPATH influences search paths globally while RUNPATH only works for
            # a single file, but not its dependencies.
            system_flags += ["-Wl,-no-as-needed", "-Wl,--disable-new-dtags"]
        elif platform.system() == "Darwin":  # pragma: nocover
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
            f"-l{openblas_lib_name}",  # required for custom_calls lib
            "-lcustom_calls",
            "-lmlir_async_runtime",
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


class Compiler:
    """Compiles MLIR modules to shared objects by executing the Catalyst compiler driver library."""

    @debug_logger_init
    def __init__(self, options: Optional[CompileOptions] = None):
        self.options = options if options is not None else CompileOptions()

    @debug_logger
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
                async_qnodes=self.options.async_qnodes,
                verbose=self.options.verbose,
                pipelines=self.options.get_pipelines(),
                lower_to_llvm=lower_to_llvm,
                checkpoint_stage=self.options.checkpoint_stage,
            )
        except RuntimeError as e:
            raise CompileError(*e.args) from e

        if self.options.verbose:
            for line in compiler_output.get_diagnostic_messages().strip().split("\n"):
                print(f"[LIB] {line}", file=self.options.logfile)

        filename = compiler_output.get_object_filename()
        out_IR = compiler_output.get_output_ir()

        if lower_to_llvm:
            output = LinkerDriver.run(filename, options=self.options)
            output_filename = str(pathlib.Path(output).absolute())
        else:
            output_filename = filename

        return output_filename, out_IR

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
