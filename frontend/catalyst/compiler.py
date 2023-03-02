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
import shutil
import subprocess
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


translate_tool = get_executable_path("llvm", "mlir-translate")
mhlo_opt_tool = get_executable_path("mhlo", "mlir-hlo-opt")
quantum_opt_tool = get_executable_path("quantum", "quantum-opt")

mhlo_lowering_pass_pipeline = [
    "--canonicalize",
    "--chlo-legalize-to-hlo",
    "--mhlo-legalize-control-flow",
    "--hlo-legalize-to-linalg",
    "--mhlo-legalize-to-std",
    "--convert-to-signless",
    "--canonicalize",
]

bufferization_pass_pipeline = [
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

llvm_lowering_pass_pipeline = [
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

compiler = get_executable_path("llvm", "llc")
compiler_flags = [
    "--filetype=obj",
    "--relocation-model=pic",
]


# pylint: disable=too-few-public-methods
class AbstractLinker:
    """Linker interface

    In order to avoid relying on a single linker at run time and allow the user some flexibility,
    this class defines a linker resolution order where multiple known linkers are attempted. The
    order is defined as follows:

    1. A user specified linker via the environment variable CATALYST_CC. It is expected that the
        user provided linker is flag compatilble with gcc.
    2. clang: May be configured to use LLD or LD. Both of which are flag compatible. Priority is
        given to clang to maintain an LLVM toolchain through all the process.
    3. gcc: Usually configured to link with LD.
    4. c99: Usually defaults to gcc, but no linker interface is specified.
    5. c89: Usually defaults to gcc, but no linker interface is specified.
    6. cc: Usually defaults to gcc, however POSIX states that it is deprecated.
    """

    _default_fallback_linkers = ["clang", "gcc", "c99", "c89", "cc"]

    @staticmethod
    def _flags():
        mlir_lib_path = get_lib_path("llvm", "MLIR_LIB_DIR")
        lrt_lib_path = get_lib_path("runtime", "RUNTIME_LIB_DIR")
        lrt_capi_path = os.path.join(lrt_lib_path, "capi")
        lrt_backend_path = os.path.join(lrt_lib_path, "backend")

        linker_flags = [
            "-Wno-unused-command-line-argument",
            "-Wno-override-module",
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

        return linker_flags

    @staticmethod
    def _lro(fallback_linkers):
        """Linker resolution order"""
        preferred_linker = os.environ.get("CATALYST_CC", None)
        preferred_linker_exists = AbstractLinker._exists(preferred_linker)
        linkers = fallback_linkers
        emit_warning = preferred_linker and not preferred_linker_exists
        if emit_warning:
            msg = f"User defined linker {preferred_linker} is not in PATH. Will attempt fallback on available linkers."
            warnings.warn(msg, UserWarning)
        else:
            linkers = [preferred_linker] + fallback_linkers
        return linkers

    @staticmethod
    def _exists(linker):
        if not linker:
            return None
        return shutil.which(linker)

    @staticmethod
    def _available_linkers(fallback_linkers):
        available_linkers = []
        for linker in AbstractLinker._lro(fallback_linkers):
            if AbstractLinker._exists(linker):
                available_linkers.append(linker)
        return available_linkers

    @staticmethod
    def _attempt_link(linker, flags, infile, outfile):
        try:
            command = [linker] + flags + [infile, "-o", outfile]
            subprocess.run(command, check=True)
            return True
        except subprocess.CalledProcessError:
            msg = f"Linker {linker} failed during execution of command {command}. Will attempt fallback on available linkers."
            warnings.warn(msg, UserWarning)
            return False

    @staticmethod
    def link(infile, outfile, fallback_linkers=None):
        """
        Link the infile against the necessary libraries and produce the outfile.

        Args:
            infile (str): input file
            outfile (str): output file
        Raises:
            EnvironmentError: The exception is raised when no linker succeeded.
        """
        if not fallback_linkers:
            fallback_linkers = AbstractLinker._default_fallback_linkers
        for linker in AbstractLinker._available_linkers(fallback_linkers):
            flags = AbstractLinker._flags()
            success = AbstractLinker._attempt_link(linker, flags, infile, outfile)
            if success:
                return
        msg = f"Unable to link {infile}. All available linker options exhausted. Please provide an available linker via $CATALYST_CC."
        raise EnvironmentError(msg)


def lower_mhlo_to_linalg(filename):
    """Translate MHLO to linalg dialect.

    Args:
        filename (str): the path to a file were the program is stored.
    Returns:
        a path to the output file
    """
    assert filename[-5:] == ".mlir", "input is not an mlir file"

    command = [mhlo_opt_tool]
    command += ["--allow-unregistered-dialect"]
    command += [filename]
    command += mhlo_lowering_pass_pipeline

    new_fname = filename.replace(".mlir", ".nohlo.mlir")

    with open(new_fname, "w", encoding="utf-8") as file:
        subprocess.run(command, stdout=file, check=True)

    return new_fname


def bufferize_tensors(filename):
    """Translate MHLO to linalg dialect.

    Args:
        filename (str): the path to a file were the program is stored.
    Returns:
        a path to the output file
    """
    assert filename[-5:] == ".mlir", "input is not an mlir file"

    command = [quantum_opt_tool]
    command += [filename]
    command += bufferization_pass_pipeline

    new_fname = filename.replace(".mlir", ".buff.mlir")

    with open(new_fname, "w", encoding="utf-8") as file:
        subprocess.run(command, stdout=file, check=True)

    return new_fname


def lower_all_to_llvm(filename):
    """Translate MLIR dialects to LLVM dialect.

    Args:
        filename (str): the path to a file were the program is stored.
    Returns:
        a path to the output file
    """
    assert filename[-10:] == ".buff.mlir", "input is not a bufferized mlir file"

    command = [quantum_opt_tool]
    command += [filename]
    command += llvm_lowering_pass_pipeline

    new_fname = filename.replace(".buff.mlir", ".llvm.mlir")
    with open(new_fname, "w", encoding="utf-8") as file:
        subprocess.run(command, stdout=file, check=True)

    return new_fname


def convert_mlir_to_llvmir(filename):
    """Translate LLVM dialect to LLVM IR.

    Args:
        filename (str): the path to a file were the program is stored.
    Returns:
        a path to the output file
    """
    assert filename[-10:] == ".llvm.mlir", "input is not an llvm dialect mlir file"

    command = [translate_tool]
    command += [filename]
    command += ["--mlir-to-llvmir"]

    new_fname = filename.replace(".llvm.mlir", ".ll")
    with open(new_fname, "w", encoding="utf-8") as file:
        subprocess.run(command, stdout=file, check=True)

    return new_fname


def compile_llvmir(filename):
    """Translate LLVM IR to an object file.

    Args:
        filename (str): the path to a file were the program is stored.
    Returns:
        a path to the output file
    """
    assert filename[-3:] == ".ll", "input is not an llvmir file"

    new_fname = filename.replace(".ll", ".o")

    command = [compiler]
    command += compiler_flags
    command += [filename]
    command += ["-o", new_fname]
    subprocess.run(command, check=True)
    return new_fname


def link_lightning_runtime(filename):
    """Link the object file as a shared object.

    Args:
        filename (str): the path to a file were the object file is stored.
    Returns:
        a path to the output file
    """
    assert filename[-2:] == ".o", "input is not an object file"

    new_fname = filename.replace(".o", ".so")

    AbstractLinker.link(filename, new_fname)

    return new_fname


def compile(mlir_module, workspace, passes):
    """Compile an MLIR module to a shared object.

    .. note::

        For compilation of hybrid quantum-classical PennyLane programs,
        please see the :func:`~.qjit` decorator.

    Args:
        mlir_module (Module): the MLIR module
        workspace (str): the absolute path to the MLIR module
        has_hlo (bool): ``True`` if the MLIR module contains HLO code. Defaults to ``False``
        passes (List[str]): the list of compilation passes

    Returns:
        Shared object
        A string representation of LLVM IR.
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

    passes["mlir"] = filename
    mlir = filename
    mlir = lower_mhlo_to_linalg(mlir)
    passes["nohlo"] = mlir
    buff = bufferize_tensors(mlir)
    passes["buff"] = buff
    llvm_dialect = lower_all_to_llvm(buff)
    passes["llvm"] = llvm_dialect
    llvmir = convert_mlir_to_llvmir(llvm_dialect)
    passes["ll"] = llvmir
    object_file = compile_llvmir(llvmir)
    shared_object = link_lightning_runtime(object_file)

    with open(llvmir, "r", encoding="utf-8") as f:
        _llvmir = f.read()

    return shared_object, _llvmir
