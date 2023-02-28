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
import subprocess

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

linker = "c99"

mlir_lib_path = get_lib_path("llvm", "MLIR_LIB_DIR")
linker_common_flags = [
    "-shared",
    "-rdynamic",
    f"-L{mlir_lib_path}",
    "-Wl,-no-as-needed",
    f"-Wl,-rpath,{mlir_lib_path}",
]

common_libs = [
    "-lpthread",
    "-lmlir_c_runner_utils",  # required for memref.copy
]

runtime_libs = [
    "-lrt_backend",
    "-lrt_capi",
    *common_libs,
]

lrt_lib_path = get_lib_path("runtime", "RUNTIME_LIB_DIR")
lrt_capi_path = os.path.join(lrt_lib_path, "capi")
lrt_backend_path = os.path.join(lrt_lib_path, "backend")
lightning_linker_flags = [
    *linker_common_flags,
    f"-L{lrt_capi_path}",
    f"-L{lrt_backend_path}",
    f"-Wl,-rpath,{lrt_capi_path}:{lrt_backend_path}",
]


def lower_mhlo_to_linalg(filename):
    """Translate MHLO to linalg dialect.

    Args:
        filename: the path to a file were the program is stored
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
        filename: the path to a file were the program is stored
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
        filename: the path to a file were the program is stored
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
        filename: the path to a file were the program is stored
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
        filename: the path to a file were the program is stored
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
        filename: the path to a file were the object file is stored
    Returns:
        a path to the output file
    """
    assert filename[-2:] == ".o", "input is not an object file"

    new_fname = filename.replace(".o", ".so")

    command = [linker]
    command += ["-Wno-unused-command-line-argument", "-Wno-override-module"]
    command += lightning_linker_flags
    command += runtime_libs
    command += [filename]
    command += ["-o", new_fname]

    subprocess.run(command, check=True)
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
