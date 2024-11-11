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

# pylint: disable=undefined-variable
from os import path

import lit.formats
from lit.llvm import llvm_config

config.name = "Frontend Tests"
config.test_format = lit.formats.ShTest(True)

# Define the file extensions to treat as test files (with the exception of this file).
config.suffixes = [".py"]
config.excludes = ["lit.cfg.py", "lit_util_printers.py"]

# Define the root path of where to look for tests.
config.test_source_root = path.dirname(__file__)

# Define where to execute tests (and produce the output).
config.test_exec_root = getattr(config, "frontend_test_dir", ".lit")

# TODO: Ideally we would test with leak detection, but this may not be possible from Python.
# TODO: Find out why we have container overflow on macOS.
config.environment["ASAN_OPTIONS"] = "detect_leaks=0,detect_container_overflow=0"

# Define substitutions used at the top of lit test files, e.g. %PYTHON.
python_executable = getattr(config, "python_executable", "python3.10")

if "Address" in getattr(config, "llvm_use_sanitizer", ""):
    # With sanitized builds, Python tests require some preloading magic to run.
    if "Linux" in config.host_os:
        python_executable = f"LD_PRELOAD=$({config.host_cxx} -print-file-name=libclang_rt.asan-{config.host_arch}.so) {python_executable}"
    elif "Darwin" in config.host_os:
        # It's important that the python executable discovered by CMake is the true interpreter
        # binary. On macOS, it appears that there can be multiple "python" executables which are
        # just wrappers around the actual interpreter. In that case, the DYLD_INSERT_LIBRARIES trick
        # will not work, as it will not be forwarded to the final process.
        # See https://stackoverflow.com/a/47853433 for more information.
        python_executable = f"DYLD_INSERT_LIBRARIES=$({config.host_cxx} -print-file-name=libclang_rt.asan_osx_dynamic.dylib) {python_executable}"
    else:
        assert False, "Testing with sanitized builds requires Linux or MacOS"

config.substitutions.append(("%PYTHON", python_executable))

# Define PATH when running frontend tests from an mlir build target.
try:
    # Access to FileCheck
    llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
    # Access to quantum-opt
    llvm_config.with_environment("PATH", config.quantum_bin_dir, append_path=True)

    # Define the location of runtime libraries when running frontend tests
    llvm_config.with_environment("RUNTIME_LIB_DIR", config.lrt_lib_dir, append_path=True)
    llvm_config.with_environment("MLIR_LIB_DIR", config.mlir_lib_dir, append_path=True)
    llvm_config.with_environment("ENZYME_LIB_DIR", config.enzyme_lib_dir, append_path=True)
    llvm_config.with_environment("DIALECTS_BUILD_DIR", config.mlir_lib_dir + "/..", append_path=True)

    # Define PYTHONPATH to include the dialect python bindings.
    # From within a build target we have access to cmake variables configured in lit.site.cfg.py.in.
    llvm_config.with_environment(
        "PYTHONPATH",
        [config.mlir_bindings_dir],
        append_path=True,
    )
except AttributeError:
    from lit.llvm.config import LLVMConfig  # fmt:skip
    llvm_config = LLVMConfig(lit_config, config)
    llvm_config.with_system_environment("PYTHONPATH")
    llvm_config.with_system_environment("RUNTIME_LIB_DIR")
    llvm_config.with_system_environment("MLIR_LIB_DIR")
    llvm_config.with_system_environment("DIALECTS_BUILD_DIR")
    llvm_config.with_system_environment("ENZYME_LIB_DIR")
