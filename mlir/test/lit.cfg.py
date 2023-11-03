import os

import lit.formats
from lit.llvm import llvm_config

config.name = "Compiler test suite"
config.test_format = lit.formats.ShTest(True)

# Define the file extensions to treat as test files.
config.suffixes = [".mlir", ".py"]
config.excludes = ["lit.cfg.py"]

# Define the root path of where to look for tests.
config.test_source_root = os.path.dirname(__file__)

# Define where to execute tests (and produce the output).
config.test_exec_root = getattr(config, "quantum_test_dir", ".lit")

# Define substitutions used at the top of lit test files, e.g. %PYTHON.
python_executable = getattr(config, "python_executable", "python3.10")

if "Address" in getattr(config, "llvm_use_sanitizer", ""):
    # With sanitized builds, Python tests require some preloading magic to run.
    assert "Linux" in config.host_os, "Testing with sanitized builds requires Linux/Clang"

    python_executable = f"LD_PRELOAD=$({config.host_cxx} -print-file-name=libclang_rt.asan-{config.host_arch}.so) {python_executable}"

config.substitutions.append(("%PYTHON", python_executable))

# Define PATH to include the various tools needed for our tests.
try:
    # From within a build target we have access to cmake variables configured in lit.site.cfg.py.in.
    llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)  # FileCheck
    llvm_config.with_environment("PATH", config.quantum_bin_dir, append_path=True)  # quantum-opt
except AttributeError as e:
    # The system PATH is available by default.
    pass

# Define PYTHONPATH to include the dialect python bindings.
try:
    # From within a build target we have access to cmake variables configured in lit.site.cfg.py.in.
    llvm_config.with_environment(
        "PYTHONPATH",
        [
            os.path.join(config.quantum_build_dir, "python_packages", "quantum"),
            os.path.join(config.quantum_build_dir, "python_packages", "gradient"),
            os.path.join(config.quantum_build_dir, "python_packages", "catalyst"),
        ],
        append_path=True,
    )
except AttributeError:
    # Else we use the system environment.
    from lit.llvm.config import LLVMConfig  # fmt:skip
    LLVMConfig(lit_config, config).with_system_environment("PYTHONPATH")
