PYTHON ?= $(shell which python3)
C_COMPILER ?= $(shell which clang)
CXX_COMPILER ?= $(shell which clang++)
COMPILER_LAUNCHER ?= $(shell which ccache)

MK_ABSPATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MK_DIR := $(dir $(MK_ABSPATH))
DIALECTS_BUILD_DIR ?= $(MK_DIR)/build
DIALECTS_DOCS_BUILD_DIR ?= $(MK_DIR)/build-docs
LLVM_BUILD_DIR ?= $(MK_DIR)/llvm-project/build
STABLEHLO_BUILD_DIR ?= $(MK_DIR)/stablehlo/build
ENZYME_BUILD_DIR ?= $(MK_DIR)/Enzyme/build
RT_BUILD_DIR ?= $(MK_DIR)/../runtime/build
ENABLE_ASAN ?= OFF
STRICT_WARNINGS ?= ON
BUILD_TYPE ?= Release
LLVM_EXTERNAL_LIT ?= $(LLVM_BUILD_DIR)/bin/llvm-lit

ifeq ($(shell uname), Darwin)
DEFAULT_ENABLE_LLD := OFF
SYMBOL_VISIBILITY := default
else
DEFAULT_ENABLE_LLD := ON
SYMBOL_VISIBILITY := default
endif

ENABLE_LLD ?= $(DEFAULT_ENABLE_LLD)
ENABLE_ZLIB ?= ON
ENABLE_ZSTD ?= OFF

ifeq ($(ENABLE_ASAN), ON)
USE_SANITIZER_NAMES="Address"
USE_SANITIZER_FLAGS="-fsanitize=address"
else
USE_SANITIZER_NAMES=""
USE_SANITIZER_FLAGS=""
endif

LLVM_PROJECTS ?= mlir
LLVM_TARGETS ?= check-mlir llvm-symbolizer

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  all                to build MLIR, Stablehlo and custom Catalyst dialects"
	@echo "  llvm               to build MLIR enabling Python bindings"
	@echo "  stablehlo          to build stablehlo"
	@echo "  enzyme             to build Enzyme"
	@echo "  dialects           to build custom Catalyst MLIR dialects"
	@echo "  dialect-docs       to build custom Catalyst MLIR dialect documentation"
	@echo "  test               to run the Catalyst MLIR dialects test suite"
	@echo "  clean              to delete all temporary, cache, and build files"
	@echo "  format [check=1]   to apply C++ formatter; use with 'check=1' to check instead of modify (requires clang-format)"
	@echo "  format [version=?] to apply C++ formatter; use with 'version={version}' to run clang-format-{version} instead of clang-format"

.PHONY: all
all: llvm stablehlo enzyme dialects dialect-docs plugin

.PHONY: llvm
llvm:
	@echo "build LLVM and MLIR enabling Python bindings"

	# Patch mlir one shot bufferization segfault
	# Remove patch after bug is resolved upstream
	# https://github.com/llvm/llvm-project/issues/150441
	@if cd llvm-project; git apply --check $(MK_DIR)/patches/llvm-bufferization-segfault.patch; then \
		git apply $(MK_DIR)/patches/llvm-bufferization-segfault.patch; \
	fi
	# Patch mlir python bindinggen annotations
	# Remove patch after JAX is updated to v0.8.2 or later
	@if cd llvm-project; git apply --check $(MK_DIR)/patches/llvm-python-bindinggen-annotations.patch; then \
		git apply $(MK_DIR)/patches/llvm-python-bindinggen-annotations.patch; \
	fi

	cmake -G Ninja -S llvm-project/llvm -B $(LLVM_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DLLVM_BUILD_EXAMPLES=OFF \
		-DLLVM_TARGETS_TO_BUILD="host;X86;AArch64;AMDGPU;NVPTX;WebAssembly" \
		-DLLVM_ENABLE_PROJECTS="$(LLVM_PROJECTS)" \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DMLIR_ENABLE_BINDINGS_PYTHON=ON \
		-DPython_EXECUTABLE=$(PYTHON) \
		-DPython3_EXECUTABLE=$(PYTHON) \
		-DPython3_NumPy_INCLUDE_DIRS=$$($(PYTHON) -c "import numpy as np; print(np.get_include())") \
		-DCMAKE_C_COMPILER=$(C_COMPILER) \
		-DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
		-DCMAKE_C_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DCMAKE_CXX_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DLLVM_USE_SANITIZER=$(USE_SANITIZER_NAMES) \
		-DLLVM_ENABLE_LLD=$(ENABLE_LLD) \
		-DLLVM_ENABLE_ZLIB=$(ENABLE_ZLIB) \
		-DLLVM_ENABLE_ZSTD=$(ENABLE_ZSTD) \
		-DCMAKE_CXX_VISIBILITY_PRESET=$(SYMBOL_VISIBILITY)

	# TODO: when updating LLVM, test to see if mlir/unittests/Bytecode/BytecodeTest.cpp:55 is passing
	# and remove filter. This tests fails on CI/CD not locally.
	# Note: the upstream lit test llvm-project/mlir/test/python/execution_engine.py requries
	# the python package `ml_dtypes`. We don't actually use the execution engine, so we skip the
	# test to reduce unnecessary dependencies.
	# Note: python_test.py fails with Python 3.14 due to type hint representation changes
	LIT_FILTER_OUT="Bytecode|tosa-to-tensor|execution_engine|python_test" cmake --build $(LLVM_BUILD_DIR) --target $(LLVM_TARGETS)

.PHONY: stablehlo
stablehlo:
	@echo "build stablehlo"

	cmake -G Ninja -S stablehlo -B $(STABLEHLO_BUILD_DIR) \
		-DSTABLEHLO_ENABLE_LLD=$(ENABLE_LLD) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DMLIR_DIR=$(LLVM_BUILD_DIR)/lib/cmake/mlir \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DLLVM_ENABLE_LLD=$(ENABLE_LLD) \
		-DLLVM_ENABLE_ZLIB=$(ENABLE_ZLIB) \
		-DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF \
		-DSTABLEHLO_ENABLE_SPLIT_DWARF=ON \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DCMAKE_C_COMPILER=$(C_COMPILER) \
		-DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
		-DCMAKE_C_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DCMAKE_CXX_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DCMAKE_EXE_LINKER_FLAGS=$(USE_SANITIZER_FLAGS) \
		-DCMAKE_CXX_VISIBILITY_PRESET=$(SYMBOL_VISIBILITY)

	cmake --build $(STABLEHLO_BUILD_DIR)

.PHONY: enzyme
enzyme: TARGET_FILE := $(MK_DIR)/Enzyme/enzyme/Enzyme/TypeAnalysis/TypeAnalysis.cpp
enzyme: PATCH_FILE := $(MK_DIR)/patches/enzyme-nvvm-fabs-intrinsics.patch
enzyme:
	@echo "build enzyme"
	# Patch enzyme's dependency on nvidia fabs llvm intrinsics
	@if cd Enzyme; git apply --check $(MK_DIR)/patches/enzyme-nvvm-fabs-intrinsics.patch; then \
		git apply $(MK_DIR)/patches/enzyme-nvvm-fabs-intrinsics.patch; \
	fi
	cmake -G Ninja -S Enzyme/enzyme -B $(ENZYME_BUILD_DIR) \
		-DENZYME_STATIC_LIB=ON \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DLLVM_DIR=$(LLVM_BUILD_DIR)/lib/cmake/llvm \
		-DCMAKE_C_COMPILER=$(C_COMPILER) \
		-DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
		-DCMAKE_C_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DCMAKE_CXX_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DCMAKE_EXE_LINKER_FLAGS=$(USE_SANITIZER_FLAGS) \
		-DCMAKE_CXX_VISIBILITY_PRESET=$(SYMBOL_VISIBILITY) \
		-DCMAKE_POLICY_DEFAULT_CMP0116=NEW

	cmake --build $(ENZYME_BUILD_DIR) --target EnzymeStatic-22

.PHONY: plugin
plugin:
	[ -f $(MK_DIR)/standalone ] || cp -r $(MK_DIR)/llvm-project/mlir/examples/standalone .
	@if patch -p0 --dry-run -N < $(MK_DIR)/patches/test-plugin-with-catalyst.patch > /dev/null 2>&1; then \
		patch -p0 < $(MK_DIR)/patches/test-plugin-with-catalyst.patch; \
	fi
	cmake -B standalone/build -G Ninja \
		-DCMAKE_C_COMPILER=$(C_COMPILER) \
		-DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
		-DCMAKE_C_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DCMAKE_CXX_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DMLIR_DIR=$(LLVM_BUILD_DIR)/lib/cmake/mlir \
		-DLLVM_EXTERNAL_LIT=$(LLVM_EXTERNAL_LIT) \
		-DCATALYST_TOOLS_DIR=$(DIALECTS_BUILD_DIR)/bin \
		-DPython_EXECUTABLE=$(PYTHON) \
		-DPython3_EXECUTABLE=$(PYTHON) \
		-DPython3_NumPy_INCLUDE_DIRS=$$($(PYTHON) -c "import numpy as np; print(np.get_include())") \
		standalone
	cmake --build standalone/build --target check-standalone
	mkdir -p $(DIALECTS_BUILD_DIR)/lib
	cp standalone/build/lib/StandalonePlugin.* $(DIALECTS_BUILD_DIR)/lib

.PHONY: dialects
dialects:

	@echo "build quantum-lsp and dialects"
	cmake -G Ninja -S . -B $(DIALECTS_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DQUANTUM_ENABLE_BINDINGS_PYTHON=ON \
		-DPython3_EXECUTABLE=$(PYTHON) \
		-DPython3_NumPy_INCLUDE_DIRS=$(shell $(PYTHON) -c "import numpy as np; print(np.get_include())") \
		-DEnzyme_DIR=$(ENZYME_BUILD_DIR) \
		-DENZYME_SRC_DIR=$(MK_DIR)/Enzyme \
		-DMLIR_DIR=$(LLVM_BUILD_DIR)/lib/cmake/mlir \
		-DSTABLEHLO_DIR=$(MK_DIR)/stablehlo \
		-DSTABLEHLO_BUILD_DIR=$(STABLEHLO_BUILD_DIR) \
		-DRUNTIME_LIB_DIR=$(RT_BUILD_DIR)/lib \
		-DMLIR_LIB_DIR=$(LLVM_BUILD_DIR)/lib \
		-DCMAKE_C_COMPILER=$(C_COMPILER) \
		-DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
		-DCMAKE_C_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DCMAKE_CXX_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DLLVM_USE_SANITIZER=$(USE_SANITIZER_NAMES) \
		-DLLVM_ENABLE_LLD=$(ENABLE_LLD) \
		-DLLVM_ENABLE_ZLIB=$(ENABLE_ZLIB) \
		-DLLVM_ENABLE_ZSTD=$(ENABLE_ZSTD) \
		-DCATALYST_ENABLE_WARNINGS=$(STRICT_WARNINGS)

	cmake --build $(DIALECTS_BUILD_DIR) --target check-dialects quantum-lsp-server catalyst-cli check-unit-tests

.PHONY: dialect-docs
dialect-docs:

	@echo "build dialect documentation"
	cmake -G Ninja -S . -B $(DIALECTS_DOCS_BUILD_DIR) \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DQUANTUM_ENABLE_BINDINGS_PYTHON=ON \
		-DPython3_EXECUTABLE=$(PYTHON) \
		-DPython3_NumPy_INCLUDE_DIRS=$(shell $(PYTHON) -c "import numpy as np; print(np.get_include())") \
		-DEnzyme_DIR=$(ENZYME_BUILD_DIR) \
		-DENZYME_SRC_DIR=$(MK_DIR)/Enzyme \
		-DMLIR_DIR=$(LLVM_BUILD_DIR)/lib/cmake/mlir \
		-DRUNTIME_LIB_DIR=$(RT_BUILD_DIR)/lib \
		-DMLIR_LIB_DIR=$(LLVM_BUILD_DIR)/lib \
		-DCATALYST_DOCS_ONLY=ON

	cmake --build $(DIALECTS_DOCS_BUILD_DIR) --target mlir-doc

.PHONY: test
test:
	@echo "test the Catalyst MLIR dialects test suite"
	cmake --build $(DIALECTS_BUILD_DIR) --target check-dialects

.PHONY: clean clean-dialects clean-enzyme clean-stablehlo clean-plugin clean-dialect-docs
clean: clean-dialects clean-llvm clean-stablehlo clean-enzyme clean-plugin clean-dialect-docs

clean-dialects:
	@echo "clean catalyst dialect build files"
	rm -rf $(DIALECTS_BUILD_DIR)

clean-llvm:
	@echo "clean llvm/mlir build files"
	rm -rf $(LLVM_BUILD_DIR)
	cd llvm-project; git clean -fd; git checkout .

clean-stablehlo:
	@echo "clean Stablehlo dialect build files"
	rm -rf $(STABLEHLO_BUILD_DIR)
	cd stablehlo; git clean -fd; git checkout .

reset-llvm:
	@echo "reset llvm git state to the commit tracked in .dep-versions without deleting llvm builds"
	cd llvm-project; git clean -fd; git checkout .

clean-enzyme:
	@echo "clean enzyme build files"
	rm -rf $(ENZYME_BUILD_DIR)
	cd Enzyme; git clean -fd; git checkout .

clean-plugin:
	@echo "clean plugin"
	rm -rf standalone
	rm -rf standalone/build
	rm -rf $(DIALECTS_BUILD_DIR)/lib/StandalonePlugin.*

clean-dialect-docs:
	@echo "clean dialect docs"
	rm -rf $(DIALECTS_DOCS_BUILD_DIR)

.PHONY: format
format:
ifdef check
	$(PYTHON) ../bin/format.py --check $(if $(version:-=),--cfversion $(version)) .
else
	$(PYTHON) ../bin/format.py $(if $(version:-=),--cfversion $(version)) .
endif
