PYTHON?=$(shell which python3)
C_COMPILER?=$(shell which clang)
CXX_COMPILER?=$(shell which clang++)
COMPILER_LAUNCHER?=$(shell which ccache)

MK_ABSPATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MK_DIR := $(dir $(MK_ABSPATH))
DIALECTS_BUILD_DIR?=$(MK_DIR)/build
LLVM_BUILD_DIR?=$(MK_DIR)/llvm-project/build
MHLO_BUILD_DIR?=$(MK_DIR)/mlir-hlo/bazel-build
ENZYME_BUILD_DIR?=$(MK_DIR)/Enzyme/build
RT_BUILD_DIR?=$(MK_DIR)/../runtime/build
ENABLE_ASAN?=OFF
BUILD_TYPE?=Release
TARGET_FILE=$(MK_DIR)/mlir-hlo/mhlo/transforms/CMakeLists.txt
PATCH_FILE=$(MK_DIR)/patches/mhlo-Add-PassesIncGen-in-transforms-CMakeList.patch

ifeq ($(shell uname), Darwin)
DEFAULT_ENABLE_LLD := OFF
SYMBOL_VISIBILITY := hidden
else
DEFAULT_ENABLE_LLD := ON
SYMBOL_VISIBILITY := protected
endif

ENABLE_LLD?=$(DEFAULT_ENABLE_LLD)
ENABLE_ZLIB?=OFF
ENABLE_ZSTD?=ON

ifeq ($(ENABLE_ASAN), ON)
USE_SANITIZER_NAMES="Address"
USE_SANITIZER_FLAGS="-fsanitize=address"
else
USE_SANITIZER_NAMES=""
USE_SANITIZER_FLAGS=""
endif

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  all                to build MLIR, MLIR-HLO and custom Catalyst dialects"
	@echo "  llvm               to build MLIR enabling Python bindings"
	@echo "  mhlo               to build MLIR-HLO"
	@echo "  enzyme             to build Enzyme"
	@echo "  dialects           to build custom Catalyst MLIR dialects"
	@echo "  test               to run the Catalyst MLIR dialects test suite"
	@echo "  clean              to delete all temporary, cache, and build files"
	@echo "  format [check=1]   to apply C++ formatter; use with 'check=1' to check instead of modify (requires clang-format)"
	@echo "  format [version=?] to apply C++ formatter; use with 'version={version}' to run clang-format-{version} instead of clang-format"

.PHONY: all
all: llvm mhlo enzyme dialects

.PHONY: llvm
llvm:
	@echo "build LLVM and MLIR enabling Python bindings"
	cmake -G Ninja -S llvm-project/llvm -B $(LLVM_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DLLVM_BUILD_EXAMPLES=OFF \
		-DLLVM_TARGETS_TO_BUILD="host" \
		-DLLVM_ENABLE_PROJECTS="mlir" \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DMLIR_ENABLE_BINDINGS_PYTHON=ON \
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
	# and remove filter
	LIT_FILTER_OUT="Bytecode" cmake --build $(LLVM_BUILD_DIR) --target check-mlir llvm-symbolizer

.PHONY: mhlo
mhlo:
	@echo "build MLIR-HLO"
	# Patch MHLO cmake dependency
	# TODO: remove once https://github.com/openxla/xla/pull/15446 is merged
	@if patch --dry-run -p1 -N $(TARGET_FILE) $(PATCH_FILE) > /dev/null 2>&1; then \
		patch -p1 $(TARGET_FILE) $(PATCH_FILE); \
	fi
	cmake -G Ninja -S mlir-hlo -B $(MHLO_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DMLIR_DIR=$(LLVM_BUILD_DIR)/lib/cmake/mlir \
		-DPython3_EXECUTABLE=$(PYTHON) \
		-DCMAKE_C_COMPILER=$(C_COMPILER) \
		-DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
		-DCMAKE_C_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DCMAKE_CXX_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DCMAKE_EXE_LINKER_FLAGS=$(USE_SANITIZER_FLAGS) \
		-DLLVM_ENABLE_LLD=$(ENABLE_LLD) \
		-DLLVM_ENABLE_ZLIB=$(ENABLE_ZLIB) \
		-DLLVM_ENABLE_ZSTD=$(ENABLE_ZSTD) \
		-DCMAKE_CXX_VISIBILITY_PRESET=$(SYMBOL_VISIBILITY)

        # TODO: figure out why this test is failing
	LIT_FILTER_OUT="chlo_legalize_to_mhlo" cmake --build $(MHLO_BUILD_DIR) --target check-mlir-hlo

.PHONY: enzyme
enzyme:
	@echo "build enzyme"
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

	cmake --build $(ENZYME_BUILD_DIR) --target EnzymeStatic-19

.PHONY: dialects
dialects:

	@echo "build quantum-lsp compiler_driver and dialects"
	cmake -G Ninja -S . -B $(DIALECTS_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DQUANTUM_ENABLE_BINDINGS_PYTHON=ON \
		-DPython3_EXECUTABLE=$(PYTHON) \
		-DPython3_NumPy_INCLUDE_DIRS=$(shell $(PYTHON) -c "import numpy as np; print(np.get_include())") \
		-DEnzyme_DIR=$(ENZYME_BUILD_DIR) \
		-DENZYME_SRC_DIR=$(MK_DIR)/Enzyme \
		-DMLIR_DIR=$(LLVM_BUILD_DIR)/lib/cmake/mlir \
		-DMHLO_DIR=$(MHLO_BUILD_DIR)/lib/cmake/mlir-hlo \
		-DMHLO_BINARY_DIR=$(MHLO_BUILD_DIR)/bin \
		-DRUNTIME_LIB_DIR=$(RT_BUILD_DIR)/lib \
		-DMLIR_LIB_DIR=$(LLVM_BUILD_DIR)/lib \
		-DCMAKE_C_COMPILER=$(C_COMPILER) \
		-DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
		-DCMAKE_C_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DCMAKE_CXX_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DLLVM_USE_SANITIZER=$(USE_SANITIZER_NAMES) \
		-DLLVM_ENABLE_LLD=$(ENABLE_LLD) \
		-DLLVM_ENABLE_ZLIB=$(ENABLE_ZLIB) \
		-DLLVM_ENABLE_ZSTD=$(ENABLE_ZSTD)

	cmake --build $(DIALECTS_BUILD_DIR) --target check-dialects quantum-lsp-server compiler_driver catalyst-cli

.PHONY: test
test:
	@echo "test the Catalyst MLIR dialects test suite"
	cmake --build $(DIALECTS_BUILD_DIR) --target check-dialects

.PHONY: clean clean-dialects clean-enzyme clean-mhlo
clean: clean-dialects clean-llvm clean-mhlo clean-enzyme

clean-dialects:
	@echo "clean catalyst dialect build files"
	rm -rf $(DIALECTS_BUILD_DIR)

clean-llvm:
	@echo "clean llvm/mlir build files"
	rm -rf $(LLVM_BUILD_DIR)

clean-mhlo:
	@echo "clean HLO dialect build files"
	rm -rf $(MHLO_BUILD_DIR)

clean-enzyme:
	@echo "clean enzyme build files"
	rm -rf $(ENZYME_BUILD_DIR)

.PHONY: format
format:
ifdef check
	$(PYTHON) ../bin/format.py --check $(if $(version:-=),--cfversion $(version)) .
else
	$(PYTHON) ../bin/format.py $(if $(version:-=),--cfversion $(version)) .
endif
