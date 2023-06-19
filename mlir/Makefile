
PYTHON?=python3
C_COMPILER?=clang
CXX_COMPILER?=clang++
COMPILER_LAUNCHER?=ccache
MK_ABSPATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MK_DIR := $(dir $(MK_ABSPATH))
DIALECTS_BUILD_DIR?=$(MK_DIR)/build
LLVM_BUILD_DIR?=$(MK_DIR)/llvm-project/build
MHLO_BUILD_DIR?=$(MK_DIR)/mlir-hlo/build
RT_BUILD_DIR?=$(MK_DIR)/../runtime/build
ENABLE_LLD?=ON

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  all                to build MLIR, MLIR-HLO and custom Catalyst dialects"
	@echo "  llvm               to build MLIR enabling Python bindings"
	@echo "  mhlo               to build build MLIR-HLO"
	@echo "  dialects           to build custom Catalyst MLIR dialects"
	@echo "  test               to run the Catalyst MLIR dialects test suite"
	@echo "  clean              to delete all temporary, cache, and build files"
	@echo "  format [check=1]   to apply C++ formatter; use with 'check=1' to check instead of modify (requires clang-format)"
	@echo "  format [version=?] to apply C++ formatter; use with 'version={version}' to run clang-format-{version} instead of clang-format"

.PHONY: all
all: llvm mhlo dialects

.PHONY: llvm
llvm:
	@echo "build LLVM and MLIR enabling Python bindings"
	cmake -G Ninja -S llvm-project/llvm -B $(LLVM_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
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
		-DLLVM_ENABLE_LLD=$(ENABLE_LLD)

	cmake --build $(LLVM_BUILD_DIR) --target check-mlir llvm-symbolizer

.PHONY: mhlo
mhlo:
	# TODO: We can remove this patch whenever this ticket is solved https://github.com/tensorflow/mlir-hlo/issues/68
	patch -d mlir-hlo/mhlo/analysis --merge --follow-symlinks < patches/add-lmhlo-dependency.patch

	@echo "build MLIR-HLO"
	cmake -G Ninja -S mlir-hlo -B $(MHLO_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DMLIR_DIR=$(LLVM_BUILD_DIR)/lib/cmake/mlir \
		-DCMAKE_C_COMPILER=$(C_COMPILER) \
		-DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
		-DCMAKE_C_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DCMAKE_CXX_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DLLVM_ENABLE_LLD=$(ENABLE_LLD)

	cmake --build $(MHLO_BUILD_DIR) --target check-mlir-hlo


.PHONY: dialects
dialects:
	@echo "build custom Catalyst MLIR Dialects"
	cmake -G Ninja -S . -B $(DIALECTS_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DQUANTUM_ENABLE_BINDINGS_PYTHON=ON \
		-DPython3_EXECUTABLE="$(PYTHON)" \
		-DMLIR_DIR=$(LLVM_BUILD_DIR)/lib/cmake/mlir \
		-DMHLO_DIR=$(MHLO_BUILD_DIR)/lib/cmake/mlir-hlo \
		-DMHLO_BINARY_DIR=$(MHLO_BUILD_DIR)/bin \
		-DRUNTIME_LIB_DIR=$(RT_BUILD_DIR)/lib \
		-DMLIR_LIB_DIR=$(LLVM_BUILD_DIR)/lib \
		-DCMAKE_C_COMPILER=$(C_COMPILER) \
		-DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
		-DCMAKE_C_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DCMAKE_CXX_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DLLVM_ENABLE_LLD=$(ENABLE_LLD)

	cmake --build $(DIALECTS_BUILD_DIR) --target check-dialects quantum-lsp-server
	patch -d $(DIALECTS_BUILD_DIR)/python_packages/quantum/mlir_quantum --follow-symlinks < patches/use-ir-from-jax.patch
	patch -d $(DIALECTS_BUILD_DIR)/python_packages/quantum/mlir_quantum/dialects --follow-symlinks < patches/use-cext-from-jax.patch
	patch -d $(DIALECTS_BUILD_DIR)/python_packages/quantum/mlir_quantum/dialects --follow-symlinks < patches/do-not-infer-result.patch
	patch -d $(DIALECTS_BUILD_DIR)/python_packages/quantum/mlir_quantum/dialects --follow-symlinks < patches/remove-gradient-cext-import.patch
	patch -d $(DIALECTS_BUILD_DIR)/python_packages/quantum/mlir_quantum/dialects --follow-symlinks < patches/remove-quantum-cext-import.patch

.PHONY: test
test:
	@echo "test the Catalyst MLIR dialects test suite"
	cmake --build $(DIALECTS_BUILD_DIR) --target check-dialects

.PHONY: clean
clean:
	@echo "clean build files"
	rm -rf $(DIALECTS_BUILD_DIR) $(LLVM_BUILD_DIR) $(MHLO_BUILD_DIR)

.PHONY: format
format:
ifdef check
	$(PYTHON) ../bin/format.py --check $(if $(version:-=),--cfversion $(version)) .
else
	$(PYTHON) ../bin/format.py $(if $(version:-=),--cfversion $(version)) .
endif
