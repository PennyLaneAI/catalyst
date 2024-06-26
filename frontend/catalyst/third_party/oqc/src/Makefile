PYTHON?=$(shell which python3)
C_COMPILER?=$(shell which clang)
CXX_COMPILER?=$(shell which clang++)
NPROC?=$(shell python3 -c "import os; print(os.cpu_count())")

MK_ABSPATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MK_DIR := $(dir $(MK_ABSPATH))
OQC_BUILD_DIR?=$(MK_DIR)/build
RT_BUILD_DIR?=$(MK_DIR)/../../../../../runtime/build
USE_ALTERNATIVE_CATALYST_PYTHON_INTERPRETER?=OFF

.PHONY: configure
configure:
	@echo "Configure OQC Runtime"

	cmake -G Ninja -B $(OQC_BUILD_DIR) \
		-DCMAKE_C_COMPILER=$(C_COMPILER) \
		-DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
		-DRUNTIME_BUILD_DIR=$(RT_BUILD_DIR) \
		-DUSE_ALTERNATIVE_CATALYST_PYTHON_INTERPRETER_PATH=$(USE_ALTERNATIVE_CATALYST_PYTHON_INTERPRETER) \
		-DPYTHON_EXECUTABLE=$(PYTHON) \
		-Dpybind11_DIR=$(shell $(PYTHON) -c "import pybind11; print(pybind11.get_cmake_dir())")

$(OQC_BUILD_DIR)/librtd_oqc.so: configure
	cmake --build $(OQC_BUILD_DIR) --target rtd_oqc -j$(NPROC)

.PHONY: oqc
oqc: $(OQC_BUILD_DIR)/librtd_oqc.so

$(OQC_BUILD_DIR)/tests/runner_tests_oqc: configure
	cmake --build $(OQC_BUILD_DIR) --target runner_tests_oqc -j$(NPROC)

.PHONY: test
test: $(OQC_BUILD_DIR)/tests/runner_tests_oqc
	@echo "test the Catalyst runtime test suite"
	$(OQC_BUILD_DIR)/tests/runner_tests_oqc

.PHONY: clean
clean:
	@echo "clean build files"
	rm -rf $(OQC_BUILD_DIR)
