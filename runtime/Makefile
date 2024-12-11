PYTHON?=$(shell which python3)
PYTHON_PREFIX:=$(shell $(PYTHON) -c "import sys; print(sys.prefix)")
PYTHON_VERSION:=$(shell $(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
C_COMPILER?=$(shell which clang)
CXX_COMPILER?=$(shell which clang++)
COMPILER_LAUNCHER?=$(shell which ccache)
NPROC?=$(shell python3 -c "import os; print(os.cpu_count())")

MK_ABSPATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MK_DIR := $(dir $(MK_ABSPATH))
RT_BUILD_DIR?=$(MK_DIR)/build
CODE_COVERAGE?=OFF
BUILD_TYPE?=RelWithDebInfo
ENABLE_OPENQASM?=ON
ENABLE_ASAN?=OFF

BUILD_TARGETS := rt_capi rtd_null_qubit
TEST_TARGETS := runner_tests_qir_runtime

ifeq ($(ENABLE_OPENQASM), ON)
	BUILD_TARGETS += rtd_openqasm
	TEST_TARGETS += runner_tests_openqasm
endif

coverage: CODE_COVERAGE=ON
coverage: BUILD_TYPE=Debug
coverage: C_COMPILER=$(shell which gcc)
coverage: CXX_COMPILER=$(shell which g++)
test: CODE_COVERAGE=OFF
test: BUILD_TYPE?=RelWithDebInfo

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  all                to build Catalyst Runtime"
	@echo "  coverage           to generate a coverage report using lcov"
	@echo "  clean              to delete all temporary, cache, and build files"
	@echo "  test               to run the Catalyst runtime test suite"
	@echo "  format [check=1]   to apply C++ formatter; use with 'check=1' to check instead of modify (requires clang-format)"
	@echo "  format [version=?] to apply C++ formatter; use with 'version={version}' to run clang-format-{version} instead of clang-format"
	@echo "  check-tidy         to build Catalyst Runtime with RUNTIME_CLANG_TIDY=ON (requires clang-tidy)"

.PHONY: configure
configure:
	@echo "Configure Catalyst Runtime"

	cmake -G Ninja -B $(RT_BUILD_DIR) . \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(RT_BUILD_DIR)/lib \
		-DCMAKE_C_COMPILER=$(C_COMPILER) \
		-DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
		-DCMAKE_C_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DCMAKE_CXX_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DENABLE_OPENQASM=$(ENABLE_OPENQASM) \
		-DENABLE_CODE_COVERAGE=$(CODE_COVERAGE) \
		-DPYTHON_EXECUTABLE=$(PYTHON) \
		-DPython_ROOT_DIR=$(PYTHON_PREFIX) \
		-DPYTHON_VERSION_TO_FIND=$(PYTHON_VERSION) \
		-Dpybind11_DIR=$(shell $(PYTHON) -c "import pybind11; print(pybind11.get_cmake_dir())") \
		-DENABLE_ADDRESS_SANITIZER=$(ENABLE_ASAN)

capi_target: configure
	cmake --build $(RT_BUILD_DIR) --target rt_capi -j$(NPROC)

.PHONY: all
all: configure
	cmake --build $(RT_BUILD_DIR) --target $(BUILD_TARGETS) -j$(NPROC)

$(RT_BUILD_DIR)/tests/runner_tests_all_targets: configure
	cmake --build $(RT_BUILD_DIR) --target $(TEST_TARGETS) -j$(NPROC)

.PHONY: test
test: $(RT_BUILD_DIR)/tests/runner_tests_all_targets
	@echo "Catalyst runtime test suite - NullQubit"
	$(ASAN_COMMAND) $(RT_BUILD_DIR)/tests/runner_tests_qir_runtime
ifeq ($(ENABLE_OPENQASM), ON)
	# Test the OpenQasm devices C++ tests
	$(ASAN_COMMAND) $(RT_BUILD_DIR)/tests/runner_tests_openqasm
endif

.PHONY: coverage
coverage: capi_target $(RT_BUILD_DIR)/tests/runner_tests_all_targets
	@echo "check C++ code coverage"
	$(RT_BUILD_DIR)/tests/runner_tests_qir_runtime

	lcov --directory $(RT_BUILD_DIR) -b $(MK_DIR)/lib --capture --output-file $(RT_BUILD_DIR)/coverage.info
	lcov --remove $(RT_BUILD_DIR)/coverage.info '/usr/*' '*/_deps/*' '*/envs/*' '*/mlir/*' --output-file $(RT_BUILD_DIR)/coverage.info
	genhtml $(RT_BUILD_DIR)/coverage.info --output-directory $(RT_BUILD_DIR)/cov -t "Catalyst Runtime C++ Coverage" --num-spaces 4

.PHONY: clean
clean:
	@echo "clean build files"
	rm -rf $(RT_BUILD_DIR) cov coverage.info $(MK_DIR)/BuildTidy

.PHONY: format
format:
ifdef check
	$(PYTHON) ../bin/format.py --check $(if $(version:-=),--cfversion $(version)) .
else
	$(PYTHON) ../bin/format.py $(if $(version:-=),--cfversion $(version)) .
endif

.PHONY: check-tidy
check-tidy:
	@echo "build Catalyst Runtime with RUNTIME_CLANG_TIDY=ON"
	cmake -G Ninja -B $(MK_DIR)/BuildTidy . \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_C_COMPILER=$(C_COMPILER) \
		-DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
		-DRUNTIME_CLANG_TIDY=ON

	cmake --build $(MK_DIR)/BuildTidy --target rt_capi -j$(NPROC)
