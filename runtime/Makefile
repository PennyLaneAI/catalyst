PYTHON?=$(shell which python3)
C_COMPILER?=$(shell which clang)
CXX_COMPILER?=$(shell which clang++)
COMPILER_LAUNCHER?=$(shell which ccache)
NPROC?=$(shell python3 -c "import os; print(os.cpu_count())")

MK_ABSPATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MK_DIR := $(dir $(MK_ABSPATH))
RT_BUILD_DIR?=$(MK_DIR)/build
CODE_COVERAGE?=OFF
BUILD_TYPE?=RelWithDebInfo
ENABLE_WARNINGS?=ON
ENABLE_LIGHTNING?=ON
ENABLE_LIGHTNING_KOKKOS?=ON
ENABLE_OPENQASM?=ON
ENABLE_ASAN?=OFF
LIGHTNING_GIT_TAG_VALUE?=eochoa/2024-08-08/setting-state-functions
ENABLE_LAPACK?=OFF

BUILD_TARGETS := rt_capi rtd_dummy
TEST_TARGETS := ""

ifeq ($(ENABLE_LIGHTNING), ON)
	BUILD_TARGETS += rtd_lightning
	TEST_TARGETS += runner_tests_lightning
endif

ifeq ($(ENABLE_OPENQASM), ON)
	BUILD_TARGETS += rtd_openqasm
	TEST_TARGETS += runner_tests_openqasm
endif

LIGHTNING_ENABLE_OPENMP?=OFF

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
	@echo "  runtime            to build Catalyst Runtime with PennyLane-Lightning"
	@echo "  coverage           to generate a coverage report using lcov"
	@echo "  clean              to delete all temporary, cache, and build files"
	@echo "  test               to run the Catalyst runtime test suite"
	@echo "  format [check=1]   to apply C++ formatter; use with 'check=1' to check instead of modify (requires clang-format)"
	@echo "  format [version=?] to apply C++ formatter; use with 'version={version}' to run clang-format-{version} instead of clang-format"
	@echo "  check-tidy         to build Catalyst Runtime with RUNTIME_CLANG_TIDY=ON (requires clang-tidy)"

.PHONY: all
all: runtime

.PHONY: configure
configure:
	@echo "Configure Catalyst Runtime"

	# TODO: Removing CMakeCache.txt before each cmake configuration will ensure that Catalyst Runtime
	# is always built with proper cmake values. There is currently an issue with caching the
	# PL_BACKEND variable by PennyLane-Lightning which can be fixed here:
	rm -f $(RT_BUILD_DIR)/CMakeCache.txt

	cmake -G Ninja -B $(RT_BUILD_DIR) . \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(RT_BUILD_DIR)/lib \
		-DCMAKE_C_COMPILER=$(C_COMPILER) \
		-DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
		-DCMAKE_C_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DCMAKE_CXX_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DENABLE_LIGHTNING=$(ENABLE_LIGHTNING) \
		-DENABLE_OPENQASM=$(ENABLE_OPENQASM) \
		-DENABLE_OPENMP=$(LIGHTNING_ENABLE_OPENMP) \
		-DENABLE_CODE_COVERAGE=$(CODE_COVERAGE) \
		-DRUNTIME_ENABLE_WARNINGS=$(ENABLE_WARNINGS) \
		-DENABLE_WARNINGS=OFF \
		-DPYTHON_EXECUTABLE=$(PYTHON) \
		-Dpybind11_DIR=$(shell $(PYTHON) -c "import pybind11; print(pybind11.get_cmake_dir())") \
		-DENABLE_ADDRESS_SANITIZER=$(ENABLE_ASAN) \
		-DLIGHTNING_GIT_TAG=$(LIGHTNING_GIT_TAG_VALUE) \
		-DENABLE_LAPACK=$(ENABLE_LAPACK) \
		$(CMAKE_ARGS)

lq_configure:
	@echo "Configure Catalyst Runtime"

	# TODO: Removing CMakeCache.txt before each cmake configuration will ensure that Catalyst Runtime
	# is always built with proper cmake values. There is currently an issue with caching the
	# PL_BACKEND variable by PennyLane-Lightning which can be fixed here:
	rm -f $(RT_BUILD_DIR)/CMakeCache.txt

	cmake -G Ninja -B $(RT_BUILD_DIR) . \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(RT_BUILD_DIR)/lib \
		-DCMAKE_C_COMPILER=$(C_COMPILER) \
		-DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
		-DCMAKE_C_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DCMAKE_CXX_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DENABLE_OPENMP=$(LIGHTNING_ENABLE_OPENMP) \
		-DENABLE_CODE_COVERAGE=$(CODE_COVERAGE) \
		-DRUNTIME_ENABLE_WARNINGS=$(ENABLE_WARNINGS) \
		-DENABLE_WARNINGS=OFF \
		-DPYTHON_EXECUTABLE=$(PYTHON) \
		-Dpybind11_DIR=$(shell $(PYTHON) -c "import pybind11; print(pybind11.get_cmake_dir())") \
		-DENABLE_ADDRESS_SANITIZER=$(ENABLE_ASAN) \
		-DLIGHTNING_GIT_TAG=$(LIGHTNING_GIT_TAG_VALUE) \
		-DENABLE_LAPACK=$(ENABLE_LAPACK) \
		$(CMAKE_ARGS)

lq_target: lq_configure
	cmake --build $(RT_BUILD_DIR) --target rt_capi rtd_lightning -j$(NPROC)

.PHONY: runtime
runtime: configure
	cmake --build $(RT_BUILD_DIR) --target $(BUILD_TARGETS) -j$(NPROC)

$(RT_BUILD_DIR)/tests/runner_tests_lightning: configure
	cmake --build $(RT_BUILD_DIR) --target $(TEST_TARGETS) -j$(NPROC)

.PHONY: dummy_device
dummy_device:
	cmake --build $(RT_BUILD_DIR) --target rtd_dummy -j$(NPROC)

.PHONY: test
test: $(RT_BUILD_DIR)/tests/runner_tests_lightning
	@echo "test the Catalyst runtime test suite"
ifeq ($(ENABLE_OPENQASM), ON)
	# Test the OpenQasm devices C++ tests
	$(ASAN_COMMAND) $(RT_BUILD_DIR)/tests/runner_tests_openqasm
endif
	# Test the Lightning suite C++ tests
	$(ASAN_COMMAND) $(RT_BUILD_DIR)/tests/runner_tests_lightning

.PHONY: coverage
coverage: lq_target
	@echo "check C++ code coverage"

	cmake --build $(RT_BUILD_DIR) --target runner_tests_lightning -j$(NPROC)
	$(RT_BUILD_DIR)/tests/runner_tests_lightning
	lcov --directory $(RT_BUILD_DIR) -b $(MK_DIR)/lib --capture --output-file $(RT_BUILD_DIR)/coverage.info
	lcov --remove $(RT_BUILD_DIR)/coverage.info '/usr/*' '*/_deps/*' '*/envs/*' '*/mlir/*' --output-file $(RT_BUILD_DIR)/coverage.info
	genhtml $(RT_BUILD_DIR)/coverage.info --output-directory $(RT_BUILD_DIR)/cov -t "Catalyst Runtime C++ Coverage" --num-spaces 4

.PHONY: examples
examples: lq_target
	$(MAKE) -C examples all

.PHONY: clean
clean:
	@echo "clean build files"
	rm -rf $(RT_BUILD_DIR) cov coverage.info $(MK_DIR)/BuildTidy
	$(MAKE) -C examples clean

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
