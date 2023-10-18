PYTHON := python3
C_COMPILER?=clang
CXX_COMPILER?=clang++
COMPILER_LAUNCHER?=ccache
MK_ABSPATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MK_DIR := $(dir $(MK_ABSPATH))
RT_BUILD_DIR?=$(MK_DIR)/build
CMAKE_ARGS?=""
CODE_COVERAGE?=OFF
BUILD_TYPE?=Release
ENABLE_WARNINGS?=ON
ENABLE_LIGHTNING_KOKKOS?=OFF
ENABLE_OPENQASM?=OFF
BUILD_QIR_STDLIB_FROM_SRC?=OFF
# TODO: Update to latest_release after v0.33.0
LIGHTNING_GIT_TAG_VALUE= "0bce777"
NPROC?=$(shell python -c "import os; print(os.cpu_count())")

coverage: CODE_COVERAGE=ON
coverage: BUILD_TYPE=Debug
test: CODE_COVERAGE=OFF
test: BUILD_TYPE?=Release

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
	cmake -G Ninja -B $(RT_BUILD_DIR) . \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(RT_BUILD_DIR)/lib \
		-DLIGHTNING_GIT_TAG=$(LIGHTNING_GIT_TAG_VALUE) \
		-DENABLE_LIGHTNING_KOKKOS=$(ENABLE_LIGHTNING_KOKKOS) \
		-DENABLE_OPENQASM=$(ENABLE_OPENQASM) \
		-DCMAKE_C_COMPILER=$(C_COMPILER) \
		-DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
		-DCMAKE_C_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DCMAKE_CXX_COMPILER_LAUNCHER=$(COMPILER_LAUNCHER) \
		-DENABLE_WARNINGS=$(ENABLE_WARNINGS) \
		-DENABLE_CODE_COVERAGE=$(CODE_COVERAGE) \
		-DBUILD_QIR_STDLIB_FROM_SRC=$(BUILD_QIR_STDLIB_FROM_SRC) \
		$(CMAKE_ARGS)

$(RT_BUILD_DIR)/lib/backend/librt_backend.so $(RT_BUILD_DIR)/lib/librt_capi.so: configure
	cmake --build $(RT_BUILD_DIR) --target rt_capi -j$(NPROC)

.PHONY: runtime
runtime: $(RT_BUILD_DIR)/lib/backend/librt_backend.so $(RT_BUILD_DIR)/lib/librt_capi.so

$(RT_BUILD_DIR)/tests/runner_tests: configure
	cmake --build $(RT_BUILD_DIR) --target runner_tests -j$(NPROC)

.PHONY: test
test: $(RT_BUILD_DIR)/tests/runner_tests
	@echo "test the Catalyst runtime test suite"
	$(RT_BUILD_DIR)/tests/runner_tests

.PHONY: coverage
coverage: $(RT_BUILD_DIR)/tests/runner_tests
	@echo "check C++ code coverage"

	$(RT_BUILD_DIR)/tests/runner_tests
	lcov --directory $(RT_BUILD_DIR) -b $(MK_DIR)/lib --capture --output-file $(RT_BUILD_DIR)/coverage.info
	lcov --remove $(RT_BUILD_DIR)/coverage.info '/usr/*' '*/_deps/*' '*/envs/*' --output-file $(RT_BUILD_DIR)/coverage.info
	genhtml $(RT_BUILD_DIR)/coverage.info --output-directory $(RT_BUILD_DIR)/cov -t "Catalyst Runtime C++ Coverage" --num-spaces 4

.PHONY: examples
examples: runtime
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
