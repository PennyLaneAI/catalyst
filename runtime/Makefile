PYTHON := python3
C_COMPILER?=clang
CXX_COMPILER?=clang++
MK_ABSPATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MK_DIR := $(dir $(MK_ABSPATH))
RT_BUILD_DIR?=$(MK_DIR)/build
QIR_STDLIB_DIR?=$(MK_DIR)/qir-stdlib/target/release/
QIR_STDLIB_INCLUDES_DIR?=$(MK_DIR)/qir-stdlib/target/release/build/include/
LIGHTNING_GIT_TAG_VALUE?=latest_release
CODE_COVERAGE?=OFF
ENABLE_KOKKOS?=OFF
BUILD_TYPE?=Release

coverage: CODE_COVERAGE=ON
coverage: BUILD_TYPE=Debug
test: BUILD_TYPE=Debug

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  all                to build Catalyst Runtime and the QIR-stdlib Rust crate"
	@echo "  runtime            to build Catalyst Runtime with PennyLane-Lightning"
	@echo "  runtime kokkos=ON  to build Catalyst Runtime with PennyLane-Lightning-Kokkos"
	@echo "  qir                to build the QIR-stdlib Rust crate"
	@echo "  coverage           to generate a coverage report using lcov"
	@echo "  clean              to delete all temporary, cache, and build files"
	@echo "  test               to run the Catalyst runtime test suite"
	@echo "  format [check=1]   to apply C++ formatter; use with 'check=1' to check instead of modify (requires clang-format)"
	@echo "  format [version=?] to apply C++ formatter; use with 'version={version}' to run clang-format-{version} instead of clang-format"
	@echo "  check-tidy         to build Catalyst Runtime with RUNTIME_CLANG_TIDY=ON (requires clang-tidy)"

.PHONY: all
all: qir | runtime

.PHONY: configure
configure:
	@echo "Configure Catalyst Runtime"
	cmake -G Ninja -B $(RT_BUILD_DIR) . \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DLIGHTNING_GIT_TAG=$(LIGHTNING_GIT_TAG_VALUE) \
		-DENABLE_LIGHTNING_KOKKOS=$(ENABLE_KOKKOS) \
		-DCMAKE_C_COMPILER=$(C_COMPILER) \
		-DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
		-DQIR_STDLIB_LIB=$(QIR_STDLIB_DIR) \
		-DQIR_STDLIB_INCLUDES=$(QIR_STDLIB_INCLUDES_DIR) \
		-DENABLE_CODE_COVERAGE=$(CODE_COVERAGE)

.PHONY: runtime qir
runtime: configure
	@echo "Build Catalyst Runtime"
	cmake --build $(RT_BUILD_DIR) --target rt_capi -j$(if $(nprocs:-=),$(nprocs),$$(nproc))

qir:
	$(MAKE) -C qir-stdlib qir

.PHONY: examples
examples: | runtime
	$(MAKE) -C examples all

.PHONY: test
test: | qir configure
	@echo "test the Catalyst runtime test suite"
	cmake --build $(RT_BUILD_DIR) --target runner_tests -j$(if $(nprocs:-=),$(nprocs),$$(nproc))
	$(RT_BUILD_DIR)/tests/runner_tests

.PHONY: coverage
coverage: | qir configure
	@echo "check C++ code coverage"

	cmake --build $(RT_BUILD_DIR) --target runner_tests -j$(if $(nprocs:-=),$(nprocs),$$(nproc))
	$(RT_BUILD_DIR)/tests/runner_tests

	lcov --directory $(RT_BUILD_DIR) -b $(MK_DIR)/lib --capture --output-file $(RT_BUILD_DIR)/coverage.info
	lcov --remove $(RT_BUILD_DIR)/coverage.info '/usr/*' '*/_deps/*' --output-file $(RT_BUILD_DIR)/coverage.info
	genhtml $(RT_BUILD_DIR)/coverage.info --output-directory $(RT_BUILD_DIR)/cov -t "Catalyst Runtime C++ Coverage" --num-spaces 4

.PHONY: clean
clean:
	@echo "clean build files"
	rm -rf $(RT_BUILD_DIR)
	rm -rf cov coverage.info
	rm -rf $(MK_DIR)/BuildTidy
	$(MAKE) -C qir-stdlib clean
	$(MAKE) -C examples clean

.PHONY: format
format:
ifdef check
	$(PYTHON) ../bin/format.py --check $(if $(version:-=),--cfversion $(version)) .
else
	$(PYTHON) ../bin/format.py $(if $(version:-=),--cfversion $(version)) .
endif

.PHONY: check-tidy
check-tidy: | qir
	@echo "build Catalyst Runtime with RUNTIME_CLANG_TIDY=ON"
	rm -rf 
	cmake -G Ninja -B $(MK_DIR)/BuildTidy . \
		-DCMAKE_C_COMPILER=$(C_COMPILER) \
		-DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
		-DQIR_STDLIB_LIB=$(QIR_STDLIB_DIR) \
		-DQIR_STDLIB_INCLUDES=$(QIR_STDLIB_INCLUDES_DIR) \
		-DRUNTIME_CLANG_TIDY=ON 

	cmake --build $(MK_DIR)/BuildTidy --target rt_capi -j$(if $(nprocs:-=),$(nprocs),$$(nproc))
