PYTHON := python3
C_COMPILER := $(if $(c_compiler:-=),$(c_compiler),clang)
CPP_COMPILER := $(if $(cpp_compiler:-=),$(cpp_compiler),clang++)
MK_ABSPATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MK_DIR := $(dir $(MK_ABSPATH))
RT_BUILD_DIR := $(if $(runtime_build_dir:-=),$(runtime_build_dir),$(MK_DIR)/build)
QIR_STDLIB_DIR = $(if $(qir_stdlib_dir:-=),$(qir_stdlib_dir),"$(MK_DIR)/qir-stdlib/target/release/")
QIR_STDLIB_INCLUDES_DIR=$(if $(qir_stdlib_include_dir:-=),$(qir_stdlib_include_dir),"$(MK_DIR)/qir-stdlib/target/release/build/include/")


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

.PHONY: all
all: qir | runtime

.PHONY: runtime qir
runtime:
	@echo "build Catalyst Runtime"
	cmake -G Ninja -B $(RT_BUILD_DIR) . \
		-DENABLE_LIGHTNING_KOKKOS=$(if $(kokkos:-=),$(kokkos),OFF) \
		-DCMAKE_C_COMPILER=$(C_COMPILER) \
		-DCMAKE_CXX_COMPILER=$(CPP_COMPILER) \
		-DQIR_STDLIB_LIB=$(QIR_STDLIB_DIR) \
		-DQIR_STDLIB_INCLUDES=$(QIR_STDLIB_INCLUDES_DIR) \
		-DENABLE_CODE_COVERAGE=OFF

	cmake --build $(RT_BUILD_DIR) --target rt_capi -j$(if $(nprocs:-=),$(nprocs),$$(nproc))

qir:
	$(MAKE) -C qir-stdlib qir

.PHONY: examples
examples: | runtime
	$(MAKE) -C examples all

.PHONY: test
test: | runtime
	@echo "test the Catalyst runtime test suite"
	cmake --build $(RT_BUILD_DIR) --target runner_tests -j$(if $(nprocs:-=),$(nprocs),$$(nproc))
	$(RT_BUILD_DIR)/tests/runner_tests

.PHONY: coverage
coverage: | qir
	@echo "check C++ code coverage"
	cmake -G Ninja -B $(RT_BUILD_DIR) . \
		-DCMAKE_BUILD_TYPE=Debug \
		-DENABLE_LIGHTNING_KOKKOS=$(if $(kokkos:-=),$(kokkos),OFF) \
		-DCMAKE_C_COMPILER=$(C_COMPILER) \
		-DCMAKE_CXX_COMPILER=$(CPP_COMPILER) \
		-DQIR_STDLIB_LIB=$(QIR_STDLIB_DIR) \
		-DQIR_STDLIB_INCLUDES=$(QIR_STDLIB_INCLUDES_DIR) \
		-DENABLE_CODE_COVERAGE=ON

	cmake --build $(RT_BUILD_DIR) --target runner_tests -j$(if $(nprocs:-=),$(nprocs),$$(nproc))
	$(RT_BUILD_DIR)/tests/runner_tests

	lcov --directory $(RT_BUILD_DIR) -b $(RT_BUILD_DIR) --capture --output-file $(RT_BUILD_DIR)/coverage.info
	lcov --remove $(RT_BUILD_DIR)/coverage.info '/usr/*' '*/_deps/*' --output-file $(RT_BUILD_DIR)/coverage.info
	genhtml $(RT_BUILD_DIR)/coverage.info --output-directory $(RT_BUILD_DIR)/cov -t "Catalyst Runtime C++ Coverage" --num-spaces 4

.PHONY: clean
clean:
	@echo "clean build files"
	rm -rf $(RT_BUILD_DIR)
	rm -rf cov coverage.info
	$(MAKE) -C qir-stdlib clean
	$(MAKE) -C examples clean

.PHONY: format
format:
ifdef check
	$(PYTHON) ../bin/format.py --check $(if $(version:-=),--cfversion $(version)) .
else
	$(PYTHON) ../bin/format.py $(if $(version:-=),--cfversion $(version)) .
endif
