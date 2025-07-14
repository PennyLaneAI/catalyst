PYTHON ?= $(shell which python3)
C_COMPILER ?= $(shell which clang)
CXX_COMPILER ?= $(shell which clang++)
BLACKVERSIONMAJOR := $(shell black --version 2> /dev/null | head -n1 | awk '{ print $$2 }' | cut -d. -f1)
BLACKVERSIONMAJOR := $(if $(BLACKVERSIONMAJOR),$(BLACKVERSIONMAJOR),0)
BLACKVERSIONMINOR := $(shell black --version 2> /dev/null | head -n1 | awk '{ print $$2 }' | cut -d. -f2)
BLACKVERSIONMINOR := $(if $(BLACKVERSIONMINOR),$(BLACKVERSIONMINOR),0)
MK_ABSPATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MK_DIR := $(dir $(MK_ABSPATH))
LLVM_BUILD_DIR ?= $(MK_DIR)/mlir/llvm-project/build
MHLO_BUILD_DIR ?= $(MK_DIR)/mlir/mlir-hlo/bazel-build
DIALECTS_SRC_DIR ?= $(MK_DIR)/mlir
DIALECTS_BUILD_DIR ?= $(MK_DIR)/mlir/build
RT_BUILD_DIR ?= $(MK_DIR)/runtime/build
OQC_BUILD_DIR ?= $(MK_DIR)/frontend/catalyst/third_party/oqc/src/build
ENZYME_BUILD_DIR ?= $(MK_DIR)/mlir/Enzyme/build
COVERAGE_REPORT ?= term-missing
ENABLE_OPENQASM ?= ON
ENABLE_OQD ?= OFF
TEST_BACKEND ?= "lightning.qubit"
TEST_BRAKET ?= NONE
ENABLE_ASAN ?= OFF
TOML_SPECS ?= $(shell find ./runtime ./frontend -name '*.toml' -not -name 'pyproject.toml')
ENABLE_FLAKY ?= OFF

PLATFORM := $(shell uname -s)
ifeq ($(PLATFORM),Linux)
COPY_FLAGS := --dereference
endif

# Note: ASAN replaces dlopen calls, which means that when we open other libraries via dlopen that
#       relied on the parent's library's RPATH, these libraries are no longer found.
#         e.g. `dlopen(catalyst_callback_registry.so)` from RuntimeCAPI.cpp
#              `dlopen(libscipy_openblas.dylib)` from lightning's BLASLibLoaderManager.hpp
#       We can fix this using LD_LIBRARY_PATH (for dev builds).
ifeq ($(PLATFORM) $(findstring clang,$(C_COMPILER)),Linux clang)
ASAN_FLAGS := LD_PRELOAD="$(shell clang  -print-file-name=libclang_rt.asan-x86_64.so)"
ASAN_FLAGS += LD_LIBRARY_PATH="$(RT_BUILD_DIR)/lib:$(LD_LIBRARY_PATH)"
else ifeq ($(PLATFORM) $(findstring gcc,$(C_COMPILER)),Linux gcc)
ASAN_FLAGS := LD_PRELOAD="$(shell gcc  -print-file-name=libasan.so)"
ASAN_FLAGS += LD_LIBRARY_PATH="$(RT_BUILD_DIR)/lib:$(LD_LIBRARY_PATH)"
else ifeq ($(PLATFORM),Darwin)
ASAN_FLAGS := DYLD_INSERT_LIBRARIES="$(shell clang -print-file-name=libclang_rt.asan_osx_dynamic.dylib)"
SCIPY_DIR := $(shell python -c 'import os, scipy_openblas32; print(os.path.dirname(scipy_openblas32.__file__))')
ASAN_FLAGS += DYLD_LIBRARY_PATH="$(RT_BUILD_DIR)/lib:$(SCIPY_DIR)/lib:$(DYLD_LIBRARY_PATH)"
endif

PARALLELIZE := -n auto
ifeq ($(ENABLE_ASAN),ON)
ifeq ($(PLATFORM),Darwin)
# Launching subprocesses with ASAN on macOS is not supported (see https://stackoverflow.com/a/47853433).
PARALLELIZE :=
endif
# These tests build a standalone executable from the Python frontend, which would have to be built
# with the ASAN runtime. Since we don't exert much control over the "user" compiler, skip them.
TEST_EXCLUDES := -k "not test_executable_generation"
endif
FLAKY :=
ifeq ($(ENABLE_FLAKY),ON)
FLAKY := --force-flaky --max-runs=5 --min-passes=5
endif
PYTEST_FLAGS := $(PARALLELIZE) $(TEST_EXCLUDES) $(FLAKY)

# TODO: Find out why we have container overflow on macOS.
ASAN_OPTIONS := ASAN_OPTIONS="detect_leaks=0,detect_container_overflow=0"

ifeq ($(ENABLE_OPENQASM), ON)
# A global 'mutex' is added to protect `pybind11::exec` calls concurrently in `OpenQasmRunner`.
# TODO: Remove this global 'mutex' to enable concurrent execution of remote calls.
# This 'mutex' leads to an ODR violation when using ASAN
ASAN_OPTIONS := ASAN_OPTIONS="detect_leaks=0,detect_container_overflow=0,detect_odr_violation=0"
endif

ifeq ($(ENABLE_ASAN),ON)
ASAN_COMMAND := $(ASAN_OPTIONS) $(ASAN_FLAGS)
else
ASAN_COMMAND :=
endif

# Flag for verbose pip install output
PIP_VERBOSE_FLAG :=
ifeq ($(VERBOSE),1)
PIP_VERBOSE_FLAG := --verbose
endif

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  all                to build and install all Catalyst modules and its MLIR dependencies"
	@echo "  frontend           to install Catalyst Frontend"
	@echo "  mlir               to build MLIR and custom Catalyst dialects"
	@echo "  runtime            to build Catalyst Runtime"
	@echo "  oqc                to build Catalyst-OQC Runtime"
	@echo "  test               to run the Catalyst test suites"
	@echo "  docs               to build the documentation for Catalyst"
	@echo "  wheel              to build the Catalyst wheel"
	@echo "  clean              to uninstall Catalyst and delete frontend build and cache files"
	@echo "  clean-mlir         to clean build files of MLIR and custom Catalyst dialects"
	@echo "  clean-runtime      to clean build files of Catalyst Runtime"
	@echo "  clean-oqc          to clean build files of OQC Runtime"
	@echo "  clean-all          to uninstall Catalyst and delete all temporary, cache, and build files"
	@echo "  clean-catalyst     to uninstall Catalyst and delete all temporary, cache, and build files for catalyst dialects and runtime (but keeping llvm build files)"
	@echo "  clean-docs         to delete all built documentation"
	@echo "  coverage           to generate a coverage report"
	@echo "  format [check=1]   to apply C++ and Python formatter; use with 'check=1' to check instead of modify (requires black, pylint and clang-format)"
	@echo "  format [version=?] to apply C++ and Python formatter; use with 'version={version}' to run clang-format-{version} instead of clang-format"


.PHONY: all catalyst
all: runtime oqc mlir frontend
catalyst: runtime dialects plugin frontend oqc

.PHONY: frontend
frontend:
	@echo "install Catalyst Frontend"
	# Uninstall pennylane before updating Catalyst, since pip will not replace two development
	# versions of a package with the same version tag (e.g. 0.38-dev0).
	$(PYTHON) -m pip uninstall -y pennylane
	$(PYTHON) -m pip install -e . --extra-index-url https://test.pypi.org/simple $(PIP_VERBOSE_FLAG)
	rm -r frontend/pennylane_catalyst.egg-info

.PHONY: mlir llvm mhlo enzyme dialects runtime oqc
mlir:
	$(MAKE) -C mlir all

llvm:
	$(MAKE) -C mlir llvm

mhlo:
	$(MAKE) -C mlir mhlo

enzyme:
	$(MAKE) -C mlir enzyme

dialects:
	$(MAKE) -C mlir dialects

runtime:
	$(MAKE) -C runtime runtime ENABLE_OQD=$(ENABLE_OQD)

oqc:
	$(MAKE) -C frontend/catalyst/third_party/oqc/src oqc

.PHONY: test test-runtime test-frontend lit pytest test-demos test-oqc test-toml-spec
test: test-runtime test-frontend test-demos

test-toml-spec:
	$(PYTHON) ./bin/toml-check.py $(TOML_SPECS)

test-runtime:
	$(MAKE) -C runtime test

test-mlir:
	$(MAKE) -C mlir test

test-frontend: lit pytest

test-oqc:
	$(MAKE) -C frontend/catalyst/third_party/oqc/src test

lit:
ifeq ($(ENABLE_ASAN),ON)
ifneq ($(findstring clang,$(C_COMPILER)),clang)
	@echo "Running Python tests with Address Sanitizer is only supported with Clang, but provided $(C_COMPILER)"
	@exit 1
endif
endif
	@echo "check the Catalyst lit test suite"
	cmake --build $(DIALECTS_BUILD_DIR) --target check-frontend

pytest:
ifeq ($(ENABLE_ASAN),ON)
ifneq ($(findstring clang,$(C_COMPILER)),clang)
	@echo "Running Python tests with Address Sanitizer is only supported with Clang, but provided $(C_COMPILER)"
	@exit 1
endif
endif
	@echo "check the Catalyst PyTest suite"
	$(ASAN_COMMAND) $(PYTHON) -m pytest frontend/test/pytest --tb=native --backend=$(TEST_BACKEND) --runbraket=$(TEST_BRAKET) $(PYTEST_FLAGS)
ifeq ($(TEST_BRAKET), NONE)
	$(ASAN_COMMAND) $(PYTHON) -m pytest frontend/test/async_tests --tb=native --backend=$(TEST_BACKEND)
	$(ASAN_COMMAND) $(PYTHON) -m pytest frontend/test/test_oqc/oqc
ifeq ($(ENABLE_OQD), ON)
	$(ASAN_COMMAND) $(PYTHON) -m pytest frontend/test/test_oqd/oqd
endif
endif

test-demos:
ifeq ($(ENABLE_ASAN) $(PLATFORM),ON Darwin)
	@echo "Cannot run Jupyter Notebooks with ASAN on macOS, likely due to subprocess invocation."
	@exit 1
endif
	@echo "check the Catalyst demos"
	MDD_BENCHMARK_PRECISION=1 \
	$(ASAN_COMMAND) $(PYTHON) -m pytest demos --nbmake $(PYTEST_FLAGS)

wheel:
	echo "INSTALLED = True" > $(MK_DIR)/frontend/catalyst/_configuration.py

	# Copy libs to frontend/catalyst/lib
	mkdir -p $(MK_DIR)/frontend/catalyst/lib/backend
	cp $(RT_BUILD_DIR)/lib/librtd* $(MK_DIR)/frontend/catalyst/lib
	cp $(RT_BUILD_DIR)/lib/catalyst_callback_registry.so $(MK_DIR)/frontend/catalyst/lib
	cp $(RT_BUILD_DIR)/lib/openqasm_python_module.so $(MK_DIR)/frontend/catalyst/lib
	cp $(RT_BUILD_DIR)/lib/liblapacke.* $(MK_DIR)/frontend/catalyst/lib || true  # optional
	cp $(RT_BUILD_DIR)/lib/librt_capi.* $(MK_DIR)/frontend/catalyst/lib
	cp $(RT_BUILD_DIR)/lib/backend/*.toml $(MK_DIR)/frontend/catalyst/lib/backend
	cp $(OQC_BUILD_DIR)/librtd_oqc* $(MK_DIR)/frontend/catalyst/lib
	cp $(OQC_BUILD_DIR)/oqc_python_module.so $(MK_DIR)/frontend/catalyst/lib
	cp $(OQC_BUILD_DIR)/backend/*.toml $(MK_DIR)/frontend/catalyst/lib/backend
	cp $(COPY_FLAGS) $(LLVM_BUILD_DIR)/lib/libmlir_float16_utils.* $(MK_DIR)/frontend/catalyst/lib
	cp $(COPY_FLAGS) $(LLVM_BUILD_DIR)/lib/libmlir_c_runner_utils.* $(MK_DIR)/frontend/catalyst/lib
	cp $(COPY_FLAGS) $(LLVM_BUILD_DIR)/lib/libmlir_async_runtime.* $(MK_DIR)/frontend/catalyst/lib

	# Copy mlir bindings & compiler driver to frontend/mlir_quantum
	mkdir -p $(MK_DIR)/frontend/mlir_quantum/dialects
	cp -R $(COPY_FLAGS) $(DIALECTS_BUILD_DIR)/python_packages/quantum/mlir_quantum/runtime $(MK_DIR)/frontend/mlir_quantum/runtime
	for file in gradient quantum _ods_common catalyst mbqc mitigation _transform; do \
		cp $(COPY_FLAGS) $(DIALECTS_BUILD_DIR)/python_packages/quantum/mlir_quantum/dialects/*$${file}* $(MK_DIR)/frontend/mlir_quantum/dialects ; \
	done
	mkdir -p $(MK_DIR)/frontend/bin
	cp $(COPY_FLAGS) $(DIALECTS_BUILD_DIR)/bin/catalyst $(MK_DIR)/frontend/bin/
	find $(MK_DIR)/frontend -type d -name __pycache__ -exec rm -rf {} +

	# Copy selected headers to `frontend/include' to include them in the wheel
	mkdir -p $(MK_DIR)/frontend/catalyst/include
	find $(DIALECTS_SRC_DIR)/include/Quantum $(DIALECTS_BUILD_DIR)/include/Quantum \
	    $(DIALECTS_SRC_DIR)/include/Gradient $(DIALECTS_BUILD_DIR)/include/Gradient \
	    $(DIALECTS_SRC_DIR)/include/Mitigation $(DIALECTS_BUILD_DIR)/include/Mitigation \
	    \( -name "*.h" -o -name "*.h.inc" \) -type f -exec sh -c \
	    'for file do \
	        if [ "$$file" = "$${file#$(DIALECTS_BUILD_DIR)}" ]; then \
				base_dir=$(DIALECTS_SRC_DIR); \
			else \
				base_dir=$(DIALECTS_BUILD_DIR); \
			fi; \
			dest_dir=$(MK_DIR)/frontend/catalyst/include/$$(dirname $${file#$${base_dir}/include/}); \
			mkdir -p $$dest_dir; \
		    cp $(COPY_FLAGS) $$file $$dest_dir; \
	    done' sh {} +

	$(PYTHON) -m pip wheel --no-deps . -w dist

	rm -r $(MK_DIR)/build
	rm -r frontend/pennylane_catalyst.egg-info

plugin-wheel: plugin
	mkdir -p $(MK_DIR)/standalone_plugin_wheel/standalone_plugin/lib
	cp $(COPY_FLAGS) $(DIALECTS_BUILD_DIR)/lib/StandalonePlugin.* $(MK_DIR)/standalone_plugin_wheel/standalone_plugin/lib

	$(PYTHON) -m pip wheel --no-deps $(MK_DIR)/standalone_plugin_wheel -w $(MK_DIR)/standalone_plugin_wheel/dist

	rm -r $(MK_DIR)/standalone_plugin_wheel/standalone_plugin/lib
	rm -r $(MK_DIR)/standalone_plugin_wheel/standalone_plugin.egg-info
	rm -r $(MK_DIR)/standalone_plugin_wheel/build

.PHONY: clean clean-all clean-catalyst
clean:
	@echo "uninstall catalyst and delete all temporary and cache files"
	$(PYTHON) -m pip uninstall -y pennylane-catalyst
	find frontend/catalyst -name "*.so" -not -path "*/third_party/*" -exec rm -v {} +
	git restore frontend/catalyst/_configuration.py
	rm -rf $(MK_DIR)/frontend/catalyst/_revision.py
	rm -rf $(MK_DIR)/frontend/mlir_quantum $(MK_DIR)/frontend/catalyst/lib $(MK_DIR)/frontend/catalyst/bin
	rm -rf dist __pycache__
	rm -rf .coverage coverage_html_report
	rm -rf .benchmarks

clean-all: clean clean-mlir clean-runtime clean-oqc
clean-catalyst: clean clean-dialects clean-runtime clean-oqc

.PHONY: clean-mlir clean-dialects clean-plugin clean-llvm clean-mhlo clean-enzyme
clean-mlir:
	$(MAKE) -C mlir clean

clean-dialects:
	$(MAKE) -C mlir clean-dialects

clean-plugin:
	$(MAKE) -C mlir clean-plugin

clean-llvm:
	$(MAKE) -C mlir clean-llvm

clean-mhlo:
	$(MAKE) -C mlir clean-mhlo

clean-enzyme:
	$(MAKE) -C mlir clean-enzyme

.PHONY: clean-runtime clean-oqc
clean-runtime:
	$(MAKE) -C runtime clean

clean-oqc:
	$(MAKE) -C frontend/catalyst/third_party/oqc/src clean

.PHONY: coverage coverage-frontend coverage-runtime
coverage: coverage-frontend coverage-runtime

coverage-frontend:
ifeq ($(ENABLE_ASAN),ON)
ifneq ($(findstring clang,$(C_COMPILER)),clang)
	@echo "Running Python tests with Address Sanitizer is only supported with Clang, but provided $(C_COMPILER)"
	@exit 1
endif
endif
	@echo "Generating coverage report for the frontend"
	$(ASAN_COMMAND) $(PYTHON) -m pytest frontend/test/pytest $(PYTEST_FLAGS) --cov=catalyst --tb=native --cov-report=$(COVERAGE_REPORT)
	$(ASAN_COMMAND) $(PYTHON) -m pytest frontend/test/test_oqc/oqc $(PYTEST_FLAGS) --cov=catalyst --cov-append --tb=native --cov-report=$(COVERAGE_REPORT)
ifeq ($(ENABLE_OQD), ON)
	$(ASAN_COMMAND) $(PYTHON) -m pytest frontend/test/test_oqd/oqd $(PYTEST_FLAGS) --cov=catalyst --cov-append --tb=native --cov-report=$(COVERAGE_REPORT)
endif
ifeq ($(TEST_BRAKET), NONE)
	$(ASAN_COMMAND) $(PYTHON) -m pytest frontend/test/async_tests --tb=native --backend=$(TEST_BACKEND) --tb=native
endif

coverage-runtime:
	$(MAKE) -C runtime coverage

.PHONY: plugin
plugin:
	$(MAKE) -C mlir plugin

.PHONY: format
format:
ifeq ($(shell test $(BLACKVERSIONMAJOR) -lt 22; echo $$?), 0)
	$(error black version is too old, please update to at least 22.10)
endif
ifeq ($(shell test $(BLACKVERSIONMAJOR) -eq 22 -a $(BLACKVERSIONMINOR) -lt 10; echo $$?), 0)
	$(error black version is too old, please update to at least 22.10)
endif
	$(MAKE) -C mlir format
	$(MAKE) -C runtime format
	$(MAKE) format-frontend

.PHONY: format-frontend
format-frontend:
ifdef check
	$(PYTHON) ./bin/format.py --check $(if $(version:-=),--cfversion $(version)) ./frontend
	black --check --verbose .
	isort --check --diff .
else
	$(PYTHON) ./bin/format.py $(if $(version:-=),--cfversion $(version)) ./frontend
	black .
	isort .
endif

.PHONY: docs clean-docs
docs:
	$(MAKE) -C doc html

clean-docs:
	$(MAKE) -C doc clean
