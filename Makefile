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
DIALECTS_BUILD_DIR ?= $(MK_DIR)/mlir/build
RT_BUILD_DIR ?= $(MK_DIR)/runtime/build
OQC_BUILD_DIR ?= $(MK_DIR)/frontend/catalyst/third_party/oqc/src/build
OQD_BUILD_DIR ?= $(MK_DIR)/frontend/catalyst/third_party/oqd/src/build
ENZYME_BUILD_DIR ?= $(MK_DIR)/mlir/Enzyme/build
COVERAGE_REPORT ?= term-missing
ENABLE_OPENQASM?=ON
TEST_BACKEND ?= "lightning.qubit"
TEST_BRAKET ?= NONE
SKIP_OQD ?= false
ENABLE_ASAN ?= OFF
TOML_SPECS ?= $(shell find ./runtime ./frontend -name '*.toml' -not -name 'pyproject.toml')

PLATFORM := $(shell uname -s)
ifeq ($(PLATFORM),Linux)
COPY_FLAGS := --dereference
endif

ifeq ($(PLATFORM) $(findstring clang,$(C_COMPILER)),Linux clang)
ASAN_FLAGS := LD_PRELOAD="$(shell clang  -print-file-name=libclang_rt.asan-x86_64.so)"
else ifeq ($(PLATFORM) $(findstring gcc,$(C_COMPILER)),Linux gcc)
ASAN_FLAGS := LD_PRELOAD="$(shell gcc  -print-file-name=libasan.so)"
else ifeq ($(PLATFORM),Darwin)
ASAN_FLAGS := DYLD_INSERT_LIBRARIES="$(shell clang -print-file-name=libclang_rt.asan_osx_dynamic.dylib)"
endif

PARALLELIZE := -n auto
ifeq ($(ENABLE_ASAN) $(PLATFORM),ON Darwin)
# Launching subprocesses with ASAN on macOS is not supported (see https://stackoverflow.com/a/47853433).
PARALLELIZE :=
endif

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

# Export variables so that they can be set here without needing to also set them in sub-make files.
export ENABLE_ASAN ASAN_COMMAND

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  all                to build and install all Catalyst modules and its MLIR dependencies"
	@echo "  frontend           to install Catalyst Frontend"
	@echo "  mlir               to build MLIR and custom Catalyst dialects"
	@echo "  runtime            to build Catalyst Runtime"
	@echo "  oqc                to build Catalyst-OQC Runtime"
	@echo "  oqd                to build Catalyst-OQD Runtime"
	@echo "  test               to run the Catalyst test suites"
	@echo "  docs               to build the documentation for Catalyst"
	@echo "  clean              to uninstall Catalyst and delete all temporary and cache files"
	@echo "  clean-frontend     to clean build files of Catalyst Frontend"
	@echo "  clean-mlir         to clean build files of MLIR and custom Catalyst dialects"
	@echo "  clean-runtime      to clean build files of Catalyst Runtime"
	@echo "  clean-oqc          to clean build files of OQC Runtime"
	@echo "  clean-oqd          to clean build files of OQD Runtime"
	@echo "  clean-all          to uninstall Catalyst and delete all temporary, cache, and build files"
	@echo "  clean-docs         to delete all built documentation"
	@echo "  coverage           to generate a coverage report"
	@echo "  format [check=1]   to apply C++ and Python formatter; use with 'check=1' to check instead of modify (requires black, pylint and clang-format)"
	@echo "  format [version=?] to apply C++ and Python formatter; use with 'version={version}' to run clang-format-{version} instead of clang-format"


.PHONY: all catalyst
all: runtime oqc oqd mlir frontend
catalyst: runtime dialects frontend

.PHONY: frontend
frontend:
	@echo "install Catalyst Frontend"
	# Uninstall pennylane before updating Catalyst, since pip will not replace two development
	# versions of a package with the same version tag (e.g. 0.38-dev0).
	$(PYTHON) -m pip uninstall -y pennylane
	$(PYTHON) -m pip install -e . --extra-index-url https://test.pypi.org/simple
	rm -r frontend/PennyLane_Catalyst.egg-info

.PHONY: mlir llvm mhlo enzyme dialects runtime oqc oqd
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
	$(MAKE) -C runtime all

oqc:
	$(MAKE) -C frontend/catalyst/third_party/oqc/src oqc

oqd:
	$(MAKE) -C frontend/catalyst/third_party/oqd/src oqd

.PHONY: test test-runtime test-frontend lit pytest test-demos test-oqc test-oqd test-toml-spec
test: test-runtime standalone-plugin test-frontend test-demos

test-toml-spec:
	$(PYTHON) ./bin/toml-check.py $(TOML_SPECS)

test-runtime:
	$(MAKE) -C runtime test

test-mlir:
	$(MAKE) -C mlir test

test-frontend: lit pytest

test-oqc:
	$(MAKE) -C frontend/catalyst/third_party/oqc/src test

test-oqd:
	$(MAKE) -C frontend/catalyst/third_party/oqd/src test

lit: standalone-plugin
ifeq ($(ENABLE_ASAN),ON)
ifneq ($(findstring clang,$(C_COMPILER)),clang)
	@echo "Build and Test with Address Sanitizer are only supported by Clang, but provided $(C_COMPILER)"
	@exit 1
endif
endif
	@echo "check the Catalyst lit test suite"
	cmake --build $(DIALECTS_BUILD_DIR) --target check-frontend

pytest:
ifeq ($(ENABLE_ASAN),ON)
ifneq ($(findstring clang,$(C_COMPILER)),clang)
	@echo "Build and Test with Address Sanitizer are only supported by Clang, but provided $(C_COMPILER)"
	@exit 1
endif
endif
	@echo "check the Catalyst PyTest suite"
	$(ASAN_COMMAND) $(PYTHON) -m pytest frontend/test/pytest --tb=native --backend=$(TEST_BACKEND) --runbraket=$(TEST_BRAKET) $(PARALLELIZE)
	$(ASAN_COMMAND) $(PYTHON) -m pytest frontend/test/test_oqc/oqc
	$(ASAN_COMMAND) $(PYTHON) -m pytest frontend/test/test_oqd/oqd --skip-oqd=$(SKIP_OQD) $(PARALLELIZE)
ifeq ($(TEST_BRAKET), NONE)
	$(ASAN_COMMAND) $(PYTHON) -m pytest frontend/test/async_tests --tb=native --backend=$(TEST_BACKEND)
endif

test-demos:
ifeq ($(ENABLE_ASAN) $(PLATFORM),ON Darwin)
	@echo "Cannot run Jupyter Notebooks with ASAN on macOS, likely due to subprocess invocation."
	@exit 1
endif
	@echo "check the Catalyst demos"
	MDD_BENCHMARK_PRECISION=1 \
	$(ASAN_COMMAND) $(PYTHON) -m pytest demos --nbmake $(PARALLELIZE)

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
	cp $(OQC_BUILD_DIR)/backend/*.toml $(MK_DIR)/frontend/catalyst/lib/backend
	cp $(OQD_BUILD_DIR)/librtd_oqd* $(MK_DIR)/frontend/catalyst/lib
	cp $(OQD_BUILD_DIR)/backend/*.toml $(MK_DIR)/frontend/catalyst/lib/backend
	cp $(COPY_FLAGS) $(LLVM_BUILD_DIR)/lib/libmlir_float16_utils.* $(MK_DIR)/frontend/catalyst/lib
	cp $(COPY_FLAGS) $(LLVM_BUILD_DIR)/lib/libmlir_c_runner_utils.* $(MK_DIR)/frontend/catalyst/lib
	cp $(COPY_FLAGS) $(LLVM_BUILD_DIR)/lib/libmlir_async_runtime.* $(MK_DIR)/frontend/catalyst/lib

	# Copy mlir bindings & compiler driver to frontend/mlir_quantum
	mkdir -p $(MK_DIR)/frontend/mlir_quantum/dialects
	cp -R $(COPY_FLAGS) $(DIALECTS_BUILD_DIR)/python_packages/quantum/mlir_quantum/runtime $(MK_DIR)/frontend/mlir_quantum/runtime
	for file in gradient quantum _ods_common catalyst mitigation _transform; do \
		cp $(COPY_FLAGS) $(DIALECTS_BUILD_DIR)/python_packages/quantum/mlir_quantum/dialects/*$${file}* $(MK_DIR)/frontend/mlir_quantum/dialects ; \
	done
	mkdir -p $(MK_DIR)/frontend/catalyst/bin
	cp $(COPY_FLAGS) $(DIALECTS_BUILD_DIR)/bin/catalyst-cli $(MK_DIR)/frontend/catalyst/bin
	cp $(COPY_FLAGS) $(DIALECTS_BUILD_DIR)/lib/StandalonePlugin.* $(MK_DIR)/frontend/catalyst/lib
	find $(MK_DIR)/frontend -type d -name __pycache__ -exec rm -rf {} +

	$(PYTHON) -m pip wheel --no-deps . -w dist

	rm -r $(MK_DIR)/build

standalone-plugin-wheel: standalone-plugin
	mkdir -p $(MK_DIR)/standalone_plugin_wheel/standalone_plugin/lib
	cp $(COPY_FLAGS) $(DIALECTS_BUILD_DIR)/lib/StandalonePlugin.* $(MK_DIR)/standalone_plugin_wheel/standalone_plugin/lib

	$(PYTHON) -m pip wheel --no-deps $(MK_DIR)/standalone_plugin_wheel -w $(MK_DIR)/standalone_plugin_wheel/dist

	rm -r $(MK_DIR)/standalone_plugin_wheel/standalone_plugin/lib
	rm -r $(MK_DIR)/standalone_plugin_wheel/standalone_plugin.egg-info
	rm -r $(MK_DIR)/standalone_plugin_wheel/build

.PHONY: clean clean-all
clean:
	@echo "uninstall catalyst and delete all temporary and cache files"
	$(PYTHON) -m pip uninstall -y pennylane-catalyst
	rm -rf $(MK_DIR)/frontend/mlir_quantum $(MK_DIR)/frontend/catalyst/lib
	rm -rf dist __pycache__
	rm -rf .coverage coverage_html_report

clean-all: clean-frontend clean-mlir clean-runtime clean-oqc clean-oqd clean-standalone-plugin
	@echo "uninstall catalyst and delete all temporary, cache, and build files"
	$(PYTHON) -m pip uninstall -y pennylane-catalyst
	rm -rf dist __pycache__
	rm -rf .coverage coverage_html_report/

.PHONY: clean-standalone-plugin
clean-standalone-plugin:
	rm -rf  $(MK_DIR)/mlir/build/lib/StandalonePlugin.*
	rm -rf  $(MK_DIR)/mlir/standalone/build/lib/StandalonePlugin.*

.PHONY: clean-frontend
clean-frontend:
	find frontend/catalyst -name "*.so" -exec rm -v {} +

.PHONY: clean-mlir clean-dialects clean-llvm clean-mhlo clean-enzyme
clean-mlir:
	$(MAKE) -C mlir clean

clean-dialects:
	$(MAKE) -C mlir clean-dialects

clean-llvm:
	$(MAKE) -C mlir clean-llvm

clean-mhlo:
	$(MAKE) -C mlir clean-mhlo

clean-enzyme:
	$(MAKE) -C mlir clean-enzyme

.PHONY: clean-runtime clean-oqc clean-oqd
clean-runtime:
	$(MAKE) -C runtime clean

clean-oqc:
	$(MAKE) -C frontend/catalyst/third_party/oqc/src clean

clean-oqd:
	$(MAKE) -C frontend/catalyst/third_party/oqd/src clean

.PHONY: coverage coverage-frontend coverage-runtime
coverage: coverage-frontend coverage-runtime

coverage-frontend:
	@echo "Generating coverage report for the frontend"
	$(ASAN_COMMAND) $(PYTHON) -m pytest frontend/test/pytest $(PARALLELIZE) --cov=catalyst --tb=native --cov-report=$(COVERAGE_REPORT)
	$(ASAN_COMMAND) $(PYTHON) -m pytest frontend/test/test_oqc/oqc $(PARALLELIZE) --cov=catalyst --cov-append --tb=native --cov-report=$(COVERAGE_REPORT)
	$(ASAN_COMMAND) $(PYTHON) -m pytest frontend/test/test_oqd/oqd $(PARALLELIZE) --cov=catalyst --cov-append --tb=native --cov-report=$(COVERAGE_REPORT)
ifeq ($(TEST_BRAKET), NONE)
	$(ASAN_COMMAND) $(PYTHON) -m pytest frontend/test/async_tests --tb=native --backend=$(TEST_BACKEND) --tb=native
endif

coverage-runtime:
	$(MAKE) -C runtime coverage

.PHONY: standalone-plugin
standalone-plugin:
	$(MAKE) -C mlir standalone-plugin

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
