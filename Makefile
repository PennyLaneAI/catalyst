PYTHON ?= python3
C_COMPILER ?= clang
CXX_COMPILER ?= clang++
BLACKVERSIONMAJOR := $(shell black --version 2> /dev/null | head -n1 | awk '{ print $$2 }' | cut -d. -f1)
BLACKVERSIONMAJOR := $(if $(BLACKVERSIONMAJOR),$(BLACKVERSIONMAJOR),0)
BLACKVERSIONMINOR := $(shell black --version 2> /dev/null | head -n1 | awk '{ print $$2 }' | cut -d. -f2)
BLACKVERSIONMINOR := $(if $(BLACKVERSIONMINOR),$(BLACKVERSIONMINOR),0)
MK_ABSPATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MK_DIR := $(dir $(MK_ABSPATH))
LLVM_BUILD_DIR ?= $(MK_DIR)/mlir/llvm-project/build
MHLO_BUILD_DIR ?= $(MK_DIR)/mlir/mlir-hlo/build
DIALECTS_BUILD_DIR ?= $(MK_DIR)/mlir/build
RT_BUILD_DIR ?= $(MK_DIR)/runtime/build
ENZYME_BUILD_DIR ?= $(MK_DIR)/mlir/Enzyme/build
COVERAGE_REPORT ?= term-missing
TEST_BACKEND ?= "lightning.qubit"
TEST_BRAKET ?= NONE
ENABLE_ASAN ?= OFF
COPY_FLAGS = $(shell python -c "import platform; print('--dereference' if platform.system() == 'Linux' else '')")

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  all                to build and install all Catalyst modules and its MLIR dependencies"
	@echo "  frontend           to install Catalyst Frontend"
	@echo "  mlir               to build MLIR and custom Catalyst dialects"
	@echo "  runtime            to build Catalyst Runtime with PennyLane-Lightning"
	@echo "  dummy_device       needed for frontend tests"
	@echo "  test               to run the Catalyst test suites"
	@echo "  docs               to build the documentation for Catalyst"
	@echo "  clean              to uninstall Catalyst and delete all temporary and cache files"
	@echo "  clean-all          to uninstall Catalyst and delete all temporary, cache, and build files"
	@echo "  clean-docs         to delete all built documentation"
	@echo "  coverage           to generate a coverage report"
	@echo "  format [check=1]   to apply C++ and Python formatter; use with 'check=1' to check instead of modify (requires black, pylint and clang-format)"
	@echo "  format [version=?] to apply C++ and Python formatter; use with 'version={version}' to run clang-format-{version} instead of clang-format"


.PHONY: all
all: runtime mlir frontend

.PHONY: frontend
frontend:
	@echo "install Catalyst Frontend"
	$(PYTHON) -m pip install -e .

.PHONY: mlir llvm mhlo enzyme dialects runtime
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

dummy_device:
	$(MAKE) -C runtime dummy_device

.PHONY: test test-runtime test-frontend lit pytest test-demos
test: test-runtime test-frontend test-demos

test-runtime:
	$(MAKE) -C runtime test

test-frontend: lit pytest

lit:
	@echo "check the Catalyst lit test suite"
	cmake --build $(DIALECTS_BUILD_DIR) --target check-frontend

pytest:
	@echo "check the Catalyst PyTest suite"
ifeq ($(ENABLE_ASAN), ON)
ifneq ($(findstring clang,$(C_COMPILER)), clang)
	@echo "Build and Test with Address Sanitizer are only supported by Clang, but provided $(C_COMPILER)"
	@exit 1
endif
	ASAN_OPTIONS=detect_leaks=0 \
	LD_PRELOAD="$(shell clang  -print-file-name=libclang_rt.asan-x86_64.so)" \
	$(PYTHON) -m pytest frontend/test/pytest --tb=native --backend=$(TEST_BACKEND) --runbraket=$(TEST_BRAKET) -n auto
else
	$(PYTHON) -m pytest frontend/test/pytest --tb=native --backend=$(TEST_BACKEND) --runbraket=$(TEST_BRAKET) -n auto
endif

test-demos:
	@echo "check the Catalyst demos"
ifeq ($(ENABLE_ASAN), ON)
ifneq ($(findstring clang,$(C_COMPILER)), clang)
	@echo "Build and Test with Address Sanitizer are only supported by Clang, but provided $(C_COMPILER)"
	@exit 1
endif
	ASAN_OPTIONS=detect_leaks=0 \
	LD_PRELOAD="$(shell clang  -print-file-name=libclang_rt.asan-x86_64.so)" \
	MDD_BENCHMARK_PRECISION=1 \
	$(PYTHON) -m pytest demos/*.ipynb --nbmake -n auto
else
	MDD_BENCHMARK_PRECISION=1 \
	$(PYTHON) -m pytest demos/*.ipynb --nbmake -n auto
endif

wheel:
	echo "INSTALLED = True" > $(MK_DIR)/frontend/catalyst/_configuration.py

	# Copy libs to frontend/catalyst/lib
	mkdir -p $(MK_DIR)/frontend/catalyst/lib
	cp $(RT_BUILD_DIR)/lib/librt_backend.* $(MK_DIR)/frontend/catalyst/lib
	cp $(RT_BUILD_DIR)/lib/librt_capi.* $(MK_DIR)/frontend/catalyst/lib
	cp $(COPY_FLAGS) $(LLVM_BUILD_DIR)/lib/libmlir_float16_utils.* $(MK_DIR)/frontend/catalyst/lib
	cp $(COPY_FLAGS) $(LLVM_BUILD_DIR)/lib/libmlir_c_runner_utils.* $(MK_DIR)/frontend/catalyst/lib

	# Copy mlir bindings & compiler driver to frontend/mlir_quantum
	mkdir -p $(MK_DIR)/frontend/mlir_quantum/dialects
	cp -R $(COPY_FLAGS) $(DIALECTS_BUILD_DIR)/python_packages/quantum/mlir_quantum/runtime $(MK_DIR)/frontend/mlir_quantum/runtime
	for file in gradient quantum _ods_common ; do \
		cp $(COPY_FLAGS) $(DIALECTS_BUILD_DIR)/python_packages/quantum/mlir_quantum/dialects/*$${file}* $(MK_DIR)/frontend/mlir_quantum/dialects ; \
	done
	cp $(COPY_FLAGS) $(DIALECTS_BUILD_DIR)/python_packages/quantum/mlir_quantum/compiler_driver.so $(MK_DIR)/frontend/mlir_quantum/
	find $(MK_DIR)/frontend -type d -name __pycache__ -exec rm -rf {} +

	$(PYTHON) $(MK_DIR)/setup.py bdist_wheel

.PHONY: clean clean-all
clean:
	@echo "uninstall catalyst and delete all temporary and cache files"
	$(PYTHON) -m pip uninstall -y pennylane-catalyst
	rm -rf $(MK_DIR)/frontend/mlir_quantum $(MK_DIR)/frontend/catalyst/lib
	rm -rf dist __pycache__
	rm -rf .coverage coverage_html_report

clean-all:
	@echo "uninstall catalyst and delete all temporary, cache files"
	$(PYTHON) -m pip uninstall -y pennylane-catalyst
	rm -rf dist __pycache__
	rm -rf .coverage coverage_html_report/
	$(MAKE) -C mlir clean
	$(MAKE) -C runtime clean

.PHONY: coverage coverage-frontend coverage-runtime
coverage: coverage-frontend coverage-runtime

coverage-frontend:
	@echo "Generating coverage report for the frontend"
	$(PYTHON) -m pytest frontend/test/pytest -n auto --cov=catalyst --tb=native --cov-report=$(COVERAGE_REPORT)

coverage-runtime:
	$(MAKE) -C runtime coverage

.PHONY: examples-runtime
examples-runtime:
	$(MAKE) -C runtime examples

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
ifdef check
	python3 ./bin/format.py --check $(if $(version:-=),--cfversion $(version)) ./frontend/catalyst/utils
	black --check --verbose .
	isort --check --diff .
else
	python3 ./bin/format.py $(if $(version:-=),--cfversion $(version)) ./frontend/catalyst/utils
	black .
	isort .
endif
	pylint frontend

.PHONY: docs clean-docs
docs:
	$(MAKE) -C doc html

clean-docs:
	$(MAKE) -C doc clean
