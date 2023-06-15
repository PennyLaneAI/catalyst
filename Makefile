PYTHON := python3 -m
BLACKVERSIONMAJOR := $(shell black --version 2> /dev/null | head -n1 | awk '{ print $$2 }' | cut -d. -f1)
BLACKVERSIONMAJOR := $(if $(BLACKVERSIONMAJOR),$(BLACKVERSIONMAJOR),0)
BLACKVERSIONMINOR := $(shell black --version 2> /dev/null | head -n1 | awk '{ print $$2 }' | cut -d. -f2)
BLACKVERSIONMINOR := $(if $(BLACKVERSIONMINOR),$(BLACKVERSIONMINOR),0)
MK_ABSPATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MK_DIR := $(dir $(MK_ABSPATH))
DIALECTS_BUILD_DIR ?= $(MK_DIR)/mlir/build
COVERAGE_REPORT ?= term-missing
TEST_BACKEND ?= "lightning.qubit"

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  all                to build and install all Catalyst modules and its MLIR dependencies"
	@echo "  frontend           to install Catalyst Frontend"
	@echo "  mlir               to build MLIR and custom Catalyst dialects"
	@echo "  runtime            to build Catalyst Runtime with PennyLane-Lightning"
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
	$(PYTHON) pip install -e .

.PHONY: mlir llvm mhlo dialects runtime qir enzyme
mlir:
	$(MAKE) -C mlir all

llvm:
	$(MAKE) -C mlir llvm

mhlo:
	$(MAKE) -C mlir mhlo

enzyme:
	$(MAKE) -C mlir llvm enzyme

dialects:
	$(MAKE) -C mlir dialects

runtime:
	$(MAKE) -C runtime all

qir:
	$(MAKE) -C runtime qir

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
	$(PYTHON) pytest frontend/test/pytest --tb=native --backend=$(TEST_BACKEND) -n auto

test-demos:
	@echo "check the Catalyst demos"
	MDD_BENCHMARK_PRECISION=1 \
	$(PYTHON) pytest demos/*.ipynb --nbmake -n auto

.PHONY: clean clean-all
clean:
	@echo "uninstall catalyst and delete all temporary and cache files"
	$(PYTHON) pip uninstall -y pennylane-catalyst
	rm -rf dist __pycache__
	rm -rf .coverage coverage_html_report

clean-all:
	@echo "uninstall catalyst and delete all temporary, cache files"
	$(PYTHON) pip uninstall -y pennylane-catalyst
	rm -rf dist __pycache__
	rm -rf .coverage coverage_html_report/
	$(MAKE) -C mlir clean
	$(MAKE) -C runtime clean

.PHONY: coverage coverage-frontend coverage-runtime
coverage: coverage-frontend coverage-runtime

coverage-frontend:
	@echo "Generating coverage report for the frontend"
	$(PYTHON) pytest frontend/test/pytest -n auto --cov=catalyst --tb=native --cov-report=$(COVERAGE_REPORT)

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
	black --check --verbose .
	isort --check --diff .
else
	black .
	isort .
endif
	pylint frontend

.PHONY: docs clean-docs
docs:
	$(MAKE) -C doc html

clean-docs:
	$(MAKE) -C doc clean
