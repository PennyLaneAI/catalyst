MK_ABSPATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MK_DIR := $(dir $(MK_ABSPATH))
OQC_BUILD_DIR?=$(MK_DIR)/build
RUNTIME_BUILD_DIR?=$(MK_DIR)/../../../../runtime/build
C_COMPILER?=$(shell which clang)
CXX_COMPILER?=$(shell which clang++)

.PHONY: oqc
oqc:
	@echo "build oqc"
	cmake -G Ninja -B $(OQC_BUILD_DIR) \
		-DCMAKE_C_COMPILER=$(C_COMPILER) \
		-DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
		-DRUNTIME_BUILD_DIR=$(RUNTIME_BUILD_DIR) \

	cmake --build $(OQC_BUILD_DIR)

.PHONY: test
test:
	@echo "test the Catalyst oqc test suite"
	$(OQC_BUILD_DIR)/tests/runner_tests_oqc
