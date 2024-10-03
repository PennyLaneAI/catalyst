PYTHON?=$(shell which python3)
C_COMPILER?=$(shell which clang)
CXX_COMPILER?=$(shell which clang++)
NPROC?=$(shell python3 -c "import os; print(os.cpu_count())")

MK_ABSPATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MK_DIR := $(dir $(MK_ABSPATH))
FRONTEND_BUILD_DIR?=$(MK_DIR)/build
BUILD_TYPE?=RelWithDebInfo


.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  all    to build Catalyst Frontend"
	@echo "  clean  to delete all temporary, cache, and build files"

.PHONY: all
all: frontend

.PHONY: configure
configure:
	@echo "Configure Catalyst Frontend"

	cmake -G Ninja -B $(FRONTEND_BUILD_DIR) . \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(FRONTEND_BUILD_DIR)/lib \
		-DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
		$(CMAKE_ARGS) \

.PHONY: frontend
frontend: configure
	cmake --build $(FRONTEND_BUILD_DIR) --target install -j$(NPROC) -v

.PHONY: clean
clean:
	@echo "clean build files"
	rm -rf $(FRONTEND_BUILD_DIR)
