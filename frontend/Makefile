PYTHON?=$(shell which python3)
C_COMPILER?=$(shell which clang)
CXX_COMPILER?=$(shell which clang++)
COMPILER_LAUNCHER?=$(shell which ccache)
NPROC?=$(shell python3 -c "import os; print(os.cpu_count())")

MK_ABSPATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MK_DIR := $(dir $(MK_ABSPATH))
FRONTEND_BUILD_DIR?=$(MK_DIR)/build


.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  all    to build Catalyst Frontend"
	@echo "  clean  to delete all temporary, cache, and build files"

.PHONY: all
all: frontend

.PHONY: frontend
frontend:
	cmake -G Ninja -B $(FRONTEND_BUILD_DIR) .
	cmake --build $(FRONTEND_BUILD_DIR) --target install -j$(NPROC)

.PHONY: clean
clean:
	@echo "clean build files"
	rm -rf $(FRONTEND_BUILD_DIR)
