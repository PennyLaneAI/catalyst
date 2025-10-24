C_COMPILER?=$(shell which clang)
CXX_COMPILER?=$(shell which clang++)
NPROC?=$(shell python3 -c "import os; print(os.cpu_count())")

MK_ABSPATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MK_DIR := $(dir $(MK_ABSPATH))
FTQC_BUILD_DIR?=$(MK_DIR)/build
RT_BUILD_DIR?=$(MK_DIR)/../../../../../runtime/build

.PHONY: configure
configure:
	@echo "Configure FTQC Runtime"

	cmake -G Ninja -B $(FTQC_BUILD_DIR) \
		-DCMAKE_C_COMPILER=$(C_COMPILER) \
		-DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
		-DRUNTIME_BUILD_DIR=$(RT_BUILD_DIR)

$(FTQC_BUILD_DIR)/librtd_ftqc.so: configure
	cmake --build $(FTQC_BUILD_DIR) --target rtd_ftqc -j$(NPROC)

.PHONY: ftqc
ftqc: $(FTQC_BUILD_DIR)/librtd_ftqc.so

.PHONY: clean
clean:
	@echo "clean build files"
	rm -rf $(FTQC_BUILD_DIR)