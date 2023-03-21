COMPILER := clang

BUILD_DIR := $(if ,$(BUILD_DIR),../build)

QIR_LINK := -L$(BUILD_DIR)/lib/capi -L$(BUILD_DIR)/lib/backend -L$(LLVM_BUILD_DIR)/lib
QIR_RPATH := -Wl,-rpath,$(BUILD_DIR)/lib/capi:$(BUILD_DIR)/lib/backend:$(LLVM_BUILD_DIR)/lib
QIR_LIBS := -lrt_capi -lrt_backend -lmlir_c_runner_utils

.PHONY: all
all: grad_qfunc probs_qfunc state_qfunc

grad_qfunc:
	$(COMPILER) $@.ll -o $@.bin $(QIR_LINK) $(QIR_LIBS) $(QIR_RPATH)
	./$@.bin

probs_qfunc:
	$(COMPILER) $@.ll -o $@.bin $(QIR_LINK) $(QIR_LIBS) $(QIR_RPATH)
	./$@.bin

state_qfunc:
	$(COMPILER) $@.ll -o $@.bin $(QIR_LINK) $(QIR_LIBS) $(QIR_RPATH)
	./$@.bin

.PHONY: clean
clean:
	rm -rf *.bin
