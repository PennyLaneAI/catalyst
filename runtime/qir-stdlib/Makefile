.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  all                to build the QIR-stdlib Rust crate"
	@echo "  qir                to build the QIR-stdlib Rust crate using cargo; note it requires the llvm-tools-preview rustup component"
	@echo "  clean              to delete all temporary, cache, and build files"

.PHONY: all
all: qir

target:
	@echo "build the QIR-stdlib Rust crate using cargo;"
	@echo "this cargo requires the llvm-tools-preview rustup component"
	cargo build --release

.PHONY: qir
qir: target

.PHONY: clean
clean:
	cargo clean
	rm -rf Cargo.lock
