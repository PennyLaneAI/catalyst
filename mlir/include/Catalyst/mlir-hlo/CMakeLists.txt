# The following is taken from mhlo to build the --legalize-to-std pass
set(LLVM_TARGET_DEFINITIONS mhlo_legalize_to_standard_patterns.td)
include_directories(
    ${CATALYST_MAIN_INCLUDE_DIR}/../mlir-hlo)
mlir_tablegen(generated_mhlo_legalize_to_standard.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRMhloLegalizeToStandardIncGen)
