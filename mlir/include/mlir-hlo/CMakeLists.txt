set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name mlir-hlo)
add_public_tablegen_target(MLIRHLOCatalystPassIncGen)

# The following is modified from the
#   tensorflow/mlir-hlo
# repository at
#   https://github.com/tensorflow/mlir-hlo/blob/a5529d99fc4d1132b0c282a053d26c11e6636b3a/mhlo/transforms/CMakeLists.txt
# to build the rewrite patterns for the --mhlo-legalize-to-std pass
set(LLVM_TARGET_DEFINITIONS mhlo_legalize_to_standard_patterns.td)
include_directories(
    ${CATALYST_MAIN_INCLUDE_DIR}/../mlir-hlo)
mlir_tablegen(generated_mhlo_legalize_to_standard.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRMhloLegalizeToStandardIncGen)
