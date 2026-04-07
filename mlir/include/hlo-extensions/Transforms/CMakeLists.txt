set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Stablehlo)
add_public_tablegen_target(STABLEHLOCatalystPassIncGen
    DEPENDS ExternalStablehloLib
)

# The following is modified from the
#   tensorflow/mlir-hlo
# repository at
#   https://github.com/tensorflow/mlir-hlo/blob/a5529d99fc4d1132b0c282a053d26c11e6636b3a/mhlo/transforms/CMakeLists.txt
# to build the rewrite patterns for the --stablehlo-legalize-to-std pass
set(LLVM_TARGET_DEFINITIONS stablehlo_legalize_to_standard_patterns.td)
include_directories(
    ${CATALYST_MAIN_INCLUDE_DIR}/../stablehlo)
mlir_tablegen(generated_stablehlo_legalize_to_standard.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRStablehloLegalizeToStandardIncGen)
