set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name catalyst)
add_public_tablegen_target(MLIRCatalystPassIncGen)
add_mlir_doc(Passes CatalystPasses ./ -gen-pass-doc)

# The following is taken from mhlo to build the --legalize-to-std pass
set(LLVM_TARGET_DEFINITIONS stablehlo_legalize_to_standard_patterns.td)
include_directories(
    ${CATALYST_MAIN_INCLUDE_DIR}/../stablehlo)
mlir_tablegen(generated_stablehlo_legalize_to_standard.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRStablehloLegalizeToStandardIncGen)
