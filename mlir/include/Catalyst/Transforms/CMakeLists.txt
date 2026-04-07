set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Catalyst)
add_public_tablegen_target(MLIRCatalystPassIncGen
    DEPENDS ExternalStablehloLib
)
add_mlir_doc(Passes CatalystPasses ./ -gen-pass-doc)
