
add_mlir_translation_library(
  MlirToQasm3Translation
  TranslateToQASM3.cpp
  
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRQuantum
)
