# Taken from llvm/tools/llc/CMakeLists.txt
# TODO: May be possible to simplify linked libs
set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  IRReader
  MC
  MIRParser
  Remarks
  ScalarOpts
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
  Vectorize
  )

add_mlir_library(CatalystCompilerDriver
    CatalystLLVMTarget.cpp
    Pipelines.cpp

    LINK_LIBS PRIVATE
)
