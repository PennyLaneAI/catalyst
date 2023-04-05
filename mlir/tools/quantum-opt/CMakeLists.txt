get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIROptLib
    MLIRQuantum
    quantum-transforms
    MLIRGradient
    gradient-transforms
    GmlStDialect
    GmlStPasses
GmlStDialect;GmlStTilingInterface;GmlStTilingInterfaceImpl;GmlStBufferizableOpInterface;GmlStPasses;GmlStTransforms;GmlStTestPasses;MLIRGmlStUtils;LmhloStructuredInterface;LmhloDialect;LmhloPasses;LmhloGPUDialect;MLIRHLOAnalysis;MLIRHLOTestAnalysis;MLIRBufferTransforms;MLIRHLOGPUTransforms;HloOpsCommon;MhloDialect;MhloRegisterDialects;MhloPasses;MhloToThloConversion;MhloToLhloConversion;MhloToMemrefConversion;MhloToArithmeticConversion;MhloToStandard;ChloPasses;MhloToLinalg;MhloShapeOpsToStandard;MhloToStablehlo;StablehloToMhlo;MhloScatterUtils;MhloTypeConversion;HloToLinalgUtils;StablehloBase;StablehloBroadcastUtils;ChloOps;StablehloRegister;StablehloAssemblyFormat;StablehloTypeInference;StablehloOps;VhloOps;StablehloTypeConversion;StablehloPasses;THLODialect;ThloPasses;ThloBufferizableOpInterface;MLIRMhloUtils
)

add_llvm_executable(quantum-opt quantum-opt.cpp)
target_link_libraries(quantum-opt PRIVATE ${LIBS})
llvm_update_compile_flags(quantum-opt)
mlir_check_all_link_libraries(quantum-opt)
