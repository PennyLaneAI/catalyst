// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Catalyst/Driver/CatalystLLVMTarget.h"
#include "Catalyst/Driver/Pipelines.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Catalyst/Transforms/Passes.h"
#include "Gradient/IR/GradientDialect.h"
#include "Gradient/Transforms/Passes.h"
#include "Quantum-c/Dialects.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/Transforms/Passes.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

#include "memory"

#include "mhlo/IR/register.h"
#include "mhlo/transforms/passes.h"
#include "stablehlo/dialect/Register.h"

using namespace mlir;
using namespace catalyst;

namespace {
/// Parse an MLIR module given in textual ASM representation.
OwningOpRef<ModuleOp> parseMLIRSource(MLIRContext *ctx, const char *source)
{
    auto moduleBuffer = llvm::MemoryBuffer::getMemBufferCopy(source, "jit source");
    auto sourceMgr = std::make_shared<llvm::SourceMgr>();
    sourceMgr->AddNewSourceBuffer(std::move(moduleBuffer), SMLoc());

    FallbackAsmResourceMap fallbackResourceMap;
    ParserConfig parserConfig{ctx, /*verifyAfterParse=*/true, &fallbackResourceMap};

    SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, ctx);
    return parseSourceFile<ModuleOp>(sourceMgr, parserConfig);
}

std::unique_ptr<llvm::Module> parseLLVMSource(llvm::LLVMContext &context, llvm::SMDiagnostic &err,
                                              const char *source)
{
    auto moduleBuffer = llvm::MemoryBuffer::getMemBufferCopy(source, "jit source");
    return llvm::parseIR(llvm::MemoryBufferRef(*moduleBuffer), err, context);
}

/// Register all dialects required by the Catalyst compiler.
void registerAllCatalystDialects(DialectRegistry &registry)
{
    // MLIR Core dialects
    registerAllDialects(registry);
    registerAllExtensions(registry);

    // HLO
    mhlo::registerAllMhloDialects(registry);
    stablehlo::registerAllDialects(registry);

    // Catalyst
    registry.insert<CatalystDialect>();
    registry.insert<quantum::QuantumDialect>();
    registry.insert<gradient::GradientDialect>();
}

void registerAllCatalystPasses()
{
    mlir::registerPass(catalyst::createArrayListToMemRefPass);
    mlir::registerPass(catalyst::createGradientBufferizationPass);
    mlir::registerPass(catalyst::createGradientLoweringPass);
    mlir::registerPass(catalyst::createGradientConversionPass);
    mlir::registerPass(catalyst::createQuantumBufferizationPass);
    mlir::registerPass(catalyst::createQuantumConversionPass);
    mlir::registerPass(catalyst::createEmitCatalystPyInterfacePass);
    mlir::registerPass(catalyst::createCopyGlobalMemRefPass);

    mhlo::registerAllMhloPasses();
}
} // namespace

template <typename MLIRObject> void serializeMLIRObject(MLIRObject &obj, char **dest)
{
    std::string output;
    llvm::raw_string_ostream ostream{output};
    obj.print(ostream);

    // Need to explicitly allocate the char buffer for C interop - don't forget the + 1 for the null
    // terminator
    *dest = static_cast<char *>(std::malloc(output.size() + 1));
    std::strcpy(*dest, output.c_str());
}

CatalystCReturnCode RunPassPipeline(const char *source, const char *passes, char **dest)
{
    DialectRegistry registry;
    registerAllCatalystDialects(registry);
    MLIRContext context{registry};
    OwningOpRef<ModuleOp> op = parseMLIRSource(&context, source);

    auto pm = PassManager::on<ModuleOp>(&context);
    if (!op || failed(parsePassPipeline(passes, pm))) {
        return ReturnParsingFailed;
    }

    ModuleOp moduleOp = op.get();
    if (failed(pm.run(moduleOp))) {
        return ReturnLoweringFailed;
    }

    serializeMLIRObject(moduleOp, dest);
    return ReturnOk;
}

llvm::Function *getJITFunction(llvm::Module &llvmModule)
{
    for (auto &function : llvmModule.functions()) {
        if (function.getName().starts_with("jit_")) {
            return &function;
        }
    }
    assert(false && "Failed to find JIT function in module");
}

RankedTensorType inferMLIRReturnType(MLIRContext *ctx, llvm::Type *memRefDescType,
                                     Type assumedElementType)
{
    SmallVector<int64_t> resultShape;
    auto *structType = cast<llvm::StructType>(memRefDescType);
    assert(structType->getNumElements() >= 3 &&
           "Expected MemRef descriptor struct to have at least 3 entries");
    if (structType->getNumElements() == 3) {
        // resultShape is empty
    }
    else {
        auto *arrayType = cast<llvm::ArrayType>(structType->getTypeAtIndex(3));
        size_t rank = arrayType->getNumElements();
        for (size_t i = 0; i < rank; i++) {
            resultShape.push_back(ShapedType::kDynamic);
        }
    }
    return RankedTensorType::get(resultShape, assumedElementType);
}

std::optional<SourceType> symbolizeSourceType(llvm::StringRef str)
{
    return llvm::StringSwitch<std::optional<SourceType>>(str)
        .Case("mlir", SourceMLIR)
        .Case("llvm", SourceLLVMIR)
        .Default(std::nullopt);
}

CatalystCReturnCode QuantumDriverMain(const char *source, const char *dest,
                                      const char *sourceTypeStr, FunctionData *functionData)
{
    auto sourceType = symbolizeSourceType(sourceTypeStr);
    if (!sourceType) {
        return ReturnUnrecognizedSourceType;
    }

    registerAllCatalystPasses();

    DialectRegistry registry;
    registerAllCatalystDialects(registry);
    registerLLVMTranslations(registry);
    MLIRContext context{registry};

    llvm::LLVMContext llvmContext;
    std::unique_ptr<llvm::Module> llvmModule;
    if (sourceType == SourceMLIR) {
        OwningOpRef<ModuleOp> op = parseMLIRSource(&context, source);
        if (!op) {
            return ReturnParsingFailed;
        }

        if (failed(runDefaultLowering(&context, *op))) {
            return ReturnLoweringFailed;
        }

        llvmModule = translateModuleToLLVMIR(*op, llvmContext);
        if (!llvmModule) {
            return ReturnTranslationFailed;
        }
    }
    else if (sourceType == SourceLLVMIR) {
        llvm::SMDiagnostic err;
        llvmModule = parseLLVMSource(llvmContext, err, source);
        if (!llvmModule) {
            return ReturnParsingFailed;
        }
    }

    // The user has requested that we infer the name and return type of the JIT'ed function.
    if (functionData != nullptr) {
        auto *function = getJITFunction(*llvmModule);
        functionData->functionName =
            static_cast<char *>(std::malloc(function->getName().size() + 1));
        std::strcpy(functionData->functionName, function->getName().data());

        // When inferring the return type from LLVM, assume a f64 element type.
        // This is because the LLVM pointer type is opaque and requires looking into its uses to
        // infer its type.
        auto tensorType =
            inferMLIRReturnType(&context, function->getReturnType(), Float64Type::get(&context));
        serializeMLIRObject(tensorType, &functionData->returnType);
    }

    if (failed(compileObjectFile(std::move(llvmModule), dest))) {
        return ReturnObjectCompilationFailed;
    }

    return ReturnOk;
}
