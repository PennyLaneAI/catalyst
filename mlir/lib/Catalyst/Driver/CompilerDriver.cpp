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

#include "Catalyst/Driver/CompilerDriver.h"
#include "Catalyst/Driver/CatalystLLVMTarget.h"
#include "Catalyst/Driver/Pipelines.h"
#include "Catalyst/Driver/Support.h"

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
/// Parse an MLIR module given in textual ASM representation. Any errors during parsing will be
/// output to diagnosticStream.
OwningOpRef<ModuleOp> parseMLIRSource(MLIRContext *ctx, StringRef source, StringRef moduleName,
                                      raw_ostream &diagnosticStream)
{
    auto moduleBuffer = llvm::MemoryBuffer::getMemBufferCopy(source, moduleName);
    auto sourceMgr = std::make_shared<llvm::SourceMgr>();
    sourceMgr->AddNewSourceBuffer(std::move(moduleBuffer), SMLoc());

    FallbackAsmResourceMap fallbackResourceMap;
    ParserConfig parserConfig{ctx, /*verifyAfterParse=*/true, &fallbackResourceMap};

    SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, ctx, diagnosticStream);
    return parseSourceFile<ModuleOp>(sourceMgr, parserConfig);
}

/// Parse an LLVM module given in textual representation. Any parse errors will be output to
/// the provided SMDiagnostic.
std::unique_ptr<llvm::Module> parseLLVMSource(llvm::LLVMContext &context, StringRef source,
                                              StringRef moduleName, llvm::SMDiagnostic &err)
{
    auto moduleBuffer = llvm::MemoryBuffer::getMemBufferCopy(source, moduleName);
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
} // namespace

template <typename MLIRObject> std::string serializeMLIRObject(MLIRObject &obj)
{
    std::string output;
    llvm::raw_string_ostream ostream{output};
    obj.print(ostream);
    return output;
}

FailureOr<std::string> RunPassPipeline(StringRef source, StringRef passes)
{
    DialectRegistry registry;
    registerAllCatalystDialects(registry);
    MLIRContext context{registry};
    OwningOpRef<ModuleOp> op = parseMLIRSource(&context, source, "jit source", llvm::errs());

    auto pm = PassManager::on<ModuleOp>(&context);
    if (!op || failed(parsePassPipeline(passes, pm))) {
        return failure();
    }

    ModuleOp moduleOp = op.get();
    if (failed(pm.run(moduleOp))) {
        return failure();
    }

    return serializeMLIRObject(moduleOp);
}

FailureOr<llvm::Function *> getJITFunction(MLIRContext *ctx, llvm::Module &llvmModule)
{
    Location loc = NameLoc::get(StringAttr::get(ctx, llvmModule.getName()));
    for (auto &function : llvmModule.functions()) {
        emitRemark(loc) << "Found LLVM function: " << function.getName() << "\n";
        if (function.getName().starts_with("jit_")) {
            return &function;
        }
    }
    emitError(loc, "Failed to find JIT function");
    return failure();
}

LogicalResult inferMLIRReturnTypes(MLIRContext *ctx, llvm::Type *returnType,
                                   Type assumedElementType,
                                   SmallVectorImpl<RankedTensorType> &inferredTypes)
{
    auto inferSingleMemRef = [&](llvm::StructType *descriptorType) {
        SmallVector<int64_t> resultShape;
        assert(descriptorType->getNumElements() >= 3 &&
               "Expected MemRef descriptor struct to have at least 3 entries");
        if (descriptorType->getNumElements() == 3) {
            // resultShape is empty
        }
        else {
            auto *arrayType = cast<llvm::ArrayType>(descriptorType->getTypeAtIndex(3));
            size_t rank = arrayType->getNumElements();
            for (size_t i = 0; i < rank; i++) {
                resultShape.push_back(ShapedType::kDynamic);
            }
        };
        return RankedTensorType::get(resultShape, assumedElementType);
    };
    if (returnType->isVoidTy()) {
        return success();
    }
    if (auto *structType = dyn_cast<llvm::StructType>(returnType)) {
        // The return type could be a single memref descriptor or a struct of multiple memref
        // descriptors.
        if (isa<llvm::StructType>(structType->getElementType(0))) {
            for (size_t i = 0; i < structType->getNumElements(); i++) {
                inferredTypes.push_back(
                    inferSingleMemRef(cast<llvm::StructType>(structType->getTypeAtIndex(i))));
            }
        }
        else {
            // Assume the function returns a single memref
            inferredTypes.push_back(inferSingleMemRef(structType));
        }
        return success();
    }
    return failure();
}

LogicalResult QuantumDriverMain(const CompilerSpec &spec,
                                const CompilerOptions &options,
                                FunctionAttributes &inferredData)
{
    registerAllCatalystPasses();
    mhlo::registerAllMhloPasses();

    DialectRegistry registry;
    registerAllCatalystDialects(registry);
    registerLLVMTranslations(registry);
    MLIRContext *ctx = options.ctx;
    ctx->appendDialectRegistry(registry);
    ctx->disableMultithreading();
    ScopedDiagnosticHandler scopedHandler(
        ctx, [&](Diagnostic &diag) { diag.print(options.diagnosticStream); });

    llvm::LLVMContext llvmContext;
    std::unique_ptr<llvm::Module> llvmModule;

    // First attempt to parse the input as an MLIR module.
    OwningOpRef<ModuleOp> op =
        parseMLIRSource(ctx, options.source, options.moduleName, options.diagnosticStream);
    if (op) {
        if (failed(runDefaultLowering(spec, options, *op))) {
            return failure();
        }

        llvmModule = translateModuleToLLVMIR(*op, llvmContext);
        if (!llvmModule) {
            return failure();
        }

        if (options.keepIntermediate) {
            if (failed(catalyst::dumpToFile(options, "llvm_ir.ll", *llvmModule))) {
                return failure();
            }
        }
    }
    else {
        // If parsing as an MLIR module failed, attempt to parse as an LLVM IR module.
        llvm::SMDiagnostic err;
        llvmModule = parseLLVMSource(llvmContext, options.source, options.moduleName, err);
        if (!llvmModule) {
            // If both MLIR and LLVM failed to parse, exit.
            err.print(options.moduleName.data(), options.diagnosticStream);
            return failure();
        }
    }

    // For feature parity with the previous driver, always return the LLVM module.
    llvm::raw_string_ostream llvmIRStream(inferredData.llvmir);
    llvmIRStream << *llvmModule;

    // Attempt to infer the name and return type of the module from LLVM IR. This information is
    // required when executing a module given as textual IR.
    auto function = getJITFunction(options.ctx, *llvmModule);
    if (succeeded(function)) {
        inferredData.functionName = function.value()->getName().str();

        // When inferring the return type from LLVM, assume a f64
        // element type. This is because the LLVM pointer type is
        // opaque and requires looking into its uses to infer its type.
        SmallVector<RankedTensorType> returnTypes;
        if (failed(inferMLIRReturnTypes(ctx, function.value()->getReturnType(),
                                        Float64Type::get(ctx), returnTypes))) {
            // Inferred return types are only required when compiling from textual IR. This
            // inference failing is not a problem when compiling from Python.
        }
        llvm::raw_string_ostream returnTypeStream(inferredData.returnType);
        llvm::interleaveComma(returnTypes, returnTypeStream, [](RankedTensorType tensorType) {
            return serializeMLIRObject(tensorType);
        });
    }

    if (failed(compileObjectFile(std::move(llvmModule), options.getObjectFile()))) {
        return failure();
    }
    return success();
}
