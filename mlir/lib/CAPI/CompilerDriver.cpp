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
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/SourceMgr.h"

#include "memory"

#include "mhlo/IR/register.h"
#include "mhlo/transforms/passes.h"
#include "stablehlo/dialect/Register.h"

using namespace mlir;
using namespace catalyst;

namespace {
/// Parse an MLIR module given in textual ASM representation.
OwningOpRef<ModuleOp> parseSource(MLIRContext *ctx, const char *source)
{
    auto moduleBuffer = llvm::MemoryBuffer::getMemBufferCopy(source, "jit source");
    auto sourceMgr = std::make_shared<llvm::SourceMgr>();
    sourceMgr->AddNewSourceBuffer(std::move(moduleBuffer), SMLoc());

    FallbackAsmResourceMap fallbackResourceMap;
    ParserConfig parserConfig{ctx, /*verifyAfterParse=*/true, &fallbackResourceMap};

    SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, ctx);
    return parseSourceFile<ModuleOp>(sourceMgr, parserConfig);
}

/// Register all dialects required by the Catalyst compiler.
void registerAllCatalystDialects(DialectRegistry &registry)
{
    // MLIR Core dialects
    registerAllDialects(registry);

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

CatalystCReturnCode RunPassPipeline(const char *source, const char *passes, char **dest)
{
    DialectRegistry registry;
    registerAllCatalystDialects(registry);
    MLIRContext context{registry};
    OwningOpRef<ModuleOp> op = parseSource(&context, source);

    auto pm = PassManager::on<ModuleOp>(&context);
    if (!op || failed(parsePassPipeline(passes, pm))) {
        return ReturnParsingFailed;
    }

    if (failed(pm.run(*op))) {
        return ReturnLoweringFailed;
    }

    std::string output;
    llvm::raw_string_ostream ostream{output};
    op->print(ostream);

    // Need to explicitly allocate the char buffer for C interop - don't forget the + 1 for the null
    // terminator
    *dest = static_cast<char *>(std::malloc(output.size() + 1));
    std::strcpy(*dest, output.c_str());
    return ReturnOk;
}

CatalystCReturnCode QuantumDriverMain(const char *source, const char *dest, FunctionData *functionData)
{
    registerAllCatalystPasses();

    DialectRegistry registry;
    registerAllCatalystDialects(registry);
    registerLLVMTranslations(registry);
    MLIRContext context{registry};
    OwningOpRef<ModuleOp> op = parseSource(&context, source);
    if (!op) {
        return ReturnParsingFailed;
    }

    if (failed(runDefaultLowering(&context, *op))) {
        return ReturnLoweringFailed;
    }

    llvm::LLVMContext llvmContext;
    auto llvmModule = translateModuleToLLVMIR(*op, llvmContext);
    if (!llvmModule) {
        return ReturnTranslationFailed;
    }

    if (functionData != nullptr) {
        for (auto &function : llvmModule->functions()) {
            if (function.getName().starts_with("@jit")) {
                llvm::errs() << "found jit function: " << function.getName() << "\n";
            }
        }
    }
    if (failed(compileObjectFile(std::move(llvmModule), dest))) {
        return ReturnObjectCompilationFailed;
    }

    return ReturnOk;
}
