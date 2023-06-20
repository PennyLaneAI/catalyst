#include "Quantum-c/Dialects.h"

#include "memory"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/SourceMgr.h"

#include "mhlo/IR/register.h"
#include "mhlo/transforms/passes.h"
#include "stablehlo/dialect/Register.h"

using namespace mlir;
using namespace llvm;

OwningOpRef<ModuleOp> parseSource(MLIRContext *ctx, const char *source)
{
    auto moduleBuffer = MemoryBuffer::getMemBufferCopy(source, "jit source");
    auto sourceMgr = std::make_shared<llvm::SourceMgr>();
    sourceMgr->AddNewSourceBuffer(std::move(moduleBuffer), SMLoc());

    FallbackAsmResourceMap fallbackResourceMap;
    ParserConfig parserConfig{ctx, /*verifyAfterParse=*/true, &fallbackResourceMap};

    SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, ctx);
    return parseSourceFile<ModuleOp>(sourceMgr, parserConfig);
}

void QuantumDriverMain(const char *source)
{
    DialectRegistry registry;
    registerAllDialects(registry);
    mhlo::registerAllMhloDialects(registry);
    stablehlo::registerAllDialects(registry);

    MLIRContext context{registry};
    OwningOpRef<ModuleOp> op = parseSource(&context, source);

    if (!op) {
        return;
    }

    auto pm = PassManager::on<ModuleOp>(&context);
    pm.addNestedPass<func::FuncOp>(
        mhlo::createLegalizeHloToLinalgPass(/*enable-primitive-ops=*/false));
    pm.addPass(createCanonicalizerPass());

    if (failed(pm.run(*op))) {
        return;
    }

    op->dump();
}
