#define DEBUG_TYPE "to_ppr"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "QEC/IR/QECDialect.h"
#include "QEC/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::quantum;
using namespace catalyst::qec;

namespace catalyst {
namespace qec {

#define GEN_PASS_DEF_COUNTPPMSPECSPASS
#define GEN_PASS_DECL_COUNTPPMSPECSPASS
#include "QEC/Transforms/Passes.h.inc"


struct CountPPMSpecsPass : impl::CountPPMSpecsPassBase<CountPPMSpecsPass> {
    using CountPPMSpecsPassBase::CountPPMSpecsPassBase;

    void runOnOperation() final
    {
        auto ctx = &getContext();
        auto module = getOperation();

        // Phase 1: Convert Clifford+T to PPR representation
        {
            ConversionTarget target(*ctx);
            target.addIllegalDialect<quantum::QuantumDialect>();
            target.addLegalOp<quantum::InitializeOp, quantum::FinalizeOp>();
            target.addLegalOp<quantum::DeviceInitOp, quantum::DeviceReleaseOp>();
            target.addLegalOp<quantum::AllocOp, quantum::DeallocOp>();
            target.addLegalOp<quantum::InsertOp, quantum::ExtractOp>();
            target.addLegalDialect<qec::QECDialect>();

            RewritePatternSet patterns(ctx);
            populateCliffordTToPPRPatterns(patterns);

            if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
                return signalPassFailure();
            }
            else {
                llvm::outs() << "TEST for Count PPM Specs\n";
            }
        }
    }
};

} // namespace qec

/// Create a pass for lowering operations in the `QECDialect`.
std::unique_ptr<mlir::Pass> createCountPPMSpecsPass()
{
    return std::make_unique<qec::CountPPMSpecsPass>();
}

} // namespace catalyst
