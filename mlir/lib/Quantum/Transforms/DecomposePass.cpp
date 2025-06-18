#define DEBUG_TYPE "decompose"
#include <iostream>

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DenseMap.h"

struct Modifiers {
    std::string gateName;
    bool adjoint;
    unsigned numCtrlQubits;
    unsigned numCtrlValues;
};

using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace catalyst {
#define GEN_PASS_DEF_DECOMPOSEPASS
#define GEN_PASS_DECL_DECOMPOSEPASS
#include "Quantum/Transforms/Passes.h.inc"

struct DecomposeCustomOp : public OpRewritePattern<quantum::CustomOp> {
    using mlir::OpRewritePattern<quantum::CustomOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(quantum::CustomOp op, PatternRewriter &rewriter) const override
    {
        // TODO: Implement the actual decomposition logic.
        return failure();
    }
};

struct DecomposePass : public impl::DecomposePassBase<DecomposePass> {
    using impl::DecomposePassBase<DecomposePass>::DecomposePassBase;

    void runOnOperation() override
    {
        llvm::DenseMap<mlir::Value, unsigned> qregs_map;
        std::vector<Modifiers> CollectedCustomOps;
        // We optionally reserve space to avoid memory reallocations
        CollectedCustomOps.reserve(32);

        auto module = getOperation();

        module->walk([&](quantum::ExtractOp extractOp) {
            auto IdxAttr = extractOp.getIdxAttr();
            if (!IdxAttr)
                return;
            unsigned wireIndex = static_cast<unsigned>(IdxAttr.value());
            qregs_map[extractOp.getResult()] = wireIndex;
        });

        module->walk([&](quantum::CustomOp op) {
            Modifiers collected_op;
            collected_op.gateName = op.getGateName();
            collected_op.adjoint = op.getAdjoint();
            collected_op.numCtrlQubits = op.getInCtrlQubits().size();
            collected_op.numCtrlValues = op.getInCtrlValues().size();
            CollectedCustomOps.push_back(collected_op);

            auto inQubits = op.getInQubits();
            auto outQubits = op.getOutQubits();
            assert(inQubits.size() == outQubits.size());

            for (unsigned i = 0; i < inQubits.size(); ++i) {
                auto it = qregs_map.find(inQubits[i]);
                if (it == qregs_map.end())
                    continue;
                qregs_map[outQubits[i]] = it->second;
            }
        });

        RewritePatternSet patterns(&getContext());
        patterns.add<DecomposeCustomOp>(&getContext());

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createDecomposePass() { return std::make_unique<DecomposePass>(); }

} // namespace catalyst