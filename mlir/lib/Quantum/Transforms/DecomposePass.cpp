#define DEBUG_TYPE "decompose"
#include <iostream>

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
// #include <unordered_map>
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
        std::vector<Modifiers> modifiers;
        llvm::DenseMap<mlir::Value, unsigned> qregs_map;

        getOperation()->walk([&](quantum::ExtractOp extractOp) {
            auto IdxAttr = extractOp.getIdxAttr();
            if (!IdxAttr)
                return;
            unsigned wireIndex = static_cast<unsigned>(IdxAttr.value());
            qregs_map[extractOp.getResult()] = wireIndex;
        });

        getOperation()->walk([&](quantum::CustomOp op) {
            Modifiers m;
            m.gateName = op.getGateName();
            m.adjoint = op.getAdjoint();
            m.numCtrlQubits = op.getInCtrlQubits().size();
            m.numCtrlValues = op.getInCtrlValues().size();
            modifiers.push_back(m);

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