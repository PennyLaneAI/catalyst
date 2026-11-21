// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "to-pbc"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "PBC/IR/PBCDialect.h"
#include "PBC/IR/PBCOps.h"
#include "PBC/Transforms/Passes.h"
#include "PBC/Transforms/Patterns.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Passes.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::pbc;

namespace catalyst {
namespace pbc {

#define GEN_PASS_DECL_TOPBCPASS
#define GEN_PASS_DEF_TOPBCPASS
#include "PBC/Transforms/Passes.h.inc"

namespace {

LogicalResult runToPBC(Operation *module, MLIRContext *ctx) {
    // Remove unused decomposition rules before lowering quantum.ctrl and quantum.adjoint regions
    // (`modifiers-lowering`) so dead rule bodies are ignored and modifier regions are reduced to
    // op-level modifiers before PBC conversion.
    OpPassManager pm("builtin.module");
    pm.addPass(mlir::createSymbolDCEPass());
    pm.addPass(quantum::createModifiersLoweringPass());
    if (failed(runPipeline(pm, module))) {
        return failure();
    }

    ConversionTarget target(*ctx);
    target.addLegalDialect<pbc::PBCDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();

    // Allowlist of Quantum ops that may stay in the PBC representation: runtime and device
    // management, qubit/register handling, uncontrolled global phase, and the terminal observables
    // and measurement processes. Gates, mid-circuit measurements, and any remaining modifier
    // regions must be converted.
    target.addDynamicallyLegalDialect<quantum::QuantumDialect>([](Operation *op) {
        if (auto gphase = dyn_cast<quantum::GlobalPhaseOp>(op)) {
            return gphase.getCtrlQubitOperands().empty();
        }
        return isa<quantum::InitializeOp, quantum::FinalizeOp, quantum::DeviceInitOp,
                   quantum::DeviceReleaseOp, quantum::NumQubitsOp, quantum::AllocOp,
                   quantum::AllocQubitOp, quantum::DeallocOp, quantum::DeallocQubitOp,
                   quantum::ExtractOp, quantum::InsertOp, quantum::Observable,
                   quantum::MeasurementProcess>(op);
    });

    RewritePatternSet patterns(ctx);
    populateToPBCPatterns(patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        return failure();
    }

    // Apply canonicalization patterns for generated PPR operations
    RewritePatternSet canonPatterns(ctx);
    catalyst::pbc::PPRotationOp::getCanonicalizationPatterns(canonPatterns, ctx);
    catalyst::pbc::PPRotationArbitraryOp::getCanonicalizationPatterns(canonPatterns, ctx);
    if (failed(applyPatternsGreedily(module, std::move(canonPatterns)))) {
        return failure();
    }

    return success();
}

// Deprecated alias for `--to-pbc`, kept for backward compatibility with `pass_name="to-ppr"`.
struct DeprecatedToPPRPass : public PassWrapper<DeprecatedToPPRPass, OperationPass<>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeprecatedToPPRPass)

    StringRef getArgument() const final { return "to-ppr"; }
    StringRef getDescription() const final {
        return "Deprecated alias for to-pbc: convert quantum dialects to the PBC dialect.";
    }
    void getDependentDialects(DialectRegistry &registry) const final {
        registry.insert<arith::ArithDialect, catalyst::CatalystDialect, catalyst::pbc::PBCDialect,
                        complex::ComplexDialect, index::IndexDialect, memref::MemRefDialect,
                        scf::SCFDialect, tensor::TensorDialect>();
    }
    void runOnOperation() final {
        if (failed(runToPBC(getOperation(), &getContext()))) {
            return signalPassFailure();
        }
    }
};

} // namespace

struct ToPBCPass : public impl::ToPBCPassBase<ToPBCPass> {
    using ToPBCPassBase::ToPBCPassBase;

    void runOnOperation() final {
        if (failed(runToPBC(getOperation(), &getContext()))) {
            return signalPassFailure();
        }
    }
};

void registerDeprecatedToPPRPass() {
    mlir::registerPass(
        []() -> std::unique_ptr<mlir::Pass> { return std::make_unique<DeprecatedToPPRPass>(); });
}

} // namespace pbc
} // namespace catalyst
