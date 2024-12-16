//===- StandalonePasses.cpp - Standalone passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Standalone/StandalonePasses.h"
#include "Quantum/IR/QuantumOps.h"

namespace mlir::standalone {
#define GEN_PASS_DEF_STANDALONESWITCHBARFOO
#include "Standalone/StandalonePasses.h.inc"

namespace {
class StandaloneSwitchBarFooRewriter : public OpRewritePattern<catalyst::quantum::AllocOp> {
public:
  using OpRewritePattern<catalyst::quantum::AllocOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(catalyst::quantum::AllocOp op,
                                PatternRewriter &rewriter) const final {
    // get the number of qubits allocated
    if (op.getNqubitsAttr().value_or(0) == 1) {
      Type i64 = rewriter.getI64Type();
      auto fortytwo = rewriter.getIntegerAttr(i64, 42);

      // modify the allocation to change the number of qubits to 42.
      rewriter.modifyOpInPlace(op, [&]() { op.setNqubitsAttrAttr(fortytwo); });
      return success();
    }
    // failure indicates that nothing was modified.
    return failure();
  }
};

class StandaloneSwitchBarFoo
    : public impl::StandaloneSwitchBarFooBase<StandaloneSwitchBarFoo> {
public:
  using impl::StandaloneSwitchBarFooBase<
      StandaloneSwitchBarFoo>::StandaloneSwitchBarFooBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<StandaloneSwitchBarFooRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::standalone
