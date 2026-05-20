// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "QecLogical/IR/QecLogicalOps.h"

#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"

#include "QecLogical/IR/QecLogicalDialect.h"

using namespace mlir;
using namespace catalyst::qecl;

//===----------------------------------------------------------------------===//
// QecLogical op definitions.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "QecLogical/IR/QecLogicalOps.cpp.inc"

//===----------------------------------------------------------------------===//
// QecLogical op verifiers.
//===----------------------------------------------------------------------===//

LogicalResult ExtractCodeblockOp::verify() {
    if (!(getIdx() || getIdxAttr().has_value())) {
        return emitOpError() << "expected to have a non-null index";
    }

    const auto hyperRegType = getHyperReg().getType();
    const auto codeblockType = getCodeblock().getType();

    if (hyperRegType.getK() != codeblockType.getK()) {
        return emitOpError()
               << "expected hyper-register and codeblock types to have same value of k, "
                  "but got hyper-register with k = "
               << hyperRegType.getK() << " and codeblock with k = " << codeblockType.getK();
    }

    if (getIdxAttr().has_value()) {
        auto idx = getIdxAttr()->getSExtValue();
        if (idx < 0 || idx >= hyperRegType.getWidth()) {
            return emitOpError() << "has out-of-bounds index attribute: extracting from index "
                                 << idx << " but hyper-register has width "
                                 << hyperRegType.getWidth();
        }
    }

    return success();
}

LogicalResult InsertCodeblockOp::verify() {
    if (!(getIdx() || getIdxAttr().has_value())) {
        return emitOpError() << "expected to have a non-null index";
    }

    // In and out hyper-register types are already constrained to be the same
    const auto hyperRegType = getInHyperReg().getType();
    const auto codeblockType = getCodeblock().getType();

    if (hyperRegType.getK() != codeblockType.getK()) {
        return emitOpError()
               << "expected hyper-register and codeblock types to have same value of k, "
                  "but got hyper-register with k = "
               << hyperRegType.getK() << " and codeblock with k = " << codeblockType.getK();
    }

    if (getIdxAttr().has_value()) {
        auto idx = getIdxAttr()->getSExtValue();
        if (idx < 0 || idx >= hyperRegType.getWidth()) {
            return emitOpError() << "has out-of-bounds index attribute: inserting at index " << idx
                                 << " but hyper-register has width " << hyperRegType.getWidth();
        }
    }

    return success();
}

template <typename OpTy> static LogicalResult verifySingleQubitLogicalGateOp(OpTy op) {
    if (!(op.getIdx() || op.getIdxAttr().has_value())) {
        return op.emitOpError() << "expected to have a non-null index";
    }

    // In and out codeblocks types are already constrained to be the same
    const auto codeblockType = op.getInCodeblock().getType();

    if (auto idxAttr = op.getIdxAttr()) {
        int64_t idx = idxAttr->getSExtValue();
        int64_t k = codeblockType.getK();

        if (idx < 0 || idx >= k) {
            return op.emitOpError()
                   << "has out-of-bounds index attribute: applying gate to logical qubit at index "
                   << idx << " in codeblock with k = " << k;
        }
    }

    return success();
}

LogicalResult IdentityOp::verify() { return verifySingleQubitLogicalGateOp(*this); }

LogicalResult PauliXOp::verify() { return verifySingleQubitLogicalGateOp(*this); }

LogicalResult PauliYOp::verify() { return verifySingleQubitLogicalGateOp(*this); }

LogicalResult PauliZOp::verify() { return verifySingleQubitLogicalGateOp(*this); }

LogicalResult HadamardOp::verify() { return verifySingleQubitLogicalGateOp(*this); }

LogicalResult SOp::verify() { return verifySingleQubitLogicalGateOp(*this); }

LogicalResult CnotOp::verify() {
    if (!(getIdxCtrl() || getIdxCtrlAttr().has_value())) {
        return emitOpError() << "expected to have a non-null ctrl index";
    }
    if (!(getIdxTrgt() || getIdxTrgtAttr().has_value())) {
        return emitOpError() << "expected to have a non-null target index";
    }

    // In and out codeblocks types are already constrained to be the same
    const auto ctrlCodeblockType = getInCtrlCodeblock().getType();
    const auto trgtCodeblockType = getInTrgtCodeblock().getType();

    if (getIdxCtrlAttr().has_value()) {
        auto idx = getIdxCtrlAttr()->getSExtValue();
        if (idx < 0 || idx >= ctrlCodeblockType.getK()) {
            return emitOpError()
                   << "has out-of-bounds index attribute: applying gate to logical qubit at index "
                   << idx << " in ctrl codeblock with k = " << ctrlCodeblockType.getK();
        }
    }

    if (getIdxTrgtAttr().has_value()) {
        auto idx = getIdxTrgtAttr()->getSExtValue();
        if (idx < 0 || idx >= trgtCodeblockType.getK()) {
            return emitOpError()
                   << "has out-of-bounds index attribute: applying gate to logical qubit at index "
                   << idx << " in target codeblock with k = " << trgtCodeblockType.getK();
        }
    }

    return success();
}

LogicalResult MeasureOp::verify() {
    if (!(getIdx() || getIdxAttr().has_value())) {
        return emitOpError() << "expected to have a non-null index";
    }

    // In and out codeblocks types are already constrained to be the same
    const auto codeblockType = getInCodeblock().getType();

    if (getIdxAttr().has_value()) {
        auto idx = getIdxAttr()->getSExtValue();
        if (idx < 0 || idx >= codeblockType.getK()) {
            return emitOpError()
                   << "has out-of-bounds index attribute: applying measurement to logical qubit at "
                      "index "
                   << idx << " in codeblock with k = " << codeblockType.getK();
        }
    }

    return success();
}

//===----------------------------------------------------------------------===//
// QecLogical op canonicalizers.
//===----------------------------------------------------------------------===//

/**
 * @brief Canonicalize logical hyper-register allocation op.
 *
 * Erase alloc op if it has no uses.
 */
LogicalResult AllocOp::canonicalize(AllocOp alloc, mlir::PatternRewriter &rewriter) {
    if (alloc->use_empty()) {
        rewriter.eraseOp(alloc);
        return success();
    }

    return failure();
}

/**
 * @brief Canonicalize logical hyper-register deallocation op.
 *
 * Erase alloc/dealloc op pairs if allocated hyper-register is immediately deallocated.
 */
LogicalResult DeallocOp::canonicalize(DeallocOp dealloc, mlir::PatternRewriter &rewriter) {
    const auto hyperReg = dealloc.getHyperReg();
    if (auto alloc = dyn_cast_if_present<AllocOp>(hyperReg.getDefiningOp())) {
        if (hyperReg.hasOneUse()) {
            rewriter.eraseOp(dealloc);
            rewriter.eraseOp(alloc);
            return success();
        }
    }

    return failure();
}

/**
 * @brief Canonicalize extract-codeblock op.
 *
 * Removes sequential insert-extract op pairs acting on the same index, e.g. handles the pattern:
 *
 *   %r0 = ... : !qecl.hyperreg<3 x 1>
 *   %b0 = ... : !qecl.codeblock<1>
 *   %r1 = insert_block %r0[0], %b0
 *   %b1 = extract_block %r1[0]
 *   %r2 = test.op %r1
 *   %b2 = test.op %b1
 *
 * and converts to:
 *
 *   %r0 = ... : !qecl.hyperreg<3 x 1>
 *   %b0 = ... : !qecl.codeblock<1>
 *   %r2 = test.op %r0
 *   %b2 = test.op %b0
 */
LogicalResult ExtractCodeblockOp::canonicalize(ExtractCodeblockOp extract,
                                               mlir::PatternRewriter &rewriter) {
    if (auto insert =
            dyn_cast_if_present<InsertCodeblockOp>(extract.getHyperReg().getDefiningOp())) {
        bool bothStatic = extract.getIdxAttr().has_value() && insert.getIdxAttr().has_value();
        bool bothDynamic = !extract.getIdxAttr().has_value() && !insert.getIdxAttr().has_value();

        bool staticallyEqual = bothStatic && extract.getIdxAttrAttr() == insert.getIdxAttrAttr();
        bool dynamicallyEqual = bothDynamic && extract.getIdx() == insert.getIdx();

        bool inSameBlock = extract->getBlock() == insert->getBlock();

        if ((staticallyEqual || dynamicallyEqual) && inSameBlock) {
            rewriter.replaceOp(extract, insert.getCodeblock());
            rewriter.replaceOp(insert, insert.getInHyperReg());
            return success();
        }
    }
    return failure();
}

/**
 * @brief Canonicalize insert-codeblock op.
 *
 * Removes sequential extract-insert op pairs acting on the same index, e.g. handles the pattern:
 *
 *   %r0 = ... : !qecl.hyperreg<3 x 1>
 *   %b0 = extract_block %r0[0]
 *   %r1 = insert_block %r0[0], %b0
 *   %r2 = test.op %r1
 *
 * and converts to:
 *
 *   %r0 = ... : !qecl.hyperreg<3 x 1>
 *   %r2 = test.op %r0
 */
LogicalResult InsertCodeblockOp::canonicalize(InsertCodeblockOp insert,
                                              mlir::PatternRewriter &rewriter) {
    if (auto extract =
            dyn_cast_if_present<ExtractCodeblockOp>(insert.getCodeblock().getDefiningOp())) {
        bool bothStatic = extract.getIdxAttr().has_value() && insert.getIdxAttr().has_value();
        bool bothDynamic = !extract.getIdxAttr().has_value() && !insert.getIdxAttr().has_value();

        bool staticallyEqual = bothStatic && extract.getIdxAttrAttr() == insert.getIdxAttrAttr();
        bool dynamicallyEqual = bothDynamic && extract.getIdx() == insert.getIdx();

        bool sameHyperReg = extract.getHyperReg() == insert.getInHyperReg();
        bool oneUse = extract.getResult().hasOneUse();

        if ((staticallyEqual || dynamicallyEqual) && oneUse && sameHyperReg) {
            rewriter.replaceOp(insert, insert.getInHyperReg());
            rewriter.eraseOp(extract);
            return success();
        }
    }

    return failure();
}

//===----------------------------------------------------------------------===//
// QecLogical op folders.
//===----------------------------------------------------------------------===//

/**
 * @brief Prefer using an attribute when the index is constant.
 */
template <typename IndexingOp> LogicalResult foldConstantIndexingOp(IndexingOp op, Attribute idx) {
    bool hasNoIdxAttr = !op.getIdxAttr().has_value();
    bool isConstantIdx = isa_and_nonnull<IntegerAttr>(idx);
    if (hasNoIdxAttr && isConstantIdx) {
        auto constantIdx = cast<IntegerAttr>(idx);
        op.setIdxAttr(constantIdx.getValue());

        // Remove the dynamic Value
        op.getIdxMutable().clear();
        return success();
    }
    return failure();
}

/**
 * @brief Fold method for extract-codeblock op.
 */
OpFoldResult ExtractCodeblockOp::fold(FoldAdaptor adaptor) {
    if (succeeded(foldConstantIndexingOp(*this, adaptor.getIdx()))) {
        return getResult();
    }
    // Returning nullptr tells the caller the op was unchanged.
    return nullptr;
}

/**
 * @brief Fold method for insert-codeblock op.
 */
OpFoldResult InsertCodeblockOp::fold(FoldAdaptor adaptor) {
    if (succeeded(foldConstantIndexingOp(*this, adaptor.getIdx()))) {
        return getResult();
    }
    // Returning nullptr tells the caller the op was unchanged.
    return nullptr;
}
