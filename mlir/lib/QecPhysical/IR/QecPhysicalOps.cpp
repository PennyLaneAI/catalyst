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

#include "QecPhysical/IR/QecPhysicalOps.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"

#include "QecPhysical/IR/QecPhysicalAttrDefs.h"
#include "QecPhysical/IR/QecPhysicalDialect.h"
#include "QecPhysical/IR/QecPhysicalTypes.h"

using namespace mlir;
using namespace catalyst::qecp;

//===----------------------------------------------------------------------===//
// QecPhysical op definitions.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "QecPhysical/IR/QecPhysicalOps.cpp.inc"

//===----------------------------------------------------------------------===//
// QecPhysical op verifiers.
//===----------------------------------------------------------------------===//

LogicalResult AllocAuxQubitOp::verify()
{
    const auto qubitRole = getQubit().getType().getRole();
    if (qubitRole != QecPhysicalQubitRole::Auxiliary) {
        return emitOpError() << "expected a QEC physical qubit with role '"
                             << stringifyQecPhysicalQubitRole(QecPhysicalQubitRole::Auxiliary)
                             << "', but got '" << stringifyQecPhysicalQubitRole(qubitRole) << "'";
    }
    return success();
}

LogicalResult DeallocAuxQubitOp::verify()
{
    const auto qubitRole = getQubit().getType().getRole();
    if (qubitRole != QecPhysicalQubitRole::Auxiliary) {
        return emitOpError() << "expected a QEC physical qubit with role '"
                             << stringifyQecPhysicalQubitRole(QecPhysicalQubitRole::Auxiliary)
                             << "', but got '" << stringifyQecPhysicalQubitRole(qubitRole) << "'";
    }
    return success();
}

LogicalResult ExtractCodeblockOp::verify()
{
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

    if (hyperRegType.getN() != codeblockType.getN()) {
        return emitOpError()
               << "expected hyper-register and codeblock types to have same value of n, "
                  "but got hyper-register with n = "
               << hyperRegType.getN() << " and codeblock with n = " << codeblockType.getN();
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

LogicalResult InsertCodeblockOp::verify()
{
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

    if (hyperRegType.getN() != codeblockType.getN()) {
        return emitOpError()
               << "expected hyper-register and codeblock types to have same value of n, "
                  "but got hyper-register with n = "
               << hyperRegType.getN() << " and codeblock with n = " << codeblockType.getN();
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

LogicalResult ExtractQubitOp::verify()
{
    if (!(getIdx() || getIdxAttr().has_value())) {
        return emitOpError() << "expected to have a non-null index";
    }

    const auto codeblockType = getCodeblock().getType();

    if (getIdxAttr().has_value()) {
        auto idx = getIdxAttr()->getSExtValue();
        if (idx < 0 || idx >= codeblockType.getN()) {
            return emitOpError() << "has out-of-bounds index attribute: extracting from index "
                                 << idx << " but codeblock has n = " << codeblockType.getN();
        }
    }

    const auto qubitTypeRole = getQubit().getType().getRole();

    if (qubitTypeRole != QecPhysicalQubitRole::Data) {
        return emitOpError() << "only physical qubits with role '"
                             << stringifyQecPhysicalQubitRole(QecPhysicalQubitRole::Data)
                             << "' should be extracted from a physical codeblock, but got '"
                             << stringifyQecPhysicalQubitRole(qubitTypeRole) << "'";
    }

    return success();
}

LogicalResult InsertQubitOp::verify()
{
    if (!(getIdx() || getIdxAttr().has_value())) {
        return emitOpError() << "expected to have a non-null index";
    }

    // In and out codeblock types are already constrained to be the same
    const auto codeblockType = getInCodeblock().getType();

    if (getIdxAttr().has_value()) {
        auto idx = getIdxAttr()->getSExtValue();
        if (idx < 0 || idx >= codeblockType.getN()) {
            return emitOpError() << "has out-of-bounds index attribute: inserting at index " << idx
                                 << " but codeblock has n = " << codeblockType.getN();
        }
    }

    const auto qubitTypeRole = getQubit().getType().getRole();

    if (qubitTypeRole != QecPhysicalQubitRole::Data) {
        return emitOpError() << "only physical qubits with role '"
                             << stringifyQecPhysicalQubitRole(QecPhysicalQubitRole::Data)
                             << "' should be inserted into a physical codeblock, but got '"
                             << stringifyQecPhysicalQubitRole(qubitTypeRole) << "'";
    }

    return success();
}

LogicalResult AssembleTannerGraphOp::verify()
{
    const auto rowIdxType = dyn_cast<ShapedType>(getRowIdx().getType());
    const auto colPtrType = dyn_cast<ShapedType>(getColPtr().getType());
    const auto tannerGraphType = getTannerGraph().getType();

    const auto rowIdxElementType = rowIdxType.getElementType();
    const auto colPtrElementType = colPtrType.getElementType();
    const auto tannerGraphElementType = tannerGraphType.getElementType();

    if (rowIdxElementType != colPtrElementType) {
        return emitOpError()
               << "expected row_idx and col_ptr types to have same element type, but got "
               << rowIdxElementType << " and " << colPtrElementType << ", respectively";
    }

    if (rowIdxElementType != tannerGraphElementType) {
        return emitOpError()
               << "expected input operands and returned Tanner graph to have same element types, "
                  "but got "
               << rowIdxElementType << ", " << colPtrElementType << " and "
               << tannerGraphElementType << ", respectively";
    }

    const auto rowIdxSize = rowIdxType.getDimSize(0);
    const auto colPtrSize = colPtrType.getDimSize(0);
    const auto tannerGraphRowIdxSize = tannerGraphType.getRowIdxSize();
    const auto tannerGraphColPtrSize = tannerGraphType.getColPtrSize();

    if (rowIdxSize != tannerGraphRowIdxSize || colPtrSize != tannerGraphColPtrSize) {
        return emitOpError() << "expected input row_idx and col_ptr sizes to match returned Tanner "
                                "graph sizes, but got row_idx sizes "
                             << rowIdxSize << " (in), " << tannerGraphRowIdxSize
                             << " (out), and col_ptr sizes " << colPtrSize << " (in), "
                             << tannerGraphColPtrSize << " (out)";
    }

    return success();
}

//===----------------------------------------------------------------------===//
// QecPhysical op canonicalizers.
//===----------------------------------------------------------------------===//

/**
 * @brief Canonicalize physical hyper-register allocation op.
 *
 * Erase alloc op if it has no uses.
 */
LogicalResult AllocOp::canonicalize(AllocOp alloc, mlir::PatternRewriter &rewriter)
{
    if (alloc->use_empty()) {
        rewriter.eraseOp(alloc);
        return success();
    }

    return failure();
}

/**
 * @brief Canonicalize physical hyper-register deallocation op.
 *
 * Erase alloc/dealloc op pairs if allocated hyper-register is immediately deallocated.
 */
LogicalResult DeallocOp::canonicalize(DeallocOp dealloc, mlir::PatternRewriter &rewriter)
{
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
 * @brief Canonicalize aux qubit allocation op.
 *
 * Erase alloc_aux op if it has no uses.
 */
LogicalResult AllocAuxQubitOp::canonicalize(AllocAuxQubitOp alloc, mlir::PatternRewriter &rewriter)
{
    if (alloc->use_empty()) {
        rewriter.eraseOp(alloc);
        return success();
    }

    return failure();
}

/**
 * @brief Canonicalize aux qubit deallocation op.
 *
 * Erase alloc/dealloc op pairs if allocated aux qubit is immediately deallocated.
 */
LogicalResult DeallocAuxQubitOp::canonicalize(DeallocAuxQubitOp dealloc,
                                              mlir::PatternRewriter &rewriter)
{
    const auto qubit = dealloc.getQubit();
    if (auto alloc = dyn_cast_if_present<AllocAuxQubitOp>(qubit.getDefiningOp())) {
        if (qubit.hasOneUse()) {
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
 *   %r0 = ... : !qecp.hyperreg<3 x 1>
 *   %b0 = ... : !qecp.codeblock<1>
 *   %r1 = insert_block %r0[0], %b0
 *   %b1 = extract_block %r1[0]
 *   %r2 = test.op %r1
 *   %b2 = test.op %b1
 *
 * and converts to:
 *
 *   %r0 = ... : !qecp.hyperreg<3 x 1>
 *   %b0 = ... : !qecp.codeblock<1>
 *   %r2 = test.op %r0
 *   %b2 = test.op %b0
 */
LogicalResult ExtractCodeblockOp::canonicalize(ExtractCodeblockOp extract,
                                               mlir::PatternRewriter &rewriter)
{
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
 *   %r1 = insert_block %r1[0], %b0
 *   %r2 = test.op %r1
 *
 * and converts to:
 *
 *   %r0 = ... : !qecl.hyperreg<3 x 1>
 *   %r2 = test.op %r0
 */
LogicalResult InsertCodeblockOp::canonicalize(InsertCodeblockOp insert,
                                              mlir::PatternRewriter &rewriter)
{
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

/**
 * @brief Canonicalize extract-qubit op.
 *
 * Analogous to ExtractCodeblockOp::canonicalize() above.
 */
LogicalResult ExtractQubitOp::canonicalize(ExtractQubitOp extract, mlir::PatternRewriter &rewriter)
{
    if (auto insert = dyn_cast_if_present<InsertQubitOp>(extract.getCodeblock().getDefiningOp())) {
        bool bothStatic = extract.getIdxAttr().has_value() && insert.getIdxAttr().has_value();
        bool bothDynamic = !extract.getIdxAttr().has_value() && !insert.getIdxAttr().has_value();

        bool staticallyEqual = bothStatic && extract.getIdxAttrAttr() == insert.getIdxAttrAttr();
        bool dynamicallyEqual = bothDynamic && extract.getIdx() == insert.getIdx();

        bool inSameBlock = extract->getBlock() == insert->getBlock();

        if ((staticallyEqual || dynamicallyEqual) && inSameBlock) {
            rewriter.replaceOp(extract, insert.getQubit());
            rewriter.replaceOp(insert, insert.getInCodeblock());
            return success();
        }
    }
    return failure();
}

/**
 * @brief Canonicalize insert-qubit op.
 *
 * Analogous to InsertCodeblockOp::canonicalize() above
 */
LogicalResult InsertQubitOp::canonicalize(InsertQubitOp insert, mlir::PatternRewriter &rewriter)
{
    if (auto extract = dyn_cast_if_present<ExtractQubitOp>(insert.getQubit().getDefiningOp())) {
        bool bothStatic = extract.getIdxAttr().has_value() && insert.getIdxAttr().has_value();
        bool bothDynamic = !extract.getIdxAttr().has_value() && !insert.getIdxAttr().has_value();

        bool staticallyEqual = bothStatic && extract.getIdxAttrAttr() == insert.getIdxAttrAttr();
        bool dynamicallyEqual = bothDynamic && extract.getIdx() == insert.getIdx();

        bool sameCodeblock = extract.getCodeblock() == insert.getInCodeblock();
        bool oneUse = extract.getResult().hasOneUse();

        if ((staticallyEqual || dynamicallyEqual) && oneUse && sameCodeblock) {
            rewriter.replaceOp(insert, insert.getInCodeblock());
            rewriter.eraseOp(extract);
            return success();
        }
    }

    return failure();
}

//===----------------------------------------------------------------------===//
// QecPhysical op folders.
//===----------------------------------------------------------------------===//

/**
 * @brief Prefer using an attribute when the index is constant.
 */
template <typename IndexingOp> LogicalResult foldConstantIndexingOp(IndexingOp op, Attribute idx)
{
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
OpFoldResult ExtractCodeblockOp::fold(FoldAdaptor adaptor)
{
    if (succeeded(foldConstantIndexingOp(*this, adaptor.getIdx()))) {
        return getResult();
    }
    // Returning nullptr tells the caller the op was unchanged.
    return nullptr;
}

/**
 * @brief Fold method for insert-codeblock op.
 */
OpFoldResult InsertCodeblockOp::fold(FoldAdaptor adaptor)
{
    if (succeeded(foldConstantIndexingOp(*this, adaptor.getIdx()))) {
        return getResult();
    }
    // Returning nullptr tells the caller the op was unchanged.
    return nullptr;
}

/**
 * @brief Fold method for extract-qubit op.
 */
OpFoldResult ExtractQubitOp::fold(FoldAdaptor adaptor)
{
    if (succeeded(foldConstantIndexingOp(*this, adaptor.getIdx()))) {
        return getResult();
    }
    // Returning nullptr tells the caller the op was unchanged.
    return nullptr;
}

/**
 * @brief Fold method for insert-qubit op.
 */
OpFoldResult InsertQubitOp::fold(FoldAdaptor adaptor)
{
    if (succeeded(foldConstantIndexingOp(*this, adaptor.getIdx()))) {
        return getResult();
    }
    // Returning nullptr tells the caller the op was unchanged.
    return nullptr;
}
