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

// Custom assembly format and verifiers for DO-QAOA graph metadata attributes:
//   !quantum.dense_graph  — dense NxN weight matrix (DenseElementsAttr)
//   !quantum.sparse_graph — COO sparse encoding

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

#include "Quantum/IR/QuantumDialect.h"

using namespace mlir;
using namespace catalyst::quantum;

//===----------------------------------------------------------------------===//
// DenseGraphAttr — parse / print / verify
//
// Assembly syntax:
//   #quantum.dense_graph<4, dense<[[0.0,-0.5,-0.5,0.0],...] : tensor<4x4xf64>>
//===----------------------------------------------------------------------===//

Attribute DenseGraphAttr::parse(AsmParser &parser, Type)
{
    uint32_t numNodes = 0;
    DenseElementsAttr weights;

    if (parser.parseLess() || parser.parseInteger(numNodes) || parser.parseComma() ||
        parser.parseAttribute(weights) || parser.parseGreater()) {
        return {};
    }
    // Verify before get() — get() asserts on failure; we need a graceful error.
    if (failed(DenseGraphAttr::verify([&]() { return parser.emitError(parser.getNameLoc()); },
                                      numNodes, weights))) {
        return {};
    }
    return DenseGraphAttr::get(parser.getContext(), numNodes, weights);
}

void DenseGraphAttr::print(AsmPrinter &printer) const
{
    printer << "<" << getNumNodes() << ", " << getWeights() << ">";
}

LogicalResult DenseGraphAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                     uint32_t numNodes, DenseElementsAttr weights)
{
    if (!weights) {
        return emitError() << "dense_graph: weights attribute must not be null";
    }
    auto tensorTy = dyn_cast<RankedTensorType>(weights.getType());
    if (!tensorTy) {
        return emitError() << "dense_graph: weights must be a ranked tensor type";
    }
    if (!isa<Float64Type>(tensorTy.getElementType())) {
        return emitError() << "dense_graph: weights element type must be f64";
    }
    if (tensorTy.getRank() != 2) {
        return emitError() << "dense_graph: weights must be a rank-2 tensor (NxN)";
    }
    auto shape = tensorTy.getShape();
    if (shape[0] != static_cast<int64_t>(numNodes) || shape[1] != static_cast<int64_t>(numNodes)) {
        return emitError() << "dense_graph: weights shape [" << shape[0] << "x" << shape[1]
                           << "] does not match numNodes=" << numNodes;
    }
    return success();
}

//===----------------------------------------------------------------------===//
// SparseGraphAttr — parse / print / verify
//
// Assembly syntax:
//   #quantum.sparse_graph<4, 3, [0,0,1], [1,2,2], [-0.5,-0.5,-0.5]>
//===----------------------------------------------------------------------===//

// Helper: parse a comma-separated list of integers inside square brackets.
static ParseResult parseI32List(AsmParser &parser, SmallVectorImpl<int32_t> &out)
{
    return parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() -> ParseResult {
        int32_t val;
        if (parser.parseInteger(val))
            return failure();
        out.push_back(val);
        return success();
    });
}

Attribute SparseGraphAttr::parse(AsmParser &parser, Type)
{
    uint32_t numNodes = 0, numEdges = 0;
    SmallVector<int32_t> rows, cols;
    DenseElementsAttr wts;

    if (parser.parseLess() || parser.parseInteger(numNodes) || parser.parseComma() ||
        parser.parseInteger(numEdges) || parser.parseComma() || parseI32List(parser, rows) ||
        parser.parseComma() || parseI32List(parser, cols) || parser.parseComma() ||
        parser.parseAttribute(wts) || parser.parseGreater()) {
        return {};
    }
    // Verify before get() — get() asserts on failure; we need a graceful error.
    if (failed(SparseGraphAttr::verify([&]() { return parser.emitError(parser.getNameLoc()); },
                                       numNodes, numEdges, rows, cols, wts))) {
        return {};
    }
    return SparseGraphAttr::get(parser.getContext(), numNodes, numEdges, rows, cols, wts);
}

void SparseGraphAttr::print(AsmPrinter &printer) const
{
    printer << "<" << getNumNodes() << ", " << getNumEdges() << ", [";
    llvm::interleaveComma(getRowIndices(), printer.getStream());
    printer << "], [";
    llvm::interleaveComma(getColIndices(), printer.getStream());
    printer << "], " << getWeights() << ">";
}

LogicalResult SparseGraphAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                      uint32_t numNodes, uint32_t numEdges,
                                      ArrayRef<int32_t> rowIndices, ArrayRef<int32_t> colIndices,
                                      DenseElementsAttr weights)
{
    if (!weights) {
        return emitError() << "sparse_graph: weights attribute must not be null";
    }
    auto tensorTy = dyn_cast<RankedTensorType>(weights.getType());
    if (!tensorTy || tensorTy.getRank() != 1) {
        return emitError() << "sparse_graph: weights must be a rank-1 tensor";
    }
    if (!isa<Float64Type>(tensorTy.getElementType())) {
        return emitError() << "sparse_graph: weights element type must be f64";
    }
    if (rowIndices.size() != numEdges) {
        return emitError() << "sparse_graph: rowIndices length (" << rowIndices.size()
                           << ") != numEdges (" << numEdges << ")";
    }
    if (colIndices.size() != numEdges) {
        return emitError() << "sparse_graph: colIndices length (" << colIndices.size()
                           << ") != numEdges (" << numEdges << ")";
    }
    if (static_cast<uint32_t>(tensorTy.getShape()[0]) != numEdges) {
        return emitError() << "sparse_graph: weights length (" << tensorTy.getShape()[0]
                           << ") != numEdges (" << numEdges << ")";
    }
    for (uint32_t e = 0; e < numEdges; ++e) {
        int32_t r = rowIndices[e], c = colIndices[e];
        if (r < 0 || c < 0) {
            return emitError() << "sparse_graph: negative index at edge " << e;
        }
        if (r >= static_cast<int32_t>(numNodes) || c >= static_cast<int32_t>(numNodes)) {
            return emitError() << "sparse_graph: index out of range at edge " << e
                               << " (numNodes=" << numNodes << ")";
        }
        if (r >= c) {
            return emitError() << "sparse_graph: expected upper-triangle (i < j), got [" << r << ","
                               << c << "] at edge " << e;
        }
    }
    return success();
}
