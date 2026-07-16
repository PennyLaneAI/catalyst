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

#include "MBQC/IR/MBQCOps.h"

#include "mlir/IR/Builders.h"

#include "MBQC/IR/MBQCDialect.h"
#include "QRef/IR/QRefDialect.h"

using namespace mlir;
using namespace catalyst::mbqc;

//===----------------------------------------------------------------------===//
// MBQC op definitions.
//===----------------------------------------------------------------------===//

#include "MBQC/IR/MBQCEnums.cpp.inc"

#define GET_OP_CLASSES
#include "MBQC/IR/MBQCOps.cpp.inc"

namespace catalyst::mbqc {

//===----------------------------------------------------------------------===//
// MBQC op verifiers.
//===----------------------------------------------------------------------===//

LogicalResult RefGraphStatePrepOp::verify()
{
    qref::QuregType qregType = getQreg().getType();
    if (qregType.isDynamic()) {
        return emitOpError() << "expected static allocation size";
    }

    ShapedType adjMatrixType = cast<ShapedType>(getAdjMatrix().getType());
    size_t adjMatrixSize = adjMatrixType.getShape()[0];

    size_t expectedAdjMatrixSize = getAdjMatrixSizeFromNumQubits();
    if (adjMatrixSize != expectedAdjMatrixSize) {
        return emitOpError()
               << "mismatch between allocation size and size of densely packed adjacency "
                  "matrix. For an allocation size of "
               << qregType.getSize().getInt()
               << ", the densely packed adjacency matrix size is expected to be "
               << expectedAdjMatrixSize;
    }

    return success();
}

//===----------------------------------------------------------------------===//
// Implement ResourceQuantumOpInterface methods.
//===----------------------------------------------------------------------===//

llvm::StringRef MeasureInBasisOp::getResourceName() { return getOperationName(); }
llvm::StringRef RefMeasureInBasisOp::getResourceName() { return getOperationName(); }
llvm::StringRef GraphStatePrepOp::getResourceName() { return getOperationName(); }
llvm::StringRef RefGraphStatePrepOp::getResourceName() { return getOperationName(); }

uint64_t MeasureInBasisOp::getResourceNumQubits() { return 0; }
uint64_t RefMeasureInBasisOp::getResourceNumQubits() { return 0; }
uint64_t GraphStatePrepOp::getResourceNumQubits() { return 0; }
uint64_t RefGraphStatePrepOp::getResourceNumQubits() { return 0; }

uint64_t MeasureInBasisOp::getResourceNumCtrlQubits() { return 0; }
uint64_t RefMeasureInBasisOp::getResourceNumCtrlQubits() { return 0; }
uint64_t GraphStatePrepOp::getResourceNumCtrlQubits() { return 0; }
uint64_t RefGraphStatePrepOp::getResourceNumCtrlQubits() { return 0; }

uint64_t MeasureInBasisOp::getResourceNumParams() { return 0; }
uint64_t RefMeasureInBasisOp::getResourceNumParams() { return 0; }
uint64_t GraphStatePrepOp::getResourceNumParams() { return 0; }
uint64_t RefGraphStatePrepOp::getResourceNumParams() { return 0; }

uint64_t GraphStatePrepOp::getResourceNumAllocQubits() { return getNumQubitsFromAdjMatrixSize(); }
uint64_t RefGraphStatePrepOp::getResourceNumAllocQubits()
{
    return getNumQubitsFromAdjMatrixSize();
}

} // namespace catalyst::mbqc
