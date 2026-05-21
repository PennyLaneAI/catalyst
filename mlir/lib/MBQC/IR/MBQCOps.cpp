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
    ShapedType adjMatrixType = cast<ShapedType>(getAdjMatrix().getType());
    size_t adjMatrixSize = adjMatrixType.getShape()[0];

    qref::QuregType qregType = getQreg().getType();
    if (qregType.isDynamic()) {
        return emitOpError() << "expected static allocation size";
    }

    size_t qregSize = qregType.getSize().getInt();
    size_t expectedAdjMatrixSize = qregSize * (qregSize - 1) / 2;
    if (adjMatrixSize != expectedAdjMatrixSize) {
        return emitOpError()
               << "mismatch between allocation size and size of densely packed adjacency "
                  "matrix. For an allocation size of "
               << qregSize << ", the densely packed adjacency matrix size is expected to be "
               << expectedAdjMatrixSize;
    }

    return success();
}
} // namespace catalyst::mbqc
