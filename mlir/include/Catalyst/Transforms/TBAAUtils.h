// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

namespace catalyst {

class TBAATree {

    mlir::LLVM::TBAARootAttr root;

    mlir::LLVM::TBAATypeDescriptorAttr intDesc;
    mlir::LLVM::TBAATypeDescriptorAttr floatDesc;
    mlir::LLVM::TBAATypeDescriptorAttr pointerDesc;

  public:
    TBAATree(mlir::MLIRContext *ctx, StringRef rootName, StringRef intName, StringRef floatName,
             StringRef pointerName);
    mlir::LLVM::TBAATypeDescriptorAttr createTBAATypeDescriptor(mlir::MLIRContext *ctx,
                                                                mlir::LLVM::TBAARootAttr rootAttr,
                                                                StringRef typeName);
};
}; // namespace catalyst
