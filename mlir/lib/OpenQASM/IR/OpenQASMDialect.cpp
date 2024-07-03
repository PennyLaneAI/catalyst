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

#include "mlir/Transforms/InliningUtils.h"

#include "OpenQASM/IR/OpenQASMDialect.h"
#include "OpenQASM/IR/OpenQASMOps.h"

using namespace mlir;
using namespace catalyst::openqasm;

#include "OpenQASM/IR/OpenQASMOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// OpenQASM Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct OpenQASMInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    /// Operations in OpenQASM dialect are always legal to inline.
    bool isLegalToInline(Operation *, Region *, bool, IRMapping &valueMapping) const final
    {
        return true;
    }
};
} // namespace

//===----------------------------------------------------------------------===//
// OpenQASM dialect.
//===----------------------------------------------------------------------===//

void OpenQASMDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "OpenQASM/IR/OpenQASMOps.cpp.inc"
        >();
    addInterface<OpenQASMInlinerInterface>();
}
