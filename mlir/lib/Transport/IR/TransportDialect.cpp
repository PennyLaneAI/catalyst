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

#include "Transport/IR/TransportDialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "Transport/IR/TransportOps.h"

using namespace mlir;
using namespace catalyst::transport;

//===----------------------------------------------------------------------===//
// Transport dialect definitions.
//===----------------------------------------------------------------------===//

#include "Transport/IR/TransportEnums.cpp.inc"
#include "Transport/IR/TransportOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Transport type definitions.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Transport/IR/TransportOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Transport attribute definitions.
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "Transport/IR/TransportAttributes.cpp.inc"

StringAttr NodeAttr::keyOr(llvm::StringRef fallback) const
{
    if (StringAttr n = getName(); n && !n.getValue().empty()) {
        return n;
    }
    return StringAttr::get(getContext(), fallback);
}

StringAttr NodeAttr::dataPathOr(llvm::StringRef dflt) const
{
    if (StringAttr p = getDataPath(); p && !p.getValue().empty()) {
        return p;
    }
    return StringAttr::get(getContext(), dflt);
}

bool NodeAttr::isRemote() const
{
    BoolAttr r = getRemote();
    return r && r.getValue();
}

static int64_t intOr(IntegerAttr field, int64_t dflt) { return field ? field.getInt() : dflt; }

// Default per-message payload width, matching the current backend's defaults.
constexpr int64_t kDefaultMessageBytes = 8;

int64_t NodeAttr::oobPort() const { return intOr(getOobPort(), 0); }
int64_t NodeAttr::inBytes() const { return intOr(getInBytes(), kDefaultMessageBytes); }
int64_t NodeAttr::outBytes() const { return intOr(getOutBytes(), kDefaultMessageBytes); }
int64_t NodeAttr::workItemIdx() const { return intOr(getWorkItemIdx(), 0); }

LogicalResult BacklineAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                   StringAttr transport, NodeAttr controller,
                                   llvm::ArrayRef<NodeAttr> coprocessors)
{
    if (!controller) {
        return emitError() << "backline requires a controller";
    }
    for (NodeAttr c : coprocessors) {
        if (!c) {
            return emitError() << "null coprocessor";
        }
        if (!c.getPeer() || c.getPeer().getValue().empty()) {
            return emitError() << "coprocessor requires a 'peer'";
        }
        if (!c.getSymbol() || c.getSymbol().getValue().empty()) {
            return emitError() << "coprocessor requires a 'symbol'";
        }
    }
    return success();
}

void TransportDialect::initialize()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "Transport/IR/TransportOpsTypes.cpp.inc"
        >();

    addAttributes<
#define GET_ATTRDEF_LIST
#include "Transport/IR/TransportAttributes.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "Transport/IR/TransportOps.cpp.inc"
        >();
}
