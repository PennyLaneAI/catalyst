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

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "Catalyst/IR/CatalystDialect.h"

#define DEBUG_TYPE "memspace-to-address-space"

using namespace mlir;

namespace catalyst {

#define GEN_PASS_DECL_MEMSPACETOADDRESSSPACEPASS
#define GEN_PASS_DEF_MEMSPACETOADDRESSSPACEPASS
#include "Catalyst/Transforms/Passes.h.inc"

namespace {

static FailureOr<IntegerAttr> loweredAddressSpace(MemSpaceAttr attr)
{
    IntegerAttr explicitAttr = attr.getAddrSpace();
    if (!explicitAttr) {
        return failure();
    }
    return IntegerAttr::get(IntegerType::get(attr.getContext(), 64),
                            explicitAttr.getValue().getZExtValue());
}

class MemSpaceToAddressSpacePass
    : public impl::MemSpaceToAddressSpacePassBase<MemSpaceToAddressSpacePass> {
  public:
    using MemSpaceToAddressSpacePassBase::MemSpaceToAddressSpacePassBase;

    void runOnOperation() override;
};

void MemSpaceToAddressSpacePass::runOnOperation()
{
    Operation *root = getOperation();
    bool failedToLower = false;

    AttrTypeReplacer replacer;

    replacer.addReplacement([&](MemSpaceAttr attr) -> std::optional<Attribute> {
        FailureOr<IntegerAttr> lowered = loweredAddressSpace(attr);
        if (failed(lowered)) {
            root->emitError() << "catalyst.memspace for domain '" << attr.getDomain()
                              << "' is missing an explicit `addr_space`";
            failedToLower = true;
            return std::nullopt;
        }
        return *lowered;
    });

    replacer.recursivelyReplaceElementsIn(root, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);

    if (failedToLower) {
        signalPassFailure();
    }
}

} // namespace

} // namespace catalyst
