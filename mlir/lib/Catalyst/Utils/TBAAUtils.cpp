// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Catalyst/Transforms/TBAAUtils.h"

catalyst::TBAATree::TBAATree(mlir::MLIRContext *ctx, StringRef rootName, StringRef intName,
                             StringRef float32Name, StringRef float64Name, StringRef pointerName)
    : root(mlir::LLVM::TBAARootAttr::get(ctx, mlir::StringAttr::get(ctx, rootName)))
{
    intDesc = createTBAATypeDescriptor(ctx, root, intName);
    float32Desc = createTBAATypeDescriptor(ctx, root, float32Name);
    float64Desc = createTBAATypeDescriptor(ctx, root, float64Name);
    pointerDesc = createTBAATypeDescriptor(ctx, root, pointerName);
    tags = createTags();
}

mlir::LLVM::TBAATypeDescriptorAttr
catalyst::TBAATree::createTBAATypeDescriptor(mlir::MLIRContext *ctx,
                                             mlir::LLVM::TBAARootAttr rootAttr, StringRef typeName)
{
    auto memberAttr = mlir::LLVM::TBAAMemberAttr::get(rootAttr, 0);
    return mlir::LLVM::TBAATypeDescriptorAttr::get(ctx, typeName, memberAttr);
}

mlir::DenseMap<StringRef, mlir::LLVM::TBAATagAttr> catalyst::TBAATree::createTags()
{
    mlir::DenseMap<StringRef, mlir::LLVM::TBAATagAttr> map;

    mlir::LLVM::TBAATagAttr intTag = mlir::LLVM::TBAATagAttr::get(intDesc, intDesc, 0);
    map.insert({"int", intTag});
    mlir::LLVM::TBAATagAttr float32Tag = mlir::LLVM::TBAATagAttr::get(float32Desc, float32Desc, 0);
    map.insert({"float", float32Tag});
    mlir::LLVM::TBAATagAttr float64Tag = mlir::LLVM::TBAATagAttr::get(float64Desc, float64Desc, 0);
    map.insert({"double", float64Tag});
    mlir::LLVM::TBAATagAttr pointerTag = mlir::LLVM::TBAATagAttr::get(pointerDesc, pointerDesc, 0);
    map.insert({"any pointer", pointerTag});
    return map;
}

mlir::LLVM::TBAATagAttr catalyst::TBAATree::getTag(StringRef typeName)
{
    return tags.find(typeName)->getSecond();
}
