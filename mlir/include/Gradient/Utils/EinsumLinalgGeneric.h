// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Value.h"

namespace catalyst {

llvm::SmallVector<mlir::Value> einsumLinalgGeneric(
  mlir::OpBuilder& ob,
  mlir::Location loc,
  llvm::ArrayRef<size_t> a_axis,
  llvm::ArrayRef<size_t> b_axis,
  llvm::ArrayRef<size_t> r_axis,
  mlir::Value a,
  mlir::Value b);




}

