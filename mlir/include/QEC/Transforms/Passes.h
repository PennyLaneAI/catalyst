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

#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"

#include "QEC/Transforms/PassesEnums.h.inc"

namespace catalyst {

std::unique_ptr<mlir::Pass> createLowerToQECPass();
std::unique_ptr<mlir::Pass> createCommutePPRPass();
std::unique_ptr<mlir::Pass> createCliffordTToPPRPass();
std::unique_ptr<mlir::Pass> createMergePPRIntoPPMPass();
std::unique_ptr<mlir::Pass> createDecomposeNonCliffordPPRPass();
std::unique_ptr<mlir::Pass> createDecomposeCliffordPPRPass();
std::unique_ptr<mlir::Pass> createPPMCompilationPass();
std::unique_ptr<mlir::Pass> createCountPPMSpecsPass();
} // namespace catalyst
