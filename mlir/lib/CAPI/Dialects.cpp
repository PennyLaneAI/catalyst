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

#include "Quantum-c/Dialects.h"

#include "Gradient/IR/GradientDialect.h"
#include "Quantum/IR/QuantumDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Quantum, quantum, catalyst::quantum::QuantumDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Gradient, gradient, catalyst::gradient::GradientDialect)
