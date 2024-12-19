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

#include "mlir/CAPI/Registration.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Quantum, quantum);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Gradient, gradient);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Mitigation, mitigation);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Catalyst, catalyst);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Ion, ion);

#ifdef __cplusplus
}
#endif
