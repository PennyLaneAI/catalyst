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

#pragma once

#include "Catalyst/Transforms/Passes.h"
#include "Gradient/Transforms/Passes.h"
#include "Ion/Transforms/Passes.h"
#include "MBQC/Transforms/Passes.h"
#include "Mitigation/Transforms/Passes.h"
#include "QEC/Transforms/Passes.h"
#include "Quantum/Transforms/Passes.h"
#include "Test/Transforms/Passes.h"
#include "mlir-hlo/Transforms/Passes.h"

namespace catalyst {

inline void registerAllPasses()
{
    registerCatalystPasses();
    gradient::registerGradientPasses();
    ion::registerIonPasses();
    mbqc::registerMBQCPasses();
    mhlo::registerMhloPasses();
    mitigation::registerMitigationPasses();
    qec::registerQECPasses();
    quantum::registerQuantumPasses();
    test::registerTestPasses();
}

} // namespace catalyst
