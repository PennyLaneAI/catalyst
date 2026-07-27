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

#pragma once

#include <string>
#include <vector>

#include "Quantum/IR/QuantumInterfaces.h"

namespace catalyst::quantum {

using PythonRuleLoweringFn = std::string (*)(DecomposableGate op);

extern PythonRuleLoweringFn pythonRuleLowering;

bool loadQPD(std::string libQPDPath, std::string libpythonPath);

} // namespace catalyst::quantum
