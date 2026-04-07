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

#include "Quantum/IR/QuantumDialect.h"

#include "MBQC/IR/MBQCDialect.h"

//===----------------------------------------------------------------------===//
// MBQC ops declarations.
//===----------------------------------------------------------------------===//

#include "MBQC/IR/MBQCEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "MBQC/IR/MBQCAttributes.h.inc"

#define GET_OP_CLASSES
#include "MBQC/IR/MBQCOps.h.inc"
