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

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"

enum class Gate { I, H, X, Y, Z, S, T, RZ, CNOT, SWAP, U, GP };

inline static constexpr size_t PRIMITIV_GATES_COUNT = 12;

inline static constexpr size_t DYNAMIC_ARITY = 3;
static constexpr size_t arity(Gate gate)
{
    switch (gate) {
    case Gate::GP:
        return 0;
    case Gate::CNOT:
    case Gate::SWAP:
        return 2;
    case Gate::I:
    case Gate::U:
        return DYNAMIC_ARITY;
    default:
        return 1;
    };
}

inline static constexpr double PI = llvm::numbers::pi;
inline static constexpr double UNKNOWN_ANGLE = -1.0;
static constexpr double rotAngle(Gate gate)
{
    switch (gate) {
    case Gate::Z:
    case Gate::Y:
        return PI;
    case Gate::S:
        return (PI * 0.5);
    case Gate::T:
        return (PI * 0.25);
    case Gate::RZ:
        return UNKNOWN_ANGLE;
    default:
        return 0.0;
    };
}

static constexpr Gate gateWithAngle(double angle)
{
    if (angle == 0.0) {
        return Gate::I;
    }
    if (angle == rotAngle(Gate::Z)) {
        return Gate::Z;
    }
    if (angle == rotAngle(Gate::S)) {
        return Gate::S;
    }
    if (angle == rotAngle(Gate::T)) {
        return Gate::T;
    }
    return Gate::RZ;
}

static constexpr bool isPhaseGate(Gate gate)
{
    return ((gate == Gate::RZ) || (gate == Gate::T) || (gate == Gate::S) || (gate == Gate::Z) ||
            (gate == Gate::Y));
}

inline static constexpr llvm::StringLiteral GATE_NAME[] = {
    "Identity", "Hadamard", "PauliX", "PauliY", "PauliZ", "S",
    "T",        "RZ",       "CNOT",   "SWAP",   "_",      "GlobalPhase"};

static int initialGateCount[PRIMITIV_GATES_COUNT] = {0};
static int insertedGateCount[PRIMITIV_GATES_COUNT] = {0};
