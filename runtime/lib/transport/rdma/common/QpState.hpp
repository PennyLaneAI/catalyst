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
#include <cstdint>

#include "Error.hpp"

namespace catalyst::transport::common {

enum class QpState : std::uint8_t { RESET, INIT, RTR, RTS, ERROR };

inline const char *to_string(QpState s) {
    switch (s) {
    case QpState::RESET:
        return "Reset";
    case QpState::INIT:
        return "Init";
    case QpState::RTR:
        return "Rtr";
    case QpState::RTS:
        return "Rts";
    case QpState::ERROR:
        return "Error";
    }
    return "?";
}

// Checks whether a given transition is valid.
constexpr bool is_valid_transition(QpState from, QpState to) {
    if (to == QpState::ERROR || to == QpState::RESET) {
        return true;
    }
    switch (from) {
    case QpState::RESET:
        return to == QpState::INIT;
    case QpState::INIT:
        return to == QpState::RTR;
    case QpState::RTR:
        return to == QpState::RTS;
    default:
        return false;
    }
}

class BadTransition : public TransportError {
  public:
    using TransportError::TransportError;
};

} // namespace catalyst::transport::common
