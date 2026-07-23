#pragma once
#include "Error.hpp"

namespace rdma::devices::common {

enum class QpState { RESET, INIT, RTR, RTS, ERROR };

inline const char *to_string(QpState s)
{
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
constexpr bool is_valid_transition(QpState from, QpState to)
{
    if (to == QpState::ERROR || to == QpState::RESET)
        return true;
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

class BadTransition : public RdmaError {
  public:
    using RdmaError::RdmaError;
};

} // namespace rdma::devices::common
