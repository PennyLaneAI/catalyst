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

// The plugin ABI for out-of-tree transport backends.
//
// A transport backend is a shared library that implements a TransportSession role and exports a
// factory symbol.

#pragma once
#ifndef TRANSPORTBACKEND_H
#define TRANSPORTBACKEND_H

#include <cstdio>
#include <exception>

#include "Transport.hpp"

#define CATALYST_TRANSPORT_CONTROLLER_FACTORY_SYMBOL "CatalystTransportControllerFactory"

// The factory signature backends must export with C linkage.
extern "C" {
using CatalystTransportControllerFactoryFn = catalyst::transport::ControllerSession *(const char *);
}

// A helper template macro to generate the <IDENTIFIER>Factory function.
// e.g. GENERATE_TRANSPORT_CONTROLLER_FACTORY(CatalystTransportController, make_controller)
// where `make_controller(const std::string &config) -> ControllerSession*`.
#define GENERATE_TRANSPORT_CONTROLLER_FACTORY(IDENTIFIER, BUILDER)                                 \
    extern "C" catalyst::transport::ControllerSession *IDENTIFIER##Factory(const char *config)     \
    {                                                                                              \
        try {                                                                                      \
            return (BUILDER)(config ? std::string(config) : std::string());                        \
        }                                                                                          \
        catch (const std::exception &e) {                                                          \
            std::fprintf(stderr, "[transport] controller factory failed: %s\n", e.what());         \
            return nullptr;                                                                        \
        }                                                                                          \
        catch (...) {                                                                              \
            std::fprintf(stderr, "[transport] controller factory failed: unknown exception\n");    \
            return nullptr;                                                                        \
        }                                                                                          \
    }

#endif // TRANSPORTBACKEND_H
