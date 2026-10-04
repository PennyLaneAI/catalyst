// Copyright 2026
// SPDX-License-Identifier: Apache-2.0
//
// The only job of this translation unit is to emit the factory entry point that
// the Catalyst runtime dlsym's. The symbol MUST be named
// "<DeviceIdentifier>Factory" and MUST have C linkage.

#include "DefaultTensor.hpp"

// Expands to:
//   extern "C" Catalyst::Runtime::QuantumDevice *
//   DefaultTensorFactory(const char *kwargs) {
//       return new Catalyst::Runtime::Devices::DefaultTensor(std::string(kwargs));
//   }
//
// The identifier "DefaultTensor" must match the first element returned by
// the frontend's SUPPORTED_RT_DEVICES entry on the Python side.
GENERATE_DEVICE_FACTORY(DefaultTensor, Catalyst::Runtime::Devices::DefaultTensor);
