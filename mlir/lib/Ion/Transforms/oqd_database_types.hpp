// Copyright 2024 Xanadu Quantum Technologies Inc.

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

#include <vector>

namespace {

struct Beam {
    // This struct contains the calibrated beam parameters.
    double rabi, detuning;
    std::vector<int64_t> polarization, wavevector;

    Beam(double _rabi, double _detuning, std::vector<int64_t> _polarization,
         std::vector<int64_t> _wavevector)
        : rabi(_rabi), detuning(_detuning), polarization(_polarization), wavevector(_wavevector)
    {
    }
};

struct Phonon {
    // This struct contains the calibrated phonon parameters on one axis.
    double energy;
    std::vector<int64_t> eigenvector;

    Phonon(double _energy, std::vector<int64_t> _eigenvector)
        : energy(_energy), eigenvector(_eigenvector)
    {
    }
};

struct PhononMode {
    // This struct contains the calibrated phonon parameters for one ion.
    Phonon COM_x;
    Phonon COM_y;
    Phonon COM_z;

    PhononMode(Phonon x, Phonon y, Phonon z) : COM_x(x), COM_y(y), COM_z(z) {}
};

} // namespace
