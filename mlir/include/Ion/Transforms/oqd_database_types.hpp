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

#include <cstdint>
#include <string>
#include <vector>

namespace {

//
// Calibrated parameters
//

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

//
// Innate atomic parameters
//

struct Level {
    // This class represents an atomic level.
    // It contains the innate properties of the qubit.
    std::string label;
    int64_t principal;
    double spin, orbital, nuclear, spin_orbital, spin_orbital_nuclear,
        spin_orbital_nuclear_magnetization, energy;

    Level(std::string _label, int64_t _principal, double _spin, double _orbital, double _nuclear,
          double _spin_orbital, double _spin_orbital_nuclear,
          double _spin_orbital_nuclear_magnetization, double _energy)
        : label(_label), principal(_principal), spin(_spin), orbital(_orbital), nuclear(_nuclear),
          spin_orbital(_spin_orbital), spin_orbital_nuclear(_spin_orbital_nuclear),
          spin_orbital_nuclear_magnetization(_spin_orbital_nuclear_magnetization), energy(_energy)
    {
    }
};

struct Transition {
    // This class represents a transition between two atomic levels.
    // It contains the innate properties of the qubit.
    std::string level_0, level_1;
    double einstein_a;

    Transition(std::string _level_0, std::string _level_1, double _einstein_a)
        : level_0(_level_0), level_1(_level_1), einstein_a(_einstein_a)
    {
    }
};

struct Ion {
    // This class represents an ion.
    // It contains the innate properties of the qubit.
    std::string name;
    double mass, charge;
    std::vector<int64_t> position;
    std::vector<Level> levels;
    std::vector<Transition> transitions;

    Ion(std::string _name, double _mass, double _charge, std::vector<int64_t> _position,
        std::vector<Level> _levels, std::vector<Transition> _transitions)
        : name(_name), mass(_mass), charge(_charge), position(_position), levels(_levels),
          transitions(_transitions)
    {
    }
};

} // namespace
