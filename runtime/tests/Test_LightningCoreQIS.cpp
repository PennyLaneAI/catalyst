// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

#include "LightningUtils.hpp"
#include "QuantumDevice.hpp"
#include "RuntimeCAPI.h"

#include <catch2/catch.hpp>

using namespace Catalyst::Runtime;

TEST_CASE("Qubits: allocate, release, dump", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *q = __quantum__rt__qubit_allocate();

    QirString *zero_str = __quantum__rt__int_to_string(0);
    QirString *one_str = __quantum__rt__int_to_string(1);
    QirString *three_str = __quantum__rt__int_to_string(3);

    QirString *qstr = __quantum__rt__qubit_to_string(q);

    REQUIRE(__quantum__rt__string_equal(qstr, zero_str));

    __quantum__rt__string_update_reference_count(qstr, -1);

    __quantum__rt__qubit_release(q);

    QirArray *qs = __quantum__rt__qubit_allocate_array(3);

    REQUIRE(__quantum__rt__array_get_size_1d(qs) == 3);

    QUBIT *first = *reinterpret_cast<QUBIT **>(__quantum__rt__array_get_element_ptr_1d(qs, 0));
    qstr = __quantum__rt__qubit_to_string(first);
    REQUIRE(__quantum__rt__string_equal(qstr, one_str));

    QUBIT *last = *reinterpret_cast<QUBIT **>(__quantum__rt__array_get_element_ptr_1d(qs, 2));
    qstr = __quantum__rt__qubit_to_string(last);
    REQUIRE(__quantum__rt__string_equal(qstr, three_str));

    __quantum__rt__string_update_reference_count(qstr, -1);

    QirArray *copy = __quantum__rt__array_copy(qs, true /*force*/);

    __quantum__rt__qubit_release_array(qs); // The `qs` is a dangling pointer from now on.
    __quantum__rt__array_update_reference_count(copy, -1);

    __quantum__rt__finalize();
}

TEST_CASE("Test lightning__core__qis methods", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    constexpr double angle = 0.42;

    QirArray *reg = __quantum__rt__qubit_allocate_array(3);
    QUBIT *target = *reinterpret_cast<QUBIT **>(__quantum__rt__array_get_element_ptr_1d(reg, 2));

    __quantum__qis__RY(angle, target);
    __quantum__qis__RX(angle, target);

    // The `ctrls` is a dangling pointer from now on.
    __quantum__rt__qubit_release_array(reg);

    __quantum__rt__finalize();

    REQUIRE(true); // if the __quantum__qis__ operations can be called
}

TEST_CASE("Test __quantum__rt__initialize multiple times", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    REQUIRE_THROWS_WITH(__quantum__rt__initialize(),
                        Catch::Contains("Invalid initialization of the global simulator"));

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__rt__print_state", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    __quantum__rt__qubit_allocate_array(2);

    std::string expected = "*** State-Vector of Size 4 ***\n[(1,0), (0,0), (0,0), (0,0)]\n";
    std::stringstream buffer;

    std::streambuf *prevcoutbuf = std::cout.rdbuf(buffer.rdbuf());
    __quantum__rt__print_state();
    std::cout.rdbuf(prevcoutbuf);

    std::string result = buffer.str();
    REQUIRE(!result.compare(expected));

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__State with wires", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *wire0 = __quantum__rt__qubit_allocate();

    __quantum__rt__qubit_allocate();

    MemRefT_CplxT_double_1d *result = new MemRefT_CplxT_double_1d;
    REQUIRE_THROWS_WITH(__quantum__qis__State(result, 1, wire0),
                        Catch::Contains("Partial State-Vector not supported yet"));

    delete result;
    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__Identity", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *wire0 = __quantum__rt__qubit_allocate();

    __quantum__rt__qubit_allocate();

    __quantum__qis__Identity(wire0);

    MemRefT_CplxT_double_1d *result = new MemRefT_CplxT_double_1d;
    __quantum__qis__State(result, 0);
    CplxT_double *state = result->data_allocated;

    REQUIRE(state[0].real == Approx(1.0).margin(1e-5));
    REQUIRE(state[0].imag == Approx(0.0).margin(1e-5));
    REQUIRE(state[1].real == Approx(0.0).margin(1e-5));
    REQUIRE(state[1].imag == Approx(0.0).margin(1e-5));

    free(state);
    delete result;

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__PauliX", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *wire0 = __quantum__rt__qubit_allocate();

    __quantum__rt__qubit_allocate();

    __quantum__qis__PauliX(wire0);

    MemRefT_CplxT_double_1d *result = new MemRefT_CplxT_double_1d;
    __quantum__qis__State(result, 0);
    CplxT_double *state = result->data_allocated;

    REQUIRE(
        (state[0].real == Approx(0.0).margin(1e-5) && state[0].imag == Approx(0.0).margin(1e-5)));
    REQUIRE(
        (state[2].real == Approx(1.0).margin(1e-5) && state[2].imag == Approx(0.0).margin(1e-5)));

    free(state);
    delete result;

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__ PauliY and Rot", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *wire0 = __quantum__rt__qubit_allocate();

    __quantum__rt__qubit_allocate();

    __quantum__qis__PauliY(wire0);
    __quantum__qis__Rot(0.4, 0.6, -0.2, wire0);

    MemRefT_CplxT_double_1d *result = new MemRefT_CplxT_double_1d;
    __quantum__qis__State(result, 0);
    CplxT_double *state = result->data_allocated;

    REQUIRE((state[0].real == Approx(0.0873321925).margin(1e-5) &&
             state[0].imag == Approx(-0.2823212367).margin(1e-5)));
    REQUIRE((state[2].real == Approx(-0.0953745058).margin(1e-5) &&
             state[2].imag == Approx(0.9505637859).margin(1e-5)));

    free(state);
    delete result;
    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__Measure", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *wire0 = __quantum__rt__qubit_allocate();

    __quantum__rt__qubit_allocate();

    __quantum__qis__PauliX(wire0);

    Result m = __quantum__qis__Measure(wire0);

    Result one = __quantum__rt__result_get_one();
    REQUIRE(*m == *one);

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__ Hadamard, PauliZ, IsingXX, IsingZZ, and SWAP",
          "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate();
    QirArray *ctrls_arr = __quantum__rt__qubit_allocate_array(1);

    QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);

    __quantum__qis__Hadamard(target);
    __quantum__qis__PauliZ(target);
    __quantum__qis__IsingXX(0.2, *ctrls, target);
    __quantum__qis__IsingZZ(0.5, *ctrls, target);
    __quantum__qis__SWAP(*ctrls, target);

    MemRefT_CplxT_double_1d *result = new MemRefT_CplxT_double_1d;
    __quantum__qis__State(result, 0);
    CplxT_double *state = result->data_allocated;

    REQUIRE((state[0].real == Approx(0.6817017748).margin(1e-5) &&
             state[0].imag == Approx(-0.1740670409).margin(1e-5)));
    REQUIRE((state[1].real == Approx(-0.6817017748).margin(1e-5) &&
             state[1].imag == Approx(-0.1740670409).margin(1e-5)));
    REQUIRE((state[2].real == Approx(-0.0174649595).margin(1e-5) &&
             state[2].imag == Approx(0.068398324).margin(1e-5)));
    REQUIRE((state[3].real == Approx(-0.0174649595).margin(1e-5) &&
             state[3].imag == Approx(-0.068398324).margin(1e-5)));

    free(state);
    delete result;

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__ CRot, IsingXY and Toffoli", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate();
    QirArray *ctrls_arr = __quantum__rt__qubit_allocate_array(2);

    QUBIT **ctrls_0 = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);
    QUBIT **ctrls_1 = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 1);

    __quantum__qis__Hadamard(target);
    __quantum__qis__PauliZ(target);
    __quantum__qis__CRot(0.2, 0.5, 0.7, *ctrls_0, target);
    __quantum__qis__IsingXY(0.2, *ctrls_0, target);
    __quantum__qis__SWAP(*ctrls_0, target);
    __quantum__qis__Toffoli(*ctrls_0, *ctrls_1, target);

    MemRefT_CplxT_double_1d *result = new MemRefT_CplxT_double_1d;
    __quantum__qis__State(result, 0);
    CplxT_double *state = result->data_allocated;

    REQUIRE((state[0].real == Approx(0.70710678).margin(1e-5) &&
             state[0].imag == Approx(0.0).margin(1e-5)));
    REQUIRE(
        (state[1].real == Approx(0.0).margin(1e-5) && state[1].imag == Approx(0.0).margin(1e-5)));
    REQUIRE((state[2].real == Approx(-0.7035741926).margin(1e-5) &&
             state[2].imag == Approx(0.0).margin(1e-5)));
    REQUIRE(
        (state[3].real == Approx(0.0).margin(1e-5) && state[3].imag == Approx(0.0).margin(1e-5)));

    free(state);
    delete result;

    __quantum__rt__finalize();
}

#ifndef _KOKKOS
TEST_CASE("Test __quantum__qis__ Hadamard, PauliX, IsingYY, CRX, and Expval",
          "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate();
    QirArray *ctrls_arr = __quantum__rt__qubit_allocate_array(1);

    QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);

    // qml.Hadamard(wires=0)
    __quantum__qis__Hadamard(target);
    // qml.PauliX(wires=0)
    __quantum__qis__PauliX(target);
    // qml.IsingYY(0.2, wires=[1,0])
    __quantum__qis__IsingYY(0.2, *ctrls, target);
    // qml.CRX(0.4, wires=[1,0])
    __quantum__qis__CRX(0.4, *ctrls, target);

    MemRefT_CplxT_double_1d *result = new MemRefT_CplxT_double_1d;
    __quantum__qis__State(result, 0);
    CplxT_double *state = result->data_allocated;

    REQUIRE((state[0].real == Approx(0.70357419).margin(1e-5) &&
             state[0].imag == Approx(0.0).margin(1e-5)));
    REQUIRE((state[1].real == Approx(0.01402464).margin(1e-5) &&
             state[1].imag == Approx(-0.06918573).margin(1e-5)));
    REQUIRE((state[2].real == Approx(0.70357419).margin(1e-5) &&
             state[2].imag == Approx(0).margin(1e-5)));
    REQUIRE((state[3].real == Approx(-0.01402464).margin(1e-5) &&
             state[3].imag == Approx(0.06918573).margin(1e-5)));

    // qml.expval(qml.Hadamard(wires=1))
    auto obs = __quantum__qis__NamedObs(ObsId::Hadamard, *ctrls);

    REQUIRE(__quantum__qis__Expval(obs) == Approx(0.69301172).margin(1e-5));

    free(state);
    delete result;

    __quantum__rt__finalize();
}
#endif

TEST_CASE("Test __quantum__qis__ PhaseShift", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate();
    QirArray *ctrls_arr = __quantum__rt__qubit_allocate_array(1);

    QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);

    // qml.Hadamard(wires=0)
    __quantum__qis__Hadamard(target);
    // qml.RX(0.123, wires=1)
    __quantum__qis__RX(0.123, *ctrls);
    // qml.PhaseShift(0.456, wires=0)
    __quantum__qis__PhaseShift(0.456, target);

    MemRefT_CplxT_double_1d *result = new MemRefT_CplxT_double_1d;
    __quantum__qis__State(result, 0);
    CplxT_double *state = result->data_allocated;

    REQUIRE((state[0].real == Approx(0.7057699753).margin(1e-5) &&
             state[0].imag == Approx(0.0).margin(1e-5)));
    REQUIRE((state[1].real == Approx(0.0).margin(1e-5) &&
             state[1].imag == Approx(-0.04345966).margin(1e-5)));
    REQUIRE((state[2].real == Approx(0.63365519).margin(1e-5) &&
             state[2].imag == Approx(0.31079312).margin(1e-5)));
    REQUIRE((state[3].real == Approx(0.01913791).margin(1e-5) &&
             state[3].imag == Approx(-0.039019).margin(1e-5)));

    free(state);
    delete result;

    __quantum__rt__finalize();
}

#ifndef _KOKKOS
TEST_CASE("Test __quantum__qis__HermitianObs with an uninitialized matrix", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    MemRefT_CplxT_double_2d *matrix = nullptr;
    REQUIRE_THROWS_WITH(__quantum__qis__HermitianObs(matrix, 0),
                        Catch::Contains("The Hermitian matrix must be initialized"));

    delete matrix;
    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__HermitianObs with invalid Hermitian matrix", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    MemRefT_CplxT_double_2d *matrix = new MemRefT_CplxT_double_2d;
    REQUIRE_THROWS_WITH(__quantum__qis__HermitianObs(matrix, 0),
                        Catch::Contains("Invalid given Hermitian matrix"));

    delete matrix;
    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__HermitianObs with invalid number of wires", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate();

    MemRefT_CplxT_double_2d *matrix = new MemRefT_CplxT_double_2d;
    matrix->offset = 0;
    matrix->sizes[0] = 4;
    matrix->sizes[1] = 4;
    matrix->strides[0] = 1;
    REQUIRE_THROWS_WITH(__quantum__qis__HermitianObs(matrix, 2, target, target),
                        Catch::Contains("Invalid number of wires"));

    delete matrix;
    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__HermitianObs and Expval", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate();
    QirArray *ctrls_arr = __quantum__rt__qubit_allocate_array(1);

    QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);

    // qml.Hadamard(wires=0)
    __quantum__qis__Hadamard(target);
    // qml.PauliX(wires=0)
    __quantum__qis__PauliX(target);
    // qml.IsingYY(0.2, wires=[1,0])
    __quantum__qis__IsingYY(0.2, *ctrls, target);
    // qml.CRX(0.4, wires=[1,0])
    __quantum__qis__CRX(0.4, *ctrls, target);

    MemRefT_CplxT_double_1d *result = new MemRefT_CplxT_double_1d;
    __quantum__qis__State(result, 0);
    CplxT_double *state = result->data_allocated;

    REQUIRE((state[0].real == Approx(0.70357419).margin(1e-5) &&
             state[0].imag == Approx(0.0).margin(1e-5)));
    REQUIRE((state[1].real == Approx(0.01402464).margin(1e-5) &&
             state[1].imag == Approx(-0.06918573).margin(1e-5)));
    REQUIRE((state[2].real == Approx(0.70357419).margin(1e-5) &&
             state[2].imag == Approx(0).margin(1e-5)));
    REQUIRE((state[3].real == Approx(-0.01402464).margin(1e-5) &&
             state[3].imag == Approx(0.06918573).margin(1e-5)));

    // qml.Hermitan(qml.Hermitian({non-zero-matrix}, wires=[0,1]))

    // just to update obs_index
    __quantum__qis__NamedObs(ObsId::Hadamard, *ctrls);

    CplxT_double matrix_data[4] = {{1.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}, {0.0, 0.0}};

    MemRefT_CplxT_double_2d *h_matrix = new MemRefT_CplxT_double_2d;
    h_matrix->data_allocated = matrix_data;
    h_matrix->data_aligned = matrix_data;
    h_matrix->offset = 0;
    h_matrix->sizes[0] = 2;
    h_matrix->sizes[1] = 2;
    h_matrix->strides[0] = 1;

    auto obs_h = __quantum__qis__HermitianObs(h_matrix, 1, *ctrls);

    REQUIRE(obs_h == 1);
    REQUIRE(__quantum__qis__Expval(obs_h) == Approx(0.9900332889).margin(1e-5));

    free(state);
    delete result;
    delete h_matrix;

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__TensorObs with invalid number of observables",
          "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    REQUIRE_THROWS_WITH(__quantum__qis__TensorObs(0),
                        Catch::Contains("Invalid number of observables to create TensorProdObs"));

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__TensorProdObs and Expval", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate();
    QirArray *ctrls_arr = __quantum__rt__qubit_allocate_array(1);

    QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);

    // qml.Hadamard(wires=0)
    __quantum__qis__Hadamard(target);
    // qml.Hadamard(wires=1)
    __quantum__qis__Hadamard(*ctrls);
    // qml.IsingYY(0.2, wires=[1,0])
    __quantum__qis__IsingYY(0.6, *ctrls, target);
    // qml.CRX(0.4, wires=[1,0])
    __quantum__qis__CRX(0.3, *ctrls, target);

    MemRefT_CplxT_double_1d *result = new MemRefT_CplxT_double_1d;
    __quantum__qis__State(result, 0);
    CplxT_double *state = result->data_allocated;

    auto obs_x = __quantum__qis__NamedObs(ObsId::PauliX, target);

    CplxT_double matrix_data[4] = {{1.0, 0.0}, {0.0, 3.0}, {2.0, 0.0}, {0.0, 5.0}};

    MemRefT_CplxT_double_2d *h_matrix = new MemRefT_CplxT_double_2d;
    h_matrix->data_allocated = matrix_data;
    h_matrix->data_aligned = matrix_data;
    h_matrix->offset = 0;
    h_matrix->sizes[0] = 2;
    h_matrix->sizes[1] = 2;
    h_matrix->strides[0] = 1;

    auto obs_h = __quantum__qis__HermitianObs(h_matrix, 1, *ctrls);

    auto obs_t = __quantum__qis__TensorObs(2, obs_h, obs_x);

    REQUIRE(obs_t == 2);
    REQUIRE(__quantum__qis__Expval(obs_t) == Approx(1.5864438048).margin(1e-5));

    free(state);
    delete result;
    delete h_matrix;

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__HamiltonianObs with invalid coefficients", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    MemRefT_double_1d *coeffs = nullptr;

    REQUIRE_THROWS_WITH(__quantum__qis__HamiltonianObs(coeffs, 0),
                        Catch::Contains("Invalid coefficients for computing Hamiltonian"));

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__HamiltonianObs with invalid number of coefficients and observables",
          "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    MemRefT_double_1d *coeffs = new MemRefT_double_1d;
    coeffs->offset = 0;
    coeffs->sizes[0] = 2;
    coeffs->strides[0] = 1;

    REQUIRE_THROWS_WITH(__quantum__qis__HamiltonianObs(coeffs, 0),
                        Catch::Contains("Invalid coefficients for computing Hamiltonian"));

    delete coeffs;
    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__HamiltonianObs(h, x) and Expval", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate();
    QirArray *ctrls_arr = __quantum__rt__qubit_allocate_array(1);

    QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);

    // qml.Hadamard(wires=0)
    __quantum__qis__Hadamard(target);
    // qml.Hadamard(wires=1)
    __quantum__qis__Hadamard(*ctrls);
    // qml.IsingYY(0.2, wires=[1,0])
    __quantum__qis__IsingYY(0.6, *ctrls, target);
    // qml.CRX(0.4, wires=[1,0])
    __quantum__qis__CRX(0.3, *ctrls, target);

    MemRefT_CplxT_double_1d *result = new MemRefT_CplxT_double_1d;
    __quantum__qis__State(result, 0);
    CplxT_double *state = result->data_allocated;

    auto obs_x = __quantum__qis__NamedObs(ObsId::PauliX, target);

    CplxT_double matrix_data[4] = {{1.0, 0.0}, {0.0, 3.0}, {2.0, 0.0}, {0.0, 5.0}};

    MemRefT_CplxT_double_2d *h_matrix = new MemRefT_CplxT_double_2d;
    h_matrix->data_allocated = matrix_data;
    h_matrix->data_aligned = matrix_data;
    h_matrix->offset = 0;
    h_matrix->sizes[0] = 2;
    h_matrix->sizes[1] = 2;
    h_matrix->strides[0] = 1;

    auto obs_h = __quantum__qis__HermitianObs(h_matrix, 1, *ctrls);

    double coeffs_data[2] = {0.4, 0.7};
    MemRefT_double_1d *coeffs = new MemRefT_double_1d;
    coeffs->data_allocated = coeffs_data;
    coeffs->data_aligned = coeffs_data;
    coeffs->offset = 0;
    coeffs->sizes[0] = 2;
    coeffs->strides[0] = 1;

    auto obs_hamiltonian = __quantum__qis__HamiltonianObs(coeffs, 2, obs_h, obs_x);

    REQUIRE(obs_hamiltonian == 2);
    REQUIRE(__quantum__qis__Expval(obs_hamiltonian) == Approx(1.1938250042).margin(1e-5));

    free(state);
    delete result;
    delete h_matrix;
    delete coeffs;

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__HamiltonianObs(t) and Expval", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate();
    QirArray *ctrls_arr = __quantum__rt__qubit_allocate_array(1);

    QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);

    // qml.Hadamard(wires=0)
    __quantum__qis__Hadamard(target);
    // qml.Hadamard(wires=1)
    __quantum__qis__Hadamard(*ctrls);
    // qml.IsingYY(0.2, wires=[1,0])
    __quantum__qis__IsingYY(0.6, *ctrls, target);
    // qml.CRX(0.4, wires=[1,0])
    __quantum__qis__CRX(0.3, *ctrls, target);

    MemRefT_CplxT_double_1d *result = new MemRefT_CplxT_double_1d;
    __quantum__qis__State(result, 0);
    CplxT_double *state = result->data_allocated;

    auto obs_x = __quantum__qis__NamedObs(ObsId::PauliX, target);

    CplxT_double matrix_data[4] = {{1.0, 0.0}, {0.0, 3.0}, {2.0, 0.0}, {0.0, 5.0}};

    MemRefT_CplxT_double_2d *h_matrix = new MemRefT_CplxT_double_2d;
    h_matrix->data_allocated = matrix_data;
    h_matrix->data_aligned = matrix_data;
    h_matrix->offset = 0;
    h_matrix->sizes[0] = 2;
    h_matrix->sizes[1] = 2;
    h_matrix->strides[0] = 1;

    auto obs_h = __quantum__qis__HermitianObs(h_matrix, 1, *ctrls);

    auto obs_t = __quantum__qis__TensorObs(2, obs_h, obs_x);

    double coeffs_data[1] = {0.4};
    MemRefT_double_1d *coeffs = new MemRefT_double_1d;
    coeffs->data_allocated = coeffs_data;
    coeffs->data_aligned = coeffs_data;
    coeffs->offset = 0;
    coeffs->sizes[0] = 1;
    coeffs->strides[0] = 1;

    auto obs_hamiltonian = __quantum__qis__HamiltonianObs(coeffs, 1, obs_t);

    REQUIRE(obs_hamiltonian == 3);
    REQUIRE(__quantum__qis__Expval(obs_hamiltonian) == Approx(0.6345775219).margin(1e-5));

    free(state);
    delete result;
    delete h_matrix;
    delete coeffs;

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__ Hadamard, PauliX, IsingYY, CRX, and Expval_arr",
          "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate();
    QirArray *ctrls_arr = __quantum__rt__qubit_allocate_array(1);

    QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);

    // qml.Hadamard(wires=0)
    __quantum__qis__Hadamard(target);
    // qml.PauliX(wires=0)
    __quantum__qis__PauliX(target);
    // qml.IsingYY(0.2, wires=[1,0])
    __quantum__qis__IsingYY(0.2, *ctrls, target);
    // qml.CRX(0.4, wires=[1,0])
    __quantum__qis__CRX(0.4, *ctrls, target);

    MemRefT_CplxT_double_1d *result = new MemRefT_CplxT_double_1d;
    __quantum__qis__State(result, 0);
    CplxT_double *state = result->data_allocated;

    REQUIRE((state[0].real == Approx(0.70357419).margin(1e-5) &&
             state[0].imag == Approx(0.0).margin(1e-5)));
    REQUIRE((state[1].real == Approx(0.01402464).margin(1e-5) &&
             state[1].imag == Approx(-0.06918573).margin(1e-5)));
    REQUIRE((state[2].real == Approx(0.70357419).margin(1e-5) &&
             state[2].imag == Approx(0).margin(1e-5)));
    REQUIRE((state[3].real == Approx(-0.01402464).margin(1e-5) &&
             state[3].imag == Approx(0.06918573).margin(1e-5)));

    // qml.expval(qml.Hadamard(wires=1))
    QUBIT **qubit = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);
    auto obs = __quantum__qis__NamedObs(ObsId::Hadamard, *qubit);

    REQUIRE(__quantum__qis__Expval(obs) == Approx(0.69301172).margin(1e-5));

    free(state);
    delete result;

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__ Hadamard, ControlledPhaseShift, IsingYY, CRX, and Var",
          "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate();              // id = 0
    QirArray *ctrls_arr = __quantum__rt__qubit_allocate_array(1); // id = 1

    QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);

    // qml.Hadamard(wires=0)
    __quantum__qis__Hadamard(target);
    // qml.ControlledPhaseShift(0.6, wires=[0,1])
    __quantum__qis__ControlledPhaseShift(0.6, target, *ctrls);
    // qml.IsingYY(0.2, wires=[0, 1])
    __quantum__qis__IsingYY(0.2, target, *ctrls);
    // qml.CRX(0.4, wires=[1,0])
    __quantum__qis__CRX(0.4, target, *ctrls);

    MemRefT_CplxT_double_1d *result = new MemRefT_CplxT_double_1d;
    __quantum__qis__State(result, 0);
    CplxT_double *state = result->data_allocated;

    REQUIRE((state[0].real == Approx(0.70357419).margin(1e-5) &&
             state[0].imag == Approx(0.0).margin(1e-5)));
    REQUIRE((state[1].real == Approx(0.0).margin(1e-5) &&
             state[1].imag == Approx(-0.0705929).margin(1e-5)));
    REQUIRE((state[2].real == Approx(0.70357419).margin(1e-5) &&
             state[2].imag == Approx(0).margin(1e-5)));
    REQUIRE((state[3].real == Approx(0.0).margin(1e-5) &&
             state[3].imag == Approx(-0.0705929).margin(1e-5)));

    // qml.var(qml.PauliZ(wires=1))
    auto obs = __quantum__qis__NamedObs(ObsId::PauliZ, *ctrls);

    REQUIRE(__quantum__qis__Variance(obs) == Approx(0.0394695).margin(1e-5));

    free(state);
    delete result;

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__ Hadamard, ControlledPhaseShift, IsingYgitY, "
          "CRX, and Var_arr",
          "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate();              // id = 0
    QirArray *ctrls_arr = __quantum__rt__qubit_allocate_array(1); // id = 1

    QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);

    // qml.Hadamard(wires=0)
    __quantum__qis__Hadamard(target);
    // qml.ControlledPhaseShift(0.6, wires=[0,1])
    __quantum__qis__ControlledPhaseShift(0.6, target, *ctrls);
    // qml.IsingYY(0.2, wires=[0, 1])
    __quantum__qis__IsingYY(0.2, target, *ctrls);
    // qml.CRX(0.4, wires=[1,0])
    __quantum__qis__CRX(0.4, target, *ctrls);

    MemRefT_CplxT_double_1d *result = new MemRefT_CplxT_double_1d;
    __quantum__qis__State(result, 0);
    CplxT_double *state = result->data_allocated;

    REQUIRE((state[0].real == Approx(0.70357419).margin(1e-5) &&
             state[0].imag == Approx(0.0).margin(1e-5)));
    REQUIRE((state[1].real == Approx(0.0).margin(1e-5) &&
             state[1].imag == Approx(-0.0705929).margin(1e-5)));
    REQUIRE((state[2].real == Approx(0.70357419).margin(1e-5) &&
             state[2].imag == Approx(0).margin(1e-5)));
    REQUIRE((state[3].real == Approx(0.0).margin(1e-5) &&
             state[3].imag == Approx(-0.0705929).margin(1e-5)));

    // qml.var(qml.PauliZ(wires=1))
    QUBIT **qubit = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);
    auto obs = __quantum__qis__NamedObs(ObsId::Hadamard, *qubit);

    REQUIRE(__quantum__qis__Variance(obs) == Approx(0.5197347515).margin(1e-5));

    free(state);
    delete result;

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__ Hadamard, PauliX, IsingYY, CRX, and Probs", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate();
    QirArray *ctrls_arr = __quantum__rt__qubit_allocate_array(1);

    QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);

    // qml.Hadamard(wires=0)
    __quantum__qis__Hadamard(target);
    // qml.PauliX(wires=0)
    __quantum__qis__PauliX(target);
    // qml.IsingYY(0.2, wires=[1,0])
    __quantum__qis__IsingYY(0.2, *ctrls, target);
    // qml.CRX(0.4, wires=[1,0])
    __quantum__qis__CRX(0.4, *ctrls, target);

    // double probs[2];
    MemRefT_double_1d *result = new MemRefT_double_1d;
    __quantum__qis__Probs(result, 0);
    double *probs = result->data_allocated;

    REQUIRE((probs[0] + probs[2]) == Approx(0.9900332889).margin(1e-5));
    REQUIRE((probs[1] + probs[3]) == Approx(0.0099667111).margin(1e-5));

    free(probs);
    delete result;

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__ Hadamard, PauliX, IsingYY, CRX, and partial Probs",
          "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate();
    QirArray *ctrls_arr = __quantum__rt__qubit_allocate_array(1);

    QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);

    // qml.Hadamard(wires=0)
    __quantum__qis__Hadamard(target);
    // qml.PauliX(wires=0)
    __quantum__qis__PauliX(target);
    // qml.IsingYY(0.2, wires=[1,0])
    __quantum__qis__IsingYY(0.2, *ctrls, target);
    // qml.CRX(0.4, wires=[1,0])
    __quantum__qis__CRX(0.4, *ctrls, target);

    MemRefT_double_1d *result = new MemRefT_double_1d;
    __quantum__qis__Probs(result, 1, ctrls[0]);
    double *probs = result->data_allocated;

    REQUIRE(probs[0] == Approx(0.9900332889).margin(1e-5));
    REQUIRE(probs[1] == Approx(0.0099667111).margin(1e-5));

    free(probs);
    delete result;

    __quantum__rt__finalize();
}
#endif

TEST_CASE("Test __quantum__qis__State on the heap using malloc", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate();
    QirArray *ctrls_arr = __quantum__rt__qubit_allocate_array(1);

    QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);

    // qml.Hadamard(wires=0)
    __quantum__qis__Hadamard(target);
    // qml.ControlledPhaseShift(0.6, wires=[0,1])
    __quantum__qis__ControlledPhaseShift(0.6, target, *ctrls);
    // qml.IsingYY(0.2, wires=[0, 1])
    __quantum__qis__IsingYY(0.2, target, *ctrls);
    // qml.CRX(0.4, wires=[1,0])
    __quantum__qis__CRX(0.4, target, *ctrls);

    MemRefT_CplxT_double_1d *result = new MemRefT_CplxT_double_1d;
    __quantum__qis__State(result, 0);
    CplxT_double *stateVec = result->data_allocated;

    REQUIRE(stateVec[0].real == Approx(0.7035741926).margin(1e-5));
    REQUIRE(stateVec[0].imag == 0.0);

    REQUIRE(stateVec[1].real == 0.0);
    REQUIRE(stateVec[1].imag == Approx(-0.070592886).margin(1e-5));

    REQUIRE(stateVec[2].real == Approx(0.7035741926).margin(1e-5));
    REQUIRE(stateVec[2].imag == 0.0);

    REQUIRE(stateVec[3].real == 0.0);
    REQUIRE(stateVec[3].imag == Approx(-0.070592886).margin(1e-5));

    free(stateVec);
    delete result;

    __quantum__rt__finalize();
}

#ifndef _KOKKOS
TEST_CASE("Test __quantum__qis__Measure with false", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate(); // id = 0

    // qml.Hadamard(wires=0)
    __quantum__qis__RY(0.0, target);

    Result mres = __quantum__qis__Measure(target);

    Result zero = __quantum__rt__result_get_zero();
    REQUIRE(__quantum__rt__result_equal(mres, zero));

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__Measure with true", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate(); // id = 0

    // qml.Hadamard(wires=0)
    __quantum__qis__RY(3.14, target);

    Result mres = __quantum__qis__Measure(target);

    Result one = __quantum__rt__result_get_one();
    REQUIRE(__quantum__rt__result_equal(mres, one));

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__MultiRZ", "[qir_lightning_core]")
{
    __quantum__rt__initialize();

    QUBIT *q0 = __quantum__rt__qubit_allocate();
    QUBIT *q1 = __quantum__rt__qubit_allocate();
    __quantum__qis__RX(M_PI, q0);
    __quantum__qis__Hadamard(q0);
    __quantum__qis__Hadamard(q1);
    __quantum__qis__MultiRZ(M_PI, 2, q0, q1);
    __quantum__qis__Hadamard(q0);
    __quantum__qis__Hadamard(q1);

    Result q0_m = __quantum__qis__Measure(q0);
    Result q1_m = __quantum__qis__Measure(q1);

    Result zero = __quantum__rt__result_get_zero();
    Result one = __quantum__rt__result_get_one();
    REQUIRE(__quantum__rt__result_equal(q0_m, zero));
    REQUIRE(__quantum__rt__result_equal(q1_m, one));

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__CSWAP ", "[qir_lightning_core]")
{
    __quantum__rt__initialize();

    QUBIT *q0 = __quantum__rt__qubit_allocate();
    QUBIT *q1 = __quantum__rt__qubit_allocate();
    QUBIT *q2 = __quantum__rt__qubit_allocate();
    __quantum__qis__RX(M_PI, q0);
    __quantum__qis__RX(M_PI, q1);
    __quantum__qis__CSWAP(q0, q1, q2);

    Result q1_m = __quantum__qis__Measure(q1);
    Result q2_m = __quantum__qis__Measure(q2);

    Result zero = __quantum__rt__result_get_zero();
    Result one = __quantum__rt__result_get_one();

    // Test via rt__result_to_string
    QirString *zero_str = __quantum__rt__result_to_string(zero);
    QirString *q1_m_str = __quantum__rt__result_to_string(q1_m);
    REQUIRE(__quantum__rt__string_equal(zero_str, q1_m_str));

    QirString *one_str = __quantum__rt__result_to_string(one);
    QirString *q2_m_str = __quantum__rt__result_to_string(q2_m);
    REQUIRE(__quantum__rt__string_equal(one_str, q2_m_str));

    // Test via rt__result_equal
    REQUIRE(__quantum__rt__result_equal(q1_m, zero));
    REQUIRE(__quantum__rt__result_equal(q2_m, one));

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__Counts with num_qubits=2 calling Hadamard, ControlledPhaseShift, "
          "IsingYY, and CRX quantum operations",
          "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate();              // id = 0
    QirArray *ctrls_arr = __quantum__rt__qubit_allocate_array(1); // id = 1

    QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);

    // qml.Hadamard(wires=0)
    __quantum__qis__Hadamard(target);
    // qml.ControlledPhaseShift(0.6, wires=[0,1])
    __quantum__qis__ControlledPhaseShift(0.6, target, *ctrls);
    // qml.IsingYY(0.2, wires=[0, 1])
    __quantum__qis__IsingYY(0.2, target, *ctrls);
    // qml.CRX(0.4, wires=[1,0])
    __quantum__qis__CRX(0.4, target, *ctrls);

    constexpr size_t shots = 1000;

    PairT_MemRefT_double_int64_1d *result = new PairT_MemRefT_double_int64_1d();
    __quantum__qis__Counts(result, shots, 0);
    int64_t *counts = result->second.data_allocated;
    double *eigvals = result->first.data_allocated;

    for (int i = 0; i < 4; i++) {
        REQUIRE(eigvals[i] == (double)i);
    }

    size_t sum = 0;
    for (int i = 0; i < 4; i++) {
        sum += counts[i];
    }
    REQUIRE(sum == shots);

    __quantum__rt__finalize();

    free(counts);
    free(eigvals);

    delete result;
}

TEST_CASE("Test __quantum__qis__Counts with num_qubits=2 PartialCounts calling Hadamard, "
          "ControlledPhaseShift, IsingYY, and CRX quantum operations",
          "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate();              // id = 0
    QirArray *ctrls_arr = __quantum__rt__qubit_allocate_array(1); // id = 1

    QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);

    // qml.Hadamard(wires=0)
    __quantum__qis__Hadamard(target);
    // qml.ControlledPhaseShift(0.6, wires=[0,1])
    __quantum__qis__ControlledPhaseShift(0.6, target, *ctrls);
    // qml.IsingYY(0.2, wires=[0, 1])
    __quantum__qis__IsingYY(0.2, target, *ctrls);
    // qml.CRX(0.4, wires=[1,0])
    __quantum__qis__CRX(0.4, target, *ctrls);

    constexpr size_t shots = 1000;

    PairT_MemRefT_double_int64_1d *result = new PairT_MemRefT_double_int64_1d();
    __quantum__qis__Counts(result, shots, 1, ctrls[0]);
    int64_t *counts = result->second.data_allocated;
    double *eigvals = result->first.data_allocated;

    REQUIRE(counts[0] + counts[1] == shots);
    REQUIRE(eigvals[0] + 1 == eigvals[1]);

    __quantum__rt__finalize();

    free(counts);
    free(eigvals);

    delete result;
}

TEST_CASE("Test __quantum__qis__Sample with num_qubits=2 calling Hadamard, ControlledPhaseShift, "
          "IsingYY, and CRX quantum operations",
          "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate();              // id = 0
    QirArray *ctrls_arr = __quantum__rt__qubit_allocate_array(1); // id = 1

    QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);

    // qml.Hadamard(wires=0)
    __quantum__qis__Hadamard(target);
    // qml.ControlledPhaseShift(0.6, wires=[0,1])
    __quantum__qis__ControlledPhaseShift(0.6, target, *ctrls);
    // qml.IsingYY(0.2, wires=[0, 1])
    __quantum__qis__IsingYY(0.2, target, *ctrls);
    // qml.CRX(0.4, wires=[1,0])
    __quantum__qis__CRX(0.4, target, *ctrls);

    constexpr size_t n = 2;
    constexpr size_t shots = 1000;

    MemRefT_double_2d *result = new MemRefT_double_2d;
    __quantum__qis__Sample(result, shots, 0);
    double *samples = result->data_allocated;

    size_t counts0[2] = {0, 0};
    for (size_t idx = 0; idx < shots * n; idx += n) {
        if (samples[idx] == 0) {
            counts0[0]++;
        }
        else {
            counts0[1]++;
        }
    }

    REQUIRE(counts0[0] + counts0[1] == shots);

    size_t counts1[2] = {0, 0};
    for (size_t idx = 1; idx < shots * n; idx += n) {
        if (samples[idx] == 0) {
            counts1[0]++;
        }
        else {
            counts1[1]++;
        }
    }

    REQUIRE(counts1[0] + counts1[1] == shots);

    QUBIT **qubit = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);
    auto obs = __quantum__qis__NamedObs(ObsId::PauliZ, *qubit);

    REQUIRE(__quantum__qis__Variance(obs) == Approx(0.0394695).margin(1e-5));

    __quantum__rt__finalize();

    free(samples);
    delete result;
}

TEST_CASE("Test __quantum__qis__Sample with num_qubits=2 and PartialSample calling Hadamard, "
          "ControlledPhaseShift, IsingYY, and CRX quantum operations",
          "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate();              // id = 0
    QirArray *ctrls_arr = __quantum__rt__qubit_allocate_array(1); // id = 1

    QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);

    // qml.Hadamard(wires=0)
    __quantum__qis__Hadamard(target);
    // qml.ControlledPhaseShift(0.6, wires=[0,1])
    __quantum__qis__ControlledPhaseShift(0.6, target, *ctrls);
    // qml.IsingYY(0.2, wires=[0, 1])
    __quantum__qis__IsingYY(0.2, target, *ctrls);
    // qml.CRX(0.4, wires=[1,0])
    __quantum__qis__CRX(0.4, target, *ctrls);

    constexpr size_t n = 1;
    constexpr size_t shots = 1000;

    MemRefT_double_2d *result = new MemRefT_double_2d;
    __quantum__qis__Sample(result, shots, 1, ctrls[0]);
    double *samples = result->data_allocated;

    size_t counts0[2] = {0, 0};
    for (size_t idx = 0; idx < shots * n; idx += n) {
        if (samples[idx] == 0) {
            counts0[0]++;
        }
        else {
            counts0[1]++;
        }
    }

    REQUIRE(counts0[0] + counts0[1] == shots);

    QUBIT **qubit = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);
    auto obs = __quantum__qis__NamedObs(ObsId::PauliZ, *qubit);

    REQUIRE(__quantum__qis__Variance(obs) == Approx(0.0394695).margin(1e-5));

    __quantum__rt__finalize();

    free(samples);
    delete result;
}

TEST_CASE("Test __quantum__qis__QubitUnitary with an uninitialized matrix", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate(); // id = 0
    MemRefT_CplxT_double_2d *matrix = nullptr;

    REQUIRE_THROWS_WITH(__quantum__qis__QubitUnitary(matrix, 1, target),
                        Catch::Contains("The QubitUnitary matrix must be initialized"));

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__QubitUnitary with invalid number of wires", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate(); // id = 0
    MemRefT_CplxT_double_2d *matrix = new MemRefT_CplxT_double_2d;

    REQUIRE_THROWS_WITH(__quantum__qis__QubitUnitary(matrix, 3, target),
                        Catch::Contains("Invalid number of wires"));

    delete matrix;
    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__QubitUnitary with invalid matrix", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate(); // id = 0
    MemRefT_CplxT_double_2d *matrix = new MemRefT_CplxT_double_2d;

    matrix->offset = 0;
    matrix->sizes[0] = 1;
    matrix->sizes[1] = 1;
    matrix->strides[0] = 1;

    REQUIRE_THROWS_WITH(__quantum__qis__QubitUnitary(matrix, 1, target),
                        Catch::Contains("Invalid given QubitUnitary matrix"));

    delete matrix;
    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__QubitUnitary with num_qubits=2", "[qir_lightning_core]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate();              // id = 0
    QirArray *ctrls_arr = __quantum__rt__qubit_allocate_array(1); // id = 1

    QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);

    __quantum__qis__Hadamard(target);
    __quantum__qis__CNOT(target, *ctrls);

    CplxT_double matrix_data[4] = {
        {-0.6709485262524046, -0.6304426335363695},
        {-0.14885403153998722, 0.3608498832392019},
        {-0.2376311670004963, 0.3096798175687841},
        {-0.8818365947322423, -0.26456390390903695},
    };

    MemRefT_CplxT_double_2d *matrix = new MemRefT_CplxT_double_2d;
    matrix->data_allocated = matrix_data;
    matrix->data_aligned = matrix_data;
    matrix->offset = 0;
    matrix->sizes[0] = 2;
    matrix->sizes[1] = 2;
    matrix->strides[0] = 1;

    __quantum__qis__QubitUnitary(matrix, 1, target);

    MemRefT_CplxT_double_1d *result = new MemRefT_CplxT_double_1d;
    __quantum__qis__State(result, 0);
    CplxT_double *state = result->data_allocated;

    REQUIRE(state[0].real == Approx(-0.474432).margin(1e-5));
    REQUIRE(state[0].imag == Approx(-0.44579).margin(1e-5));
    REQUIRE(state[1].real == Approx(-0.105256).margin(1e-5));
    REQUIRE(state[1].imag == Approx(0.255159).margin(1e-5));
    REQUIRE(state[2].real == Approx(-0.168031).margin(1e-5));
    REQUIRE(state[2].imag == Approx(0.218977).margin(1e-5));
    REQUIRE(state[3].real == Approx(-0.623553).margin(1e-5));
    REQUIRE(state[3].imag == Approx(-0.187075).margin(1e-5));

    free(state);
    delete result;
    delete matrix;
    __quantum__rt__finalize();
}
#endif
