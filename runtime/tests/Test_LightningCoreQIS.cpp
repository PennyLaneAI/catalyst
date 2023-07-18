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

#include "MemRefUtils.hpp"
#include "QuantumDevice.hpp"
#include "RuntimeCAPI.h"
#include "Utils.hpp"

#include "TestUtils.hpp"

using namespace Catalyst::Runtime;

MemRefT_CplxT_double_1d getState(size_t buffer_len)
{
    CplxT_double *buffer = new CplxT_double[buffer_len];
    MemRefT_CplxT_double_1d result = {buffer, buffer, 0, {buffer_len}, {1}};
    return result;
}

void freeState(MemRefT_CplxT_double_1d &result) { delete[] result.data_allocated; }

PairT_MemRefT_double_int64_1d getCounts(size_t buffer_len)
{
    double *buff_e = new double[buffer_len];
    long *buff_c = new long[buffer_len];
    PairT_MemRefT_double_int64_1d result = {{buff_e, buff_e, 0, {buffer_len}, {1}},
                                            {buff_c, buff_c, 0, {buffer_len}, {1}}};
    return result;
}

void freeCounts(PairT_MemRefT_double_int64_1d &result)
{
    delete[] result.first.data_allocated;
    delete[] result.second.data_allocated;
}

TEST_CASE("Test __quantum__rt__fail_cstr", "[qir_lightning_core]")
{
    REQUIRE_THROWS_WITH(
        __quantum__rt__fail_cstr("Test!"),
        Catch::Contains("[Function:__quantum__rt__fail_cstr] Error in Catalyst Runtime: Test!"));
}

TEST_CASE("Qubits: allocate, release, dump", "[CoreQIS]")
{
    __quantum__rt__initialize();
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QUBIT *q = __quantum__rt__qubit_allocate();

        QirString *zero_str = __quantum__rt__int_to_string(0);
        QirString *one_str = __quantum__rt__int_to_string(1);
        QirString *three_str = __quantum__rt__int_to_string(3);

        QirString *qstr = __quantum__rt__qubit_to_string(q);

        CHECK(__quantum__rt__string_equal(qstr, zero_str));

        __quantum__rt__string_update_reference_count(qstr, -1);

        __quantum__rt__qubit_release(q);

        QirArray *qs = __quantum__rt__qubit_allocate_array(3);

        CHECK(__quantum__rt__array_get_size_1d(qs) == 3);

        QUBIT *first = *reinterpret_cast<QUBIT **>(__quantum__rt__array_get_element_ptr_1d(qs, 0));
        qstr = __quantum__rt__qubit_to_string(first);
        CHECK(__quantum__rt__string_equal(qstr, one_str));

        __quantum__rt__string_update_reference_count(qstr, -1);

        QUBIT *last = *reinterpret_cast<QUBIT **>(__quantum__rt__array_get_element_ptr_1d(qs, 2));
        qstr = __quantum__rt__qubit_to_string(last);
        CHECK(__quantum__rt__string_equal(qstr, three_str));

        __quantum__rt__string_update_reference_count(qstr, -1);

        QirArray *copy = __quantum__rt__array_copy(qs, true /*force*/);

        __quantum__rt__string_update_reference_count(zero_str, -1);
        __quantum__rt__string_update_reference_count(one_str, -1);
        __quantum__rt__string_update_reference_count(three_str, -1);

        __quantum__rt__qubit_release_array(qs); // The `qs` is a dangling pointer from now on.
        __quantum__rt__array_update_reference_count(copy, -1);
    }
    __quantum__rt__finalize();
}

TEST_CASE("Test lightning__core__qis methods", "[CoreQIS]")
{
    __quantum__rt__initialize();
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        constexpr double angle = 0.42;

        QirArray *reg = __quantum__rt__qubit_allocate_array(3);
        QUBIT *target =
            *reinterpret_cast<QUBIT **>(__quantum__rt__array_get_element_ptr_1d(reg, 2));

        __quantum__qis__RY(angle, target, false);
        __quantum__qis__RX(angle, target, false);

        // The `ctrls` is a dangling pointer from now on.
        __quantum__rt__qubit_release_array(reg);

        CHECK(true); // if the __quantum__qis__ operations can be called
    }
    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__rt__print_state", "[CoreQIS]")
{
    __quantum__rt__initialize();
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QirArray *reg = __quantum__rt__qubit_allocate_array(2);

        std::string expected = "*** State-Vector of Size 4 ***\n[(1,0), (0,0), (0,0), (0,0)]\n";
        std::stringstream buffer;

        std::streambuf *prevcoutbuf = std::cout.rdbuf(buffer.rdbuf());
        __quantum__rt__print_state();
        std::cout.rdbuf(prevcoutbuf);

        std::string result = buffer.str();
        CHECK(!result.compare(expected));

        __quantum__rt__qubit_release_array(reg);
    }
    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__State with wires", "[CoreQIS]")
{
    __quantum__rt__initialize();
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QUBIT *wire0 = __quantum__rt__qubit_allocate();

        QUBIT *wire1 = __quantum__rt__qubit_allocate();

        MemRefT_CplxT_double_1d result = getState(8);

        REQUIRE_THROWS_WITH(__quantum__qis__State(&result, 1, wire0),
                            Catch::Contains("[Function:__quantum__qis__State] Error in Catalyst "
                                            "Runtime: Partial State-Vector not supported yet"));

        freeState(result);
        __quantum__rt__qubit_release(wire1);
        __quantum__rt__qubit_release(wire0);
    }
    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__Identity", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QUBIT *wire0 = __quantum__rt__qubit_allocate();

        QUBIT *wire1 = __quantum__rt__qubit_allocate();

        __quantum__qis__Identity(wire0, false);

        MemRefT_CplxT_double_1d result = getState(4);
        __quantum__qis__State(&result, 0);
        CplxT_double *state = result.data_allocated;

        CHECK(state[0].real == Approx(1.0).margin(1e-5));
        CHECK(state[0].imag == Approx(0.0).margin(1e-5));
        CHECK(state[1].real == Approx(0.0).margin(1e-5));
        CHECK(state[1].imag == Approx(0.0).margin(1e-5));

        freeState(result);
        __quantum__rt__qubit_release(wire1);
        __quantum__rt__qubit_release(wire0);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__PauliX", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QUBIT *wire0 = __quantum__rt__qubit_allocate();

        QUBIT *wire1 = __quantum__rt__qubit_allocate();

        __quantum__qis__PauliX(wire0, false);

        MemRefT_CplxT_double_1d result = getState(4);
        __quantum__qis__State(&result, 0);
        CplxT_double *state = result.data_allocated;

        CHECK((state[0].real == Approx(0.0).margin(1e-5) &&
               state[0].imag == Approx(0.0).margin(1e-5)));
        CHECK((state[2].real == Approx(1.0).margin(1e-5) &&
               state[2].imag == Approx(0.0).margin(1e-5)));

        freeState(result);
        __quantum__rt__qubit_release(wire1);
        __quantum__rt__qubit_release(wire0);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__ PauliY and Rot", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QUBIT *wire0 = __quantum__rt__qubit_allocate();
        QUBIT *wire1 = __quantum__rt__qubit_allocate();

        __quantum__qis__PauliY(wire0, false);
        __quantum__qis__Rot(0.4, 0.6, -0.2, wire0, false);

        MemRefT_CplxT_double_1d result = getState(4);
        __quantum__qis__State(&result, 0);
        CplxT_double *state = result.data_allocated;

        CHECK((state[0].real == Approx(0.0873321925).margin(1e-5) &&
               state[0].imag == Approx(-0.2823212367).margin(1e-5)));
        CHECK((state[2].real == Approx(-0.0953745058).margin(1e-5) &&
               state[2].imag == Approx(0.9505637859).margin(1e-5)));

        freeState(result);
        __quantum__rt__qubit_release(wire1);
        __quantum__rt__qubit_release(wire0);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test rank=0 and empty DataView", "[qir_lightning_core]")
{
    std::vector<double> empty_vec;
    DataView<double, 1> zero_size(empty_vec);
    CHECK(zero_size.size() == 0);

    DataView<double, 1> zero_rank(nullptr, 0, nullptr, nullptr);
    CHECK(zero_rank.size() == 0);
}

TEST_CASE("Test copy to strided array", "[qir_lightning_core]")
{
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8};
    size_t buffer_len = data.size();
    double *buffer = data.data();
    MemRefT<double, 2> src = {buffer, buffer, 0, {buffer_len / 2, 2}, {2, 1}};
    DataView<double, 2> src_view(src.data_aligned, src.offset, src.sizes, src.strides);
    CHECK(src_view(0, 0) == buffer[0]);
    CHECK(src_view(0, 1) == buffer[1]);
    CHECK(src_view(1, 0) == buffer[2]);
    CHECK(src_view(1, 1) == buffer[3]);
    CHECK(src_view(2, 0) == buffer[4]);
    CHECK(src_view(2, 1) == buffer[5]);
    CHECK(src_view(3, 0) == buffer[6]);
    CHECK(src_view(3, 1) == buffer[7]);

    size_t buffer_strided_len = buffer_len * 2;
    double *buffer_strided = new double[buffer_strided_len];
    MemRefT<double, 2> dst = {
        buffer_strided, buffer_strided, 0, {buffer_strided_len / 4, 2}, {4, 2}};
    DataView<double, 2> dst_view(dst.data_aligned, dst.offset, dst.sizes, dst.strides);
    for (auto iterD = dst_view.begin(), iterS = src_view.begin(); iterD != dst_view.end();
         iterS++, iterD++) {
        *iterD = *iterS;
    }

    CHECK(buffer_strided[0] == buffer[0]);
    CHECK(buffer_strided[2] == buffer[1]);
    CHECK(buffer_strided[4] == buffer[2]);
    CHECK(buffer_strided[6] == buffer[3]);
    CHECK(buffer_strided[8] == buffer[4]);
    CHECK(buffer_strided[10] == buffer[5]);
    CHECK(buffer_strided[12] == buffer[6]);
    CHECK(buffer_strided[14] == buffer[7]);

    CHECK(src_view(0, 0) == dst_view(0, 0));
    CHECK(src_view(0, 1) == dst_view(0, 1));
    CHECK(src_view(1, 0) == dst_view(1, 0));
    CHECK(src_view(1, 1) == dst_view(1, 1));
    CHECK(src_view(2, 0) == dst_view(2, 0));
    CHECK(src_view(2, 1) == dst_view(2, 1));
    CHECK(src_view(3, 0) == dst_view(3, 0));
    CHECK(src_view(3, 1) == dst_view(3, 1));

    REQUIRE_THROWS_WITH(dst_view(4, 1),
                        Catch::Contains("[Function:operator()] Error in Catalyst Runtime: "
                                        "Assertion: indices[axis] < sizes[axis]"));
    REQUIRE_THROWS_WITH(dst_view(3, 2),
                        Catch::Contains("[Function:operator()] Error in Catalyst Runtime: "
                                        "Assertion: indices[axis] < sizes[axis]"));

    delete[] buffer_strided;
}

TEST_CASE("Test memref alloc", "[CoreQIS]")
{
    __quantum__rt__initialize();
    int *a = (int *)_mlir_memref_to_llvm_alloc(sizeof(int));
    CHECK(a != NULL);
    *a = 1;
    __quantum__rt__finalize();
}

TEST_CASE("Test memref aligned alloc", "[CoreQIS]")
{
    __quantum__rt__initialize();
    int *a = (int *)_mlir_memref_to_llvm_aligned_alloc(sizeof(int), sizeof(int));
    CHECK(a != NULL);
    *a = 1;
    __quantum__rt__finalize();
}

TEST_CASE("Test memref free", "[CoreQIS]")
{
    __quantum__rt__initialize();
    int *a = (int *)_mlir_memref_to_llvm_alloc(sizeof(int));
    CHECK(a != NULL);
    *a = 1;
    _mlir_memref_to_llvm_free(a);
    __quantum__rt__finalize();
}

TEST_CASE("Test memory transfer in rt", "[CoreQIS]")
{
    __quantum__rt__initialize();
    int *a = (int *)_mlir_memref_to_llvm_alloc(sizeof(int));
    bool is_in_rt = _mlir_memory_transfer(a);
    CHECK(is_in_rt);
    __quantum__rt__finalize();
    free(a);
}

TEST_CASE("Test memory transfer not in rt", "[CoreQIS]")
{
    __quantum__rt__initialize();
    int *a = (int *)malloc(sizeof(int));
    bool is_in_rt = _mlir_memory_transfer(a);
    CHECK(!is_in_rt);
    __quantum__rt__finalize();
    free(a);
}

TEST_CASE("Test __quantum__qis__Measure", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QUBIT *wire0 = __quantum__rt__qubit_allocate();

        QUBIT *wire1 = __quantum__rt__qubit_allocate();

        __quantum__qis__PauliX(wire0, false);

        Result m = __quantum__qis__Measure(wire0);

        Result one = __quantum__rt__result_get_one();
        CHECK(*m == *one);

        __quantum__rt__qubit_release(wire1);
        __quantum__rt__qubit_release(wire0);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__ Hadamard, PauliZ, IsingXX, IsingZZ, and SWAP", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QirArray *qs = __quantum__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 1);

        __quantum__qis__Hadamard(*target, false);
        __quantum__qis__PauliZ(*target, false);
        __quantum__qis__IsingXX(0.2, *ctrls, *target, false);
        __quantum__qis__IsingZZ(0.5, *ctrls, *target, false);
        __quantum__qis__SWAP(*ctrls, *target, false);

        MemRefT_CplxT_double_1d result = getState(4);
        __quantum__qis__State(&result, 0);
        CplxT_double *state = result.data_allocated;

        CHECK((state[0].real == Approx(0.6817017748).margin(1e-5) &&
               state[0].imag == Approx(-0.1740670409).margin(1e-5)));
        CHECK((state[1].real == Approx(-0.6817017748).margin(1e-5) &&
               state[1].imag == Approx(-0.1740670409).margin(1e-5)));
        CHECK((state[2].real == Approx(-0.0174649595).margin(1e-5) &&
               state[2].imag == Approx(0.068398324).margin(1e-5)));
        CHECK((state[3].real == Approx(-0.0174649595).margin(1e-5) &&
               state[3].imag == Approx(-0.068398324).margin(1e-5)));

        freeState(result);
        __quantum__rt__qubit_release_array(qs);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__ CRot, IsingXY and Toffoli", "[CoreQIS]")
{
    __quantum__rt__initialize();
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QirArray *qs = __quantum__rt__qubit_allocate_array(3);

        QUBIT **target = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls_0 = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 1);
        QUBIT **ctrls_1 = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 2);

        __quantum__qis__Hadamard(*target, false);
        __quantum__qis__PauliZ(*target, false);
        __quantum__qis__CRot(0.2, 0.5, 0.7, *ctrls_0, *target, false);
        __quantum__qis__IsingXY(0.2, *ctrls_0, *target, false);
        __quantum__qis__SWAP(*ctrls_0, *target, false);
        __quantum__qis__Toffoli(*ctrls_0, *ctrls_1, *target, false);

        MemRefT_CplxT_double_1d result = getState(8);
        __quantum__qis__State(&result, 0);
        CplxT_double *state = result.data_allocated;

        CHECK((state[0].real == Approx(0.70710678).margin(1e-5) &&
               state[0].imag == Approx(0.0).margin(1e-5)));
        CHECK((state[1].real == Approx(0.0).margin(1e-5) &&
               state[1].imag == Approx(0.0).margin(1e-5)));
        CHECK((state[2].real == Approx(-0.7035741926).margin(1e-5) &&
               state[2].imag == Approx(0.0).margin(1e-5)));
        CHECK((state[3].real == Approx(0.0).margin(1e-5) &&
               state[3].imag == Approx(0.0).margin(1e-5)));

        freeState(result);
        __quantum__rt__qubit_release_array(qs);
    }
    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__ Hadamard, PauliX, IsingYY, CRX, and Expval", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QirArray *qs = __quantum__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __quantum__qis__Hadamard(*target, false);
        // qml.PauliX(wires=0)
        __quantum__qis__PauliX(*target, false);
        // qml.IsingYY(0.2, wires=[1,0])
        __quantum__qis__IsingYY(0.2, *ctrls, *target, false);
        // qml.CRX(0.4, wires=[1,0])
        __quantum__qis__CRX(0.4, *ctrls, *target, false);

        MemRefT_CplxT_double_1d result = getState(4);
        __quantum__qis__State(&result, 0);
        CplxT_double *state = result.data_allocated;

        CHECK((state[0].real == Approx(0.70357419).margin(1e-5) &&
               state[0].imag == Approx(0.0).margin(1e-5)));
        CHECK((state[1].real == Approx(0.01402464).margin(1e-5) &&
               state[1].imag == Approx(-0.06918573).margin(1e-5)));
        CHECK((state[2].real == Approx(0.70357419).margin(1e-5) &&
               state[2].imag == Approx(0).margin(1e-5)));
        CHECK((state[3].real == Approx(-0.01402464).margin(1e-5) &&
               state[3].imag == Approx(0.06918573).margin(1e-5)));

        // qml.expval(qml.Hadamard(wires=1))
        auto obs = __quantum__qis__NamedObs(ObsId::Hadamard, *ctrls);

        CHECK(__quantum__qis__Expval(obs) == Approx(0.69301172).margin(1e-5));

        freeState(result);
        __quantum__rt__qubit_release_array(qs);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__ PhaseShift", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QirArray *qs = __quantum__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __quantum__qis__Hadamard(*target, false);
        // qml.RX(0.123, wires=1)
        __quantum__qis__RX(0.123, *ctrls, false);
        // qml.PhaseShift(0.456, wires=0)
        __quantum__qis__PhaseShift(0.456, *target, false);

        MemRefT_CplxT_double_1d result = getState(4);
        __quantum__qis__State(&result, 0);
        CplxT_double *state = result.data_allocated;

        CHECK((state[0].real == Approx(0.7057699753).margin(1e-5) &&
               state[0].imag == Approx(0.0).margin(1e-5)));
        CHECK((state[1].real == Approx(0.0).margin(1e-5) &&
               state[1].imag == Approx(-0.04345966).margin(1e-5)));
        CHECK((state[2].real == Approx(0.63365519).margin(1e-5) &&
               state[2].imag == Approx(0.31079312).margin(1e-5)));
        CHECK((state[3].real == Approx(0.01913791).margin(1e-5) &&
               state[3].imag == Approx(-0.039019).margin(1e-5)));

        freeState(result);
        __quantum__rt__qubit_release_array(qs);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__HermitianObs with an uninitialized matrix", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        MemRefT_CplxT_double_2d *matrix = nullptr;
        REQUIRE_THROWS_WITH(
            __quantum__qis__HermitianObs(matrix, 0),
            Catch::Contains("[Function:__quantum__qis__HermitianObs] Error in Catalyst Runtime: "
                            "The Hermitian matrix must be initialized"));

        delete matrix;
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__HermitianObs with invalid Hermitian matrix", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        MemRefT_CplxT_double_2d *matrix = new MemRefT_CplxT_double_2d;
        REQUIRE_THROWS_WITH(__quantum__qis__HermitianObs(matrix, 0),
                            Catch::Contains("Invalid given Hermitian matrix"));

        delete matrix;
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__HermitianObs with invalid number of wires", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QirArray *qs = __quantum__rt__qubit_allocate_array(1);

        QUBIT **target = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 0);

        MemRefT_CplxT_double_2d *matrix = new MemRefT_CplxT_double_2d;
        matrix->offset = 0;
        matrix->sizes[0] = 4;
        matrix->sizes[1] = 4;
        matrix->strides[0] = 1;
        REQUIRE_THROWS_WITH(__quantum__qis__HermitianObs(matrix, 2, *target, *target),
                            Catch::Contains("Invalid number of wires"));

        delete matrix;
        __quantum__rt__qubit_release_array(qs);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__HermitianObs and Expval", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QirArray *qs = __quantum__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __quantum__qis__Hadamard(*target, false);
        // qml.PauliX(wires=0)
        __quantum__qis__PauliX(*target, false);
        // qml.IsingYY(0.2, wires=[1,0])
        __quantum__qis__IsingYY(0.2, *ctrls, *target, false);
        // qml.CRX(0.4, wires=[1,0])
        __quantum__qis__CRX(0.4, *ctrls, *target, false);

        MemRefT_CplxT_double_1d result = getState(4);
        __quantum__qis__State(&result, 0);
        CplxT_double *state = result.data_allocated;

        CHECK((state[0].real == Approx(0.70357419).margin(1e-5) &&
               state[0].imag == Approx(0.0).margin(1e-5)));
        CHECK((state[1].real == Approx(0.01402464).margin(1e-5) &&
               state[1].imag == Approx(-0.06918573).margin(1e-5)));
        CHECK((state[2].real == Approx(0.70357419).margin(1e-5) &&
               state[2].imag == Approx(0).margin(1e-5)));
        CHECK((state[3].real == Approx(-0.01402464).margin(1e-5) &&
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

        CHECK(obs_h == 1);
        CHECK(__quantum__qis__Expval(obs_h) == Approx(0.9900332889).margin(1e-5));

        freeState(result);
        delete h_matrix;
        __quantum__rt__qubit_release_array(qs);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__TensorObs with invalid number of observables", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        REQUIRE_THROWS_WITH(
            __quantum__qis__TensorObs(0),
            Catch::Contains("[Function:__quantum__qis__TensorObs] Error in Catalyst Runtime: "
                            "Invalid number of observables to create TensorProdObs"));

        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__TensorProdObs and Expval", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QirArray *qs = __quantum__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __quantum__qis__Hadamard(*target, false);
        // qml.Hadamard(wires=1)
        __quantum__qis__Hadamard(*ctrls, false);
        // qml.IsingYY(0.2, wires=[1,0])
        __quantum__qis__IsingYY(0.6, *ctrls, *target, false);
        // qml.CRX(0.4, wires=[1,0])
        __quantum__qis__CRX(0.3, *ctrls, *target, false);

        MemRefT_CplxT_double_1d result = getState(4);
        __quantum__qis__State(&result, 0);

        auto obs_x = __quantum__qis__NamedObs(ObsId::PauliX, *target);

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

        CHECK(obs_t == 2);

        CHECK(__quantum__qis__Expval(obs_t) == Approx(1.5864438048).margin(1e-5));

        freeState(result);
        delete h_matrix;
        __quantum__rt__qubit_release_array(qs);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__HamiltonianObs with invalid coefficients", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        MemRefT_double_1d *coeffs = nullptr;

        REQUIRE_THROWS_WITH(
            __quantum__qis__HamiltonianObs(coeffs, 0),
            Catch::Contains("[Function:__quantum__qis__HamiltonianObs] Error in Catalyst Runtime: "
                            "Invalid coefficients for computing Hamiltonian"));

        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__HamiltonianObs with invalid number of coefficients and observables",
          "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        MemRefT_double_1d *coeffs = new MemRefT_double_1d;
        coeffs->offset = 0;
        coeffs->sizes[0] = 2;
        coeffs->strides[0] = 1;

        REQUIRE_THROWS_WITH(__quantum__qis__HamiltonianObs(coeffs, 0),
                            Catch::Contains("Invalid coefficients for computing Hamiltonian"));

        delete coeffs;

        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__HamiltonianObs(h, x) and Expval", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QirArray *qs = __quantum__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __quantum__qis__Hadamard(*target, false);
        // qml.Hadamard(wires=1)
        __quantum__qis__Hadamard(*ctrls, false);
        // qml.IsingYY(0.2, wires=[1,0])
        __quantum__qis__IsingYY(0.6, *ctrls, *target, false);
        // qml.CRX(0.4, wires=[1,0])
        __quantum__qis__CRX(0.3, *ctrls, *target, false);

        MemRefT_CplxT_double_1d result = getState(4);
        __quantum__qis__State(&result, 0);

        auto obs_x = __quantum__qis__NamedObs(ObsId::PauliX, *target);

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

        CHECK(obs_hamiltonian == 2);

        CHECK(__quantum__qis__Expval(obs_hamiltonian) == Approx(1.1938250042).margin(1e-5));

        freeState(result);
        delete h_matrix;
        delete coeffs;
        __quantum__rt__qubit_release_array(qs);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__HamiltonianObs(t) and Expval", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QirArray *qs = __quantum__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __quantum__qis__Hadamard(*target, false);
        // qml.Hadamard(wires=1)
        __quantum__qis__Hadamard(*ctrls, false);
        // qml.IsingYY(0.2, wires=[1,0])
        __quantum__qis__IsingYY(0.6, *ctrls, *target, false);
        // qml.CRX(0.4, wires=[1,0])
        __quantum__qis__CRX(0.3, *ctrls, *target, false);

        MemRefT_CplxT_double_1d result = getState(4);
        __quantum__qis__State(&result, 0);

        auto obs_x = __quantum__qis__NamedObs(ObsId::PauliX, *target);

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

        CHECK(obs_hamiltonian == 3);

        CHECK(__quantum__qis__Expval(obs_hamiltonian) == Approx(0.6345775219).margin(1e-5));

        freeState(result);
        delete h_matrix;
        delete coeffs;
        __quantum__rt__qubit_release_array(qs);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__ Hadamard, PauliX, IsingYY, CRX, and Expval_arr", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QirArray *qs = __quantum__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __quantum__qis__Hadamard(*target, false);
        // qml.PauliX(wires=0)
        __quantum__qis__PauliX(*target, false);
        // qml.IsingYY(0.2, wires=[1,0])
        __quantum__qis__IsingYY(0.2, *ctrls, *target, false);
        // qml.CRX(0.4, wires=[1,0])
        __quantum__qis__CRX(0.4, *ctrls, *target, false);

        MemRefT_CplxT_double_1d result = getState(4);
        __quantum__qis__State(&result, 0);
        CplxT_double *state = result.data_allocated;

        CHECK((state[0].real == Approx(0.70357419).margin(1e-5) &&
               state[0].imag == Approx(0.0).margin(1e-5)));
        CHECK((state[1].real == Approx(0.01402464).margin(1e-5) &&
               state[1].imag == Approx(-0.06918573).margin(1e-5)));
        CHECK((state[2].real == Approx(0.70357419).margin(1e-5) &&
               state[2].imag == Approx(0).margin(1e-5)));
        CHECK((state[3].real == Approx(-0.01402464).margin(1e-5) &&
               state[3].imag == Approx(0.06918573).margin(1e-5)));

        // qml.expval(qml.Hadamard(wires=1))
        auto obs = __quantum__qis__NamedObs(ObsId::Hadamard, *ctrls);

        CHECK(__quantum__qis__Expval(obs) == Approx(0.69301172).margin(1e-5));

        freeState(result);
        __quantum__rt__qubit_release_array(qs);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__ Hadamard, ControlledPhaseShift, IsingYY, CRX, and Var",
          "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QirArray *qs = __quantum__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __quantum__qis__Hadamard(*target, false);
        // qml.ControlledPhaseShift(0.6, wires=[0,1])
        __quantum__qis__ControlledPhaseShift(0.6, *target, *ctrls, false);
        // qml.IsingYY(0.2, wires=[0, 1])
        __quantum__qis__IsingYY(0.2, *target, *ctrls, false);
        // qml.CRX(0.4, wires=[1,0])
        __quantum__qis__CRX(0.4, *target, *ctrls, false);

        MemRefT_CplxT_double_1d result = getState(4);
        __quantum__qis__State(&result, 0);
        CplxT_double *state = result.data_allocated;

        CHECK((state[0].real == Approx(0.70357419).margin(1e-5) &&
               state[0].imag == Approx(0.0).margin(1e-5)));
        CHECK((state[1].real == Approx(0.0).margin(1e-5) &&
               state[1].imag == Approx(-0.0705929).margin(1e-5)));
        CHECK((state[2].real == Approx(0.70357419).margin(1e-5) &&
               state[2].imag == Approx(0).margin(1e-5)));
        CHECK((state[3].real == Approx(0.0).margin(1e-5) &&
               state[3].imag == Approx(-0.0705929).margin(1e-5)));

        // qml.var(qml.PauliZ(wires=1))
        auto obs = __quantum__qis__NamedObs(ObsId::PauliZ, *ctrls);

        CHECK(__quantum__qis__Variance(obs) == Approx(0.0394695).margin(1e-5));

        freeState(result);
        __quantum__rt__qubit_release_array(qs);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__ Hadamard, PauliX, IsingYY, CRX, and Probs", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QirArray *qs = __quantum__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __quantum__qis__Hadamard(*target, false);
        // qml.PauliX(wires=0)
        __quantum__qis__PauliX(*target, false);
        // qml.IsingYY(0.2, wires=[1,0])
        __quantum__qis__IsingYY(0.2, *ctrls, *target, false);
        // qml.CRX(0.4, wires=[1,0])
        __quantum__qis__CRX(0.4, *ctrls, *target, false);

        size_t buffer_len = 4;
        double *buffer = new double[buffer_len];
        MemRefT_double_1d result = {buffer, buffer, 0, {buffer_len}, {1}};
        __quantum__qis__Probs(&result, 0);
        double *probs = result.data_allocated;

        CHECK((probs[0] + probs[2]) == Approx(0.9900332889).margin(1e-5));
        CHECK((probs[1] + probs[3]) == Approx(0.0099667111).margin(1e-5));

        delete[] buffer;
        __quantum__rt__qubit_release_array(qs);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__ Hadamard, PauliX, IsingYY, CRX, and partial Probs", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QirArray *qs = __quantum__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __quantum__qis__Hadamard(*target, false);
        // qml.PauliX(wires=0)
        __quantum__qis__PauliX(*target, false);
        // qml.IsingYY(0.2, wires=[1,0])
        __quantum__qis__IsingYY(0.2, *ctrls, *target, false);
        // qml.CRX(0.4, wires=[1,0])
        __quantum__qis__CRX(0.4, *ctrls, *target, false);

        size_t buffer_len = 2;
        double *buffer = new double[buffer_len];
        MemRefT_double_1d result = {buffer, buffer, 0, {buffer_len}, {1}};
        __quantum__qis__Probs(&result, 1, ctrls[0]);
        double *probs = result.data_allocated;

        CHECK(probs[0] == Approx(0.9900332889).margin(1e-5));
        CHECK(probs[1] == Approx(0.0099667111).margin(1e-5));

        delete[] buffer;
        __quantum__rt__qubit_release_array(qs);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__State on the heap using malloc", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QirArray *qs = __quantum__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __quantum__qis__Hadamard(*target, false);
        // qml.ControlledPhaseShift(0.6, wires=[0,1])
        __quantum__qis__ControlledPhaseShift(0.6, *target, *ctrls, false);
        // qml.IsingYY(0.2, wires=[0, 1])
        __quantum__qis__IsingYY(0.2, *target, *ctrls, false);
        // qml.CRX(0.4, wires=[1,0])
        __quantum__qis__CRX(0.4, *target, *ctrls, false);

        MemRefT_CplxT_double_1d result = getState(4);
        __quantum__qis__State(&result, 0);
        CplxT_double *stateVec = result.data_allocated;

        CHECK(stateVec[0].real == Approx(0.7035741926).margin(1e-5));
        CHECK(stateVec[0].imag == 0.0);

        CHECK(stateVec[1].real == 0.0);
        CHECK(stateVec[1].imag == Approx(-0.070592886).margin(1e-5));

        CHECK(stateVec[2].real == Approx(0.7035741926).margin(1e-5));
        CHECK(stateVec[2].imag == 0.0);

        CHECK(stateVec[3].real == 0.0);
        CHECK(stateVec[3].imag == Approx(-0.070592886).margin(1e-5));

        freeState(result);
        __quantum__rt__qubit_release_array(qs);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__Measure with false", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QUBIT *target = __quantum__rt__qubit_allocate(); // id = 0

        // qml.Hadamard(wires=0)
        __quantum__qis__RY(0.0, target, false);

        Result mres = __quantum__qis__Measure(target);

        Result zero = __quantum__rt__result_get_zero();
        CHECK(__quantum__rt__result_equal(mres, zero));

        __quantum__rt__qubit_release(target);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__Measure with true", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QUBIT *target = __quantum__rt__qubit_allocate(); // id = 0

        // qml.Hadamard(wires=0)
        __quantum__qis__RY(3.14, target, false);

        Result mres = __quantum__qis__Measure(target);

        Result one = __quantum__rt__result_get_one();
        CHECK(__quantum__rt__result_equal(mres, one));

        __quantum__rt__qubit_release(target);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__MultiRZ", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QirArray *qs = __quantum__rt__qubit_allocate_array(2);

        QUBIT **q0 = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **q1 = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 1);

        __quantum__qis__RX(M_PI, *q0, false);
        __quantum__qis__Hadamard(*q0, false);
        __quantum__qis__Hadamard(*q1, false);
        __quantum__qis__MultiRZ(M_PI, false, 2, *q0, *q1);
        __quantum__qis__Hadamard(*q0, false);
        __quantum__qis__Hadamard(*q1, false);

        Result q0_m = __quantum__qis__Measure(*q0);
        Result q1_m = __quantum__qis__Measure(*q1);

        Result zero = __quantum__rt__result_get_zero();
        Result one = __quantum__rt__result_get_one();
        CHECK(__quantum__rt__result_equal(q0_m, zero));
        CHECK(__quantum__rt__result_equal(q1_m, one));

        __quantum__rt__qubit_release_array(qs);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__CSWAP ", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QirArray *qs = __quantum__rt__qubit_allocate_array(3);

        QUBIT **q0 = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **q1 = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 1);
        QUBIT **q2 = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 2);

        __quantum__qis__RX(M_PI, *q0, false);
        __quantum__qis__RX(M_PI, *q1, false);
        __quantum__qis__CSWAP(*q0, *q1, *q2, false);

        Result q1_m = __quantum__qis__Measure(*q1);
        Result q2_m = __quantum__qis__Measure(*q2);

        Result zero = __quantum__rt__result_get_zero();
        Result one = __quantum__rt__result_get_one();

        // Test via rt__result_to_string
        QirString *zero_str = __quantum__rt__result_to_string(zero);
        QirString *q1_m_str = __quantum__rt__result_to_string(q1_m);
        CHECK(__quantum__rt__string_equal(zero_str, q1_m_str));

        QirString *one_str = __quantum__rt__result_to_string(one);
        QirString *q2_m_str = __quantum__rt__result_to_string(q2_m);
        CHECK(__quantum__rt__string_equal(one_str, q2_m_str));

        // Test via rt__result_equal
        CHECK(__quantum__rt__result_equal(q1_m, zero));
        CHECK(__quantum__rt__result_equal(q2_m, one));

        __quantum__rt__string_update_reference_count(zero_str, -1);
        __quantum__rt__string_update_reference_count(q1_m_str, -1);
        __quantum__rt__string_update_reference_count(one_str, -1);
        __quantum__rt__string_update_reference_count(q2_m_str, -1);

        __quantum__rt__qubit_release_array(qs);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__Counts with num_qubits=2 calling Hadamard, ControlledPhaseShift, "
          "IsingYY, and CRX quantum operations",
          "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QirArray *qs = __quantum__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __quantum__qis__Hadamard(*target, false);
        // qml.ControlledPhaseShift(0.6, wires=[0,1])
        __quantum__qis__ControlledPhaseShift(0.6, *target, *ctrls, false);
        // qml.IsingYY(0.2, wires=[0, 1])
        __quantum__qis__IsingYY(0.2, *target, *ctrls, false);
        // qml.CRX(0.4, wires=[1,0])
        __quantum__qis__CRX(0.4, *target, *ctrls, false);

        constexpr size_t shots = 1000;

        PairT_MemRefT_double_int64_1d result = getCounts(4);
        __quantum__qis__Counts(&result, shots, 0);
        double *eigvals = result.first.data_allocated;
        int64_t *counts = result.second.data_allocated;

        for (int i = 0; i < 4; i++) {
            CHECK(eigvals[i] == (double)i);
        }

        size_t sum = 0;
        for (int i = 0; i < 4; i++) {
            sum += counts[i];
        }
        CHECK(sum == shots);

        freeCounts(result);
        __quantum__rt__qubit_release_array(qs);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__Counts with num_qubits=2 PartialCounts calling Hadamard, "
          "ControlledPhaseShift, IsingYY, and CRX quantum operations",
          "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QirArray *qs = __quantum__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __quantum__qis__Hadamard(*target, false);
        // qml.ControlledPhaseShift(0.6, wires=[0,1])
        __quantum__qis__ControlledPhaseShift(0.6, *target, *ctrls, false);
        // qml.IsingYY(0.2, wires=[0, 1])
        __quantum__qis__IsingYY(0.2, *target, *ctrls, false);
        // qml.CRX(0.4, wires=[1,0])
        __quantum__qis__CRX(0.4, *target, *ctrls, false);

        constexpr size_t shots = 1000;

        PairT_MemRefT_double_int64_1d result = getCounts(2);
        __quantum__qis__Counts(&result, shots, 1, ctrls[0]);
        double *eigvals = result.first.data_allocated;
        int64_t *counts = result.second.data_allocated;

        CHECK(counts[0] + counts[1] == shots);
        CHECK(eigvals[0] + 1 == eigvals[1]);

        freeCounts(result);
        __quantum__rt__qubit_release_array(qs);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__Sample with num_qubits=2 calling Hadamard, ControlledPhaseShift, "
          "IsingYY, and CRX quantum operations",
          "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QirArray *qs = __quantum__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __quantum__qis__Hadamard(*target, false);
        // qml.ControlledPhaseShift(0.6, wires=[0,1])
        __quantum__qis__ControlledPhaseShift(0.6, *target, *ctrls, false);
        // qml.IsingYY(0.2, wires=[0, 1])
        __quantum__qis__IsingYY(0.2, *target, *ctrls, false);
        // qml.CRX(0.4, wires=[1,0])
        __quantum__qis__CRX(0.4, *target, *ctrls, false);

        constexpr size_t n = 2;
        constexpr size_t shots = 1000;

        double *buffer = new double[shots * n];
        MemRefT_double_2d result = {buffer, buffer, 0, {shots, n}, {n, 1}};
        __quantum__qis__Sample(&result, shots, 0);
        double *samples = result.data_allocated;

        size_t counts0[2] = {0, 0};
        for (size_t idx = 0; idx < shots * n; idx += n) {
            if (samples[idx] == 0) {
                counts0[0]++;
            }
            else {
                counts0[1]++;
            }
        }

        CHECK(counts0[0] + counts0[1] == shots);

        size_t counts1[2] = {0, 0};
        for (size_t idx = 1; idx < shots * n; idx += n) {
            if (samples[idx] == 0) {
                counts1[0]++;
            }
            else {
                counts1[1]++;
            }
        }

        CHECK(counts1[0] + counts1[1] == shots);

        auto obs = __quantum__qis__NamedObs(ObsId::PauliZ, *ctrls);

        CHECK(__quantum__qis__Variance(obs) == Approx(0.0394695).margin(1e-5));

        delete[] buffer;

        __quantum__rt__qubit_release_array(qs);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__Sample with num_qubits=2 and PartialSample calling Hadamard, "
          "ControlledPhaseShift, IsingYY, and CRX quantum operations",
          "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QirArray *qs = __quantum__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __quantum__qis__Hadamard(*target, false);
        // qml.ControlledPhaseShift(0.6, wires=[0,1])
        __quantum__qis__ControlledPhaseShift(0.6, *target, *ctrls, false);
        // qml.IsingYY(0.2, wires=[0, 1])
        __quantum__qis__IsingYY(0.2, *target, *ctrls, false);
        // qml.CRX(0.4, wires=[1,0])
        __quantum__qis__CRX(0.4, *target, *ctrls, false);

        constexpr size_t n = 1;
        constexpr size_t shots = 1000;

        double *buffer = new double[shots * n];
        MemRefT_double_2d result = {buffer, buffer, 0, {shots, n}, {n, 1}};
        __quantum__qis__Sample(&result, shots, 1, ctrls[0]);
        double *samples = result.data_allocated;

        size_t counts0[2] = {0, 0};
        for (size_t idx = 0; idx < shots * n; idx += n) {
            if (samples[idx] == 0) {
                counts0[0]++;
            }
            else {
                counts0[1]++;
            }
        }

        CHECK(counts0[0] + counts0[1] == shots);

        auto obs = __quantum__qis__NamedObs(ObsId::PauliZ, *ctrls);

        CHECK(__quantum__qis__Variance(obs) == Approx(0.0394695).margin(1e-5));

        delete[] buffer;

        __quantum__rt__qubit_release_array(qs);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__QubitUnitary with an uninitialized matrix", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QUBIT *target = __quantum__rt__qubit_allocate(); // id = 0
        MemRefT_CplxT_double_2d *matrix = nullptr;

        REQUIRE_THROWS_WITH(
            __quantum__qis__QubitUnitary(matrix, false, 1, target),
            Catch::Contains("[Function:__quantum__qis__QubitUnitary] Error in Catalyst Runtime: "
                            "The QubitUnitary matrix must be initialized"));

        __quantum__rt__qubit_release(target);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__QubitUnitary with invalid number of wires", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QUBIT *target = __quantum__rt__qubit_allocate(); // id = 0
        MemRefT_CplxT_double_2d *matrix = new MemRefT_CplxT_double_2d;

        REQUIRE_THROWS_WITH(__quantum__qis__QubitUnitary(matrix, false, 3, target, false),
                            Catch::Contains("Invalid number of wires"));
        REQUIRE_THROWS_WITH(__quantum__qis__QubitUnitary(matrix, false, 3, target, true),
                            Catch::Contains("Invalid number of wires"));

        delete matrix;
        __quantum__rt__qubit_release(target);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__QubitUnitary with invalid matrix", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QUBIT *target = __quantum__rt__qubit_allocate(); // id = 0
        MemRefT_CplxT_double_2d *matrix = new MemRefT_CplxT_double_2d;

        matrix->offset = 0;
        matrix->sizes[0] = 1;
        matrix->sizes[1] = 1;
        matrix->strides[0] = 1;

        REQUIRE_THROWS_WITH(__quantum__qis__QubitUnitary(matrix, false, 1, target),
                            Catch::Contains("Invalid given QubitUnitary matrix"));
        REQUIRE_THROWS_WITH(__quantum__qis__QubitUnitary(matrix, false, 1, target, true),
                            Catch::Contains("Invalid given QubitUnitary matrix"));

        delete matrix;
        __quantum__rt__qubit_release(target);
        __quantum__rt__finalize();
    }
}

TEST_CASE("Test __quantum__qis__QubitUnitary with num_qubits=2", "[CoreQIS]")
{
    __quantum__rt__initialize();
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QirArray *qs = __quantum__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 1);

        __quantum__qis__Hadamard(*target, false);
        __quantum__qis__CNOT(*target, *ctrls, false);

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

        __quantum__qis__QubitUnitary(matrix, false, 1, *target);

        MemRefT_CplxT_double_1d result = getState(4);
        __quantum__qis__State(&result, 0);
        CplxT_double *state = result.data_allocated;

        CHECK(state[0].real == Approx(-0.474432).margin(1e-5));
        CHECK(state[0].imag == Approx(-0.44579).margin(1e-5));
        CHECK(state[1].real == Approx(-0.105256).margin(1e-5));
        CHECK(state[1].imag == Approx(0.255159).margin(1e-5));
        CHECK(state[2].real == Approx(-0.168031).margin(1e-5));
        CHECK(state[2].imag == Approx(0.218977).margin(1e-5));
        CHECK(state[3].real == Approx(-0.623553).margin(1e-5));
        CHECK(state[3].imag == Approx(-0.187075).margin(1e-5));

        freeState(result);
        delete matrix;
        __quantum__rt__qubit_release_array(qs);
    }
    __quantum__rt__finalize();
}

TEST_CASE("Test __rt__device registering a custom device with shots=500 and device=lightning",
          "[CoreQIS]")
{
    __quantum__rt__initialize();

    char dev[8] = "backend";
    char dev_value[17] = "lightning.qubit";
    __quantum__rt__device((int8_t *)dev, (int8_t *)dev_value);

    char dev2[7] = "device";
    char dev2_value[15] = "backend.other";
    REQUIRE_THROWS_WITH(__quantum__rt__device((int8_t *)dev2, (int8_t *)dev_value),
                        Catch::Contains("[Function:__quantum__rt__device] Error in Catalyst "
                                        "Runtime: Invalid device specification"));

    REQUIRE_THROWS_WITH(__quantum__rt__device((int8_t *)dev, (int8_t *)dev2_value),
                        Catch::Contains("Failed initialization of the backend device"));

    REQUIRE_THROWS_WITH(__quantum__rt__device(nullptr, nullptr),
                        Catch::Contains("Invalid device specification"));

    __quantum__rt__finalize();

    REQUIRE_THROWS_WITH(__quantum__rt__device((int8_t *)dev, (int8_t *)dev_value),
                        Catch::Contains("Invalid use of the global driver before initialization"));
}

TEST_CASE("Test __rt__device registering the OpenQasm device", "[CoreQIS]")
{
    __quantum__rt__initialize();

    char dev[8] = "backend";
    char dev_value[30] = "braket.aws.qubit";

#if __has_include("OpenQasmDevice.hpp")
    __quantum__rt__device((int8_t *)dev, (int8_t *)dev_value);
#else
    REQUIRE_THROWS_WITH(__quantum__rt__device((int8_t *)dev, (int8_t *)dev_value),
                        Catch::Contains("Failed initialization of the backend device"));
#endif

    __quantum__rt__finalize();

    __quantum__rt__initialize();

    char dev_kwargs[20] = "kwargs";
    char dev_value_kwargs[70] = "device_arn : arn:aws:braket:::device/quantum-simulator/amazon/sv1";

    __quantum__rt__device((int8_t *)dev_kwargs, (int8_t *)dev_value_kwargs);

#if __has_include("OpenQasmDevice.hpp")
    __quantum__rt__device((int8_t *)dev, (int8_t *)dev_value);
#else
    REQUIRE_THROWS_WITH(__quantum__rt__device((int8_t *)dev, (int8_t *)dev_value),
                        Catch::Contains("Failed initialization of the backend device"));
#endif

    __quantum__rt__finalize();

    __quantum__rt__initialize();

    char dev_lcl[8] = "backend";
    char dev_value_lcl[30] = "braket.local.qubit";

#if __has_include("OpenQasmDevice.hpp")
    __quantum__rt__device((int8_t *)dev_lcl, (int8_t *)dev_value_lcl);
#else
    REQUIRE_THROWS_WITH(__quantum__rt__device((int8_t *)dev_lcl, (int8_t *)dev_value_lcl),
                        Catch::Contains("Failed initialization of the backend device"));
#endif

    __quantum__rt__finalize();
}

TEST_CASE("Test the main porperty of the adjoint quantum operations", "[CoreQIS]")
{
    for (const auto &[key, val] : getDevices()) {
        __quantum__rt__initialize();
        __quantum__rt__device((int8_t *)key.c_str(), (int8_t *)val.c_str());

        QirArray *qs = __quantum__rt__qubit_allocate_array(3);

        QUBIT **q0 = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **q1 = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 1);
        QUBIT **q2 = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(qs, 2);

        double theta = 3.14 / 2.0;
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

        __quantum__qis__QubitUnitary(matrix, false, 1, *q0);
        __quantum__qis__MultiRZ(theta, false, 2, *q0, *q1);
        __quantum__qis__Toffoli(*q0, *q1, *q2, false);
        __quantum__qis__CSWAP(*q0, *q1, *q2, false);
        __quantum__qis__CRot(theta, theta, theta, *q0, *q1, false);
        __quantum__qis__CRZ(theta, *q0, *q1, false);
        __quantum__qis__CRY(theta, *q0, *q1, false);
        __quantum__qis__CRX(theta, *q0, *q1, false);
        __quantum__qis__ControlledPhaseShift(theta, *q0, *q1, false);
        __quantum__qis__IsingZZ(theta, *q0, *q1, false);
        __quantum__qis__IsingXY(theta, *q0, *q1, false);
        __quantum__qis__IsingYY(theta, *q0, *q1, false);
        __quantum__qis__IsingXX(theta, *q0, *q1, false);
        __quantum__qis__SWAP(*q0, *q1, false);
        __quantum__qis__CZ(*q0, *q1, false);
        __quantum__qis__CY(*q0, *q1, false);
        __quantum__qis__CNOT(*q0, *q1, false);
        __quantum__qis__Rot(theta, theta, theta, *q0, false);
        __quantum__qis__RZ(theta, *q0, false);
        __quantum__qis__RY(theta, *q0, false);
        __quantum__qis__RX(theta, *q0, false);
        __quantum__qis__PhaseShift(theta, *q0, false);
        __quantum__qis__T(*q0, false);
        __quantum__qis__S(*q0, false);
        __quantum__qis__Hadamard(*q0, false);
        __quantum__qis__PauliZ(*q0, false);
        __quantum__qis__PauliY(*q0, false);
        __quantum__qis__PauliX(*q0, false);
        __quantum__qis__Identity(*q0, false);

        __quantum__qis__Identity(*q0, true);
        __quantum__qis__PauliX(*q0, true);
        __quantum__qis__PauliY(*q0, true);
        __quantum__qis__PauliZ(*q0, true);
        __quantum__qis__Hadamard(*q0, true);
        __quantum__qis__S(*q0, true);
        __quantum__qis__T(*q0, true);
        __quantum__qis__PhaseShift(theta, *q0, true);
        __quantum__qis__RX(theta, *q0, true);
        __quantum__qis__RY(theta, *q0, true);
        __quantum__qis__RZ(theta, *q0, true);
        __quantum__qis__Rot(theta, theta, theta, *q0, true);
        __quantum__qis__CNOT(*q0, *q1, true);
        __quantum__qis__CY(*q0, *q1, true);
        __quantum__qis__CZ(*q0, *q1, true);
        __quantum__qis__SWAP(*q0, *q1, true);
        __quantum__qis__IsingXX(theta, *q0, *q1, true);
        __quantum__qis__IsingYY(theta, *q0, *q1, true);
        __quantum__qis__IsingXY(theta, *q0, *q1, true);
        __quantum__qis__IsingZZ(theta, *q0, *q1, true);
        __quantum__qis__ControlledPhaseShift(theta, *q0, *q1, true);
        __quantum__qis__CRX(theta, *q0, *q1, true);
        __quantum__qis__CRY(theta, *q0, *q1, true);
        __quantum__qis__CRZ(theta, *q0, *q1, true);
        __quantum__qis__CRot(theta, theta, theta, *q0, *q1, true);
        __quantum__qis__CSWAP(*q0, *q1, *q2, true);
        __quantum__qis__Toffoli(*q0, *q1, *q2, true);
        __quantum__qis__MultiRZ(theta, true, 2, *q0, *q1);
        __quantum__qis__QubitUnitary(matrix, true, 1, *q0);

        MemRefT_CplxT_double_1d result = getState(8);
        __quantum__qis__State(&result, 0);
        CplxT_double *stateVec = result.data_allocated;

        CHECK(stateVec[0].real == Approx(1.0).margin(1e-5));
        CHECK(stateVec[0].imag == Approx(0.0).margin(1e-5));
        for (size_t i = 1; i < 8; i++) {
            CHECK(stateVec[i].real == Approx(0.0).margin(1e-5));
            CHECK(stateVec[i].imag == Approx(0.0).margin(1e-5));
        }

        freeState(result);
        delete matrix;
        __quantum__rt__qubit_release_array(qs);
        __quantum__rt__finalize();
    }
}
