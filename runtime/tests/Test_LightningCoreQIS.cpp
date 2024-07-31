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

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "DataView.hpp"
#include "MemRefUtils.hpp"
#include "QuantumDevice.hpp"
#include "RuntimeCAPI.h"
#include "Utils.hpp"

#include "TestUtils.hpp"

using namespace Catalyst::Runtime;

TEST_CASE("Test __catalyst__rt__print_tensor i1, i8, i16, i32, f32, and c64",
          "[qir_lightning_core]")
{
    std::array<bool, 2> buffer_i1;
    MemRefT<bool, 1> mr_i1_1d{buffer_i1.data(), buffer_i1.data(), 0, {2}, {1}};
    CHECK(mr_i1_1d.sizes[0] == 2);
    OpaqueMemRefT omr_i1_1d{1, (void *)(&mr_i1_1d), NumericType::i1};
    CHECK(omr_i1_1d.rank == 1);
    __catalyst__rt__print_tensor(&omr_i1_1d, false);

    std::vector<int8_t> buffer_i8(2, 1);
    MemRefT<int8_t, 1> mr_i8_1d{buffer_i8.data(), buffer_i8.data(), 0, {2}, {1}};
    CHECK(mr_i8_1d.sizes[0] == 2);
    OpaqueMemRefT omr_i8_1d{1, (void *)(&mr_i8_1d), NumericType::i8};
    CHECK(omr_i8_1d.rank == 1);
    __catalyst__rt__print_tensor(&omr_i8_1d, false);

    std::vector<int16_t> buffer_i16(1, 1);
    MemRefT<int16_t, 1> mr_i16_1d{buffer_i16.data(), buffer_i16.data(), 0, {1}, {1}};
    CHECK(mr_i16_1d.sizes[0] == 1);
    OpaqueMemRefT omr_i16_1d{1, (void *)(&mr_i16_1d), NumericType::i16};
    CHECK(omr_i16_1d.rank == 1);
    __catalyst__rt__print_tensor(&omr_i16_1d, false);

    std::vector<int32_t> buffer_i32(1, 1);
    MemRefT<int32_t, 1> mr_i32_1d{buffer_i32.data(), buffer_i32.data(), 0, {1}, {1}};
    CHECK(mr_i32_1d.sizes[0] == 1);
    OpaqueMemRefT omr_i32_1d{1, (void *)(&mr_i32_1d), NumericType::i32};
    CHECK(omr_i32_1d.rank == 1);
    __catalyst__rt__print_tensor(&omr_i32_1d, false);

    std::vector<size_t> buffer_idx(1, 1);
    MemRefT<size_t, 1> mr_idx_1d{buffer_idx.data(), buffer_idx.data(), 0, {1}, {1}};
    CHECK(mr_idx_1d.sizes[0] == 1);
    OpaqueMemRefT omr_idx_1d{1, (void *)(&mr_idx_1d), NumericType::idx};
    CHECK(omr_idx_1d.rank == 1);
    __catalyst__rt__print_tensor(&omr_idx_1d, false);

    std::vector<float> buffer_f32(1, 1.0);
    MemRefT<float, 1> mr_f32_1d{buffer_f32.data(), buffer_f32.data(), 0, {1}, {1}};
    CHECK(mr_f32_1d.sizes[0] == 1);
    OpaqueMemRefT omr_f32_1d{1, (void *)(&mr_f32_1d), NumericType::f32};
    CHECK(omr_f32_1d.rank == 1);
    __catalyst__rt__print_tensor(&omr_f32_1d, false);

    CplxT_float matrix_data[2] = {
        {-0.67, -0.63},
        {-0.14, 0.36},
    };
    MemRefT<CplxT_float, 1> mr_c32_1d{matrix_data, matrix_data, 0, {2}, {1}};
    CHECK(mr_c32_1d.sizes[0] == 2);
    OpaqueMemRefT omr_c32_1d{1, (void *)(&mr_c32_1d), NumericType::c64};
    CHECK(omr_c32_1d.rank == 1);
    __catalyst__rt__print_tensor(&omr_c32_1d, false);
}

TEST_CASE("Test __catalyst__rt__print_tensor for an unsupported datatype", "[qir_lightning_core]")
{
    std::vector<std::vector<size_t>> buffer(2);
    MemRefT<std::vector<size_t>, 1> mr_vidx_1d{buffer.data(), buffer.data(), 0, {2}, {1}};

    OpaqueMemRefT omr_vidx_1d{1, (void *)(&mr_vidx_1d), static_cast<NumericType>(1000)};

    REQUIRE_THROWS_WITH(
        __catalyst__rt__print_tensor(&omr_vidx_1d, false),
        Catch::Contains("[Function:__catalyst__rt__print_tensor] Error in Catalyst Runtime: Unkown "
                        "numeric type encoding for array printing."));
}

TEST_CASE("Test __catalyst__rt__print_tensor i64 1-dim", "[qir_lightning_core]")
{
    std::vector<int64_t> buffer(100);
    std::iota(buffer.begin(), buffer.end(), 0);

    MemRefT_int64_1d mr_i64_1d{buffer.data(), buffer.data(), 0, {100}, {1}};
    CHECK(mr_i64_1d.sizes[0] == 100);

    OpaqueMemRefT omr_i64_1d{1, (void *)(&mr_i64_1d), NumericType::i64};
    CHECK(omr_i64_1d.rank == 1);

    __catalyst__rt__print_tensor(&omr_i64_1d, false);
}

TEST_CASE("Test __catalyst__rt__print_tensor dbl 2-dim", "[qir_lightning_core]")
{
    std::vector<double> buffer(100 * 2);
    buffer[0] = 1.0;

    MemRefT_double_2d mr_dbl_2d{buffer.data(), buffer.data(), 0, {100, 2}, {2, 1}};
    CHECK(mr_dbl_2d.sizes[0] == 100);
    CHECK(mr_dbl_2d.sizes[1] == 2);

    OpaqueMemRefT omr_dbl_2d{2, (void *)(&mr_dbl_2d), NumericType::f64};
    CHECK(omr_dbl_2d.rank == 2);

    __catalyst__rt__print_tensor(&omr_dbl_2d, true);
}

TEST_CASE("Test __catalyst__rt__print_tensor cplx 2-dim", "[qir_lightning_core]")
{
    CplxT_double matrix_data[4] = {
        {-0.67, -0.63},
        {-0.14, 0.36},
        {-0.23, 0.30},
        {-0.88, -0.26},
    };

    MemRefT_CplxT_double_2d mr_cplx_2d{matrix_data, matrix_data, 0, {2, 2}, {1, 0}};
    CHECK(mr_cplx_2d.sizes[0] == 2);
    CHECK(mr_cplx_2d.sizes[1] == 2);

    OpaqueMemRefT omr_cplx_2d{2, (void *)(&mr_cplx_2d), NumericType::c128};
    CHECK(omr_cplx_2d.rank == 2);

    __catalyst__rt__print_tensor(&omr_cplx_2d, false);
}

PairT_MemRefT_double_int64_1d getCounts(size_t buffer_len)
{
    double *buff_e = new double[buffer_len];
    int64_t *buff_c = new int64_t[buffer_len];
    PairT_MemRefT_double_int64_1d result = {{buff_e, buff_e, 0, {buffer_len}, {1}},
                                            {buff_c, buff_c, 0, {buffer_len}, {1}}};
    return result;
}

void freeCounts(PairT_MemRefT_double_int64_1d &result)
{
    delete[] result.first.data_allocated;
    delete[] result.second.data_allocated;
}

TEST_CASE("Test __catalyst__rt__fail_cstr", "[qir_lightning_core]")
{
    REQUIRE_THROWS_WITH(
        __catalyst__rt__fail_cstr("Test!"),
        Catch::Contains("[Function:__catalyst__rt__fail_cstr] Error in Catalyst Runtime: Test!"));
}

TEST_CASE("Test device release after driver release", "[CoreQIS]")
{
    auto devices = getDevices();
    auto &[rtd_lib, rtd_name, rtd_kwargs] = devices[0];

    __catalyst__rt__initialize(nullptr);
    __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                (int8_t *)rtd_kwargs.c_str());
    QUBIT *q = __catalyst__rt__qubit_allocate();
    __catalyst__rt__qubit_release(q);
    __catalyst__rt__finalize();

    REQUIRE_THROWS_WITH(
        __catalyst__rt__device_release(),
        Catch::Contains("Cannot release an ACTIVE device out of scope of the global driver"));
}

TEST_CASE("Test device init before device release", "[CoreQIS]")
{
    auto devices = getDevices();
    auto &[rtd_lib, rtd_name, rtd_kwargs] = devices[0];

    REQUIRE_THROWS_WITH(__catalyst__rt__device_init((int8_t *)rtd_lib.c_str(),
                                                    (int8_t *)rtd_name.c_str(),
                                                    (int8_t *)rtd_kwargs.c_str()),
                        Catch::Contains("Invalid use of the global driver before initialization"));
}

TEST_CASE("Qubits: allocate, release, dump", "[CoreQIS]")
{
    __catalyst__rt__initialize(nullptr);
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QUBIT *q = __catalyst__rt__qubit_allocate();

        __catalyst__rt__qubit_release(q);

        QirArray *qs = __catalyst__rt__qubit_allocate_array(3);

        CHECK(__catalyst__rt__array_get_size_1d(qs) == 3);

        __catalyst__rt__array_get_element_ptr_1d(qs, 0);
        __catalyst__rt__array_get_element_ptr_1d(qs, 2);

        REQUIRE_THROWS_WITH(
            __catalyst__rt__array_get_element_ptr_1d(qs, 3),
            Catch::Contains("The qubit register does not contain the requested wire: 3"));

        __catalyst__rt__qubit_release_array(qs); // The `qs` is a dangling pointer from now on.
        __catalyst__rt__device_release();
    }
    __catalyst__rt__finalize();
}

TEST_CASE("Test lightning__core__qis methods", "[CoreQIS]")
{
    __catalyst__rt__initialize(nullptr);
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        constexpr double angle = 0.42;

        QirArray *reg = __catalyst__rt__qubit_allocate_array(3);
        QUBIT *target =
            *reinterpret_cast<QUBIT **>(__catalyst__rt__array_get_element_ptr_1d(reg, 2));

        __catalyst__qis__RY(angle, target, NO_MODIFIERS);
        __catalyst__qis__RX(angle, target, NO_MODIFIERS);

        // The `ctrls` is a dangling pointer from now on.
        __catalyst__rt__qubit_release_array(reg);

        CHECK(true); // if the __catalyst__qis__ operations can be called
        __catalyst__rt__device_release();
    }
    __catalyst__rt__finalize();
}

TEST_CASE("Test __catalyst__rt__print_state", "[CoreQIS]")
{
    __catalyst__rt__initialize(nullptr);
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *reg = __catalyst__rt__qubit_allocate_array(2);

        std::string expected = "*** State-Vector of Size 4 ***\n[(1,0), (0,0), (0,0), (0,0)]\n";
        std::stringstream buffer;

        std::streambuf *prevcoutbuf = std::cout.rdbuf(buffer.rdbuf());
        __catalyst__rt__print_state();
        std::cout.rdbuf(prevcoutbuf);

        std::string result = buffer.str();
        CHECK(!result.compare(expected));

        __catalyst__rt__qubit_release_array(reg);
        __catalyst__rt__device_release();
    }
    __catalyst__rt__finalize();
}

TEST_CASE("Test __catalyst__qis__State with wires", "[CoreQIS]")
{
    __catalyst__rt__initialize(nullptr);
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QUBIT *wire0 = __catalyst__rt__qubit_allocate();

        QUBIT *wire1 = __catalyst__rt__qubit_allocate();

        MemRefT_CplxT_double_1d result = getState(8);

        REQUIRE_THROWS_WITH(__catalyst__qis__State(&result, 1, wire0),
                            Catch::Contains("[Function:__catalyst__qis__State] Error in Catalyst "
                                            "Runtime: Partial State-Vector not supported yet"));

        freeState(result);
        __catalyst__rt__qubit_release(wire1);
        __catalyst__rt__qubit_release(wire0);
        __catalyst__rt__device_release();
    }
    __catalyst__rt__finalize();
}

TEST_CASE("Test __catalyst__qis__Identity", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QUBIT *wire0 = __catalyst__rt__qubit_allocate();

        QUBIT *wire1 = __catalyst__rt__qubit_allocate();

        __catalyst__qis__Identity(wire0, NO_MODIFIERS);

        MemRefT_CplxT_double_1d result = getState(4);
        __catalyst__qis__State(&result, 0);
        CplxT_double *state = result.data_allocated;

        CHECK(state[0].real == Approx(1.0).margin(1e-5));
        CHECK(state[0].imag == Approx(0.0).margin(1e-5));
        CHECK(state[1].real == Approx(0.0).margin(1e-5));
        CHECK(state[1].imag == Approx(0.0).margin(1e-5));

        freeState(result);
        __catalyst__rt__qubit_release(wire1);
        __catalyst__rt__qubit_release(wire0);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__PauliX", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QUBIT *wire0 = __catalyst__rt__qubit_allocate();

        QUBIT *wire1 = __catalyst__rt__qubit_allocate();

        __catalyst__qis__PauliX(wire0, NO_MODIFIERS);

        MemRefT_CplxT_double_1d result = getState(4);
        __catalyst__qis__State(&result, 0);
        CplxT_double *state = result.data_allocated;

        CHECK((state[0].real == Approx(0.0).margin(1e-5) &&
               state[0].imag == Approx(0.0).margin(1e-5)));
        CHECK((state[2].real == Approx(1.0).margin(1e-5) &&
               state[2].imag == Approx(0.0).margin(1e-5)));

        freeState(result);
        __catalyst__rt__qubit_release(wire1);
        __catalyst__rt__qubit_release(wire0);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__ PauliY and Rot", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QUBIT *wire0 = __catalyst__rt__qubit_allocate();
        QUBIT *wire1 = __catalyst__rt__qubit_allocate();

        __catalyst__qis__PauliY(wire0, NO_MODIFIERS);
        __catalyst__qis__Rot(0.4, 0.6, -0.2, wire0, NO_MODIFIERS);

        MemRefT_CplxT_double_1d result = getState(4);
        __catalyst__qis__State(&result, 0);
        CplxT_double *state = result.data_allocated;

        CHECK((state[0].real == Approx(0.0873321925).margin(1e-5) &&
               state[0].imag == Approx(-0.2823212367).margin(1e-5)));
        CHECK((state[2].real == Approx(-0.0953745058).margin(1e-5) &&
               state[2].imag == Approx(0.9505637859).margin(1e-5)));

        freeState(result);
        __catalyst__rt__qubit_release(wire1);
        __catalyst__rt__qubit_release(wire0);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
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
    __catalyst__rt__initialize(nullptr);
    int *a = (int *)_mlir_memref_to_llvm_alloc(sizeof(int));
    CHECK(a != NULL);
    *a = 1;
    __catalyst__rt__finalize();
}

TEST_CASE("Test memref aligned alloc", "[CoreQIS]")
{
    __catalyst__rt__initialize(nullptr);

    // macOS requires the alignment of 'aligned_alloc' is at least 8-byte
    int *a = (int *)_mlir_memref_to_llvm_aligned_alloc(8, 2 * sizeof(int));
    CHECK(a != NULL);
    *a = 1;
    __catalyst__rt__finalize();
}

TEST_CASE("Test memref free", "[CoreQIS]")
{
    __catalyst__rt__initialize(nullptr);
    int *a = (int *)_mlir_memref_to_llvm_alloc(sizeof(int));
    CHECK(a != NULL);
    *a = 1;
    _mlir_memref_to_llvm_free(a);
    __catalyst__rt__finalize();
}

TEST_CASE("Test memory transfer in rt", "[CoreQIS]")
{
    __catalyst__rt__initialize(nullptr);
    int *a = (int *)_mlir_memref_to_llvm_alloc(sizeof(int));
    bool is_in_rt = _mlir_memory_transfer(a);
    CHECK(is_in_rt);
    __catalyst__rt__finalize();
    free(a);
}

TEST_CASE("Test memory transfer not in rt", "[CoreQIS]")
{
    __catalyst__rt__initialize(nullptr);
    int *a = (int *)malloc(sizeof(int));
    bool is_in_rt = _mlir_memory_transfer(a);
    CHECK(!is_in_rt);
    __catalyst__rt__finalize();
    free(a);
}

TEST_CASE("Test __catalyst__qis__Measure", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QUBIT *wire0 = __catalyst__rt__qubit_allocate();

        QUBIT *wire1 = __catalyst__rt__qubit_allocate();

        __catalyst__qis__PauliX(wire0, NO_MODIFIERS);

        Result m = __catalyst__qis__Measure(wire0, -1);

        Result one = __catalyst__rt__result_get_one();
        CHECK(*m == *one);

        __catalyst__rt__qubit_release(wire1);
        __catalyst__rt__qubit_release(wire0);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__ Hadamard, PauliZ, IsingXX, IsingZZ, and SWAP", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

        __catalyst__qis__Hadamard(*target, NO_MODIFIERS);
        __catalyst__qis__PauliZ(*target, NO_MODIFIERS);
        __catalyst__qis__IsingXX(0.2, *ctrls, *target, NO_MODIFIERS);
        __catalyst__qis__IsingZZ(0.5, *ctrls, *target, NO_MODIFIERS);
        __catalyst__qis__SWAP(*ctrls, *target, NO_MODIFIERS);

        MemRefT_CplxT_double_1d result = getState(4);
        __catalyst__qis__State(&result, 0);
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
        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__ CRot, IsingXY and Toffoli", "[CoreQIS]")
{
    __catalyst__rt__initialize(nullptr);
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(3);

        QUBIT **target = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls_0 = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);
        QUBIT **ctrls_1 = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 2);

        __catalyst__qis__Hadamard(*target, NO_MODIFIERS);
        __catalyst__qis__PauliZ(*target, NO_MODIFIERS);
        __catalyst__qis__CRot(0.2, 0.5, 0.7, *ctrls_0, *target, NO_MODIFIERS);
        __catalyst__qis__IsingXY(0.2, *ctrls_0, *target, NO_MODIFIERS);
        __catalyst__qis__SWAP(*ctrls_0, *target, NO_MODIFIERS);
        __catalyst__qis__Toffoli(*ctrls_0, *ctrls_1, *target, NO_MODIFIERS);

        MemRefT_CplxT_double_1d result = getState(8);
        __catalyst__qis__State(&result, 0);
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
        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
    }
    __catalyst__rt__finalize();
}

TEST_CASE("Test __catalyst__qis__ GlobalPhase", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(1);

        __catalyst__qis__GlobalPhase(M_PI / 4, NO_MODIFIERS);

        MemRefT_CplxT_double_1d result = getState(2);
        __catalyst__qis__State(&result, 0);
        CplxT_double *state = result.data_allocated;

        CHECK((state[0].real == Approx(0.70710678).margin(1e-5) &&
               state[0].imag == Approx(-0.70710678).margin(1e-5)));

        freeState(result);
        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__ Hadamard, PauliX, IsingYY, CRX, and Expval", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __catalyst__qis__Hadamard(*target, NO_MODIFIERS);
        // qml.PauliX(wires=0)
        __catalyst__qis__PauliX(*target, NO_MODIFIERS);
        // qml.IsingYY(0.2, wires=[1,0])
        __catalyst__qis__IsingYY(0.2, *ctrls, *target, NO_MODIFIERS);
        // qml.CRX(0.4, wires=[1,0])
        __catalyst__qis__CRX(0.4, *ctrls, *target, NO_MODIFIERS);

        MemRefT_CplxT_double_1d result = getState(4);
        __catalyst__qis__State(&result, 0);
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
        auto obs = __catalyst__qis__NamedObs(ObsId::Hadamard, *ctrls);

        CHECK(__catalyst__qis__Expval(obs) == Approx(0.69301172).margin(1e-5));

        freeState(result);
        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__ PhaseShift", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __catalyst__qis__Hadamard(*target, NO_MODIFIERS);
        // qml.RX(0.123, wires=1)
        __catalyst__qis__RX(0.123, *ctrls, NO_MODIFIERS);
        // qml.PhaseShift(0.456, wires=0)
        __catalyst__qis__PhaseShift(0.456, *target, NO_MODIFIERS);

        MemRefT_CplxT_double_1d result = getState(4);
        __catalyst__qis__State(&result, 0);
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
        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__HermitianObs with an uninitialized matrix", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        MemRefT_CplxT_double_2d *matrix = nullptr;
        REQUIRE_THROWS_WITH(
            __catalyst__qis__HermitianObs(matrix, 0),
            Catch::Contains("[Function:__catalyst__qis__HermitianObs] Error in Catalyst Runtime: "
                            "The Hermitian matrix must be initialized"));

        delete matrix;
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__HermitianObs with invalid Hermitian matrix", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        MemRefT_CplxT_double_2d *matrix = new MemRefT_CplxT_double_2d;
        REQUIRE_THROWS_WITH(__catalyst__qis__HermitianObs(matrix, 0),
                            Catch::Contains("Invalid given Hermitian matrix"));

        delete matrix;
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__HermitianObs with invalid number of wires", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(1);

        QUBIT **target = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);

        MemRefT_CplxT_double_2d *matrix = new MemRefT_CplxT_double_2d;
        matrix->offset = 0;
        matrix->sizes[0] = 4;
        matrix->sizes[1] = 4;
        matrix->strides[0] = 1;
        REQUIRE_THROWS_WITH(__catalyst__qis__HermitianObs(matrix, 2, *target, *target),
                            Catch::Contains("Invalid number of wires"));

        delete matrix;
        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__HermitianObs and Expval", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __catalyst__qis__Hadamard(*target, NO_MODIFIERS);
        // qml.PauliX(wires=0)
        __catalyst__qis__PauliX(*target, NO_MODIFIERS);
        // qml.IsingYY(0.2, wires=[1,0])
        __catalyst__qis__IsingYY(0.2, *ctrls, *target, NO_MODIFIERS);
        // qml.CRX(0.4, wires=[1,0])
        __catalyst__qis__CRX(0.4, *ctrls, *target, NO_MODIFIERS);

        MemRefT_CplxT_double_1d result = getState(4);
        __catalyst__qis__State(&result, 0);
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
        __catalyst__qis__NamedObs(ObsId::Hadamard, *ctrls);

        CplxT_double matrix_data[4] = {{1.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}, {0.0, 0.0}};

        MemRefT_CplxT_double_2d *h_matrix = new MemRefT_CplxT_double_2d;
        h_matrix->data_allocated = matrix_data;
        h_matrix->data_aligned = matrix_data;
        h_matrix->offset = 0;
        h_matrix->sizes[0] = 2;
        h_matrix->sizes[1] = 2;
        h_matrix->strides[0] = 1;

        auto obs_h = __catalyst__qis__HermitianObs(h_matrix, 1, *ctrls);

        CHECK(obs_h == 1);
        CHECK(__catalyst__qis__Expval(obs_h) == Approx(0.9900332889).margin(1e-5));

        freeState(result);
        delete h_matrix;
        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__TensorObs with invalid number of observables", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        REQUIRE_THROWS_WITH(
            __catalyst__qis__TensorObs(0),
            Catch::Contains("[Function:__catalyst__qis__TensorObs] Error in Catalyst Runtime: "
                            "Invalid number of observables to create TensorProdObs"));

        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__TensorProdObs and Expval", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __catalyst__qis__Hadamard(*target, NO_MODIFIERS);
        // qml.Hadamard(wires=1)
        __catalyst__qis__Hadamard(*ctrls, NO_MODIFIERS);
        // qml.IsingYY(0.2, wires=[1,0])
        __catalyst__qis__IsingYY(0.6, *ctrls, *target, NO_MODIFIERS);
        // qml.CRX(0.4, wires=[1,0])
        __catalyst__qis__CRX(0.3, *ctrls, *target, NO_MODIFIERS);

        MemRefT_CplxT_double_1d result = getState(4);
        __catalyst__qis__State(&result, 0);

        auto obs_x = __catalyst__qis__NamedObs(ObsId::PauliX, *target);

        CplxT_double matrix_data[4] = {{1.0, 0.0}, {0.0, 3.0}, {2.0, 0.0}, {0.0, 5.0}};

        MemRefT_CplxT_double_2d *h_matrix = new MemRefT_CplxT_double_2d;
        h_matrix->data_allocated = matrix_data;
        h_matrix->data_aligned = matrix_data;
        h_matrix->offset = 0;
        h_matrix->sizes[0] = 2;
        h_matrix->sizes[1] = 2;
        h_matrix->strides[0] = 1;

        auto obs_h = __catalyst__qis__HermitianObs(h_matrix, 1, *ctrls);
        auto obs_t = __catalyst__qis__TensorObs(2, obs_h, obs_x);

        CHECK(obs_t == 2);

        CHECK(__catalyst__qis__Expval(obs_t) == Approx(1.5864438048).margin(1e-5));

        freeState(result);
        delete h_matrix;
        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__HamiltonianObs with invalid coefficients", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        MemRefT_double_1d *coeffs = nullptr;

        REQUIRE_THROWS_WITH(
            __catalyst__qis__HamiltonianObs(coeffs, 0),
            Catch::Contains("[Function:__catalyst__qis__HamiltonianObs] Error in Catalyst Runtime: "
                            "Invalid coefficients for computing Hamiltonian"));

        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE(
    "Test __catalyst__qis__HamiltonianObs with invalid number of coefficients and observables",
    "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        MemRefT_double_1d *coeffs = new MemRefT_double_1d;
        coeffs->offset = 0;
        coeffs->sizes[0] = 2;
        coeffs->strides[0] = 1;

        REQUIRE_THROWS_WITH(__catalyst__qis__HamiltonianObs(coeffs, 0),
                            Catch::Contains("Invalid coefficients for computing Hamiltonian"));

        delete coeffs;

        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__HamiltonianObs(h, x) and Expval", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __catalyst__qis__Hadamard(*target, NO_MODIFIERS);
        // qml.Hadamard(wires=1)
        __catalyst__qis__Hadamard(*ctrls, NO_MODIFIERS);
        // qml.IsingYY(0.2, wires=[1,0])
        __catalyst__qis__IsingYY(0.6, *ctrls, *target, NO_MODIFIERS);
        // qml.CRX(0.4, wires=[1,0])
        __catalyst__qis__CRX(0.3, *ctrls, *target, NO_MODIFIERS);

        MemRefT_CplxT_double_1d result = getState(4);
        __catalyst__qis__State(&result, 0);

        auto obs_x = __catalyst__qis__NamedObs(ObsId::PauliX, *target);

        CplxT_double matrix_data[4] = {{1.0, 0.0}, {0.0, 3.0}, {2.0, 0.0}, {0.0, 5.0}};

        MemRefT_CplxT_double_2d *h_matrix = new MemRefT_CplxT_double_2d;
        h_matrix->data_allocated = matrix_data;
        h_matrix->data_aligned = matrix_data;
        h_matrix->offset = 0;
        h_matrix->sizes[0] = 2;
        h_matrix->sizes[1] = 2;
        h_matrix->strides[0] = 1;

        auto obs_h = __catalyst__qis__HermitianObs(h_matrix, 1, *ctrls);

        double coeffs_data[2] = {0.4, 0.7};
        MemRefT_double_1d *coeffs = new MemRefT_double_1d;
        coeffs->data_allocated = coeffs_data;
        coeffs->data_aligned = coeffs_data;
        coeffs->offset = 0;
        coeffs->sizes[0] = 2;
        coeffs->strides[0] = 1;

        auto obs_hamiltonian = __catalyst__qis__HamiltonianObs(coeffs, 2, obs_h, obs_x);

        CHECK(obs_hamiltonian == 2);

        CHECK(__catalyst__qis__Expval(obs_hamiltonian) == Approx(1.1938250042).margin(1e-5));

        freeState(result);
        delete h_matrix;
        delete coeffs;
        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__HamiltonianObs(t) and Expval", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __catalyst__qis__Hadamard(*target, NO_MODIFIERS);
        // qml.Hadamard(wires=1)
        __catalyst__qis__Hadamard(*ctrls, NO_MODIFIERS);
        // qml.IsingYY(0.2, wires=[1,0])
        __catalyst__qis__IsingYY(0.6, *ctrls, *target, NO_MODIFIERS);
        // qml.CRX(0.4, wires=[1,0])
        __catalyst__qis__CRX(0.3, *ctrls, *target, NO_MODIFIERS);

        MemRefT_CplxT_double_1d result = getState(4);
        __catalyst__qis__State(&result, 0);

        auto obs_x = __catalyst__qis__NamedObs(ObsId::PauliX, *target);

        CplxT_double matrix_data[4] = {{1.0, 0.0}, {0.0, 3.0}, {2.0, 0.0}, {0.0, 5.0}};

        MemRefT_CplxT_double_2d *h_matrix = new MemRefT_CplxT_double_2d;
        h_matrix->data_allocated = matrix_data;
        h_matrix->data_aligned = matrix_data;
        h_matrix->offset = 0;
        h_matrix->sizes[0] = 2;
        h_matrix->sizes[1] = 2;
        h_matrix->strides[0] = 1;

        auto obs_h = __catalyst__qis__HermitianObs(h_matrix, 1, *ctrls);
        auto obs_t = __catalyst__qis__TensorObs(2, obs_h, obs_x);

        double coeffs_data[1] = {0.4};
        MemRefT_double_1d *coeffs = new MemRefT_double_1d;
        coeffs->data_allocated = coeffs_data;
        coeffs->data_aligned = coeffs_data;
        coeffs->offset = 0;
        coeffs->sizes[0] = 1;
        coeffs->strides[0] = 1;

        auto obs_hamiltonian = __catalyst__qis__HamiltonianObs(coeffs, 1, obs_t);

        CHECK(obs_hamiltonian == 3);

        CHECK(__catalyst__qis__Expval(obs_hamiltonian) == Approx(0.6345775219).margin(1e-5));

        freeState(result);
        delete h_matrix;
        delete coeffs;
        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__HamiltonianObs(h, Ham(x)) and Expval", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __catalyst__qis__Hadamard(*target, NO_MODIFIERS);
        // qml.Hadamard(wires=1)
        __catalyst__qis__Hadamard(*ctrls, NO_MODIFIERS);
        // qml.IsingYY(0.2, wires=[1,0])
        __catalyst__qis__IsingYY(0.6, *ctrls, *target, NO_MODIFIERS);
        // qml.CRX(0.4, wires=[1,0])
        __catalyst__qis__CRX(0.3, *ctrls, *target, NO_MODIFIERS);

        MemRefT_CplxT_double_1d result = getState(4);
        __catalyst__qis__State(&result, 0);

        auto obs_x = __catalyst__qis__NamedObs(ObsId::PauliX, *target);

        double coeffs1_data[1] = {0.2};
        MemRefT_double_1d *coeffs1 = new MemRefT_double_1d;
        coeffs1->data_allocated = coeffs1_data;
        coeffs1->data_aligned = coeffs1_data;
        coeffs1->offset = 0;
        coeffs1->sizes[0] = 1;
        coeffs1->strides[0] = 1;
        auto obs_ham1 = __catalyst__qis__HamiltonianObs(coeffs1, 1, obs_x);

        CplxT_double matrix_data[4] = {{1.0, 0.0}, {0.0, 3.0}, {2.0, 0.0}, {0.0, 5.0}};

        MemRefT_CplxT_double_2d *h_matrix = new MemRefT_CplxT_double_2d;
        h_matrix->data_allocated = matrix_data;
        h_matrix->data_aligned = matrix_data;
        h_matrix->offset = 0;
        h_matrix->sizes[0] = 2;
        h_matrix->sizes[1] = 2;
        h_matrix->strides[0] = 1;

        auto obs_h = __catalyst__qis__HermitianObs(h_matrix, 1, *ctrls);
        double coeffs_data[2] = {0.4, 0.7};
        MemRefT_double_1d *coeffs2 = new MemRefT_double_1d;
        coeffs2->data_allocated = coeffs_data;
        coeffs2->data_aligned = coeffs_data;
        coeffs2->offset = 0;
        coeffs2->sizes[0] = 2;
        coeffs2->strides[0] = 1;

        auto obs_hamiltonian = __catalyst__qis__HamiltonianObs(coeffs2, 2, obs_h, obs_ham1);

        CHECK(obs_hamiltonian == 3);

        CHECK(__catalyst__qis__Expval(obs_hamiltonian) == Approx(0.7316370598).margin(1e-5));

        freeState(result);
        delete coeffs1;
        delete h_matrix;
        delete coeffs2;
        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__ Hadamard, PauliX, IsingYY, CRX, and Expval_arr", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __catalyst__qis__Hadamard(*target, NO_MODIFIERS);
        // qml.PauliX(wires=0)
        __catalyst__qis__PauliX(*target, NO_MODIFIERS);
        // qml.IsingYY(0.2, wires=[1,0])
        __catalyst__qis__IsingYY(0.2, *ctrls, *target, NO_MODIFIERS);
        // qml.CRX(0.4, wires=[1,0])
        __catalyst__qis__CRX(0.4, *ctrls, *target, NO_MODIFIERS);

        MemRefT_CplxT_double_1d result = getState(4);
        __catalyst__qis__State(&result, 0);
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
        auto obs = __catalyst__qis__NamedObs(ObsId::Hadamard, *ctrls);

        CHECK(__catalyst__qis__Expval(obs) == Approx(0.69301172).margin(1e-5));

        freeState(result);
        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__ Hadamard, ControlledPhaseShift, IsingYY, CRX, and Var",
          "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __catalyst__qis__Hadamard(*target, NO_MODIFIERS);
        // qml.ControlledPhaseShift(0.6, wires=[0,1])
        __catalyst__qis__ControlledPhaseShift(0.6, *target, *ctrls, NO_MODIFIERS);
        // qml.IsingYY(0.2, wires=[0, 1])
        __catalyst__qis__IsingYY(0.2, *target, *ctrls, NO_MODIFIERS);
        // qml.CRX(0.4, wires=[1,0])
        __catalyst__qis__CRX(0.4, *target, *ctrls, NO_MODIFIERS);

        MemRefT_CplxT_double_1d result = getState(4);
        __catalyst__qis__State(&result, 0);
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
        auto obs = __catalyst__qis__NamedObs(ObsId::PauliZ, *ctrls);

        CHECK(__catalyst__qis__Variance(obs) == Approx(0.0394695).margin(1e-5));

        freeState(result);
        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__ Hadamard, PauliX, IsingYY, CRX, and Probs", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __catalyst__qis__Hadamard(*target, NO_MODIFIERS);
        // qml.PauliX(wires=0)
        __catalyst__qis__PauliX(*target, NO_MODIFIERS);
        // qml.IsingYY(0.2, wires=[1,0])
        __catalyst__qis__IsingYY(0.2, *ctrls, *target, NO_MODIFIERS);
        // qml.CRX(0.4, wires=[1,0])
        __catalyst__qis__CRX(0.4, *ctrls, *target, NO_MODIFIERS);

        size_t buffer_len = 4;
        double *buffer = new double[buffer_len];
        MemRefT_double_1d result = {buffer, buffer, 0, {buffer_len}, {1}};
        __catalyst__qis__Probs(&result, 0);
        double *probs = result.data_allocated;

        CHECK((probs[0] + probs[2]) == Approx(0.9900332889).margin(1e-5));
        CHECK((probs[1] + probs[3]) == Approx(0.0099667111).margin(1e-5));

        delete[] buffer;
        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__ Hadamard, PauliX, IsingYY, CRX, and partial Probs", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __catalyst__qis__Hadamard(*target, NO_MODIFIERS);
        // qml.PauliX(wires=0)
        __catalyst__qis__PauliX(*target, NO_MODIFIERS);
        // qml.IsingYY(0.2, wires=[1,0])
        __catalyst__qis__IsingYY(0.2, *ctrls, *target, NO_MODIFIERS);
        // qml.CRX(0.4, wires=[1,0])
        __catalyst__qis__CRX(0.4, *ctrls, *target, NO_MODIFIERS);

        size_t buffer_len = 2;
        double *buffer = new double[buffer_len];
        MemRefT_double_1d result = {buffer, buffer, 0, {buffer_len}, {1}};
        __catalyst__qis__Probs(&result, 1, ctrls[0]);
        double *probs = result.data_allocated;

        CHECK(probs[0] == Approx(0.9900332889).margin(1e-5));
        CHECK(probs[1] == Approx(0.0099667111).margin(1e-5));

        delete[] buffer;
        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__State on the heap using malloc", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __catalyst__qis__Hadamard(*target, NO_MODIFIERS);
        // qml.ControlledPhaseShift(0.6, wires=[0,1])
        __catalyst__qis__ControlledPhaseShift(0.6, *target, *ctrls, NO_MODIFIERS);
        // qml.IsingYY(0.2, wires=[0, 1])
        __catalyst__qis__IsingYY(0.2, *target, *ctrls, NO_MODIFIERS);
        // qml.CRX(0.4, wires=[1,0])
        __catalyst__qis__CRX(0.4, *target, *ctrls, NO_MODIFIERS);

        MemRefT_CplxT_double_1d result = getState(4);
        __catalyst__qis__State(&result, 0);
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
        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__Measure with false", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QUBIT *target = __catalyst__rt__qubit_allocate(); // id = 0

        // qml.Hadamard(wires=0)
        __catalyst__qis__RY(0.0, target, NO_MODIFIERS);

        Result mres = __catalyst__qis__Measure(target, -1);

        Result zero = __catalyst__rt__result_get_zero();
        CHECK(__catalyst__rt__result_equal(mres, zero));

        __catalyst__rt__qubit_release(target);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__Measure with true", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QUBIT *target = __catalyst__rt__qubit_allocate(); // id = 0

        // qml.Hadamard(wires=0)
        __catalyst__qis__RY(3.14, target, NO_MODIFIERS);

        Result mres = __catalyst__qis__Measure(target, -1);

        Result one = __catalyst__rt__result_get_one();
        CHECK(__catalyst__rt__result_equal(mres, one));

        __catalyst__rt__qubit_release(target);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__MultiRZ", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(2);

        QUBIT **q0 = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **q1 = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

        __catalyst__qis__RX(M_PI, *q0, NO_MODIFIERS);
        __catalyst__qis__Hadamard(*q0, NO_MODIFIERS);
        __catalyst__qis__Hadamard(*q1, NO_MODIFIERS);
        __catalyst__qis__MultiRZ(M_PI, NO_MODIFIERS, 2, *q0, *q1);
        __catalyst__qis__Hadamard(*q0, NO_MODIFIERS);
        __catalyst__qis__Hadamard(*q1, NO_MODIFIERS);

        Result q0_m = __catalyst__qis__Measure(*q0, -1);
        Result q1_m = __catalyst__qis__Measure(*q1, -1);

        Result zero = __catalyst__rt__result_get_zero();
        Result one = __catalyst__rt__result_get_one();
        CHECK(__catalyst__rt__result_equal(q0_m, zero));
        CHECK(__catalyst__rt__result_equal(q1_m, one));

        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__CSWAP ", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(3);

        QUBIT **q0 = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **q1 = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);
        QUBIT **q2 = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 2);

        __catalyst__qis__RX(M_PI, *q0, NO_MODIFIERS);
        __catalyst__qis__RX(M_PI, *q1, NO_MODIFIERS);
        __catalyst__qis__CSWAP(*q0, *q1, *q2, NO_MODIFIERS);

        Result q1_m = __catalyst__qis__Measure(*q1, -1);
        Result q2_m = __catalyst__qis__Measure(*q2, -1);

        Result zero = __catalyst__rt__result_get_zero();
        Result one = __catalyst__rt__result_get_one();

        CHECK(__catalyst__rt__result_equal(q1_m, zero));
        CHECK(__catalyst__rt__result_equal(q2_m, one));

        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__Counts with num_qubits=2 calling Hadamard, ControlledPhaseShift, "
          "IsingYY, and CRX quantum operations",
          "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __catalyst__qis__Hadamard(*target, NO_MODIFIERS);
        // qml.ControlledPhaseShift(0.6, wires=[0,1])
        __catalyst__qis__ControlledPhaseShift(0.6, *target, *ctrls, NO_MODIFIERS);
        // qml.IsingYY(0.2, wires=[0, 1])
        __catalyst__qis__IsingYY(0.2, *target, *ctrls, NO_MODIFIERS);
        // qml.CRX(0.4, wires=[1,0])
        __catalyst__qis__CRX(0.4, *target, *ctrls, NO_MODIFIERS);

        constexpr size_t shots = 1000;

        PairT_MemRefT_double_int64_1d result = getCounts(4);
        __catalyst__qis__Counts(&result, shots, 0);
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
        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__Counts with num_qubits=2 PartialCounts calling Hadamard, "
          "ControlledPhaseShift, IsingYY, and CRX quantum operations",
          "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __catalyst__qis__Hadamard(*target, NO_MODIFIERS);
        // qml.ControlledPhaseShift(0.6, wires=[0,1])
        __catalyst__qis__ControlledPhaseShift(0.6, *target, *ctrls, NO_MODIFIERS);
        // qml.IsingYY(0.2, wires=[0, 1])
        __catalyst__qis__IsingYY(0.2, *target, *ctrls, NO_MODIFIERS);
        // qml.CRX(0.4, wires=[1,0])
        __catalyst__qis__CRX(0.4, *target, *ctrls, NO_MODIFIERS);

        constexpr size_t shots = 1000;

        PairT_MemRefT_double_int64_1d result = getCounts(2);
        __catalyst__qis__Counts(&result, shots, 1, ctrls[0]);
        double *eigvals = result.first.data_allocated;
        int64_t *counts = result.second.data_allocated;

        CHECK(counts[0] + counts[1] == shots);
        CHECK(eigvals[0] + 1 == eigvals[1]);

        freeCounts(result);
        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__Sample with num_qubits=2 calling Hadamard, ControlledPhaseShift, "
          "IsingYY, and CRX quantum operations",
          "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __catalyst__qis__Hadamard(*target, NO_MODIFIERS);
        // qml.ControlledPhaseShift(0.6, wires=[0,1])
        __catalyst__qis__ControlledPhaseShift(0.6, *target, *ctrls, NO_MODIFIERS);
        // qml.IsingYY(0.2, wires=[0, 1])
        __catalyst__qis__IsingYY(0.2, *target, *ctrls, NO_MODIFIERS);
        // qml.CRX(0.4, wires=[1,0])
        __catalyst__qis__CRX(0.4, *target, *ctrls, NO_MODIFIERS);

        constexpr size_t n = 2;
        constexpr size_t shots = 1000;

        double *buffer = new double[shots * n];
        MemRefT_double_2d result = {buffer, buffer, 0, {shots, n}, {n, 1}};
        __catalyst__qis__Sample(&result, shots, 0);
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

        auto obs = __catalyst__qis__NamedObs(ObsId::PauliZ, *ctrls);

        CHECK(__catalyst__qis__Variance(obs) == Approx(0.0394695).margin(1e-5));

        delete[] buffer;

        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__Sample with num_qubits=2 and PartialSample calling Hadamard, "
          "ControlledPhaseShift, IsingYY, and CRX quantum operations",
          "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __catalyst__qis__Hadamard(*target, NO_MODIFIERS);
        // qml.ControlledPhaseShift(0.6, wires=[0,1])
        __catalyst__qis__ControlledPhaseShift(0.6, *target, *ctrls, NO_MODIFIERS);
        // qml.IsingYY(0.2, wires=[0, 1])
        __catalyst__qis__IsingYY(0.2, *target, *ctrls, NO_MODIFIERS);
        // qml.CRX(0.4, wires=[1,0])
        __catalyst__qis__CRX(0.4, *target, *ctrls, NO_MODIFIERS);

        constexpr size_t n = 1;
        constexpr size_t shots = 1000;

        double *buffer = new double[shots * n];
        MemRefT_double_2d result = {buffer, buffer, 0, {shots, n}, {n, 1}};
        __catalyst__qis__Sample(&result, shots, 1, ctrls[0]);
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

        auto obs = __catalyst__qis__NamedObs(ObsId::PauliZ, *ctrls);

        CHECK(__catalyst__qis__Variance(obs) == Approx(0.0394695).margin(1e-5));

        delete[] buffer;

        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__QubitUnitary with an uninitialized matrix", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QUBIT *target = __catalyst__rt__qubit_allocate(); // id = 0
        MemRefT_CplxT_double_2d *matrix = nullptr;

        REQUIRE_THROWS_WITH(
            __catalyst__qis__QubitUnitary(matrix, NO_MODIFIERS, 1, target),
            Catch::Contains("[Function:__catalyst__qis__QubitUnitary] Error in Catalyst Runtime: "
                            "The QubitUnitary matrix must be initialized"));

        __catalyst__rt__qubit_release(target);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__QubitUnitary with invalid number of wires", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QUBIT *target = __catalyst__rt__qubit_allocate(); // id = 0
        MemRefT_CplxT_double_2d *matrix = new MemRefT_CplxT_double_2d;

        REQUIRE_THROWS_WITH(__catalyst__qis__QubitUnitary(matrix, NO_MODIFIERS, 3, target, false),
                            Catch::Contains("Invalid number of wires"));
        REQUIRE_THROWS_WITH(__catalyst__qis__QubitUnitary(matrix, NO_MODIFIERS, 3, target, true),
                            Catch::Contains("Invalid number of wires"));

        delete matrix;
        __catalyst__rt__qubit_release(target);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__QubitUnitary with invalid matrix", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QUBIT *target = __catalyst__rt__qubit_allocate(); // id = 0
        MemRefT_CplxT_double_2d *matrix = new MemRefT_CplxT_double_2d;

        matrix->offset = 0;
        matrix->sizes[0] = 1;
        matrix->sizes[1] = 1;
        matrix->strides[0] = 1;

        REQUIRE_THROWS_WITH(__catalyst__qis__QubitUnitary(matrix, NO_MODIFIERS, 1, target),
                            Catch::Contains("Invalid given QubitUnitary matrix"));
        REQUIRE_THROWS_WITH(__catalyst__qis__QubitUnitary(matrix, NO_MODIFIERS, 1, target, true),
                            Catch::Contains("Invalid given QubitUnitary matrix"));

        delete matrix;
        __catalyst__rt__qubit_release(target);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__QubitUnitary with num_qubits=2", "[CoreQIS]")
{
    __catalyst__rt__initialize(nullptr);
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(2);

        QUBIT **target = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **ctrls = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

        __catalyst__qis__Hadamard(*target, NO_MODIFIERS);
        __catalyst__qis__CNOT(*target, *ctrls, NO_MODIFIERS);

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

        __catalyst__qis__QubitUnitary(matrix, NO_MODIFIERS, 1, *target);

        MemRefT_CplxT_double_1d result = getState(4);
        __catalyst__qis__State(&result, 0);
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
        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
    }
    __catalyst__rt__finalize();
}

TEST_CASE("Test the main porperty of the adjoint quantum operations", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(3);

        QUBIT **q0 = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **q1 = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);
        QUBIT **q2 = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 2);

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

        Modifiers adjoint_modifier = {true, 0, nullptr, nullptr};

        __catalyst__qis__QubitUnitary(matrix, NO_MODIFIERS, 1, *q0);
        __catalyst__qis__MultiRZ(theta, NO_MODIFIERS, 2, *q0, *q1);
        __catalyst__qis__Toffoli(*q0, *q1, *q2, NO_MODIFIERS);
        __catalyst__qis__CSWAP(*q0, *q1, *q2, NO_MODIFIERS);
        __catalyst__qis__CRot(theta, theta, theta, *q0, *q1, NO_MODIFIERS);
        __catalyst__qis__CRZ(theta, *q0, *q1, NO_MODIFIERS);
        __catalyst__qis__CRY(theta, *q0, *q1, NO_MODIFIERS);
        __catalyst__qis__CRX(theta, *q0, *q1, NO_MODIFIERS);
        __catalyst__qis__ControlledPhaseShift(theta, *q0, *q1, NO_MODIFIERS);
        __catalyst__qis__IsingZZ(theta, *q0, *q1, NO_MODIFIERS);
        __catalyst__qis__IsingXY(theta, *q0, *q1, NO_MODIFIERS);
        __catalyst__qis__IsingYY(theta, *q0, *q1, NO_MODIFIERS);
        __catalyst__qis__IsingXX(theta, *q0, *q1, NO_MODIFIERS);
        __catalyst__qis__SWAP(*q0, *q1, NO_MODIFIERS);
        __catalyst__qis__CZ(*q0, *q1, NO_MODIFIERS);
        __catalyst__qis__CY(*q0, *q1, NO_MODIFIERS);
        __catalyst__qis__CNOT(*q0, *q1, NO_MODIFIERS);
        __catalyst__qis__Rot(theta, theta, theta, *q0, NO_MODIFIERS);
        __catalyst__qis__RZ(theta, *q0, NO_MODIFIERS);
        __catalyst__qis__RY(theta, *q0, NO_MODIFIERS);
        __catalyst__qis__RX(theta, *q0, NO_MODIFIERS);
        __catalyst__qis__PhaseShift(theta, *q0, NO_MODIFIERS);
        __catalyst__qis__T(*q0, NO_MODIFIERS);
        __catalyst__qis__S(*q0, NO_MODIFIERS);
        __catalyst__qis__Hadamard(*q0, NO_MODIFIERS);
        __catalyst__qis__PauliZ(*q0, NO_MODIFIERS);
        __catalyst__qis__PauliY(*q0, NO_MODIFIERS);
        __catalyst__qis__PauliX(*q0, NO_MODIFIERS);
        __catalyst__qis__Identity(*q0, NO_MODIFIERS);

        __catalyst__qis__Identity(*q0, &adjoint_modifier);
        __catalyst__qis__PauliX(*q0, &adjoint_modifier);
        __catalyst__qis__PauliY(*q0, &adjoint_modifier);
        __catalyst__qis__PauliZ(*q0, &adjoint_modifier);
        __catalyst__qis__Hadamard(*q0, &adjoint_modifier);
        __catalyst__qis__S(*q0, &adjoint_modifier);
        __catalyst__qis__T(*q0, &adjoint_modifier);
        __catalyst__qis__PhaseShift(theta, *q0, &adjoint_modifier);
        __catalyst__qis__RX(theta, *q0, &adjoint_modifier);
        __catalyst__qis__RY(theta, *q0, &adjoint_modifier);
        __catalyst__qis__RZ(theta, *q0, &adjoint_modifier);
        __catalyst__qis__Rot(theta, theta, theta, *q0, &adjoint_modifier);
        __catalyst__qis__CNOT(*q0, *q1, &adjoint_modifier);
        __catalyst__qis__CY(*q0, *q1, &adjoint_modifier);
        __catalyst__qis__CZ(*q0, *q1, &adjoint_modifier);
        __catalyst__qis__SWAP(*q0, *q1, &adjoint_modifier);
        __catalyst__qis__IsingXX(theta, *q0, *q1, &adjoint_modifier);
        __catalyst__qis__IsingYY(theta, *q0, *q1, &adjoint_modifier);
        __catalyst__qis__IsingXY(theta, *q0, *q1, &adjoint_modifier);
        __catalyst__qis__IsingZZ(theta, *q0, *q1, &adjoint_modifier);
        __catalyst__qis__ControlledPhaseShift(theta, *q0, *q1, &adjoint_modifier);
        __catalyst__qis__CRX(theta, *q0, *q1, &adjoint_modifier);
        __catalyst__qis__CRY(theta, *q0, *q1, &adjoint_modifier);
        __catalyst__qis__CRZ(theta, *q0, *q1, &adjoint_modifier);
        __catalyst__qis__CRot(theta, theta, theta, *q0, *q1, &adjoint_modifier);
        __catalyst__qis__CSWAP(*q0, *q1, *q2, &adjoint_modifier);
        __catalyst__qis__Toffoli(*q0, *q1, *q2, &adjoint_modifier);
        __catalyst__qis__MultiRZ(theta, &adjoint_modifier, 2, *q0, *q1);
        __catalyst__qis__QubitUnitary(matrix, &adjoint_modifier, 1, *q0);

        MemRefT_CplxT_double_1d result = getState(8);
        __catalyst__qis__State(&result, 0);
        CplxT_double *stateVec = result.data_allocated;

        CHECK(stateVec[0].real == Approx(1.0).margin(1e-5));
        CHECK(stateVec[0].imag == Approx(0.0).margin(1e-5));
        for (size_t i = 1; i < 8; i++) {
            CHECK(stateVec[i].real == Approx(0.0).margin(1e-5));
            CHECK(stateVec[i].imag == Approx(0.0).margin(1e-5));
        }

        freeState(result);
        delete matrix;
        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test that an exception is raised unconditionally", "[CoreQIS]")
{
    auto devices = getDevices();
    auto &[rtd_lib, rtd_name, rtd_kwargs] = devices[0];

    __catalyst__rt__initialize(nullptr);
    __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                (int8_t *)rtd_kwargs.c_str());

    REQUIRE_THROWS_WITH(__catalyst__host__rt__unrecoverable_error(),
                        Catch::Contains("Unrecoverable error"));

    __catalyst__rt__device_release();
    __catalyst__rt__finalize();
}

TEST_CASE("Test that an exception if CNOT is controlled with the same qubit", "[CoreQIS]")
{
    auto devices = getDevices();
    auto &[rtd_lib, rtd_name, rtd_kwargs] = devices[0];

    __catalyst__rt__initialize(nullptr);
    __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                (int8_t *)rtd_kwargs.c_str());

    QirArray *qs = __catalyst__rt__qubit_allocate_array(1);

    QUBIT **target = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);

    REQUIRE_THROWS_WITH(__catalyst__qis__CNOT(*target, *target, NO_MODIFIERS),
                        Catch::Contains("Invalid input for CNOT gate."));

    __catalyst__rt__qubit_release_array(qs);
    __catalyst__rt__device_release();
    __catalyst__rt__finalize();
}

TEST_CASE("Test __catalyst__qis__ Hadamard, IsingZZ", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(2);

        QUBIT **wire0 = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
        QUBIT **wire1 = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

        // qml.Hadamard(wires=0)
        __catalyst__qis__Hadamard(*wire0, NO_MODIFIERS);
        // qml.Hadamard(wires=1)
        __catalyst__qis__Hadamard(*wire1, NO_MODIFIERS);
        // qml.IsingZZ(M_PI_4, wires=[1,0])
        __catalyst__qis__IsingZZ(M_PI_4, *wire1, *wire0, NO_MODIFIERS);

        MemRefT_CplxT_double_1d result = getState(4);
        __catalyst__qis__State(&result, 0);
        CplxT_double *state = result.data_allocated;

        CHECK(state[0].real == Approx(0.4619397663).margin(1e-5));
        CHECK(state[0].imag == Approx(-0.1913417162).margin(1e-5));
        CHECK(state[1].real == Approx(0.4619397663).margin(1e-5));
        CHECK(state[1].imag == Approx(0.1913417162).margin(1e-5));
        CHECK(state[2].real == Approx(0.4619397663).margin(1e-5));
        CHECK(state[2].imag == Approx(0.1913417162).margin(1e-5));
        CHECK(state[3].real == Approx(0.4619397663).margin(1e-5));
        CHECK(state[3].imag == Approx(-0.1913417162).margin(1e-5));

        freeState(result);
        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

TEST_CASE("Test __catalyst__qis__SetState", "[CoreQIS]")
{
    for (const auto &[rtd_lib, rtd_name, rtd_kwargs] : getDevices()) {
        __catalyst__rt__initialize(nullptr);
        __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                    (int8_t *)rtd_kwargs.c_str());

        QirArray *qs = __catalyst__rt__qubit_allocate_array(1);

        MemRefT_CplxT_double_1d state = getState(2);
        state.data_aligned[0] = {0.5, 0.5};
        state.data_aligned[1] = {0.0, 0.0};

        __catalyst__qis__SetState(&state);
        MemRefT_CplxT_double_1d result = getState(2);

        __catalyst__qis__State(&result, 0);
        CplxT_double *buffer = result.data_allocated;

        CHECK(buffer[0].real == Approx(0.5).margin(1e-5));
        CHECK(buffer[0].imag == Approx(0.5).margin(1e-5));
        CHECK(buffer[1].real == Approx(0.0).margin(1e-5));
        CHECK(buffer[1].imag == Approx(0.0).margin(1e-5));

        freeState(state);
        freeState(result);
        __catalyst__rt__qubit_release_array(qs);
        __catalyst__rt__device_release();
        __catalyst__rt__finalize();
    }
}

