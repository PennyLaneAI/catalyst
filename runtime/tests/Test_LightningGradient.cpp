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

#include "LightningUtils.hpp"
#include "QuantumDevice.hpp"
#include "RuntimeCAPI.h"

#include <catch2/catch.hpp>

using namespace Catalyst::Runtime;

TEST_CASE("Test __quantum__qis__Gradient with numAlloc=0", "[Gradient]")
{
    __quantum__rt__initialize();

    REQUIRE_NOTHROW(__quantum__qis__Gradient(0, nullptr));

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__Gradient_params with numAlloc=0", "[Gradient]")
{
    __quantum__rt__initialize();

    REQUIRE_THROWS_WITH(__quantum__qis__Gradient_params(nullptr, 0, nullptr),
                        Catch::Contains("Invalid number of trainable parameters"));

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__Gradient_params for zero number of obs", "[Gradient]")
{
    // Test qis__Gradient:
    int64_t trainParams[1] = {0};
    MemRefT_double_1d *results = new MemRefT_double_1d();
    MemRefT_int64_1d *tp = new MemRefT_int64_1d();
    tp->data_aligned = trainParams;
    tp->data_allocated = trainParams;
    tp->sizes[0] = 1;
    tp->strides[0] = 0;

    __quantum__rt__initialize();

    QUBIT *q = __quantum__rt__qubit_allocate();

    __quantum__rt__toggle_recorder(/* activate_cm */ true);

    __quantum__qis__S(q);
    __quantum__qis__T(q);

    REQUIRE_NOTHROW(__quantum__qis__Gradient_params(tp, 0, results));

    __quantum__rt__toggle_recorder(/* activate_cm */ false);

    delete results;
    delete tp;

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__Gradient and __quantum__qis__Gradient_params "
          "with invalid results",
          "[Gradient]")
{
    // Test qis__Gradient:
    int64_t trainParams[1] = {0};
    MemRefT_double_1d *results = new MemRefT_double_1d();
    MemRefT_int64_1d *tp = new MemRefT_int64_1d();
    tp->data_aligned = trainParams;
    tp->data_allocated = trainParams;
    tp->sizes[0] = 1;
    tp->strides[0] = 0;

    __quantum__rt__initialize();

    QUBIT *q = __quantum__rt__qubit_allocate();

    __quantum__rt__toggle_recorder(/* activate_cm */ true);

    __quantum__qis__RX(-M_PI / 7, q);

    auto obs = __quantum__qis__NamedObs(ObsId::PauliZ, q);

    __quantum__qis__Expval(obs);

    REQUIRE_THROWS_WITH(__quantum__qis__Gradient(2, results),
                        Catch::Contains("Invalid number of results"));

    REQUIRE_THROWS_WITH(__quantum__qis__Gradient_params(tp, 2, results),
                        Catch::Contains("Invalid number of results"));

    __quantum__rt__toggle_recorder(/* activate_cm */ false);

    delete results;
    delete tp;

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__Gradient and __quantum__qis__Gradient_params "
          "with Var",
          "[Gradient]")
{
    // Test qis__Gradient:
    int64_t trainParams[1] = {0};
    MemRefT_double_1d *results = new MemRefT_double_1d();
    MemRefT_int64_1d *tp = new MemRefT_int64_1d();
    tp->data_aligned = trainParams;
    tp->data_allocated = trainParams;
    tp->sizes[0] = 1;
    tp->strides[0] = 0;

    __quantum__rt__initialize();

    QUBIT *q = __quantum__rt__qubit_allocate();

    __quantum__rt__toggle_recorder(/* activate_cm */ true);

    __quantum__qis__RX(-M_PI / 7, q);

    auto obs = __quantum__qis__NamedObs(ObsId::PauliZ, q);

    __quantum__qis__Variance(obs);

    REQUIRE_THROWS_WITH(__quantum__qis__Gradient(1, results),
                        Catch::Contains("Unsupported measurements to compute gradient"));

    REQUIRE_THROWS_WITH(__quantum__qis__Gradient_params(tp, 1, results),
                        Catch::Contains("Unsupported measurements to compute gradient"));

    __quantum__rt__toggle_recorder(/* activate_cm */ false);

    delete results;
    delete tp;

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__Gradient and __quantum__qis__Gradient_params "
          "Op=RX, Obs=Z",
          "[Gradient]")
{
    // Test qis__Gradient:
    int64_t trainParams[1] = {0};
    MemRefT_double_1d *result = new MemRefT_double_1d();
    MemRefT_double_1d *result_tp = new MemRefT_double_1d();
    MemRefT_int64_1d *tp_memref = new MemRefT_int64_1d();
    tp_memref->data_aligned = trainParams;
    tp_memref->data_allocated = trainParams;
    tp_memref->sizes[0] = 1;
    tp_memref->strides[0] = 0;

    __quantum__rt__initialize();

    QUBIT *q = __quantum__rt__qubit_allocate();

    __quantum__rt__toggle_recorder(/* activate_cm */ true);

    __quantum__qis__RX(-M_PI / 7, q);

    auto obs_idx_0 = __quantum__qis__NamedObs(ObsId::PauliZ, q);

    __quantum__qis__Expval(obs_idx_0);

    __quantum__qis__Gradient_params(tp_memref, 1, result_tp);

    __quantum__qis__Gradient(1, result);

    __quantum__rt__toggle_recorder(/* activate_cm */ false);

    CHECK(-sin(-M_PI / 7) == Approx(result_tp->data_aligned[0]));
    CHECK(-sin(-M_PI / 7) == Approx(result->data_aligned[0]));

    delete result_tp;
    delete result;
    delete tp_memref;

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__Gradient and __quantum__qis__Gradient_params "
          "Op=RX, Obs=Hermitian",
          "[Gradient]")
{
    // Test qis__Gradient:
    int64_t trainParams[1] = {0};
    MemRefT_double_1d *result = new MemRefT_double_1d();
    MemRefT_double_1d *result_tp = new MemRefT_double_1d();
    MemRefT_int64_1d *tp_memref = new MemRefT_int64_1d();
    tp_memref->data_aligned = trainParams;
    tp_memref->data_allocated = trainParams;
    tp_memref->sizes[0] = 1;
    tp_memref->strides[0] = 0;

    __quantum__rt__initialize();

    QUBIT *q = __quantum__rt__qubit_allocate();

    __quantum__rt__toggle_recorder(/* activate_cm */ true);

    __quantum__qis__RX(-M_PI / 7, q);

    CplxT_double matrix_data[4] = {{1.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}, {0.0, 0.0}};

    MemRefT_CplxT_double_2d *h_matrix = new MemRefT_CplxT_double_2d;
    h_matrix->data_allocated = matrix_data;
    h_matrix->data_aligned = matrix_data;
    h_matrix->offset = 0;
    h_matrix->sizes[0] = 2;
    h_matrix->sizes[1] = 2;
    h_matrix->strides[0] = 1;

    auto obs_h = __quantum__qis__HermitianObs(h_matrix, 1, q);

    __quantum__qis__Expval(obs_h);

    __quantum__qis__Gradient_params(tp_memref, 1, result_tp);

    __quantum__qis__Gradient(1, result);

    __quantum__rt__toggle_recorder(/* activate_cm */ false);

    const double expected{0.2169418696};
    CHECK(expected == Approx(result_tp->data_aligned[0]));
    CHECK(expected == Approx(result->data_aligned[0]));

    delete result_tp;
    delete result;
    delete tp_memref;
    delete h_matrix;

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__Gradient_params and __quantum__qis__Gradient "
          "Op=RY, Obs=X",
          "[Gradient]")
{
    const std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};

    int64_t trainParams[1] = {0};
    MemRefT_double_1d *result = new MemRefT_double_1d();
    MemRefT_double_1d *result_tp = new MemRefT_double_1d();
    MemRefT_int64_1d *tp_memref = new MemRefT_int64_1d();
    tp_memref->data_aligned = trainParams;
    tp_memref->data_allocated = trainParams;
    tp_memref->sizes[0] = 1;
    tp_memref->strides[0] = 0;

    for (const auto &p : param) {
        // Test qis__Gradient:

        __quantum__rt__initialize();

        QUBIT *q = __quantum__rt__qubit_allocate();

        __quantum__rt__toggle_recorder(/* activate_cm */ true);

        __quantum__qis__RY(p, q);

        auto obs_idx_0 = __quantum__qis__NamedObs(ObsId::PauliX, q);

        __quantum__qis__Expval(obs_idx_0);

        __quantum__qis__Gradient_params(tp_memref, 1, result_tp);

        __quantum__qis__Gradient(1, result);

        __quantum__rt__toggle_recorder(/* activate_cm */ false);

        CHECK(cos(p) == Approx(result_tp->data_aligned[0]).margin(1e-5));
        CHECK(cos(p) == Approx(result->data_aligned[0]).margin(1e-5));

        __quantum__rt__finalize();
    }

    delete result_tp;
    delete result;
    delete tp_memref;
}

TEST_CASE("Test __quantum__qis__Gradient_params Op=[Hadamard,RZ,RY,RZ,S,T,ParamShift], "
          "Obs=[X]",
          "[Gradient]")
{
    const std::vector<double> param{0.3, 0.7, 0.4};
    const std::vector<double> expected{
        -0.1496908292,
        0.5703010745,
        -0.0008403297,
    };

    // Test qis__Gradient:
    int64_t trainParams[3] = {0, 1, 2};
    MemRefT_double_1d *result = new MemRefT_double_1d();
    MemRefT_double_1d *result_tp = new MemRefT_double_1d();
    MemRefT_int64_1d *tp_memref = new MemRefT_int64_1d();
    tp_memref->data_aligned = trainParams;
    tp_memref->data_allocated = trainParams;
    tp_memref->sizes[0] = 3;
    tp_memref->strides[0] = 0;

    __quantum__rt__initialize();

    QUBIT *q0 = __quantum__rt__qubit_allocate();
    __quantum__rt__qubit_allocate();

    __quantum__rt__toggle_recorder(/* activate_cm */ true);

    __quantum__qis__Hadamard(q0);
    __quantum__qis__RZ(param[0], q0);
    __quantum__qis__RY(param[1], q0);
    __quantum__qis__RZ(param[2], q0);
    __quantum__qis__S(q0);
    __quantum__qis__T(q0);

    auto obs_idx_0 = __quantum__qis__NamedObs(ObsId::PauliX, q0);

    __quantum__qis__Expval(obs_idx_0);

    __quantum__qis__Gradient_params(tp_memref, 1, result_tp);

    __quantum__qis__Gradient(1, result);

    __quantum__rt__toggle_recorder(/* activate_cm */ false);

    CHECK(expected[0] == Approx(result_tp->data_aligned[0]).margin(1e-5));
    CHECK(expected[1] == Approx(result_tp->data_aligned[1]).margin(1e-5));
    CHECK(expected[2] == Approx(result_tp->data_aligned[2]).margin(1e-5));

    CHECK(expected[0] == Approx(result->data_aligned[0]).margin(1e-5));
    CHECK(expected[1] == Approx(result->data_aligned[1]).margin(1e-5));
    CHECK(expected[2] == Approx(result->data_aligned[2]).margin(1e-5));

    delete result;
    delete result_tp;
    delete tp_memref;

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__Gradient Op=[RX,CY], Obs=[Z,Z]", "[Gradient]")
{
    // Test qis__Gradient:
    int64_t trainParams[1] = {0};
    MemRefT_double_1d *result0 = new MemRefT_double_1d();
    MemRefT_double_1d *result1 = new MemRefT_double_1d();
    MemRefT_int64_1d *tp_memref = new MemRefT_int64_1d();
    tp_memref->data_aligned = trainParams;
    tp_memref->data_allocated = trainParams;
    tp_memref->sizes[0] = 1;
    tp_memref->strides[0] = 0;

    __quantum__rt__initialize();

    QUBIT *q0 = __quantum__rt__qubit_allocate();
    QUBIT *q1 = __quantum__rt__qubit_allocate();

    __quantum__rt__toggle_recorder(/* activate_cm */ true);

    __quantum__qis__RX(-M_PI / 7, q0);
    __quantum__qis__CY(q0, q1);

    auto obs_idx_0 = __quantum__qis__NamedObs(ObsId::PauliZ, q0);
    auto obs_idx_1 = __quantum__qis__NamedObs(ObsId::PauliZ, q1);

    __quantum__qis__Expval(obs_idx_0);
    __quantum__qis__Expval(obs_idx_1);

    __quantum__qis__Gradient(2, result0, result1);

    __quantum__rt__toggle_recorder(/* activate_cm */ false);

    const std::vector<double> expected{-sin(-M_PI / 7), 0.4338837391};
    CHECK(expected[0] == Approx(result0->data_aligned[0]).margin(1e-5));
    CHECK(expected[1] == Approx(result1->data_aligned[0]).margin(1e-5));

    delete result0;
    delete result1;
    delete tp_memref;

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__Gradient_params Op=[RX,RX,RX,CZ], Obs=[Z,Z,Z]", "[Gradient]")
{
    const std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    const std::vector<double> expected{-sin(param[0]), -sin(param[1]), -sin(param[2])};

    // Test qis__Gradient:
    int64_t trainParams[3] = {0, 1, 2};
    MemRefT_double_1d *result0 = new MemRefT_double_1d();
    MemRefT_double_1d *result1 = new MemRefT_double_1d();
    MemRefT_double_1d *result2 = new MemRefT_double_1d();
    MemRefT_int64_1d *tp_memref = new MemRefT_int64_1d();
    tp_memref->data_aligned = trainParams;
    tp_memref->data_allocated = trainParams;
    tp_memref->sizes[0] = 3;
    tp_memref->strides[0] = 0;

    __quantum__rt__initialize();

    QUBIT *q0 = __quantum__rt__qubit_allocate();
    QUBIT *q1 = __quantum__rt__qubit_allocate();
    QUBIT *q2 = __quantum__rt__qubit_allocate();

    __quantum__rt__toggle_recorder(/* activate_cm */ true);

    __quantum__qis__RX(param[0], q0);
    __quantum__qis__RX(param[1], q1);
    __quantum__qis__RX(param[2], q2);
    __quantum__qis__CZ(q0, q2);

    auto obs_idx_0 = __quantum__qis__NamedObs(ObsId::PauliZ, q0);
    auto obs_idx_1 = __quantum__qis__NamedObs(ObsId::PauliZ, q1);
    auto obs_idx_2 = __quantum__qis__NamedObs(ObsId::PauliZ, q2);

    __quantum__qis__Expval(obs_idx_0);
    __quantum__qis__Expval(obs_idx_1);
    __quantum__qis__Expval(obs_idx_2);

    __quantum__qis__Gradient_params(tp_memref, 3, result0, result1, result2);

    __quantum__rt__toggle_recorder(/* activate_cm */ false);

    CHECK(expected[0] == Approx(result0->data_aligned[0]).margin(1e-5));
    CHECK(expected[1] == Approx(result1->data_aligned[1]).margin(1e-5));
    CHECK(expected[2] == Approx(result2->data_aligned[2]).margin(1e-5));

    delete result0;
    delete result1;
    delete result2;
    delete tp_memref;

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__Gradient and __quantum__qis__Gradient_params "
          "Op=Mixed, Obs=X@X@X",
          "[Gradient]")
{
    const std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    const std::vector<double> expected{0.0,         -0.6742144271, 0.275139672,
                                       0.275139672, -0.0129093062, 0.3238461564};

    // Test qis__Gradient:
    int64_t trainParams[6] = {0, 1, 2, 3, 4, 5};
    MemRefT_double_1d *result = new MemRefT_double_1d();
    MemRefT_double_1d *result_tp = new MemRefT_double_1d();
    MemRefT_int64_1d *tp_memref = new MemRefT_int64_1d();
    tp_memref->data_aligned = trainParams;
    tp_memref->data_allocated = trainParams;
    tp_memref->sizes[0] = 6;
    tp_memref->strides[0] = 0;

    __quantum__rt__initialize();

    QUBIT *q0 = __quantum__rt__qubit_allocate();
    QUBIT *q1 = __quantum__rt__qubit_allocate();
    QUBIT *q2 = __quantum__rt__qubit_allocate();

    __quantum__rt__toggle_recorder(/* activate_cm */ true);

    __quantum__qis__RZ(param[0], q0);
    __quantum__qis__RY(param[1], q0);
    __quantum__qis__RZ(param[2], q0);
    __quantum__qis__CNOT(q0, q1);
    __quantum__qis__CNOT(q1, q2);
    __quantum__qis__RZ(param[0], q1);
    __quantum__qis__RY(param[1], q1);
    __quantum__qis__RZ(param[2], q1);

    auto obs_idx_0 = __quantum__qis__NamedObs(ObsId::PauliX, q0);
    auto obs_idx_1 = __quantum__qis__NamedObs(ObsId::PauliX, q1);
    auto obs_idx_2 = __quantum__qis__NamedObs(ObsId::PauliX, q2);
    auto obs_tp = __quantum__qis__TensorObs(3, obs_idx_0, obs_idx_1, obs_idx_2);

    __quantum__qis__Expval(obs_tp);

    __quantum__qis__Gradient_params(tp_memref, 1, result_tp);

    __quantum__qis__Gradient(1, result);

    __quantum__rt__toggle_recorder(/* activate_cm */ false);

    CHECK(expected[0] == Approx(result_tp->data_aligned[0]).margin(1e-5));
    CHECK(expected[1] == Approx(result_tp->data_aligned[1]).margin(1e-5));
    CHECK(expected[2] == Approx(result_tp->data_aligned[2]).margin(1e-5));
    CHECK(expected[3] == Approx(result_tp->data_aligned[3]).margin(1e-5));
    CHECK(expected[4] == Approx(result_tp->data_aligned[4]).margin(1e-5));
    CHECK(expected[5] == Approx(result_tp->data_aligned[5]).margin(1e-5));

    CHECK(expected[0] == Approx(result->data_aligned[0]).margin(1e-5));
    CHECK(expected[1] == Approx(result->data_aligned[1]).margin(1e-5));
    CHECK(expected[2] == Approx(result->data_aligned[2]).margin(1e-5));
    CHECK(expected[3] == Approx(result->data_aligned[3]).margin(1e-5));
    CHECK(expected[4] == Approx(result->data_aligned[4]).margin(1e-5));
    CHECK(expected[5] == Approx(result->data_aligned[5]).margin(1e-5));

    delete result;
    delete result_tp;
    delete tp_memref;

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__Gradient and __quantum__qis__Gradient_params "
          "Op=Mixed, Obs=Z@Z@Z",
          "[Gradient]")
{
    const std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    const std::vector<double> expected{0.0, -0.414506421, 0.0, 0.0, -0.4643270456, 0.0210905264};

    // Test qis__Gradient:
    int64_t trainParams[6] = {0, 1, 2, 3, 4, 5};
    MemRefT_double_1d *result = new MemRefT_double_1d();
    MemRefT_double_1d *result_tp = new MemRefT_double_1d();
    MemRefT_int64_1d *tp_memref = new MemRefT_int64_1d();
    tp_memref->data_aligned = trainParams;
    tp_memref->data_allocated = trainParams;
    tp_memref->sizes[0] = 6;
    tp_memref->strides[0] = 0;

    __quantum__rt__initialize();

    QUBIT *q0 = __quantum__rt__qubit_allocate();
    QUBIT *q1 = __quantum__rt__qubit_allocate();
    QUBIT *q2 = __quantum__rt__qubit_allocate();

    __quantum__rt__toggle_recorder(/* activate_cm */ true);

    __quantum__qis__RZ(param[0], q0);
    __quantum__qis__RY(param[1], q0);
    __quantum__qis__RZ(param[2], q0);
    __quantum__qis__CNOT(q0, q1);
    __quantum__qis__CNOT(q1, q2);
    __quantum__qis__RZ(param[0], q1);
    __quantum__qis__RY(param[1], q1);
    __quantum__qis__RZ(param[2], q1);
    __quantum__qis__CRY(param[0], q0, q1);
    __quantum__qis__CRZ(param[1], q0, q2);

    auto obs_idx_0 = __quantum__qis__NamedObs(ObsId::PauliZ, q0);
    auto obs_idx_1 = __quantum__qis__NamedObs(ObsId::PauliZ, q1);
    auto obs_idx_2 = __quantum__qis__NamedObs(ObsId::PauliZ, q2);
    auto obs_tensor = __quantum__qis__TensorObs(3, obs_idx_0, obs_idx_1, obs_idx_2);

    __quantum__qis__Expval(obs_tensor);

    __quantum__qis__Gradient_params(tp_memref, 1, result_tp);

    __quantum__qis__Gradient(1, result);

    __quantum__rt__toggle_recorder(/* activate_cm */ false);

    CHECK(expected[0] == Approx(result_tp->data_aligned[0]).margin(1e-5));
    CHECK(expected[1] == Approx(result_tp->data_aligned[1]).margin(1e-5));
    CHECK(expected[2] == Approx(result_tp->data_aligned[2]).margin(1e-5));
    CHECK(expected[3] == Approx(result_tp->data_aligned[3]).margin(1e-5));
    CHECK(expected[4] == Approx(result_tp->data_aligned[4]).margin(1e-5));
    CHECK(expected[5] == Approx(result_tp->data_aligned[5]).margin(1e-5));

    CHECK(expected[0] == Approx(result->data_aligned[0]).margin(1e-5));
    CHECK(expected[1] == Approx(result->data_aligned[1]).margin(1e-5));
    CHECK(expected[2] == Approx(result->data_aligned[2]).margin(1e-5));
    CHECK(expected[3] == Approx(result->data_aligned[3]).margin(1e-5));
    CHECK(expected[4] == Approx(result->data_aligned[4]).margin(1e-5));
    CHECK(expected[5] == Approx(result->data_aligned[5]).margin(1e-5));

    delete result;
    delete result_tp;
    delete tp_memref;

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__Gradient and __quantum__qis__Gradient_params "
          "Op=Mixed, "
          "Obs=Hamiltonian([Z@Z, H], {0.2, 0.6})",
          "[Gradient]")
{
    const std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    const std::vector<double> expected{0.0, -0.2493761627, 0.0, 0.0, -0.1175570505, 0.0};

    // Test qis__Gradient:
    int64_t trainParams[6] = {0, 1, 2, 3, 4, 5};
    MemRefT_double_1d *result = new MemRefT_double_1d();
    MemRefT_double_1d *result_tp = new MemRefT_double_1d();
    MemRefT_int64_1d *tp_memref = new MemRefT_int64_1d();
    tp_memref->data_aligned = trainParams;
    tp_memref->data_allocated = trainParams;
    tp_memref->sizes[0] = 6;
    tp_memref->strides[0] = 0;

    __quantum__rt__initialize();

    QUBIT *q0 = __quantum__rt__qubit_allocate();
    QUBIT *q1 = __quantum__rt__qubit_allocate();
    QUBIT *q2 = __quantum__rt__qubit_allocate();

    __quantum__rt__toggle_recorder(/* activate_cm */ true);

    __quantum__qis__RZ(param[0], q0);
    __quantum__qis__RY(param[1], q0);
    __quantum__qis__RZ(param[2], q0);
    __quantum__qis__CNOT(q0, q1);
    __quantum__qis__CNOT(q1, q2);
    __quantum__qis__RZ(param[0], q1);
    __quantum__qis__RY(param[1], q1);
    __quantum__qis__RZ(param[2], q1);

    double coeffs_data[2] = {0.2, 0.6};
    MemRefT_double_1d *coeffs = new MemRefT_double_1d;
    coeffs->data_allocated = coeffs_data;
    coeffs->data_aligned = coeffs_data;
    coeffs->offset = 0;
    coeffs->sizes[0] = 2;
    coeffs->strides[0] = 1;

    auto obs_idx_0 = __quantum__qis__NamedObs(ObsId::PauliZ, q0);
    auto obs_idx_1 = __quantum__qis__NamedObs(ObsId::PauliZ, q1);
    auto obs_idx_2 = __quantum__qis__NamedObs(ObsId::Hadamard, q2);
    auto obs_tensor = __quantum__qis__TensorObs(2, obs_idx_0, obs_idx_1);
    auto obs_hamiltonian = __quantum__qis__HamiltonianObs(coeffs, 2, obs_tensor, obs_idx_2);

    __quantum__qis__Expval(obs_hamiltonian);

    __quantum__qis__Gradient_params(tp_memref, 1, result_tp);

    __quantum__qis__Gradient(1, result);

    __quantum__rt__toggle_recorder(/* activate_cm */ false);

    CHECK(expected[0] == Approx(result_tp->data_aligned[0]).margin(1e-5));
    CHECK(expected[1] == Approx(result_tp->data_aligned[1]).margin(1e-5));
    CHECK(expected[2] == Approx(result_tp->data_aligned[2]).margin(1e-5));
    CHECK(expected[3] == Approx(result_tp->data_aligned[3]).margin(1e-5));
    CHECK(expected[4] == Approx(result_tp->data_aligned[4]).margin(1e-5));
    CHECK(expected[5] == Approx(result_tp->data_aligned[5]).margin(1e-5));

    CHECK(expected[0] == Approx(result->data_aligned[0]).margin(1e-5));
    CHECK(expected[1] == Approx(result->data_aligned[1]).margin(1e-5));
    CHECK(expected[2] == Approx(result->data_aligned[2]).margin(1e-5));
    CHECK(expected[3] == Approx(result->data_aligned[3]).margin(1e-5));
    CHECK(expected[4] == Approx(result->data_aligned[4]).margin(1e-5));
    CHECK(expected[5] == Approx(result->data_aligned[5]).margin(1e-5));

    delete coeffs;
    delete result;
    delete result_tp;
    delete tp_memref;

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__Gradient and __quantum__qis__Gradient_params "
          "for a nontrivial qubits map in the qubit-manager Op=RX, Obs=Hermitian",
          "[Gradient]")
{
    // Test qis__Gradient:
    int64_t trainParams[1] = {0};
    MemRefT_double_1d *result = new MemRefT_double_1d();
    MemRefT_double_1d *result_tp = new MemRefT_double_1d();
    MemRefT_int64_1d *tp_memref = new MemRefT_int64_1d();
    tp_memref->data_aligned = trainParams;
    tp_memref->data_allocated = trainParams;
    tp_memref->sizes[0] = 1;
    tp_memref->strides[0] = 0;

    __quantum__rt__initialize();

    QirArray *qubit_arr = __quantum__rt__qubit_allocate_array(2);

    __quantum__rt__qubit_release_array(qubit_arr);

    QUBIT *q = __quantum__rt__qubit_allocate();

    QirString *qstr = __quantum__rt__qubit_to_string(q);

    QirString *expected_str = __quantum__rt__int_to_string(2);

    CHECK(__quantum__rt__string_equal(qstr, expected_str));

    __quantum__rt__toggle_recorder(/* activate_cm */ true);

    __quantum__qis__RX(-M_PI / 7, q);

    CplxT_double matrix_data[4] = {{1.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}, {0.0, 0.0}};

    MemRefT_CplxT_double_2d *h_matrix = new MemRefT_CplxT_double_2d;
    h_matrix->data_allocated = matrix_data;
    h_matrix->data_aligned = matrix_data;
    h_matrix->offset = 0;
    h_matrix->sizes[0] = 2;
    h_matrix->sizes[1] = 2;
    h_matrix->strides[0] = 1;

    auto obs_h = __quantum__qis__HermitianObs(h_matrix, 1, q);

    __quantum__qis__Expval(obs_h);

    __quantum__qis__Gradient_params(tp_memref, 1, result_tp);

    __quantum__qis__Gradient(1, result);

    __quantum__rt__toggle_recorder(/* activate_cm */ false);

    const double expected{0.2169418696};
    CHECK(expected == Approx(result_tp->data_aligned[0]));
    CHECK(expected == Approx(result->data_aligned[0]));

    delete result_tp;
    delete result;
    delete tp_memref;
    delete h_matrix;

    __quantum__rt__finalize();
}
