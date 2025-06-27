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

#pragma once
#ifndef RUNTIMECAPI_H
#define RUNTIMECAPI_H

#include "Types.h"

#ifdef __cplusplus
extern "C" {
#endif

static long long total_memory_consumption = 0;
static long long memory_tracker = 0;
static long long peak_memory_consumption = 0;
static int pystdout = 0;

struct Tree {
    std::string _frame_name;
    long long _total_memory_consumption;
    long long _memory_tracker;
    long long _peak_memory_consumption;
    std::vector<Tree *> _children;
    Tree *_parent;
    Tree(std::string name)
        : _frame_name(name), _total_memory_consumption(0), _memory_tracker(0),
          _peak_memory_consumption(0), _parent(nullptr)
    {
    }

    ~Tree()
    {
        for (auto child : _children) {
            delete child;
        }
    }

    Tree *add_child(std::string child_name)
    {
        auto child = new Tree(child_name);
        child->_parent = this;
        _children.push_back(child);
        return child;
    }

    void show_stats()
    {
        fprintf(stdout, "%s total memory %lld\n", this->_frame_name.c_str(),
                this->_total_memory_consumption << 3);
        fprintf(stdout, "%s peak memory %lld\n", this->_frame_name.c_str(),
                this->_peak_memory_consumption << 3);
        for (auto child : _children) {
            child->show_stats();
        }
    }
};

struct Tree *call_tree = nullptr;

// Quantum Runtime Instructions
void __catalyst__rt__fail_cstr(const char *);
void __catalyst__rt__initialize(uint32_t *);
void __catalyst__rt__device_init(int8_t *, int8_t *, int8_t *, int64_t, bool);
void __catalyst__rt__device_release();
void __catalyst__rt__finalize(bool);
void __catalyst__rt__toggle_recorder(bool);
void __catalyst__rt__print_state();
void __catalyst__rt__print_tensor(OpaqueMemRefT *, bool);
void __catalyst__rt__print_string(char *);
void __catalyst__rt__assert_bool(bool, char *);
int64_t __catalyst__rt__array_get_size_1d(QirArray *);
int8_t *__catalyst__rt__array_get_element_ptr_1d(QirArray *, int64_t);

// Profiling functions
int64_t __catalyst__rt__profiler_get_timestamp();
void __catalyst__rt__profiler_record(const char *file_name, uint32_t line, uint32_t column,
                                     int64_t start_time, int64_t end_time);
void __catalyst__rt__profiler_print_stats();

QUBIT *__catalyst__rt__qubit_allocate();
QirArray *__catalyst__rt__qubit_allocate_array(int64_t);
void __catalyst__rt__qubit_release(QUBIT *);
void __catalyst__rt__qubit_release_array(QirArray *);

int64_t __catalyst__rt__num_qubits();

bool __catalyst__rt__result_equal(RESULT *, RESULT *);
RESULT *__catalyst__rt__result_get_one();
RESULT *__catalyst__rt__result_get_zero();

// Quantum Gate Set Instructions
void __catalyst__qis__SetState(MemRefT_CplxT_double_1d *, uint64_t, ...);
void __catalyst__qis__SetBasisState(MemRefT_int8_1d *, uint64_t, ...);
void __catalyst__qis__Identity(QUBIT *, const Modifiers *);
void __catalyst__qis__PauliX(QUBIT *, const Modifiers *);
void __catalyst__qis__PauliY(QUBIT *, const Modifiers *);
void __catalyst__qis__PauliZ(QUBIT *, const Modifiers *);
void __catalyst__qis__Hadamard(QUBIT *, const Modifiers *);
void __catalyst__qis__S(QUBIT *, const Modifiers *);
void __catalyst__qis__T(QUBIT *, const Modifiers *);
void __catalyst__qis__PhaseShift(double, QUBIT *, const Modifiers *);
void __catalyst__qis__RX(double, QUBIT *, const Modifiers *);
void __catalyst__qis__RY(double, QUBIT *, const Modifiers *);
void __catalyst__qis__RZ(double, QUBIT *, const Modifiers *);
void __catalyst__qis__Rot(double, double, double, QUBIT *, const Modifiers *);
void __catalyst__qis__CNOT(QUBIT *, QUBIT *, const Modifiers *);
void __catalyst__qis__CY(QUBIT *, QUBIT *, const Modifiers *);
void __catalyst__qis__CZ(QUBIT *, QUBIT *, const Modifiers *);
void __catalyst__qis__SWAP(QUBIT *, QUBIT *, const Modifiers *);
void __catalyst__qis__IsingXX(double, QUBIT *, QUBIT *, const Modifiers *);
void __catalyst__qis__IsingYY(double, QUBIT *, QUBIT *, const Modifiers *);
void __catalyst__qis__IsingXY(double, QUBIT *, QUBIT *, const Modifiers *);
void __catalyst__qis__IsingZZ(double, QUBIT *, QUBIT *, const Modifiers *);
void __catalyst__qis__ControlledPhaseShift(double, QUBIT *, QUBIT *, const Modifiers *);
void __catalyst__qis__CRX(double, QUBIT *, QUBIT *, const Modifiers *);
void __catalyst__qis__CRY(double, QUBIT *, QUBIT *, const Modifiers *);
void __catalyst__qis__CRZ(double, QUBIT *, QUBIT *, const Modifiers *);
void __catalyst__qis__MS(double, QUBIT *, QUBIT *, const Modifiers *);
void __catalyst__qis__CRot(double, double, double, QUBIT *, QUBIT *, const Modifiers *);
void __catalyst__qis__CSWAP(QUBIT *, QUBIT *, QUBIT *, const Modifiers *);
void __catalyst__qis__Toffoli(QUBIT *, QUBIT *, QUBIT *, const Modifiers *);
void __catalyst__qis__MultiRZ(double, const Modifiers *, int64_t, /*qubits*/...);
void __catalyst__qis__GlobalPhase(double, const Modifiers *);
void __catalyst__qis__ISWAP(QUBIT *, QUBIT *, const Modifiers *);
void __catalyst__qis__PSWAP(double, QUBIT *, QUBIT *, const Modifiers *);

// Struct pointer arguments for these instructions represent real arguments,
// as passing structs by value is too unreliable / compiler dependant.
void __catalyst__qis__QubitUnitary(MemRefT_CplxT_double_2d *, const Modifiers *, int64_t,
                                   /*qubits*/...);

ObsIdType __catalyst__qis__NamedObs(int64_t, QUBIT *);
ObsIdType __catalyst__qis__HermitianObs(MemRefT_CplxT_double_2d *, int64_t, /*qubits*/...);
ObsIdType __catalyst__qis__TensorObs(int64_t, /*obsKeys*/...);
ObsIdType __catalyst__qis__HamiltonianObs(MemRefT_double_1d *, int64_t, /*obsKeys*/...);

// Struct pointers arguments here represent return values.
RESULT *__catalyst__qis__Measure(QUBIT *, int32_t);
double __catalyst__qis__Expval(ObsIdType);
double __catalyst__qis__Variance(ObsIdType);
void __catalyst__qis__Probs(MemRefT_double_1d *, int64_t, /*qubits*/...);
void __catalyst__qis__Sample(MemRefT_double_2d *, int64_t, /*qubits*/...);
void __catalyst__qis__Counts(PairT_MemRefT_double_int64_1d *, int64_t, /*qubits*/...);
void __catalyst__qis__State(MemRefT_CplxT_double_1d *, int64_t, /*qubits*/...);
void __catalyst__qis__Gradient(int64_t, /*results*/...);
void __catalyst__qis__Gradient_params(MemRefT_int64_1d *, int64_t, /*results*/...);

// MBQC operations
RESULT *__catalyst__mbqc__measure_in_basis(QUBIT *, uint32_t, double, int32_t);

// Async runtime error
void __catalyst__host__rt__unrecoverable_error();

#ifdef __cplusplus
} // extern "C"
#endif

#endif
