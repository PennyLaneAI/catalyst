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

// RUN: quantum-opt %s --split-input-file --verify-diagnostics

//===----------------------------------------------------------------------===//
// Smoke tests: Can we use each qubit type?
//===----------------------------------------------------------------------===//

func.func @test_default_type(%0 : !quantum.bit, %1 : !quantum.bit) {
    %2 = quantum.custom ""() %0 : !quantum.bit
    %3, %4 = quantum.custom ""() %1, %2 : !quantum.bit, !quantum.bit
    return
}

// -----

func.func @test_abstract_type(%0 : !quantum.bit<abstract>, %1 : !quantum.bit<abstract>) {
    %2 = quantum.custom ""() %0 : !quantum.bit<abstract>
    %3, %4 = quantum.custom ""() %1, %2 : !quantum.bit<abstract>, !quantum.bit<abstract>
    return
}

// -----

func.func @test_logical_type(%0 : !quantum.bit<logical>, %1 : !quantum.bit<logical>) {
    %2 = quantum.custom ""() %0 : !quantum.bit<logical>
    %3, %4 = quantum.custom ""() %1, %2 : !quantum.bit<logical>, !quantum.bit<logical>
    return
}

// -----

func.func @test_qec_type(%0 : !quantum.bit<qec>, %1 : !quantum.bit<qec>) {
    %2 = quantum.custom ""() %0 : !quantum.bit<qec>
    %3, %4 = quantum.custom ""() %1, %2 : !quantum.bit<qec>, !quantum.bit<qec>
    return
}

// -----

func.func @test_physical_type(%0 : !quantum.bit<physical>, %1 : !quantum.bit<physical>) {
    %2 = quantum.custom ""() %0 : !quantum.bit<physical>
    %3, %4 = quantum.custom ""() %1, %2 : !quantum.bit<physical>, !quantum.bit<physical>
    return
}

// -----

func.func @test_ctrl_default(%0 : !quantum.bit, %1 : !quantum.bit) {
    %true = llvm.mlir.constant (1 : i1) : i1
    %2, %3 = quantum.custom ""() %0 ctrls (%1) ctrlvals (%true) : !quantum.bit ctrls !quantum.bit
    return
}

// -----

func.func @test_ctrl_abstract(%0 : !quantum.bit<abstract>, %1 : !quantum.bit<abstract>) {
    %true = llvm.mlir.constant (1 : i1) : i1
    %2, %3 = quantum.custom ""() %0 ctrls (%1) ctrlvals (%true) : !quantum.bit<abstract> ctrls !quantum.bit<abstract>
    return
}

// -----

func.func @test_ctrl_logical(%0 : !quantum.bit<logical>, %1 : !quantum.bit<logical>) {
    %true = llvm.mlir.constant (1 : i1) : i1
    %2, %3 = quantum.custom ""() %0 ctrls (%1) ctrlvals (%true) : !quantum.bit<logical> ctrls !quantum.bit<logical>
    return
}

// -----

func.func @test_ctrl_qec(%0 : !quantum.bit<qec>, %1 : !quantum.bit<qec>) {
    %true = llvm.mlir.constant (1 : i1) : i1
    %2, %3 = quantum.custom ""() %0 ctrls (%1) ctrlvals (%true) : !quantum.bit<qec> ctrls !quantum.bit<qec>
    return
}

// -----

func.func @test_ctrl_physical(%0 : !quantum.bit<physical>, %1 : !quantum.bit<physical>) {
    %true = llvm.mlir.constant (1 : i1) : i1
    %2, %3 = quantum.custom ""() %0 ctrls (%1) ctrlvals (%true) : !quantum.bit<physical> ctrls !quantum.bit<physical>
    return
}

// -----


//===----------------------------------------------------------------------===//
// Verifier tests: Does incorrect usage result in expected errors?
//===----------------------------------------------------------------------===//

// COM: In the tests below, note that we can safely "convert" between `!quantum.bit`
// COM: and `!quantum.bit<abstract>` since they are equivalent.

func.func @test_mix_types_abs_log(%0 : !quantum.bit) {
    %1 = quantum.custom ""() %0 : !quantum.bit<abstract>  // expected-note {{prior use here}}
    // expected-error @below {{expects different type than prior uses}}
    %2 = quantum.custom ""() %1 : !quantum.bit<logical>
    return
}

// -----

func.func @test_mix_types_abs_qec(%0 : !quantum.bit) {
    %1 = quantum.custom ""() %0 : !quantum.bit<abstract>  // expected-note {{prior use here}}
    // expected-error @below {{expects different type than prior uses}}
    %2 = quantum.custom ""() %1 : !quantum.bit<qec>
    return
}

// -----

func.func @test_mix_types_abs_phy(%0 : !quantum.bit) {
    %1 = quantum.custom ""() %0 : !quantum.bit<abstract>  // expected-note {{prior use here}}
    // expected-error @below {{expects different type than prior uses}}
    %2 = quantum.custom ""() %1 : !quantum.bit<physical>
    return
}

// -----

func.func @test_mix_types_log_qec(%0 : !quantum.bit<logical>) {
    %1 = quantum.custom ""() %0 : !quantum.bit<logical>  // expected-note {{prior use here}}
    // expected-error @below {{expects different type than prior uses}}
    %2 = quantum.custom ""() %1 : !quantum.bit<qec>
    return
}

// -----

func.func @test_mix_types_log_phy(%0 : !quantum.bit<logical>) {
    %1 = quantum.custom ""() %0 : !quantum.bit<logical>  // expected-note {{prior use here}}
    // expected-error @below {{expects different type than prior uses}}
    %2 = quantum.custom ""() %1 : !quantum.bit<physical>
    return
}

// -----

func.func @test_mix_types_qec_phy(%0 : !quantum.bit<qec>) {
    %1 = quantum.custom ""() %0 : !quantum.bit<qec>  // expected-note {{prior use here}}
    // expected-error @below {{expects different type than prior uses}}
    %2 = quantum.custom ""() %1 : !quantum.bit<physical>
    return
}

// -----

func.func @test_2q_mix_input_types_default_log(%0 : !quantum.bit, %1 : !quantum.bit<logical>) {
    // expected-error @below {{requires all qubit operands and results to have the same type}}
    %2, %3 = quantum.custom ""() %0, %1 : !quantum.bit, !quantum.bit<logical>
    return
}

// -----

func.func @test_2q_mix_input_types_abs_log(%0 : !quantum.bit<abstract>, %1 : !quantum.bit<logical>) {
    // expected-error @below {{requires all qubit operands and results to have the same type}}
    %2, %3 = quantum.custom ""() %0, %1 : !quantum.bit<abstract>, !quantum.bit<logical>
    return
}

// -----

func.func @test_2q_mix_input_types_abs_phy(%0 : !quantum.bit<abstract>, %1 : !quantum.bit<physical>) {
    // expected-error @below {{requires all qubit operands and results to have the same type}}
    %2, %3 = quantum.custom ""() %0, %1 : !quantum.bit<abstract>, !quantum.bit<physical>
    return
}

// -----

func.func @test_2q_mix_input_ctrl_types_abs_log(%0 : !quantum.bit<abstract>, %1 : !quantum.bit<logical>) {
    %true = llvm.mlir.constant (1 : i1) : i1
    // expected-error @below {{requires all qubit operands and results to have the same type}}
    %2, %3 = quantum.custom ""() %0 ctrls (%1) ctrlvals (%true) : !quantum.bit<abstract> ctrls !quantum.bit<logical>
    return
}

// -----

func.func @test_2q_mix_input_ctrl_types_abs_phy(%0 : !quantum.bit<abstract>, %1 : !quantum.bit<physical>) {
    %true = llvm.mlir.constant (1 : i1) : i1
    // expected-error @below {{requires all qubit operands and results to have the same type}}
    %2, %3 = quantum.custom ""() %0 ctrls (%1) ctrlvals (%true) : !quantum.bit<abstract> ctrls !quantum.bit<physical>
    return
}
