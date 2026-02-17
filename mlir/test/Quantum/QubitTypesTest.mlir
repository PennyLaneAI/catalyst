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

func.func @test_pbc_type(%0 : !quantum.bit<pbc>, %1 : !quantum.bit<pbc>) {
    %2 = quantum.custom ""() %0 : !quantum.bit<pbc>
    %3, %4 = quantum.custom ""() %1, %2 : !quantum.bit<pbc>, !quantum.bit<pbc>
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

func.func @test_ctrl_pbc(%0 : !quantum.bit<pbc>, %1 : !quantum.bit<pbc>) {
    %true = llvm.mlir.constant (1 : i1) : i1
    %2, %3 = quantum.custom ""() %0 ctrls (%1) ctrlvals (%true) : !quantum.bit<pbc> ctrls !quantum.bit<pbc>
    return
}

// -----

func.func @test_ctrl_physical(%0 : !quantum.bit<physical>, %1 : !quantum.bit<physical>) {
    %true = llvm.mlir.constant (1 : i1) : i1
    %2, %3 = quantum.custom ""() %0 ctrls (%1) ctrlvals (%true) : !quantum.bit<physical> ctrls !quantum.bit<physical>
    return
}

// -----

func.func @test_alloc_qb_default() {
    %0 = quantum.alloc_qb : !quantum.bit
    return
}

// -----

func.func @test_alloc_qb_abs() {
    %0 = quantum.alloc_qb : !quantum.bit<abstract>
    return
}

// -----

func.func @test_alloc_qb_log() {
    %0 = quantum.alloc_qb : !quantum.bit<logical>
    return
}

// -----

func.func @test_alloc_qb_phy() {
    %0 = quantum.alloc_qb : !quantum.bit<physical>
    return
}

// -----

func.func @test_dealloc_qb_default(%0 : !quantum.bit) {
    quantum.dealloc_qb %0 : !quantum.bit
    return
}

// -----

func.func @test_dealloc_qb_abs(%0 : !quantum.bit<abstract>) {
    quantum.dealloc_qb %0 : !quantum.bit<abstract>
    return
}

// -----

func.func @test_dealloc_qb_log(%0 : !quantum.bit<logical>) {
    quantum.dealloc_qb %0 : !quantum.bit<logical>
    return
}

// -----

func.func @test_dealloc_qb_phy(%0 : !quantum.bit<physical>) {
    quantum.dealloc_qb %0 : !quantum.bit<physical>
    return
}

// -----

func.func @test_abs_null_type(%0 : !quantum.bit<abstract, null>) {
    %1 = quantum.custom ""() %0 : !quantum.bit<abstract, null>
    return
}

// -----

func.func @test_log_null_type(%0 : !quantum.bit<logical, null>) {
    %1 = quantum.custom ""() %0 : !quantum.bit<logical, null>
    return
}

// -----

func.func @test_pbc_data_type(%0 : !quantum.bit<pbc, data>) {
    %1 = quantum.custom ""() %0 : !quantum.bit<pbc, data>
    return
}

// -----

func.func @test_pbc_xcheck_type(%0 : !quantum.bit<pbc, xcheck>) {
    %1 = quantum.custom ""() %0 : !quantum.bit<pbc, xcheck>
    return
}

// -----

func.func @test_pbc_zcheck_type(%0 : !quantum.bit<pbc, zcheck>) {
    %1 = quantum.custom ""() %0 : !quantum.bit<pbc, zcheck>
    return
}

// -----

func.func @test_phy_data_type(%0 : !quantum.bit<physical, data>) {
    %1 = quantum.custom ""() %0 : !quantum.bit<physical, data>
    return
}

// -----

func.func @test_phy_xcheck_type(%0 : !quantum.bit<physical, xcheck>) {
    %1 = quantum.custom ""() %0 : !quantum.bit<physical, xcheck>
    return
}

// -----

func.func @test_phy_zcheck_type(%0 : !quantum.bit<physical, zcheck>) {
    %1 = quantum.custom ""() %0 : !quantum.bit<physical, zcheck>
    return
}

// -----

//===----------------------------------------------------------------------===//
// Smoke tests for register types
//===----------------------------------------------------------------------===//

func.func @test_alloc_reg_default() {
    %0 = quantum.alloc(1) : !quantum.reg
    return
}

// -----

func.func @test_alloc_reg_abs() {
    %0 = quantum.alloc(1) : !quantum.reg<abstract>
    return
}

// -----

func.func @test_alloc_reg_log() {
    %0 = quantum.alloc(1) : !quantum.reg<logical>
    return
}

// -----

func.func @test_alloc_reg_pbc() {
    %0 = quantum.alloc(1) : !quantum.reg<pbc>
    return
}

// -----

func.func @test_dealloc_reg_default(%0 : !quantum.reg) {
    quantum.dealloc %0 : !quantum.reg
    return
}

// -----

func.func @test_dealloc_reg_abs(%0 : !quantum.reg<abstract>) {
    quantum.dealloc %0 : !quantum.reg<abstract>
    return
}

// -----

func.func @test_dealloc_reg_log(%0 : !quantum.reg<logical>) {
    quantum.dealloc %0 : !quantum.reg<logical>
    return
}

// -----

func.func @test_dealloc_reg_pbc(%0 : !quantum.reg<pbc>) {
    quantum.dealloc %0 : !quantum.reg<pbc>
    return
}

// -----

func.func @test_dealloc_reg_phy(%0 : !quantum.reg<physical>) {
    quantum.dealloc %0 : !quantum.reg<physical>
    return
}

// -----

func.func @test_extract_reg_default(%0 : !quantum.reg) {
    %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
    return
}

// -----

func.func @test_extract_reg_abs(%0 : !quantum.reg<abstract>) {
    %1 = quantum.extract %0[0] : !quantum.reg<abstract> -> !quantum.bit<abstract>
    return
}

// -----

func.func @test_extract_reg_log(%0 : !quantum.reg<logical>) {
    %1 = quantum.extract %0[0] : !quantum.reg<logical> -> !quantum.bit<logical>
    return
}

// -----

func.func @test_extract_reg_pbc(%0 : !quantum.reg<pbc>) {
    %1 = quantum.extract %0[0] : !quantum.reg<pbc> -> !quantum.bit<pbc>
    return
}

// -----

func.func @test_extract_reg_pbc_to_pbc_data_bit(%0 : !quantum.reg<pbc>) {
    // COM: It's permitted to extract a pbc qubit with a role other than
    // COM: `null` from a pbc register
    %1 = quantum.extract %0[0] : !quantum.reg<pbc> -> !quantum.bit<pbc, data>
    return
}

// -----

func.func @test_extract_reg_phy(%0 : !quantum.reg<physical>) {
    %1 = quantum.extract %0[0] : !quantum.reg<physical> -> !quantum.bit<physical>
    return
}

// -----

func.func @test_extract_reg_phy_to_phy_data_bit(%0 : !quantum.reg<physical>) {
    // COM: It's permitted to extract a physical qubit with a role other than
    // COM: `null` from a physical register
    %1 = quantum.extract %0[0] : !quantum.reg<physical> -> !quantum.bit<physical, data>
    return
}

// -----

func.func @test_insert_reg_default(%0 : !quantum.reg, %q : !quantum.bit) {
    %1 = quantum.insert %0[0], %q : !quantum.reg, !quantum.bit
    return
}

// -----

func.func @test_insert_reg_abs(%0 : !quantum.reg<abstract>, %q : !quantum.bit<abstract>) {
    %1 = quantum.insert %0[0], %q : !quantum.reg<abstract>, !quantum.bit<abstract>
    return
}

// -----

func.func @test_insert_reg_log(%0 : !quantum.reg<logical>, %q : !quantum.bit<logical>) {
    %1 = quantum.insert %0[0], %q : !quantum.reg<logical>, !quantum.bit<logical>
    return
}

// -----

func.func @test_insert_reg_pbc(%0 : !quantum.reg<pbc>, %q : !quantum.bit<pbc>) {
    %1 = quantum.insert %0[0], %q : !quantum.reg<pbc>, !quantum.bit<pbc>
    return
}

// -----

func.func @test_insert_reg_phy(%0 : !quantum.reg<physical>, %q : !quantum.bit<physical>) {
    %1 = quantum.insert %0[0], %q : !quantum.reg<physical>, !quantum.bit<physical>
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

func.func @test_mix_types_abs_pbc(%0 : !quantum.bit) {
    %1 = quantum.custom ""() %0 : !quantum.bit<abstract>  // expected-note {{prior use here}}
    // expected-error @below {{expects different type than prior uses}}
    %2 = quantum.custom ""() %1 : !quantum.bit<pbc>
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

func.func @test_mix_types_log_pbc(%0 : !quantum.bit<logical>) {
    %1 = quantum.custom ""() %0 : !quantum.bit<logical>  // expected-note {{prior use here}}
    // expected-error @below {{expects different type than prior uses}}
    %2 = quantum.custom ""() %1 : !quantum.bit<pbc>
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

func.func @test_mix_types_pbc_phy(%0 : !quantum.bit<pbc>) {
    %1 = quantum.custom ""() %0 : !quantum.bit<pbc>  // expected-note {{prior use here}}
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

// -----

func.func @test_abs_data_type(%0 : !quantum.bit<abstract, data>) {
    // expected-error @above {{qubit role 'data' is only permitted for pbc and physical qubits}}
    return
}

// -----

func.func @test_abs_xcheck_type(%0 : !quantum.bit<abstract, xcheck>) {
    // expected-error @above {{qubit role 'xcheck' is only permitted for pbc and physical qubits}}
    return
}

// -----

func.func @test_abs_zcheck_type(%0 : !quantum.bit<abstract, zcheck>) {
    // expected-error @above {{qubit role 'zcheck' is only permitted for pbc and physical qubits}}
    return
}

// -----

func.func @test_abs_data_type(%0 : !quantum.bit<logical, data>) {
    // expected-error @above {{qubit role 'data' is only permitted for pbc and physical qubits}}
    return
}

// -----

func.func @test_abs_xcheck_type(%0 : !quantum.bit<logical, xcheck>) {
    // expected-error @above {{qubit role 'xcheck' is only permitted for pbc and physical qubits}}
    return
}

// -----

func.func @test_abs_zcheck_type(%0 : !quantum.bit<logical, zcheck>) {
    // expected-error @above {{qubit role 'zcheck' is only permitted for pbc and physical qubits}}
    return
}

// -----

//===----------------------------------------------------------------------===//
// Verifier tests for register types
//===----------------------------------------------------------------------===//

func.func @test_extract_default_reg_to_log_bit(%0 : !quantum.reg) {
    // expected-error @below {{type mismatch}}
    %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit<logical>
    return
}

// -----

func.func @test_extract_default_reg_to_pbc_bit(%0 : !quantum.reg) {
    // expected-error @below {{type mismatch}}
    %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit<pbc>
    return
}

// -----

func.func @test_extract_default_reg_to_phy_bit(%0 : !quantum.reg) {
    // expected-error @below {{type mismatch}}
    %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit<physical>
    return
}

// -----

func.func @test_extract_log_reg_to_pbc_bit(%0 : !quantum.reg<logical>) {
    // expected-error @below {{type mismatch}}
    %1 = quantum.extract %0[0] : !quantum.reg<logical> -> !quantum.bit<pbc>
    return
}

// -----

func.func @test_extract_pbc_reg_to_phy_bit(%0 : !quantum.reg<pbc>) {
    // expected-error @below {{type mismatch}}
    %1 = quantum.extract %0[0] : !quantum.reg<pbc> -> !quantum.bit<physical>
    return
}

// -----

func.func @test_insert_log_bit_to_default_reg(%0 : !quantum.reg, %q : !quantum.bit<logical>) {
    // expected-error @below {{type mismatch}}
    %1 = quantum.insert %0[0], %q : !quantum.reg, !quantum.bit<logical>
    return
}

// -----

func.func @test_insert_pbc_bit_to_default_reg(%0 : !quantum.reg, %q : !quantum.bit<pbc>) {
    // expected-error @below {{type mismatch}}
    %1 = quantum.insert %0[0], %q : !quantum.reg, !quantum.bit<pbc>
    return
}
// -----

func.func @test_insert_phy_bit_to_default_reg(%0 : !quantum.reg, %q : !quantum.bit<physical>) {
    // expected-error @below {{type mismatch}}
    %1 = quantum.insert %0[0], %q : !quantum.reg, !quantum.bit<physical>
    return
}

// -----

func.func @test_insert_pbc_bit_to_log_reg(%0 : !quantum.reg<logical>, %q : !quantum.bit<pbc>) {
    // expected-error @below {{type mismatch}}
    %1 = quantum.insert %0[0], %q : !quantum.reg<logical>, !quantum.bit<pbc>
    return
}

// -----

func.func @test_insert_phy_bit_to_log_reg(%0 : !quantum.reg<pbc>, %q : !quantum.bit<physical>) {
    // expected-error @below {{type mismatch}}
    %1 = quantum.insert %0[0], %q : !quantum.reg<pbc>, !quantum.bit<physical>
    return
}

// -----

func.func @test_insert_mix_types_default_log(%0 : !quantum.reg, %q : !quantum.bit) {
    // expected-note @above {{prior use here}}
    // expected-error @below {{expects different type than prior uses}}
    %1 = quantum.insert %0[0], %q : !quantum.reg<logical>, !quantum.bit
    return
}

// -----

func.func @test_insert_mix_types_log_phy(%0 : !quantum.reg<logical>, %q : !quantum.bit<logical>) {
    // expected-note @above {{prior use here}}
    // expected-error @below {{expects different type than prior uses}}
    %1 = quantum.insert %0[0], %q : !quantum.reg<physical>, !quantum.bit<logical>
    return
}
