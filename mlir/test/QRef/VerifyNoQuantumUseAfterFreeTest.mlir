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

// RUN: quantum-opt --split-input-file --verify-diagnostics --verify-no-quantum-use-after-free %s


func.func @test_use_after_free() {
    %r = qref.alloc(3) : !qref.reg<3>
    %q = qref.get %r[0] : !qref.reg<3> -> !qref.bit
    qref.custom "PauliX"() %q : !qref.bit
    qref.dealloc %r : !qref.reg<3>

    // expected-error@+1 {{Detected use of a qubit after deallocation}}
    qref.custom "PauliX"() %q : !qref.bit
    return
}

// -----

func.func @test_use_after_free_single_qubit_alloc() {
    %q = qref.alloc_qb : !qref.bit
    qref.custom "PauliX"() %q : !qref.bit
    qref.dealloc_qb %q : !qref.bit

    // expected-error@+1 {{Detected use of a qubit after deallocation}}
    qref.custom "PauliX"() %q : !qref.bit
    return
}

// -----

func.func @test_use_after_free_from_arg(%r: !qref.reg<3>) {
    %q = qref.get %r[0] : !qref.reg<3> -> !qref.bit
    qref.custom "PauliX"() %q : !qref.bit
    qref.dealloc %r : !qref.reg<3>

    // expected-error@+1 {{Detected use of a qubit after deallocation}}
    qref.custom "PauliX"() %q : !qref.bit
    return
}

// -----

func.func @test_use_after_free_single_qubit_alloc_from_arg(%q: !qref.bit) {
    qref.custom "PauliX"() %q : !qref.bit
    qref.dealloc_qb %q : !qref.bit

    // expected-error@+1 {{Detected use of a qubit after deallocation}}
    qref.custom "PauliX"() %q : !qref.bit
    return
}

// -----

func.func @test_use_after_free_alias() {
    %r = qref.alloc(3) : !qref.reg<3>
    %q = qref.get %r[0] : !qref.reg<3> -> !qref.bit
    qref.custom "PauliX"() %q : !qref.bit
    qref.dealloc %r : !qref.reg<3>

    %q_alias = qref.get %r[0] : !qref.reg<3> -> !qref.bit
    // expected-error@+1 {{Detected use of a qubit after deallocation}}
    qref.custom "PauliX"() %q_alias : !qref.bit
    return
}

// -----

func.func @test_no_errors() {
    %r = qref.alloc(3) : !qref.reg<3>
    %q = qref.get %r[0] : !qref.reg<3> -> !qref.bit
    qref.custom "PauliX"() %q : !qref.bit
    qref.dealloc %r : !qref.reg<3>
    return
}
