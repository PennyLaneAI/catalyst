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


func.func @__builtin_h_to_rz_ry(%q0 : !quantum.bit) -> !quantum.bit attributes {target_gate="Hadamard"} {
    %piby2 = arith.constant 1.57 : f64
    %pi = arith.constant 3.14 : f64
    %negpiby2 = arith.constant -3.14 : f64
    %q1 = quantum.custom "RZ"(%pi) %q0 : !quantum.bit
    %q2 = quantum.custom "RY"(%piby2) %q1 : !quantum.bit
    quantum.gphase(%negpiby2)
    return %q2 : !quantum.bit
}

func.func @__builtin_h_to_rz_rx(%q0 : !quantum.bit) -> !quantum.bit attributes {target_gate="Hadamard"} {
    %piby2 = arith.constant 1.57 : f64
    %negpiby2 = arith.constant -3.14 : f64
    %q1 = quantum.custom "RZ"(%piby2) %q0 : !quantum.bit
    %q2 = quantum.custom "RX"(%piby2) %q1 : !quantum.bit
    %q3 = quantum.custom "RZ"(%piby2) %q2 : !quantum.bit
    quantum.gphase(%negpiby2)
    return %q3 : !quantum.bit
}

func.func @__builtin_x_to_rx(%q0 : !quantum.bit) -> !quantum.bit attributes {target_gate="PauliX"} {
    %pi = arith.constant 3.14 : f64
    %negpiby2 = arith.constant -3.14 : f64
    %q1 = quantum.custom "RX"(%pi) %q0 : !quantum.bit
    quantum.gphase(%negpiby2)
    return %q1 : !quantum.bit
}

func.func @__builtin_rx_to_rz_ry(%angle : f64, %q0 : !quantum.bit) -> !quantum.bit attributes {target_gate="RX"} {
    %piby2 = arith.constant 1.57 : f64
    %negpiby2 = arith.constant -3.14 : f64
    %q1 = quantum.custom "RZ"(%piby2) %q0 : !quantum.bit
    %q2 = quantum.custom "RY"(%angle) %q1 : !quantum.bit
    %q3 = quantum.custom "RZ"(%negpiby2) %q2 : !quantum.bit
    return %q3 : !quantum.bit
}
