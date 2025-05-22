// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt %s

func.func @foo(%q1 : !quantum.bit, %q2 : !quantum.bit) {
    qec.ppr ["X", "Z"] (4) %q1, %q2 : !quantum.bit, !quantum.bit
    func.return
}

func.func @boo(%q1 : !quantum.bit) {
    %0 = qec.prepare zero %q1 : !quantum.bit
    %1 = qec.prepare one %0 : !quantum.bit
    %2 = qec.prepare plus %1 : !quantum.bit
    %3 = qec.prepare minus %2 : !quantum.bit
    %4 = qec.prepare plus_i %3 : !quantum.bit
    %5 = qec.prepare minus_i %4 : !quantum.bit
    func.return
}

func.func @magic() {
    %0 = qec.fabricate magic : !quantum.bit
    %1 = qec.fabricate magic_conj : !quantum.bit
    %2 = qec.fabricate plus_i : !quantum.bit
    %3 = qec.fabricate minus_i : !quantum.bit
    func.return
}

func.func @bar(%q1 : !quantum.bit, %q2 : !quantum.bit) {
    %m_0, %0 = qec.ppm ["Z"] %q1 : !quantum.bit
    %m_1, %1 = qec.select.ppm (%m_0, ["X"], ["Z"]) %q2 : !quantum.bit
    func.return
}

func.func @baz(%q1 : !quantum.bit, %q2 : !quantum.bit) {
    %m_0, %0 = qec.ppm ["Z"] %q1 : !quantum.bit
    %1:2 = qec.ppr ["Y", "Y"] (4) %0, %q2 cond(%m_0) : !quantum.bit, !quantum.bit
    func.return
}
