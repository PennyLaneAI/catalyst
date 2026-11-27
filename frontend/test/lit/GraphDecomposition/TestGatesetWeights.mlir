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

// see https://github.com/llvm/llvm-project/blob/cc6b3b5a1d0fba10d92be143cd52268c62f2a2a9/mlir/test/IR/parser.mlir#L1078 for examples of F64 special values

// Invalid weights must fail with a diagnostic rather than throwing from the solver.
// RUN: not catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=TestGate=-1.0})' %s 2>&1 | FileCheck %s --check-prefix=NEGATIVE
// RUN: not catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set={TestGate=-1.0 : f64}})' %s 2>&1 | FileCheck %s --check-prefix=NEGATIVE-F64
// RUN: not catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=TestGate=-inf})' %s 2>&1 | FileCheck %s --check-prefix=NEGATIVE-INF
// this is the hex code for MLIR's negative infinity f64
// RUN: not catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set={TestGate = 0xFFF0000000000000 : f64}})' %s 2>&1 | FileCheck %s --check-prefix=NEGATIVE-INF-F64
// RUN: not catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=TestGate=nan})' %s 2>&1 | FileCheck %s --check-prefix=NAN
// signalling NaNs
// RUN: not catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set={TestGate = 0x7FF0000000000001 : f64}})' %s 2>&1 | FileCheck %s --check-prefix=NAN-F64
// RUN: not catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set={TestGate = 0x7FF8000000000000 : f64}})' %s 2>&1 | FileCheck %s --check-prefix=NAN-F64
// quiet NaNs
// RUN: not catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set={TestGate = 0x7FF0000001000000 : f64}})' %s 2>&1 | FileCheck %s --check-prefix=NAN-F64
// RUN: not catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set={TestGate = 0xFFF0000001000000 : f64}})' %s 2>&1 | FileCheck %s --check-prefix=NAN-F64
// RUN: not catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=TestGate=invalid})' %s 2>&1 | FileCheck %s --check-prefix=MALFORMED
// RUN: not catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=TestGate=})' %s 2>&1 | FileCheck %s --check-prefix=MISSING
// NEGATIVE: error: Invalid cost '-1.0' for gate 'TestGate' in the target gateset: expected a non-negative number (not NaN).
// NEGATIVE-F64: error: Invalid cost '-1.0 : f64' for gate 'TestGate' in the target gateset: expected a non-negative number (not NaN).
// NEGATIVE-INF: error: Invalid cost '-inf' for gate 'TestGate' in the target gateset: expected a non-negative number (not NaN).
// NEGATIVE-INF-F64: error: Invalid cost '0xFFF0000000000000 : f64' for gate 'TestGate' in the target gateset: expected a non-negative number (not NaN).
// NAN: error: Invalid cost 'nan' for gate 'TestGate' in the target gateset: expected a non-negative number (not NaN).
// NAN-F64: error: Invalid cost '{{.+}} : f64' for gate 'TestGate' in the target gateset: expected a non-negative number (not NaN).
// MALFORMED: error: Invalid cost 'invalid' for gate 'TestGate' in the target gateset: expected a non-negative number (not NaN).
// MISSING: error: Invalid cost '' for gate 'TestGate' in the target gateset: expected a non-negative number (not NaN).

// Zero, fractional, and positive infinite weights are permitted, as strings or as float attrs
// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=TestGate=0.0})' %s | FileCheck %s
// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set={TestGate=0.0 : f64}})' %s | FileCheck %s
// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=TestGate=0.5})' %s | FileCheck %s
// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set={TestGate=0.5 : f64}})' %s | FileCheck %s
// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=TestGate=inf})' %s | FileCheck %s
// This is the hex code for MLIR's infinite f64
// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set={TestGate = 0x7FF0000000000000 : f64}})' %s | FileCheck %s

// CHECK-LABEL: func.func @circuit
// CHECK: quantum.custom "TestGate"
func.func @circuit(%q: !quantum.bit) -> !quantum.bit {
  %out = quantum.custom "TestGate"() %q : !quantum.bit
  return %out : !quantum.bit
}
