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

// RUN: quantum-opt %s --annotate-function -split-input-file -verify-diagnostics

// -----

func.func private @foo(%arg0: f64) -> f64

%0 = arith.constant 1.2 : f64

gradient.grad "fd" @foo(%0) : (f64) -> f64
gradient.grad "auto" @foo(%0) : (f64) -> f64

// expected-error@+1 {{got invalid differentiation method: none}}
gradient.grad "none" @foo(%0) : (f64) -> f64

// -----

// expected-error@+1 {{invalid function name specified: @foo}}
gradient.grad "fd" @foo() : () -> ()

// -----

func.func private @integer(%arg0: i64) -> f64

%i0 = arith.constant 3 : i64

gradient.grad "fd" @integer(%i0) : (i64) -> f64
// expected-error@-1 {{invalid numeric base type: callee operand at position 0 must be floating point to be differentiable}}
// expected-error@-2 {{invalid result type: grad result at position 0 must be 'i64' but got 'f64'}}

// -----

func.func private @scalar_scalar(%arg0: f64) -> f64

%f0 = arith.constant 0.7 : f64
%f1 = arith.constant 1.2 : f16

gradient.grad "fd" @scalar_scalar(%f0) : (f64) -> f64

// expected-error@+1 {{incorrect number of operands for callee, expected 1 but got 2}}
gradient.grad "fd" @scalar_scalar(%f0, %f1) : (f64, f16) -> f64

// -----

func.func private @scalar_scalar(%arg0: f64) -> f64

%f1 = arith.constant 1.2 : f16

// expected-error@+1 {{operand type mismatch: expected operand type 'f64', but provided 'f16' for operand number 0}}
gradient.grad "fd" @scalar_scalar(%f1) : (f16) -> f64

// -----

func.func private @scalar_tensor(%arg0: f16) -> tensor<2x3xf64>

%f1 = arith.constant 1.2 : f16

gradient.grad "fd" @scalar_tensor(%f1) : (f16) -> tensor<2x3xf64>
// expected-error@-1 {{invalid result type: grad result at position 0 must be 'tensor<2x3xf16>' but got 'tensor<2x3xf64>'}}

// -----

func.func private @tensor_scalar(%arg0: tensor<3xf64>) -> f32

%f0 = arith.constant 0.7 : f64
%t0 = tensor.from_elements %f0, %f0, %f0 : tensor<3xf64>

gradient.grad "fd" @tensor_scalar(%t0) : (tensor<3xf64>) -> tensor<3xf32>
// expected-error@-1 {{invalid result type: grad result at position 0 must be 'tensor<3xf64>' but got 'tensor<3xf32>}}

gradient.grad "fd" @tensor_scalar(%t0) : (tensor<3xf64>) -> f32

// -----

func.func private @tensor_tensor(%arg0: tensor<3xf64>) -> tensor<2xf64>

%f0 = arith.constant 0.7 : f64
%t0 = tensor.from_elements %f0, %f0, %f0 : tensor<3xf64>

gradient.grad "fd" @tensor_tensor(%t0) : (tensor<3xf64>) -> tensor<3x2xf64>
// expected-error@-1 {{op invalid result type: grad result at position 0 must be 'tensor<2x3xf64>' but got 'tensor<3x2xf64>}}


gradient.grad "fd" @tensor_tensor(%t0) : (tensor<3xf64>) -> tensor<2x3xf64>

// -----

func.func private @multiple_args_res(%arg0: f64, %arg1: tensor<3xf64>) -> (f64, tensor<2x3xf64>)

%f0 = arith.constant 0.7 : f64
%t0 = tensor.from_elements %f0, %f0, %f0 : tensor<3xf64>

gradient.grad "fd" @multiple_args_res(%f0, %t0)
    : (f64, tensor<3xf64>) -> (f64, tensor<2x3xf64>)

gradient.grad "fd" @multiple_args_res(%f0, %t0) {diffArgIndices = dense<[0, 1]> : tensor<2xindex>}
    : (f64, tensor<3xf64>) -> (f64, tensor<3xf64>, tensor<2x3xf64>, tensor<2x3x3xf64>)

// expected-error@+1 {{incorrect number of results in the gradient of the callee, expected 4 results but got 2}}
gradient.grad "fd" @multiple_args_res(%f0, %t0) {diffArgIndices = dense<[0, 1]> : tensor<2xindex>}
    : (f64, tensor<3xf64>) -> (f64, tensor<2x3xf64>)

// -----

func.func private @foo(%arg0: tensor<f64>)

%f0 = arith.constant 0.7 : f64
%t0 = tensor.from_elements %f0 : tensor<f64>
%m0 = memref.alloc() : memref<f64>

// expected-error@+1 {{cannot have both tensor results and memref output arguments}}
%grad = gradient.backprop @foo(%t0) grad_out(%m0 : memref<f64>) cotangents(%t0: tensor<f64>) : (tensor<f64>) -> tensor<f64>

// -----

func.func private @foo(%arg0: tensor<f64>)

%f0 = arith.constant 0.7 : f64
%t0 = tensor.from_elements %f0 : tensor<f64>
%m0 = memref.alloc() : memref<f64>

// expected-error@+1 {{cannot have callee result buffers before bufferization}}
%grad = gradient.backprop @foo(%t0) callee_out(%m0 : memref<f64>) cotangents(%t0: tensor<f64>) : (tensor<f64>) -> tensor<f64>

// -----

func.func private @foo(%arg0: memref<f64>)

%m0 = memref.alloc() : memref<f64>

// expected-error@+1 {{need as many callee result buffers as there are cotangents, expected 1 but got 0}}
gradient.backprop @foo(%m0) grad_out(%m0 : memref<f64>) cotangents(%m0: memref<f64>) : (memref<f64>) -> ()

// -----

func.func private @multiple_args(%arg0: tensor<f64>, %arg1: tensor<f64>)

%f0 = arith.constant 0.7 : f64
%t0 = tensor.from_elements %f0 : tensor<f64>

// expected-error@+1 {{number of gradient results did not match number of differentiable arguments, expected 1 but got 2}}
%grad:2 = gradient.backprop @multiple_args(%t0, %t0) cotangents(%t0: tensor<f64>) {diffArgIndices = dense<0> : tensor<1xindex>}: (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)

