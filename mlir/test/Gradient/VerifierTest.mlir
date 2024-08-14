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
%grad = gradient.backprop @foo(%t0) grad_out(%m0 : memref<f64>) cotangents(%t0: tensor<f64>) {resultSegmentSizes = array<i32: 0, 1>} : (tensor<f64>) -> tensor<f64>

// -----

func.func private @foo(%arg0: tensor<f64>)

%f0 = arith.constant 0.7 : f64
%t0 = tensor.from_elements %f0 : tensor<f64>
%m0 = memref.alloc() : memref<f64>

// expected-error@+1 {{cannot have callee result buffers before bufferization}}
%grad = gradient.backprop @foo(%t0) callee_out(%m0 : memref<f64>) cotangents(%t0: tensor<f64>) {resultSegmentSizes = array<i32: 0, 1>}: (tensor<f64>) -> tensor<f64>

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
%grad:2 = gradient.backprop @multiple_args(%t0, %t0) cotangents(%t0: tensor<f64>) {diffArgIndices = dense<0> : tensor<1xindex>, resultSegmentSizes = array<i32: 0, 2>}: (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)

// -----

func.func @measure(%arg0: f64) -> f64 {

    %c0 = arith.constant 0 : i64
    %0 = quantum.alloc(2) : !quantum.reg
    %1 = quantum.extract %0[%c0] : !quantum.reg -> !quantum.bit
    %res, %new_q = quantum.measure %1 : i1, !quantum.bit
    %c1 = arith.constant 1.0 : f64

    return %c1 : f64
}

%f0 = arith.constant 0.0 : f64
// expected-error@+1 {{An operation without a valid gradient was found}}
gradient.grad "auto" @measure(%f0) : (f64) -> (f64)

// -----

func.func @measure(%arg0: f64) -> f64 {

    %c0 = arith.constant 0 : i64
    %0 = quantum.alloc(2) : !quantum.reg
    %1 = quantum.extract %0[%c0] : !quantum.reg -> !quantum.bit
    %res, %new_q = quantum.measure %1 : i1, !quantum.bit
    %c1 = arith.constant 1.0 : f64

    return %c1 : f64
}

func.func @foo(%arg0 : f64) -> f64 {
    %0 = func.call @measure(%arg0) : (f64) -> f64
    return %0 : f64
}

%f0 = arith.constant 0.0 : f64
// expected-error@+1 {{An operation without a valid gradient was found}}
gradient.grad "auto" @foo(%f0) : (f64) -> (f64)

// Check that finite difference does not raise an error 
// -----

func.func @measure(%arg0: f64) -> f64 {

    %c0 = arith.constant 0 : i64
    %0 = quantum.alloc(2) : !quantum.reg
    %1 = quantum.extract %0[%c0] : !quantum.reg -> !quantum.bit
    %res, %new_q = quantum.measure %1 : i1, !quantum.bit
    %c1 = arith.constant 1.0 : f64

    return %c1 : f64
}

%f0 = arith.constant 0.0 : f64
gradient.grad "fd" @measure(%f0) : (f64) -> (f64)

// -----

func.func @measure(%arg0: f64) -> f64 {

    %c0 = arith.constant 0 : i64
    %0 = quantum.alloc(2) : !quantum.reg
    %1 = quantum.extract %0[%c0] : !quantum.reg -> !quantum.bit
    %res, %new_q = quantum.measure %1 : i1, !quantum.bit
    %c1 = arith.constant 1.0 : f64

    return %c1 : f64
}

func.func @foo(%arg0 : f64) -> f64 {
    %0 = func.call @measure(%arg0) : (f64) -> f64
    return %0 : f64
}

%f0 = arith.constant 0.0 : f64
gradient.grad "fd" @foo(%f0) : (f64) -> (f64)

// -----

func.func @measure(%arg0: tensor<2xf64>) -> tensor<2xf64> {

    %c0 = arith.constant 0 : i64
    %0 = quantum.alloc(2) : !quantum.reg
    %1 = quantum.extract %0[%c0] : !quantum.reg -> !quantum.bit
    %res, %new_q = quantum.measure %1 : i1, !quantum.bit
    %c1 = arith.constant dense<[1.0, 0.0]> : tensor<2xf64>

    return %c1 : tensor<2xf64>
}

%cst0 = arith.constant dense<[1.0, 0.0]> : tensor<2xf64>
%cst1 = arith.constant dense<[1.0, 0.0]> : tensor<2xf64>
// expected-error@+1 {{An operation without a valid gradient was found}}
gradient.jvp "auto" @measure(%cst0) tangents(%cst1) : (tensor<2xf64>, tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>)

// -----

func.func @measure(%arg0: tensor<2xf64>) -> tensor<2xf64> {

    %c0 = arith.constant 0 : i64
    %0 = quantum.alloc(2) : !quantum.reg
    %1 = quantum.extract %0[%c0] : !quantum.reg -> !quantum.bit
    %res, %new_q = quantum.measure %1 : i1, !quantum.bit
    %c1 = arith.constant dense<[1.0, 0.0]> : tensor<2xf64>

    return %c1 : tensor<2xf64>
}

%cst0 = arith.constant dense<[1.0, 0.0]> : tensor<2xf64>
%cst1 = arith.constant dense<[1.0, 0.0]> : tensor<2xf64>
gradient.jvp "fd" @measure(%cst0) tangents(%cst1) : (tensor<2xf64>, tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>)

// -----

func.func @foo(%arg0: tensor<f64>) -> (tensor<f64>, tensor<f64>) {

    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %0 = stablehlo.multiply %cst, %arg0 : tensor<f64>
    %1 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
    return %0, %1 : tensor<f64>, tensor<f64>

}

%cst0 = arith.constant dense<1.0> : tensor<f64>
%cst1 = arith.constant dense<1.0> : tensor<f64>
gradient.jvp "auto" @foo(%cst0) tangents(%cst1) : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>)

// -----

func.func @foo(%arg0: tensor<f64>) -> (tensor<f64>, tensor<f64>) {

    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %0 = stablehlo.multiply %cst, %arg0 : tensor<f64>
    %1 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
    return %0, %1 : tensor<f64>, tensor<f64>

}

%cst0 = arith.constant dense<1.0> : tensor<f64>
%cst1 = arith.constant dense<1> : tensor<i64>
// expected-error@+1 {{callee input type does not match the tangent type}}
gradient.jvp "auto" @foo(%cst0) tangents(%cst1) : (tensor<f64>, tensor<i64>) -> (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>)

// -----

func.func @measure(%arg0: tensor<2xf64>) -> tensor<2xf64> {

    %c0 = arith.constant 0 : i64
    %0 = quantum.alloc(2) : !quantum.reg
    %1 = quantum.extract %0[%c0] : !quantum.reg -> !quantum.bit
    %res, %new_q = quantum.measure %1 : i1, !quantum.bit
    %c1 = arith.constant dense<[1.0, 0.0]> : tensor<2xf64>

    return %c1 : tensor<2xf64>
}

%cst0 = arith.constant dense<[1.0, 0.0]> : tensor<2xf64>
%cst1 = arith.constant dense<[1.0, 0.0]> : tensor<2xf64>
// expected-error@+1 {{An operation without a valid gradient was found}}
gradient.vjp "auto" @measure(%cst0) cotangents(%cst1) {resultSegmentSizes = array<i32: 1, 1>} : (tensor<2xf64>, tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>)

// -----

func.func @measure(%arg0: tensor<2xf64>) -> tensor<2xf64> {

    %c0 = arith.constant 0 : i64
    %0 = quantum.alloc(2) : !quantum.reg
    %1 = quantum.extract %0[%c0] : !quantum.reg -> !quantum.bit
    %res, %new_q = quantum.measure %1 : i1, !quantum.bit
    %c1 = arith.constant dense<[1.0, 0.0]> : tensor<2xf64>

    return %c1 : tensor<2xf64>
}

%cst0 = arith.constant dense<[1.0, 0.0]> : tensor<2xf64>
%cst1 = arith.constant dense<[1.0, 0.0]> : tensor<2xf64>
gradient.vjp "fd" @measure(%cst0) cotangents(%cst1) {resultSegmentSizes = array<i32: 1, 1>} : (tensor<2xf64>, tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>)

// -----

func.func @foo(%arg0: tensor<f64>) -> (tensor<f64>, tensor<f64>) {

    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %0 = stablehlo.multiply %cst, %arg0 : tensor<f64>
    %1 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
    return %0, %1 : tensor<f64>, tensor<f64>

}

%cst0 = arith.constant dense<1.0> : tensor<f64>
%cst1 = arith.constant dense<1.0> : tensor<f64>
%cst2 = arith.constant dense<1.0> : tensor<f64>
gradient.vjp "auto" @foo(%cst0) cotangents(%cst1, %cst2) {resultSegmentSizes = array<i32: 2, 1>} : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>)

// -----

func.func @foo(%arg0: tensor<f64>) -> (tensor<f64>, tensor<f64>) {

    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %0 = stablehlo.multiply %cst, %arg0 : tensor<f64>
    %1 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
    return %0, %1 : tensor<f64>, tensor<f64>

}

%cst0 = arith.constant dense<1.0> : tensor<f64>
%cst1 = arith.constant dense<1> : tensor<i64>
%cst2 = arith.constant dense<1> : tensor<i64>
// expected-error@+1 {{callee result type does not match the cotangent type}}
gradient.vjp "auto" @foo(%cst0) cotangents(%cst1, %cst2) {resultSegmentSizes = array<i32: 2, 1>} : (tensor<f64>, tensor<i64>, tensor<i64>) -> (tensor<f64>, tensor<f64>, tensor<f64>)

// -----

module @grad.wrapper {
  func.func public @jit_grad.wrapper(%arg0: tensor<2xf64>) -> tensor<2xf64> attributes {llvm.emit_c_interface} {
    %0 = gradient.grad "auto" @wrapper(%arg0) {diffArgIndices = dense<0> : tensor<1xi64>} : (tensor<2xf64>) -> tensor<2xf64>
    return %0 : tensor<2xf64>
  }
  func.func private @wrapper(%arg0: tensor<2xf64>) -> tensor<f64> attributes {llvm.linkage = #llvm.linkage<internal>} {
    %0 = catalyst.callback_call @callback_139793003716976(%arg0) : (tensor<2xf64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
  catalyst.callback @callback_139793003716976(tensor<2xf64>) -> tensor<f64> attributes {argc = 1 : i64, id = 139793003716976 : i64, llvm.linkage = #llvm.linkage<internal>, resc = 1 : i64}
  func.func private @fwd(%arg0: tensor<2xf64>) -> (tensor<f64>, tensor<i64>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %0 = catalyst.callback_call @callback_139793003716976(%arg0) : (tensor<2xf64>) -> tensor<f64>
    %1 = stablehlo.constant dense<1> : tensor<i64>
    return %0, %1 : tensor<f64>, tensor<i64>
  }
  func.func private @bwd(%arg0: tensor<i64>, %arg1: tensor<2xf64>) -> tensor<2xf64> attributes {llvm.linkage = #llvm.linkage<internal>} {
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<f64>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f64>) -> tensor<2xf64>
    %2 = stablehlo.multiply %1, %arg1 : tensor<2xf64>
    return %2 : tensor<2xf64>
  }
  gradient.forward @fwd.fwd(tensor<2xf64>) -> (tensor<f64>, tensor<i64>) attributes {argc = 1 : i64, implementation = @fwd, llvm.linkage = #llvm.linkage<internal>, resc = 1 : i64, tape = 1 : i64}
  gradient.reverse @bwd.rev(tensor<i64>, tensor<2xf64>) -> tensor<2xf64> attributes {argc = 1 : i64, implementation = @bwd, llvm.linkage = #llvm.linkage<internal>, resc = 1 : i64, tape = 1 : i64}
  gradient.custom_grad @callback_139793003716976 @fwd.fwd @bwd.rev {llvm.linkage = #llvm.linkage<internal>}
}

// -----

func.func @measure(%arg0: f64) -> f64 {

    %c0 = arith.constant 0 : i64
    %0 = quantum.alloc(2) : !quantum.reg
    %1 = quantum.extract %0[%c0] : !quantum.reg -> !quantum.bit
    %res, %new_q = quantum.measure %1 : i1, !quantum.bit
    %c1 = arith.constant 1.0 : f64

    return %c1 : f64
}

%f0 = arith.constant 0.0 : f64
// expected-error@+1 {{An operation without a valid gradient was found}}
gradient.value_and_grad "auto" @measure(%f0) : (f64) -> (f64, f64)

// -----

func.func @measure(%arg0: f64) -> f64 {

    %c0 = arith.constant 0 : i64
    %0 = quantum.alloc(2) : !quantum.reg
    %1 = quantum.extract %0[%c0] : !quantum.reg -> !quantum.bit
    %res, %new_q = quantum.measure %1 : i1, !quantum.bit
    %c1 = arith.constant 1.0 : f64

    return %c1 : f64
}

%f0 = arith.constant 0.0 : f64
gradient.value_and_grad "fd" @measure(%f0) : (f64) -> (f64, f64)

