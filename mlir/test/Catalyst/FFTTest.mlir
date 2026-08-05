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

// RUN: quantum-opt %s --fft-lowering --split-input-file --verify-diagnostics | FileCheck %s

// 1D complex-to-complex forward FFT on a non-power-of-two length.
// The forward twiddle angle factor is -2*pi/6.
// CHECK-LABEL: func.func @fft_1d
func.func @fft_1d(%arg0: tensor<6xcomplex<f64>>) -> tensor<6xcomplex<f64>> {
    // CHECK-NOT: stablehlo.fft
    // CHECK-DAG: arith.constant -1.0471975511965976 : f64
    // CHECK-DAG: [[INIT:%.+]] = tensor.empty() : tensor<6xcomplex<f64>>
    // CHECK-DAG: [[ZERO:%.+]] = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
    // CHECK:     [[FILLED:%.+]] = linalg.fill ins([[ZERO]] : complex<f64>) outs([[INIT]] : tensor<6xcomplex<f64>>)
    // CHECK:     [[RES:%.+]] = linalg.generic
    // CHECK-SAME: indexing_maps = [#map, #map1]
    // CHECK-SAME: iterator_types = ["parallel", "reduction"]
    // CHECK-SAME: ins(%arg0 : tensor<6xcomplex<f64>>) outs([[FILLED]] : tensor<6xcomplex<f64>>)
    // CHECK-DAG:    linalg.index 0
    // CHECK-DAG:    linalg.index 1
    // CHECK-DAG:    arith.muli
    // CHECK-DAG:    arith.remui
    // CHECK-DAG:    math.cos
    // CHECK-DAG:    math.sin
    // CHECK:        complex.create
    // CHECK:     return [[RES]]
    %0 = stablehlo.fft %arg0, type = FFT, length = [6] : (tensor<6xcomplex<f64>>) -> tensor<6xcomplex<f64>>
    return %0 : tensor<6xcomplex<f64>>
}

// -----

// 1D inverse FFT with positive twiddle angle +2*pi/6 and the 1/6
// normalization folded into the twiddles.
// CHECK-LABEL: func.func @ifft_1d
func.func @ifft_1d(%arg0: tensor<6xcomplex<f64>>) -> tensor<6xcomplex<f64>> {
    // CHECK-NOT: stablehlo.fft
    // CHECK-DAG: arith.constant 1.0471975511965976 : f64
    // CHECK-DAG: arith.constant 0.16666666666666666 : f64
    // CHECK:     linalg.generic
    // CHECK:     complex.create
    %0 = stablehlo.fft %arg0, type = IFFT, length = [6] : (tensor<6xcomplex<f64>>) -> tensor<6xcomplex<f64>>
    return %0 : tensor<6xcomplex<f64>>
}

// -----

// 1D real-to-complex FFT where odd length 9 stores floor(9/2)+1 = 5 bins.
// CHECK-LABEL: func.func @rfft_1d
func.func @rfft_1d(%arg0: tensor<9xf64>) -> tensor<5xcomplex<f64>> {
    // CHECK-NOT: stablehlo.fft
    // CHECK:     tensor.empty() : tensor<5xcomplex<f64>>
    // CHECK:     [[RES:%.+]] = linalg.generic
    // CHECK-SAME: ins(%arg0 : tensor<9xf64>)
    // CHECK-SAME: outs({{.*}} : tensor<5xcomplex<f64>>)
    // CHECK:        complex.create
    // CHECK:     return [[RES]]
    %0 = stablehlo.fft %arg0, type = RFFT, length = [9] : (tensor<9xf64>) -> tensor<5xcomplex<f64>>
    return %0 : tensor<5xcomplex<f64>>
}

// -----

// 1D complex-to-real inverse FFT reading 5 stored bins to reconstruct
// length 8. Interior bins are double weighted due to Hermitian symmetry
// while the DC bin and the Nyquist bin count once.
// CHECK-LABEL: func.func @irfft_1d
func.func @irfft_1d(%arg0: tensor<5xcomplex<f64>>) -> tensor<8xf64> {
    // CHECK-NOT: stablehlo.fft
    // CHECK-DAG: arith.constant 1.250000e-01 : f64
    // CHECK-DAG: arith.constant 4 : index
    // CHECK:     tensor.empty() : tensor<8xf64>
    // CHECK:     [[RES:%.+]] = linalg.generic
    // CHECK-SAME: ins(%arg0 : tensor<5xcomplex<f64>>)
    // CHECK-SAME: outs({{.*}} : tensor<8xf64>)
    // CHECK-DAG:    arith.cmpi eq
    // CHECK-DAG:    arith.select
    // CHECK:     return [[RES]]
    %0 = stablehlo.fft %arg0, type = IRFFT, length = [8] : (tensor<5xcomplex<f64>>) -> tensor<8xf64>
    return %0 : tensor<8xf64>
}

// -----

// 2D transform. The separable lowering emits one batched 1D stage per axis.
// CHECK-LABEL: func.func @fft_2d
func.func @fft_2d(%arg0: tensor<4x6xcomplex<f64>>) -> tensor<4x6xcomplex<f64>> {
    // CHECK-NOT: stablehlo.fft
    // CHECK:     [[S1:%.+]] = linalg.generic
    // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
    // CHECK:     [[S2:%.+]] = linalg.generic
    // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
    // CHECK-SAME: ins([[S1]] : tensor<4x6xcomplex<f64>>)
    // CHECK:     return [[S2]]
    %0 = stablehlo.fft %arg0, type = FFT, length = [4, 6] : (tensor<4x6xcomplex<f64>>) -> tensor<4x6xcomplex<f64>>
    return %0 : tensor<4x6xcomplex<f64>>
}

// -----

// Batched transform with a dynamic batch dimension.
// CHECK-LABEL: func.func @fft_batched_dynamic
func.func @fft_batched_dynamic(%arg0: tensor<?x6xcomplex<f64>>) -> tensor<?x6xcomplex<f64>> {
    // CHECK-NOT: stablehlo.fft
    // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[DIM:%.+]] = tensor.dim %arg0, [[C0]]
    // CHECK:     tensor.empty([[DIM]]) : tensor<?x6xcomplex<f64>>
    // CHECK:     [[RES:%.+]] = linalg.generic
    // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
    // CHECK:     return [[RES]]
    %0 = stablehlo.fft %arg0, type = FFT, length = [6] : (tensor<?x6xcomplex<f64>>) -> tensor<?x6xcomplex<f64>>
    return %0 : tensor<?x6xcomplex<f64>>
}

// -----

// f32 transforms compute twiddles in f64 and truncate.
// CHECK-LABEL: func.func @fft_f32
func.func @fft_f32(%arg0: tensor<6xcomplex<f32>>) -> tensor<6xcomplex<f32>> {
    // CHECK-NOT: stablehlo.fft
    // CHECK:     linalg.generic
    // CHECK-DAG:    math.cos {{.*}} : f64
    // CHECK-DAG:    arith.truncf {{.*}} : f64 to f32
    // CHECK:        complex.create {{.*}} : complex<f32>
    %0 = stablehlo.fft %arg0, type = FFT, length = [6] : (tensor<6xcomplex<f32>>) -> tensor<6xcomplex<f32>>
    return %0 : tensor<6xcomplex<f32>>
}
