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

// RUN: quantum-opt --scalarize-tensor-extracts --canonicalize --split-input-file %s | FileCheck %s

// The extract_slice + collapse_shape + extract chain produced when tracing
// indexes a 1-D runtime tensor collapses to a single extract.

// CHECK-LABEL: @slice_collapse_extract
// CHECK-SAME:    (%[[ARG:.+]]: tensor<15xf64>)
// CHECK:         %[[C14:.+]] = arith.constant 14 : index
// CHECK:         %[[RES:.+]] = tensor.extract %[[ARG]][%[[C14]]] : tensor<15xf64>
// CHECK-NOT:     tensor.extract_slice
// CHECK-NOT:     tensor.collapse_shape
// CHECK:         return %[[RES]]
func.func @slice_collapse_extract(%arg0: tensor<15xf64>) -> f64 {
  %s = tensor.extract_slice %arg0[14] [1] [1] : tensor<15xf64> to tensor<1xf64>
  %c = tensor.collapse_shape %s [] : tensor<1xf64> into tensor<f64>
  %e = tensor.extract %c[] : tensor<f64>
  return %e : f64
}

// -----

// Extracting one element of an elementwise linalg.generic inlines the scalar
// payload; the generic and its tensor.empty become dead.

// CHECK-LABEL: @extract_of_generic
// CHECK-SAME:    (%[[A:.+]]: tensor<2x2xf64>, %[[B:.+]]: tensor<2x2xf64>)
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[EA:.+]] = tensor.extract %[[A]][%[[C0]], %[[C1]]]
// CHECK-DAG:     %[[EB:.+]] = tensor.extract %[[B]][%[[C0]], %[[C1]]]
// CHECK:         %[[RES:.+]] = arith.mulf %[[EA]], %[[EB]] : f64
// CHECK-NOT:     linalg.generic
// CHECK:         return %[[RES]]
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @extract_of_generic(%arg0: tensor<2x2xf64>, %arg1: tensor<2x2xf64>) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %empty = tensor.empty() : tensor<2x2xf64>
  %prod = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<2x2xf64>, tensor<2x2xf64>) outs(%empty : tensor<2x2xf64>) {
  ^bb0(%in0: f64, %in1: f64, %out: f64):
    %m = arith.mulf %in0, %in1 : f64
    linalg.yield %m : f64
  } -> tensor<2x2xf64>
  %res = tensor.extract %prod[%c0, %c1] : tensor<2x2xf64>
  return %res : f64
}

// -----

// Broadcast (rank-0 to 2x2) generics fold to an extract of the rank-0 source.

// CHECK-LABEL: @extract_of_broadcast
// CHECK-SAME:    (%[[A:.+]]: tensor<f64>)
// CHECK:         %[[E:.+]] = tensor.extract %[[A]][] : tensor<f64>
// CHECK-NOT:     linalg.generic
// CHECK:         return %[[E]]
#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @extract_of_broadcast(%arg0: tensor<f64>) -> f64 {
  %c1 = arith.constant 1 : index
  %empty = tensor.empty() : tensor<2x2xf64>
  %bcast = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<f64>) outs(%empty : tensor<2x2xf64>) {
  ^bb0(%in: f64, %out: f64):
    linalg.yield %in : f64
  } -> tensor<2x2xf64>
  %res = tensor.extract %bcast[%c1, %c1] : tensor<2x2xf64>
  return %res : f64
}

// -----

// Reductions must not be scalarized: the payload reads the accumulator.

// CHECK-LABEL: @reduction_untouched
// CHECK:         linalg.generic
// CHECK:         tensor.extract
#map_in = affine_map<(d0) -> (d0)>
#map_out = affine_map<(d0) -> ()>
func.func @reduction_untouched(%arg0: tensor<8xf64>) -> f64 {
  %cst = arith.constant 0.0 : f64
  %empty = tensor.empty() : tensor<f64>
  %fill = linalg.fill ins(%cst : f64) outs(%empty : tensor<f64>) -> tensor<f64>
  %sum = linalg.generic {indexing_maps = [#map_in, #map_out], iterator_types = ["reduction"]}
      ins(%arg0 : tensor<8xf64>) outs(%fill : tensor<f64>) {
  ^bb0(%in: f64, %acc: f64):
    %a = arith.addf %in, %acc : f64
    linalg.yield %a : f64
  } -> tensor<f64>
  %res = tensor.extract %sum[] : tensor<f64>
  return %res : f64
}

// -----

// Large tensors must not be scalarized (payload cloning is capped).

// CHECK-LABEL: @large_tensor_untouched
// CHECK:         linalg.generic
// CHECK:         tensor.extract
#map2 = affine_map<(d0) -> (d0)>
func.func @large_tensor_untouched(%arg0: tensor<100xf64>) -> f64 {
  %c5 = arith.constant 5 : index
  %empty = tensor.empty() : tensor<100xf64>
  %sq = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]}
      ins(%arg0 : tensor<100xf64>) outs(%empty : tensor<100xf64>) {
  ^bb0(%in: f64, %out: f64):
    %m = arith.mulf %in, %in : f64
    linalg.yield %m : f64
  } -> tensor<100xf64>
  %res = tensor.extract %sq[%c5] : tensor<100xf64>
  return %res : f64
}

// -----

// Rank-reducing extract_slice: index arithmetic offset + i * stride.

// CHECK-LABEL: @strided_slice_extract
// CHECK-SAME:    (%[[ARG:.+]]: tensor<4x6xf64>, %[[I:.+]]: index)
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[SCALED:.+]] = arith.muli %[[I]], %[[C2]]
// CHECK-DAG:     %[[COL:.+]] = arith.addi %[[SCALED]], %[[C1]]
// CHECK:         %[[RES:.+]] = tensor.extract %[[ARG]][%[[C2]], %[[COL]]] : tensor<4x6xf64>
// CHECK:         return %[[RES]]
func.func @strided_slice_extract(%arg0: tensor<4x6xf64>, %i: index) -> f64 {
  %s = tensor.extract_slice %arg0[2, 1] [1, 3] [1, 2] : tensor<4x6xf64> to tensor<3xf64>
  %e = tensor.extract %s[%i] : tensor<3xf64>
  return %e : f64
}
