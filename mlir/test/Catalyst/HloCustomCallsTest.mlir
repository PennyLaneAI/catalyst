// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --hlo-custom-call-lowering --split-input-file %s | FileCheck %s

func.func @custom_call(%arg0: tensor<3x3xf64>) -> tensor<3x3xf64> {
    // CHECK: %cst = arith.constant dense<1> : tensor<i32>
    // CHECK: %cst_0 = arith.constant dense<3> : tensor<i32>
    // CHECK: %0 = catalyst.custom_call fn("lapack_dgesdd_ffi") (%cst, %cst_0, %cst_0, %arg0) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<3x3xf64>) -> tensor<3x3xf64>
    // CHECK: return %0 : tensor<3x3xf64>
    %0 = mhlo.custom_call @lapack_dgesdd_ffi(%arg0) {api_version = 2 : i32, backend_config = "", operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#mhlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<3x3xf64>) -> tensor<3x3xf64>
    return %0 : tensor<3x3xf64>
}
