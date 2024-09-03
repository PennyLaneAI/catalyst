// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt %s --split-multiple-tapes --split-input-file --verify-diagnostics | FileCheck %s


module @circuit_twotapes_module {
  func.func private @circuit_twotapes(%arg0: tensor<4xcomplex<f64>>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %cst = stablehlo.constant dense<4.000000e-01> : tensor<f64>
    %cst_0 = stablehlo.constant dense<8.000000e-01> : tensor<f64>
    quantum.device["librtd_lightning.so", "LightningSimulator", "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3:2 = quantum.set_state(%arg0) %1, %2 : (tensor<4xcomplex<f64>>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
    %extracted = tensor.extract %arg1[] : tensor<f64>
    %out_qubits = quantum.custom "RY"(%extracted) %3#0 {adjoint} : !quantum.bit
    %4 = quantum.namedobs %out_qubits[ PauliX] : !quantum.obs
    %5 = quantum.expval %4 : f64
    %from_elements = tensor.from_elements %5 : tensor<f64>
    %6 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
    %7 = stablehlo.add %arg2, %cst_0 : tensor<f64>
    %extracted_1 = tensor.extract %7[] : tensor<f64>
    %out_qubits_2 = quantum.custom "RX"(%extracted_1) %3#1 : !quantum.bit
    %8 = quantum.insert %6[ 1], %out_qubits_2 : !quantum.reg, !quantum.bit
    quantum.dealloc %8 : !quantum.reg
    quantum.device_release
    quantum.device["librtd_lightning.so", "LightningSimulator", "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %9 = quantum.alloc( 2) : !quantum.reg
    %10 = quantum.extract %9[ 0] : !quantum.reg -> !quantum.bit
    %11 = stablehlo.add %arg1, %cst : tensor<f64>
    %extracted_3 = tensor.extract %11[] : tensor<f64>
    %out_qubits_4 = quantum.custom "RY"(%extracted_3) %10 : !quantum.bit
    %12 = quantum.namedobs %out_qubits_4[ PauliX] : !quantum.obs
    %13 = quantum.expval %12 : f64
    %from_elements_5 = tensor.from_elements %13 : tensor<f64>
    %14 = quantum.insert %9[ 0], %out_qubits_4 : !quantum.reg, !quantum.bit
    quantum.dealloc %14 : !quantum.reg
    quantum.device_release
    %15 = stablehlo.add %from_elements, %from_elements_5 : tensor<f64>
    return %15 : tensor<f64>
  }
}

// CHECK-LABEL: circuit_twotapes_tape_1
// CHECK: quantum.device
// CHECK: quantum.dealloc {{%.+}} : !quantum.reg
// CHECK: quantum.device_release
// CHECK: return %from_elements : tensor<f64>
// CHECK-NEXT: }

// CHECK-LABEL: circuit_twotapes_tape_0
// CHECK: quantum.device
// CHECK: quantum.dealloc {{%.+}} : !quantum.reg
// CHECK: quantum.device_release
// CHECK: return %from_elements : tensor<f64>
// CHECK-NEXT: }

// CHECK-LABEL: circuit_twotapes
// CHECK: func.call @circuit_twotapes_tape_0
// CHECK: func.call @circuit_twotapes_tape_1
// CHECK: {{%.+}} = stablehlo.add {{%.+}}, {{%.+}} : tensor<f64>


// -----


// Should do nothing when there's no tapes
module @empty_workflow {

}

// CHECK: module @empty_workflow {
// CHECK-NEXT: }


// -----


// Should do nothing when there's only one tape
module @circuit_one_tape {
  func.func private @circuit(%arg0: tensor<f64>) -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    quantum.device["librtd_lightning.so", "LightningSimulator", "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %extracted = tensor.extract %arg0[] : tensor<f64>
    %out_qubits = quantum.custom "RX"(%extracted) %1 : !quantum.bit
    %2 = quantum.namedobs %out_qubits[ PauliZ] : !quantum.obs
    %3 = quantum.expval %2 : f64
    %from_elements = tensor.from_elements %3 : tensor<f64>
    %4 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
    quantum.dealloc %4 : !quantum.reg
    quantum.device_release
    return %from_elements : tensor<f64>
  }
}

// CHECK: module @circuit_one_tape {
// CHECK-NEXT:  func.func private @circuit(%arg0: tensor<f64>) -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
// CHECK-NEXT:    quantum.device["librtd_lightning.so", "LightningSimulator", "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
// CHECK-NEXT:    %0 = quantum.alloc( 1) : !quantum.reg
// CHECK-NEXT:    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
// CHECK-NEXT:    %extracted = tensor.extract %arg0[] : tensor<f64>
// CHECK-NEXT:    %out_qubits = quantum.custom "RX"(%extracted) %1 : !quantum.bit
// CHECK-NEXT:    %2 = quantum.namedobs %out_qubits[ PauliZ] : !quantum.obs
// CHECK-NEXT:    %3 = quantum.expval %2 : f64
// CHECK-NEXT:    %from_elements = tensor.from_elements %3 : tensor<f64>
// CHECK-NEXT:    %4 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
// CHECK-NEXT:    quantum.dealloc %4 : !quantum.reg
// CHECK-NEXT:    quantum.device_release
// CHECK-NEXT:    return %from_elements : tensor<f64>
// CHECK-NEXT:  }
// CHECK-NEXT:}
