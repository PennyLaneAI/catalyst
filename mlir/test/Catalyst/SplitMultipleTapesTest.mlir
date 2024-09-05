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

// Test that --split-multiple-tapes pass correctly outlines each tape into its own function and calls the outlined functions.
// A tape is the operations between a "quantum.device" and a "quantum.device_release", inclusive.


// Basic test with a two-tape function
// Each tape take in different number elements in different order

module @circuit_twotapes_module {
  func.func private @circuit_twotapes(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %cst = stablehlo.constant dense<4.000000e-01> : tensor<f64>
    %cst_0 = stablehlo.constant dense<8.000000e-01> : tensor<f64>
    quantum.device["librtd_lightning.so", "LightningSimulator", "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %tape0_out = stablehlo.add %arg1, %cst_0 : tensor<f64>
    quantum.device_release
    quantum.device["librtd_lightning.so", "LightningSimulator", "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %extracted = tensor.extract %arg1[] : tensor<f64>
    %tape1_out = stablehlo.multiply %arg0, %cst : tensor<f64>
    quantum.device_release
    %result = stablehlo.subtract %tape0_out, %tape1_out : tensor<f64>
    return %result : tensor<f64>
  }
}

// CHECK: module @circuit_twotapes_module {
// CHECK: func.func private @circuit_twotapes
// CHECK: func.call @circuit_twotapes_tape_0(%arg1, %cst_0)
// CHECK: func.call @circuit_twotapes_tape_1(%arg1, %arg0, %cst)
// CHECK: {{%.+}} = stablehlo.subtract {{%.+}}, {{%.+}} : tensor<f64>

// CHECK: func.func private @circuit_twotapes_tape_0(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode}
// CHECK: quantum.device
// CHECK: {{%.+}} = stablehlo.add %arg0, %arg1 : tensor<f64>
// CHECK: quantum.device_release
// CHECK: return {{%.+}} : tensor<f64>
// CHECK-NEXT: }

// CHECK: func.func private @circuit_twotapes_tape_1(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode}
// CHECK: quantum.device
// CHECK: tensor.extract %arg0[]
// CHECK: {{%.+}} = stablehlo.multiply %arg1, %arg2 : tensor<f64>
// CHECK: quantum.device_release
// CHECK: return {{%.+}} : tensor<f64>
// CHECK-NEXT: }

// CHECK: }


// -----


// A module with two two-tape functions
// Both of them needs to be split

module @circuit_twotapes_module {
  func.func private @circuit_twotapes(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %cst = stablehlo.constant dense<4.000000e-01> : tensor<f64>
    %cst_0 = stablehlo.constant dense<8.000000e-01> : tensor<f64>
    quantum.device["librtd_lightning.so", "LightningSimulator", "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %tape0_out = stablehlo.add %arg1, %cst_0 : tensor<f64>
    quantum.device_release
    quantum.device["librtd_lightning.so", "LightningSimulator", "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %extracted = tensor.extract %arg1[] : tensor<f64>
    %tape1_out = stablehlo.multiply %arg0, %cst : tensor<f64>
    quantum.device_release
    %result = stablehlo.subtract %tape0_out, %tape1_out : tensor<f64>
    return %result : tensor<f64>
  }
  func.func private @circuit_twotapes_doppleganger(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %cst = stablehlo.constant dense<4.000000e-01> : tensor<f64>
    %cst_0 = stablehlo.constant dense<8.000000e-01> : tensor<f64>
    quantum.device["librtd_lightning.so", "LightningSimulator", "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %tape0_out = stablehlo.add %arg1, %cst_0 : tensor<f64>
    quantum.device_release
    quantum.device["librtd_lightning.so", "LightningSimulator", "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %extracted = tensor.extract %arg1[] : tensor<f64>
    %tape1_out = stablehlo.multiply %arg0, %cst : tensor<f64>
    quantum.device_release
    %result = stablehlo.subtract %tape0_out, %tape1_out : tensor<f64>
    return %result : tensor<f64>
  }
}

// CHECK: module @circuit_twotapes_module {
// CHECK: func.func private @circuit_twotapes
// CHECK: func.call @circuit_twotapes_tape_0(%arg1, %cst_0)
// CHECK: func.call @circuit_twotapes_tape_1(%arg1, %arg0, %cst)
// CHECK: {{%.+}} = stablehlo.subtract {{%.+}}, {{%.+}} : tensor<f64>

// CHECK: func.func private @circuit_twotapes_tape_0(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode}
// CHECK: quantum.device
// CHECK: {{%.+}} = stablehlo.add %arg0, %arg1 : tensor<f64>
// CHECK: quantum.device_release
// CHECK: return {{%.+}} : tensor<f64>
// CHECK-NEXT: }

// CHECK: func.func private @circuit_twotapes_tape_1(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode}
// CHECK: quantum.device
// CHECK: tensor.extract %arg0[]
// CHECK: {{%.+}} = stablehlo.multiply %arg1, %arg2 : tensor<f64>
// CHECK: quantum.device_release
// CHECK: return {{%.+}} : tensor<f64>
// CHECK-NEXT: }

// CHECK: func.func private @circuit_twotapes_doppleganger
// CHECK: func.call @circuit_twotapes_doppleganger_tape_0(%arg1, %cst_0)
// CHECK: func.call @circuit_twotapes_doppleganger_tape_1(%arg1, %arg0, %cst)
// CHECK: {{%.+}} = stablehlo.subtract {{%.+}}, {{%.+}} : tensor<f64>

// CHECK: func.func private @circuit_twotapes_doppleganger_tape_0(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode}
// CHECK: quantum.device
// CHECK: {{%.+}} = stablehlo.add %arg0, %arg1 : tensor<f64>
// CHECK: quantum.device_release
// CHECK: return {{%.+}} : tensor<f64>
// CHECK-NEXT: }

// CHECK: func.func private @circuit_twotapes_doppleganger_tape_1(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode}
// CHECK: quantum.device
// CHECK: tensor.extract %arg0[]
// CHECK: {{%.+}} = stablehlo.multiply %arg1, %arg2 : tensor<f64>
// CHECK: quantum.device_release
// CHECK: return {{%.+}} : tensor<f64>
// CHECK-NEXT: }

// CHECK: }


// -----


// Test when a post processing op is nested and the necessary values are not exposed at the upmost layer

module @circuit_twotapes_module {
  func.func private @circuit_twotapes(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %cst = stablehlo.constant dense<4.000000e-01> : tensor<f64>
    %cst_0 = stablehlo.constant dense<8.000000e-01> : tensor<f64>
    quantum.device["librtd_lightning.so", "LightningSimulator", "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %tape0_out = stablehlo.add %arg1, %cst_0 : tensor<f64>
    quantum.device_release
    quantum.device["librtd_lightning.so", "LightningSimulator", "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %extracted = tensor.extract %arg1[] : tensor<f64>
    %tape1_out = stablehlo.multiply %arg0, %cst : tensor<f64>
    quantum.device_release
    %result = scf.execute_region -> (tensor<f64>) {
      %result_in_scf = stablehlo.subtract %tape0_out, %tape1_out : tensor<f64>
      scf.yield %result_in_scf : tensor<f64>
    }
    return %result : tensor<f64>
  }
}

// CHECK: module @circuit_twotapes_module {
// CHECK: func.func private @circuit_twotapes
// CHECK: func.call @circuit_twotapes_tape_0(%arg1, %cst_0)
// CHECK: func.call @circuit_twotapes_tape_1(%arg1, %arg0, %cst)
// CHECK: {{%.+}} = scf.execute_region
// CHECK: {{%.+}} = stablehlo.subtract {{%.+}}, {{%.+}} : tensor<f64>
// CHECK: scf.yield

// CHECK: func.func private @circuit_twotapes_tape_0(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode}
// CHECK: quantum.device
// CHECK: {{%.+}} = stablehlo.add %arg0, %arg1 : tensor<f64>
// CHECK: quantum.device_release
// CHECK: return {{%.+}} : tensor<f64>
// CHECK-NEXT: }

// CHECK: func.func private @circuit_twotapes_tape_1(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode}
// CHECK: quantum.device
// CHECK: tensor.extract %arg0[]
// CHECK: {{%.+}} = stablehlo.multiply %arg1, %arg2 : tensor<f64>
// CHECK: quantum.device_release
// CHECK: return {{%.+}} : tensor<f64>
// CHECK-NEXT: }

// CHECK: }


// -----


// Should do nothing when there's no tapes
module @empty_workflow {

}

module @classical_workflow {
  func.func private @func() -> tensor<f64> {
    %cst = stablehlo.constant dense<4.000000e-01> : tensor<f64>
    return %cst : tensor<f64>
  }
}

// CHECK: module @empty_workflow {
// CHECK-NEXT: }

// CHECK-NEXT: module @classical_workflow {
// CHCEK-NEXT: func.func private @func() -> tensor<f64> {
// CHECK: %cst = stablehlo.constant dense<4.000000e-01> : tensor<f64>
// CHECK-NEXT: return %cst : tensor<f64>
// CHECK-NEXT: }
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
