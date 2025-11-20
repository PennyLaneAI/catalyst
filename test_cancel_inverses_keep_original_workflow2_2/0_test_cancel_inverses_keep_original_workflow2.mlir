module @test_cancel_inverses_keep_original_workflow2 {
  func.func public @jit_test_cancel_inverses_keep_original_workflow2() -> (tensor<f64>, tensor<f64>) attributes {llvm.emit_c_interface} {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %0 = catalyst.launch_kernel @module_f::@f(%cst) : (tensor<f64>) -> tensor<f64>
    %1 = catalyst.launch_kernel @module_f_0::@f_1(%cst) : (tensor<f64>) -> tensor<f64>
    return %0, %1 : tensor<f64>, tensor<f64>
  }
  module @module_f {
    module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
        transform.yield 
      }
    }
    func.func public @f(%arg0: tensor<f64>) -> tensor<f64> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
      %c = stablehlo.constant dense<0> : tensor<i64>
      %extracted = tensor.extract %c[] : tensor<i64>
      quantum.device shots(%extracted) ["/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
      %c_0 = stablehlo.constant dense<1> : tensor<i64>
      %0 = quantum.alloc( 1) : !quantum.reg
      %extracted_1 = tensor.extract %c[] : tensor<i64>
      %1 = quantum.extract %0[%extracted_1] : !quantum.reg -> !quantum.bit
      %extracted_2 = tensor.extract %arg0[] : tensor<f64>
      %out_qubits = quantum.custom "RX"(%extracted_2) %1 : !quantum.bit
      %out_qubits_3 = quantum.custom "Hadamard"() %out_qubits : !quantum.bit
      %out_qubits_4 = quantum.custom "Hadamard"() %out_qubits_3 : !quantum.bit
      %2 = quantum.namedobs %out_qubits_4[ PauliZ] : !quantum.obs
      %3 = quantum.expval %2 : f64
      %from_elements = tensor.from_elements %3 : tensor<f64>
      %extracted_5 = tensor.extract %c[] : tensor<i64>
      %4 = quantum.insert %0[%extracted_5], %out_qubits_4 : !quantum.reg, !quantum.bit
      quantum.dealloc %4 : !quantum.reg
      quantum.device_release
      return %from_elements : tensor<f64>
    }
  }
  module @module_f_0 {
    module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
        %0 = transform.apply_registered_pass "cancel-inverses" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
        transform.yield 
      }
    }
    func.func public @f_1(%arg0: tensor<f64>) -> tensor<f64> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
      %c = stablehlo.constant dense<0> : tensor<i64>
      %extracted = tensor.extract %c[] : tensor<i64>
      quantum.device shots(%extracted) ["/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
      %c_0 = stablehlo.constant dense<1> : tensor<i64>
      %0 = quantum.alloc( 1) : !quantum.reg
      %extracted_1 = tensor.extract %c[] : tensor<i64>
      %1 = quantum.extract %0[%extracted_1] : !quantum.reg -> !quantum.bit
      %extracted_2 = tensor.extract %arg0[] : tensor<f64>
      %out_qubits = quantum.custom "RX"(%extracted_2) %1 : !quantum.bit
      %out_qubits_3 = quantum.custom "Hadamard"() %out_qubits : !quantum.bit
      %out_qubits_4 = quantum.custom "Hadamard"() %out_qubits_3 : !quantum.bit
      %2 = quantum.namedobs %out_qubits_4[ PauliZ] : !quantum.obs
      %3 = quantum.expval %2 : f64
      %from_elements = tensor.from_elements %3 : tensor<f64>
      %extracted_5 = tensor.extract %c[] : tensor<i64>
      %4 = quantum.insert %0[%extracted_5], %out_qubits_4 : !quantum.reg, !quantum.bit
      quantum.dealloc %4 : !quantum.reg
      quantum.device_release
      return %from_elements : tensor<f64>
    }
  }
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
  }
}