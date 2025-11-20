module @test_pipeline_lowering_workflow {
  func.func public @jit_test_pipeline_lowering_workflow(%arg0: tensor<f64>) -> tensor<f64> attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_test_pipeline_lowering_workflow::@test_pipeline_lowering_workflow(%arg0) : (tensor<f64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
  module @module_test_pipeline_lowering_workflow {
    module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
        %0 = transform.apply_registered_pass "cancel-inverses" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
        %1 = transform.apply_registered_pass "merge-rotations" to %0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
        transform.yield 
      }
    }
    func.func public @test_pipeline_lowering_workflow(%arg0: tensor<f64>) -> tensor<f64> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
      %c = stablehlo.constant dense<0> : tensor<i64>
      %extracted = tensor.extract %c[] : tensor<i64>
      quantum.device shots(%extracted) ["/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
      %c_0 = stablehlo.constant dense<2> : tensor<i64>
      %0 = quantum.alloc( 2) : !quantum.reg
      %extracted_1 = tensor.extract %c[] : tensor<i64>
      %1 = quantum.extract %0[%extracted_1] : !quantum.reg -> !quantum.bit
      %extracted_2 = tensor.extract %arg0[] : tensor<f64>
      %out_qubits = quantum.custom "RX"(%extracted_2) %1 : !quantum.bit
      %c_3 = stablehlo.constant dense<1> : tensor<i64>
      %extracted_4 = tensor.extract %c_3[] : tensor<i64>
      %2 = quantum.extract %0[%extracted_4] : !quantum.reg -> !quantum.bit
      %out_qubits_5 = quantum.custom "Hadamard"() %2 : !quantum.bit
      %out_qubits_6 = quantum.custom "Hadamard"() %out_qubits_5 : !quantum.bit
      %3 = quantum.namedobs %out_qubits[ PauliY] : !quantum.obs
      %4 = quantum.expval %3 : f64
      %from_elements = tensor.from_elements %4 : tensor<f64>
      %extracted_7 = tensor.extract %c[] : tensor<i64>
      %5 = quantum.insert %0[%extracted_7], %out_qubits : !quantum.reg, !quantum.bit
      %extracted_8 = tensor.extract %c_3[] : tensor<i64>
      %6 = quantum.insert %5[%extracted_8], %out_qubits_6 : !quantum.reg, !quantum.bit
      quantum.dealloc %6 : !quantum.reg
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