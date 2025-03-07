module @f {
  func.func public @jit_f(%arg0: tensor<i64>) -> tensor<?xf64> attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_circ::@circ(%arg0) : (tensor<i64>) -> tensor<?xf64>
    return %0 : tensor<?xf64>
  }
  module @module_circ {
    module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
        transform.yield 
      }
    }
    func.func public @circ(%arg0: tensor<i64>) -> tensor<?xf64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
      %c = stablehlo.constant dense<0> : tensor<i64>
      %extracted = tensor.extract %c[] : tensor<i64>
      quantum.device shots(%extracted) ["/home/paul.wang/catalyst_new/catalyst/cat_pyenv/lib/python3.10/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", "LightningSimulator", "{'shots': 1000, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
      %extracted_0 = tensor.extract %arg0[] : tensor<i64>
      %0 = quantum.alloc(%extracted_0) : !quantum.reg
      %c_1 = stablehlo.constant dense<2> : tensor<i64>
      %extracted_2 = tensor.extract %c_1[] : tensor<i64>
      %1 = quantum.extract %0[%extracted_2] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.static_custom "RX" [1.230000e+00] %1 : !quantum.bit
      //%c_3 = stablehlo.constant dense<0> : tensor<i64>
      //%extracted_4 = tensor.extract %c_3[] : tensor<i64>
      //%2 = quantum.extract %0[%extracted_4] : !quantum.reg -> !quantum.bit
      //%c_5_probs = stablehlo.constant dense<2> : tensor<i64>
      //%extracted_6_probs = tensor.extract %c_5_probs[] : tensor<i64>
      //%probs_5 = quantum.insert %0[%extracted_6_probs], %out_qubits : !quantum.reg, !quantum.bit
      //%3 = quantum.compbasis num_qubits %extracted_0 : !quantum.obs
      //%3 = quantum.compbasis %out_qubits: !quantum.obs
      //%4 = quantum.probs %3 : tensor<?xf64>

      %c_5 = stablehlo.constant dense<2> : tensor<i64>
      %extracted_6 = tensor.extract %c_5[] : tensor<i64>
      %5 = quantum.insert %0[%extracted_6], %out_qubits : !quantum.reg, !quantum.bit
      %3 = quantum.compbasis qreg %5: !quantum.obs
      %4 = quantum.probs %3 : tensor<?xf64>
      quantum.dealloc %5 : !quantum.reg
      quantum.device_release
      return %4 : tensor<?xf64>
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
