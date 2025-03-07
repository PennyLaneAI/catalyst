module @f_cat {
  func.func public @jit_f_cat() -> tensor<2xf64> attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_circ::@circ() : () -> tensor<2xf64>
    return %0 : tensor<2xf64>
  }
  module @module_circ {
    module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
        transform.yield 
      }
    }
    func.func public @circ() -> tensor<2xf64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
      %c = stablehlo.constant dense<0> : tensor<i64>
      %extracted = tensor.extract %c[] : tensor<i64>
      quantum.device shots(%extracted) ["/home/paul.wang/catalyst_new/catalyst/cat_pyenv/lib/python3.10/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", "LightningSimulator", "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
      %c_0 = stablehlo.constant dense<3> : tensor<i64>
      %0 = quantum.alloc( 3) : !quantum.reg
      %c_1 = stablehlo.constant dense<0> : tensor<i64>
      %extracted_2 = tensor.extract %c_1[] : tensor<i64>
      %1 = quantum.extract %0[%extracted_2] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.static_custom "RX" [1.230000e+00] %1 : !quantum.bit

      // start manual ir {
      // Basic simple case:
      // Request a new aux wire
      // Do something to it
      // Measure on aux wire
      // Do things to the main wire depending on what was seen
      // Release the new aux wire

      // aux circuit workflow and measurement
      %another_reg = quantum.alloc( 1) : !quantum.reg
      %another_bit = quantum.extract %another_reg[ 0] : !quantum.reg -> !quantum.bit
      %another_out = quantum.static_custom "RX" [0.0] %another_bit : !quantum.bit
      %aux_expval_obs = quantum.namedobs %another_out[ PauliZ] : !quantum.obs
      %aux_expval = quantum.expval %aux_expval_obs : f64

      // conditional amendment to main circuit
      %point_five = arith.constant 0.5 : f64  // <0|Z|0> = 1 > 0.5
      %compare = arith.cmpf ogt, %aux_expval, %point_five : f64
      %out_qubits_if = scf.if %compare -> (!quantum.bit) {
        %out_qubits_in_scf = quantum.static_custom "RX" [-1.230000e+00] %out_qubits : !quantum.bit
        scf.yield %out_qubits_in_scf : !quantum.bit
      } else {
        scf.yield %out_qubits : !quantum.bit
      }

      // Regarding deletion:
      // In runtime, a qubit_release_array actually releases ALL qubits on the device,
      // regardless of the register.
      //
      // static int __catalyst__rt__qubit_release_array__impl(QirArray *qubit_array)
      // {
      //     getQuantumDevicePtr()->ReleaseAllQubits();
      //     std::vector<QubitIdType> *qubit_array_ptr =
      //         reinterpret_cast<std::vector<QubitIdType> *>(qubit_array);
      //     delete qubit_array_ptr;
      //     return 0;
      // }
      //
      // What if we never delete the aux qubits?...
      // But if we keep them around, what would happen with the return shapes like probs() (without wires)?
      %another_reg_free = quantum.insert %another_reg[ 0], %out_qubits_if : !quantum.reg, !quantum.bit
      quantum.dealloc %another_reg_free : !quantum.reg
      // } end manual ir

      %2 = quantum.compbasis %out_qubits_if : !quantum.obs
      %3 = quantum.probs %2 : tensor<2xf64>
      %c_3 = stablehlo.constant dense<0> : tensor<i64>
      %extracted_4 = tensor.extract %c_3[] : tensor<i64>
      %4 = quantum.insert %0[%extracted_4], %out_qubits_if : !quantum.reg, !quantum.bit
      quantum.dealloc %4 : !quantum.reg
      quantum.device_release
      return %3 : tensor<2xf64>
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
