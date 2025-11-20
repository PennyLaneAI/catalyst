module @test_pipeline_lowering_keep_original_workflow {
  func.func public @jit_test_pipeline_lowering_keep_original_workflow(%arg0: memref<f64>) -> (memref<f64>, memref<f64>) attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(3735928559 : index) : i64
    %1 = call @f_0(%arg0) : (memref<f64>) -> memref<f64>
    %2 = call @f_1_0(%arg0) : (memref<f64>) -> memref<f64>
    %3 = builtin.unrealized_conversion_cast %1 : memref<f64> to !llvm.struct<(ptr, ptr, i64)>
    %4 = llvm.extractvalue %3[0] : !llvm.struct<(ptr, ptr, i64)> 
    %5 = llvm.ptrtoint %4 : !llvm.ptr to i64
    %6 = llvm.icmp "eq" %0, %5 : i64
    %7 = scf.if %6 -> (memref<f64>) {
      %alloc = memref.alloc() : memref<f64>
      memref.copy %1, %alloc : memref<f64> to memref<f64>
      scf.yield %alloc : memref<f64>
    } else {
      scf.yield %1 : memref<f64>
    }
    %8 = builtin.unrealized_conversion_cast %2 : memref<f64> to !llvm.struct<(ptr, ptr, i64)>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr, ptr, i64)> 
    %10 = llvm.ptrtoint %9 : !llvm.ptr to i64
    %11 = llvm.icmp "eq" %0, %10 : i64
    %12 = scf.if %11 -> (memref<f64>) {
      %alloc = memref.alloc() : memref<f64>
      memref.copy %2, %alloc : memref<f64> to memref<f64>
      scf.yield %alloc : memref<f64>
    } else {
      scf.yield %2 : memref<f64>
    }
    return %7, %12 : memref<f64>, memref<f64>
  }
  func.func public @f_0(%arg0: memref<f64>) -> memref<f64> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %c0_i64 = arith.constant 0 : i64
    quantum.device shots(%c0_i64) ["/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = memref.load %arg0[] : memref<f64>
    %out_qubits = quantum.custom "RX"(%2) %1 : !quantum.bit
    %3 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %out_qubits_0 = quantum.custom "Hadamard"() %3 : !quantum.bit
    %out_qubits_1 = quantum.custom "Hadamard"() %out_qubits_0 : !quantum.bit
    %4 = quantum.namedobs %out_qubits[ PauliY] : !quantum.obs
    %5 = quantum.expval %4 : f64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f64>
    memref.store %5, %alloc[] : memref<f64>
    %6 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
    %7 = quantum.insert %6[ 1], %out_qubits_1 : !quantum.reg, !quantum.bit
    quantum.dealloc %7 : !quantum.reg
    quantum.device_release
    return %alloc : memref<f64>
  }
  func.func public @f_1_0(%arg0: memref<f64>) -> memref<f64> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %c0_i64 = arith.constant 0 : i64
    quantum.device shots(%c0_i64) ["/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = memref.load %arg0[] : memref<f64>
    %out_qubits = quantum.custom "RX"(%2) %1 : !quantum.bit
    %3 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %4 = quantum.namedobs %out_qubits[ PauliY] : !quantum.obs
    %5 = quantum.expval %4 : f64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f64>
    memref.store %5, %alloc[] : memref<f64>
    %6 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
    %7 = quantum.insert %6[ 1], %3 : !quantum.reg, !quantum.bit
    quantum.dealloc %7 : !quantum.reg
    quantum.device_release
    return %alloc : memref<f64>
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