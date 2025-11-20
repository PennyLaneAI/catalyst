module @test_pipeline_lowering_workflow {
  func.func public @jit_test_pipeline_lowering_workflow(%arg0: memref<f64>) -> memref<f64> attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(3735928559 : index) : i64
    %1 = call @test_pipeline_lowering_workflow_0(%arg0) : (memref<f64>) -> memref<f64>
    %2 = builtin.unrealized_conversion_cast %1 : memref<f64> to !llvm.struct<(ptr, ptr, i64)>
    %3 = llvm.extractvalue %2[0] : !llvm.struct<(ptr, ptr, i64)> 
    %4 = llvm.ptrtoint %3 : !llvm.ptr to i64
    %5 = llvm.icmp "eq" %0, %4 : i64
    %6 = scf.if %5 -> (memref<f64>) {
      %alloc = memref.alloc() : memref<f64>
      memref.copy %1, %alloc : memref<f64> to memref<f64>
      scf.yield %alloc : memref<f64>
    } else {
      scf.yield %1 : memref<f64>
    }
    return %6 : memref<f64>
  }
  func.func public @test_pipeline_lowering_workflow_0(%arg0: memref<f64>) -> memref<f64> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
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