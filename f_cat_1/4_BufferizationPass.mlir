module @f_cat {
  func.func public @jit_f_cat() -> memref<?xf64> attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(3735928559 : index) : i64
    %1 = call @circ_0() : () -> memref<?xf64>
    %2 = builtin.unrealized_conversion_cast %1 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.extractvalue %2[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.ptrtoint %3 : !llvm.ptr to i64
    %5 = llvm.icmp "eq" %0, %4 : i64
    %6 = scf.if %5 -> (memref<?xf64>) {
      //%alloc = memref.alloc() : memref<?xf64>
      %nq = func.call @__catalyst__rt__num_qubits() : () -> i64
      %one = arith.constant 1 : i64
      %twoToN = arith.shli %one, %nq : i64
      %idx1 = index.casts %twoToN : i64 to index
      %alloc = memref.alloc(%idx1) : memref<?xf64>
      memref.copy %1, %alloc : memref<?xf64> to memref<?xf64>
      scf.yield %alloc : memref<?xf64>
    } else {
      scf.yield %1 : memref<?xf64>
    }
    return %6 : memref<?xf64>
  }

  func.func private @__catalyst__rt__num_qubits() -> i64

  func.func public @circ_0() -> memref<?xf64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 1.230000e+00 : f64
    %cst_0 = arith.constant 4.560000e+00 : f64
    quantum.device shots(%c0_i64) ["/home/paul.wang/catalyst_new/catalyst/cat_pyenv/lib/python3.10/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", "LightningSimulator", "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %0 = quantum.alloc( 0) : !quantum.reg
    %1 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RX"(%cst) %1 : !quantum.bit
    %2 = quantum.extract %0[ 4] : !quantum.reg -> !quantum.bit
    %out_qubits_1 = quantum.custom "RX"(%cst_0) %2 : !quantum.bit
    %3 = quantum.compbasis  : !quantum.obs

    // Alloc for probs() result buffer
    %nq = func.call @__catalyst__rt__num_qubits() : () -> i64
    %one = arith.constant 1 : i64
    %twoToN = arith.shli %one, %nq : i64
    %idx1 = index.casts %twoToN : i64 to index
    %alloc = memref.alloc(%idx1) : memref<?xf64>

    quantum.probs %3 in(%alloc : memref<?xf64>)
    %4 = quantum.insert %0[ 2], %out_qubits : !quantum.reg, !quantum.bit
    %5 = quantum.insert %4[ 4], %out_qubits_1 : !quantum.reg, !quantum.bit
    quantum.dealloc %5 : !quantum.reg
    quantum.device_release
    return %alloc : memref<?xf64>
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
