module @qfunc {
  llvm.func @_mlir_memref_to_llvm_alloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_3xi64(dense<[0, 1, 2]> : tensor<3xi64>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<3 x i64>
  func.func public @jit_qfunc() -> memref<f64> attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(3735928559 : index) : i64
    %1 = call @qfunc_0() : () -> memref<f64>
    %2 = builtin.unrealized_conversion_cast %1 : memref<f64> to !llvm.struct<(ptr, ptr, i64)>
    %3 = builtin.unrealized_conversion_cast %1 : memref<f64> to !llvm.struct<(ptr, ptr, i64)>
    %4 = llvm.extractvalue %3[0] : !llvm.struct<(ptr, ptr, i64)> 
    %5 = llvm.ptrtoint %4 : !llvm.ptr to i64
    %6 = llvm.icmp "eq" %0, %5 : i64
    cf.cond_br %6, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.zero : !llvm.ptr
    %9 = llvm.getelementptr %8[%7] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %10 = llvm.ptrtoint %9 : !llvm.ptr to i64
    %11 = llvm.call @_mlir_memref_to_llvm_alloc(%10) : (i64) -> !llvm.ptr
    %12 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %13 = llvm.insertvalue %11, %12[0] : !llvm.struct<(ptr, ptr, i64)> 
    %14 = llvm.insertvalue %11, %13[1] : !llvm.struct<(ptr, ptr, i64)> 
    %15 = llvm.mlir.constant(0 : index) : i64
    %16 = llvm.insertvalue %15, %14[2] : !llvm.struct<(ptr, ptr, i64)> 
    %17 = builtin.unrealized_conversion_cast %16 : !llvm.struct<(ptr, ptr, i64)> to memref<f64>
    %18 = llvm.mlir.constant(1 : index) : i64
    %19 = llvm.mlir.zero : !llvm.ptr
    %20 = llvm.getelementptr %19[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %21 = llvm.ptrtoint %20 : !llvm.ptr to i64
    %22 = llvm.mul %18, %21 : i64
    %23 = llvm.extractvalue %2[1] : !llvm.struct<(ptr, ptr, i64)> 
    %24 = llvm.extractvalue %2[2] : !llvm.struct<(ptr, ptr, i64)> 
    %25 = llvm.getelementptr %23[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %26 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64)> 
    %27 = llvm.extractvalue %16[2] : !llvm.struct<(ptr, ptr, i64)> 
    %28 = llvm.getelementptr %26[%27] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%28, %25, %22) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    cf.br ^bb3(%17 : memref<f64>)
  ^bb2:  // pred: ^bb0
    cf.br ^bb3(%1 : memref<f64>)
  ^bb3(%29: memref<f64>):  // 2 preds: ^bb1, ^bb2
    cf.br ^bb4
  ^bb4:  // pred: ^bb3
    return %29 : memref<f64>
  }
  func.func public @qfunc_0() -> memref<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(3 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = llvm.getelementptr %3[%1] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %5 = llvm.ptrtoint %4 : !llvm.ptr to i64
    %6 = llvm.mlir.addressof @__constant_3xi64 : !llvm.ptr
    %7 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i64>
    %8 = llvm.mlir.constant(3735928559 : index) : i64
    %9 = llvm.inttoptr %8 : i64 to !llvm.ptr
    %10 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %9, %10[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %7, %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.mlir.constant(0 : index) : i64
    %14 = llvm.insertvalue %13, %12[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.insertvalue %1, %14[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.insertvalue %2, %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    quantum.device shots(%0) ["/Users/haider.sajjad/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_null_qubit.dylib", "NullQubit", "{'track_resources': False}"]
    %17 = quantum.alloc( 3) : !quantum.reg
    %18 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %19 = llvm.extractvalue %16[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.insertvalue %19, %18[0] : !llvm.struct<(ptr, ptr, i64)> 
    %22 = llvm.insertvalue %20, %21[1] : !llvm.struct<(ptr, ptr, i64)> 
    %23 = llvm.mlir.constant(2 : index) : i64
    %24 = llvm.insertvalue %23, %22[2] : !llvm.struct<(ptr, ptr, i64)> 
    %25 = llvm.extractvalue %24[1] : !llvm.struct<(ptr, ptr, i64)> 
    %26 = llvm.mlir.constant(2 : index) : i64
    %27 = llvm.getelementptr %25[%26] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %28 = llvm.load %27 : !llvm.ptr -> i64
    %29 = quantum.extract %17[%28] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %29 : !quantum.bit
    %30 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %31 = llvm.extractvalue %16[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %33 = llvm.insertvalue %31, %30[0] : !llvm.struct<(ptr, ptr, i64)> 
    %34 = llvm.insertvalue %32, %33[1] : !llvm.struct<(ptr, ptr, i64)> 
    %35 = llvm.mlir.constant(1 : index) : i64
    %36 = llvm.insertvalue %35, %34[2] : !llvm.struct<(ptr, ptr, i64)> 
    %37 = llvm.extractvalue %36[1] : !llvm.struct<(ptr, ptr, i64)> 
    %38 = llvm.mlir.constant(1 : index) : i64
    %39 = llvm.getelementptr %37[%38] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %40 = llvm.load %39 : !llvm.ptr -> i64
    %41 = quantum.extract %17[%40] : !quantum.reg -> !quantum.bit
    %out_qubits_0:2 = quantum.custom "CNOT"() %41, %out_qubits : !quantum.bit, !quantum.bit
    %42 = quantum.insert %17[%40], %out_qubits_0#0 : !quantum.reg, !quantum.bit
    %out_qubits_1 = quantum.custom "T"() %out_qubits_0#1 adj : !quantum.bit
    %43 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %44 = llvm.extractvalue %16[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %45 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %46 = llvm.insertvalue %44, %43[0] : !llvm.struct<(ptr, ptr, i64)> 
    %47 = llvm.insertvalue %45, %46[1] : !llvm.struct<(ptr, ptr, i64)> 
    %48 = llvm.mlir.constant(0 : index) : i64
    %49 = llvm.insertvalue %48, %47[2] : !llvm.struct<(ptr, ptr, i64)> 
    %50 = llvm.extractvalue %49[1] : !llvm.struct<(ptr, ptr, i64)> 
    %51 = llvm.load %50 : !llvm.ptr -> i64
    %52 = quantum.extract %42[%51] : !quantum.reg -> !quantum.bit
    %out_qubits_2:2 = quantum.custom "CNOT"() %52, %out_qubits_1 : !quantum.bit, !quantum.bit
    %53 = quantum.insert %42[%51], %out_qubits_2#0 : !quantum.reg, !quantum.bit
    %out_qubits_3 = quantum.custom "T"() %out_qubits_2#1 : !quantum.bit
    %54 = quantum.extract %53[%40] : !quantum.reg -> !quantum.bit
    %out_qubits_4:2 = quantum.custom "CNOT"() %54, %out_qubits_3 : !quantum.bit, !quantum.bit
    %55 = quantum.insert %53[%40], %out_qubits_4#0 : !quantum.reg, !quantum.bit
    %out_qubits_5 = quantum.custom "T"() %out_qubits_4#1 adj : !quantum.bit
    %56 = quantum.extract %55[%51] : !quantum.reg -> !quantum.bit
    %out_qubits_6:2 = quantum.custom "CNOT"() %56, %out_qubits_5 : !quantum.bit, !quantum.bit
    %57 = quantum.insert %55[%51], %out_qubits_6#0 : !quantum.reg, !quantum.bit
    %out_qubits_7 = quantum.custom "T"() %out_qubits_6#1 : !quantum.bit
    %58 = quantum.insert %57[%28], %out_qubits_7 : !quantum.reg, !quantum.bit
    %59 = quantum.extract %58[%40] : !quantum.reg -> !quantum.bit
    %out_qubits_8 = quantum.custom "T"() %59 : !quantum.bit
    %60 = quantum.extract %58[%51] : !quantum.reg -> !quantum.bit
    %out_qubits_9:2 = quantum.custom "CNOT"() %60, %out_qubits_8 : !quantum.bit, !quantum.bit
    %61 = quantum.insert %58[%51], %out_qubits_9#0 : !quantum.reg, !quantum.bit
    %62 = quantum.insert %61[%40], %out_qubits_9#1 : !quantum.reg, !quantum.bit
    %63 = quantum.extract %62[%28] : !quantum.reg -> !quantum.bit
    %out_qubits_10 = quantum.custom "Hadamard"() %63 : !quantum.bit
    %64 = quantum.insert %62[%28], %out_qubits_10 : !quantum.reg, !quantum.bit
    %65 = quantum.extract %64[%51] : !quantum.reg -> !quantum.bit
    %out_qubits_11 = quantum.custom "T"() %65 : !quantum.bit
    %66 = quantum.extract %64[%40] : !quantum.reg -> !quantum.bit
    %out_qubits_12 = quantum.custom "T"() %66 adj : !quantum.bit
    %out_qubits_13:2 = quantum.custom "CNOT"() %out_qubits_11, %out_qubits_12 : !quantum.bit, !quantum.bit
    %67 = quantum.insert %64[%51], %out_qubits_13#0 : !quantum.reg, !quantum.bit
    %68 = quantum.insert %67[%40], %out_qubits_13#1 : !quantum.reg, !quantum.bit
    %69 = quantum.extract %68[ 0] : !quantum.reg -> !quantum.bit
    %70 = quantum.namedobs %69[ PauliZ] : !quantum.obs
    %71 = quantum.insert %68[ 0], %69 : !quantum.reg, !quantum.bit
    %72 = quantum.expval %70 : f64
    %73 = llvm.mlir.constant(1 : index) : i64
    %74 = llvm.mlir.zero : !llvm.ptr
    %75 = llvm.getelementptr %74[%73] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %76 = llvm.ptrtoint %75 : !llvm.ptr to i64
    %77 = llvm.mlir.constant(64 : index) : i64
    %78 = llvm.add %76, %77 : i64
    %79 = llvm.call @_mlir_memref_to_llvm_alloc(%78) : (i64) -> !llvm.ptr
    %80 = llvm.ptrtoint %79 : !llvm.ptr to i64
    %81 = llvm.mlir.constant(1 : index) : i64
    %82 = llvm.sub %77, %81 : i64
    %83 = llvm.add %80, %82 : i64
    %84 = llvm.urem %83, %77 : i64
    %85 = llvm.sub %83, %84 : i64
    %86 = llvm.inttoptr %85 : i64 to !llvm.ptr
    %87 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %88 = llvm.insertvalue %79, %87[0] : !llvm.struct<(ptr, ptr, i64)> 
    %89 = llvm.insertvalue %86, %88[1] : !llvm.struct<(ptr, ptr, i64)> 
    %90 = llvm.mlir.constant(0 : index) : i64
    %91 = llvm.insertvalue %90, %89[2] : !llvm.struct<(ptr, ptr, i64)> 
    %92 = builtin.unrealized_conversion_cast %91 : !llvm.struct<(ptr, ptr, i64)> to memref<f64>
    %93 = llvm.extractvalue %91[1] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %72, %93 : f64, !llvm.ptr
    quantum.dealloc %71 : !quantum.reg
    quantum.device_release
    return %92 : memref<f64>
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