module @qfunc {
  llvm.func @__catalyst__rt__finalize()
  llvm.func @__catalyst__rt__initialize(!llvm.ptr)
  llvm.func @__catalyst__rt__device_release()
  llvm.func @__catalyst__rt__qubit_release_array(!llvm.ptr)
  llvm.func @__catalyst__qis__Expval(i64) -> f64
  llvm.func @__catalyst__qis__NamedObs(i64, !llvm.ptr) -> i64
  llvm.func @__catalyst__qis__T(!llvm.ptr, !llvm.ptr)
  llvm.func @__catalyst__qis__CNOT(!llvm.ptr, !llvm.ptr, !llvm.ptr)
  llvm.func @__catalyst__qis__Hadamard(!llvm.ptr, !llvm.ptr)
  llvm.func @__catalyst__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @__catalyst__rt__qubit_allocate_array(i64) -> !llvm.ptr
  llvm.mlir.global internal constant @"{'track_resources': False}"("{'track_resources': False}\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @NullQubit("NullQubit\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @"/Users/haider.sajjad/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_null_qubit.dylib"("/Users/haider.sajjad/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_null_qubit.dylib\00") {addr_space = 0 : i32}
  llvm.func @__catalyst__rt__device_init(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i1)
  llvm.func @_mlir_memref_to_llvm_alloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_3xi64(dense<[0, 1, 2]> : tensor<3xi64>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<3 x i64>
  llvm.func @jit_qfunc() -> !llvm.struct<(ptr, ptr, i64)> attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(3735928559 : index) : i64
    %1 = llvm.call @qfunc_0() : () -> !llvm.struct<(ptr, ptr, i64)>
    %2 = builtin.unrealized_conversion_cast %1 : !llvm.struct<(ptr, ptr, i64)> to memref<f64>
    %3 = builtin.unrealized_conversion_cast %2 : memref<f64> to !llvm.struct<(ptr, ptr, i64)>
    %4 = builtin.unrealized_conversion_cast %2 : memref<f64> to !llvm.struct<(ptr, ptr, i64)>
    %5 = llvm.extractvalue %4[0] : !llvm.struct<(ptr, ptr, i64)> 
    %6 = llvm.ptrtoint %5 : !llvm.ptr to i64
    %7 = llvm.icmp "eq" %0, %6 : i64
    llvm.cond_br %7, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.zero : !llvm.ptr
    %10 = llvm.getelementptr %9[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %11 = llvm.ptrtoint %10 : !llvm.ptr to i64
    %12 = llvm.call @_mlir_memref_to_llvm_alloc(%11) : (i64) -> !llvm.ptr
    %13 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %14 = llvm.insertvalue %12, %13[0] : !llvm.struct<(ptr, ptr, i64)> 
    %15 = llvm.insertvalue %12, %14[1] : !llvm.struct<(ptr, ptr, i64)> 
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.insertvalue %16, %15[2] : !llvm.struct<(ptr, ptr, i64)> 
    %18 = builtin.unrealized_conversion_cast %17 : !llvm.struct<(ptr, ptr, i64)> to memref<f64>
    %19 = llvm.mlir.constant(1 : index) : i64
    %20 = llvm.mlir.zero : !llvm.ptr
    %21 = llvm.getelementptr %20[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %22 = llvm.ptrtoint %21 : !llvm.ptr to i64
    %23 = llvm.mul %19, %22 : i64
    %24 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, ptr, i64)> 
    %25 = llvm.extractvalue %3[2] : !llvm.struct<(ptr, ptr, i64)> 
    %26 = llvm.getelementptr %24[%25] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %27 = llvm.extractvalue %17[1] : !llvm.struct<(ptr, ptr, i64)> 
    %28 = llvm.extractvalue %17[2] : !llvm.struct<(ptr, ptr, i64)> 
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%29, %26, %23) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb3(%17 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb3(%1 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb3(%30: !llvm.struct<(ptr, ptr, i64)>):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %30 : !llvm.struct<(ptr, ptr, i64)>
  }
  llvm.func @_mlir_ciface_jit_qfunc(%arg0: !llvm.ptr) attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.call @jit_qfunc() : () -> !llvm.struct<(ptr, ptr, i64)>
    llvm.store %0, %arg0 : !llvm.struct<(ptr, ptr, i64)>, !llvm.ptr
    llvm.return
  }
  llvm.func internal @qfunc_0() -> !llvm.struct<(ptr, ptr, i64)> attributes {diff_method = "parameter-shift", qnode} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.alloca %1 x !llvm.struct<(i1, i64, ptr, ptr)> : (i64) -> !llvm.ptr
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.alloca %3 x !llvm.struct<(i1, i64, ptr, ptr)> : (i64) -> !llvm.ptr
    %5 = llvm.mlir.constant(1 : i64) : i64
    %6 = llvm.alloca %5 x !llvm.struct<(i1, i64, ptr, ptr)> : (i64) -> !llvm.ptr
    %7 = llvm.mlir.constant(3 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.zero : !llvm.ptr
    %10 = llvm.getelementptr %9[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %11 = llvm.ptrtoint %10 : !llvm.ptr to i64
    %12 = llvm.mlir.addressof @__constant_3xi64 : !llvm.ptr
    %13 = llvm.getelementptr %12[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i64>
    %14 = llvm.mlir.constant(3735928559 : index) : i64
    %15 = llvm.inttoptr %14 : i64 to !llvm.ptr
    %16 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.insertvalue %15, %16[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.insertvalue %13, %17[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.mlir.constant(0 : index) : i64
    %20 = llvm.insertvalue %19, %18[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.insertvalue %7, %20[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.insertvalue %8, %21[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %23 = llvm.mlir.addressof @"/Users/haider.sajjad/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_null_qubit.dylib" : !llvm.ptr
    %24 = llvm.getelementptr inbounds %23[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<105 x i8>
    %25 = llvm.mlir.addressof @NullQubit : !llvm.ptr
    %26 = llvm.getelementptr inbounds %25[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i8>
    %27 = llvm.mlir.addressof @"{'track_resources': False}" : !llvm.ptr
    %28 = llvm.getelementptr inbounds %27[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<27 x i8>
    %29 = llvm.mlir.constant(false) : i1
    llvm.call @__catalyst__rt__device_init(%24, %26, %28, %0, %29) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i1) -> ()
    %30 = llvm.mlir.constant(3 : i64) : i64
    %31 = llvm.call @__catalyst__rt__qubit_allocate_array(%30) : (i64) -> !llvm.ptr
    %32 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %33 = llvm.extractvalue %22[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.insertvalue %33, %32[0] : !llvm.struct<(ptr, ptr, i64)> 
    %36 = llvm.insertvalue %34, %35[1] : !llvm.struct<(ptr, ptr, i64)> 
    %37 = llvm.mlir.constant(2 : index) : i64
    %38 = llvm.insertvalue %37, %36[2] : !llvm.struct<(ptr, ptr, i64)> 
    %39 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64)> 
    %40 = llvm.mlir.constant(2 : index) : i64
    %41 = llvm.getelementptr %39[%40] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %42 = llvm.load %41 : !llvm.ptr -> i64
    %43 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%31, %42) : (!llvm.ptr, i64) -> !llvm.ptr
    %44 = llvm.load %43 : !llvm.ptr -> !llvm.ptr
    %45 = llvm.mlir.zero : !llvm.ptr
    llvm.call @__catalyst__qis__Hadamard(%44, %45) : (!llvm.ptr, !llvm.ptr) -> ()
    %46 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %47 = llvm.extractvalue %22[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %48 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %49 = llvm.insertvalue %47, %46[0] : !llvm.struct<(ptr, ptr, i64)> 
    %50 = llvm.insertvalue %48, %49[1] : !llvm.struct<(ptr, ptr, i64)> 
    %51 = llvm.mlir.constant(1 : index) : i64
    %52 = llvm.insertvalue %51, %50[2] : !llvm.struct<(ptr, ptr, i64)> 
    %53 = llvm.extractvalue %52[1] : !llvm.struct<(ptr, ptr, i64)> 
    %54 = llvm.mlir.constant(1 : index) : i64
    %55 = llvm.getelementptr %53[%54] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %56 = llvm.load %55 : !llvm.ptr -> i64
    %57 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%31, %56) : (!llvm.ptr, i64) -> !llvm.ptr
    %58 = llvm.load %57 : !llvm.ptr -> !llvm.ptr
    %59 = llvm.mlir.zero : !llvm.ptr
    llvm.call @__catalyst__qis__CNOT(%58, %44, %59) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %60 = llvm.mlir.zero : !llvm.ptr
    %61 = llvm.mlir.constant(true) : i1
    %62 = llvm.getelementptr inbounds %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    %63 = llvm.getelementptr inbounds %6[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    %64 = llvm.getelementptr inbounds %6[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    %65 = llvm.getelementptr inbounds %6[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    llvm.store %61, %62 : i1, !llvm.ptr
    %66 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %66, %63 : i64, !llvm.ptr
    llvm.store %60, %64 : !llvm.ptr, !llvm.ptr
    llvm.store %60, %65 : !llvm.ptr, !llvm.ptr
    llvm.call @__catalyst__qis__T(%44, %6) : (!llvm.ptr, !llvm.ptr) -> ()
    %67 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %68 = llvm.extractvalue %22[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %69 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %70 = llvm.insertvalue %68, %67[0] : !llvm.struct<(ptr, ptr, i64)> 
    %71 = llvm.insertvalue %69, %70[1] : !llvm.struct<(ptr, ptr, i64)> 
    %72 = llvm.mlir.constant(0 : index) : i64
    %73 = llvm.insertvalue %72, %71[2] : !llvm.struct<(ptr, ptr, i64)> 
    %74 = llvm.extractvalue %73[1] : !llvm.struct<(ptr, ptr, i64)> 
    %75 = llvm.load %74 : !llvm.ptr -> i64
    %76 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%31, %75) : (!llvm.ptr, i64) -> !llvm.ptr
    %77 = llvm.load %76 : !llvm.ptr -> !llvm.ptr
    %78 = llvm.mlir.zero : !llvm.ptr
    llvm.call @__catalyst__qis__CNOT(%77, %44, %78) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %79 = llvm.mlir.zero : !llvm.ptr
    llvm.call @__catalyst__qis__T(%44, %79) : (!llvm.ptr, !llvm.ptr) -> ()
    %80 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%31, %56) : (!llvm.ptr, i64) -> !llvm.ptr
    %81 = llvm.load %80 : !llvm.ptr -> !llvm.ptr
    %82 = llvm.mlir.zero : !llvm.ptr
    llvm.call @__catalyst__qis__CNOT(%81, %44, %82) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %83 = llvm.mlir.zero : !llvm.ptr
    %84 = llvm.mlir.constant(true) : i1
    %85 = llvm.getelementptr inbounds %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    %86 = llvm.getelementptr inbounds %4[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    %87 = llvm.getelementptr inbounds %4[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    %88 = llvm.getelementptr inbounds %4[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    llvm.store %84, %85 : i1, !llvm.ptr
    %89 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %89, %86 : i64, !llvm.ptr
    llvm.store %83, %87 : !llvm.ptr, !llvm.ptr
    llvm.store %83, %88 : !llvm.ptr, !llvm.ptr
    llvm.call @__catalyst__qis__T(%44, %4) : (!llvm.ptr, !llvm.ptr) -> ()
    %90 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%31, %75) : (!llvm.ptr, i64) -> !llvm.ptr
    %91 = llvm.load %90 : !llvm.ptr -> !llvm.ptr
    %92 = llvm.mlir.zero : !llvm.ptr
    llvm.call @__catalyst__qis__CNOT(%91, %44, %92) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %93 = llvm.mlir.zero : !llvm.ptr
    llvm.call @__catalyst__qis__T(%44, %93) : (!llvm.ptr, !llvm.ptr) -> ()
    %94 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%31, %56) : (!llvm.ptr, i64) -> !llvm.ptr
    %95 = llvm.load %94 : !llvm.ptr -> !llvm.ptr
    %96 = llvm.mlir.zero : !llvm.ptr
    llvm.call @__catalyst__qis__T(%95, %96) : (!llvm.ptr, !llvm.ptr) -> ()
    %97 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%31, %75) : (!llvm.ptr, i64) -> !llvm.ptr
    %98 = llvm.load %97 : !llvm.ptr -> !llvm.ptr
    %99 = llvm.mlir.zero : !llvm.ptr
    llvm.call @__catalyst__qis__CNOT(%98, %95, %99) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %100 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%31, %42) : (!llvm.ptr, i64) -> !llvm.ptr
    %101 = llvm.load %100 : !llvm.ptr -> !llvm.ptr
    %102 = llvm.mlir.zero : !llvm.ptr
    llvm.call @__catalyst__qis__Hadamard(%101, %102) : (!llvm.ptr, !llvm.ptr) -> ()
    %103 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%31, %75) : (!llvm.ptr, i64) -> !llvm.ptr
    %104 = llvm.load %103 : !llvm.ptr -> !llvm.ptr
    %105 = llvm.mlir.zero : !llvm.ptr
    llvm.call @__catalyst__qis__T(%104, %105) : (!llvm.ptr, !llvm.ptr) -> ()
    %106 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%31, %56) : (!llvm.ptr, i64) -> !llvm.ptr
    %107 = llvm.load %106 : !llvm.ptr -> !llvm.ptr
    %108 = llvm.mlir.zero : !llvm.ptr
    %109 = llvm.mlir.constant(true) : i1
    %110 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    %111 = llvm.getelementptr inbounds %2[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    %112 = llvm.getelementptr inbounds %2[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    %113 = llvm.getelementptr inbounds %2[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i64, ptr, ptr)>
    llvm.store %109, %110 : i1, !llvm.ptr
    %114 = llvm.mlir.constant(0 : i64) : i64
    llvm.store %114, %111 : i64, !llvm.ptr
    llvm.store %108, %112 : !llvm.ptr, !llvm.ptr
    llvm.store %108, %113 : !llvm.ptr, !llvm.ptr
    llvm.call @__catalyst__qis__T(%107, %2) : (!llvm.ptr, !llvm.ptr) -> ()
    %115 = llvm.mlir.zero : !llvm.ptr
    llvm.call @__catalyst__qis__CNOT(%104, %107, %115) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %116 = llvm.mlir.constant(0 : i64) : i64
    %117 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%31, %116) : (!llvm.ptr, i64) -> !llvm.ptr
    %118 = llvm.load %117 : !llvm.ptr -> !llvm.ptr
    %119 = llvm.mlir.constant(3 : i64) : i64
    %120 = llvm.call @__catalyst__qis__NamedObs(%119, %118) : (i64, !llvm.ptr) -> i64
    %121 = llvm.call @__catalyst__qis__Expval(%120) : (i64) -> f64
    %122 = llvm.mlir.constant(1 : index) : i64
    %123 = llvm.mlir.zero : !llvm.ptr
    %124 = llvm.getelementptr %123[%122] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %125 = llvm.ptrtoint %124 : !llvm.ptr to i64
    %126 = llvm.mlir.constant(64 : index) : i64
    %127 = llvm.add %125, %126 : i64
    %128 = llvm.call @_mlir_memref_to_llvm_alloc(%127) : (i64) -> !llvm.ptr
    %129 = llvm.ptrtoint %128 : !llvm.ptr to i64
    %130 = llvm.mlir.constant(1 : index) : i64
    %131 = llvm.sub %126, %130 : i64
    %132 = llvm.add %129, %131 : i64
    %133 = llvm.urem %132, %126 : i64
    %134 = llvm.sub %132, %133 : i64
    %135 = llvm.inttoptr %134 : i64 to !llvm.ptr
    %136 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %137 = llvm.insertvalue %128, %136[0] : !llvm.struct<(ptr, ptr, i64)> 
    %138 = llvm.insertvalue %135, %137[1] : !llvm.struct<(ptr, ptr, i64)> 
    %139 = llvm.mlir.constant(0 : index) : i64
    %140 = llvm.insertvalue %139, %138[2] : !llvm.struct<(ptr, ptr, i64)> 
    %141 = builtin.unrealized_conversion_cast %140 : !llvm.struct<(ptr, ptr, i64)> to memref<f64>
    %142 = llvm.extractvalue %140[1] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %121, %142 : f64, !llvm.ptr
    llvm.call @__catalyst__rt__qubit_release_array(%31) : (!llvm.ptr) -> ()
    llvm.call @__catalyst__rt__device_release() : () -> ()
    llvm.return %140 : !llvm.struct<(ptr, ptr, i64)>
  }
  llvm.func @setup() {
    %0 = llvm.mlir.zero : !llvm.ptr
    llvm.call @__catalyst__rt__initialize(%0) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @teardown() {
    llvm.call @__catalyst__rt__finalize() : () -> ()
    llvm.return
  }
}