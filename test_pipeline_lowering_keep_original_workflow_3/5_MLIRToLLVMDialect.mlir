module @test_pipeline_lowering_keep_original_workflow {
  llvm.func @__catalyst__rt__finalize()
  llvm.func @__catalyst__rt__initialize(!llvm.ptr)
  llvm.func @__catalyst__rt__device_release()
  llvm.func @__catalyst__rt__qubit_release_array(!llvm.ptr)
  llvm.func @__catalyst__qis__Expval(i64) -> f64
  llvm.func @__catalyst__qis__NamedObs(i64, !llvm.ptr) -> i64
  llvm.func @__catalyst__qis__Hadamard(!llvm.ptr, !llvm.ptr)
  llvm.func @__catalyst__qis__RX(f64, !llvm.ptr, !llvm.ptr)
  llvm.func @__catalyst__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @__catalyst__rt__qubit_allocate_array(i64) -> !llvm.ptr
  llvm.mlir.global internal constant @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"("{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @LightningSimulator("LightningSimulator\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @"/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib"("/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib\00") {addr_space = 0 : i32}
  llvm.func @__catalyst__rt__device_init(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i1)
  llvm.func @_mlir_memref_to_llvm_alloc(i64) -> !llvm.ptr
  llvm.func @jit_test_pipeline_lowering_keep_original_workflow(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64) -> !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>)> attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.mlir.poison : !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>)>
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(3735928559 : index) : i64
    %5 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %6 = llvm.call @f_0(%arg0, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(ptr, ptr, i64)>
    %7 = llvm.call @f_1_0(%arg0, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(ptr, ptr, i64)>
    %8 = llvm.extractvalue %6[0] : !llvm.struct<(ptr, ptr, i64)> 
    %9 = llvm.ptrtoint %8 : !llvm.ptr to i64
    %10 = llvm.icmp "eq" %4, %9 : i64
    llvm.cond_br %10, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %11 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %12 = llvm.ptrtoint %11 : !llvm.ptr to i64
    %13 = llvm.call @_mlir_memref_to_llvm_alloc(%12) : (i64) -> !llvm.ptr
    %14 = llvm.insertvalue %13, %5[0] : !llvm.struct<(ptr, ptr, i64)> 
    %15 = llvm.insertvalue %13, %14[1] : !llvm.struct<(ptr, ptr, i64)> 
    %16 = llvm.insertvalue %1, %15[2] : !llvm.struct<(ptr, ptr, i64)> 
    %17 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %18 = llvm.ptrtoint %17 : !llvm.ptr to i64
    %19 = llvm.mul %18, %3 : i64
    %20 = llvm.extractvalue %6[1] : !llvm.struct<(ptr, ptr, i64)> 
    %21 = llvm.extractvalue %6[2] : !llvm.struct<(ptr, ptr, i64)> 
    %22 = llvm.getelementptr inbounds %20[%21] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%13, %22, %19) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb3(%16 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb3(%6 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb3(%23: !llvm.struct<(ptr, ptr, i64)>):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    %24 = llvm.extractvalue %7[0] : !llvm.struct<(ptr, ptr, i64)> 
    %25 = llvm.ptrtoint %24 : !llvm.ptr to i64
    %26 = llvm.icmp "eq" %4, %25 : i64
    llvm.cond_br %26, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %27 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %28 = llvm.ptrtoint %27 : !llvm.ptr to i64
    %29 = llvm.call @_mlir_memref_to_llvm_alloc(%28) : (i64) -> !llvm.ptr
    %30 = llvm.insertvalue %29, %5[0] : !llvm.struct<(ptr, ptr, i64)> 
    %31 = llvm.insertvalue %29, %30[1] : !llvm.struct<(ptr, ptr, i64)> 
    %32 = llvm.insertvalue %1, %31[2] : !llvm.struct<(ptr, ptr, i64)> 
    %33 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %34 = llvm.ptrtoint %33 : !llvm.ptr to i64
    %35 = llvm.mul %34, %3 : i64
    %36 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64)> 
    %37 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64)> 
    %38 = llvm.getelementptr inbounds %36[%37] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%29, %38, %35) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb7(%32 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb6:  // pred: ^bb4
    llvm.br ^bb7(%7 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb7(%39: !llvm.struct<(ptr, ptr, i64)>):  // 2 preds: ^bb5, ^bb6
    llvm.br ^bb8
  ^bb8:  // pred: ^bb7
    %40 = llvm.insertvalue %23, %0[0] : !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>)> 
    %41 = llvm.insertvalue %39, %40[1] : !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>)> 
    llvm.return %41 : !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>)>
  }
  llvm.func @_catalyst_pyface_jit_test_pipeline_lowering_keep_original_workflow(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr)> 
    llvm.call @_catalyst_ciface_jit_test_pipeline_lowering_keep_original_workflow(%arg0, %1) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_catalyst_ciface_jit_test_pipeline_lowering_keep_original_workflow(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64)> 
    %4 = llvm.call @jit_test_pipeline_lowering_keep_original_workflow(%1, %2, %3) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>)>
    llvm.store %4, %arg0 : !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>)>, !llvm.ptr
    llvm.return
  }
  llvm.func internal @f_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64) -> !llvm.struct<(ptr, ptr, i64)> attributes {diff_method = "adjoint", qnode} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(64 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = llvm.mlir.constant(2 : i64) : i64
    %6 = llvm.mlir.constant(false) : i1
    %7 = llvm.mlir.addressof @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}" : !llvm.ptr
    %8 = llvm.mlir.addressof @LightningSimulator : !llvm.ptr
    %9 = llvm.mlir.addressof @"/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib" : !llvm.ptr
    %10 = llvm.mlir.constant(0 : i64) : i64
    %11 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %12 = llvm.getelementptr inbounds %9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<118 x i8>
    %13 = llvm.getelementptr inbounds %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<19 x i8>
    %14 = llvm.getelementptr inbounds %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<54 x i8>
    llvm.call @__catalyst__rt__device_init(%12, %13, %14, %10, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i1) -> ()
    %15 = llvm.call @__catalyst__rt__qubit_allocate_array(%5) : (i64) -> !llvm.ptr
    %16 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%15, %10) : (!llvm.ptr, i64) -> !llvm.ptr
    %17 = llvm.load %16 : !llvm.ptr -> !llvm.ptr
    %18 = llvm.load %arg1 : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RX(%18, %17, %4) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %19 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%15, %3) : (!llvm.ptr, i64) -> !llvm.ptr
    %20 = llvm.load %19 : !llvm.ptr -> !llvm.ptr
    llvm.call @__catalyst__qis__Hadamard(%20, %4) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__Hadamard(%20, %4) : (!llvm.ptr, !llvm.ptr) -> ()
    %21 = llvm.call @__catalyst__qis__NamedObs(%5, %17) : (i64, !llvm.ptr) -> i64
    %22 = llvm.call @__catalyst__qis__Expval(%21) : (i64) -> f64
    %23 = llvm.getelementptr %4[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %24 = llvm.ptrtoint %23 : !llvm.ptr to i64
    %25 = llvm.add %24, %1 : i64
    %26 = llvm.call @_mlir_memref_to_llvm_alloc(%25) : (i64) -> !llvm.ptr
    %27 = llvm.ptrtoint %26 : !llvm.ptr to i64
    %28 = llvm.sub %1, %2 : i64
    %29 = llvm.add %27, %28 : i64
    %30 = llvm.urem %29, %1 : i64
    %31 = llvm.sub %29, %30 : i64
    %32 = llvm.inttoptr %31 : i64 to !llvm.ptr
    %33 = llvm.insertvalue %26, %11[0] : !llvm.struct<(ptr, ptr, i64)> 
    %34 = llvm.insertvalue %32, %33[1] : !llvm.struct<(ptr, ptr, i64)> 
    %35 = llvm.insertvalue %0, %34[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %22, %32 : f64, !llvm.ptr
    llvm.call @__catalyst__rt__qubit_release_array(%15) : (!llvm.ptr) -> ()
    llvm.call @__catalyst__rt__device_release() : () -> ()
    llvm.return %35 : !llvm.struct<(ptr, ptr, i64)>
  }
  llvm.func internal @f_1_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64) -> !llvm.struct<(ptr, ptr, i64)> attributes {diff_method = "adjoint", qnode} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(64 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = llvm.mlir.constant(2 : i64) : i64
    %6 = llvm.mlir.constant(false) : i1
    %7 = llvm.mlir.addressof @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}" : !llvm.ptr
    %8 = llvm.mlir.addressof @LightningSimulator : !llvm.ptr
    %9 = llvm.mlir.addressof @"/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib" : !llvm.ptr
    %10 = llvm.mlir.constant(0 : i64) : i64
    %11 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %12 = llvm.getelementptr inbounds %9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<118 x i8>
    %13 = llvm.getelementptr inbounds %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<19 x i8>
    %14 = llvm.getelementptr inbounds %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<54 x i8>
    llvm.call @__catalyst__rt__device_init(%12, %13, %14, %10, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i1) -> ()
    %15 = llvm.call @__catalyst__rt__qubit_allocate_array(%5) : (i64) -> !llvm.ptr
    %16 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%15, %10) : (!llvm.ptr, i64) -> !llvm.ptr
    %17 = llvm.load %16 : !llvm.ptr -> !llvm.ptr
    %18 = llvm.load %arg1 : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RX(%18, %17, %4) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %19 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%15, %3) : (!llvm.ptr, i64) -> !llvm.ptr
    %20 = llvm.call @__catalyst__qis__NamedObs(%5, %17) : (i64, !llvm.ptr) -> i64
    %21 = llvm.call @__catalyst__qis__Expval(%20) : (i64) -> f64
    %22 = llvm.getelementptr %4[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %23 = llvm.ptrtoint %22 : !llvm.ptr to i64
    %24 = llvm.add %23, %1 : i64
    %25 = llvm.call @_mlir_memref_to_llvm_alloc(%24) : (i64) -> !llvm.ptr
    %26 = llvm.ptrtoint %25 : !llvm.ptr to i64
    %27 = llvm.sub %1, %2 : i64
    %28 = llvm.add %26, %27 : i64
    %29 = llvm.urem %28, %1 : i64
    %30 = llvm.sub %28, %29 : i64
    %31 = llvm.inttoptr %30 : i64 to !llvm.ptr
    %32 = llvm.insertvalue %25, %11[0] : !llvm.struct<(ptr, ptr, i64)> 
    %33 = llvm.insertvalue %31, %32[1] : !llvm.struct<(ptr, ptr, i64)> 
    %34 = llvm.insertvalue %0, %33[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %21, %31 : f64, !llvm.ptr
    llvm.call @__catalyst__rt__qubit_release_array(%15) : (!llvm.ptr) -> ()
    llvm.call @__catalyst__rt__device_release() : () -> ()
    llvm.return %34 : !llvm.struct<(ptr, ptr, i64)>
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