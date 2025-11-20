module @test_cancel_inverses_lowering_transform_applied_workflow {
  llvm.func @__catalyst__rt__finalize()
  llvm.func @__catalyst__rt__initialize(!llvm.ptr)
  llvm.func @__catalyst__qis__Hadamard(!llvm.ptr, !llvm.ptr)
  llvm.func @__catalyst__rt__device_release()
  llvm.func @__catalyst__rt__qubit_release_array(!llvm.ptr)
  llvm.func @__catalyst__qis__Expval(i64) -> f64
  llvm.func @__catalyst__qis__NamedObs(i64, !llvm.ptr) -> i64
  llvm.func @__catalyst__qis__RX(f64, !llvm.ptr, !llvm.ptr)
  llvm.func @__catalyst__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @__catalyst__rt__qubit_allocate_array(i64) -> !llvm.ptr
  llvm.mlir.global internal constant @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"("{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @LightningSimulator("LightningSimulator\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @"/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib"("/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib\00") {addr_space = 0 : i32}
  llvm.func @__catalyst__rt__device_init(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i1)
  llvm.func @_mlir_memref_to_llvm_alloc(i64) -> !llvm.ptr
  llvm.func @jit_test_cancel_inverses_lowering_transform_applied_workflow(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64) -> !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>)> attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.mlir.poison : !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>)>
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(3735928559 : index) : i64
    %5 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %6 = llvm.call @f_0(%arg0, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(ptr, ptr, i64)>
    %7 = llvm.call @g_0(%arg0, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(ptr, ptr, i64)>
    %8 = llvm.call @h_0(%arg0, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(ptr, ptr, i64)>
    %9 = llvm.extractvalue %6[0] : !llvm.struct<(ptr, ptr, i64)> 
    %10 = llvm.ptrtoint %9 : !llvm.ptr to i64
    %11 = llvm.icmp "eq" %4, %10 : i64
    llvm.cond_br %11, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %12 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %13 = llvm.ptrtoint %12 : !llvm.ptr to i64
    %14 = llvm.call @_mlir_memref_to_llvm_alloc(%13) : (i64) -> !llvm.ptr
    %15 = llvm.insertvalue %14, %5[0] : !llvm.struct<(ptr, ptr, i64)> 
    %16 = llvm.insertvalue %14, %15[1] : !llvm.struct<(ptr, ptr, i64)> 
    %17 = llvm.insertvalue %1, %16[2] : !llvm.struct<(ptr, ptr, i64)> 
    %18 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %19 = llvm.ptrtoint %18 : !llvm.ptr to i64
    %20 = llvm.mul %19, %3 : i64
    %21 = llvm.extractvalue %6[1] : !llvm.struct<(ptr, ptr, i64)> 
    %22 = llvm.extractvalue %6[2] : !llvm.struct<(ptr, ptr, i64)> 
    %23 = llvm.getelementptr inbounds %21[%22] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%14, %23, %20) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb3(%17 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb3(%6 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb3(%24: !llvm.struct<(ptr, ptr, i64)>):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    %25 = llvm.extractvalue %7[0] : !llvm.struct<(ptr, ptr, i64)> 
    %26 = llvm.ptrtoint %25 : !llvm.ptr to i64
    %27 = llvm.icmp "eq" %4, %26 : i64
    llvm.cond_br %27, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %28 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.call @_mlir_memref_to_llvm_alloc(%29) : (i64) -> !llvm.ptr
    %31 = llvm.insertvalue %30, %5[0] : !llvm.struct<(ptr, ptr, i64)> 
    %32 = llvm.insertvalue %30, %31[1] : !llvm.struct<(ptr, ptr, i64)> 
    %33 = llvm.insertvalue %1, %32[2] : !llvm.struct<(ptr, ptr, i64)> 
    %34 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %35 = llvm.ptrtoint %34 : !llvm.ptr to i64
    %36 = llvm.mul %35, %3 : i64
    %37 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64)> 
    %38 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64)> 
    %39 = llvm.getelementptr inbounds %37[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%30, %39, %36) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb7(%33 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb6:  // pred: ^bb4
    llvm.br ^bb7(%7 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb7(%40: !llvm.struct<(ptr, ptr, i64)>):  // 2 preds: ^bb5, ^bb6
    llvm.br ^bb8
  ^bb8:  // pred: ^bb7
    %41 = llvm.extractvalue %8[0] : !llvm.struct<(ptr, ptr, i64)> 
    %42 = llvm.ptrtoint %41 : !llvm.ptr to i64
    %43 = llvm.icmp "eq" %4, %42 : i64
    llvm.cond_br %43, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    %44 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %45 = llvm.ptrtoint %44 : !llvm.ptr to i64
    %46 = llvm.call @_mlir_memref_to_llvm_alloc(%45) : (i64) -> !llvm.ptr
    %47 = llvm.insertvalue %46, %5[0] : !llvm.struct<(ptr, ptr, i64)> 
    %48 = llvm.insertvalue %46, %47[1] : !llvm.struct<(ptr, ptr, i64)> 
    %49 = llvm.insertvalue %1, %48[2] : !llvm.struct<(ptr, ptr, i64)> 
    %50 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %51 = llvm.ptrtoint %50 : !llvm.ptr to i64
    %52 = llvm.mul %51, %3 : i64
    %53 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64)> 
    %54 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64)> 
    %55 = llvm.getelementptr inbounds %53[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    "llvm.intr.memcpy"(%46, %55, %52) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb11(%49 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb10:  // pred: ^bb8
    llvm.br ^bb11(%8 : !llvm.struct<(ptr, ptr, i64)>)
  ^bb11(%56: !llvm.struct<(ptr, ptr, i64)>):  // 2 preds: ^bb9, ^bb10
    llvm.br ^bb12
  ^bb12:  // pred: ^bb11
    %57 = llvm.insertvalue %24, %0[0] : !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>)> 
    %58 = llvm.insertvalue %40, %57[1] : !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>)> 
    %59 = llvm.insertvalue %56, %58[2] : !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>)> 
    llvm.return %59 : !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>)>
  }
  llvm.func @_catalyst_pyface_jit_test_cancel_inverses_lowering_transform_applied_workflow(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr)> 
    llvm.call @_catalyst_ciface_jit_test_cancel_inverses_lowering_transform_applied_workflow(%arg0, %1) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_catalyst_ciface_jit_test_cancel_inverses_lowering_transform_applied_workflow(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64)> 
    %4 = llvm.call @jit_test_cancel_inverses_lowering_transform_applied_workflow(%1, %2, %3) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>)>
    llvm.store %4, %arg0 : !llvm.struct<(struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>, struct<(ptr, ptr, i64)>)>, !llvm.ptr
    llvm.return
  }
  llvm.func internal @f_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64) -> !llvm.struct<(ptr, ptr, i64)> attributes {diff_method = "adjoint", qnode} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(64 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(3 : i64) : i64
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = llvm.mlir.constant(1 : i64) : i64
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
    %19 = llvm.call @__catalyst__qis__NamedObs(%3, %17) : (i64, !llvm.ptr) -> i64
    %20 = llvm.call @__catalyst__qis__Expval(%19) : (i64) -> f64
    %21 = llvm.getelementptr %4[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %22 = llvm.ptrtoint %21 : !llvm.ptr to i64
    %23 = llvm.add %22, %1 : i64
    %24 = llvm.call @_mlir_memref_to_llvm_alloc(%23) : (i64) -> !llvm.ptr
    %25 = llvm.ptrtoint %24 : !llvm.ptr to i64
    %26 = llvm.sub %1, %2 : i64
    %27 = llvm.add %25, %26 : i64
    %28 = llvm.urem %27, %1 : i64
    %29 = llvm.sub %27, %28 : i64
    %30 = llvm.inttoptr %29 : i64 to !llvm.ptr
    %31 = llvm.insertvalue %24, %11[0] : !llvm.struct<(ptr, ptr, i64)> 
    %32 = llvm.insertvalue %30, %31[1] : !llvm.struct<(ptr, ptr, i64)> 
    %33 = llvm.insertvalue %0, %32[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %20, %30 : f64, !llvm.ptr
    llvm.call @__catalyst__rt__qubit_release_array(%15) : (!llvm.ptr) -> ()
    llvm.call @__catalyst__rt__device_release() : () -> ()
    llvm.return %33 : !llvm.struct<(ptr, ptr, i64)>
  }
  llvm.func internal @g_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64) -> !llvm.struct<(ptr, ptr, i64)> attributes {diff_method = "adjoint", qnode} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(64 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(3 : i64) : i64
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = llvm.mlir.constant(1 : i64) : i64
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
    llvm.call @__catalyst__qis__Hadamard(%17, %4) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__Hadamard(%17, %4) : (!llvm.ptr, !llvm.ptr) -> ()
    %19 = llvm.call @__catalyst__qis__NamedObs(%3, %17) : (i64, !llvm.ptr) -> i64
    %20 = llvm.call @__catalyst__qis__Expval(%19) : (i64) -> f64
    %21 = llvm.getelementptr %4[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %22 = llvm.ptrtoint %21 : !llvm.ptr to i64
    %23 = llvm.add %22, %1 : i64
    %24 = llvm.call @_mlir_memref_to_llvm_alloc(%23) : (i64) -> !llvm.ptr
    %25 = llvm.ptrtoint %24 : !llvm.ptr to i64
    %26 = llvm.sub %1, %2 : i64
    %27 = llvm.add %25, %26 : i64
    %28 = llvm.urem %27, %1 : i64
    %29 = llvm.sub %27, %28 : i64
    %30 = llvm.inttoptr %29 : i64 to !llvm.ptr
    %31 = llvm.insertvalue %24, %11[0] : !llvm.struct<(ptr, ptr, i64)> 
    %32 = llvm.insertvalue %30, %31[1] : !llvm.struct<(ptr, ptr, i64)> 
    %33 = llvm.insertvalue %0, %32[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %20, %30 : f64, !llvm.ptr
    llvm.call @__catalyst__rt__qubit_release_array(%15) : (!llvm.ptr) -> ()
    llvm.call @__catalyst__rt__device_release() : () -> ()
    llvm.return %33 : !llvm.struct<(ptr, ptr, i64)>
  }
  llvm.func internal @h_0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64) -> !llvm.struct<(ptr, ptr, i64)> attributes {diff_method = "adjoint", qnode} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(64 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(3 : i64) : i64
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = llvm.mlir.constant(1 : i64) : i64
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
    llvm.call @__catalyst__qis__Hadamard(%17, %4) : (!llvm.ptr, !llvm.ptr) -> ()
    %18 = llvm.load %arg1 : !llvm.ptr -> f64
    llvm.call @__catalyst__qis__RX(%18, %17, %4) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__catalyst__qis__Hadamard(%17, %4) : (!llvm.ptr, !llvm.ptr) -> ()
    %19 = llvm.call @__catalyst__qis__NamedObs(%3, %17) : (i64, !llvm.ptr) -> i64
    %20 = llvm.call @__catalyst__qis__Expval(%19) : (i64) -> f64
    %21 = llvm.getelementptr %4[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %22 = llvm.ptrtoint %21 : !llvm.ptr to i64
    %23 = llvm.add %22, %1 : i64
    %24 = llvm.call @_mlir_memref_to_llvm_alloc(%23) : (i64) -> !llvm.ptr
    %25 = llvm.ptrtoint %24 : !llvm.ptr to i64
    %26 = llvm.sub %1, %2 : i64
    %27 = llvm.add %25, %26 : i64
    %28 = llvm.urem %27, %1 : i64
    %29 = llvm.sub %27, %28 : i64
    %30 = llvm.inttoptr %29 : i64 to !llvm.ptr
    %31 = llvm.insertvalue %24, %11[0] : !llvm.struct<(ptr, ptr, i64)> 
    %32 = llvm.insertvalue %30, %31[1] : !llvm.struct<(ptr, ptr, i64)> 
    %33 = llvm.insertvalue %0, %32[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %20, %30 : f64, !llvm.ptr
    llvm.call @__catalyst__rt__qubit_release_array(%15) : (!llvm.ptr) -> ()
    llvm.call @__catalyst__rt__device_release() : () -> ()
    llvm.return %33 : !llvm.struct<(ptr, ptr, i64)>
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