; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}" = internal constant [54 x i8] c"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}\00"
@LightningSimulator = internal constant [19 x i8] c"LightningSimulator\00"
@"/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib" = internal constant [118 x i8] c"/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib\00"
@__constant_xf64 = private constant double 1.000000e+00, align 64

declare void @__catalyst__rt__finalize()

declare void @__catalyst__rt__initialize(ptr)

declare void @__catalyst__rt__device_release()

declare void @__catalyst__rt__qubit_release_array(ptr)

declare double @__catalyst__qis__Expval(i64)

declare i64 @__catalyst__qis__NamedObs(i64, ptr)

declare void @__catalyst__qis__Hadamard(ptr, ptr)

declare void @__catalyst__qis__RX(double, ptr, ptr)

declare ptr @__catalyst__rt__array_get_element_ptr_1d(ptr, i64)

declare ptr @__catalyst__rt__qubit_allocate_array(i64)

declare void @__catalyst__rt__device_init(ptr, ptr, ptr, i64, i1)

declare ptr @_mlir_memref_to_llvm_alloc(i64)

define { ptr, ptr, i64 } @jit_test_cancel_inverses_keep_original_workflow0() {
  %1 = call { ptr, ptr, i64 } @f_0(ptr inttoptr (i64 3735928559 to ptr), ptr @__constant_xf64, i64 0)
  %2 = extractvalue { ptr, ptr, i64 } %1, 0
  %3 = ptrtoint ptr %2 to i64
  %4 = icmp eq i64 3735928559, %3
  br i1 %4, label %5, label %13

5:                                                ; preds = %0
  %6 = call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %7 = insertvalue { ptr, ptr, i64 } poison, ptr %6, 0
  %8 = insertvalue { ptr, ptr, i64 } %7, ptr %6, 1
  %9 = insertvalue { ptr, ptr, i64 } %8, i64 0, 2
  %10 = extractvalue { ptr, ptr, i64 } %1, 1
  %11 = extractvalue { ptr, ptr, i64 } %1, 2
  %12 = getelementptr inbounds double, ptr %10, i64 %11
  call void @llvm.memcpy.p0.p0.i64(ptr %6, ptr %12, i64 8, i1 false)
  br label %14

13:                                               ; preds = %0
  br label %14

14:                                               ; preds = %5, %13
  %15 = phi { ptr, ptr, i64 } [ %1, %13 ], [ %9, %5 ]
  br label %16

16:                                               ; preds = %14
  ret { ptr, ptr, i64 } %15
}

define void @_catalyst_pyface_jit_test_cancel_inverses_keep_original_workflow0(ptr %0, ptr %1) {
  call void @_catalyst_ciface_jit_test_cancel_inverses_keep_original_workflow0(ptr %0)
  ret void
}

define void @_catalyst_ciface_jit_test_cancel_inverses_keep_original_workflow0(ptr %0) {
  %2 = call { ptr, ptr, i64 } @jit_test_cancel_inverses_keep_original_workflow0()
  store { ptr, ptr, i64 } %2, ptr %0, align 8
  ret void
}

define internal { ptr, ptr, i64 } @f_0(ptr %0, ptr %1, i64 %2) {
  call void @__catalyst__rt__device_init(ptr @"/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", ptr @LightningSimulator, ptr @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", i64 0, i1 false)
  %4 = call ptr @__catalyst__rt__qubit_allocate_array(i64 1)
  %5 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 0)
  %6 = load ptr, ptr %5, align 8
  %7 = load double, ptr %1, align 8
  call void @__catalyst__qis__RX(double %7, ptr %6, ptr null)
  call void @__catalyst__qis__Hadamard(ptr %6, ptr null)
  call void @__catalyst__qis__Hadamard(ptr %6, ptr null)
  %8 = call i64 @__catalyst__qis__NamedObs(i64 3, ptr %6)
  %9 = call double @__catalyst__qis__Expval(i64 %8)
  %10 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %11 = ptrtoint ptr %10 to i64
  %12 = add i64 %11, 63
  %13 = urem i64 %12, 64
  %14 = sub i64 %12, %13
  %15 = inttoptr i64 %14 to ptr
  %16 = insertvalue { ptr, ptr, i64 } poison, ptr %10, 0
  %17 = insertvalue { ptr, ptr, i64 } %16, ptr %15, 1
  %18 = insertvalue { ptr, ptr, i64 } %17, i64 0, 2
  store double %9, ptr %15, align 8
  call void @__catalyst__rt__qubit_release_array(ptr %4)
  call void @__catalyst__rt__device_release()
  ret { ptr, ptr, i64 } %18
}

define void @setup() {
  call void @__catalyst__rt__initialize(ptr null)
  ret void
}

define void @teardown() {
  call void @__catalyst__rt__finalize()
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
