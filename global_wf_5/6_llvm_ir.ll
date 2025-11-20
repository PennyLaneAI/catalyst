; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}" = internal constant [54 x i8] c"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}\00"
@LightningSimulator = internal constant [19 x i8] c"LightningSimulator\00"
@"/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib" = internal constant [118 x i8] c"/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib\00"
@__constant_xf64 = private constant double 1.200000e+00, align 64

declare void @__catalyst__rt__finalize()

declare void @__catalyst__rt__initialize(ptr)

declare void @__catalyst__rt__device_release()

declare void @__catalyst__rt__qubit_release_array(ptr)

declare double @__catalyst__qis__Expval(i64)

declare i64 @__catalyst__qis__NamedObs(i64, ptr)

declare void @__catalyst__qis__RX(double, ptr, ptr)

declare ptr @__catalyst__rt__array_get_element_ptr_1d(ptr, i64)

declare ptr @__catalyst__rt__qubit_allocate_array(i64)

declare void @__catalyst__rt__device_init(ptr, ptr, ptr, i64, i1)

declare ptr @_mlir_memref_to_llvm_alloc(i64)

define { { ptr, ptr, i64 }, { ptr, ptr, i64 } } @jit_global_wf() {
  %1 = call { ptr, ptr, i64 } @g_0(ptr inttoptr (i64 3735928559 to ptr), ptr @__constant_xf64, i64 0)
  %2 = call { ptr, ptr, i64 } @h_0(ptr inttoptr (i64 3735928559 to ptr), ptr @__constant_xf64, i64 0)
  %3 = extractvalue { ptr, ptr, i64 } %1, 0
  %4 = ptrtoint ptr %3 to i64
  %5 = icmp eq i64 3735928559, %4
  br i1 %5, label %6, label %14

6:                                                ; preds = %0
  %7 = call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %8 = insertvalue { ptr, ptr, i64 } poison, ptr %7, 0
  %9 = insertvalue { ptr, ptr, i64 } %8, ptr %7, 1
  %10 = insertvalue { ptr, ptr, i64 } %9, i64 0, 2
  %11 = extractvalue { ptr, ptr, i64 } %1, 1
  %12 = extractvalue { ptr, ptr, i64 } %1, 2
  %13 = getelementptr inbounds double, ptr %11, i64 %12
  call void @llvm.memcpy.p0.p0.i64(ptr %7, ptr %13, i64 8, i1 false)
  br label %15

14:                                               ; preds = %0
  br label %15

15:                                               ; preds = %6, %14
  %16 = phi { ptr, ptr, i64 } [ %1, %14 ], [ %10, %6 ]
  br label %17

17:                                               ; preds = %15
  %18 = extractvalue { ptr, ptr, i64 } %2, 0
  %19 = ptrtoint ptr %18 to i64
  %20 = icmp eq i64 3735928559, %19
  br i1 %20, label %21, label %29

21:                                               ; preds = %17
  %22 = call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %23 = insertvalue { ptr, ptr, i64 } poison, ptr %22, 0
  %24 = insertvalue { ptr, ptr, i64 } %23, ptr %22, 1
  %25 = insertvalue { ptr, ptr, i64 } %24, i64 0, 2
  %26 = extractvalue { ptr, ptr, i64 } %2, 1
  %27 = extractvalue { ptr, ptr, i64 } %2, 2
  %28 = getelementptr inbounds double, ptr %26, i64 %27
  call void @llvm.memcpy.p0.p0.i64(ptr %22, ptr %28, i64 8, i1 false)
  br label %30

29:                                               ; preds = %17
  br label %30

30:                                               ; preds = %21, %29
  %31 = phi { ptr, ptr, i64 } [ %2, %29 ], [ %25, %21 ]
  br label %32

32:                                               ; preds = %30
  %33 = insertvalue { { ptr, ptr, i64 }, { ptr, ptr, i64 } } poison, { ptr, ptr, i64 } %16, 0
  %34 = insertvalue { { ptr, ptr, i64 }, { ptr, ptr, i64 } } %33, { ptr, ptr, i64 } %31, 1
  ret { { ptr, ptr, i64 }, { ptr, ptr, i64 } } %34
}

define void @_catalyst_pyface_jit_global_wf(ptr %0, ptr %1) {
  call void @_catalyst_ciface_jit_global_wf(ptr %0)
  ret void
}

define void @_catalyst_ciface_jit_global_wf(ptr %0) {
  %2 = call { { ptr, ptr, i64 }, { ptr, ptr, i64 } } @jit_global_wf()
  store { { ptr, ptr, i64 }, { ptr, ptr, i64 } } %2, ptr %0, align 8
  ret void
}

define internal { ptr, ptr, i64 } @g_0(ptr %0, ptr %1, i64 %2) {
  call void @__catalyst__rt__device_init(ptr @"/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", ptr @LightningSimulator, ptr @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", i64 0, i1 false)
  %4 = call ptr @__catalyst__rt__qubit_allocate_array(i64 2)
  %5 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 0)
  %6 = load ptr, ptr %5, align 8
  %7 = load double, ptr %1, align 8
  call void @__catalyst__qis__RX(double %7, ptr %6, ptr null)
  %8 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 1)
  %9 = call i64 @__catalyst__qis__NamedObs(i64 2, ptr %6)
  %10 = call double @__catalyst__qis__Expval(i64 %9)
  %11 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %12 = ptrtoint ptr %11 to i64
  %13 = add i64 %12, 63
  %14 = urem i64 %13, 64
  %15 = sub i64 %13, %14
  %16 = inttoptr i64 %15 to ptr
  %17 = insertvalue { ptr, ptr, i64 } poison, ptr %11, 0
  %18 = insertvalue { ptr, ptr, i64 } %17, ptr %16, 1
  %19 = insertvalue { ptr, ptr, i64 } %18, i64 0, 2
  store double %10, ptr %16, align 8
  call void @__catalyst__rt__qubit_release_array(ptr %4)
  call void @__catalyst__rt__device_release()
  ret { ptr, ptr, i64 } %19
}

define internal { ptr, ptr, i64 } @h_0(ptr %0, ptr %1, i64 %2) {
  call void @__catalyst__rt__device_init(ptr @"/Users/christina/Prog/catalyst_env/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", ptr @LightningSimulator, ptr @"{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", i64 0, i1 false)
  %4 = call ptr @__catalyst__rt__qubit_allocate_array(i64 2)
  %5 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 0)
  %6 = load ptr, ptr %5, align 8
  %7 = load double, ptr %1, align 8
  call void @__catalyst__qis__RX(double %7, ptr %6, ptr null)
  %8 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 1)
  %9 = call i64 @__catalyst__qis__NamedObs(i64 2, ptr %6)
  %10 = call double @__catalyst__qis__Expval(i64 %9)
  %11 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %12 = ptrtoint ptr %11 to i64
  %13 = add i64 %12, 63
  %14 = urem i64 %13, 64
  %15 = sub i64 %13, %14
  %16 = inttoptr i64 %15 to ptr
  %17 = insertvalue { ptr, ptr, i64 } poison, ptr %11, 0
  %18 = insertvalue { ptr, ptr, i64 } %17, ptr %16, 1
  %19 = insertvalue { ptr, ptr, i64 } %18, i64 0, 2
  store double %10, ptr %16, align 8
  call void @__catalyst__rt__qubit_release_array(ptr %4)
  call void @__catalyst__rt__device_release()
  ret { ptr, ptr, i64 } %19
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
