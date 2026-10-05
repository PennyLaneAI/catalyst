; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@"{'track_resources': False}" = internal constant [27 x i8] c"{'track_resources': False}\00"
@NullQubit = internal constant [10 x i8] c"NullQubit\00"
@"/Users/haider.sajjad/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_null_qubit.dylib" = internal constant [105 x i8] c"/Users/haider.sajjad/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_null_qubit.dylib\00"
@__constant_3xi64 = private constant [3 x i64] [i64 0, i64 1, i64 2], align 64

declare void @__catalyst__rt__finalize()

declare void @__catalyst__rt__initialize(ptr)

declare void @__catalyst__rt__device_release()

declare void @__catalyst__rt__qubit_release_array(ptr)

declare double @__catalyst__qis__Expval(i64)

declare i64 @__catalyst__qis__NamedObs(i64, ptr)

declare void @__catalyst__qis__T(ptr, ptr)

declare void @__catalyst__qis__CNOT(ptr, ptr, ptr)

declare void @__catalyst__qis__Hadamard(ptr, ptr)

declare ptr @__catalyst__rt__array_get_element_ptr_1d(ptr, i64)

declare ptr @__catalyst__rt__qubit_allocate_array(i64)

declare void @__catalyst__rt__device_init(ptr, ptr, ptr, i64, i1)

declare ptr @_mlir_memref_to_llvm_alloc(i64)

define { ptr, ptr, i64 } @jit_qfunc() {
  %1 = call { ptr, ptr, i64 } @qfunc_0()
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

define void @_catalyst_pyface_jit_qfunc(ptr %0, ptr %1) {
  call void @_catalyst_ciface_jit_qfunc(ptr %0)
  ret void
}

define void @_catalyst_ciface_jit_qfunc(ptr %0) {
  %2 = call { ptr, ptr, i64 } @jit_qfunc()
  store { ptr, ptr, i64 } %2, ptr %0, align 8
  ret void
}

define internal { ptr, ptr, i64 } @qfunc_0() {
  %1 = alloca { i1, i64, ptr, ptr }, i64 1, align 8
  %2 = alloca { i1, i64, ptr, ptr }, i64 1, align 8
  %3 = alloca { i1, i64, ptr, ptr }, i64 1, align 8
  call void @__catalyst__rt__device_init(ptr @"/Users/haider.sajjad/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_null_qubit.dylib", ptr @NullQubit, ptr @"{'track_resources': False}", i64 0, i1 false)
  %4 = call ptr @__catalyst__rt__qubit_allocate_array(i64 3)
  %5 = load i64, ptr getelementptr inbounds nuw (i8, ptr @__constant_3xi64, i64 16), align 4
  %6 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 %5)
  %7 = load ptr, ptr %6, align 8
  call void @__catalyst__qis__Hadamard(ptr %7, ptr null)
  %8 = load i64, ptr getelementptr inbounds nuw (i8, ptr @__constant_3xi64, i64 8), align 4
  %9 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 %8)
  %10 = load ptr, ptr %9, align 8
  call void @__catalyst__qis__CNOT(ptr %10, ptr %7, ptr null)
  %11 = getelementptr inbounds { i1, i64, ptr, ptr }, ptr %3, i32 0, i32 0
  %12 = getelementptr inbounds { i1, i64, ptr, ptr }, ptr %3, i32 0, i32 1
  %13 = getelementptr inbounds { i1, i64, ptr, ptr }, ptr %3, i32 0, i32 2
  %14 = getelementptr inbounds { i1, i64, ptr, ptr }, ptr %3, i32 0, i32 3
  store i1 true, ptr %11, align 1
  store i64 0, ptr %12, align 4
  store ptr null, ptr %13, align 8
  store ptr null, ptr %14, align 8
  call void @__catalyst__qis__T(ptr %7, ptr %3)
  %15 = load i64, ptr @__constant_3xi64, align 4
  %16 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 %15)
  %17 = load ptr, ptr %16, align 8
  call void @__catalyst__qis__CNOT(ptr %17, ptr %7, ptr null)
  call void @__catalyst__qis__T(ptr %7, ptr null)
  %18 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 %8)
  %19 = load ptr, ptr %18, align 8
  call void @__catalyst__qis__CNOT(ptr %19, ptr %7, ptr null)
  %20 = getelementptr inbounds { i1, i64, ptr, ptr }, ptr %2, i32 0, i32 0
  %21 = getelementptr inbounds { i1, i64, ptr, ptr }, ptr %2, i32 0, i32 1
  %22 = getelementptr inbounds { i1, i64, ptr, ptr }, ptr %2, i32 0, i32 2
  %23 = getelementptr inbounds { i1, i64, ptr, ptr }, ptr %2, i32 0, i32 3
  store i1 true, ptr %20, align 1
  store i64 0, ptr %21, align 4
  store ptr null, ptr %22, align 8
  store ptr null, ptr %23, align 8
  call void @__catalyst__qis__T(ptr %7, ptr %2)
  %24 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 %15)
  %25 = load ptr, ptr %24, align 8
  call void @__catalyst__qis__CNOT(ptr %25, ptr %7, ptr null)
  call void @__catalyst__qis__T(ptr %7, ptr null)
  %26 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 %8)
  %27 = load ptr, ptr %26, align 8
  call void @__catalyst__qis__T(ptr %27, ptr null)
  %28 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 %15)
  %29 = load ptr, ptr %28, align 8
  call void @__catalyst__qis__CNOT(ptr %29, ptr %27, ptr null)
  %30 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 %5)
  %31 = load ptr, ptr %30, align 8
  call void @__catalyst__qis__Hadamard(ptr %31, ptr null)
  %32 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 %15)
  %33 = load ptr, ptr %32, align 8
  call void @__catalyst__qis__T(ptr %33, ptr null)
  %34 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 %8)
  %35 = load ptr, ptr %34, align 8
  %36 = getelementptr inbounds { i1, i64, ptr, ptr }, ptr %1, i32 0, i32 0
  %37 = getelementptr inbounds { i1, i64, ptr, ptr }, ptr %1, i32 0, i32 1
  %38 = getelementptr inbounds { i1, i64, ptr, ptr }, ptr %1, i32 0, i32 2
  %39 = getelementptr inbounds { i1, i64, ptr, ptr }, ptr %1, i32 0, i32 3
  store i1 true, ptr %36, align 1
  store i64 0, ptr %37, align 4
  store ptr null, ptr %38, align 8
  store ptr null, ptr %39, align 8
  call void @__catalyst__qis__T(ptr %35, ptr %1)
  call void @__catalyst__qis__CNOT(ptr %33, ptr %35, ptr null)
  %40 = call ptr @__catalyst__rt__array_get_element_ptr_1d(ptr %4, i64 0)
  %41 = load ptr, ptr %40, align 8
  %42 = call i64 @__catalyst__qis__NamedObs(i64 3, ptr %41)
  %43 = call double @__catalyst__qis__Expval(i64 %42)
  %44 = call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %45 = ptrtoint ptr %44 to i64
  %46 = add i64 %45, 63
  %47 = urem i64 %46, 64
  %48 = sub i64 %46, %47
  %49 = inttoptr i64 %48 to ptr
  %50 = insertvalue { ptr, ptr, i64 } poison, ptr %44, 0
  %51 = insertvalue { ptr, ptr, i64 } %50, ptr %49, 1
  %52 = insertvalue { ptr, ptr, i64 } %51, i64 0, 2
  store double %43, ptr %49, align 8
  call void @__catalyst__rt__qubit_release_array(ptr %4)
  call void @__catalyst__rt__device_release()
  ret { ptr, ptr, i64 } %52
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
