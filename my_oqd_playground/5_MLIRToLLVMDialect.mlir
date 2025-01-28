module @f {
  llvm.func @__catalyst__oqd__greetings()
  llvm.func @__catalyst__rt__finalize()
  llvm.func @__catalyst__rt__initialize(!llvm.ptr)

  llvm.func @__catalyst__oqd__ion(!llvm.ptr) -> ()

  llvm.mlir.global internal constant @upstate("upstate\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @estate("estate\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @downstate("downstate\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @Yb171("Yb171\00") {addr_space = 0 : i32}

  llvm.func @ion_op() -> () {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.mlir.addressof @Yb171 : !llvm.ptr
    %2 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<6 x i8>
    %3 = llvm.mlir.constant(1.710000e+02 : f64) : f64
    %4 = llvm.mlir.constant(56.110000e+00 : f64) : f64
    %posx = llvm.mlir.constant(123 : i64) : i64
    %posy = llvm.mlir.constant(456 : i64) : i64
    %posz = llvm.mlir.constant(789 : i64) : i64
    //%5 = llvm.mlir.constant(dense<[0, 11, 3]> : array<3xi64>) : array<3xi64>
    %6 = llvm.mlir.undef : !llvm.array<3 x struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>>
    %7 = llvm.mlir.undef : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %8 = llvm.mlir.addressof @downstate : !llvm.ptr
    %9 = llvm.getelementptr inbounds %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i8>
    %10 = llvm.insertvalue %9, %7[0] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %11 = llvm.mlir.constant(6 : i64) : i64
    %12 = llvm.insertvalue %11, %10[1] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %13 = llvm.mlir.constant(4.000000e-01 : f64) : f64
    %14 = llvm.insertvalue %13, %12[2] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %15 = llvm.mlir.constant(5.000000e-01 : f64) : f64
    %16 = llvm.insertvalue %15, %14[3] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %17 = llvm.mlir.constant(6.000000e-01 : f64) : f64
    %18 = llvm.insertvalue %17, %16[4] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %19 = llvm.mlir.constant(8.000000e-01 : f64) : f64
    %20 = llvm.insertvalue %19, %18[5] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %21 = llvm.mlir.constant(9.000000e-01 : f64) : f64
    %22 = llvm.insertvalue %21, %20[6] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %23 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %24 = llvm.insertvalue %23, %22[7] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %25 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %26 = llvm.insertvalue %25, %24[8] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %27 = llvm.insertvalue %26, %6[0] : !llvm.array<3 x struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>>
    %28 = llvm.mlir.undef : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %29 = llvm.mlir.addressof @estate : !llvm.ptr
    %30 = llvm.getelementptr inbounds %29[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x i8>
    %31 = llvm.insertvalue %30, %28[0] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %32 = llvm.mlir.constant(6 : i64) : i64
    %33 = llvm.insertvalue %32, %31[1] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %34 = llvm.mlir.constant(1.400000e+00 : f64) : f64
    %35 = llvm.insertvalue %34, %33[2] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %36 = llvm.mlir.constant(1.500000e+00 : f64) : f64
    %37 = llvm.insertvalue %36, %35[3] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %38 = llvm.mlir.constant(1.600000e+00 : f64) : f64
    %39 = llvm.insertvalue %38, %37[4] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %40 = llvm.mlir.constant(1.800000e+00 : f64) : f64
    %41 = llvm.insertvalue %40, %39[5] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %42 = llvm.mlir.constant(1.900000e+00 : f64) : f64
    %43 = llvm.insertvalue %42, %41[6] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %44 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %45 = llvm.insertvalue %44, %43[7] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %46 = llvm.mlir.constant(1.264300e+10 : f64) : f64
    %47 = llvm.insertvalue %46, %45[8] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %48 = llvm.insertvalue %47, %27[1] : !llvm.array<3 x struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>>
    %49 = llvm.mlir.undef : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %50 = llvm.mlir.addressof @upstate : !llvm.ptr
    %51 = llvm.getelementptr inbounds %50[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i8>
    %52 = llvm.insertvalue %51, %49[0] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %53 = llvm.mlir.constant(5 : i64) : i64
    %54 = llvm.insertvalue %53, %52[1] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %55 = llvm.mlir.constant(2.400000e+00 : f64) : f64
    %56 = llvm.insertvalue %55, %54[2] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %57 = llvm.mlir.constant(2.500000e+00 : f64) : f64
    %58 = llvm.insertvalue %57, %56[3] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %59 = llvm.mlir.constant(2.600000e+00 : f64) : f64
    %60 = llvm.insertvalue %59, %58[4] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %61 = llvm.mlir.constant(2.800000e+00 : f64) : f64
    %62 = llvm.insertvalue %61, %60[5] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %63 = llvm.mlir.constant(2.900000e+00 : f64) : f64
    %64 = llvm.insertvalue %63, %62[6] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %65 = llvm.mlir.constant(3.000000e+00 : f64) : f64
    %66 = llvm.insertvalue %65, %64[7] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %67 = llvm.mlir.constant(8.115200e+14 : f64) : f64
    %68 = llvm.insertvalue %67, %66[8] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %69 = llvm.insertvalue %68, %48[2] : !llvm.array<3 x struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>>
    %70 = llvm.mlir.constant(3 : i64) : i64
    %71 = llvm.alloca %70 x !llvm.array<3 x struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>> : (i64) -> !llvm.ptr
    llvm.store %69, %71 : !llvm.array<3 x struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>>, !llvm.ptr
    %72 = llvm.mlir.undef : !llvm.array<3 x struct<(ptr, ptr, f64)>>
    %73 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, f64)>
    %74 = llvm.mlir.addressof @estate : !llvm.ptr
    %75 = llvm.getelementptr inbounds %74[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x i8>
    %76 = llvm.mlir.addressof @downstate : !llvm.ptr
    %77 = llvm.getelementptr inbounds %76[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i8>
    %78 = llvm.insertvalue %75, %73[0] : !llvm.struct<(ptr, ptr, f64)>
    %79 = llvm.insertvalue %77, %78[1] : !llvm.struct<(ptr, ptr, f64)>
    %80 = llvm.mlir.constant(2.200000e+00 : f64) : f64
    %81 = llvm.insertvalue %80, %79[2] : !llvm.struct<(ptr, ptr, f64)>
    %82 = llvm.insertvalue %81, %72[0] : !llvm.array<3 x struct<(ptr, ptr, f64)>>
    %83 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, f64)>
    %84 = llvm.mlir.addressof @upstate : !llvm.ptr
    %85 = llvm.getelementptr inbounds %84[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i8>
    %86 = llvm.mlir.addressof @downstate : !llvm.ptr
    %87 = llvm.getelementptr inbounds %86[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i8>
    %88 = llvm.insertvalue %85, %83[0] : !llvm.struct<(ptr, ptr, f64)>
    %89 = llvm.insertvalue %87, %88[1] : !llvm.struct<(ptr, ptr, f64)>
    %90 = llvm.mlir.constant(1.100000e+00 : f64) : f64
    %91 = llvm.insertvalue %90, %89[2] : !llvm.struct<(ptr, ptr, f64)>
    %92 = llvm.insertvalue %91, %82[1] : !llvm.array<3 x struct<(ptr, ptr, f64)>>
    %93 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, f64)>
    %94 = llvm.mlir.addressof @upstate : !llvm.ptr
    %95 = llvm.getelementptr inbounds %94[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i8>
    %96 = llvm.mlir.addressof @estate : !llvm.ptr
    %97 = llvm.getelementptr inbounds %96[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x i8>
    %98 = llvm.insertvalue %95, %93[0] : !llvm.struct<(ptr, ptr, f64)>
    %99 = llvm.insertvalue %97, %98[1] : !llvm.struct<(ptr, ptr, f64)>
    %100 = llvm.mlir.constant(3.300000e+00 : f64) : f64
    %101 = llvm.insertvalue %100, %99[2] : !llvm.struct<(ptr, ptr, f64)>
    %102 = llvm.insertvalue %101, %92[2] : !llvm.array<3 x struct<(ptr, ptr, f64)>>
    %103 = llvm.mlir.constant(3 : i64) : i64
    %104 = llvm.alloca %103 x !llvm.array<3 x struct<(ptr, ptr, f64)>> : (i64) -> !llvm.ptr
    llvm.store %102, %104 : !llvm.array<3 x struct<(ptr, ptr, f64)>>, !llvm.ptr
    %105 = llvm.mlir.undef : !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, ptr)>
    %106 = llvm.insertvalue %2, %105[0] : !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, ptr)>
    %107 = llvm.insertvalue %3, %106[1] : !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, ptr)>
    %108 = llvm.insertvalue %4, %107[2] : !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, ptr)>
    //%109 = llvm.insertvalue %5, %108[3] : !llvm.struct<(ptr, f64, f64, array<3xi64>, ptr, ptr)>
    %200 = llvm.insertvalue %posx, %108[3, 0] : !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, ptr)>
    %201 = llvm.insertvalue %posy, %200[3, 1] : !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, ptr)>
    %202 = llvm.insertvalue %posz, %201[3, 2] : !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, ptr)>
    %110 = llvm.insertvalue %71, %202[4] : !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, ptr)>
    %111 = llvm.insertvalue %104, %110[5] : !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, ptr)>
    %112 = llvm.alloca %0 x !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, ptr)> : (i64) -> !llvm.ptr
    llvm.store %111, %112 : !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, ptr)>, !llvm.ptr
    llvm.call @__catalyst__oqd__ion(%112) : (!llvm.ptr) -> ()
    llvm.return
  }

  llvm.func @jit_f() -> () {
    llvm.call @f_0() : () -> ()
    llvm.call @ion_op() : () -> ()
    llvm.return
  }

  llvm.func @_catalyst_pyface_jit_f(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    llvm.call @_catalyst_ciface_jit_f(%arg0) : (!llvm.ptr) -> ()
    llvm.return
  }

  llvm.func @_catalyst_ciface_jit_f(%arg0: !llvm.ptr) attributes {llvm.copy_memref, llvm.emit_c_interface} {
    llvm.call @jit_f() : () -> ()
    llvm.return
  }

  llvm.func internal @f_0() -> () {
    llvm.call @__catalyst__oqd__greetings() : () -> ()
    llvm.return
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
