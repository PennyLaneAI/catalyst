module @f {
  llvm.func @__catalyst__rt__finalize()
  llvm.func @__catalyst__rt__initialize(!llvm.ptr)
  llvm.func @__catalyst__rt__device_release()
  llvm.func @__catalyst__rt__qubit_release_array(!llvm.ptr)
  llvm.func @__catalyst__qis__Counts(!llvm.ptr, i64, ...)
  llvm.func @__catalyst__rt__array_get_element_ptr_1d(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @__catalyst__rt__qubit_allocate_array(i64) -> !llvm.ptr
  llvm.mlir.global internal constant @"{'shots': 1000}"("{'shots': 1000}\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @oqd("oqd\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @"/home/paul.wang/catalyst_new/catalyst/frontend/catalyst/utils/../../catalyst/third_party/oqd/src/build/librtd_oqd.so"("/home/paul.wang/catalyst_new/catalyst/frontend/catalyst/utils/../../catalyst/third_party/oqd/src/build/librtd_oqd.so\00") {addr_space = 0 : i32}
  llvm.func @__catalyst__rt__device_init(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64)
  llvm.func @_mlir_memref_to_llvm_free(!llvm.ptr)
  llvm.func @_mlir_memref_to_llvm_alloc(i64) -> !llvm.ptr

  llvm.func @__catalyst__oqd__ion(!llvm.ptr) -> ()
  llvm.func @__catalyst__oqd__rt__initialize()
  llvm.func @__catalyst__oqd__rt__finalize()

  llvm.mlir.global internal constant @upstate("upstate\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @estate("estate\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @downstate("downstate\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @Yb171("Yb171\00") {addr_space = 0 : i32}


  llvm.func @jit_f() -> !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)> attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(4 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(3735928559 : index) : i64
    %4 = llvm.call @f_0() : () -> !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>
    %5 = llvm.extractvalue %4[0] : !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>
    %6 = llvm.extractvalue %4[1] : !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>
    %7 = llvm.extractvalue %5[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.ptrtoint %7 : !llvm.ptr to i64
    %9 = llvm.icmp "eq" %3, %8 : i64
    llvm.cond_br %9, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %10 = llvm.mlir.zero : !llvm.ptr
    %11 = llvm.getelementptr %10[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %12 = llvm.ptrtoint %11 : !llvm.ptr to i64
    %13 = llvm.call @_mlir_memref_to_llvm_alloc(%12) : (i64) -> !llvm.ptr
    %14 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.insertvalue %13, %14[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.insertvalue %13, %15[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.insertvalue %1, %16[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %18 = llvm.insertvalue %2, %17[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.insertvalue %2, %18[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %20 = llvm.extractvalue %5[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %21 = llvm.mul %20, %2 : i64
    %22 = llvm.mlir.zero : !llvm.ptr
    %23 = llvm.getelementptr %22[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %24 = llvm.ptrtoint %23 : !llvm.ptr to i64
    %25 = llvm.mul %21, %24 : i64
    %26 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %27 = llvm.extractvalue %5[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %28 = llvm.getelementptr inbounds %26[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    "llvm.intr.memcpy"(%13, %28, %25) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb3(%19 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb3(%5 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb3(%29: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    %30 = llvm.extractvalue %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %31 = llvm.ptrtoint %30 : !llvm.ptr to i64
    %32 = llvm.icmp "eq" %3, %31 : i64
    llvm.cond_br %32, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %33 = llvm.mlir.zero : !llvm.ptr
    %34 = llvm.getelementptr %33[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %35 = llvm.ptrtoint %34 : !llvm.ptr to i64
    %36 = llvm.call @_mlir_memref_to_llvm_alloc(%35) : (i64) -> !llvm.ptr
    %37 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %38 = llvm.insertvalue %36, %37[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %39 = llvm.insertvalue %36, %38[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %40 = llvm.insertvalue %1, %39[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %41 = llvm.insertvalue %0, %40[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %42 = llvm.insertvalue %2, %41[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %43 = llvm.extractvalue %6[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %44 = llvm.mul %43, %2 : i64
    %45 = llvm.mlir.zero : !llvm.ptr
    %46 = llvm.getelementptr %45[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %47 = llvm.ptrtoint %46 : !llvm.ptr to i64
    %48 = llvm.mul %44, %47 : i64
    %49 = llvm.extractvalue %6[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %50 = llvm.extractvalue %6[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %51 = llvm.getelementptr inbounds %49[%50] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    "llvm.intr.memcpy"(%36, %51, %48) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb7(%42 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb6:  // pred: ^bb4
    llvm.br ^bb7(%6 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb7(%52: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>):  // 2 preds: ^bb5, ^bb6
    llvm.br ^bb8
  ^bb8:  // pred: ^bb7
    %53 = llvm.mlir.undef : !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>
    %54 = llvm.insertvalue %29, %53[0] : !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>
    %55 = llvm.insertvalue %52, %54[1] : !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>
    llvm.return %55 : !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>
  }
  llvm.func @_catalyst_pyface_jit_f(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    llvm.call @_catalyst_ciface_jit_f(%arg0) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_catalyst_ciface_jit_f(%arg0: !llvm.ptr) attributes {llvm.copy_memref, llvm.emit_c_interface} {
    %0 = llvm.call @jit_f() : () -> !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>
    llvm.store %0, %arg0 : !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, !llvm.ptr
    llvm.return
  }
  llvm.func internal @f_0() -> !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)> attributes {diff_method = "parameter-shift", qnode} {

    // make ion ptr and send to device init
    %700 = llvm.mlir.constant(1 : i64) : i64
    %701 = llvm.mlir.addressof @Yb171 : !llvm.ptr
    %702 = llvm.getelementptr inbounds %701[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<6 x i8>
    %703 = llvm.mlir.constant(1.710000e+02 : f64) : f64
    %704 = llvm.mlir.constant(56.110000e+00 : f64) : f64
    %posx = llvm.mlir.constant(123 : i64) : i64
    %posy = llvm.mlir.constant(456 : i64) : i64
    %posz = llvm.mlir.constant(789 : i64) : i64
    //%5 = llvm.mlir.constant(dense<[0, 11, 3]> : array<3xi64>) : array<3xi64>
    %706 = llvm.mlir.undef : !llvm.array<3 x struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>>
    %707 = llvm.mlir.undef : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %708 = llvm.mlir.addressof @downstate : !llvm.ptr
    %709 = llvm.getelementptr inbounds %708[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i8>
    %7010 = llvm.insertvalue %709, %707[0] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7011 = llvm.mlir.constant(6 : i64) : i64
    %7012 = llvm.insertvalue %7011, %7010[1] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7013 = llvm.mlir.constant(4.000000e-01 : f64) : f64
    %7014 = llvm.insertvalue %7013, %7012[2] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7015 = llvm.mlir.constant(5.000000e-01 : f64) : f64
    %7016 = llvm.insertvalue %7015, %7014[3] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7017 = llvm.mlir.constant(6.000000e-01 : f64) : f64
    %7018 = llvm.insertvalue %7017, %7016[4] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7019 = llvm.mlir.constant(8.000000e-01 : f64) : f64
    %7020 = llvm.insertvalue %7019, %7018[5] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7021 = llvm.mlir.constant(9.000000e-01 : f64) : f64
    %7022 = llvm.insertvalue %7021, %7020[6] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7023 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %7024 = llvm.insertvalue %7023, %7022[7] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7025 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %7026 = llvm.insertvalue %7025, %7024[8] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7027 = llvm.insertvalue %7026, %706[0] : !llvm.array<3 x struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>>
    %7028 = llvm.mlir.undef : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7029 = llvm.mlir.addressof @estate : !llvm.ptr
    %7030 = llvm.getelementptr inbounds %7029[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x i8>
    %7031 = llvm.insertvalue %7030, %7028[0] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7032 = llvm.mlir.constant(6 : i64) : i64
    %7033 = llvm.insertvalue %7032, %7031[1] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7034 = llvm.mlir.constant(1.400000e+00 : f64) : f64
    %7035 = llvm.insertvalue %7034, %7033[2] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7036 = llvm.mlir.constant(1.500000e+00 : f64) : f64
    %7037 = llvm.insertvalue %7036, %7035[3] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7038 = llvm.mlir.constant(1.600000e+00 : f64) : f64
    %7039 = llvm.insertvalue %7038, %7037[4] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7040 = llvm.mlir.constant(1.800000e+00 : f64) : f64
    %7041 = llvm.insertvalue %7040, %7039[5] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7042 = llvm.mlir.constant(1.900000e+00 : f64) : f64
    %7043 = llvm.insertvalue %7042, %7041[6] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7044 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %7045 = llvm.insertvalue %7044, %7043[7] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7046 = llvm.mlir.constant(1.264300e+10 : f64) : f64
    %7047 = llvm.insertvalue %7046, %7045[8] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7048 = llvm.insertvalue %7047, %7027[1] : !llvm.array<3 x struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>>
    %7049 = llvm.mlir.undef : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7050 = llvm.mlir.addressof @upstate : !llvm.ptr
    %7051 = llvm.getelementptr inbounds %7050[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i8>
    %7052 = llvm.insertvalue %7051, %7049[0] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7053 = llvm.mlir.constant(5 : i64) : i64
    %7054 = llvm.insertvalue %7053, %7052[1] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7055 = llvm.mlir.constant(2.400000e+00 : f64) : f64
    %7056 = llvm.insertvalue %7055, %7054[2] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7057 = llvm.mlir.constant(2.500000e+00 : f64) : f64
    %7058 = llvm.insertvalue %7057, %7056[3] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7059 = llvm.mlir.constant(2.600000e+00 : f64) : f64
    %7060 = llvm.insertvalue %7059, %7058[4] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7061 = llvm.mlir.constant(2.800000e+00 : f64) : f64
    %7062 = llvm.insertvalue %7061, %7060[5] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7063 = llvm.mlir.constant(2.900000e+00 : f64) : f64
    %7064 = llvm.insertvalue %7063, %7062[6] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7065 = llvm.mlir.constant(3.000000e+00 : f64) : f64
    %7066 = llvm.insertvalue %7065, %7064[7] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7067 = llvm.mlir.constant(8.115200e+14 : f64) : f64
    %7068 = llvm.insertvalue %7067, %7066[8] : !llvm.struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>
    %7069 = llvm.insertvalue %7068, %7048[2] : !llvm.array<3 x struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>>
    %7070 = llvm.mlir.constant(3 : i64) : i64
    %7071 = llvm.alloca %7070 x !llvm.array<3 x struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>> : (i64) -> !llvm.ptr
    llvm.store %7069, %7071 : !llvm.array<3 x struct<(ptr, i64, f64, f64, f64, f64, f64, f64, f64)>>, !llvm.ptr
    %7072 = llvm.mlir.undef : !llvm.array<3 x struct<(ptr, ptr, f64)>>
    %7073 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, f64)>
    %7074 = llvm.mlir.addressof @estate : !llvm.ptr
    %7075 = llvm.getelementptr inbounds %7074[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x i8>
    %7076 = llvm.mlir.addressof @downstate : !llvm.ptr
    %7077 = llvm.getelementptr inbounds %7076[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i8>
    %7078 = llvm.insertvalue %7075, %7073[0] : !llvm.struct<(ptr, ptr, f64)>
    %7079 = llvm.insertvalue %7077, %7078[1] : !llvm.struct<(ptr, ptr, f64)>
    %7080 = llvm.mlir.constant(2.200000e+00 : f64) : f64
    %7081 = llvm.insertvalue %7080, %7079[2] : !llvm.struct<(ptr, ptr, f64)>
    %7082 = llvm.insertvalue %7081, %7072[0] : !llvm.array<3 x struct<(ptr, ptr, f64)>>
    %7083 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, f64)>
    %7084 = llvm.mlir.addressof @upstate : !llvm.ptr
    %7085 = llvm.getelementptr inbounds %7084[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i8>
    %7086 = llvm.mlir.addressof @downstate : !llvm.ptr
    %7087 = llvm.getelementptr inbounds %7086[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i8>
    %7088 = llvm.insertvalue %7085, %7083[0] : !llvm.struct<(ptr, ptr, f64)>
    %7089 = llvm.insertvalue %7087, %7088[1] : !llvm.struct<(ptr, ptr, f64)>
    %7090 = llvm.mlir.constant(1.100000e+00 : f64) : f64
    %7091 = llvm.insertvalue %7090, %7089[2] : !llvm.struct<(ptr, ptr, f64)>
    %7092 = llvm.insertvalue %7091, %7082[1] : !llvm.array<3 x struct<(ptr, ptr, f64)>>
    %7093 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, f64)>
    %7094 = llvm.mlir.addressof @upstate : !llvm.ptr
    %7095 = llvm.getelementptr inbounds %7094[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i8>
    %7096 = llvm.mlir.addressof @estate : !llvm.ptr
    %7097 = llvm.getelementptr inbounds %7096[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x i8>
    %7098 = llvm.insertvalue %7095, %7093[0] : !llvm.struct<(ptr, ptr, f64)>
    %7099 = llvm.insertvalue %7097, %7098[1] : !llvm.struct<(ptr, ptr, f64)>
    %70100 = llvm.mlir.constant(3.300000e+00 : f64) : f64
    %70101 = llvm.insertvalue %70100, %7099[2] : !llvm.struct<(ptr, ptr, f64)>
    %70102 = llvm.insertvalue %70101, %7092[2] : !llvm.array<3 x struct<(ptr, ptr, f64)>>
    %70103 = llvm.mlir.constant(3 : i64) : i64
    %70104 = llvm.alloca %70103 x !llvm.array<3 x struct<(ptr, ptr, f64)>> : (i64) -> !llvm.ptr
    llvm.store %70102, %70104 : !llvm.array<3 x struct<(ptr, ptr, f64)>>, !llvm.ptr
    %70105 = llvm.mlir.undef : !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, i64, ptr, i64)>
    %70106 = llvm.insertvalue %702, %70105[0] : !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, i64, ptr, i64)>
    %70107 = llvm.insertvalue %703, %70106[1] : !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, i64, ptr, i64)>
    %70108 = llvm.insertvalue %704, %70107[2] : !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, i64, ptr, i64)>
    //%109 = llvm.insertvalue %5, %108[3] : !llvm.struct<(ptr, f64, f64, array<3xi64>, ptr, ptr)>
    %70200 = llvm.insertvalue %posx, %70108[3, 0] : !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, i64, ptr, i64)>
    %70201 = llvm.insertvalue %posy, %70200[3, 1] : !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, i64, ptr, i64)>
    %70202 = llvm.insertvalue %posz, %70201[3, 2] : !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, i64, ptr, i64)>
    %70110 = llvm.insertvalue %7071, %70202[4] : !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, i64, ptr, i64)>
    %70111 = llvm.insertvalue %7070, %70110[5] : !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, i64, ptr, i64)>
    %70112 = llvm.insertvalue %70104, %70111[6] : !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, i64, ptr, i64)>
    %70113 = llvm.insertvalue %70103, %70112[7] : !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, i64, ptr, i64)>
    %70114 = llvm.alloca %700 x !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, i64, ptr, i64)> : (i64) -> !llvm.ptr
    llvm.store %70113, %70114 : !llvm.struct<(ptr, f64, f64, array<3 x i64>, ptr, i64, ptr, i64)>, !llvm.ptr
    //llvm.call @__catalyst__oqd__ion(%70114) : (!llvm.ptr) -> ()


    // regular catalyst
    %0 = llvm.mlir.constant(64 : index) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.mlir.constant(2 : i64) : i64
    %4 = llvm.mlir.addressof @"{'shots': 1000}" : !llvm.ptr
    %5 = llvm.mlir.addressof @oqd : !llvm.ptr
    %6 = llvm.mlir.addressof @"/home/paul.wang/catalyst_new/catalyst/frontend/catalyst/utils/../../catalyst/third_party/oqd/src/build/librtd_oqd.so" : !llvm.ptr
    %7 = llvm.mlir.constant(3 : index) : i64
    %8 = llvm.mlir.constant(2 : index) : i64
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.mlir.constant(1000 : i64) : i64
    %11 = llvm.mlir.constant(4 : index) : i64
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.getelementptr inbounds %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<117 x i8>
    %14 = llvm.getelementptr inbounds %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i8>
    %15 = llvm.getelementptr inbounds %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i8>
    llvm.call @__catalyst__rt__device_init(%13, %14, %15, %10, %70114) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
    %16 = llvm.call @__catalyst__rt__qubit_allocate_array(%3) : (i64) -> !llvm.ptr
    %17 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%16, %2) : (!llvm.ptr, i64) -> !llvm.ptr
    %18 = llvm.load %17 : !llvm.ptr -> !llvm.ptr
    %19 = llvm.call @__catalyst__rt__array_get_element_ptr_1d(%16, %1) : (!llvm.ptr, i64) -> !llvm.ptr
    %20 = llvm.load %19 : !llvm.ptr -> !llvm.ptr
    %21 = llvm.mlir.zero : !llvm.ptr
    %22 = llvm.getelementptr %21[4] : (!llvm.ptr) -> !llvm.ptr, f64
    %23 = llvm.ptrtoint %22 : !llvm.ptr to i64
    %24 = llvm.call @_mlir_memref_to_llvm_alloc(%23) : (i64) -> !llvm.ptr
    %25 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %26 = llvm.insertvalue %24, %25[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %27 = llvm.insertvalue %24, %26[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %28 = llvm.insertvalue %12, %27[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %29 = llvm.insertvalue %11, %28[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %30 = llvm.insertvalue %9, %29[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %31 = llvm.mlir.zero : !llvm.ptr
    %32 = llvm.getelementptr %31[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %33 = llvm.ptrtoint %32 : !llvm.ptr to i64
    %34 = llvm.call @_mlir_memref_to_llvm_alloc(%33) : (i64) -> !llvm.ptr
    %35 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %36 = llvm.insertvalue %34, %35[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %37 = llvm.insertvalue %34, %36[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %38 = llvm.insertvalue %12, %37[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %39 = llvm.insertvalue %11, %38[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %40 = llvm.insertvalue %9, %39[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %41 = llvm.alloca %1 x !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)> : (i64) -> !llvm.ptr
    %42 = llvm.mlir.undef : !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>
    %43 = llvm.insertvalue %30, %42[0] : !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>
    %44 = llvm.insertvalue %40, %43[1] : !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>
    llvm.store %44, %41 : !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, !llvm.ptr
    llvm.call @__catalyst__qis__Counts(%41, %3, %18, %20) vararg(!llvm.func<void (ptr, i64, ...)>) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> ()
    %45 = llvm.mlir.zero : !llvm.ptr
    %46 = llvm.getelementptr %45[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %47 = llvm.ptrtoint %46 : !llvm.ptr to i64
    %48 = llvm.add %47, %0 : i64
    %49 = llvm.call @_mlir_memref_to_llvm_alloc(%48) : (i64) -> !llvm.ptr
    %50 = llvm.ptrtoint %49 : !llvm.ptr to i64
    %51 = llvm.sub %0, %9 : i64
    %52 = llvm.add %50, %51 : i64
    %53 = llvm.urem %52, %0  : i64
    %54 = llvm.sub %52, %53 : i64
    %55 = llvm.inttoptr %54 : i64 to !llvm.ptr
    llvm.br ^bb1(%12 : i64)
  ^bb1(%56: i64):  // 2 preds: ^bb0, ^bb2
    %57 = llvm.icmp "slt" %56, %11 : i64
    llvm.cond_br %57, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %58 = llvm.getelementptr inbounds %24[%56] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %59 = llvm.load %58 : !llvm.ptr -> f64
    %60 = llvm.fptosi %59 : f64 to i64
    %61 = llvm.getelementptr inbounds %55[%56] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %60, %61 : i64, !llvm.ptr
    %62 = llvm.add %56, %9 : i64
    llvm.br ^bb1(%62 : i64)
  ^bb3:  // pred: ^bb1
    llvm.call @_mlir_memref_to_llvm_free(%24) : (!llvm.ptr) -> ()
    llvm.call @__catalyst__rt__qubit_release_array(%16) : (!llvm.ptr) -> ()
    llvm.call @__catalyst__rt__device_release() : () -> ()
    %63 = llvm.mlir.zero : !llvm.ptr
    %64 = llvm.getelementptr %63[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %65 = llvm.ptrtoint %64 : !llvm.ptr to i64
    %66 = llvm.add %65, %0 : i64
    %67 = llvm.call @_mlir_memref_to_llvm_alloc(%66) : (i64) -> !llvm.ptr
    %68 = llvm.ptrtoint %67 : !llvm.ptr to i64
    %69 = llvm.sub %0, %9 : i64
    %70 = llvm.add %68, %69 : i64
    %71 = llvm.urem %70, %0  : i64
    %72 = llvm.sub %70, %71 : i64
    %73 = llvm.inttoptr %72 : i64 to !llvm.ptr
    llvm.br ^bb4(%12 : i64)
  ^bb4(%74: i64):  // 2 preds: ^bb3, ^bb5
    %75 = llvm.icmp "slt" %74, %9 : i64
    llvm.cond_br %75, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %76 = llvm.load %34 : !llvm.ptr -> i64
    %77 = llvm.getelementptr inbounds %73[%74] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %76, %77 : i64, !llvm.ptr
    %78 = llvm.add %74, %9 : i64
    llvm.br ^bb4(%78 : i64)
  ^bb6:  // pred: ^bb4
    %79 = llvm.mlir.zero : !llvm.ptr
    %80 = llvm.getelementptr %79[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %81 = llvm.ptrtoint %80 : !llvm.ptr to i64
    %82 = llvm.add %81, %0 : i64
    %83 = llvm.call @_mlir_memref_to_llvm_alloc(%82) : (i64) -> !llvm.ptr
    %84 = llvm.ptrtoint %83 : !llvm.ptr to i64
    %85 = llvm.sub %0, %9 : i64
    %86 = llvm.add %84, %85 : i64
    %87 = llvm.urem %86, %0  : i64
    %88 = llvm.sub %86, %87 : i64
    %89 = llvm.inttoptr %88 : i64 to !llvm.ptr
    llvm.br ^bb7(%12 : i64)
  ^bb7(%90: i64):  // 2 preds: ^bb6, ^bb8
    %91 = llvm.icmp "slt" %90, %9 : i64
    llvm.cond_br %91, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %92 = llvm.getelementptr inbounds %34[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %93 = llvm.load %92 : !llvm.ptr -> i64
    %94 = llvm.getelementptr inbounds %89[%90] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %93, %94 : i64, !llvm.ptr
    %95 = llvm.add %90, %9 : i64
    llvm.br ^bb7(%95 : i64)
  ^bb9:  // pred: ^bb7
    %96 = llvm.mlir.zero : !llvm.ptr
    %97 = llvm.getelementptr %96[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %98 = llvm.ptrtoint %97 : !llvm.ptr to i64
    %99 = llvm.add %98, %0 : i64
    %100 = llvm.call @_mlir_memref_to_llvm_alloc(%99) : (i64) -> !llvm.ptr
    %101 = llvm.ptrtoint %100 : !llvm.ptr to i64
    %102 = llvm.sub %0, %9 : i64
    %103 = llvm.add %101, %102 : i64
    %104 = llvm.urem %103, %0  : i64
    %105 = llvm.sub %103, %104 : i64
    %106 = llvm.inttoptr %105 : i64 to !llvm.ptr
    %107 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %108 = llvm.insertvalue %100, %107[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %109 = llvm.insertvalue %106, %108[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %110 = llvm.insertvalue %12, %109[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %111 = llvm.insertvalue %9, %110[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %112 = llvm.insertvalue %9, %111[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.br ^bb10(%12 : i64)
  ^bb10(%113: i64):  // 2 preds: ^bb9, ^bb11
    %114 = llvm.icmp "slt" %113, %9 : i64
    llvm.cond_br %114, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %115 = llvm.getelementptr inbounds %34[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %116 = llvm.load %115 : !llvm.ptr -> i64
    %117 = llvm.getelementptr inbounds %106[%113] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %116, %117 : i64, !llvm.ptr
    %118 = llvm.add %113, %9 : i64
    llvm.br ^bb10(%118 : i64)
  ^bb12:  // pred: ^bb10
    %119 = llvm.mlir.zero : !llvm.ptr
    %120 = llvm.getelementptr %119[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %121 = llvm.ptrtoint %120 : !llvm.ptr to i64
    %122 = llvm.add %121, %0 : i64
    %123 = llvm.call @_mlir_memref_to_llvm_alloc(%122) : (i64) -> !llvm.ptr
    %124 = llvm.ptrtoint %123 : !llvm.ptr to i64
    %125 = llvm.sub %0, %9 : i64
    %126 = llvm.add %124, %125 : i64
    %127 = llvm.urem %126, %0  : i64
    %128 = llvm.sub %126, %127 : i64
    %129 = llvm.inttoptr %128 : i64 to !llvm.ptr
    %130 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %131 = llvm.insertvalue %123, %130[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %132 = llvm.insertvalue %129, %131[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %133 = llvm.insertvalue %12, %132[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %134 = llvm.insertvalue %9, %133[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %135 = llvm.insertvalue %9, %134[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.br ^bb13(%12 : i64)
  ^bb13(%136: i64):  // 2 preds: ^bb12, ^bb14
    %137 = llvm.icmp "slt" %136, %9 : i64
    llvm.cond_br %137, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    %138 = llvm.getelementptr inbounds %34[3] : (!llvm.ptr) -> !llvm.ptr, i64
    %139 = llvm.load %138 : !llvm.ptr -> i64
    %140 = llvm.getelementptr inbounds %129[%136] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %139, %140 : i64, !llvm.ptr
    %141 = llvm.add %136, %9 : i64
    llvm.br ^bb13(%141 : i64)
  ^bb15:  // pred: ^bb13
    llvm.call @_mlir_memref_to_llvm_free(%34) : (!llvm.ptr) -> ()
    %142 = llvm.mlir.zero : !llvm.ptr
    %143 = llvm.getelementptr %142[4] : (!llvm.ptr) -> !llvm.ptr, i64
    %144 = llvm.ptrtoint %143 : !llvm.ptr to i64
    %145 = llvm.add %144, %0 : i64
    %146 = llvm.call @_mlir_memref_to_llvm_alloc(%145) : (i64) -> !llvm.ptr
    %147 = llvm.ptrtoint %146 : !llvm.ptr to i64
    %148 = llvm.sub %0, %9 : i64
    %149 = llvm.add %147, %148 : i64
    %150 = llvm.urem %149, %0  : i64
    %151 = llvm.sub %149, %150 : i64
    %152 = llvm.inttoptr %151 : i64 to !llvm.ptr
    %153 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %154 = llvm.insertvalue %146, %153[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %155 = llvm.insertvalue %152, %154[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %156 = llvm.insertvalue %12, %155[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %157 = llvm.insertvalue %11, %156[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %158 = llvm.insertvalue %9, %157[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.br ^bb16(%12 : i64)
  ^bb16(%159: i64):  // 2 preds: ^bb15, ^bb28
    %160 = llvm.icmp "slt" %159, %11 : i64
    llvm.cond_br %160, ^bb17, ^bb29
  ^bb17:  // pred: ^bb16
    %161 = llvm.icmp "ult" %159, %9 : i64
    llvm.cond_br %161, ^bb18, ^bb19
  ^bb18:  // pred: ^bb17
    %162 = llvm.getelementptr inbounds %73[%159] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %163 = llvm.load %162 : !llvm.ptr -> i64
    llvm.br ^bb27(%163 : i64)
  ^bb19:  // pred: ^bb17
    %164 = llvm.icmp "ult" %159, %8 : i64
    llvm.cond_br %164, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %165 = llvm.sub %159, %9 : i64
    %166 = llvm.getelementptr inbounds %89[%165] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %167 = llvm.load %166 : !llvm.ptr -> i64
    llvm.br ^bb25(%167 : i64)
  ^bb21:  // pred: ^bb19
    %168 = llvm.icmp "ult" %159, %7 : i64
    llvm.cond_br %168, ^bb22(%8, %112 : i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>), ^bb22(%7, %135 : i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
  ^bb22(%169: i64, %170: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>):  // 2 preds: ^bb21, ^bb21
    %171 = llvm.sub %159, %169 : i64
    %172 = llvm.extractvalue %170[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %173 = llvm.getelementptr inbounds %172[%171] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %174 = llvm.load %173 : !llvm.ptr -> i64
    llvm.br ^bb23(%174 : i64)
  ^bb23(%175: i64):  // pred: ^bb22
    llvm.br ^bb24
  ^bb24:  // pred: ^bb23
    llvm.br ^bb25(%175 : i64)
  ^bb25(%176: i64):  // 2 preds: ^bb20, ^bb24
    llvm.br ^bb26
  ^bb26:  // pred: ^bb25
    llvm.br ^bb27(%176 : i64)
  ^bb27(%177: i64):  // 2 preds: ^bb18, ^bb26
    llvm.br ^bb28
  ^bb28:  // pred: ^bb27
    %178 = llvm.getelementptr inbounds %152[%159] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %177, %178 : i64, !llvm.ptr
    %179 = llvm.add %159, %9 : i64
    llvm.br ^bb16(%179 : i64)
  ^bb29:  // pred: ^bb16
    llvm.call @_mlir_memref_to_llvm_free(%123) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%100) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%83) : (!llvm.ptr) -> ()
    llvm.call @_mlir_memref_to_llvm_free(%67) : (!llvm.ptr) -> ()
    %180 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %181 = llvm.insertvalue %49, %180[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %182 = llvm.insertvalue %55, %181[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %183 = llvm.insertvalue %12, %182[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %184 = llvm.insertvalue %9, %183[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %185 = llvm.insertvalue %9, %184[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %186 = llvm.mlir.undef : !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>
    %187 = llvm.insertvalue %185, %186[0] : !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>
    %188 = llvm.insertvalue %158, %187[1] : !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>
    llvm.return %188 : !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>
  }
  llvm.func @setup() {
    %0 = llvm.mlir.zero : !llvm.ptr
    llvm.call @__catalyst__rt__initialize(%0) : (!llvm.ptr) -> ()
    //llvm.call @__catalyst__oqd__rt__initialize() : () -> ()
    llvm.return
  }
  llvm.func @teardown() {
    llvm.call @__catalyst__rt__finalize() : () -> ()
    //llvm.call @__catalyst__oqd__rt__initialize() : () -> ()
    llvm.return
  }
}
