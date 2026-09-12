module @module_grover_5 {
  func.func public @grover_5(%arg0: tensor<1xi64>, %arg1: tensor<i64>) attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, quantum.node} {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c0_i64 = arith.constant 0 : i64
    quantum.device shots(%c0_i64) ["/Users/sara.babaeekhanehsar/Desktop/Home/Coding/Catalyst/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_null_qubit.dylib", "NullQubit", "{'track_resources': False}"]
    %0 = qref.alloc( 129) : !qref.reg<129>
    %cst_false = arith.constant dense<[false]> : tensor<1xi1>

    %1 = qref.get %0[ 0] : !qref.reg<129> -> !qref.bit
    %130 = qref.get %0[ 1] : !qref.reg<129> -> !qref.bit
    %128 = qref.get %0[ 2] : !qref.reg<129> -> !qref.bit
    %126 = qref.get %0[ 3] : !qref.reg<129> -> !qref.bit
    %124 = qref.get %0[ 4] : !qref.reg<129> -> !qref.bit
    %122 = qref.get %0[ 5] : !qref.reg<129> -> !qref.bit
    %120 = qref.get %0[ 6] : !qref.reg<129> -> !qref.bit
    %118 = qref.get %0[ 7] : !qref.reg<129> -> !qref.bit
    %116 = qref.get %0[ 8] : !qref.reg<129> -> !qref.bit
    %114 = qref.get %0[ 9] : !qref.reg<129> -> !qref.bit
    %112 = qref.get %0[ 10] : !qref.reg<129> -> !qref.bit
    %110 = qref.get %0[ 11] : !qref.reg<129> -> !qref.bit
    %108 = qref.get %0[ 12] : !qref.reg<129> -> !qref.bit
    %106 = qref.get %0[ 13] : !qref.reg<129> -> !qref.bit
    %104 = qref.get %0[ 14] : !qref.reg<129> -> !qref.bit
    %102 = qref.get %0[ 15] : !qref.reg<129> -> !qref.bit
    %100 = qref.get %0[ 16] : !qref.reg<129> -> !qref.bit
    %98 = qref.get %0[ 17] : !qref.reg<129> -> !qref.bit
    %96 = qref.get %0[ 18] : !qref.reg<129> -> !qref.bit
    %94 = qref.get %0[ 19] : !qref.reg<129> -> !qref.bit
    %92 = qref.get %0[ 20] : !qref.reg<129> -> !qref.bit
    %90 = qref.get %0[ 21] : !qref.reg<129> -> !qref.bit
    %88 = qref.get %0[ 22] : !qref.reg<129> -> !qref.bit
    %86 = qref.get %0[ 23] : !qref.reg<129> -> !qref.bit
    %84 = qref.get %0[ 24] : !qref.reg<129> -> !qref.bit
    %82 = qref.get %0[ 25] : !qref.reg<129> -> !qref.bit
    %80 = qref.get %0[ 26] : !qref.reg<129> -> !qref.bit
    %78 = qref.get %0[ 27] : !qref.reg<129> -> !qref.bit
    %76 = qref.get %0[ 28] : !qref.reg<129> -> !qref.bit
    %74 = qref.get %0[ 29] : !qref.reg<129> -> !qref.bit
    %72 = qref.get %0[ 30] : !qref.reg<129> -> !qref.bit
    %70 = qref.get %0[ 31] : !qref.reg<129> -> !qref.bit
    %68 = qref.get %0[ 32] : !qref.reg<129> -> !qref.bit
    %66 = qref.get %0[ 33] : !qref.reg<129> -> !qref.bit
    %64 = qref.get %0[ 34] : !qref.reg<129> -> !qref.bit
    %62 = qref.get %0[ 35] : !qref.reg<129> -> !qref.bit
    %60 = qref.get %0[ 36] : !qref.reg<129> -> !qref.bit
    %58 = qref.get %0[ 37] : !qref.reg<129> -> !qref.bit
    %56 = qref.get %0[ 38] : !qref.reg<129> -> !qref.bit
    %54 = qref.get %0[ 39] : !qref.reg<129> -> !qref.bit
    %52 = qref.get %0[ 40] : !qref.reg<129> -> !qref.bit
    %50 = qref.get %0[ 41] : !qref.reg<129> -> !qref.bit
    %48 = qref.get %0[ 42] : !qref.reg<129> -> !qref.bit
    %46 = qref.get %0[ 43] : !qref.reg<129> -> !qref.bit
    %44 = qref.get %0[ 44] : !qref.reg<129> -> !qref.bit
    %42 = qref.get %0[ 45] : !qref.reg<129> -> !qref.bit
    %40 = qref.get %0[ 46] : !qref.reg<129> -> !qref.bit
    %38 = qref.get %0[ 47] : !qref.reg<129> -> !qref.bit
    %36 = qref.get %0[ 48] : !qref.reg<129> -> !qref.bit
    %34 = qref.get %0[ 49] : !qref.reg<129> -> !qref.bit
    %32 = qref.get %0[ 50] : !qref.reg<129> -> !qref.bit
    %30 = qref.get %0[ 51] : !qref.reg<129> -> !qref.bit
    %28 = qref.get %0[ 52] : !qref.reg<129> -> !qref.bit
    %26 = qref.get %0[ 53] : !qref.reg<129> -> !qref.bit
    %24 = qref.get %0[ 54] : !qref.reg<129> -> !qref.bit
    %22 = qref.get %0[ 55] : !qref.reg<129> -> !qref.bit
    %20 = qref.get %0[ 56] : !qref.reg<129> -> !qref.bit
    %18 = qref.get %0[ 57] : !qref.reg<129> -> !qref.bit
    %16 = qref.get %0[ 58] : !qref.reg<129> -> !qref.bit
    %14 = qref.get %0[ 59] : !qref.reg<129> -> !qref.bit
    %12 = qref.get %0[ 60] : !qref.reg<129> -> !qref.bit
    %10 = qref.get %0[ 61] : !qref.reg<129> -> !qref.bit
    %8 = qref.get %0[ 62] : !qref.reg<129> -> !qref.bit
    %6 = qref.get %0[ 63] : !qref.reg<129> -> !qref.bit
    %129 = qref.get %0[ 64] : !qref.reg<129> -> !qref.bit
    %127 = qref.get %0[ 65] : !qref.reg<129> -> !qref.bit
    %125 = qref.get %0[ 66] : !qref.reg<129> -> !qref.bit
    %123 = qref.get %0[ 67] : !qref.reg<129> -> !qref.bit
    %121 = qref.get %0[ 68] : !qref.reg<129> -> !qref.bit
    %119 = qref.get %0[ 69] : !qref.reg<129> -> !qref.bit
    %117 = qref.get %0[ 70] : !qref.reg<129> -> !qref.bit
    %115 = qref.get %0[ 71] : !qref.reg<129> -> !qref.bit
    %113 = qref.get %0[ 72] : !qref.reg<129> -> !qref.bit
    %111 = qref.get %0[ 73] : !qref.reg<129> -> !qref.bit
    %109 = qref.get %0[ 74] : !qref.reg<129> -> !qref.bit
    %107 = qref.get %0[ 75] : !qref.reg<129> -> !qref.bit
    %105 = qref.get %0[ 76] : !qref.reg<129> -> !qref.bit
    %103 = qref.get %0[ 77] : !qref.reg<129> -> !qref.bit
    %101 = qref.get %0[ 78] : !qref.reg<129> -> !qref.bit
    %99 = qref.get %0[ 79] : !qref.reg<129> -> !qref.bit
    %97 = qref.get %0[ 80] : !qref.reg<129> -> !qref.bit
    %95 = qref.get %0[ 81] : !qref.reg<129> -> !qref.bit
    %93 = qref.get %0[ 82] : !qref.reg<129> -> !qref.bit
    %91 = qref.get %0[ 83] : !qref.reg<129> -> !qref.bit
    %89 = qref.get %0[ 84] : !qref.reg<129> -> !qref.bit
    %87 = qref.get %0[ 85] : !qref.reg<129> -> !qref.bit
    %85 = qref.get %0[ 86] : !qref.reg<129> -> !qref.bit
    %83 = qref.get %0[ 87] : !qref.reg<129> -> !qref.bit
    %81 = qref.get %0[ 88] : !qref.reg<129> -> !qref.bit
    %79 = qref.get %0[ 89] : !qref.reg<129> -> !qref.bit
    %77 = qref.get %0[ 90] : !qref.reg<129> -> !qref.bit
    %75 = qref.get %0[ 91] : !qref.reg<129> -> !qref.bit
    %73 = qref.get %0[ 92] : !qref.reg<129> -> !qref.bit
    %71 = qref.get %0[ 93] : !qref.reg<129> -> !qref.bit
    %69 = qref.get %0[ 94] : !qref.reg<129> -> !qref.bit
    %67 = qref.get %0[ 95] : !qref.reg<129> -> !qref.bit
    %65 = qref.get %0[ 96] : !qref.reg<129> -> !qref.bit
    %63 = qref.get %0[ 97] : !qref.reg<129> -> !qref.bit
    %61 = qref.get %0[ 98] : !qref.reg<129> -> !qref.bit
    %59 = qref.get %0[ 99] : !qref.reg<129> -> !qref.bit
    %57 = qref.get %0[ 100] : !qref.reg<129> -> !qref.bit
    %55 = qref.get %0[ 101] : !qref.reg<129> -> !qref.bit
    %53 = qref.get %0[ 102] : !qref.reg<129> -> !qref.bit
    %51 = qref.get %0[ 103] : !qref.reg<129> -> !qref.bit
    %49 = qref.get %0[ 104] : !qref.reg<129> -> !qref.bit
    %47 = qref.get %0[ 105] : !qref.reg<129> -> !qref.bit
    %45 = qref.get %0[ 106] : !qref.reg<129> -> !qref.bit
    %43 = qref.get %0[ 107] : !qref.reg<129> -> !qref.bit
    %41 = qref.get %0[ 108] : !qref.reg<129> -> !qref.bit
    %39 = qref.get %0[ 109] : !qref.reg<129> -> !qref.bit
    %37 = qref.get %0[ 110] : !qref.reg<129> -> !qref.bit
    %35 = qref.get %0[ 111] : !qref.reg<129> -> !qref.bit
    %33 = qref.get %0[ 112] : !qref.reg<129> -> !qref.bit
    %31 = qref.get %0[ 113] : !qref.reg<129> -> !qref.bit
    %29 = qref.get %0[ 114] : !qref.reg<129> -> !qref.bit
    %27 = qref.get %0[ 115] : !qref.reg<129> -> !qref.bit
    %25 = qref.get %0[ 116] : !qref.reg<129> -> !qref.bit
    %23 = qref.get %0[ 117] : !qref.reg<129> -> !qref.bit
    %21 = qref.get %0[ 118] : !qref.reg<129> -> !qref.bit
    %19 = qref.get %0[ 119] : !qref.reg<129> -> !qref.bit
    %17 = qref.get %0[ 120] : !qref.reg<129> -> !qref.bit
    %15 = qref.get %0[ 121] : !qref.reg<129> -> !qref.bit
    %13 = qref.get %0[ 122] : !qref.reg<129> -> !qref.bit
    %11 = qref.get %0[ 123] : !qref.reg<129> -> !qref.bit
    %9 = qref.get %0[ 124] : !qref.reg<129> -> !qref.bit
    %7 = qref.get %0[ 125] : !qref.reg<129> -> !qref.bit
    %5 = qref.get %0[ 128] : !qref.reg<129> -> !qref.bit

    qref.set_basis_state(%cst_false) %1 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %130 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %128 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %126 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %124 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %122 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %120 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %118 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %116 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %114 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %112 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %110 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %108 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %106 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %104 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %102 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %100 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %98 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %96 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %94 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %92 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %90 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %88 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %86 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %84 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %82 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %80 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %78 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %76 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %74 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %72 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %70 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %68 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %66 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %64 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %62 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %60 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %58 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %56 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %54 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %52 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %50 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %48 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %46 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %44 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %42 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %40 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %38 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %36 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %34 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %32 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %30 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %28 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %26 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %24 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %22 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %20 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %18 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %16 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %14 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %12 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %10 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %8 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %6 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %129 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %127 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %125 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %123 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %121 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %119 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %117 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %115 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %113 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %111 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %109 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %107 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %105 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %103 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %101 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %99 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %97 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %95 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %93 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %91 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %89 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %87 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %85 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %83 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %81 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %79 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %77 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %75 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %73 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %71 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %69 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %67 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %65 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %63 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %61 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %59 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %57 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %55 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %53 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %51 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %49 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %47 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %45 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %43 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %41 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %39 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %37 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %35 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %33 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %31 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %29 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %27 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %25 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %23 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %21 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %19 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %17 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %15 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %13 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %11 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %9 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %7 : tensor<1xi1>, !qref.bit
    qref.set_basis_state(%cst_false) %5 : tensor<1xi1>, !qref.bit

    qref.custom "PauliX"() %5 : !qref.bit

    %start = arith.constant 0 : index
    %step = arith.constant 1 : index
    %stop = arith.constant 3373259427 : index

    scf.for %i = %start to %stop step %step {

      qref.custom "Hadamard"() %5 : !qref.bit
      qref.custom "Hadamard"() %5 : !qref.bit
      qref.custom "T"() %5 : !qref.bit
      
      qref.custom "Hadamard"() %6 : !qref.bit
      qref.custom "T"() %6 : !qref.bit
      qref.custom "T"() %7 : !qref.bit
      qref.custom "CNOT"() %6, %7 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %7, %5 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %5, %6 : !qref.bit, !qref.bit
      qref.custom "T"() %6 adj : !qref.bit
      qref.custom "T"() %7 adj : !qref.bit
      qref.custom "CNOT"() %7, %6 : !qref.bit, !qref.bit
      qref.custom "T"() %6 adj : !qref.bit
      qref.custom "T"() %5 : !qref.bit
      qref.custom "CNOT"() %7, %5 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %5, %6 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %5 : !qref.bit
      qref.custom "Hadamard"() %5 : !qref.bit
      qref.custom "CNOT"() %6, %7 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %6 : !qref.bit
      qref.custom "PauliX"() %6 : !qref.bit
      qref.custom "Hadamard"() %6 : !qref.bit
      qref.custom "Hadamard"() %6 : !qref.bit
      qref.custom "T"() %6 : !qref.bit
      qref.custom "Hadamard"() %8 : !qref.bit
      qref.custom "Hadamard"() %8 : !qref.bit
      qref.custom "PauliX"() %8 : !qref.bit
      qref.custom "T"() %8 : !qref.bit
      qref.custom "Hadamard"() %9 : !qref.bit
      qref.custom "T"() %9 : !qref.bit
      qref.custom "Hadamard"() %10 : !qref.bit
      qref.custom "T"() %10 : !qref.bit
      qref.custom "Hadamard"() %11 : !qref.bit
      qref.custom "T"() %11 : !qref.bit
      qref.custom "Hadamard"() %12 : !qref.bit
      qref.custom "T"() %12 : !qref.bit
      qref.custom "Hadamard"() %13 : !qref.bit
      qref.custom "T"() %13 : !qref.bit
      
      qref.custom "Hadamard"() %14 : !qref.bit
      qref.custom "T"() %14 : !qref.bit
      
      qref.custom "Hadamard"() %15 : !qref.bit
      qref.custom "T"() %15 : !qref.bit
      
      qref.custom "Hadamard"() %16 : !qref.bit
      qref.custom "T"() %16 : !qref.bit
      qref.custom "Hadamard"() %17 : !qref.bit
      qref.custom "T"() %17 : !qref.bit
      qref.custom "Hadamard"() %18 : !qref.bit
      qref.custom "T"() %18 : !qref.bit
      qref.custom "Hadamard"() %19 : !qref.bit
      qref.custom "T"() %19 : !qref.bit
      qref.custom "Hadamard"() %20 : !qref.bit
      qref.custom "T"() %20 : !qref.bit
      qref.custom "Hadamard"() %21 : !qref.bit
      qref.custom "T"() %21 : !qref.bit
      qref.custom "Hadamard"() %22 : !qref.bit
      qref.custom "T"() %22 : !qref.bit
      qref.custom "Hadamard"() %23 : !qref.bit
      qref.custom "T"() %23 : !qref.bit
      qref.custom "Hadamard"() %24 : !qref.bit
      qref.custom "T"() %24 : !qref.bit
      qref.custom "Hadamard"() %25 : !qref.bit
      qref.custom "T"() %25 : !qref.bit
      qref.custom "Hadamard"() %26 : !qref.bit
      qref.custom "T"() %26 : !qref.bit
      qref.custom "Hadamard"() %27 : !qref.bit
      qref.custom "T"() %27 : !qref.bit
      qref.custom "Hadamard"() %28 : !qref.bit
      qref.custom "T"() %28 : !qref.bit
      qref.custom "Hadamard"() %29 : !qref.bit
      qref.custom "T"() %29 : !qref.bit
      qref.custom "Hadamard"() %30 : !qref.bit
      qref.custom "T"() %30 : !qref.bit
      qref.custom "Hadamard"() %31 : !qref.bit
      qref.custom "T"() %31 : !qref.bit
      qref.custom "Hadamard"() %32 : !qref.bit
      qref.custom "T"() %32 : !qref.bit
      qref.custom "Hadamard"() %33 : !qref.bit
      qref.custom "T"() %33 : !qref.bit
      qref.custom "Hadamard"() %34 : !qref.bit
      qref.custom "T"() %34 : !qref.bit
      qref.custom "Hadamard"() %35 : !qref.bit
      qref.custom "T"() %35 : !qref.bit
      qref.custom "Hadamard"() %36 : !qref.bit
      qref.custom "T"() %36 : !qref.bit
      qref.custom "Hadamard"() %37 : !qref.bit
      qref.custom "T"() %37 : !qref.bit
      qref.custom "Hadamard"() %38 : !qref.bit
      qref.custom "T"() %38 : !qref.bit
      qref.custom "Hadamard"() %39 : !qref.bit
      qref.custom "T"() %39 : !qref.bit
      qref.custom "Hadamard"() %40 : !qref.bit
      qref.custom "T"() %40 : !qref.bit
      qref.custom "Hadamard"() %41 : !qref.bit
      qref.custom "T"() %41 : !qref.bit
      qref.custom "Hadamard"() %42 : !qref.bit
      qref.custom "T"() %42 : !qref.bit
      qref.custom "Hadamard"() %43 : !qref.bit
      qref.custom "T"() %43 : !qref.bit
      qref.custom "Hadamard"() %44 : !qref.bit
      qref.custom "T"() %44 : !qref.bit
      qref.custom "Hadamard"() %45 : !qref.bit
      qref.custom "T"() %45 : !qref.bit
      qref.custom "Hadamard"() %46 : !qref.bit
      qref.custom "T"() %46 : !qref.bit
      qref.custom "Hadamard"() %47 : !qref.bit
      qref.custom "T"() %47 : !qref.bit
      qref.custom "Hadamard"() %48 : !qref.bit
      qref.custom "T"() %48 : !qref.bit
      qref.custom "Hadamard"() %49 : !qref.bit
      qref.custom "T"() %49 : !qref.bit
      qref.custom "Hadamard"() %50 : !qref.bit
      qref.custom "T"() %50 : !qref.bit
      qref.custom "Hadamard"() %51 : !qref.bit
      qref.custom "T"() %51 : !qref.bit
      qref.custom "Hadamard"() %52 : !qref.bit
      qref.custom "T"() %52 : !qref.bit
      qref.custom "Hadamard"() %53 : !qref.bit
      qref.custom "T"() %53 : !qref.bit
      qref.custom "Hadamard"() %54 : !qref.bit
      qref.custom "T"() %54 : !qref.bit
      qref.custom "Hadamard"() %55 : !qref.bit
      qref.custom "T"() %55 : !qref.bit
      qref.custom "Hadamard"() %56 : !qref.bit
      qref.custom "T"() %56 : !qref.bit
      qref.custom "Hadamard"() %57 : !qref.bit
      qref.custom "T"() %57 : !qref.bit
      qref.custom "Hadamard"() %58 : !qref.bit
      qref.custom "T"() %58 : !qref.bit
      qref.custom "Hadamard"() %59 : !qref.bit
      qref.custom "T"() %59 : !qref.bit
      qref.custom "Hadamard"() %60 : !qref.bit
      qref.custom "T"() %60 : !qref.bit
      qref.custom "Hadamard"() %61 : !qref.bit
      qref.custom "T"() %61 : !qref.bit
      qref.custom "Hadamard"() %62 : !qref.bit
      qref.custom "T"() %62 : !qref.bit
      qref.custom "Hadamard"() %63 : !qref.bit
      qref.custom "T"() %63 : !qref.bit
      qref.custom "Hadamard"() %64 : !qref.bit
      qref.custom "T"() %64 : !qref.bit
      qref.custom "Hadamard"() %65 : !qref.bit
      qref.custom "T"() %65 : !qref.bit
      qref.custom "Hadamard"() %66 : !qref.bit
      qref.custom "T"() %66 : !qref.bit
      qref.custom "Hadamard"() %67 : !qref.bit
      qref.custom "T"() %67 : !qref.bit
      qref.custom "Hadamard"() %68 : !qref.bit
      qref.custom "T"() %68 : !qref.bit
      qref.custom "Hadamard"() %69 : !qref.bit
      qref.custom "T"() %69 : !qref.bit
      qref.custom "Hadamard"() %70 : !qref.bit
      qref.custom "T"() %70 : !qref.bit
      qref.custom "Hadamard"() %71 : !qref.bit
      qref.custom "T"() %71 : !qref.bit
      qref.custom "Hadamard"() %72 : !qref.bit
      qref.custom "T"() %72 : !qref.bit
      qref.custom "Hadamard"() %73 : !qref.bit
      qref.custom "T"() %73 : !qref.bit
      qref.custom "Hadamard"() %74 : !qref.bit
      qref.custom "T"() %74 : !qref.bit
      qref.custom "Hadamard"() %75 : !qref.bit
      qref.custom "T"() %75 : !qref.bit
      qref.custom "Hadamard"() %76 : !qref.bit
      qref.custom "T"() %76 : !qref.bit
      qref.custom "Hadamard"() %77 : !qref.bit
      qref.custom "T"() %77 : !qref.bit
      qref.custom "Hadamard"() %78 : !qref.bit
      qref.custom "T"() %78 : !qref.bit
      qref.custom "Hadamard"() %79 : !qref.bit
      qref.custom "T"() %79 : !qref.bit
      qref.custom "Hadamard"() %80 : !qref.bit
      qref.custom "T"() %80 : !qref.bit
      qref.custom "Hadamard"() %81 : !qref.bit
      qref.custom "T"() %81 : !qref.bit
      qref.custom "Hadamard"() %82 : !qref.bit
      qref.custom "T"() %82 : !qref.bit
      qref.custom "Hadamard"() %83 : !qref.bit
      qref.custom "T"() %83 : !qref.bit
      qref.custom "Hadamard"() %84 : !qref.bit
      qref.custom "T"() %84 : !qref.bit
      qref.custom "Hadamard"() %85 : !qref.bit
      qref.custom "T"() %85 : !qref.bit
      qref.custom "Hadamard"() %86 : !qref.bit
      qref.custom "T"() %86 : !qref.bit
      qref.custom "Hadamard"() %87 : !qref.bit
      qref.custom "T"() %87 : !qref.bit
      qref.custom "Hadamard"() %88 : !qref.bit
      qref.custom "T"() %88 : !qref.bit
      qref.custom "Hadamard"() %89 : !qref.bit
      qref.custom "T"() %89 : !qref.bit
      qref.custom "Hadamard"() %90 : !qref.bit
      qref.custom "T"() %90 : !qref.bit
      qref.custom "Hadamard"() %91 : !qref.bit
      qref.custom "T"() %91 : !qref.bit
      qref.custom "Hadamard"() %92 : !qref.bit
      qref.custom "T"() %92 : !qref.bit
      qref.custom "Hadamard"() %93 : !qref.bit
      qref.custom "T"() %93 : !qref.bit
      qref.custom "Hadamard"() %94 : !qref.bit
      qref.custom "T"() %94 : !qref.bit
      qref.custom "Hadamard"() %95 : !qref.bit
      qref.custom "T"() %95 : !qref.bit
      qref.custom "Hadamard"() %96 : !qref.bit
      qref.custom "T"() %96 : !qref.bit
      qref.custom "Hadamard"() %97 : !qref.bit
      qref.custom "T"() %97 : !qref.bit
      qref.custom "Hadamard"() %98 : !qref.bit
      qref.custom "T"() %98 : !qref.bit
      qref.custom "Hadamard"() %99 : !qref.bit
      qref.custom "T"() %99 : !qref.bit
      qref.custom "Hadamard"() %100 : !qref.bit
      qref.custom "T"() %100 : !qref.bit
      qref.custom "Hadamard"() %101 : !qref.bit
      qref.custom "T"() %101 : !qref.bit
      qref.custom "Hadamard"() %102 : !qref.bit
      qref.custom "T"() %102 : !qref.bit
      qref.custom "Hadamard"() %103 : !qref.bit
      qref.custom "T"() %103 : !qref.bit
      qref.custom "Hadamard"() %104 : !qref.bit
      qref.custom "T"() %104 : !qref.bit
      qref.custom "Hadamard"() %105 : !qref.bit
      qref.custom "T"() %105 : !qref.bit
      qref.custom "Hadamard"() %106 : !qref.bit
      qref.custom "T"() %106 : !qref.bit
      qref.custom "Hadamard"() %107 : !qref.bit
      qref.custom "T"() %107 : !qref.bit
      qref.custom "Hadamard"() %108 : !qref.bit
      qref.custom "T"() %108 : !qref.bit
      qref.custom "Hadamard"() %109 : !qref.bit
      qref.custom "T"() %109 : !qref.bit
      qref.custom "Hadamard"() %110 : !qref.bit
      qref.custom "T"() %110 : !qref.bit
      qref.custom "Hadamard"() %111 : !qref.bit
      qref.custom "T"() %111 : !qref.bit
      qref.custom "Hadamard"() %112 : !qref.bit
      qref.custom "T"() %112 : !qref.bit
      qref.custom "Hadamard"() %113 : !qref.bit
      qref.custom "T"() %113 : !qref.bit
      qref.custom "Hadamard"() %114 : !qref.bit
      qref.custom "T"() %114 : !qref.bit
      qref.custom "Hadamard"() %115 : !qref.bit
      qref.custom "T"() %115 : !qref.bit
      qref.custom "Hadamard"() %116 : !qref.bit
      qref.custom "T"() %116 : !qref.bit
      qref.custom "Hadamard"() %117 : !qref.bit
      qref.custom "T"() %117 : !qref.bit
      qref.custom "Hadamard"() %118 : !qref.bit
      qref.custom "T"() %118 : !qref.bit
      qref.custom "Hadamard"() %119 : !qref.bit
      qref.custom "T"() %119 : !qref.bit
      qref.custom "Hadamard"() %120 : !qref.bit
      qref.custom "T"() %120 : !qref.bit
      qref.custom "Hadamard"() %121 : !qref.bit
      qref.custom "T"() %121 : !qref.bit
      qref.custom "Hadamard"() %122 : !qref.bit
      qref.custom "T"() %122 : !qref.bit
      qref.custom "Hadamard"() %123 : !qref.bit
      qref.custom "T"() %123 : !qref.bit
      qref.custom "Hadamard"() %124 : !qref.bit
      qref.custom "T"() %124 : !qref.bit
      qref.custom "Hadamard"() %125 : !qref.bit
      qref.custom "T"() %125 : !qref.bit
      qref.custom "Hadamard"() %126 : !qref.bit
      qref.custom "T"() %126 : !qref.bit
      qref.custom "Hadamard"() %127 : !qref.bit
      qref.custom "T"() %127 : !qref.bit
      qref.custom "Hadamard"() %128 : !qref.bit
      qref.custom "T"() %128 : !qref.bit
      qref.custom "Hadamard"() %129 : !qref.bit
      qref.custom "T"() %129 : !qref.bit
      qref.custom "Hadamard"() %1 : !qref.bit
      qref.custom "T"() %1 : !qref.bit
      qref.custom "Hadamard"() %130 : !qref.bit
      qref.custom "T"() %130 : !qref.bit
      qref.custom "CNOT"() %1, %130 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %130, %129 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %129, %1 : !qref.bit, !qref.bit
      qref.custom "T"() %1 adj : !qref.bit
      qref.custom "T"() %130 adj : !qref.bit
      qref.custom "CNOT"() %130, %1 : !qref.bit, !qref.bit
      qref.custom "T"() %1 adj : !qref.bit
      qref.custom "T"() %129 : !qref.bit
      qref.custom "CNOT"() %130, %129 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %129, %1 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %129 : !qref.bit
      qref.custom "T"() %129 : !qref.bit
      qref.custom "CNOT"() %128, %129 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %129, %127 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %127, %128 : !qref.bit, !qref.bit
      qref.custom "T"() %128 adj : !qref.bit
      qref.custom "T"() %129 adj : !qref.bit
      qref.custom "CNOT"() %129, %128 : !qref.bit, !qref.bit
      qref.custom "T"() %128 adj : !qref.bit
      qref.custom "T"() %127 : !qref.bit
      qref.custom "CNOT"() %129, %127 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %127, %128 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %127 : !qref.bit
      qref.custom "T"() %127 : !qref.bit
      qref.custom "CNOT"() %126, %127 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %127, %125 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %125, %126 : !qref.bit, !qref.bit
      qref.custom "T"() %126 adj : !qref.bit
      qref.custom "T"() %127 adj : !qref.bit
      qref.custom "CNOT"() %127, %126 : !qref.bit, !qref.bit
      qref.custom "T"() %126 adj : !qref.bit
      qref.custom "T"() %125 : !qref.bit
      qref.custom "CNOT"() %127, %125 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %125, %126 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %125 : !qref.bit
      qref.custom "T"() %125 : !qref.bit
      qref.custom "CNOT"() %124, %125 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %125, %123 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %123, %124 : !qref.bit, !qref.bit
      qref.custom "T"() %124 adj : !qref.bit
      qref.custom "T"() %125 adj : !qref.bit
      qref.custom "CNOT"() %125, %124 : !qref.bit, !qref.bit
      qref.custom "T"() %124 adj : !qref.bit
      qref.custom "T"() %123 : !qref.bit
      qref.custom "CNOT"() %125, %123 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %123, %124 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %123 : !qref.bit
      qref.custom "T"() %123 : !qref.bit
      qref.custom "CNOT"() %122, %123 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %123, %121 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %121, %122 : !qref.bit, !qref.bit
      qref.custom "T"() %122 adj : !qref.bit
      qref.custom "T"() %123 adj : !qref.bit
      qref.custom "CNOT"() %123, %122 : !qref.bit, !qref.bit
      qref.custom "T"() %122 adj : !qref.bit
      qref.custom "T"() %121 : !qref.bit
      qref.custom "CNOT"() %123, %121 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %121, %122 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %121 : !qref.bit
      qref.custom "T"() %121 : !qref.bit
      qref.custom "CNOT"() %120, %121 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %121, %119 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %119, %120 : !qref.bit, !qref.bit
      qref.custom "T"() %120 adj : !qref.bit
      qref.custom "T"() %121 adj : !qref.bit
      qref.custom "CNOT"() %121, %120 : !qref.bit, !qref.bit
      qref.custom "T"() %120 adj : !qref.bit
      qref.custom "T"() %119 : !qref.bit
      qref.custom "CNOT"() %121, %119 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %119, %120 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %119 : !qref.bit
      qref.custom "T"() %119 : !qref.bit
      qref.custom "CNOT"() %118, %119 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %119, %117 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %117, %118 : !qref.bit, !qref.bit
      qref.custom "T"() %118 adj : !qref.bit
      qref.custom "T"() %119 adj : !qref.bit
      qref.custom "CNOT"() %119, %118 : !qref.bit, !qref.bit
      qref.custom "T"() %118 adj : !qref.bit
      qref.custom "T"() %117 : !qref.bit
      qref.custom "CNOT"() %119, %117 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %117, %118 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %117 : !qref.bit
      qref.custom "T"() %117 : !qref.bit
      qref.custom "CNOT"() %116, %117 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %117, %115 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %115, %116 : !qref.bit, !qref.bit
      qref.custom "T"() %116 adj : !qref.bit
      qref.custom "T"() %117 adj : !qref.bit
      qref.custom "CNOT"() %117, %116 : !qref.bit, !qref.bit
      qref.custom "T"() %116 adj : !qref.bit
      qref.custom "T"() %115 : !qref.bit
      qref.custom "CNOT"() %117, %115 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %115, %116 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %115 : !qref.bit
      qref.custom "T"() %115 : !qref.bit
      qref.custom "CNOT"() %114, %115 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %115, %113 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %113, %114 : !qref.bit, !qref.bit
      qref.custom "T"() %114 adj : !qref.bit
      qref.custom "T"() %115 adj : !qref.bit
      qref.custom "CNOT"() %115, %114 : !qref.bit, !qref.bit
      qref.custom "T"() %114 adj : !qref.bit
      qref.custom "T"() %113 : !qref.bit
      qref.custom "CNOT"() %115, %113 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %113, %114 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %113 : !qref.bit
      qref.custom "T"() %113 : !qref.bit
      qref.custom "CNOT"() %112, %113 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %113, %111 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %111, %112 : !qref.bit, !qref.bit
      qref.custom "T"() %112 adj : !qref.bit
      qref.custom "T"() %113 adj : !qref.bit
      qref.custom "CNOT"() %113, %112 : !qref.bit, !qref.bit
      qref.custom "T"() %112 adj : !qref.bit
      qref.custom "T"() %111 : !qref.bit
      qref.custom "CNOT"() %113, %111 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %111, %112 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %111 : !qref.bit
      qref.custom "T"() %111 : !qref.bit
      qref.custom "CNOT"() %110, %111 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %111, %109 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %109, %110 : !qref.bit, !qref.bit
      qref.custom "T"() %110 adj : !qref.bit
      qref.custom "T"() %111 adj : !qref.bit
      qref.custom "CNOT"() %111, %110 : !qref.bit, !qref.bit
      qref.custom "T"() %110 adj : !qref.bit
      qref.custom "T"() %109 : !qref.bit
      qref.custom "CNOT"() %111, %109 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %109, %110 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %109 : !qref.bit
      qref.custom "T"() %109 : !qref.bit
      qref.custom "CNOT"() %108, %109 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %109, %107 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %107, %108 : !qref.bit, !qref.bit
      qref.custom "T"() %108 adj : !qref.bit
      qref.custom "T"() %109 adj : !qref.bit
      qref.custom "CNOT"() %109, %108 : !qref.bit, !qref.bit
      qref.custom "T"() %108 adj : !qref.bit
      qref.custom "T"() %107 : !qref.bit
      qref.custom "CNOT"() %109, %107 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %107, %108 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %107 : !qref.bit
      qref.custom "T"() %107 : !qref.bit
      qref.custom "CNOT"() %106, %107 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %107, %105 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %105, %106 : !qref.bit, !qref.bit
      qref.custom "T"() %106 adj : !qref.bit
      qref.custom "T"() %107 adj : !qref.bit
      qref.custom "CNOT"() %107, %106 : !qref.bit, !qref.bit
      qref.custom "T"() %106 adj : !qref.bit
      qref.custom "T"() %105 : !qref.bit
      qref.custom "CNOT"() %107, %105 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %105, %106 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %105 : !qref.bit
      qref.custom "T"() %105 : !qref.bit
      qref.custom "CNOT"() %104, %105 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %105, %103 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %103, %104 : !qref.bit, !qref.bit
      qref.custom "T"() %104 adj : !qref.bit
      qref.custom "T"() %105 adj : !qref.bit
      qref.custom "CNOT"() %105, %104 : !qref.bit, !qref.bit
      qref.custom "T"() %104 adj : !qref.bit
      qref.custom "T"() %103 : !qref.bit
      qref.custom "CNOT"() %105, %103 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %103, %104 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %103 : !qref.bit
      qref.custom "T"() %103 : !qref.bit
      qref.custom "CNOT"() %102, %103 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %103, %101 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %101, %102 : !qref.bit, !qref.bit
      qref.custom "T"() %102 adj : !qref.bit
      qref.custom "T"() %103 adj : !qref.bit
      qref.custom "CNOT"() %103, %102 : !qref.bit, !qref.bit
      qref.custom "T"() %102 adj : !qref.bit
      qref.custom "T"() %101 : !qref.bit
      qref.custom "CNOT"() %103, %101 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %101, %102 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %101 : !qref.bit
      qref.custom "T"() %101 : !qref.bit
      qref.custom "CNOT"() %100, %101 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %101, %99 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %99, %100 : !qref.bit, !qref.bit
      qref.custom "T"() %100 adj : !qref.bit
      qref.custom "T"() %101 adj : !qref.bit
      qref.custom "CNOT"() %101, %100 : !qref.bit, !qref.bit
      qref.custom "T"() %100 adj : !qref.bit
      qref.custom "T"() %99 : !qref.bit
      qref.custom "CNOT"() %101, %99 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %99, %100 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %99 : !qref.bit
      qref.custom "T"() %99 : !qref.bit
      qref.custom "CNOT"() %98, %99 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %99, %97 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %97, %98 : !qref.bit, !qref.bit
      qref.custom "T"() %98 adj : !qref.bit
      qref.custom "T"() %99 adj : !qref.bit
      qref.custom "CNOT"() %99, %98 : !qref.bit, !qref.bit
      qref.custom "T"() %98 adj : !qref.bit
      qref.custom "T"() %97 : !qref.bit
      qref.custom "CNOT"() %99, %97 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %97, %98 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %97 : !qref.bit
      qref.custom "T"() %97 : !qref.bit
      qref.custom "CNOT"() %96, %97 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %97, %95 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %95, %96 : !qref.bit, !qref.bit
      qref.custom "T"() %96 adj : !qref.bit
      qref.custom "T"() %97 adj : !qref.bit
      qref.custom "CNOT"() %97, %96 : !qref.bit, !qref.bit
      qref.custom "T"() %96 adj : !qref.bit
      qref.custom "T"() %95 : !qref.bit
      qref.custom "CNOT"() %97, %95 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %95, %96 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %95 : !qref.bit
      qref.custom "T"() %95 : !qref.bit
      qref.custom "CNOT"() %94, %95 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %95, %93 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %93, %94 : !qref.bit, !qref.bit
      qref.custom "T"() %94 adj : !qref.bit
      qref.custom "T"() %95 adj : !qref.bit
      qref.custom "CNOT"() %95, %94 : !qref.bit, !qref.bit
      qref.custom "T"() %94 adj : !qref.bit
      qref.custom "T"() %93 : !qref.bit
      qref.custom "CNOT"() %95, %93 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %93, %94 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %93 : !qref.bit
      qref.custom "T"() %93 : !qref.bit
      qref.custom "CNOT"() %92, %93 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %93, %91 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %91, %92 : !qref.bit, !qref.bit
      qref.custom "T"() %92 adj : !qref.bit
      qref.custom "T"() %93 adj : !qref.bit
      qref.custom "CNOT"() %93, %92 : !qref.bit, !qref.bit
      qref.custom "T"() %92 adj : !qref.bit
      qref.custom "T"() %91 : !qref.bit
      qref.custom "CNOT"() %93, %91 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %91, %92 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %91 : !qref.bit
      qref.custom "T"() %91 : !qref.bit
      qref.custom "CNOT"() %90, %91 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %91, %89 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %89, %90 : !qref.bit, !qref.bit
      qref.custom "T"() %90 adj : !qref.bit
      qref.custom "T"() %91 adj : !qref.bit
      qref.custom "CNOT"() %91, %90 : !qref.bit, !qref.bit
      qref.custom "T"() %90 adj : !qref.bit
      qref.custom "T"() %89 : !qref.bit
      qref.custom "CNOT"() %91, %89 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %89, %90 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %89 : !qref.bit
      qref.custom "T"() %89 : !qref.bit
      qref.custom "CNOT"() %88, %89 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %89, %87 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %87, %88 : !qref.bit, !qref.bit
      qref.custom "T"() %88 adj : !qref.bit
      qref.custom "T"() %89 adj : !qref.bit
      qref.custom "CNOT"() %89, %88 : !qref.bit, !qref.bit
      qref.custom "T"() %88 adj : !qref.bit
      qref.custom "T"() %87 : !qref.bit
      qref.custom "CNOT"() %89, %87 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %87, %88 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %87 : !qref.bit
      qref.custom "T"() %87 : !qref.bit
      qref.custom "CNOT"() %86, %87 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %87, %85 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %85, %86 : !qref.bit, !qref.bit
      qref.custom "T"() %86 adj : !qref.bit
      qref.custom "T"() %87 adj : !qref.bit
      qref.custom "CNOT"() %87, %86 : !qref.bit, !qref.bit
      qref.custom "T"() %86 adj : !qref.bit
      qref.custom "T"() %85 : !qref.bit
      qref.custom "CNOT"() %87, %85 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %85, %86 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %85 : !qref.bit
      qref.custom "T"() %85 : !qref.bit
      qref.custom "CNOT"() %84, %85 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %85, %83 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %83, %84 : !qref.bit, !qref.bit
      qref.custom "T"() %84 adj : !qref.bit
      qref.custom "T"() %85 adj : !qref.bit
      qref.custom "CNOT"() %85, %84 : !qref.bit, !qref.bit
      qref.custom "T"() %84 adj : !qref.bit
      qref.custom "T"() %83 : !qref.bit
      qref.custom "CNOT"() %85, %83 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %83, %84 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %83 : !qref.bit
      qref.custom "T"() %83 : !qref.bit
      qref.custom "CNOT"() %82, %83 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %83, %81 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %81, %82 : !qref.bit, !qref.bit
      qref.custom "T"() %82 adj : !qref.bit
      qref.custom "T"() %83 adj : !qref.bit
      qref.custom "CNOT"() %83, %82 : !qref.bit, !qref.bit
      qref.custom "T"() %82 adj : !qref.bit
      qref.custom "T"() %81 : !qref.bit
      qref.custom "CNOT"() %83, %81 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %81, %82 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %81 : !qref.bit
      qref.custom "T"() %81 : !qref.bit
      qref.custom "CNOT"() %80, %81 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %81, %79 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %79, %80 : !qref.bit, !qref.bit
      qref.custom "T"() %80 adj : !qref.bit
      qref.custom "T"() %81 adj : !qref.bit
      qref.custom "CNOT"() %81, %80 : !qref.bit, !qref.bit
      qref.custom "T"() %80 adj : !qref.bit
      qref.custom "T"() %79 : !qref.bit
      qref.custom "CNOT"() %81, %79 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %79, %80 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %79 : !qref.bit
      qref.custom "T"() %79 : !qref.bit
      qref.custom "CNOT"() %78, %79 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %79, %77 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %77, %78 : !qref.bit, !qref.bit
      qref.custom "T"() %78 adj : !qref.bit
      qref.custom "T"() %79 adj : !qref.bit
      qref.custom "CNOT"() %79, %78 : !qref.bit, !qref.bit
      qref.custom "T"() %78 adj : !qref.bit
      qref.custom "T"() %77 : !qref.bit
      qref.custom "CNOT"() %79, %77 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %77, %78 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %77 : !qref.bit
      qref.custom "T"() %77 : !qref.bit
      qref.custom "CNOT"() %76, %77 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %77, %75 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %75, %76 : !qref.bit, !qref.bit
      qref.custom "T"() %76 adj : !qref.bit
      qref.custom "T"() %77 adj : !qref.bit
      qref.custom "CNOT"() %77, %76 : !qref.bit, !qref.bit
      qref.custom "T"() %76 adj : !qref.bit
      qref.custom "T"() %75 : !qref.bit
      qref.custom "CNOT"() %77, %75 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %75, %76 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %75 : !qref.bit
      qref.custom "T"() %75 : !qref.bit
      qref.custom "CNOT"() %74, %75 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %75, %73 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %73, %74 : !qref.bit, !qref.bit
      qref.custom "T"() %74 adj : !qref.bit
      qref.custom "T"() %75 adj : !qref.bit
      qref.custom "CNOT"() %75, %74 : !qref.bit, !qref.bit
      qref.custom "T"() %74 adj : !qref.bit
      qref.custom "T"() %73 : !qref.bit
      qref.custom "CNOT"() %75, %73 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %73, %74 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %73 : !qref.bit
      qref.custom "T"() %73 : !qref.bit
      qref.custom "CNOT"() %72, %73 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %73, %71 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %71, %72 : !qref.bit, !qref.bit
      qref.custom "T"() %72 adj : !qref.bit
      qref.custom "T"() %73 adj : !qref.bit
      qref.custom "CNOT"() %73, %72 : !qref.bit, !qref.bit
      qref.custom "T"() %72 adj : !qref.bit
      qref.custom "T"() %71 : !qref.bit
      qref.custom "CNOT"() %73, %71 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %71, %72 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %71 : !qref.bit
      qref.custom "T"() %71 : !qref.bit
      qref.custom "CNOT"() %70, %71 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %71, %69 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %69, %70 : !qref.bit, !qref.bit
      qref.custom "T"() %70 adj : !qref.bit
      qref.custom "T"() %71 adj : !qref.bit
      qref.custom "CNOT"() %71, %70 : !qref.bit, !qref.bit
      qref.custom "T"() %70 adj : !qref.bit
      qref.custom "T"() %69 : !qref.bit
      qref.custom "CNOT"() %71, %69 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %69, %70 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %69 : !qref.bit
      qref.custom "T"() %69 : !qref.bit
      qref.custom "CNOT"() %68, %69 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %69, %67 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %67, %68 : !qref.bit, !qref.bit
      qref.custom "T"() %68 adj : !qref.bit
      qref.custom "T"() %69 adj : !qref.bit
      qref.custom "CNOT"() %69, %68 : !qref.bit, !qref.bit
      qref.custom "T"() %68 adj : !qref.bit
      qref.custom "T"() %67 : !qref.bit
      qref.custom "CNOT"() %69, %67 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %67, %68 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %67 : !qref.bit
      qref.custom "T"() %67 : !qref.bit
      qref.custom "CNOT"() %66, %67 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %67, %65 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %65, %66 : !qref.bit, !qref.bit
      qref.custom "T"() %66 adj : !qref.bit
      qref.custom "T"() %67 adj : !qref.bit
      qref.custom "CNOT"() %67, %66 : !qref.bit, !qref.bit
      qref.custom "T"() %66 adj : !qref.bit
      qref.custom "T"() %65 : !qref.bit
      qref.custom "CNOT"() %67, %65 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %65, %66 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %65 : !qref.bit
      qref.custom "T"() %65 : !qref.bit
      qref.custom "CNOT"() %64, %65 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %65, %63 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %63, %64 : !qref.bit, !qref.bit
      qref.custom "T"() %64 adj : !qref.bit
      qref.custom "T"() %65 adj : !qref.bit
      qref.custom "CNOT"() %65, %64 : !qref.bit, !qref.bit
      qref.custom "T"() %64 adj : !qref.bit
      qref.custom "T"() %63 : !qref.bit
      qref.custom "CNOT"() %65, %63 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %63, %64 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %63 : !qref.bit
      qref.custom "T"() %63 : !qref.bit
      qref.custom "CNOT"() %62, %63 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %63, %61 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %61, %62 : !qref.bit, !qref.bit
      qref.custom "T"() %62 adj : !qref.bit
      qref.custom "T"() %63 adj : !qref.bit
      qref.custom "CNOT"() %63, %62 : !qref.bit, !qref.bit
      qref.custom "T"() %62 adj : !qref.bit
      qref.custom "T"() %61 : !qref.bit
      qref.custom "CNOT"() %63, %61 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %61, %62 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %61 : !qref.bit
      qref.custom "T"() %61 : !qref.bit
      qref.custom "CNOT"() %60, %61 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %61, %59 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %59, %60 : !qref.bit, !qref.bit
      qref.custom "T"() %60 adj : !qref.bit
      qref.custom "T"() %61 adj : !qref.bit
      qref.custom "CNOT"() %61, %60 : !qref.bit, !qref.bit
      qref.custom "T"() %60 adj : !qref.bit
      qref.custom "T"() %59 : !qref.bit
      qref.custom "CNOT"() %61, %59 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %59, %60 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %59 : !qref.bit
      qref.custom "T"() %59 : !qref.bit
      qref.custom "CNOT"() %58, %59 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %59, %57 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %57, %58 : !qref.bit, !qref.bit
      qref.custom "T"() %58 adj : !qref.bit
      qref.custom "T"() %59 adj : !qref.bit
      qref.custom "CNOT"() %59, %58 : !qref.bit, !qref.bit
      qref.custom "T"() %58 adj : !qref.bit
      qref.custom "T"() %57 : !qref.bit
      qref.custom "CNOT"() %59, %57 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %57, %58 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %57 : !qref.bit
      qref.custom "T"() %57 : !qref.bit
      qref.custom "CNOT"() %56, %57 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %57, %55 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %55, %56 : !qref.bit, !qref.bit
      qref.custom "T"() %56 adj : !qref.bit
      qref.custom "T"() %57 adj : !qref.bit
      qref.custom "CNOT"() %57, %56 : !qref.bit, !qref.bit
      qref.custom "T"() %56 adj : !qref.bit
      qref.custom "T"() %55 : !qref.bit
      qref.custom "CNOT"() %57, %55 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %55, %56 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %55 : !qref.bit
      qref.custom "T"() %55 : !qref.bit
      qref.custom "CNOT"() %54, %55 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %55, %53 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %53, %54 : !qref.bit, !qref.bit
      qref.custom "T"() %54 adj : !qref.bit
      qref.custom "T"() %55 adj : !qref.bit
      qref.custom "CNOT"() %55, %54 : !qref.bit, !qref.bit
      qref.custom "T"() %54 adj : !qref.bit
      qref.custom "T"() %53 : !qref.bit
      qref.custom "CNOT"() %55, %53 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %53, %54 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %53 : !qref.bit
      qref.custom "T"() %53 : !qref.bit
      qref.custom "CNOT"() %52, %53 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %53, %51 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %51, %52 : !qref.bit, !qref.bit
      qref.custom "T"() %52 adj : !qref.bit
      qref.custom "T"() %53 adj : !qref.bit
      qref.custom "CNOT"() %53, %52 : !qref.bit, !qref.bit
      qref.custom "T"() %52 adj : !qref.bit
      qref.custom "T"() %51 : !qref.bit
      qref.custom "CNOT"() %53, %51 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %51, %52 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %51 : !qref.bit
      qref.custom "T"() %51 : !qref.bit
      qref.custom "CNOT"() %50, %51 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %51, %49 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %49, %50 : !qref.bit, !qref.bit
      qref.custom "T"() %50 adj : !qref.bit
      qref.custom "T"() %51 adj : !qref.bit
      qref.custom "CNOT"() %51, %50 : !qref.bit, !qref.bit
      qref.custom "T"() %50 adj : !qref.bit
      qref.custom "T"() %49 : !qref.bit
      qref.custom "CNOT"() %51, %49 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %49, %50 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %49 : !qref.bit
      qref.custom "T"() %49 : !qref.bit
      qref.custom "CNOT"() %48, %49 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %49, %47 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %47, %48 : !qref.bit, !qref.bit
      qref.custom "T"() %48 adj : !qref.bit
      qref.custom "T"() %49 adj : !qref.bit
      qref.custom "CNOT"() %49, %48 : !qref.bit, !qref.bit
      qref.custom "T"() %48 adj : !qref.bit
      qref.custom "T"() %47 : !qref.bit
      qref.custom "CNOT"() %49, %47 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %47, %48 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %47 : !qref.bit
      qref.custom "T"() %47 : !qref.bit
      qref.custom "CNOT"() %46, %47 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %47, %45 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %45, %46 : !qref.bit, !qref.bit
      qref.custom "T"() %46 adj : !qref.bit
      qref.custom "T"() %47 adj : !qref.bit
      qref.custom "CNOT"() %47, %46 : !qref.bit, !qref.bit
      qref.custom "T"() %46 adj : !qref.bit
      qref.custom "T"() %45 : !qref.bit
      qref.custom "CNOT"() %47, %45 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %45, %46 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %45 : !qref.bit
      qref.custom "T"() %45 : !qref.bit
      qref.custom "CNOT"() %44, %45 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %45, %43 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %43, %44 : !qref.bit, !qref.bit
      qref.custom "T"() %44 adj : !qref.bit
      qref.custom "T"() %45 adj : !qref.bit
      qref.custom "CNOT"() %45, %44 : !qref.bit, !qref.bit
      qref.custom "T"() %44 adj : !qref.bit
      qref.custom "T"() %43 : !qref.bit
      qref.custom "CNOT"() %45, %43 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %43, %44 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %43 : !qref.bit
      qref.custom "T"() %43 : !qref.bit
      qref.custom "CNOT"() %42, %43 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %43, %41 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %41, %42 : !qref.bit, !qref.bit
      qref.custom "T"() %42 adj : !qref.bit
      qref.custom "T"() %43 adj : !qref.bit
      qref.custom "CNOT"() %43, %42 : !qref.bit, !qref.bit
      qref.custom "T"() %42 adj : !qref.bit
      qref.custom "T"() %41 : !qref.bit
      qref.custom "CNOT"() %43, %41 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %41, %42 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %41 : !qref.bit
      qref.custom "T"() %41 : !qref.bit
      qref.custom "CNOT"() %40, %41 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %41, %39 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %39, %40 : !qref.bit, !qref.bit
      qref.custom "T"() %40 adj : !qref.bit
      qref.custom "T"() %41 adj : !qref.bit
      qref.custom "CNOT"() %41, %40 : !qref.bit, !qref.bit
      qref.custom "T"() %40 adj : !qref.bit
      qref.custom "T"() %39 : !qref.bit
      qref.custom "CNOT"() %41, %39 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %39, %40 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %39 : !qref.bit
      qref.custom "T"() %39 : !qref.bit
      qref.custom "CNOT"() %38, %39 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %39, %37 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %37, %38 : !qref.bit, !qref.bit
      qref.custom "T"() %38 adj : !qref.bit
      qref.custom "T"() %39 adj : !qref.bit
      qref.custom "CNOT"() %39, %38 : !qref.bit, !qref.bit
      qref.custom "T"() %38 adj : !qref.bit
      qref.custom "T"() %37 : !qref.bit
      qref.custom "CNOT"() %39, %37 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %37, %38 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %37 : !qref.bit
      qref.custom "T"() %37 : !qref.bit
      qref.custom "CNOT"() %36, %37 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %37, %35 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %35, %36 : !qref.bit, !qref.bit
      qref.custom "T"() %36 adj : !qref.bit
      qref.custom "T"() %37 adj : !qref.bit
      qref.custom "CNOT"() %37, %36 : !qref.bit, !qref.bit
      qref.custom "T"() %36 adj : !qref.bit
      qref.custom "T"() %35 : !qref.bit
      qref.custom "CNOT"() %37, %35 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %35, %36 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %35 : !qref.bit
      qref.custom "T"() %35 : !qref.bit
      qref.custom "CNOT"() %34, %35 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %35, %33 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %33, %34 : !qref.bit, !qref.bit
      qref.custom "T"() %34 adj : !qref.bit
      qref.custom "T"() %35 adj : !qref.bit
      qref.custom "CNOT"() %35, %34 : !qref.bit, !qref.bit
      qref.custom "T"() %34 adj : !qref.bit
      qref.custom "T"() %33 : !qref.bit
      qref.custom "CNOT"() %35, %33 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %33, %34 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %33 : !qref.bit
      qref.custom "T"() %33 : !qref.bit
      qref.custom "CNOT"() %32, %33 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %33, %31 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %31, %32 : !qref.bit, !qref.bit
      qref.custom "T"() %32 adj : !qref.bit
      qref.custom "T"() %33 adj : !qref.bit
      qref.custom "CNOT"() %33, %32 : !qref.bit, !qref.bit
      qref.custom "T"() %32 adj : !qref.bit
      qref.custom "T"() %31 : !qref.bit
      qref.custom "CNOT"() %33, %31 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %31, %32 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %31 : !qref.bit
      qref.custom "T"() %31 : !qref.bit
      qref.custom "CNOT"() %30, %31 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %31, %29 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %29, %30 : !qref.bit, !qref.bit
      qref.custom "T"() %30 adj : !qref.bit
      qref.custom "T"() %31 adj : !qref.bit
      qref.custom "CNOT"() %31, %30 : !qref.bit, !qref.bit
      qref.custom "T"() %30 adj : !qref.bit
      qref.custom "T"() %29 : !qref.bit
      qref.custom "CNOT"() %31, %29 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %29, %30 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %29 : !qref.bit
      qref.custom "T"() %29 : !qref.bit
      qref.custom "CNOT"() %28, %29 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %29, %27 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %27, %28 : !qref.bit, !qref.bit
      qref.custom "T"() %28 adj : !qref.bit
      qref.custom "T"() %29 adj : !qref.bit
      qref.custom "CNOT"() %29, %28 : !qref.bit, !qref.bit
      qref.custom "T"() %28 adj : !qref.bit
      qref.custom "T"() %27 : !qref.bit
      qref.custom "CNOT"() %29, %27 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %27, %28 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %27 : !qref.bit
      qref.custom "T"() %27 : !qref.bit
      qref.custom "CNOT"() %26, %27 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %27, %25 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %25, %26 : !qref.bit, !qref.bit
      qref.custom "T"() %26 adj : !qref.bit
      qref.custom "T"() %27 adj : !qref.bit
      qref.custom "CNOT"() %27, %26 : !qref.bit, !qref.bit
      qref.custom "T"() %26 adj : !qref.bit
      qref.custom "T"() %25 : !qref.bit
      qref.custom "CNOT"() %27, %25 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %25, %26 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %25 : !qref.bit
      qref.custom "T"() %25 : !qref.bit
      qref.custom "CNOT"() %24, %25 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %25, %23 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %23, %24 : !qref.bit, !qref.bit
      qref.custom "T"() %24 adj : !qref.bit
      qref.custom "T"() %25 adj : !qref.bit
      qref.custom "CNOT"() %25, %24 : !qref.bit, !qref.bit
      qref.custom "T"() %24 adj : !qref.bit
      qref.custom "T"() %23 : !qref.bit
      qref.custom "CNOT"() %25, %23 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %23, %24 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %23 : !qref.bit
      qref.custom "T"() %23 : !qref.bit
      qref.custom "CNOT"() %22, %23 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %23, %21 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %21, %22 : !qref.bit, !qref.bit
      qref.custom "T"() %22 adj : !qref.bit
      qref.custom "T"() %23 adj : !qref.bit
      qref.custom "CNOT"() %23, %22 : !qref.bit, !qref.bit
      qref.custom "T"() %22 adj : !qref.bit
      qref.custom "T"() %21 : !qref.bit
      qref.custom "CNOT"() %23, %21 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %21, %22 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %21 : !qref.bit
      qref.custom "T"() %21 : !qref.bit
      qref.custom "CNOT"() %20, %21 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %21, %19 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %19, %20 : !qref.bit, !qref.bit
      qref.custom "T"() %20 adj : !qref.bit
      qref.custom "T"() %21 adj : !qref.bit
      qref.custom "CNOT"() %21, %20 : !qref.bit, !qref.bit
      qref.custom "T"() %20 adj : !qref.bit
      qref.custom "T"() %19 : !qref.bit
      qref.custom "CNOT"() %21, %19 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %19, %20 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %19 : !qref.bit
      qref.custom "T"() %19 : !qref.bit
      qref.custom "CNOT"() %18, %19 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %19, %17 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %17, %18 : !qref.bit, !qref.bit
      qref.custom "T"() %18 adj : !qref.bit
      qref.custom "T"() %19 adj : !qref.bit
      qref.custom "CNOT"() %19, %18 : !qref.bit, !qref.bit
      qref.custom "T"() %18 adj : !qref.bit
      qref.custom "T"() %17 : !qref.bit
      qref.custom "CNOT"() %19, %17 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %17, %18 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %17 : !qref.bit
      qref.custom "T"() %17 : !qref.bit
      qref.custom "CNOT"() %16, %17 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %17, %15 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %15, %16 : !qref.bit, !qref.bit
      qref.custom "T"() %16 adj : !qref.bit
      qref.custom "T"() %17 adj : !qref.bit
      qref.custom "CNOT"() %17, %16 : !qref.bit, !qref.bit
      qref.custom "T"() %16 adj : !qref.bit
      qref.custom "T"() %15 : !qref.bit
      qref.custom "CNOT"() %17, %15 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %15, %16 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %15 : !qref.bit
      qref.custom "T"() %15 : !qref.bit
      qref.custom "CNOT"() %14, %15 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %15, %13 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %13, %14 : !qref.bit, !qref.bit
      qref.custom "T"() %14 adj : !qref.bit
      qref.custom "T"() %15 adj : !qref.bit
      qref.custom "CNOT"() %15, %14 : !qref.bit, !qref.bit
      qref.custom "T"() %14 adj : !qref.bit
      qref.custom "T"() %13 : !qref.bit
      qref.custom "CNOT"() %15, %13 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %13, %14 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %13 : !qref.bit
      qref.custom "T"() %13 : !qref.bit
      qref.custom "CNOT"() %12, %13 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %13, %11 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %11, %12 : !qref.bit, !qref.bit
      qref.custom "T"() %12 adj : !qref.bit
      qref.custom "T"() %13 adj : !qref.bit
      qref.custom "CNOT"() %13, %12 : !qref.bit, !qref.bit
      qref.custom "T"() %12 adj : !qref.bit
      qref.custom "T"() %11 : !qref.bit
      qref.custom "CNOT"() %13, %11 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %11, %12 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %11 : !qref.bit
      qref.custom "T"() %11 : !qref.bit
      qref.custom "CNOT"() %10, %11 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %11, %9 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %9, %10 : !qref.bit, !qref.bit
      qref.custom "T"() %10 adj : !qref.bit
      qref.custom "T"() %11 adj : !qref.bit
      qref.custom "CNOT"() %11, %10 : !qref.bit, !qref.bit
      qref.custom "T"() %10 adj : !qref.bit
      qref.custom "T"() %9 : !qref.bit
      qref.custom "CNOT"() %11, %9 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %9, %10 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %9 : !qref.bit
      qref.custom "Hadamard"() %9 : !qref.bit
      qref.custom "T"() %9 : !qref.bit
      qref.custom "CNOT"() %10, %11 : !qref.bit, !qref.bit
      qref.custom "T"() %10 : !qref.bit
      qref.custom "Hadamard"() %11 : !qref.bit
      qref.custom "T"() %11 : !qref.bit
      qref.custom "CNOT"() %12, %13 : !qref.bit, !qref.bit
      qref.custom "T"() %12 : !qref.bit
      qref.custom "Hadamard"() %13 : !qref.bit
      qref.custom "T"() %13 : !qref.bit
      qref.custom "CNOT"() %14, %15 : !qref.bit, !qref.bit
      qref.custom "T"() %14 : !qref.bit
      qref.custom "Hadamard"() %15 : !qref.bit
      qref.custom "T"() %15 : !qref.bit
      qref.custom "CNOT"() %16, %17 : !qref.bit, !qref.bit
      qref.custom "T"() %16 : !qref.bit
      qref.custom "Hadamard"() %17 : !qref.bit
      qref.custom "T"() %17 : !qref.bit
      qref.custom "CNOT"() %18, %19 : !qref.bit, !qref.bit
      qref.custom "T"() %18 : !qref.bit
      qref.custom "Hadamard"() %19 : !qref.bit
      qref.custom "T"() %19 : !qref.bit
      qref.custom "CNOT"() %20, %21 : !qref.bit, !qref.bit
      qref.custom "T"() %20 : !qref.bit
      qref.custom "Hadamard"() %21 : !qref.bit
      qref.custom "T"() %21 : !qref.bit
      qref.custom "CNOT"() %22, %23 : !qref.bit, !qref.bit
      qref.custom "T"() %22 : !qref.bit
      qref.custom "Hadamard"() %23 : !qref.bit
      qref.custom "T"() %23 : !qref.bit
      qref.custom "CNOT"() %24, %25 : !qref.bit, !qref.bit
      qref.custom "T"() %24 : !qref.bit
      qref.custom "Hadamard"() %25 : !qref.bit
      qref.custom "T"() %25 : !qref.bit
      qref.custom "CNOT"() %26, %27 : !qref.bit, !qref.bit
      qref.custom "T"() %26 : !qref.bit
      qref.custom "Hadamard"() %27 : !qref.bit
      qref.custom "T"() %27 : !qref.bit
      qref.custom "CNOT"() %28, %29 : !qref.bit, !qref.bit
      qref.custom "T"() %28 : !qref.bit
      qref.custom "Hadamard"() %29 : !qref.bit
      qref.custom "T"() %29 : !qref.bit
      qref.custom "CNOT"() %30, %31 : !qref.bit, !qref.bit
      qref.custom "T"() %30 : !qref.bit
      qref.custom "Hadamard"() %31 : !qref.bit
      qref.custom "T"() %31 : !qref.bit
      qref.custom "CNOT"() %32, %33 : !qref.bit, !qref.bit
      qref.custom "T"() %32 : !qref.bit
      qref.custom "Hadamard"() %33 : !qref.bit
      qref.custom "T"() %33 : !qref.bit
      qref.custom "CNOT"() %34, %35 : !qref.bit, !qref.bit
      qref.custom "T"() %34 : !qref.bit
      qref.custom "Hadamard"() %35 : !qref.bit
      qref.custom "T"() %35 : !qref.bit
      qref.custom "CNOT"() %36, %37 : !qref.bit, !qref.bit
      qref.custom "T"() %36 : !qref.bit
      qref.custom "Hadamard"() %37 : !qref.bit
      qref.custom "T"() %37 : !qref.bit
      qref.custom "CNOT"() %38, %39 : !qref.bit, !qref.bit
      qref.custom "T"() %38 : !qref.bit
      qref.custom "Hadamard"() %39 : !qref.bit
      qref.custom "T"() %39 : !qref.bit
      qref.custom "CNOT"() %40, %41 : !qref.bit, !qref.bit
      qref.custom "T"() %40 : !qref.bit
      qref.custom "Hadamard"() %41 : !qref.bit
      qref.custom "T"() %41 : !qref.bit
      qref.custom "CNOT"() %42, %43 : !qref.bit, !qref.bit
      qref.custom "T"() %42 : !qref.bit
      qref.custom "Hadamard"() %43 : !qref.bit
      qref.custom "T"() %43 : !qref.bit
      qref.custom "CNOT"() %44, %45 : !qref.bit, !qref.bit
      qref.custom "T"() %44 : !qref.bit
      qref.custom "Hadamard"() %45 : !qref.bit
      qref.custom "T"() %45 : !qref.bit
      qref.custom "CNOT"() %46, %47 : !qref.bit, !qref.bit
      qref.custom "T"() %46 : !qref.bit
      qref.custom "Hadamard"() %47 : !qref.bit
      qref.custom "T"() %47 : !qref.bit
      qref.custom "CNOT"() %48, %49 : !qref.bit, !qref.bit
      qref.custom "T"() %48 : !qref.bit
      qref.custom "Hadamard"() %49 : !qref.bit
      qref.custom "T"() %49 : !qref.bit
      qref.custom "CNOT"() %50, %51 : !qref.bit, !qref.bit
      qref.custom "T"() %50 : !qref.bit
      qref.custom "Hadamard"() %51 : !qref.bit
      qref.custom "T"() %51 : !qref.bit
      qref.custom "CNOT"() %52, %53 : !qref.bit, !qref.bit
      qref.custom "T"() %52 : !qref.bit
      qref.custom "Hadamard"() %53 : !qref.bit
      qref.custom "T"() %53 : !qref.bit
      qref.custom "CNOT"() %54, %55 : !qref.bit, !qref.bit
      qref.custom "T"() %54 : !qref.bit
      qref.custom "Hadamard"() %55 : !qref.bit
      qref.custom "T"() %55 : !qref.bit
      qref.custom "CNOT"() %56, %57 : !qref.bit, !qref.bit
      qref.custom "T"() %56 : !qref.bit
      qref.custom "Hadamard"() %57 : !qref.bit
      qref.custom "T"() %57 : !qref.bit
      qref.custom "CNOT"() %58, %59 : !qref.bit, !qref.bit
      qref.custom "T"() %58 : !qref.bit
      qref.custom "Hadamard"() %59 : !qref.bit
      qref.custom "T"() %59 : !qref.bit
      qref.custom "CNOT"() %60, %61 : !qref.bit, !qref.bit
      qref.custom "T"() %60 : !qref.bit
      qref.custom "Hadamard"() %61 : !qref.bit
      qref.custom "T"() %61 : !qref.bit
      qref.custom "CNOT"() %62, %63 : !qref.bit, !qref.bit
      qref.custom "T"() %62 : !qref.bit
      qref.custom "Hadamard"() %63 : !qref.bit
      qref.custom "T"() %63 : !qref.bit
      qref.custom "CNOT"() %64, %65 : !qref.bit, !qref.bit
      qref.custom "T"() %64 : !qref.bit
      qref.custom "Hadamard"() %65 : !qref.bit
      qref.custom "T"() %65 : !qref.bit
      qref.custom "CNOT"() %66, %67 : !qref.bit, !qref.bit
      qref.custom "T"() %66 : !qref.bit
      qref.custom "Hadamard"() %67 : !qref.bit
      qref.custom "T"() %67 : !qref.bit
      qref.custom "CNOT"() %68, %69 : !qref.bit, !qref.bit
      qref.custom "T"() %68 : !qref.bit
      qref.custom "Hadamard"() %69 : !qref.bit
      qref.custom "T"() %69 : !qref.bit
      qref.custom "CNOT"() %70, %71 : !qref.bit, !qref.bit
      qref.custom "T"() %70 : !qref.bit
      qref.custom "Hadamard"() %71 : !qref.bit
      qref.custom "T"() %71 : !qref.bit
      qref.custom "CNOT"() %72, %73 : !qref.bit, !qref.bit
      qref.custom "T"() %72 : !qref.bit
      qref.custom "Hadamard"() %73 : !qref.bit
      qref.custom "T"() %73 : !qref.bit
      qref.custom "CNOT"() %74, %75 : !qref.bit, !qref.bit
      qref.custom "T"() %74 : !qref.bit
      qref.custom "Hadamard"() %75 : !qref.bit
      qref.custom "T"() %75 : !qref.bit
      qref.custom "CNOT"() %76, %77 : !qref.bit, !qref.bit
      qref.custom "T"() %76 : !qref.bit
      qref.custom "Hadamard"() %77 : !qref.bit
      qref.custom "T"() %77 : !qref.bit
      qref.custom "CNOT"() %78, %79 : !qref.bit, !qref.bit
      qref.custom "T"() %78 : !qref.bit
      qref.custom "Hadamard"() %79 : !qref.bit
      qref.custom "T"() %79 : !qref.bit
      qref.custom "CNOT"() %80, %81 : !qref.bit, !qref.bit
      qref.custom "T"() %80 : !qref.bit
      qref.custom "Hadamard"() %81 : !qref.bit
      qref.custom "T"() %81 : !qref.bit
      qref.custom "CNOT"() %82, %83 : !qref.bit, !qref.bit
      qref.custom "T"() %82 : !qref.bit
      qref.custom "Hadamard"() %83 : !qref.bit
      qref.custom "T"() %83 : !qref.bit
      qref.custom "CNOT"() %84, %85 : !qref.bit, !qref.bit
      qref.custom "T"() %84 : !qref.bit
      qref.custom "Hadamard"() %85 : !qref.bit
      qref.custom "T"() %85 : !qref.bit
      qref.custom "CNOT"() %86, %87 : !qref.bit, !qref.bit
      qref.custom "T"() %86 : !qref.bit
      qref.custom "Hadamard"() %87 : !qref.bit
      qref.custom "T"() %87 : !qref.bit
      qref.custom "CNOT"() %88, %89 : !qref.bit, !qref.bit
      qref.custom "T"() %88 : !qref.bit
      qref.custom "Hadamard"() %89 : !qref.bit
      qref.custom "T"() %89 : !qref.bit
      qref.custom "CNOT"() %90, %91 : !qref.bit, !qref.bit
      qref.custom "T"() %90 : !qref.bit
      qref.custom "Hadamard"() %91 : !qref.bit
      qref.custom "T"() %91 : !qref.bit
      qref.custom "CNOT"() %92, %93 : !qref.bit, !qref.bit
      qref.custom "T"() %92 : !qref.bit
      qref.custom "Hadamard"() %93 : !qref.bit
      qref.custom "T"() %93 : !qref.bit
      qref.custom "CNOT"() %94, %95 : !qref.bit, !qref.bit
      qref.custom "T"() %94 : !qref.bit
      qref.custom "Hadamard"() %95 : !qref.bit
      qref.custom "T"() %95 : !qref.bit
      qref.custom "CNOT"() %96, %97 : !qref.bit, !qref.bit
      qref.custom "T"() %96 : !qref.bit
      qref.custom "Hadamard"() %97 : !qref.bit
      qref.custom "T"() %97 : !qref.bit
      qref.custom "CNOT"() %98, %99 : !qref.bit, !qref.bit
      qref.custom "T"() %98 : !qref.bit
      qref.custom "Hadamard"() %99 : !qref.bit
      qref.custom "T"() %99 : !qref.bit
      qref.custom "CNOT"() %100, %101 : !qref.bit, !qref.bit
      qref.custom "T"() %100 : !qref.bit
      qref.custom "Hadamard"() %101 : !qref.bit
      qref.custom "T"() %101 : !qref.bit
      qref.custom "CNOT"() %102, %103 : !qref.bit, !qref.bit
      qref.custom "T"() %102 : !qref.bit
      qref.custom "Hadamard"() %103 : !qref.bit
      qref.custom "T"() %103 : !qref.bit
      qref.custom "CNOT"() %104, %105 : !qref.bit, !qref.bit
      qref.custom "T"() %104 : !qref.bit
      qref.custom "Hadamard"() %105 : !qref.bit
      qref.custom "T"() %105 : !qref.bit
      qref.custom "CNOT"() %106, %107 : !qref.bit, !qref.bit
      qref.custom "T"() %106 : !qref.bit
      qref.custom "Hadamard"() %107 : !qref.bit
      qref.custom "T"() %107 : !qref.bit
      qref.custom "CNOT"() %108, %109 : !qref.bit, !qref.bit
      qref.custom "T"() %108 : !qref.bit
      qref.custom "Hadamard"() %109 : !qref.bit
      qref.custom "T"() %109 : !qref.bit
      qref.custom "CNOT"() %110, %111 : !qref.bit, !qref.bit
      qref.custom "T"() %110 : !qref.bit
      qref.custom "Hadamard"() %111 : !qref.bit
      qref.custom "T"() %111 : !qref.bit
      qref.custom "CNOT"() %112, %113 : !qref.bit, !qref.bit
      qref.custom "T"() %112 : !qref.bit
      qref.custom "Hadamard"() %113 : !qref.bit
      qref.custom "T"() %113 : !qref.bit
      qref.custom "CNOT"() %114, %115 : !qref.bit, !qref.bit
      qref.custom "T"() %114 : !qref.bit
      qref.custom "Hadamard"() %115 : !qref.bit
      qref.custom "T"() %115 : !qref.bit
      qref.custom "CNOT"() %116, %117 : !qref.bit, !qref.bit
      qref.custom "T"() %116 : !qref.bit
      qref.custom "Hadamard"() %117 : !qref.bit
      qref.custom "T"() %117 : !qref.bit
      qref.custom "CNOT"() %118, %119 : !qref.bit, !qref.bit
      qref.custom "T"() %118 : !qref.bit
      qref.custom "Hadamard"() %119 : !qref.bit
      qref.custom "T"() %119 : !qref.bit
      qref.custom "CNOT"() %120, %121 : !qref.bit, !qref.bit
      qref.custom "T"() %120 : !qref.bit
      qref.custom "Hadamard"() %121 : !qref.bit
      qref.custom "T"() %121 : !qref.bit
      qref.custom "CNOT"() %122, %123 : !qref.bit, !qref.bit
      qref.custom "T"() %122 : !qref.bit
      qref.custom "Hadamard"() %123 : !qref.bit
      qref.custom "T"() %123 : !qref.bit
      qref.custom "CNOT"() %124, %125 : !qref.bit, !qref.bit
      qref.custom "T"() %124 : !qref.bit
      qref.custom "Hadamard"() %125 : !qref.bit
      qref.custom "T"() %125 : !qref.bit
      qref.custom "CNOT"() %126, %127 : !qref.bit, !qref.bit
      qref.custom "T"() %126 : !qref.bit
      qref.custom "Hadamard"() %127 : !qref.bit
      qref.custom "T"() %127 : !qref.bit
      qref.custom "CNOT"() %128, %129 : !qref.bit, !qref.bit
      qref.custom "T"() %128 : !qref.bit
      qref.custom "T"() %129 : !qref.bit
      qref.custom "CNOT"() %128, %129 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %129, %127 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %127, %128 : !qref.bit, !qref.bit
      qref.custom "T"() %128 adj : !qref.bit
      qref.custom "T"() %129 adj : !qref.bit
      qref.custom "CNOT"() %129, %128 : !qref.bit, !qref.bit
      qref.custom "T"() %128 adj : !qref.bit
      qref.custom "T"() %127 : !qref.bit
      qref.custom "CNOT"() %129, %127 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %127, %128 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %127 : !qref.bit
      qref.custom "T"() %127 : !qref.bit
      qref.custom "CNOT"() %126, %127 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %127, %125 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %125, %126 : !qref.bit, !qref.bit
      qref.custom "T"() %126 adj : !qref.bit
      qref.custom "T"() %127 adj : !qref.bit
      qref.custom "CNOT"() %127, %126 : !qref.bit, !qref.bit
      qref.custom "T"() %126 adj : !qref.bit
      qref.custom "T"() %125 : !qref.bit
      qref.custom "CNOT"() %127, %125 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %125, %126 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %125 : !qref.bit
      qref.custom "T"() %125 : !qref.bit
      qref.custom "CNOT"() %124, %125 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %125, %123 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %123, %124 : !qref.bit, !qref.bit
      qref.custom "T"() %124 adj : !qref.bit
      qref.custom "T"() %125 adj : !qref.bit
      qref.custom "CNOT"() %125, %124 : !qref.bit, !qref.bit
      qref.custom "T"() %124 adj : !qref.bit
      qref.custom "T"() %123 : !qref.bit
      qref.custom "CNOT"() %125, %123 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %123, %124 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %123 : !qref.bit
      qref.custom "T"() %123 : !qref.bit
      qref.custom "CNOT"() %122, %123 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %123, %121 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %121, %122 : !qref.bit, !qref.bit
      qref.custom "T"() %122 adj : !qref.bit
      qref.custom "T"() %123 adj : !qref.bit
      qref.custom "CNOT"() %123, %122 : !qref.bit, !qref.bit
      qref.custom "T"() %122 adj : !qref.bit
      qref.custom "T"() %121 : !qref.bit
      qref.custom "CNOT"() %123, %121 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %121, %122 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %121 : !qref.bit
      qref.custom "T"() %121 : !qref.bit
      qref.custom "CNOT"() %120, %121 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %121, %119 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %119, %120 : !qref.bit, !qref.bit
      qref.custom "T"() %120 adj : !qref.bit
      qref.custom "T"() %121 adj : !qref.bit
      qref.custom "CNOT"() %121, %120 : !qref.bit, !qref.bit
      qref.custom "T"() %120 adj : !qref.bit
      qref.custom "T"() %119 : !qref.bit
      qref.custom "CNOT"() %121, %119 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %119, %120 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %119 : !qref.bit
      qref.custom "T"() %119 : !qref.bit
      qref.custom "CNOT"() %118, %119 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %119, %117 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %117, %118 : !qref.bit, !qref.bit
      qref.custom "T"() %118 adj : !qref.bit
      qref.custom "T"() %119 adj : !qref.bit
      qref.custom "CNOT"() %119, %118 : !qref.bit, !qref.bit
      qref.custom "T"() %118 adj : !qref.bit
      qref.custom "T"() %117 : !qref.bit
      qref.custom "CNOT"() %119, %117 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %117, %118 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %117 : !qref.bit
      qref.custom "T"() %117 : !qref.bit
      qref.custom "CNOT"() %116, %117 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %117, %115 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %115, %116 : !qref.bit, !qref.bit
      qref.custom "T"() %116 adj : !qref.bit
      qref.custom "T"() %117 adj : !qref.bit
      qref.custom "CNOT"() %117, %116 : !qref.bit, !qref.bit
      qref.custom "T"() %116 adj : !qref.bit
      qref.custom "T"() %115 : !qref.bit
      qref.custom "CNOT"() %117, %115 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %115, %116 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %115 : !qref.bit
      qref.custom "T"() %115 : !qref.bit
      qref.custom "CNOT"() %114, %115 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %115, %113 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %113, %114 : !qref.bit, !qref.bit
      qref.custom "T"() %114 adj : !qref.bit
      qref.custom "T"() %115 adj : !qref.bit
      qref.custom "CNOT"() %115, %114 : !qref.bit, !qref.bit
      qref.custom "T"() %114 adj : !qref.bit
      qref.custom "T"() %113 : !qref.bit
      qref.custom "CNOT"() %115, %113 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %113, %114 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %113 : !qref.bit
      qref.custom "T"() %113 : !qref.bit
      qref.custom "CNOT"() %112, %113 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %113, %111 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %111, %112 : !qref.bit, !qref.bit
      qref.custom "T"() %112 adj : !qref.bit
      qref.custom "T"() %113 adj : !qref.bit
      qref.custom "CNOT"() %113, %112 : !qref.bit, !qref.bit
      qref.custom "T"() %112 adj : !qref.bit
      qref.custom "T"() %111 : !qref.bit
      qref.custom "CNOT"() %113, %111 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %111, %112 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %111 : !qref.bit
      qref.custom "T"() %111 : !qref.bit
      qref.custom "CNOT"() %110, %111 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %111, %109 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %109, %110 : !qref.bit, !qref.bit
      qref.custom "T"() %110 adj : !qref.bit
      qref.custom "T"() %111 adj : !qref.bit
      qref.custom "CNOT"() %111, %110 : !qref.bit, !qref.bit
      qref.custom "T"() %110 adj : !qref.bit
      qref.custom "T"() %109 : !qref.bit
      qref.custom "CNOT"() %111, %109 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %109, %110 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %109 : !qref.bit
      qref.custom "T"() %109 : !qref.bit
      qref.custom "CNOT"() %108, %109 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %109, %107 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %107, %108 : !qref.bit, !qref.bit
      qref.custom "T"() %108 adj : !qref.bit
      qref.custom "T"() %109 adj : !qref.bit
      qref.custom "CNOT"() %109, %108 : !qref.bit, !qref.bit
      qref.custom "T"() %108 adj : !qref.bit
      qref.custom "T"() %107 : !qref.bit
      qref.custom "CNOT"() %109, %107 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %107, %108 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %107 : !qref.bit
      qref.custom "T"() %107 : !qref.bit
      qref.custom "CNOT"() %106, %107 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %107, %105 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %105, %106 : !qref.bit, !qref.bit
      qref.custom "T"() %106 adj : !qref.bit
      qref.custom "T"() %107 adj : !qref.bit
      qref.custom "CNOT"() %107, %106 : !qref.bit, !qref.bit
      qref.custom "T"() %106 adj : !qref.bit
      qref.custom "T"() %105 : !qref.bit
      qref.custom "CNOT"() %107, %105 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %105, %106 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %105 : !qref.bit
      qref.custom "T"() %105 : !qref.bit
      qref.custom "CNOT"() %104, %105 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %105, %103 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %103, %104 : !qref.bit, !qref.bit
      qref.custom "T"() %104 adj : !qref.bit
      qref.custom "T"() %105 adj : !qref.bit
      qref.custom "CNOT"() %105, %104 : !qref.bit, !qref.bit
      qref.custom "T"() %104 adj : !qref.bit
      qref.custom "T"() %103 : !qref.bit
      qref.custom "CNOT"() %105, %103 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %103, %104 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %103 : !qref.bit
      qref.custom "T"() %103 : !qref.bit
      qref.custom "CNOT"() %102, %103 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %103, %101 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %101, %102 : !qref.bit, !qref.bit
      qref.custom "T"() %102 adj : !qref.bit
      qref.custom "T"() %103 adj : !qref.bit
      qref.custom "CNOT"() %103, %102 : !qref.bit, !qref.bit
      qref.custom "T"() %102 adj : !qref.bit
      qref.custom "T"() %101 : !qref.bit
      qref.custom "CNOT"() %103, %101 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %101, %102 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %101 : !qref.bit
      qref.custom "T"() %101 : !qref.bit
      qref.custom "CNOT"() %100, %101 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %101, %99 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %99, %100 : !qref.bit, !qref.bit
      qref.custom "T"() %100 adj : !qref.bit
      qref.custom "T"() %101 adj : !qref.bit
      qref.custom "CNOT"() %101, %100 : !qref.bit, !qref.bit
      qref.custom "T"() %100 adj : !qref.bit
      qref.custom "T"() %99 : !qref.bit
      qref.custom "CNOT"() %101, %99 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %99, %100 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %99 : !qref.bit
      qref.custom "T"() %99 : !qref.bit
      qref.custom "CNOT"() %98, %99 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %99, %97 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %97, %98 : !qref.bit, !qref.bit
      qref.custom "T"() %98 adj : !qref.bit
      qref.custom "T"() %99 adj : !qref.bit
      qref.custom "CNOT"() %99, %98 : !qref.bit, !qref.bit
      qref.custom "T"() %98 adj : !qref.bit
      qref.custom "T"() %97 : !qref.bit
      qref.custom "CNOT"() %99, %97 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %97, %98 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %97 : !qref.bit
      qref.custom "T"() %97 : !qref.bit
      qref.custom "CNOT"() %96, %97 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %97, %95 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %95, %96 : !qref.bit, !qref.bit
      qref.custom "T"() %96 adj : !qref.bit
      qref.custom "T"() %97 adj : !qref.bit
      qref.custom "CNOT"() %97, %96 : !qref.bit, !qref.bit
      qref.custom "T"() %96 adj : !qref.bit
      qref.custom "T"() %95 : !qref.bit
      qref.custom "CNOT"() %97, %95 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %95, %96 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %95 : !qref.bit
      qref.custom "T"() %95 : !qref.bit
      qref.custom "CNOT"() %94, %95 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %95, %93 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %93, %94 : !qref.bit, !qref.bit
      qref.custom "T"() %94 adj : !qref.bit
      qref.custom "T"() %95 adj : !qref.bit
      qref.custom "CNOT"() %95, %94 : !qref.bit, !qref.bit
      qref.custom "T"() %94 adj : !qref.bit
      qref.custom "T"() %93 : !qref.bit
      qref.custom "CNOT"() %95, %93 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %93, %94 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %93 : !qref.bit
      qref.custom "T"() %93 : !qref.bit
      qref.custom "CNOT"() %92, %93 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %93, %91 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %91, %92 : !qref.bit, !qref.bit
      qref.custom "T"() %92 adj : !qref.bit
      qref.custom "T"() %93 adj : !qref.bit
      qref.custom "CNOT"() %93, %92 : !qref.bit, !qref.bit
      qref.custom "T"() %92 adj : !qref.bit
      qref.custom "T"() %91 : !qref.bit
      qref.custom "CNOT"() %93, %91 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %91, %92 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %91 : !qref.bit
      qref.custom "T"() %91 : !qref.bit
      qref.custom "CNOT"() %90, %91 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %91, %89 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %89, %90 : !qref.bit, !qref.bit
      qref.custom "T"() %90 adj : !qref.bit
      qref.custom "T"() %91 adj : !qref.bit
      qref.custom "CNOT"() %91, %90 : !qref.bit, !qref.bit
      qref.custom "T"() %90 adj : !qref.bit
      qref.custom "T"() %89 : !qref.bit
      qref.custom "CNOT"() %91, %89 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %89, %90 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %89 : !qref.bit
      qref.custom "T"() %89 : !qref.bit
      qref.custom "CNOT"() %88, %89 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %89, %87 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %87, %88 : !qref.bit, !qref.bit
      qref.custom "T"() %88 adj : !qref.bit
      qref.custom "T"() %89 adj : !qref.bit
      qref.custom "CNOT"() %89, %88 : !qref.bit, !qref.bit
      qref.custom "T"() %88 adj : !qref.bit
      qref.custom "T"() %87 : !qref.bit
      qref.custom "CNOT"() %89, %87 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %87, %88 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %87 : !qref.bit
      qref.custom "T"() %87 : !qref.bit
      qref.custom "CNOT"() %86, %87 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %87, %85 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %85, %86 : !qref.bit, !qref.bit
      qref.custom "T"() %86 adj : !qref.bit
      qref.custom "T"() %87 adj : !qref.bit
      qref.custom "CNOT"() %87, %86 : !qref.bit, !qref.bit
      qref.custom "T"() %86 adj : !qref.bit
      qref.custom "T"() %85 : !qref.bit
      qref.custom "CNOT"() %87, %85 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %85, %86 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %85 : !qref.bit
      qref.custom "T"() %85 : !qref.bit
      qref.custom "CNOT"() %84, %85 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %85, %83 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %83, %84 : !qref.bit, !qref.bit
      qref.custom "T"() %84 adj : !qref.bit
      qref.custom "T"() %85 adj : !qref.bit
      qref.custom "CNOT"() %85, %84 : !qref.bit, !qref.bit
      qref.custom "T"() %84 adj : !qref.bit
      qref.custom "T"() %83 : !qref.bit
      qref.custom "CNOT"() %85, %83 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %83, %84 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %83 : !qref.bit
      qref.custom "T"() %83 : !qref.bit
      qref.custom "CNOT"() %82, %83 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %83, %81 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %81, %82 : !qref.bit, !qref.bit
      qref.custom "T"() %82 adj : !qref.bit
      qref.custom "T"() %83 adj : !qref.bit
      qref.custom "CNOT"() %83, %82 : !qref.bit, !qref.bit
      qref.custom "T"() %82 adj : !qref.bit
      qref.custom "T"() %81 : !qref.bit
      qref.custom "CNOT"() %83, %81 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %81, %82 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %81 : !qref.bit
      qref.custom "T"() %81 : !qref.bit
      qref.custom "CNOT"() %80, %81 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %81, %79 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %79, %80 : !qref.bit, !qref.bit
      qref.custom "T"() %80 adj : !qref.bit
      qref.custom "T"() %81 adj : !qref.bit
      qref.custom "CNOT"() %81, %80 : !qref.bit, !qref.bit
      qref.custom "T"() %80 adj : !qref.bit
      qref.custom "T"() %79 : !qref.bit
      qref.custom "CNOT"() %81, %79 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %79, %80 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %79 : !qref.bit
      qref.custom "T"() %79 : !qref.bit
      qref.custom "CNOT"() %78, %79 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %79, %77 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %77, %78 : !qref.bit, !qref.bit
      qref.custom "T"() %78 adj : !qref.bit
      qref.custom "T"() %79 adj : !qref.bit
      qref.custom "CNOT"() %79, %78 : !qref.bit, !qref.bit
      qref.custom "T"() %78 adj : !qref.bit
      qref.custom "T"() %77 : !qref.bit
      qref.custom "CNOT"() %79, %77 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %77, %78 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %77 : !qref.bit
      qref.custom "T"() %77 : !qref.bit
      qref.custom "CNOT"() %76, %77 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %77, %75 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %75, %76 : !qref.bit, !qref.bit
      qref.custom "T"() %76 adj : !qref.bit
      qref.custom "T"() %77 adj : !qref.bit
      qref.custom "CNOT"() %77, %76 : !qref.bit, !qref.bit
      qref.custom "T"() %76 adj : !qref.bit
      qref.custom "T"() %75 : !qref.bit
      qref.custom "CNOT"() %77, %75 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %75, %76 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %75 : !qref.bit
      qref.custom "T"() %75 : !qref.bit
      qref.custom "CNOT"() %74, %75 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %75, %73 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %73, %74 : !qref.bit, !qref.bit
      qref.custom "T"() %74 adj : !qref.bit
      qref.custom "T"() %75 adj : !qref.bit
      qref.custom "CNOT"() %75, %74 : !qref.bit, !qref.bit
      qref.custom "T"() %74 adj : !qref.bit
      qref.custom "T"() %73 : !qref.bit
      qref.custom "CNOT"() %75, %73 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %73, %74 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %73 : !qref.bit
      qref.custom "T"() %73 : !qref.bit
      qref.custom "CNOT"() %72, %73 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %73, %71 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %71, %72 : !qref.bit, !qref.bit
      qref.custom "T"() %72 adj : !qref.bit
      qref.custom "T"() %73 adj : !qref.bit
      qref.custom "CNOT"() %73, %72 : !qref.bit, !qref.bit
      qref.custom "T"() %72 adj : !qref.bit
      qref.custom "T"() %71 : !qref.bit
      qref.custom "CNOT"() %73, %71 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %71, %72 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %71 : !qref.bit
      qref.custom "T"() %71 : !qref.bit
      qref.custom "CNOT"() %70, %71 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %71, %69 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %69, %70 : !qref.bit, !qref.bit
      qref.custom "T"() %70 adj : !qref.bit
      qref.custom "T"() %71 adj : !qref.bit
      qref.custom "CNOT"() %71, %70 : !qref.bit, !qref.bit
      qref.custom "T"() %70 adj : !qref.bit
      qref.custom "T"() %69 : !qref.bit
      qref.custom "CNOT"() %71, %69 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %69, %70 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %69 : !qref.bit
      qref.custom "T"() %69 : !qref.bit
      qref.custom "CNOT"() %68, %69 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %69, %67 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %67, %68 : !qref.bit, !qref.bit
      qref.custom "T"() %68 adj : !qref.bit
      qref.custom "T"() %69 adj : !qref.bit
      qref.custom "CNOT"() %69, %68 : !qref.bit, !qref.bit
      qref.custom "T"() %68 adj : !qref.bit
      qref.custom "T"() %67 : !qref.bit
      qref.custom "CNOT"() %69, %67 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %67, %68 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %67 : !qref.bit
      qref.custom "T"() %67 : !qref.bit
      qref.custom "CNOT"() %66, %67 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %67, %65 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %65, %66 : !qref.bit, !qref.bit
      qref.custom "T"() %66 adj : !qref.bit
      qref.custom "T"() %67 adj : !qref.bit
      qref.custom "CNOT"() %67, %66 : !qref.bit, !qref.bit
      qref.custom "T"() %66 adj : !qref.bit
      qref.custom "T"() %65 : !qref.bit
      qref.custom "CNOT"() %67, %65 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %65, %66 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %65 : !qref.bit
      qref.custom "T"() %65 : !qref.bit
      qref.custom "CNOT"() %64, %65 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %65, %63 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %63, %64 : !qref.bit, !qref.bit
      qref.custom "T"() %64 adj : !qref.bit
      qref.custom "T"() %65 adj : !qref.bit
      qref.custom "CNOT"() %65, %64 : !qref.bit, !qref.bit
      qref.custom "T"() %64 adj : !qref.bit
      qref.custom "T"() %63 : !qref.bit
      qref.custom "CNOT"() %65, %63 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %63, %64 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %63 : !qref.bit
      qref.custom "T"() %63 : !qref.bit
      qref.custom "CNOT"() %62, %63 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %63, %61 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %61, %62 : !qref.bit, !qref.bit
      qref.custom "T"() %62 adj : !qref.bit
      qref.custom "T"() %63 adj : !qref.bit
      qref.custom "CNOT"() %63, %62 : !qref.bit, !qref.bit
      qref.custom "T"() %62 adj : !qref.bit
      qref.custom "T"() %61 : !qref.bit
      qref.custom "CNOT"() %63, %61 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %61, %62 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %61 : !qref.bit
      qref.custom "T"() %61 : !qref.bit
      qref.custom "CNOT"() %60, %61 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %61, %59 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %59, %60 : !qref.bit, !qref.bit
      qref.custom "T"() %60 adj : !qref.bit
      qref.custom "T"() %61 adj : !qref.bit
      qref.custom "CNOT"() %61, %60 : !qref.bit, !qref.bit
      qref.custom "T"() %60 adj : !qref.bit
      qref.custom "T"() %59 : !qref.bit
      qref.custom "CNOT"() %61, %59 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %59, %60 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %59 : !qref.bit
      qref.custom "T"() %59 : !qref.bit
      qref.custom "CNOT"() %58, %59 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %59, %57 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %57, %58 : !qref.bit, !qref.bit
      qref.custom "T"() %58 adj : !qref.bit
      qref.custom "T"() %59 adj : !qref.bit
      qref.custom "CNOT"() %59, %58 : !qref.bit, !qref.bit
      qref.custom "T"() %58 adj : !qref.bit
      qref.custom "T"() %57 : !qref.bit
      qref.custom "CNOT"() %59, %57 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %57, %58 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %57 : !qref.bit
      qref.custom "T"() %57 : !qref.bit
      qref.custom "CNOT"() %56, %57 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %57, %55 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %55, %56 : !qref.bit, !qref.bit
      qref.custom "T"() %56 adj : !qref.bit
      qref.custom "T"() %57 adj : !qref.bit
      qref.custom "CNOT"() %57, %56 : !qref.bit, !qref.bit
      qref.custom "T"() %56 adj : !qref.bit
      qref.custom "T"() %55 : !qref.bit
      qref.custom "CNOT"() %57, %55 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %55, %56 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %55 : !qref.bit
      qref.custom "T"() %55 : !qref.bit
      qref.custom "CNOT"() %54, %55 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %55, %53 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %53, %54 : !qref.bit, !qref.bit
      qref.custom "T"() %54 adj : !qref.bit
      qref.custom "T"() %55 adj : !qref.bit
      qref.custom "CNOT"() %55, %54 : !qref.bit, !qref.bit
      qref.custom "T"() %54 adj : !qref.bit
      qref.custom "T"() %53 : !qref.bit
      qref.custom "CNOT"() %55, %53 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %53, %54 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %53 : !qref.bit
      qref.custom "T"() %53 : !qref.bit
      qref.custom "CNOT"() %52, %53 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %53, %51 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %51, %52 : !qref.bit, !qref.bit
      qref.custom "T"() %52 adj : !qref.bit
      qref.custom "T"() %53 adj : !qref.bit
      qref.custom "CNOT"() %53, %52 : !qref.bit, !qref.bit
      qref.custom "T"() %52 adj : !qref.bit
      qref.custom "T"() %51 : !qref.bit
      qref.custom "CNOT"() %53, %51 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %51, %52 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %51 : !qref.bit
      qref.custom "T"() %51 : !qref.bit
      qref.custom "CNOT"() %50, %51 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %51, %49 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %49, %50 : !qref.bit, !qref.bit
      qref.custom "T"() %50 adj : !qref.bit
      qref.custom "T"() %51 adj : !qref.bit
      qref.custom "CNOT"() %51, %50 : !qref.bit, !qref.bit
      qref.custom "T"() %50 adj : !qref.bit
      qref.custom "T"() %49 : !qref.bit
      qref.custom "CNOT"() %51, %49 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %49, %50 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %49 : !qref.bit
      qref.custom "T"() %49 : !qref.bit
      qref.custom "CNOT"() %48, %49 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %49, %47 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %47, %48 : !qref.bit, !qref.bit
      qref.custom "T"() %48 adj : !qref.bit
      qref.custom "T"() %49 adj : !qref.bit
      qref.custom "CNOT"() %49, %48 : !qref.bit, !qref.bit
      qref.custom "T"() %48 adj : !qref.bit
      qref.custom "T"() %47 : !qref.bit
      qref.custom "CNOT"() %49, %47 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %47, %48 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %47 : !qref.bit
      qref.custom "T"() %47 : !qref.bit
      qref.custom "CNOT"() %46, %47 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %47, %45 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %45, %46 : !qref.bit, !qref.bit
      qref.custom "T"() %46 adj : !qref.bit
      qref.custom "T"() %47 adj : !qref.bit
      qref.custom "CNOT"() %47, %46 : !qref.bit, !qref.bit
      qref.custom "T"() %46 adj : !qref.bit
      qref.custom "T"() %45 : !qref.bit
      qref.custom "CNOT"() %47, %45 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %45, %46 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %45 : !qref.bit
      qref.custom "T"() %45 : !qref.bit
      qref.custom "CNOT"() %44, %45 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %45, %43 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %43, %44 : !qref.bit, !qref.bit
      qref.custom "T"() %44 adj : !qref.bit
      qref.custom "T"() %45 adj : !qref.bit
      qref.custom "CNOT"() %45, %44 : !qref.bit, !qref.bit
      qref.custom "T"() %44 adj : !qref.bit
      qref.custom "T"() %43 : !qref.bit
      qref.custom "CNOT"() %45, %43 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %43, %44 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %43 : !qref.bit
      qref.custom "T"() %43 : !qref.bit
      qref.custom "CNOT"() %42, %43 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %43, %41 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %41, %42 : !qref.bit, !qref.bit
      qref.custom "T"() %42 adj : !qref.bit
      qref.custom "T"() %43 adj : !qref.bit
      qref.custom "CNOT"() %43, %42 : !qref.bit, !qref.bit
      qref.custom "T"() %42 adj : !qref.bit
      qref.custom "T"() %41 : !qref.bit
      qref.custom "CNOT"() %43, %41 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %41, %42 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %41 : !qref.bit
      qref.custom "T"() %41 : !qref.bit
      qref.custom "CNOT"() %40, %41 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %41, %39 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %39, %40 : !qref.bit, !qref.bit
      qref.custom "T"() %40 adj : !qref.bit
      qref.custom "T"() %41 adj : !qref.bit
      qref.custom "CNOT"() %41, %40 : !qref.bit, !qref.bit
      qref.custom "T"() %40 adj : !qref.bit
      qref.custom "T"() %39 : !qref.bit
      qref.custom "CNOT"() %41, %39 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %39, %40 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %39 : !qref.bit
      qref.custom "T"() %39 : !qref.bit
      qref.custom "CNOT"() %38, %39 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %39, %37 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %37, %38 : !qref.bit, !qref.bit
      qref.custom "T"() %38 adj : !qref.bit
      qref.custom "T"() %39 adj : !qref.bit
      qref.custom "CNOT"() %39, %38 : !qref.bit, !qref.bit
      qref.custom "T"() %38 adj : !qref.bit
      qref.custom "T"() %37 : !qref.bit
      qref.custom "CNOT"() %39, %37 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %37, %38 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %37 : !qref.bit
      qref.custom "T"() %37 : !qref.bit
      qref.custom "CNOT"() %36, %37 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %37, %35 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %35, %36 : !qref.bit, !qref.bit
      qref.custom "T"() %36 adj : !qref.bit
      qref.custom "T"() %37 adj : !qref.bit
      qref.custom "CNOT"() %37, %36 : !qref.bit, !qref.bit
      qref.custom "T"() %36 adj : !qref.bit
      qref.custom "T"() %35 : !qref.bit
      qref.custom "CNOT"() %37, %35 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %35, %36 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %35 : !qref.bit
      qref.custom "T"() %35 : !qref.bit
      qref.custom "CNOT"() %34, %35 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %35, %33 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %33, %34 : !qref.bit, !qref.bit
      qref.custom "T"() %34 adj : !qref.bit
      qref.custom "T"() %35 adj : !qref.bit
      qref.custom "CNOT"() %35, %34 : !qref.bit, !qref.bit
      qref.custom "T"() %34 adj : !qref.bit
      qref.custom "T"() %33 : !qref.bit
      qref.custom "CNOT"() %35, %33 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %33, %34 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %33 : !qref.bit
      qref.custom "T"() %33 : !qref.bit
      qref.custom "CNOT"() %32, %33 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %33, %31 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %31, %32 : !qref.bit, !qref.bit
      qref.custom "T"() %32 adj : !qref.bit
      qref.custom "T"() %33 adj : !qref.bit
      qref.custom "CNOT"() %33, %32 : !qref.bit, !qref.bit
      qref.custom "T"() %32 adj : !qref.bit
      qref.custom "T"() %31 : !qref.bit
      qref.custom "CNOT"() %33, %31 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %31, %32 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %31 : !qref.bit
      qref.custom "T"() %31 : !qref.bit
      qref.custom "CNOT"() %30, %31 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %31, %29 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %29, %30 : !qref.bit, !qref.bit
      qref.custom "T"() %30 adj : !qref.bit
      qref.custom "T"() %31 adj : !qref.bit
      qref.custom "CNOT"() %31, %30 : !qref.bit, !qref.bit
      qref.custom "T"() %30 adj : !qref.bit
      qref.custom "T"() %29 : !qref.bit
      qref.custom "CNOT"() %31, %29 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %29, %30 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %29 : !qref.bit
      qref.custom "T"() %29 : !qref.bit
      qref.custom "CNOT"() %28, %29 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %29, %27 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %27, %28 : !qref.bit, !qref.bit
      qref.custom "T"() %28 adj : !qref.bit
      qref.custom "T"() %29 adj : !qref.bit
      qref.custom "CNOT"() %29, %28 : !qref.bit, !qref.bit
      qref.custom "T"() %28 adj : !qref.bit
      qref.custom "T"() %27 : !qref.bit
      qref.custom "CNOT"() %29, %27 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %27, %28 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %27 : !qref.bit
      qref.custom "T"() %27 : !qref.bit
      qref.custom "CNOT"() %26, %27 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %27, %25 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %25, %26 : !qref.bit, !qref.bit
      qref.custom "T"() %26 adj : !qref.bit
      qref.custom "T"() %27 adj : !qref.bit
      qref.custom "CNOT"() %27, %26 : !qref.bit, !qref.bit
      qref.custom "T"() %26 adj : !qref.bit
      qref.custom "T"() %25 : !qref.bit
      qref.custom "CNOT"() %27, %25 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %25, %26 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %25 : !qref.bit
      qref.custom "T"() %25 : !qref.bit
      qref.custom "CNOT"() %24, %25 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %25, %23 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %23, %24 : !qref.bit, !qref.bit
      qref.custom "T"() %24 adj : !qref.bit
      qref.custom "T"() %25 adj : !qref.bit
      qref.custom "CNOT"() %25, %24 : !qref.bit, !qref.bit
      qref.custom "T"() %24 adj : !qref.bit
      qref.custom "T"() %23 : !qref.bit
      qref.custom "CNOT"() %25, %23 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %23, %24 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %23 : !qref.bit
      qref.custom "T"() %23 : !qref.bit
      qref.custom "CNOT"() %22, %23 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %23, %21 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %21, %22 : !qref.bit, !qref.bit
      qref.custom "T"() %22 adj : !qref.bit
      qref.custom "T"() %23 adj : !qref.bit
      qref.custom "CNOT"() %23, %22 : !qref.bit, !qref.bit
      qref.custom "T"() %22 adj : !qref.bit
      qref.custom "T"() %21 : !qref.bit
      qref.custom "CNOT"() %23, %21 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %21, %22 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %21 : !qref.bit
      qref.custom "T"() %21 : !qref.bit
      qref.custom "CNOT"() %20, %21 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %21, %19 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %19, %20 : !qref.bit, !qref.bit
      qref.custom "T"() %20 adj : !qref.bit
      qref.custom "T"() %21 adj : !qref.bit
      qref.custom "CNOT"() %21, %20 : !qref.bit, !qref.bit
      qref.custom "T"() %20 adj : !qref.bit
      qref.custom "T"() %19 : !qref.bit
      qref.custom "CNOT"() %21, %19 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %19, %20 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %19 : !qref.bit
      qref.custom "T"() %19 : !qref.bit
      qref.custom "CNOT"() %18, %19 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %19, %17 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %17, %18 : !qref.bit, !qref.bit
      qref.custom "T"() %18 adj : !qref.bit
      qref.custom "T"() %19 adj : !qref.bit
      qref.custom "CNOT"() %19, %18 : !qref.bit, !qref.bit
      qref.custom "T"() %18 adj : !qref.bit
      qref.custom "T"() %17 : !qref.bit
      qref.custom "CNOT"() %19, %17 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %17, %18 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %17 : !qref.bit
      qref.custom "T"() %17 : !qref.bit
      qref.custom "CNOT"() %16, %17 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %17, %15 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %15, %16 : !qref.bit, !qref.bit
      qref.custom "T"() %16 adj : !qref.bit
      qref.custom "T"() %17 adj : !qref.bit
      qref.custom "CNOT"() %17, %16 : !qref.bit, !qref.bit
      qref.custom "T"() %16 adj : !qref.bit
      qref.custom "T"() %15 : !qref.bit
      qref.custom "CNOT"() %17, %15 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %15, %16 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %15 : !qref.bit
      qref.custom "T"() %15 : !qref.bit
      qref.custom "CNOT"() %14, %15 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %15, %13 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %13, %14 : !qref.bit, !qref.bit
      qref.custom "T"() %14 adj : !qref.bit
      qref.custom "T"() %15 adj : !qref.bit
      qref.custom "CNOT"() %15, %14 : !qref.bit, !qref.bit
      qref.custom "T"() %14 adj : !qref.bit
      qref.custom "T"() %13 : !qref.bit
      qref.custom "CNOT"() %15, %13 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %13, %14 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %13 : !qref.bit
      qref.custom "T"() %13 : !qref.bit
      qref.custom "CNOT"() %12, %13 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %13, %11 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %11, %12 : !qref.bit, !qref.bit
      qref.custom "T"() %12 adj : !qref.bit
      qref.custom "T"() %13 adj : !qref.bit
      qref.custom "CNOT"() %13, %12 : !qref.bit, !qref.bit
      qref.custom "T"() %12 adj : !qref.bit
      qref.custom "T"() %11 : !qref.bit
      qref.custom "CNOT"() %13, %11 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %11, %12 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %11 : !qref.bit
      qref.custom "T"() %11 : !qref.bit
      qref.custom "CNOT"() %10, %11 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %11, %9 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %9, %10 : !qref.bit, !qref.bit
      qref.custom "T"() %10 adj : !qref.bit
      qref.custom "T"() %11 adj : !qref.bit
      qref.custom "CNOT"() %11, %10 : !qref.bit, !qref.bit
      qref.custom "T"() %10 adj : !qref.bit
      qref.custom "T"() %9 : !qref.bit
      qref.custom "CNOT"() %11, %9 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %9, %10 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %9 : !qref.bit
      qref.custom "Hadamard"() %9 : !qref.bit
      qref.custom "T"() %9 : !qref.bit
      qref.custom "CNOT"() %10, %11 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %10 : !qref.bit
      qref.custom "PauliX"() %10 : !qref.bit
      qref.custom "T"() %10 : !qref.bit
      qref.custom "Hadamard"() %11 : !qref.bit
      qref.custom "T"() %11 : !qref.bit
      qref.custom "CNOT"() %12, %13 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %12 : !qref.bit
      qref.custom "PauliX"() %12 : !qref.bit
      qref.custom "T"() %12 : !qref.bit
      qref.custom "Hadamard"() %13 : !qref.bit
      qref.custom "T"() %13 : !qref.bit
      qref.custom "CNOT"() %14, %15 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %14 : !qref.bit
      qref.custom "PauliX"() %14 : !qref.bit
      qref.custom "T"() %14 : !qref.bit
      qref.custom "Hadamard"() %15 : !qref.bit
      qref.custom "T"() %15 : !qref.bit
      qref.custom "CNOT"() %16, %17 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %16 : !qref.bit
      qref.custom "PauliX"() %16 : !qref.bit
      qref.custom "T"() %16 : !qref.bit
      qref.custom "Hadamard"() %17 : !qref.bit
      qref.custom "T"() %17 : !qref.bit
      qref.custom "CNOT"() %18, %19 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %18 : !qref.bit
      qref.custom "PauliX"() %18 : !qref.bit
      qref.custom "T"() %18 : !qref.bit
      qref.custom "Hadamard"() %19 : !qref.bit
      qref.custom "T"() %19 : !qref.bit
      qref.custom "CNOT"() %20, %21 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %20 : !qref.bit
      qref.custom "PauliX"() %20 : !qref.bit
      qref.custom "T"() %20 : !qref.bit
      qref.custom "Hadamard"() %21 : !qref.bit
      qref.custom "T"() %21 : !qref.bit
      qref.custom "CNOT"() %22, %23 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %22 : !qref.bit
      qref.custom "PauliX"() %22 : !qref.bit
      qref.custom "T"() %22 : !qref.bit
      qref.custom "Hadamard"() %23 : !qref.bit
      qref.custom "T"() %23 : !qref.bit
      qref.custom "CNOT"() %24, %25 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %24 : !qref.bit
      qref.custom "PauliX"() %24 : !qref.bit
      qref.custom "T"() %24 : !qref.bit
      qref.custom "Hadamard"() %25 : !qref.bit
      qref.custom "T"() %25 : !qref.bit
      qref.custom "CNOT"() %26, %27 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %26 : !qref.bit
      qref.custom "PauliX"() %26 : !qref.bit
      qref.custom "T"() %26 : !qref.bit
      qref.custom "Hadamard"() %27 : !qref.bit
      qref.custom "T"() %27 : !qref.bit
      qref.custom "CNOT"() %28, %29 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %28 : !qref.bit
      qref.custom "PauliX"() %28 : !qref.bit
      qref.custom "T"() %28 : !qref.bit
      qref.custom "Hadamard"() %29 : !qref.bit
      qref.custom "T"() %29 : !qref.bit
      qref.custom "CNOT"() %30, %31 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %30 : !qref.bit
      qref.custom "PauliX"() %30 : !qref.bit
      qref.custom "T"() %30 : !qref.bit
      qref.custom "Hadamard"() %31 : !qref.bit
      qref.custom "T"() %31 : !qref.bit
      qref.custom "CNOT"() %32, %33 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %32 : !qref.bit
      qref.custom "PauliX"() %32 : !qref.bit
      qref.custom "T"() %32 : !qref.bit
      qref.custom "Hadamard"() %33 : !qref.bit
      qref.custom "T"() %33 : !qref.bit
      qref.custom "CNOT"() %34, %35 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %34 : !qref.bit
      qref.custom "PauliX"() %34 : !qref.bit
      qref.custom "T"() %34 : !qref.bit
      qref.custom "Hadamard"() %35 : !qref.bit
      qref.custom "T"() %35 : !qref.bit
      qref.custom "CNOT"() %36, %37 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %36 : !qref.bit
      qref.custom "PauliX"() %36 : !qref.bit
      qref.custom "T"() %36 : !qref.bit
      qref.custom "Hadamard"() %37 : !qref.bit
      qref.custom "T"() %37 : !qref.bit
      qref.custom "CNOT"() %38, %39 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %38 : !qref.bit
      qref.custom "PauliX"() %38 : !qref.bit
      qref.custom "T"() %38 : !qref.bit
      qref.custom "Hadamard"() %39 : !qref.bit
      qref.custom "T"() %39 : !qref.bit
      qref.custom "CNOT"() %40, %41 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %40 : !qref.bit
      qref.custom "PauliX"() %40 : !qref.bit
      qref.custom "T"() %40 : !qref.bit
      qref.custom "Hadamard"() %41 : !qref.bit
      qref.custom "T"() %41 : !qref.bit
      qref.custom "CNOT"() %42, %43 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %42 : !qref.bit
      qref.custom "PauliX"() %42 : !qref.bit
      qref.custom "T"() %42 : !qref.bit
      qref.custom "Hadamard"() %43 : !qref.bit
      qref.custom "T"() %43 : !qref.bit
      qref.custom "CNOT"() %44, %45 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %44 : !qref.bit
      qref.custom "PauliX"() %44 : !qref.bit
      qref.custom "T"() %44 : !qref.bit
      qref.custom "Hadamard"() %45 : !qref.bit
      qref.custom "T"() %45 : !qref.bit
      qref.custom "CNOT"() %46, %47 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %46 : !qref.bit
      qref.custom "PauliX"() %46 : !qref.bit
      qref.custom "T"() %46 : !qref.bit
      qref.custom "Hadamard"() %47 : !qref.bit
      qref.custom "T"() %47 : !qref.bit
      qref.custom "CNOT"() %48, %49 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %48 : !qref.bit
      qref.custom "PauliX"() %48 : !qref.bit
      qref.custom "T"() %48 : !qref.bit
      qref.custom "Hadamard"() %49 : !qref.bit
      qref.custom "T"() %49 : !qref.bit
      qref.custom "CNOT"() %50, %51 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %50 : !qref.bit
      qref.custom "PauliX"() %50 : !qref.bit
      qref.custom "T"() %50 : !qref.bit
      qref.custom "Hadamard"() %51 : !qref.bit
      qref.custom "T"() %51 : !qref.bit
      qref.custom "CNOT"() %52, %53 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %52 : !qref.bit
      qref.custom "PauliX"() %52 : !qref.bit
      qref.custom "T"() %52 : !qref.bit
      qref.custom "Hadamard"() %53 : !qref.bit
      qref.custom "T"() %53 : !qref.bit
      qref.custom "CNOT"() %54, %55 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %54 : !qref.bit
      qref.custom "PauliX"() %54 : !qref.bit
      qref.custom "T"() %54 : !qref.bit
      qref.custom "Hadamard"() %55 : !qref.bit
      qref.custom "T"() %55 : !qref.bit
      qref.custom "CNOT"() %56, %57 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %56 : !qref.bit
      qref.custom "PauliX"() %56 : !qref.bit
      qref.custom "T"() %56 : !qref.bit
      qref.custom "Hadamard"() %57 : !qref.bit
      qref.custom "T"() %57 : !qref.bit
      qref.custom "CNOT"() %58, %59 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %58 : !qref.bit
      qref.custom "PauliX"() %58 : !qref.bit
      qref.custom "T"() %58 : !qref.bit
      qref.custom "Hadamard"() %59 : !qref.bit
      qref.custom "T"() %59 : !qref.bit
      qref.custom "CNOT"() %60, %61 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %60 : !qref.bit
      qref.custom "PauliX"() %60 : !qref.bit
      qref.custom "T"() %60 : !qref.bit
      qref.custom "Hadamard"() %61 : !qref.bit
      qref.custom "T"() %61 : !qref.bit
      qref.custom "CNOT"() %62, %63 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %62 : !qref.bit
      qref.custom "PauliX"() %62 : !qref.bit
      qref.custom "T"() %62 : !qref.bit
      qref.custom "Hadamard"() %63 : !qref.bit
      qref.custom "T"() %63 : !qref.bit
      qref.custom "CNOT"() %64, %65 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %64 : !qref.bit
      qref.custom "PauliX"() %64 : !qref.bit
      qref.custom "T"() %64 : !qref.bit
      qref.custom "Hadamard"() %65 : !qref.bit
      qref.custom "T"() %65 : !qref.bit
      qref.custom "CNOT"() %66, %67 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %66 : !qref.bit
      qref.custom "PauliX"() %66 : !qref.bit
      qref.custom "T"() %66 : !qref.bit
      qref.custom "Hadamard"() %67 : !qref.bit
      qref.custom "T"() %67 : !qref.bit
      qref.custom "CNOT"() %68, %69 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %68 : !qref.bit
      qref.custom "PauliX"() %68 : !qref.bit
      qref.custom "T"() %68 : !qref.bit
      qref.custom "Hadamard"() %69 : !qref.bit
      qref.custom "T"() %69 : !qref.bit
      qref.custom "CNOT"() %70, %71 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %70 : !qref.bit
      qref.custom "PauliX"() %70 : !qref.bit
      qref.custom "T"() %70 : !qref.bit
      qref.custom "Hadamard"() %71 : !qref.bit
      qref.custom "T"() %71 : !qref.bit
      qref.custom "CNOT"() %72, %73 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %72 : !qref.bit
      qref.custom "PauliX"() %72 : !qref.bit
      qref.custom "T"() %72 : !qref.bit
      qref.custom "Hadamard"() %73 : !qref.bit
      qref.custom "T"() %73 : !qref.bit
      qref.custom "CNOT"() %74, %75 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %74 : !qref.bit
      qref.custom "PauliX"() %74 : !qref.bit
      qref.custom "T"() %74 : !qref.bit
      qref.custom "Hadamard"() %75 : !qref.bit
      qref.custom "T"() %75 : !qref.bit
      qref.custom "CNOT"() %76, %77 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %76 : !qref.bit
      qref.custom "PauliX"() %76 : !qref.bit
      qref.custom "T"() %76 : !qref.bit
      qref.custom "Hadamard"() %77 : !qref.bit
      qref.custom "T"() %77 : !qref.bit
      qref.custom "CNOT"() %78, %79 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %78 : !qref.bit
      qref.custom "PauliX"() %78 : !qref.bit
      qref.custom "T"() %78 : !qref.bit
      qref.custom "Hadamard"() %79 : !qref.bit
      qref.custom "T"() %79 : !qref.bit
      qref.custom "CNOT"() %80, %81 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %80 : !qref.bit
      qref.custom "PauliX"() %80 : !qref.bit
      qref.custom "T"() %80 : !qref.bit
      qref.custom "Hadamard"() %81 : !qref.bit
      qref.custom "T"() %81 : !qref.bit
      qref.custom "CNOT"() %82, %83 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %82 : !qref.bit
      qref.custom "PauliX"() %82 : !qref.bit
      qref.custom "T"() %82 : !qref.bit
      qref.custom "Hadamard"() %83 : !qref.bit
      qref.custom "T"() %83 : !qref.bit
      qref.custom "CNOT"() %84, %85 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %84 : !qref.bit
      qref.custom "PauliX"() %84 : !qref.bit
      qref.custom "T"() %84 : !qref.bit
      qref.custom "Hadamard"() %85 : !qref.bit
      qref.custom "T"() %85 : !qref.bit
      qref.custom "CNOT"() %86, %87 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %86 : !qref.bit
      qref.custom "PauliX"() %86 : !qref.bit
      qref.custom "T"() %86 : !qref.bit
      qref.custom "Hadamard"() %87 : !qref.bit
      qref.custom "T"() %87 : !qref.bit
      qref.custom "CNOT"() %88, %89 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %88 : !qref.bit
      qref.custom "PauliX"() %88 : !qref.bit
      qref.custom "T"() %88 : !qref.bit
      qref.custom "Hadamard"() %89 : !qref.bit
      qref.custom "T"() %89 : !qref.bit
      qref.custom "CNOT"() %90, %91 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %90 : !qref.bit
      qref.custom "PauliX"() %90 : !qref.bit
      qref.custom "T"() %90 : !qref.bit
      qref.custom "Hadamard"() %91 : !qref.bit
      qref.custom "T"() %91 : !qref.bit
      qref.custom "CNOT"() %92, %93 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %92 : !qref.bit
      qref.custom "PauliX"() %92 : !qref.bit
      qref.custom "T"() %92 : !qref.bit
      qref.custom "Hadamard"() %93 : !qref.bit
      qref.custom "T"() %93 : !qref.bit
      qref.custom "CNOT"() %94, %95 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %94 : !qref.bit
      qref.custom "PauliX"() %94 : !qref.bit
      qref.custom "T"() %94 : !qref.bit
      qref.custom "Hadamard"() %95 : !qref.bit
      qref.custom "T"() %95 : !qref.bit
      qref.custom "CNOT"() %96, %97 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %96 : !qref.bit
      qref.custom "PauliX"() %96 : !qref.bit
      qref.custom "T"() %96 : !qref.bit
      qref.custom "Hadamard"() %97 : !qref.bit
      qref.custom "T"() %97 : !qref.bit
      qref.custom "CNOT"() %98, %99 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %98 : !qref.bit
      qref.custom "PauliX"() %98 : !qref.bit
      qref.custom "T"() %98 : !qref.bit
      qref.custom "Hadamard"() %99 : !qref.bit
      qref.custom "T"() %99 : !qref.bit
      qref.custom "CNOT"() %100, %101 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %100 : !qref.bit
      qref.custom "PauliX"() %100 : !qref.bit
      qref.custom "T"() %100 : !qref.bit
      qref.custom "Hadamard"() %101 : !qref.bit
      qref.custom "T"() %101 : !qref.bit
      qref.custom "CNOT"() %102, %103 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %102 : !qref.bit
      qref.custom "PauliX"() %102 : !qref.bit
      qref.custom "T"() %102 : !qref.bit
      qref.custom "Hadamard"() %103 : !qref.bit
      qref.custom "T"() %103 : !qref.bit
      qref.custom "CNOT"() %104, %105 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %104 : !qref.bit
      qref.custom "PauliX"() %104 : !qref.bit
      qref.custom "T"() %104 : !qref.bit
      qref.custom "Hadamard"() %105 : !qref.bit
      qref.custom "T"() %105 : !qref.bit
      qref.custom "CNOT"() %106, %107 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %106 : !qref.bit
      qref.custom "PauliX"() %106 : !qref.bit
      qref.custom "T"() %106 : !qref.bit
      qref.custom "Hadamard"() %107 : !qref.bit
      qref.custom "T"() %107 : !qref.bit
      qref.custom "CNOT"() %108, %109 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %108 : !qref.bit
      qref.custom "PauliX"() %108 : !qref.bit
      qref.custom "T"() %108 : !qref.bit
      qref.custom "Hadamard"() %109 : !qref.bit
      qref.custom "T"() %109 : !qref.bit
      qref.custom "CNOT"() %110, %111 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %110 : !qref.bit
      qref.custom "PauliX"() %110 : !qref.bit
      qref.custom "T"() %110 : !qref.bit
      qref.custom "Hadamard"() %111 : !qref.bit
      qref.custom "T"() %111 : !qref.bit
      qref.custom "CNOT"() %112, %113 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %112 : !qref.bit
      qref.custom "PauliX"() %112 : !qref.bit
      qref.custom "T"() %112 : !qref.bit
      qref.custom "Hadamard"() %113 : !qref.bit
      qref.custom "T"() %113 : !qref.bit
      qref.custom "CNOT"() %114, %115 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %114 : !qref.bit
      qref.custom "PauliX"() %114 : !qref.bit
      qref.custom "T"() %114 : !qref.bit
      qref.custom "Hadamard"() %115 : !qref.bit
      qref.custom "T"() %115 : !qref.bit
      qref.custom "CNOT"() %116, %117 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %116 : !qref.bit
      qref.custom "PauliX"() %116 : !qref.bit
      qref.custom "T"() %116 : !qref.bit
      qref.custom "Hadamard"() %117 : !qref.bit
      qref.custom "T"() %117 : !qref.bit
      qref.custom "CNOT"() %118, %119 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %118 : !qref.bit
      qref.custom "PauliX"() %118 : !qref.bit
      qref.custom "T"() %118 : !qref.bit
      qref.custom "Hadamard"() %119 : !qref.bit
      qref.custom "T"() %119 : !qref.bit
      qref.custom "CNOT"() %120, %121 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %120 : !qref.bit
      qref.custom "PauliX"() %120 : !qref.bit
      qref.custom "T"() %120 : !qref.bit
      qref.custom "Hadamard"() %121 : !qref.bit
      qref.custom "T"() %121 : !qref.bit
      qref.custom "CNOT"() %122, %123 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %122 : !qref.bit
      qref.custom "PauliX"() %122 : !qref.bit
      qref.custom "T"() %122 : !qref.bit
      qref.custom "Hadamard"() %123 : !qref.bit
      qref.custom "T"() %123 : !qref.bit
      qref.custom "CNOT"() %124, %125 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %124 : !qref.bit
      qref.custom "PauliX"() %124 : !qref.bit
      qref.custom "T"() %124 : !qref.bit
      qref.custom "Hadamard"() %125 : !qref.bit
      qref.custom "T"() %125 : !qref.bit
      qref.custom "CNOT"() %126, %127 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %126 : !qref.bit
      qref.custom "PauliX"() %126 : !qref.bit
      qref.custom "T"() %126 : !qref.bit
      qref.custom "Hadamard"() %127 : !qref.bit
      qref.custom "T"() %127 : !qref.bit
      qref.custom "CNOT"() %128, %129 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %128 : !qref.bit
      qref.custom "PauliX"() %128 : !qref.bit
      qref.custom "T"() %128 : !qref.bit
      qref.custom "Hadamard"() %129 : !qref.bit
      qref.custom "T"() %129 : !qref.bit
      qref.custom "CNOT"() %1, %130 : !qref.bit, !qref.bit
      qref.custom "T"() %1 : !qref.bit
      qref.custom "T"() %130 : !qref.bit
      qref.custom "CNOT"() %1, %130 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %130, %129 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %129, %1 : !qref.bit, !qref.bit
      qref.custom "T"() %1 adj : !qref.bit
      qref.custom "T"() %130 adj : !qref.bit
      qref.custom "CNOT"() %130, %1 : !qref.bit, !qref.bit
      qref.custom "T"() %1 adj : !qref.bit
      qref.custom "T"() %129 : !qref.bit
      qref.custom "CNOT"() %130, %129 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %129, %1 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %129 : !qref.bit
      qref.custom "Hadamard"() %129 : !qref.bit
      qref.custom "T"() %129 : !qref.bit
      qref.custom "CNOT"() %1, %130 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %1 : !qref.bit
      qref.custom "PauliX"() %1 : !qref.bit
      qref.custom "T"() %1 : !qref.bit
      qref.custom "Hadamard"() %130 : !qref.bit
      qref.custom "PauliX"() %130 : !qref.bit
      qref.custom "T"() %130 : !qref.bit
      qref.custom "CNOT"() %1, %130 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %130, %129 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %129, %1 : !qref.bit, !qref.bit
      qref.custom "T"() %1 adj : !qref.bit
      qref.custom "T"() %130 adj : !qref.bit
      qref.custom "CNOT"() %130, %1 : !qref.bit, !qref.bit
      qref.custom "T"() %1 adj : !qref.bit
      qref.custom "T"() %129 : !qref.bit
      qref.custom "CNOT"() %130, %129 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %129, %1 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %129 : !qref.bit
      qref.custom "T"() %129 : !qref.bit
      qref.custom "CNOT"() %128, %129 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %129, %127 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %127, %128 : !qref.bit, !qref.bit
      qref.custom "T"() %128 adj : !qref.bit
      qref.custom "T"() %129 adj : !qref.bit
      qref.custom "CNOT"() %129, %128 : !qref.bit, !qref.bit
      qref.custom "T"() %128 adj : !qref.bit
      qref.custom "T"() %127 : !qref.bit
      qref.custom "CNOT"() %129, %127 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %127, %128 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %127 : !qref.bit
      qref.custom "T"() %127 : !qref.bit
      qref.custom "CNOT"() %126, %127 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %127, %125 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %125, %126 : !qref.bit, !qref.bit
      qref.custom "T"() %126 adj : !qref.bit
      qref.custom "T"() %127 adj : !qref.bit
      qref.custom "CNOT"() %127, %126 : !qref.bit, !qref.bit
      qref.custom "T"() %126 adj : !qref.bit
      qref.custom "T"() %125 : !qref.bit
      qref.custom "CNOT"() %127, %125 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %125, %126 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %125 : !qref.bit
      qref.custom "T"() %125 : !qref.bit
      qref.custom "CNOT"() %124, %125 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %125, %123 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %123, %124 : !qref.bit, !qref.bit
      qref.custom "T"() %124 adj : !qref.bit
      qref.custom "T"() %125 adj : !qref.bit
      qref.custom "CNOT"() %125, %124 : !qref.bit, !qref.bit
      qref.custom "T"() %124 adj : !qref.bit
      qref.custom "T"() %123 : !qref.bit
      qref.custom "CNOT"() %125, %123 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %123, %124 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %123 : !qref.bit
      qref.custom "T"() %123 : !qref.bit
      qref.custom "CNOT"() %122, %123 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %123, %121 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %121, %122 : !qref.bit, !qref.bit
      qref.custom "T"() %122 adj : !qref.bit
      qref.custom "T"() %123 adj : !qref.bit
      qref.custom "CNOT"() %123, %122 : !qref.bit, !qref.bit
      qref.custom "T"() %122 adj : !qref.bit
      qref.custom "T"() %121 : !qref.bit
      qref.custom "CNOT"() %123, %121 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %121, %122 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %121 : !qref.bit
      qref.custom "T"() %121 : !qref.bit
      qref.custom "CNOT"() %120, %121 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %121, %119 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %119, %120 : !qref.bit, !qref.bit
      qref.custom "T"() %120 adj : !qref.bit
      qref.custom "T"() %121 adj : !qref.bit
      qref.custom "CNOT"() %121, %120 : !qref.bit, !qref.bit
      qref.custom "T"() %120 adj : !qref.bit
      qref.custom "T"() %119 : !qref.bit
      qref.custom "CNOT"() %121, %119 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %119, %120 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %119 : !qref.bit
      qref.custom "T"() %119 : !qref.bit
      qref.custom "CNOT"() %118, %119 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %119, %117 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %117, %118 : !qref.bit, !qref.bit
      qref.custom "T"() %118 adj : !qref.bit
      qref.custom "T"() %119 adj : !qref.bit
      qref.custom "CNOT"() %119, %118 : !qref.bit, !qref.bit
      qref.custom "T"() %118 adj : !qref.bit
      qref.custom "T"() %117 : !qref.bit
      qref.custom "CNOT"() %119, %117 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %117, %118 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %117 : !qref.bit
      qref.custom "T"() %117 : !qref.bit
      qref.custom "CNOT"() %116, %117 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %117, %115 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %115, %116 : !qref.bit, !qref.bit
      qref.custom "T"() %116 adj : !qref.bit
      qref.custom "T"() %117 adj : !qref.bit
      qref.custom "CNOT"() %117, %116 : !qref.bit, !qref.bit
      qref.custom "T"() %116 adj : !qref.bit
      qref.custom "T"() %115 : !qref.bit
      qref.custom "CNOT"() %117, %115 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %115, %116 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %115 : !qref.bit
      qref.custom "T"() %115 : !qref.bit
      qref.custom "CNOT"() %114, %115 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %115, %113 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %113, %114 : !qref.bit, !qref.bit
      qref.custom "T"() %114 adj : !qref.bit
      qref.custom "T"() %115 adj : !qref.bit
      qref.custom "CNOT"() %115, %114 : !qref.bit, !qref.bit
      qref.custom "T"() %114 adj : !qref.bit
      qref.custom "T"() %113 : !qref.bit
      qref.custom "CNOT"() %115, %113 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %113, %114 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %113 : !qref.bit
      qref.custom "T"() %113 : !qref.bit
      qref.custom "CNOT"() %112, %113 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %113, %111 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %111, %112 : !qref.bit, !qref.bit
      qref.custom "T"() %112 adj : !qref.bit
      qref.custom "T"() %113 adj : !qref.bit
      qref.custom "CNOT"() %113, %112 : !qref.bit, !qref.bit
      qref.custom "T"() %112 adj : !qref.bit
      qref.custom "T"() %111 : !qref.bit
      qref.custom "CNOT"() %113, %111 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %111, %112 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %111 : !qref.bit
      qref.custom "T"() %111 : !qref.bit
      qref.custom "CNOT"() %110, %111 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %111, %109 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %109, %110 : !qref.bit, !qref.bit
      qref.custom "T"() %110 adj : !qref.bit
      qref.custom "T"() %111 adj : !qref.bit
      qref.custom "CNOT"() %111, %110 : !qref.bit, !qref.bit
      qref.custom "T"() %110 adj : !qref.bit
      qref.custom "T"() %109 : !qref.bit
      qref.custom "CNOT"() %111, %109 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %109, %110 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %109 : !qref.bit
      qref.custom "T"() %109 : !qref.bit
      qref.custom "CNOT"() %108, %109 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %109, %107 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %107, %108 : !qref.bit, !qref.bit
      qref.custom "T"() %108 adj : !qref.bit
      qref.custom "T"() %109 adj : !qref.bit
      qref.custom "CNOT"() %109, %108 : !qref.bit, !qref.bit
      qref.custom "T"() %108 adj : !qref.bit
      qref.custom "T"() %107 : !qref.bit
      qref.custom "CNOT"() %109, %107 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %107, %108 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %107 : !qref.bit
      qref.custom "T"() %107 : !qref.bit
      qref.custom "CNOT"() %106, %107 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %107, %105 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %105, %106 : !qref.bit, !qref.bit
      qref.custom "T"() %106 adj : !qref.bit
      qref.custom "T"() %107 adj : !qref.bit
      qref.custom "CNOT"() %107, %106 : !qref.bit, !qref.bit
      qref.custom "T"() %106 adj : !qref.bit
      qref.custom "T"() %105 : !qref.bit
      qref.custom "CNOT"() %107, %105 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %105, %106 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %105 : !qref.bit
      qref.custom "T"() %105 : !qref.bit
      qref.custom "CNOT"() %104, %105 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %105, %103 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %103, %104 : !qref.bit, !qref.bit
      qref.custom "T"() %104 adj : !qref.bit
      qref.custom "T"() %105 adj : !qref.bit
      qref.custom "CNOT"() %105, %104 : !qref.bit, !qref.bit
      qref.custom "T"() %104 adj : !qref.bit
      qref.custom "T"() %103 : !qref.bit
      qref.custom "CNOT"() %105, %103 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %103, %104 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %103 : !qref.bit
      qref.custom "T"() %103 : !qref.bit
      qref.custom "CNOT"() %102, %103 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %103, %101 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %101, %102 : !qref.bit, !qref.bit
      qref.custom "T"() %102 adj : !qref.bit
      qref.custom "T"() %103 adj : !qref.bit
      qref.custom "CNOT"() %103, %102 : !qref.bit, !qref.bit
      qref.custom "T"() %102 adj : !qref.bit
      qref.custom "T"() %101 : !qref.bit
      qref.custom "CNOT"() %103, %101 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %101, %102 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %101 : !qref.bit
      qref.custom "T"() %101 : !qref.bit
      qref.custom "CNOT"() %100, %101 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %101, %99 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %99, %100 : !qref.bit, !qref.bit
      qref.custom "T"() %100 adj : !qref.bit
      qref.custom "T"() %101 adj : !qref.bit
      qref.custom "CNOT"() %101, %100 : !qref.bit, !qref.bit
      qref.custom "T"() %100 adj : !qref.bit
      qref.custom "T"() %99 : !qref.bit
      qref.custom "CNOT"() %101, %99 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %99, %100 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %99 : !qref.bit
      qref.custom "T"() %99 : !qref.bit
      qref.custom "CNOT"() %98, %99 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %99, %97 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %97, %98 : !qref.bit, !qref.bit
      qref.custom "T"() %98 adj : !qref.bit
      qref.custom "T"() %99 adj : !qref.bit
      qref.custom "CNOT"() %99, %98 : !qref.bit, !qref.bit
      qref.custom "T"() %98 adj : !qref.bit
      qref.custom "T"() %97 : !qref.bit
      qref.custom "CNOT"() %99, %97 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %97, %98 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %97 : !qref.bit
      qref.custom "T"() %97 : !qref.bit
      qref.custom "CNOT"() %96, %97 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %97, %95 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %95, %96 : !qref.bit, !qref.bit
      qref.custom "T"() %96 adj : !qref.bit
      qref.custom "T"() %97 adj : !qref.bit
      qref.custom "CNOT"() %97, %96 : !qref.bit, !qref.bit
      qref.custom "T"() %96 adj : !qref.bit
      qref.custom "T"() %95 : !qref.bit
      qref.custom "CNOT"() %97, %95 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %95, %96 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %95 : !qref.bit
      qref.custom "T"() %95 : !qref.bit
      qref.custom "CNOT"() %94, %95 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %95, %93 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %93, %94 : !qref.bit, !qref.bit
      qref.custom "T"() %94 adj : !qref.bit
      qref.custom "T"() %95 adj : !qref.bit
      qref.custom "CNOT"() %95, %94 : !qref.bit, !qref.bit
      qref.custom "T"() %94 adj : !qref.bit
      qref.custom "T"() %93 : !qref.bit
      qref.custom "CNOT"() %95, %93 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %93, %94 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %93 : !qref.bit
      qref.custom "T"() %93 : !qref.bit
      qref.custom "CNOT"() %92, %93 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %93, %91 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %91, %92 : !qref.bit, !qref.bit
      qref.custom "T"() %92 adj : !qref.bit
      qref.custom "T"() %93 adj : !qref.bit
      qref.custom "CNOT"() %93, %92 : !qref.bit, !qref.bit
      qref.custom "T"() %92 adj : !qref.bit
      qref.custom "T"() %91 : !qref.bit
      qref.custom "CNOT"() %93, %91 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %91, %92 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %91 : !qref.bit
      qref.custom "T"() %91 : !qref.bit
      qref.custom "CNOT"() %90, %91 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %91, %89 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %89, %90 : !qref.bit, !qref.bit
      qref.custom "T"() %90 adj : !qref.bit
      qref.custom "T"() %91 adj : !qref.bit
      qref.custom "CNOT"() %91, %90 : !qref.bit, !qref.bit
      qref.custom "T"() %90 adj : !qref.bit
      qref.custom "T"() %89 : !qref.bit
      qref.custom "CNOT"() %91, %89 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %89, %90 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %89 : !qref.bit
      qref.custom "T"() %89 : !qref.bit
      qref.custom "CNOT"() %88, %89 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %89, %87 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %87, %88 : !qref.bit, !qref.bit
      qref.custom "T"() %88 adj : !qref.bit
      qref.custom "T"() %89 adj : !qref.bit
      qref.custom "CNOT"() %89, %88 : !qref.bit, !qref.bit
      qref.custom "T"() %88 adj : !qref.bit
      qref.custom "T"() %87 : !qref.bit
      qref.custom "CNOT"() %89, %87 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %87, %88 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %87 : !qref.bit
      qref.custom "T"() %87 : !qref.bit
      qref.custom "CNOT"() %86, %87 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %87, %85 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %85, %86 : !qref.bit, !qref.bit
      qref.custom "T"() %86 adj : !qref.bit
      qref.custom "T"() %87 adj : !qref.bit
      qref.custom "CNOT"() %87, %86 : !qref.bit, !qref.bit
      qref.custom "T"() %86 adj : !qref.bit
      qref.custom "T"() %85 : !qref.bit
      qref.custom "CNOT"() %87, %85 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %85, %86 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %85 : !qref.bit
      qref.custom "T"() %85 : !qref.bit
      qref.custom "CNOT"() %84, %85 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %85, %83 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %83, %84 : !qref.bit, !qref.bit
      qref.custom "T"() %84 adj : !qref.bit
      qref.custom "T"() %85 adj : !qref.bit
      qref.custom "CNOT"() %85, %84 : !qref.bit, !qref.bit
      qref.custom "T"() %84 adj : !qref.bit
      qref.custom "T"() %83 : !qref.bit
      qref.custom "CNOT"() %85, %83 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %83, %84 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %83 : !qref.bit
      qref.custom "T"() %83 : !qref.bit
      qref.custom "CNOT"() %82, %83 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %83, %81 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %81, %82 : !qref.bit, !qref.bit
      qref.custom "T"() %82 adj : !qref.bit
      qref.custom "T"() %83 adj : !qref.bit
      qref.custom "CNOT"() %83, %82 : !qref.bit, !qref.bit
      qref.custom "T"() %82 adj : !qref.bit
      qref.custom "T"() %81 : !qref.bit
      qref.custom "CNOT"() %83, %81 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %81, %82 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %81 : !qref.bit
      qref.custom "T"() %81 : !qref.bit
      qref.custom "CNOT"() %80, %81 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %81, %79 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %79, %80 : !qref.bit, !qref.bit
      qref.custom "T"() %80 adj : !qref.bit
      qref.custom "T"() %81 adj : !qref.bit
      qref.custom "CNOT"() %81, %80 : !qref.bit, !qref.bit
      qref.custom "T"() %80 adj : !qref.bit
      qref.custom "T"() %79 : !qref.bit
      qref.custom "CNOT"() %81, %79 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %79, %80 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %79 : !qref.bit
      qref.custom "T"() %79 : !qref.bit
      qref.custom "CNOT"() %78, %79 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %79, %77 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %77, %78 : !qref.bit, !qref.bit
      qref.custom "T"() %78 adj : !qref.bit
      qref.custom "T"() %79 adj : !qref.bit
      qref.custom "CNOT"() %79, %78 : !qref.bit, !qref.bit
      qref.custom "T"() %78 adj : !qref.bit
      qref.custom "T"() %77 : !qref.bit
      qref.custom "CNOT"() %79, %77 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %77, %78 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %77 : !qref.bit
      qref.custom "T"() %77 : !qref.bit
      qref.custom "CNOT"() %76, %77 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %77, %75 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %75, %76 : !qref.bit, !qref.bit
      qref.custom "T"() %76 adj : !qref.bit
      qref.custom "T"() %77 adj : !qref.bit
      qref.custom "CNOT"() %77, %76 : !qref.bit, !qref.bit
      qref.custom "T"() %76 adj : !qref.bit
      qref.custom "T"() %75 : !qref.bit
      qref.custom "CNOT"() %77, %75 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %75, %76 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %75 : !qref.bit
      qref.custom "T"() %75 : !qref.bit
      qref.custom "CNOT"() %74, %75 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %75, %73 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %73, %74 : !qref.bit, !qref.bit
      qref.custom "T"() %74 adj : !qref.bit
      qref.custom "T"() %75 adj : !qref.bit
      qref.custom "CNOT"() %75, %74 : !qref.bit, !qref.bit
      qref.custom "T"() %74 adj : !qref.bit
      qref.custom "T"() %73 : !qref.bit
      qref.custom "CNOT"() %75, %73 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %73, %74 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %73 : !qref.bit
      qref.custom "T"() %73 : !qref.bit
      qref.custom "CNOT"() %72, %73 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %73, %71 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %71, %72 : !qref.bit, !qref.bit
      qref.custom "T"() %72 adj : !qref.bit
      qref.custom "T"() %73 adj : !qref.bit
      qref.custom "CNOT"() %73, %72 : !qref.bit, !qref.bit
      qref.custom "T"() %72 adj : !qref.bit
      qref.custom "T"() %71 : !qref.bit
      qref.custom "CNOT"() %73, %71 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %71, %72 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %71 : !qref.bit
      qref.custom "T"() %71 : !qref.bit
      qref.custom "CNOT"() %70, %71 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %71, %69 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %69, %70 : !qref.bit, !qref.bit
      qref.custom "T"() %70 adj : !qref.bit
      qref.custom "T"() %71 adj : !qref.bit
      qref.custom "CNOT"() %71, %70 : !qref.bit, !qref.bit
      qref.custom "T"() %70 adj : !qref.bit
      qref.custom "T"() %69 : !qref.bit
      qref.custom "CNOT"() %71, %69 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %69, %70 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %69 : !qref.bit
      qref.custom "T"() %69 : !qref.bit
      qref.custom "CNOT"() %68, %69 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %69, %67 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %67, %68 : !qref.bit, !qref.bit
      qref.custom "T"() %68 adj : !qref.bit
      qref.custom "T"() %69 adj : !qref.bit
      qref.custom "CNOT"() %69, %68 : !qref.bit, !qref.bit
      qref.custom "T"() %68 adj : !qref.bit
      qref.custom "T"() %67 : !qref.bit
      qref.custom "CNOT"() %69, %67 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %67, %68 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %67 : !qref.bit
      qref.custom "T"() %67 : !qref.bit
      qref.custom "CNOT"() %66, %67 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %67, %65 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %65, %66 : !qref.bit, !qref.bit
      qref.custom "T"() %66 adj : !qref.bit
      qref.custom "T"() %67 adj : !qref.bit
      qref.custom "CNOT"() %67, %66 : !qref.bit, !qref.bit
      qref.custom "T"() %66 adj : !qref.bit
      qref.custom "T"() %65 : !qref.bit
      qref.custom "CNOT"() %67, %65 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %65, %66 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %65 : !qref.bit
      qref.custom "T"() %65 : !qref.bit
      qref.custom "CNOT"() %64, %65 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %65, %63 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %63, %64 : !qref.bit, !qref.bit
      qref.custom "T"() %64 adj : !qref.bit
      qref.custom "T"() %65 adj : !qref.bit
      qref.custom "CNOT"() %65, %64 : !qref.bit, !qref.bit
      qref.custom "T"() %64 adj : !qref.bit
      qref.custom "T"() %63 : !qref.bit
      qref.custom "CNOT"() %65, %63 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %63, %64 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %63 : !qref.bit
      qref.custom "T"() %63 : !qref.bit
      qref.custom "CNOT"() %62, %63 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %63, %61 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %61, %62 : !qref.bit, !qref.bit
      qref.custom "T"() %62 adj : !qref.bit
      qref.custom "T"() %63 adj : !qref.bit
      qref.custom "CNOT"() %63, %62 : !qref.bit, !qref.bit
      qref.custom "T"() %62 adj : !qref.bit
      qref.custom "T"() %61 : !qref.bit
      qref.custom "CNOT"() %63, %61 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %61, %62 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %61 : !qref.bit
      qref.custom "T"() %61 : !qref.bit
      qref.custom "CNOT"() %60, %61 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %61, %59 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %59, %60 : !qref.bit, !qref.bit
      qref.custom "T"() %60 adj : !qref.bit
      qref.custom "T"() %61 adj : !qref.bit
      qref.custom "CNOT"() %61, %60 : !qref.bit, !qref.bit
      qref.custom "T"() %60 adj : !qref.bit
      qref.custom "T"() %59 : !qref.bit
      qref.custom "CNOT"() %61, %59 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %59, %60 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %59 : !qref.bit
      qref.custom "T"() %59 : !qref.bit
      qref.custom "CNOT"() %58, %59 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %59, %57 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %57, %58 : !qref.bit, !qref.bit
      qref.custom "T"() %58 adj : !qref.bit
      qref.custom "T"() %59 adj : !qref.bit
      qref.custom "CNOT"() %59, %58 : !qref.bit, !qref.bit
      qref.custom "T"() %58 adj : !qref.bit
      qref.custom "T"() %57 : !qref.bit
      qref.custom "CNOT"() %59, %57 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %57, %58 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %57 : !qref.bit
      qref.custom "T"() %57 : !qref.bit
      qref.custom "CNOT"() %56, %57 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %57, %55 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %55, %56 : !qref.bit, !qref.bit
      qref.custom "T"() %56 adj : !qref.bit
      qref.custom "T"() %57 adj : !qref.bit
      qref.custom "CNOT"() %57, %56 : !qref.bit, !qref.bit
      qref.custom "T"() %56 adj : !qref.bit
      qref.custom "T"() %55 : !qref.bit
      qref.custom "CNOT"() %57, %55 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %55, %56 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %55 : !qref.bit
      qref.custom "T"() %55 : !qref.bit
      qref.custom "CNOT"() %54, %55 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %55, %53 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %53, %54 : !qref.bit, !qref.bit
      qref.custom "T"() %54 adj : !qref.bit
      qref.custom "T"() %55 adj : !qref.bit
      qref.custom "CNOT"() %55, %54 : !qref.bit, !qref.bit
      qref.custom "T"() %54 adj : !qref.bit
      qref.custom "T"() %53 : !qref.bit
      qref.custom "CNOT"() %55, %53 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %53, %54 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %53 : !qref.bit
      qref.custom "T"() %53 : !qref.bit
      qref.custom "CNOT"() %52, %53 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %53, %51 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %51, %52 : !qref.bit, !qref.bit
      qref.custom "T"() %52 adj : !qref.bit
      qref.custom "T"() %53 adj : !qref.bit
      qref.custom "CNOT"() %53, %52 : !qref.bit, !qref.bit
      qref.custom "T"() %52 adj : !qref.bit
      qref.custom "T"() %51 : !qref.bit
      qref.custom "CNOT"() %53, %51 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %51, %52 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %51 : !qref.bit
      qref.custom "T"() %51 : !qref.bit
      qref.custom "CNOT"() %50, %51 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %51, %49 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %49, %50 : !qref.bit, !qref.bit
      qref.custom "T"() %50 adj : !qref.bit
      qref.custom "T"() %51 adj : !qref.bit
      qref.custom "CNOT"() %51, %50 : !qref.bit, !qref.bit
      qref.custom "T"() %50 adj : !qref.bit
      qref.custom "T"() %49 : !qref.bit
      qref.custom "CNOT"() %51, %49 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %49, %50 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %49 : !qref.bit
      qref.custom "T"() %49 : !qref.bit
      qref.custom "CNOT"() %48, %49 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %49, %47 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %47, %48 : !qref.bit, !qref.bit
      qref.custom "T"() %48 adj : !qref.bit
      qref.custom "T"() %49 adj : !qref.bit
      qref.custom "CNOT"() %49, %48 : !qref.bit, !qref.bit
      qref.custom "T"() %48 adj : !qref.bit
      qref.custom "T"() %47 : !qref.bit
      qref.custom "CNOT"() %49, %47 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %47, %48 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %47 : !qref.bit
      qref.custom "T"() %47 : !qref.bit
      qref.custom "CNOT"() %46, %47 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %47, %45 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %45, %46 : !qref.bit, !qref.bit
      qref.custom "T"() %46 adj : !qref.bit
      qref.custom "T"() %47 adj : !qref.bit
      qref.custom "CNOT"() %47, %46 : !qref.bit, !qref.bit
      qref.custom "T"() %46 adj : !qref.bit
      qref.custom "T"() %45 : !qref.bit
      qref.custom "CNOT"() %47, %45 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %45, %46 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %45 : !qref.bit
      qref.custom "T"() %45 : !qref.bit
      qref.custom "CNOT"() %44, %45 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %45, %43 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %43, %44 : !qref.bit, !qref.bit
      qref.custom "T"() %44 adj : !qref.bit
      qref.custom "T"() %45 adj : !qref.bit
      qref.custom "CNOT"() %45, %44 : !qref.bit, !qref.bit
      qref.custom "T"() %44 adj : !qref.bit
      qref.custom "T"() %43 : !qref.bit
      qref.custom "CNOT"() %45, %43 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %43, %44 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %43 : !qref.bit
      qref.custom "T"() %43 : !qref.bit
      qref.custom "CNOT"() %42, %43 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %43, %41 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %41, %42 : !qref.bit, !qref.bit
      qref.custom "T"() %42 adj : !qref.bit
      qref.custom "T"() %43 adj : !qref.bit
      qref.custom "CNOT"() %43, %42 : !qref.bit, !qref.bit
      qref.custom "T"() %42 adj : !qref.bit
      qref.custom "T"() %41 : !qref.bit
      qref.custom "CNOT"() %43, %41 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %41, %42 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %41 : !qref.bit
      qref.custom "T"() %41 : !qref.bit
      qref.custom "CNOT"() %40, %41 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %41, %39 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %39, %40 : !qref.bit, !qref.bit
      qref.custom "T"() %40 adj : !qref.bit
      qref.custom "T"() %41 adj : !qref.bit
      qref.custom "CNOT"() %41, %40 : !qref.bit, !qref.bit
      qref.custom "T"() %40 adj : !qref.bit
      qref.custom "T"() %39 : !qref.bit
      qref.custom "CNOT"() %41, %39 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %39, %40 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %39 : !qref.bit
      qref.custom "T"() %39 : !qref.bit
      qref.custom "CNOT"() %38, %39 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %39, %37 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %37, %38 : !qref.bit, !qref.bit
      qref.custom "T"() %38 adj : !qref.bit
      qref.custom "T"() %39 adj : !qref.bit
      qref.custom "CNOT"() %39, %38 : !qref.bit, !qref.bit
      qref.custom "T"() %38 adj : !qref.bit
      qref.custom "T"() %37 : !qref.bit
      qref.custom "CNOT"() %39, %37 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %37, %38 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %37 : !qref.bit
      qref.custom "T"() %37 : !qref.bit
      qref.custom "CNOT"() %36, %37 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %37, %35 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %35, %36 : !qref.bit, !qref.bit
      qref.custom "T"() %36 adj : !qref.bit
      qref.custom "T"() %37 adj : !qref.bit
      qref.custom "CNOT"() %37, %36 : !qref.bit, !qref.bit
      qref.custom "T"() %36 adj : !qref.bit
      qref.custom "T"() %35 : !qref.bit
      qref.custom "CNOT"() %37, %35 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %35, %36 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %35 : !qref.bit
      qref.custom "T"() %35 : !qref.bit
      qref.custom "CNOT"() %34, %35 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %35, %33 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %33, %34 : !qref.bit, !qref.bit
      qref.custom "T"() %34 adj : !qref.bit
      qref.custom "T"() %35 adj : !qref.bit
      qref.custom "CNOT"() %35, %34 : !qref.bit, !qref.bit
      qref.custom "T"() %34 adj : !qref.bit
      qref.custom "T"() %33 : !qref.bit
      qref.custom "CNOT"() %35, %33 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %33, %34 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %33 : !qref.bit
      qref.custom "T"() %33 : !qref.bit
      qref.custom "CNOT"() %32, %33 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %33, %31 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %31, %32 : !qref.bit, !qref.bit
      qref.custom "T"() %32 adj : !qref.bit
      qref.custom "T"() %33 adj : !qref.bit
      qref.custom "CNOT"() %33, %32 : !qref.bit, !qref.bit
      qref.custom "T"() %32 adj : !qref.bit
      qref.custom "T"() %31 : !qref.bit
      qref.custom "CNOT"() %33, %31 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %31, %32 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %31 : !qref.bit
      qref.custom "T"() %31 : !qref.bit
      qref.custom "CNOT"() %30, %31 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %31, %29 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %29, %30 : !qref.bit, !qref.bit
      qref.custom "T"() %30 adj : !qref.bit
      qref.custom "T"() %31 adj : !qref.bit
      qref.custom "CNOT"() %31, %30 : !qref.bit, !qref.bit
      qref.custom "T"() %30 adj : !qref.bit
      qref.custom "T"() %29 : !qref.bit
      qref.custom "CNOT"() %31, %29 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %29, %30 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %29 : !qref.bit
      qref.custom "T"() %29 : !qref.bit
      qref.custom "CNOT"() %28, %29 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %29, %27 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %27, %28 : !qref.bit, !qref.bit
      qref.custom "T"() %28 adj : !qref.bit
      qref.custom "T"() %29 adj : !qref.bit
      qref.custom "CNOT"() %29, %28 : !qref.bit, !qref.bit
      qref.custom "T"() %28 adj : !qref.bit
      qref.custom "T"() %27 : !qref.bit
      qref.custom "CNOT"() %29, %27 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %27, %28 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %27 : !qref.bit
      qref.custom "T"() %27 : !qref.bit
      qref.custom "CNOT"() %26, %27 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %27, %25 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %25, %26 : !qref.bit, !qref.bit
      qref.custom "T"() %26 adj : !qref.bit
      qref.custom "T"() %27 adj : !qref.bit
      qref.custom "CNOT"() %27, %26 : !qref.bit, !qref.bit
      qref.custom "T"() %26 adj : !qref.bit
      qref.custom "T"() %25 : !qref.bit
      qref.custom "CNOT"() %27, %25 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %25, %26 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %25 : !qref.bit
      qref.custom "T"() %25 : !qref.bit
      qref.custom "CNOT"() %24, %25 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %25, %23 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %23, %24 : !qref.bit, !qref.bit
      qref.custom "T"() %24 adj : !qref.bit
      qref.custom "T"() %25 adj : !qref.bit
      qref.custom "CNOT"() %25, %24 : !qref.bit, !qref.bit
      qref.custom "T"() %24 adj : !qref.bit
      qref.custom "T"() %23 : !qref.bit
      qref.custom "CNOT"() %25, %23 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %23, %24 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %23 : !qref.bit
      qref.custom "T"() %23 : !qref.bit
      qref.custom "CNOT"() %22, %23 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %23, %21 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %21, %22 : !qref.bit, !qref.bit
      qref.custom "T"() %22 adj : !qref.bit
      qref.custom "T"() %23 adj : !qref.bit
      qref.custom "CNOT"() %23, %22 : !qref.bit, !qref.bit
      qref.custom "T"() %22 adj : !qref.bit
      qref.custom "T"() %21 : !qref.bit
      qref.custom "CNOT"() %23, %21 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %21, %22 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %21 : !qref.bit
      qref.custom "T"() %21 : !qref.bit
      qref.custom "CNOT"() %20, %21 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %21, %19 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %19, %20 : !qref.bit, !qref.bit
      qref.custom "T"() %20 adj : !qref.bit
      qref.custom "T"() %21 adj : !qref.bit
      qref.custom "CNOT"() %21, %20 : !qref.bit, !qref.bit
      qref.custom "T"() %20 adj : !qref.bit
      qref.custom "T"() %19 : !qref.bit
      qref.custom "CNOT"() %21, %19 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %19, %20 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %19 : !qref.bit
      qref.custom "T"() %19 : !qref.bit
      qref.custom "CNOT"() %18, %19 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %19, %17 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %17, %18 : !qref.bit, !qref.bit
      qref.custom "T"() %18 adj : !qref.bit
      qref.custom "T"() %19 adj : !qref.bit
      qref.custom "CNOT"() %19, %18 : !qref.bit, !qref.bit
      qref.custom "T"() %18 adj : !qref.bit
      qref.custom "T"() %17 : !qref.bit
      qref.custom "CNOT"() %19, %17 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %17, %18 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %17 : !qref.bit
      qref.custom "T"() %17 : !qref.bit
      qref.custom "CNOT"() %16, %17 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %17, %15 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %15, %16 : !qref.bit, !qref.bit
      qref.custom "T"() %16 adj : !qref.bit
      qref.custom "T"() %17 adj : !qref.bit
      qref.custom "CNOT"() %17, %16 : !qref.bit, !qref.bit
      qref.custom "T"() %16 adj : !qref.bit
      qref.custom "T"() %15 : !qref.bit
      qref.custom "CNOT"() %17, %15 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %15, %16 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %15 : !qref.bit
      qref.custom "T"() %15 : !qref.bit
      qref.custom "CNOT"() %14, %15 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %15, %13 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %13, %14 : !qref.bit, !qref.bit
      qref.custom "T"() %14 adj : !qref.bit
      qref.custom "T"() %15 adj : !qref.bit
      qref.custom "CNOT"() %15, %14 : !qref.bit, !qref.bit
      qref.custom "T"() %14 adj : !qref.bit
      qref.custom "T"() %13 : !qref.bit
      qref.custom "CNOT"() %15, %13 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %13, %14 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %13 : !qref.bit
      qref.custom "T"() %13 : !qref.bit
      qref.custom "CNOT"() %12, %13 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %13, %11 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %11, %12 : !qref.bit, !qref.bit
      qref.custom "T"() %12 adj : !qref.bit
      qref.custom "T"() %13 adj : !qref.bit
      qref.custom "CNOT"() %13, %12 : !qref.bit, !qref.bit
      qref.custom "T"() %12 adj : !qref.bit
      qref.custom "T"() %11 : !qref.bit
      qref.custom "CNOT"() %13, %11 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %11, %12 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %11 : !qref.bit
      qref.custom "T"() %11 : !qref.bit
      qref.custom "CNOT"() %10, %11 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %11, %9 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %9, %10 : !qref.bit, !qref.bit
      qref.custom "T"() %10 adj : !qref.bit
      qref.custom "T"() %11 adj : !qref.bit
      qref.custom "CNOT"() %11, %10 : !qref.bit, !qref.bit
      qref.custom "T"() %10 adj : !qref.bit
      qref.custom "T"() %9 : !qref.bit
      qref.custom "CNOT"() %11, %9 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %9, %10 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %9 : !qref.bit
      qref.custom "T"() %9 : !qref.bit
      qref.custom "CNOT"() %8, %9 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %9, %6 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %6, %8 : !qref.bit, !qref.bit
      qref.custom "T"() %8 adj : !qref.bit
      qref.custom "T"() %9 adj : !qref.bit
      qref.custom "CNOT"() %9, %8 : !qref.bit, !qref.bit
      qref.custom "T"() %8 adj : !qref.bit
      qref.custom "T"() %6 : !qref.bit
      qref.custom "CNOT"() %9, %6 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %6, %8 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %8, %9 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %9 : !qref.bit
      qref.custom "T"() %9 : !qref.bit
      qref.custom "CNOT"() %10, %11 : !qref.bit, !qref.bit
      qref.custom "T"() %10 : !qref.bit
      qref.custom "Hadamard"() %11 : !qref.bit
      qref.custom "T"() %11 : !qref.bit
      qref.custom "CNOT"() %12, %13 : !qref.bit, !qref.bit
      qref.custom "T"() %12 : !qref.bit
      qref.custom "Hadamard"() %13 : !qref.bit
      qref.custom "T"() %13 : !qref.bit
      qref.custom "CNOT"() %14, %15 : !qref.bit, !qref.bit
      qref.custom "T"() %14 : !qref.bit
      qref.custom "Hadamard"() %15 : !qref.bit
      qref.custom "T"() %15 : !qref.bit
      qref.custom "CNOT"() %16, %17 : !qref.bit, !qref.bit
      qref.custom "T"() %16 : !qref.bit
      qref.custom "Hadamard"() %17 : !qref.bit
      qref.custom "T"() %17 : !qref.bit
      qref.custom "CNOT"() %18, %19 : !qref.bit, !qref.bit
      qref.custom "T"() %18 : !qref.bit
      qref.custom "Hadamard"() %19 : !qref.bit
      qref.custom "T"() %19 : !qref.bit
      qref.custom "CNOT"() %20, %21 : !qref.bit, !qref.bit
      qref.custom "T"() %20 : !qref.bit
      qref.custom "Hadamard"() %21 : !qref.bit
      qref.custom "T"() %21 : !qref.bit
      qref.custom "CNOT"() %22, %23 : !qref.bit, !qref.bit
      qref.custom "T"() %22 : !qref.bit
      qref.custom "Hadamard"() %23 : !qref.bit
      qref.custom "T"() %23 : !qref.bit
      qref.custom "CNOT"() %24, %25 : !qref.bit, !qref.bit
      qref.custom "T"() %24 : !qref.bit
      qref.custom "Hadamard"() %25 : !qref.bit
      qref.custom "T"() %25 : !qref.bit
      qref.custom "CNOT"() %26, %27 : !qref.bit, !qref.bit
      qref.custom "T"() %26 : !qref.bit
      qref.custom "Hadamard"() %27 : !qref.bit
      qref.custom "T"() %27 : !qref.bit
      qref.custom "CNOT"() %28, %29 : !qref.bit, !qref.bit
      qref.custom "T"() %28 : !qref.bit
      qref.custom "Hadamard"() %29 : !qref.bit
      qref.custom "T"() %29 : !qref.bit
      qref.custom "CNOT"() %30, %31 : !qref.bit, !qref.bit
      qref.custom "T"() %30 : !qref.bit
      qref.custom "Hadamard"() %31 : !qref.bit
      qref.custom "T"() %31 : !qref.bit
      qref.custom "CNOT"() %32, %33 : !qref.bit, !qref.bit
      qref.custom "T"() %32 : !qref.bit
      qref.custom "Hadamard"() %33 : !qref.bit
      qref.custom "T"() %33 : !qref.bit
      qref.custom "CNOT"() %34, %35 : !qref.bit, !qref.bit
      qref.custom "T"() %34 : !qref.bit
      qref.custom "Hadamard"() %35 : !qref.bit
      qref.custom "T"() %35 : !qref.bit
      qref.custom "CNOT"() %36, %37 : !qref.bit, !qref.bit
      qref.custom "T"() %36 : !qref.bit
      qref.custom "Hadamard"() %37 : !qref.bit
      qref.custom "T"() %37 : !qref.bit
      qref.custom "CNOT"() %38, %39 : !qref.bit, !qref.bit
      qref.custom "T"() %38 : !qref.bit
      qref.custom "Hadamard"() %39 : !qref.bit
      qref.custom "T"() %39 : !qref.bit
      qref.custom "CNOT"() %40, %41 : !qref.bit, !qref.bit
      qref.custom "T"() %40 : !qref.bit
      qref.custom "Hadamard"() %41 : !qref.bit
      qref.custom "T"() %41 : !qref.bit
      qref.custom "CNOT"() %42, %43 : !qref.bit, !qref.bit
      qref.custom "T"() %42 : !qref.bit
      qref.custom "Hadamard"() %43 : !qref.bit
      qref.custom "T"() %43 : !qref.bit
      qref.custom "CNOT"() %44, %45 : !qref.bit, !qref.bit
      qref.custom "T"() %44 : !qref.bit
      qref.custom "Hadamard"() %45 : !qref.bit
      qref.custom "T"() %45 : !qref.bit
      qref.custom "CNOT"() %46, %47 : !qref.bit, !qref.bit
      qref.custom "T"() %46 : !qref.bit
      qref.custom "Hadamard"() %47 : !qref.bit
      qref.custom "T"() %47 : !qref.bit
      qref.custom "CNOT"() %48, %49 : !qref.bit, !qref.bit
      qref.custom "T"() %48 : !qref.bit
      qref.custom "Hadamard"() %49 : !qref.bit
      qref.custom "T"() %49 : !qref.bit
      qref.custom "CNOT"() %50, %51 : !qref.bit, !qref.bit
      qref.custom "T"() %50 : !qref.bit
      qref.custom "Hadamard"() %51 : !qref.bit
      qref.custom "T"() %51 : !qref.bit
      qref.custom "CNOT"() %52, %53 : !qref.bit, !qref.bit
      qref.custom "T"() %52 : !qref.bit
      qref.custom "Hadamard"() %53 : !qref.bit
      qref.custom "T"() %53 : !qref.bit
      qref.custom "CNOT"() %54, %55 : !qref.bit, !qref.bit
      qref.custom "T"() %54 : !qref.bit
      qref.custom "Hadamard"() %55 : !qref.bit
      qref.custom "T"() %55 : !qref.bit
      qref.custom "CNOT"() %56, %57 : !qref.bit, !qref.bit
      qref.custom "T"() %56 : !qref.bit
      qref.custom "Hadamard"() %57 : !qref.bit
      qref.custom "T"() %57 : !qref.bit
      qref.custom "CNOT"() %58, %59 : !qref.bit, !qref.bit
      qref.custom "T"() %58 : !qref.bit
      qref.custom "Hadamard"() %59 : !qref.bit
      qref.custom "T"() %59 : !qref.bit
      qref.custom "CNOT"() %60, %61 : !qref.bit, !qref.bit
      qref.custom "T"() %60 : !qref.bit
      qref.custom "Hadamard"() %61 : !qref.bit
      qref.custom "T"() %61 : !qref.bit
      qref.custom "CNOT"() %62, %63 : !qref.bit, !qref.bit
      qref.custom "T"() %62 : !qref.bit
      qref.custom "Hadamard"() %63 : !qref.bit
      qref.custom "T"() %63 : !qref.bit
      qref.custom "CNOT"() %64, %65 : !qref.bit, !qref.bit
      qref.custom "T"() %64 : !qref.bit
      qref.custom "Hadamard"() %65 : !qref.bit
      qref.custom "T"() %65 : !qref.bit
      qref.custom "CNOT"() %66, %67 : !qref.bit, !qref.bit
      qref.custom "T"() %66 : !qref.bit
      qref.custom "Hadamard"() %67 : !qref.bit
      qref.custom "T"() %67 : !qref.bit
      qref.custom "CNOT"() %68, %69 : !qref.bit, !qref.bit
      qref.custom "T"() %68 : !qref.bit
      qref.custom "Hadamard"() %69 : !qref.bit
      qref.custom "T"() %69 : !qref.bit
      qref.custom "CNOT"() %70, %71 : !qref.bit, !qref.bit
      qref.custom "T"() %70 : !qref.bit
      qref.custom "Hadamard"() %71 : !qref.bit
      qref.custom "T"() %71 : !qref.bit
      qref.custom "CNOT"() %72, %73 : !qref.bit, !qref.bit
      qref.custom "T"() %72 : !qref.bit
      qref.custom "Hadamard"() %73 : !qref.bit
      qref.custom "T"() %73 : !qref.bit
      qref.custom "CNOT"() %74, %75 : !qref.bit, !qref.bit
      qref.custom "T"() %74 : !qref.bit
      qref.custom "Hadamard"() %75 : !qref.bit
      qref.custom "T"() %75 : !qref.bit
      qref.custom "CNOT"() %76, %77 : !qref.bit, !qref.bit
      qref.custom "T"() %76 : !qref.bit
      qref.custom "Hadamard"() %77 : !qref.bit
      qref.custom "T"() %77 : !qref.bit
      qref.custom "CNOT"() %78, %79 : !qref.bit, !qref.bit
      qref.custom "T"() %78 : !qref.bit
      qref.custom "Hadamard"() %79 : !qref.bit
      qref.custom "T"() %79 : !qref.bit
      qref.custom "CNOT"() %80, %81 : !qref.bit, !qref.bit
      qref.custom "T"() %80 : !qref.bit
      qref.custom "Hadamard"() %81 : !qref.bit
      qref.custom "T"() %81 : !qref.bit
      qref.custom "CNOT"() %82, %83 : !qref.bit, !qref.bit
      qref.custom "T"() %82 : !qref.bit
      qref.custom "Hadamard"() %83 : !qref.bit
      qref.custom "T"() %83 : !qref.bit
      qref.custom "CNOT"() %84, %85 : !qref.bit, !qref.bit
      qref.custom "T"() %84 : !qref.bit
      qref.custom "Hadamard"() %85 : !qref.bit
      qref.custom "T"() %85 : !qref.bit
      qref.custom "CNOT"() %86, %87 : !qref.bit, !qref.bit
      qref.custom "T"() %86 : !qref.bit
      qref.custom "Hadamard"() %87 : !qref.bit
      qref.custom "T"() %87 : !qref.bit
      qref.custom "CNOT"() %88, %89 : !qref.bit, !qref.bit
      qref.custom "T"() %88 : !qref.bit
      qref.custom "Hadamard"() %89 : !qref.bit
      qref.custom "T"() %89 : !qref.bit
      qref.custom "CNOT"() %90, %91 : !qref.bit, !qref.bit
      qref.custom "T"() %90 : !qref.bit
      qref.custom "Hadamard"() %91 : !qref.bit
      qref.custom "T"() %91 : !qref.bit
      qref.custom "CNOT"() %92, %93 : !qref.bit, !qref.bit
      qref.custom "T"() %92 : !qref.bit
      qref.custom "Hadamard"() %93 : !qref.bit
      qref.custom "T"() %93 : !qref.bit
      qref.custom "CNOT"() %94, %95 : !qref.bit, !qref.bit
      qref.custom "T"() %94 : !qref.bit
      qref.custom "Hadamard"() %95 : !qref.bit
      qref.custom "T"() %95 : !qref.bit
      qref.custom "CNOT"() %96, %97 : !qref.bit, !qref.bit
      qref.custom "T"() %96 : !qref.bit
      qref.custom "Hadamard"() %97 : !qref.bit
      qref.custom "T"() %97 : !qref.bit
      qref.custom "CNOT"() %98, %99 : !qref.bit, !qref.bit
      qref.custom "T"() %98 : !qref.bit
      qref.custom "Hadamard"() %99 : !qref.bit
      qref.custom "T"() %99 : !qref.bit
      qref.custom "CNOT"() %100, %101 : !qref.bit, !qref.bit
      qref.custom "T"() %100 : !qref.bit
      qref.custom "Hadamard"() %101 : !qref.bit
      qref.custom "T"() %101 : !qref.bit
      qref.custom "CNOT"() %102, %103 : !qref.bit, !qref.bit
      qref.custom "T"() %102 : !qref.bit
      qref.custom "Hadamard"() %103 : !qref.bit
      qref.custom "T"() %103 : !qref.bit
      qref.custom "CNOT"() %104, %105 : !qref.bit, !qref.bit
      qref.custom "T"() %104 : !qref.bit
      qref.custom "Hadamard"() %105 : !qref.bit
      qref.custom "T"() %105 : !qref.bit
      qref.custom "CNOT"() %106, %107 : !qref.bit, !qref.bit
      qref.custom "T"() %106 : !qref.bit
      qref.custom "Hadamard"() %107 : !qref.bit
      qref.custom "T"() %107 : !qref.bit
      qref.custom "CNOT"() %108, %109 : !qref.bit, !qref.bit
      qref.custom "T"() %108 : !qref.bit
      qref.custom "Hadamard"() %109 : !qref.bit
      qref.custom "T"() %109 : !qref.bit
      qref.custom "CNOT"() %110, %111 : !qref.bit, !qref.bit
      qref.custom "T"() %110 : !qref.bit
      qref.custom "Hadamard"() %111 : !qref.bit
      qref.custom "T"() %111 : !qref.bit
      qref.custom "CNOT"() %112, %113 : !qref.bit, !qref.bit
      qref.custom "T"() %112 : !qref.bit
      qref.custom "Hadamard"() %113 : !qref.bit
      qref.custom "T"() %113 : !qref.bit
      qref.custom "CNOT"() %114, %115 : !qref.bit, !qref.bit
      qref.custom "T"() %114 : !qref.bit
      qref.custom "Hadamard"() %115 : !qref.bit
      qref.custom "T"() %115 : !qref.bit
      qref.custom "CNOT"() %116, %117 : !qref.bit, !qref.bit
      qref.custom "T"() %116 : !qref.bit
      qref.custom "Hadamard"() %117 : !qref.bit
      qref.custom "T"() %117 : !qref.bit
      qref.custom "CNOT"() %118, %119 : !qref.bit, !qref.bit
      qref.custom "T"() %118 : !qref.bit
      qref.custom "Hadamard"() %119 : !qref.bit
      qref.custom "T"() %119 : !qref.bit
      qref.custom "CNOT"() %120, %121 : !qref.bit, !qref.bit
      qref.custom "T"() %120 : !qref.bit
      qref.custom "Hadamard"() %121 : !qref.bit
      qref.custom "T"() %121 : !qref.bit
      qref.custom "CNOT"() %122, %123 : !qref.bit, !qref.bit
      qref.custom "T"() %122 : !qref.bit
      qref.custom "Hadamard"() %123 : !qref.bit
      qref.custom "T"() %123 : !qref.bit
      qref.custom "CNOT"() %124, %125 : !qref.bit, !qref.bit
      qref.custom "T"() %124 : !qref.bit
      qref.custom "Hadamard"() %125 : !qref.bit
      qref.custom "T"() %125 : !qref.bit
      qref.custom "CNOT"() %126, %127 : !qref.bit, !qref.bit
      qref.custom "T"() %126 : !qref.bit
      qref.custom "Hadamard"() %127 : !qref.bit
      qref.custom "T"() %127 : !qref.bit
      qref.custom "CNOT"() %128, %129 : !qref.bit, !qref.bit
      qref.custom "T"() %128 : !qref.bit
      qref.custom "T"() %129 : !qref.bit
      qref.custom "CNOT"() %128, %129 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %129, %127 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %127, %128 : !qref.bit, !qref.bit
      qref.custom "T"() %128 adj : !qref.bit
      qref.custom "T"() %129 adj : !qref.bit
      qref.custom "CNOT"() %129, %128 : !qref.bit, !qref.bit
      qref.custom "T"() %128 adj : !qref.bit
      qref.custom "T"() %127 : !qref.bit
      qref.custom "CNOT"() %129, %127 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %127, %128 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %127 : !qref.bit
      qref.custom "T"() %127 : !qref.bit
      qref.custom "CNOT"() %126, %127 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %127, %125 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %125, %126 : !qref.bit, !qref.bit
      qref.custom "T"() %126 adj : !qref.bit
      qref.custom "T"() %127 adj : !qref.bit
      qref.custom "CNOT"() %127, %126 : !qref.bit, !qref.bit
      qref.custom "T"() %126 adj : !qref.bit
      qref.custom "T"() %125 : !qref.bit
      qref.custom "CNOT"() %127, %125 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %125, %126 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %125 : !qref.bit
      qref.custom "T"() %125 : !qref.bit
      qref.custom "CNOT"() %124, %125 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %125, %123 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %123, %124 : !qref.bit, !qref.bit
      qref.custom "T"() %124 adj : !qref.bit
      qref.custom "T"() %125 adj : !qref.bit
      qref.custom "CNOT"() %125, %124 : !qref.bit, !qref.bit
      qref.custom "T"() %124 adj : !qref.bit
      qref.custom "T"() %123 : !qref.bit
      qref.custom "CNOT"() %125, %123 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %123, %124 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %123 : !qref.bit
      qref.custom "T"() %123 : !qref.bit
      qref.custom "CNOT"() %122, %123 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %123, %121 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %121, %122 : !qref.bit, !qref.bit
      qref.custom "T"() %122 adj : !qref.bit
      qref.custom "T"() %123 adj : !qref.bit
      qref.custom "CNOT"() %123, %122 : !qref.bit, !qref.bit
      qref.custom "T"() %122 adj : !qref.bit
      qref.custom "T"() %121 : !qref.bit
      qref.custom "CNOT"() %123, %121 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %121, %122 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %121 : !qref.bit
      qref.custom "T"() %121 : !qref.bit
      qref.custom "CNOT"() %120, %121 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %121, %119 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %119, %120 : !qref.bit, !qref.bit
      qref.custom "T"() %120 adj : !qref.bit
      qref.custom "T"() %121 adj : !qref.bit
      qref.custom "CNOT"() %121, %120 : !qref.bit, !qref.bit
      qref.custom "T"() %120 adj : !qref.bit
      qref.custom "T"() %119 : !qref.bit
      qref.custom "CNOT"() %121, %119 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %119, %120 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %119 : !qref.bit
      qref.custom "T"() %119 : !qref.bit
      qref.custom "CNOT"() %118, %119 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %119, %117 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %117, %118 : !qref.bit, !qref.bit
      qref.custom "T"() %118 adj : !qref.bit
      qref.custom "T"() %119 adj : !qref.bit
      qref.custom "CNOT"() %119, %118 : !qref.bit, !qref.bit
      qref.custom "T"() %118 adj : !qref.bit
      qref.custom "T"() %117 : !qref.bit
      qref.custom "CNOT"() %119, %117 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %117, %118 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %117 : !qref.bit
      qref.custom "T"() %117 : !qref.bit
      qref.custom "CNOT"() %116, %117 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %117, %115 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %115, %116 : !qref.bit, !qref.bit
      qref.custom "T"() %116 adj : !qref.bit
      qref.custom "T"() %117 adj : !qref.bit
      qref.custom "CNOT"() %117, %116 : !qref.bit, !qref.bit
      qref.custom "T"() %116 adj : !qref.bit
      qref.custom "T"() %115 : !qref.bit
      qref.custom "CNOT"() %117, %115 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %115, %116 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %115 : !qref.bit
      qref.custom "T"() %115 : !qref.bit
      qref.custom "CNOT"() %114, %115 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %115, %113 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %113, %114 : !qref.bit, !qref.bit
      qref.custom "T"() %114 adj : !qref.bit
      qref.custom "T"() %115 adj : !qref.bit
      qref.custom "CNOT"() %115, %114 : !qref.bit, !qref.bit
      qref.custom "T"() %114 adj : !qref.bit
      qref.custom "T"() %113 : !qref.bit
      qref.custom "CNOT"() %115, %113 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %113, %114 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %113 : !qref.bit
      qref.custom "T"() %113 : !qref.bit
      qref.custom "CNOT"() %112, %113 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %113, %111 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %111, %112 : !qref.bit, !qref.bit
      qref.custom "T"() %112 adj : !qref.bit
      qref.custom "T"() %113 adj : !qref.bit
      qref.custom "CNOT"() %113, %112 : !qref.bit, !qref.bit
      qref.custom "T"() %112 adj : !qref.bit
      qref.custom "T"() %111 : !qref.bit
      qref.custom "CNOT"() %113, %111 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %111, %112 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %111 : !qref.bit
      qref.custom "T"() %111 : !qref.bit
      qref.custom "CNOT"() %110, %111 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %111, %109 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %109, %110 : !qref.bit, !qref.bit
      qref.custom "T"() %110 adj : !qref.bit
      qref.custom "T"() %111 adj : !qref.bit
      qref.custom "CNOT"() %111, %110 : !qref.bit, !qref.bit
      qref.custom "T"() %110 adj : !qref.bit
      qref.custom "T"() %109 : !qref.bit
      qref.custom "CNOT"() %111, %109 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %109, %110 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %109 : !qref.bit
      qref.custom "T"() %109 : !qref.bit
      qref.custom "CNOT"() %108, %109 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %109, %107 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %107, %108 : !qref.bit, !qref.bit
      qref.custom "T"() %108 adj : !qref.bit
      qref.custom "T"() %109 adj : !qref.bit
      qref.custom "CNOT"() %109, %108 : !qref.bit, !qref.bit
      qref.custom "T"() %108 adj : !qref.bit
      qref.custom "T"() %107 : !qref.bit
      qref.custom "CNOT"() %109, %107 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %107, %108 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %107 : !qref.bit
      qref.custom "T"() %107 : !qref.bit
      qref.custom "CNOT"() %106, %107 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %107, %105 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %105, %106 : !qref.bit, !qref.bit
      qref.custom "T"() %106 adj : !qref.bit
      qref.custom "T"() %107 adj : !qref.bit
      qref.custom "CNOT"() %107, %106 : !qref.bit, !qref.bit
      qref.custom "T"() %106 adj : !qref.bit
      qref.custom "T"() %105 : !qref.bit
      qref.custom "CNOT"() %107, %105 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %105, %106 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %105 : !qref.bit
      qref.custom "T"() %105 : !qref.bit
      qref.custom "CNOT"() %104, %105 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %105, %103 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %103, %104 : !qref.bit, !qref.bit
      qref.custom "T"() %104 adj : !qref.bit
      qref.custom "T"() %105 adj : !qref.bit
      qref.custom "CNOT"() %105, %104 : !qref.bit, !qref.bit
      qref.custom "T"() %104 adj : !qref.bit
      qref.custom "T"() %103 : !qref.bit
      qref.custom "CNOT"() %105, %103 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %103, %104 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %103 : !qref.bit
      qref.custom "T"() %103 : !qref.bit
      qref.custom "CNOT"() %102, %103 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %103, %101 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %101, %102 : !qref.bit, !qref.bit
      qref.custom "T"() %102 adj : !qref.bit
      qref.custom "T"() %103 adj : !qref.bit
      qref.custom "CNOT"() %103, %102 : !qref.bit, !qref.bit
      qref.custom "T"() %102 adj : !qref.bit
      qref.custom "T"() %101 : !qref.bit
      qref.custom "CNOT"() %103, %101 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %101, %102 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %101 : !qref.bit
      qref.custom "T"() %101 : !qref.bit
      qref.custom "CNOT"() %100, %101 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %101, %99 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %99, %100 : !qref.bit, !qref.bit
      qref.custom "T"() %100 adj : !qref.bit
      qref.custom "T"() %101 adj : !qref.bit
      qref.custom "CNOT"() %101, %100 : !qref.bit, !qref.bit
      qref.custom "T"() %100 adj : !qref.bit
      qref.custom "T"() %99 : !qref.bit
      qref.custom "CNOT"() %101, %99 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %99, %100 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %99 : !qref.bit
      qref.custom "T"() %99 : !qref.bit
      qref.custom "CNOT"() %98, %99 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %99, %97 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %97, %98 : !qref.bit, !qref.bit
      qref.custom "T"() %98 adj : !qref.bit
      qref.custom "T"() %99 adj : !qref.bit
      qref.custom "CNOT"() %99, %98 : !qref.bit, !qref.bit
      qref.custom "T"() %98 adj : !qref.bit
      qref.custom "T"() %97 : !qref.bit
      qref.custom "CNOT"() %99, %97 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %97, %98 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %97 : !qref.bit
      qref.custom "T"() %97 : !qref.bit
      qref.custom "CNOT"() %96, %97 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %97, %95 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %95, %96 : !qref.bit, !qref.bit
      qref.custom "T"() %96 adj : !qref.bit
      qref.custom "T"() %97 adj : !qref.bit
      qref.custom "CNOT"() %97, %96 : !qref.bit, !qref.bit
      qref.custom "T"() %96 adj : !qref.bit
      qref.custom "T"() %95 : !qref.bit
      qref.custom "CNOT"() %97, %95 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %95, %96 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %95 : !qref.bit
      qref.custom "T"() %95 : !qref.bit
      qref.custom "CNOT"() %94, %95 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %95, %93 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %93, %94 : !qref.bit, !qref.bit
      qref.custom "T"() %94 adj : !qref.bit
      qref.custom "T"() %95 adj : !qref.bit
      qref.custom "CNOT"() %95, %94 : !qref.bit, !qref.bit
      qref.custom "T"() %94 adj : !qref.bit
      qref.custom "T"() %93 : !qref.bit
      qref.custom "CNOT"() %95, %93 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %93, %94 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %93 : !qref.bit
      qref.custom "T"() %93 : !qref.bit
      qref.custom "CNOT"() %92, %93 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %93, %91 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %91, %92 : !qref.bit, !qref.bit
      qref.custom "T"() %92 adj : !qref.bit
      qref.custom "T"() %93 adj : !qref.bit
      qref.custom "CNOT"() %93, %92 : !qref.bit, !qref.bit
      qref.custom "T"() %92 adj : !qref.bit
      qref.custom "T"() %91 : !qref.bit
      qref.custom "CNOT"() %93, %91 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %91, %92 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %91 : !qref.bit
      qref.custom "T"() %91 : !qref.bit
      qref.custom "CNOT"() %90, %91 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %91, %89 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %89, %90 : !qref.bit, !qref.bit
      qref.custom "T"() %90 adj : !qref.bit
      qref.custom "T"() %91 adj : !qref.bit
      qref.custom "CNOT"() %91, %90 : !qref.bit, !qref.bit
      qref.custom "T"() %90 adj : !qref.bit
      qref.custom "T"() %89 : !qref.bit
      qref.custom "CNOT"() %91, %89 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %89, %90 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %89 : !qref.bit
      qref.custom "T"() %89 : !qref.bit
      qref.custom "CNOT"() %88, %89 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %89, %87 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %87, %88 : !qref.bit, !qref.bit
      qref.custom "T"() %88 adj : !qref.bit
      qref.custom "T"() %89 adj : !qref.bit
      qref.custom "CNOT"() %89, %88 : !qref.bit, !qref.bit
      qref.custom "T"() %88 adj : !qref.bit
      qref.custom "T"() %87 : !qref.bit
      qref.custom "CNOT"() %89, %87 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %87, %88 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %87 : !qref.bit
      qref.custom "T"() %87 : !qref.bit
      qref.custom "CNOT"() %86, %87 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %87, %85 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %85, %86 : !qref.bit, !qref.bit
      qref.custom "T"() %86 adj : !qref.bit
      qref.custom "T"() %87 adj : !qref.bit
      qref.custom "CNOT"() %87, %86 : !qref.bit, !qref.bit
      qref.custom "T"() %86 adj : !qref.bit
      qref.custom "T"() %85 : !qref.bit
      qref.custom "CNOT"() %87, %85 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %85, %86 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %85 : !qref.bit
      qref.custom "T"() %85 : !qref.bit
      qref.custom "CNOT"() %84, %85 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %85, %83 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %83, %84 : !qref.bit, !qref.bit
      qref.custom "T"() %84 adj : !qref.bit
      qref.custom "T"() %85 adj : !qref.bit
      qref.custom "CNOT"() %85, %84 : !qref.bit, !qref.bit
      qref.custom "T"() %84 adj : !qref.bit
      qref.custom "T"() %83 : !qref.bit
      qref.custom "CNOT"() %85, %83 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %83, %84 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %83 : !qref.bit
      qref.custom "T"() %83 : !qref.bit
      qref.custom "CNOT"() %82, %83 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %83, %81 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %81, %82 : !qref.bit, !qref.bit
      qref.custom "T"() %82 adj : !qref.bit
      qref.custom "T"() %83 adj : !qref.bit
      qref.custom "CNOT"() %83, %82 : !qref.bit, !qref.bit
      qref.custom "T"() %82 adj : !qref.bit
      qref.custom "T"() %81 : !qref.bit
      qref.custom "CNOT"() %83, %81 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %81, %82 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %81 : !qref.bit
      qref.custom "T"() %81 : !qref.bit
      qref.custom "CNOT"() %80, %81 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %81, %79 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %79, %80 : !qref.bit, !qref.bit
      qref.custom "T"() %80 adj : !qref.bit
      qref.custom "T"() %81 adj : !qref.bit
      qref.custom "CNOT"() %81, %80 : !qref.bit, !qref.bit
      qref.custom "T"() %80 adj : !qref.bit
      qref.custom "T"() %79 : !qref.bit
      qref.custom "CNOT"() %81, %79 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %79, %80 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %79 : !qref.bit
      qref.custom "T"() %79 : !qref.bit
      qref.custom "CNOT"() %78, %79 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %79, %77 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %77, %78 : !qref.bit, !qref.bit
      qref.custom "T"() %78 adj : !qref.bit
      qref.custom "T"() %79 adj : !qref.bit
      qref.custom "CNOT"() %79, %78 : !qref.bit, !qref.bit
      qref.custom "T"() %78 adj : !qref.bit
      qref.custom "T"() %77 : !qref.bit
      qref.custom "CNOT"() %79, %77 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %77, %78 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %77 : !qref.bit
      qref.custom "T"() %77 : !qref.bit
      qref.custom "CNOT"() %76, %77 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %77, %75 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %75, %76 : !qref.bit, !qref.bit
      qref.custom "T"() %76 adj : !qref.bit
      qref.custom "T"() %77 adj : !qref.bit
      qref.custom "CNOT"() %77, %76 : !qref.bit, !qref.bit
      qref.custom "T"() %76 adj : !qref.bit
      qref.custom "T"() %75 : !qref.bit
      qref.custom "CNOT"() %77, %75 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %75, %76 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %75 : !qref.bit
      qref.custom "T"() %75 : !qref.bit
      qref.custom "CNOT"() %74, %75 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %75, %73 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %73, %74 : !qref.bit, !qref.bit
      qref.custom "T"() %74 adj : !qref.bit
      qref.custom "T"() %75 adj : !qref.bit
      qref.custom "CNOT"() %75, %74 : !qref.bit, !qref.bit
      qref.custom "T"() %74 adj : !qref.bit
      qref.custom "T"() %73 : !qref.bit
      qref.custom "CNOT"() %75, %73 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %73, %74 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %73 : !qref.bit
      qref.custom "T"() %73 : !qref.bit
      qref.custom "CNOT"() %72, %73 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %73, %71 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %71, %72 : !qref.bit, !qref.bit
      qref.custom "T"() %72 adj : !qref.bit
      qref.custom "T"() %73 adj : !qref.bit
      qref.custom "CNOT"() %73, %72 : !qref.bit, !qref.bit
      qref.custom "T"() %72 adj : !qref.bit
      qref.custom "T"() %71 : !qref.bit
      qref.custom "CNOT"() %73, %71 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %71, %72 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %71 : !qref.bit
      qref.custom "T"() %71 : !qref.bit
      qref.custom "CNOT"() %70, %71 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %71, %69 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %69, %70 : !qref.bit, !qref.bit
      qref.custom "T"() %70 adj : !qref.bit
      qref.custom "T"() %71 adj : !qref.bit
      qref.custom "CNOT"() %71, %70 : !qref.bit, !qref.bit
      qref.custom "T"() %70 adj : !qref.bit
      qref.custom "T"() %69 : !qref.bit
      qref.custom "CNOT"() %71, %69 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %69, %70 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %69 : !qref.bit
      qref.custom "T"() %69 : !qref.bit
      qref.custom "CNOT"() %68, %69 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %69, %67 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %67, %68 : !qref.bit, !qref.bit
      qref.custom "T"() %68 adj : !qref.bit
      qref.custom "T"() %69 adj : !qref.bit
      qref.custom "CNOT"() %69, %68 : !qref.bit, !qref.bit
      qref.custom "T"() %68 adj : !qref.bit
      qref.custom "T"() %67 : !qref.bit
      qref.custom "CNOT"() %69, %67 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %67, %68 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %67 : !qref.bit
      qref.custom "T"() %67 : !qref.bit
      qref.custom "CNOT"() %66, %67 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %67, %65 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %65, %66 : !qref.bit, !qref.bit
      qref.custom "T"() %66 adj : !qref.bit
      qref.custom "T"() %67 adj : !qref.bit
      qref.custom "CNOT"() %67, %66 : !qref.bit, !qref.bit
      qref.custom "T"() %66 adj : !qref.bit
      qref.custom "T"() %65 : !qref.bit
      qref.custom "CNOT"() %67, %65 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %65, %66 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %65 : !qref.bit
      qref.custom "T"() %65 : !qref.bit
      qref.custom "CNOT"() %64, %65 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %65, %63 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %63, %64 : !qref.bit, !qref.bit
      qref.custom "T"() %64 adj : !qref.bit
      qref.custom "T"() %65 adj : !qref.bit
      qref.custom "CNOT"() %65, %64 : !qref.bit, !qref.bit
      qref.custom "T"() %64 adj : !qref.bit
      qref.custom "T"() %63 : !qref.bit
      qref.custom "CNOT"() %65, %63 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %63, %64 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %63 : !qref.bit
      qref.custom "T"() %63 : !qref.bit
      qref.custom "CNOT"() %62, %63 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %63, %61 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %61, %62 : !qref.bit, !qref.bit
      qref.custom "T"() %62 adj : !qref.bit
      qref.custom "T"() %63 adj : !qref.bit
      qref.custom "CNOT"() %63, %62 : !qref.bit, !qref.bit
      qref.custom "T"() %62 adj : !qref.bit
      qref.custom "T"() %61 : !qref.bit
      qref.custom "CNOT"() %63, %61 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %61, %62 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %61 : !qref.bit
      qref.custom "T"() %61 : !qref.bit
      qref.custom "CNOT"() %60, %61 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %61, %59 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %59, %60 : !qref.bit, !qref.bit
      qref.custom "T"() %60 adj : !qref.bit
      qref.custom "T"() %61 adj : !qref.bit
      qref.custom "CNOT"() %61, %60 : !qref.bit, !qref.bit
      qref.custom "T"() %60 adj : !qref.bit
      qref.custom "T"() %59 : !qref.bit
      qref.custom "CNOT"() %61, %59 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %59, %60 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %59 : !qref.bit
      qref.custom "T"() %59 : !qref.bit
      qref.custom "CNOT"() %58, %59 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %59, %57 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %57, %58 : !qref.bit, !qref.bit
      qref.custom "T"() %58 adj : !qref.bit
      qref.custom "T"() %59 adj : !qref.bit
      qref.custom "CNOT"() %59, %58 : !qref.bit, !qref.bit
      qref.custom "T"() %58 adj : !qref.bit
      qref.custom "T"() %57 : !qref.bit
      qref.custom "CNOT"() %59, %57 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %57, %58 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %57 : !qref.bit
      qref.custom "T"() %57 : !qref.bit
      qref.custom "CNOT"() %56, %57 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %57, %55 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %55, %56 : !qref.bit, !qref.bit
      qref.custom "T"() %56 adj : !qref.bit
      qref.custom "T"() %57 adj : !qref.bit
      qref.custom "CNOT"() %57, %56 : !qref.bit, !qref.bit
      qref.custom "T"() %56 adj : !qref.bit
      qref.custom "T"() %55 : !qref.bit
      qref.custom "CNOT"() %57, %55 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %55, %56 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %55 : !qref.bit
      qref.custom "T"() %55 : !qref.bit
      qref.custom "CNOT"() %54, %55 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %55, %53 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %53, %54 : !qref.bit, !qref.bit
      qref.custom "T"() %54 adj : !qref.bit
      qref.custom "T"() %55 adj : !qref.bit
      qref.custom "CNOT"() %55, %54 : !qref.bit, !qref.bit
      qref.custom "T"() %54 adj : !qref.bit
      qref.custom "T"() %53 : !qref.bit
      qref.custom "CNOT"() %55, %53 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %53, %54 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %53 : !qref.bit
      qref.custom "T"() %53 : !qref.bit
      qref.custom "CNOT"() %52, %53 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %53, %51 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %51, %52 : !qref.bit, !qref.bit
      qref.custom "T"() %52 adj : !qref.bit
      qref.custom "T"() %53 adj : !qref.bit
      qref.custom "CNOT"() %53, %52 : !qref.bit, !qref.bit
      qref.custom "T"() %52 adj : !qref.bit
      qref.custom "T"() %51 : !qref.bit
      qref.custom "CNOT"() %53, %51 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %51, %52 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %51 : !qref.bit
      qref.custom "T"() %51 : !qref.bit
      qref.custom "CNOT"() %50, %51 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %51, %49 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %49, %50 : !qref.bit, !qref.bit
      qref.custom "T"() %50 adj : !qref.bit
      qref.custom "T"() %51 adj : !qref.bit
      qref.custom "CNOT"() %51, %50 : !qref.bit, !qref.bit
      qref.custom "T"() %50 adj : !qref.bit
      qref.custom "T"() %49 : !qref.bit
      qref.custom "CNOT"() %51, %49 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %49, %50 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %49 : !qref.bit
      qref.custom "T"() %49 : !qref.bit
      qref.custom "CNOT"() %48, %49 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %49, %47 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %47, %48 : !qref.bit, !qref.bit
      qref.custom "T"() %48 adj : !qref.bit
      qref.custom "T"() %49 adj : !qref.bit
      qref.custom "CNOT"() %49, %48 : !qref.bit, !qref.bit
      qref.custom "T"() %48 adj : !qref.bit
      qref.custom "T"() %47 : !qref.bit
      qref.custom "CNOT"() %49, %47 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %47, %48 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %47 : !qref.bit
      qref.custom "T"() %47 : !qref.bit
      qref.custom "CNOT"() %46, %47 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %47, %45 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %45, %46 : !qref.bit, !qref.bit
      qref.custom "T"() %46 adj : !qref.bit
      qref.custom "T"() %47 adj : !qref.bit
      qref.custom "CNOT"() %47, %46 : !qref.bit, !qref.bit
      qref.custom "T"() %46 adj : !qref.bit
      qref.custom "T"() %45 : !qref.bit
      qref.custom "CNOT"() %47, %45 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %45, %46 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %45 : !qref.bit
      qref.custom "T"() %45 : !qref.bit
      qref.custom "CNOT"() %44, %45 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %45, %43 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %43, %44 : !qref.bit, !qref.bit
      qref.custom "T"() %44 adj : !qref.bit
      qref.custom "T"() %45 adj : !qref.bit
      qref.custom "CNOT"() %45, %44 : !qref.bit, !qref.bit
      qref.custom "T"() %44 adj : !qref.bit
      qref.custom "T"() %43 : !qref.bit
      qref.custom "CNOT"() %45, %43 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %43, %44 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %43 : !qref.bit
      qref.custom "T"() %43 : !qref.bit
      qref.custom "CNOT"() %42, %43 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %43, %41 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %41, %42 : !qref.bit, !qref.bit
      qref.custom "T"() %42 adj : !qref.bit
      qref.custom "T"() %43 adj : !qref.bit
      qref.custom "CNOT"() %43, %42 : !qref.bit, !qref.bit
      qref.custom "T"() %42 adj : !qref.bit
      qref.custom "T"() %41 : !qref.bit
      qref.custom "CNOT"() %43, %41 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %41, %42 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %41 : !qref.bit
      qref.custom "T"() %41 : !qref.bit
      qref.custom "CNOT"() %40, %41 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %41, %39 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %39, %40 : !qref.bit, !qref.bit
      qref.custom "T"() %40 adj : !qref.bit
      qref.custom "T"() %41 adj : !qref.bit
      qref.custom "CNOT"() %41, %40 : !qref.bit, !qref.bit
      qref.custom "T"() %40 adj : !qref.bit
      qref.custom "T"() %39 : !qref.bit
      qref.custom "CNOT"() %41, %39 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %39, %40 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %39 : !qref.bit
      qref.custom "T"() %39 : !qref.bit
      qref.custom "CNOT"() %38, %39 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %39, %37 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %37, %38 : !qref.bit, !qref.bit
      qref.custom "T"() %38 adj : !qref.bit
      qref.custom "T"() %39 adj : !qref.bit
      qref.custom "CNOT"() %39, %38 : !qref.bit, !qref.bit
      qref.custom "T"() %38 adj : !qref.bit
      qref.custom "T"() %37 : !qref.bit
      qref.custom "CNOT"() %39, %37 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %37, %38 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %37 : !qref.bit
      qref.custom "T"() %37 : !qref.bit
      qref.custom "CNOT"() %36, %37 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %37, %35 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %35, %36 : !qref.bit, !qref.bit
      qref.custom "T"() %36 adj : !qref.bit
      qref.custom "T"() %37 adj : !qref.bit
      qref.custom "CNOT"() %37, %36 : !qref.bit, !qref.bit
      qref.custom "T"() %36 adj : !qref.bit
      qref.custom "T"() %35 : !qref.bit
      qref.custom "CNOT"() %37, %35 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %35, %36 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %35 : !qref.bit
      qref.custom "T"() %35 : !qref.bit
      qref.custom "CNOT"() %34, %35 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %35, %33 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %33, %34 : !qref.bit, !qref.bit
      qref.custom "T"() %34 adj : !qref.bit
      qref.custom "T"() %35 adj : !qref.bit
      qref.custom "CNOT"() %35, %34 : !qref.bit, !qref.bit
      qref.custom "T"() %34 adj : !qref.bit
      qref.custom "T"() %33 : !qref.bit
      qref.custom "CNOT"() %35, %33 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %33, %34 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %33 : !qref.bit
      qref.custom "T"() %33 : !qref.bit
      qref.custom "CNOT"() %32, %33 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %33, %31 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %31, %32 : !qref.bit, !qref.bit
      qref.custom "T"() %32 adj : !qref.bit
      qref.custom "T"() %33 adj : !qref.bit
      qref.custom "CNOT"() %33, %32 : !qref.bit, !qref.bit
      qref.custom "T"() %32 adj : !qref.bit
      qref.custom "T"() %31 : !qref.bit
      qref.custom "CNOT"() %33, %31 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %31, %32 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %31 : !qref.bit
      qref.custom "T"() %31 : !qref.bit
      qref.custom "CNOT"() %30, %31 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %31, %29 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %29, %30 : !qref.bit, !qref.bit
      qref.custom "T"() %30 adj : !qref.bit
      qref.custom "T"() %31 adj : !qref.bit
      qref.custom "CNOT"() %31, %30 : !qref.bit, !qref.bit
      qref.custom "T"() %30 adj : !qref.bit
      qref.custom "T"() %29 : !qref.bit
      qref.custom "CNOT"() %31, %29 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %29, %30 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %29 : !qref.bit
      qref.custom "T"() %29 : !qref.bit
      qref.custom "CNOT"() %28, %29 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %29, %27 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %27, %28 : !qref.bit, !qref.bit
      qref.custom "T"() %28 adj : !qref.bit
      qref.custom "T"() %29 adj : !qref.bit
      qref.custom "CNOT"() %29, %28 : !qref.bit, !qref.bit
      qref.custom "T"() %28 adj : !qref.bit
      qref.custom "T"() %27 : !qref.bit
      qref.custom "CNOT"() %29, %27 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %27, %28 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %27 : !qref.bit
      qref.custom "T"() %27 : !qref.bit
      qref.custom "CNOT"() %26, %27 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %27, %25 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %25, %26 : !qref.bit, !qref.bit
      qref.custom "T"() %26 adj : !qref.bit
      qref.custom "T"() %27 adj : !qref.bit
      qref.custom "CNOT"() %27, %26 : !qref.bit, !qref.bit
      qref.custom "T"() %26 adj : !qref.bit
      qref.custom "T"() %25 : !qref.bit
      qref.custom "CNOT"() %27, %25 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %25, %26 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %25 : !qref.bit
      qref.custom "T"() %25 : !qref.bit
      qref.custom "CNOT"() %24, %25 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %25, %23 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %23, %24 : !qref.bit, !qref.bit
      qref.custom "T"() %24 adj : !qref.bit
      qref.custom "T"() %25 adj : !qref.bit
      qref.custom "CNOT"() %25, %24 : !qref.bit, !qref.bit
      qref.custom "T"() %24 adj : !qref.bit
      qref.custom "T"() %23 : !qref.bit
      qref.custom "CNOT"() %25, %23 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %23, %24 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %23 : !qref.bit
      qref.custom "T"() %23 : !qref.bit
      qref.custom "CNOT"() %22, %23 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %23, %21 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %21, %22 : !qref.bit, !qref.bit
      qref.custom "T"() %22 adj : !qref.bit
      qref.custom "T"() %23 adj : !qref.bit
      qref.custom "CNOT"() %23, %22 : !qref.bit, !qref.bit
      qref.custom "T"() %22 adj : !qref.bit
      qref.custom "T"() %21 : !qref.bit
      qref.custom "CNOT"() %23, %21 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %21, %22 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %21 : !qref.bit
      qref.custom "T"() %21 : !qref.bit
      qref.custom "CNOT"() %20, %21 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %21, %19 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %19, %20 : !qref.bit, !qref.bit
      qref.custom "T"() %20 adj : !qref.bit
      qref.custom "T"() %21 adj : !qref.bit
      qref.custom "CNOT"() %21, %20 : !qref.bit, !qref.bit
      qref.custom "T"() %20 adj : !qref.bit
      qref.custom "T"() %19 : !qref.bit
      qref.custom "CNOT"() %21, %19 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %19, %20 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %19 : !qref.bit
      qref.custom "T"() %19 : !qref.bit
      qref.custom "CNOT"() %18, %19 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %19, %17 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %17, %18 : !qref.bit, !qref.bit
      qref.custom "T"() %18 adj : !qref.bit
      qref.custom "T"() %19 adj : !qref.bit
      qref.custom "CNOT"() %19, %18 : !qref.bit, !qref.bit
      qref.custom "T"() %18 adj : !qref.bit
      qref.custom "T"() %17 : !qref.bit
      qref.custom "CNOT"() %19, %17 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %17, %18 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %17 : !qref.bit
      qref.custom "T"() %17 : !qref.bit
      qref.custom "CNOT"() %16, %17 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %17, %15 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %15, %16 : !qref.bit, !qref.bit
      qref.custom "T"() %16 adj : !qref.bit
      qref.custom "T"() %17 adj : !qref.bit
      qref.custom "CNOT"() %17, %16 : !qref.bit, !qref.bit
      qref.custom "T"() %16 adj : !qref.bit
      qref.custom "T"() %15 : !qref.bit
      qref.custom "CNOT"() %17, %15 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %15, %16 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %15 : !qref.bit
      qref.custom "T"() %15 : !qref.bit
      qref.custom "CNOT"() %14, %15 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %15, %13 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %13, %14 : !qref.bit, !qref.bit
      qref.custom "T"() %14 adj : !qref.bit
      qref.custom "T"() %15 adj : !qref.bit
      qref.custom "CNOT"() %15, %14 : !qref.bit, !qref.bit
      qref.custom "T"() %14 adj : !qref.bit
      qref.custom "T"() %13 : !qref.bit
      qref.custom "CNOT"() %15, %13 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %13, %14 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %13 : !qref.bit
      qref.custom "T"() %13 : !qref.bit
      qref.custom "CNOT"() %12, %13 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %13, %11 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %11, %12 : !qref.bit, !qref.bit
      qref.custom "T"() %12 adj : !qref.bit
      qref.custom "T"() %13 adj : !qref.bit
      qref.custom "CNOT"() %13, %12 : !qref.bit, !qref.bit
      qref.custom "T"() %12 adj : !qref.bit
      qref.custom "T"() %11 : !qref.bit
      qref.custom "CNOT"() %13, %11 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %11, %12 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %11 : !qref.bit
      qref.custom "T"() %11 : !qref.bit
      qref.custom "CNOT"() %10, %11 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %11, %9 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %9, %10 : !qref.bit, !qref.bit
      qref.custom "T"() %10 adj : !qref.bit
      qref.custom "T"() %11 adj : !qref.bit
      qref.custom "CNOT"() %11, %10 : !qref.bit, !qref.bit
      qref.custom "T"() %10 adj : !qref.bit
      qref.custom "T"() %9 : !qref.bit
      qref.custom "CNOT"() %11, %9 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %9, %10 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %9 : !qref.bit
      qref.custom "CNOT"() %128, %129 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %129 : !qref.bit
      qref.custom "T"() %129 : !qref.bit
      qref.custom "CNOT"() %1, %130 : !qref.bit, !qref.bit
      qref.custom "T"() %1 : !qref.bit
      qref.custom "T"() %130 : !qref.bit
      qref.custom "CNOT"() %1, %130 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %130, %129 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %129, %1 : !qref.bit, !qref.bit
      qref.custom "T"() %1 adj : !qref.bit
      qref.custom "T"() %130 adj : !qref.bit
      qref.custom "CNOT"() %130, %1 : !qref.bit, !qref.bit
      qref.custom "T"() %1 adj : !qref.bit
      qref.custom "T"() %129 : !qref.bit
      qref.custom "CNOT"() %130, %129 : !qref.bit, !qref.bit
      qref.custom "CNOT"() %129, %1 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %129 : !qref.bit
      qref.custom "CNOT"() %1, %130 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %1 : !qref.bit
      qref.custom "PauliX"() %1 : !qref.bit
      qref.custom "Hadamard"() %130 : !qref.bit
      qref.custom "PauliX"() %130 : !qref.bit
      qref.custom "Hadamard"() %128 : !qref.bit
      qref.custom "PauliX"() %128 : !qref.bit
      qref.custom "CNOT"() %126, %127 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %126 : !qref.bit
      qref.custom "PauliX"() %126 : !qref.bit
      qref.custom "CNOT"() %124, %125 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %124 : !qref.bit
      qref.custom "PauliX"() %124 : !qref.bit
      qref.custom "CNOT"() %122, %123 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %122 : !qref.bit
      qref.custom "PauliX"() %122 : !qref.bit
      qref.custom "CNOT"() %120, %121 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %120 : !qref.bit
      qref.custom "PauliX"() %120 : !qref.bit
      qref.custom "CNOT"() %118, %119 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %118 : !qref.bit
      qref.custom "PauliX"() %118 : !qref.bit
      qref.custom "CNOT"() %116, %117 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %116 : !qref.bit
      qref.custom "PauliX"() %116 : !qref.bit
      qref.custom "CNOT"() %114, %115 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %114 : !qref.bit
      qref.custom "PauliX"() %114 : !qref.bit
      qref.custom "CNOT"() %112, %113 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %112 : !qref.bit
      qref.custom "PauliX"() %112 : !qref.bit
      qref.custom "CNOT"() %110, %111 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %110 : !qref.bit
      qref.custom "PauliX"() %110 : !qref.bit
      qref.custom "CNOT"() %108, %109 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %108 : !qref.bit
      qref.custom "PauliX"() %108 : !qref.bit
      qref.custom "CNOT"() %106, %107 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %106 : !qref.bit
      qref.custom "PauliX"() %106 : !qref.bit
      qref.custom "CNOT"() %104, %105 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %104 : !qref.bit
      qref.custom "PauliX"() %104 : !qref.bit
      qref.custom "CNOT"() %102, %103 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %102 : !qref.bit
      qref.custom "PauliX"() %102 : !qref.bit
      qref.custom "CNOT"() %100, %101 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %100 : !qref.bit
      qref.custom "PauliX"() %100 : !qref.bit
      qref.custom "CNOT"() %98, %99 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %98 : !qref.bit
      qref.custom "PauliX"() %98 : !qref.bit
      qref.custom "CNOT"() %96, %97 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %96 : !qref.bit
      qref.custom "PauliX"() %96 : !qref.bit
      qref.custom "CNOT"() %94, %95 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %94 : !qref.bit
      qref.custom "PauliX"() %94 : !qref.bit
      qref.custom "CNOT"() %92, %93 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %92 : !qref.bit
      qref.custom "PauliX"() %92 : !qref.bit
      qref.custom "CNOT"() %90, %91 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %90 : !qref.bit
      qref.custom "PauliX"() %90 : !qref.bit
      qref.custom "CNOT"() %88, %89 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %88 : !qref.bit
      qref.custom "PauliX"() %88 : !qref.bit
      qref.custom "CNOT"() %86, %87 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %86 : !qref.bit
      qref.custom "PauliX"() %86 : !qref.bit
      qref.custom "CNOT"() %84, %85 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %84 : !qref.bit
      qref.custom "PauliX"() %84 : !qref.bit
      qref.custom "CNOT"() %82, %83 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %82 : !qref.bit
      qref.custom "PauliX"() %82 : !qref.bit
      qref.custom "CNOT"() %80, %81 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %80 : !qref.bit
      qref.custom "PauliX"() %80 : !qref.bit
      qref.custom "CNOT"() %78, %79 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %78 : !qref.bit
      qref.custom "PauliX"() %78 : !qref.bit
      qref.custom "CNOT"() %76, %77 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %76 : !qref.bit
      qref.custom "PauliX"() %76 : !qref.bit
      qref.custom "CNOT"() %74, %75 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %74 : !qref.bit
      qref.custom "PauliX"() %74 : !qref.bit
      qref.custom "CNOT"() %72, %73 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %72 : !qref.bit
      qref.custom "PauliX"() %72 : !qref.bit
      qref.custom "CNOT"() %70, %71 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %70 : !qref.bit
      qref.custom "PauliX"() %70 : !qref.bit
      qref.custom "CNOT"() %68, %69 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %68 : !qref.bit
      qref.custom "PauliX"() %68 : !qref.bit
      qref.custom "CNOT"() %66, %67 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %66 : !qref.bit
      qref.custom "PauliX"() %66 : !qref.bit
      qref.custom "CNOT"() %64, %65 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %64 : !qref.bit
      qref.custom "PauliX"() %64 : !qref.bit
      qref.custom "CNOT"() %62, %63 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %62 : !qref.bit
      qref.custom "PauliX"() %62 : !qref.bit
      qref.custom "CNOT"() %60, %61 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %60 : !qref.bit
      qref.custom "PauliX"() %60 : !qref.bit
      qref.custom "CNOT"() %58, %59 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %58 : !qref.bit
      qref.custom "PauliX"() %58 : !qref.bit
      qref.custom "CNOT"() %56, %57 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %56 : !qref.bit
      qref.custom "PauliX"() %56 : !qref.bit
      qref.custom "CNOT"() %54, %55 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %54 : !qref.bit
      qref.custom "PauliX"() %54 : !qref.bit
      qref.custom "CNOT"() %52, %53 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %52 : !qref.bit
      qref.custom "PauliX"() %52 : !qref.bit
      qref.custom "CNOT"() %50, %51 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %50 : !qref.bit
      qref.custom "PauliX"() %50 : !qref.bit
      qref.custom "CNOT"() %48, %49 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %48 : !qref.bit
      qref.custom "PauliX"() %48 : !qref.bit
      qref.custom "CNOT"() %46, %47 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %46 : !qref.bit
      qref.custom "PauliX"() %46 : !qref.bit
      qref.custom "CNOT"() %44, %45 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %44 : !qref.bit
      qref.custom "PauliX"() %44 : !qref.bit
      qref.custom "CNOT"() %42, %43 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %42 : !qref.bit
      qref.custom "PauliX"() %42 : !qref.bit
      qref.custom "CNOT"() %40, %41 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %40 : !qref.bit
      qref.custom "PauliX"() %40 : !qref.bit
      qref.custom "CNOT"() %38, %39 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %38 : !qref.bit
      qref.custom "PauliX"() %38 : !qref.bit
      qref.custom "CNOT"() %36, %37 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %36 : !qref.bit
      qref.custom "PauliX"() %36 : !qref.bit
      qref.custom "CNOT"() %34, %35 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %34 : !qref.bit
      qref.custom "PauliX"() %34 : !qref.bit
      qref.custom "CNOT"() %32, %33 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %32 : !qref.bit
      qref.custom "PauliX"() %32 : !qref.bit
      qref.custom "CNOT"() %30, %31 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %30 : !qref.bit
      qref.custom "PauliX"() %30 : !qref.bit
      qref.custom "CNOT"() %28, %29 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %28 : !qref.bit
      qref.custom "PauliX"() %28 : !qref.bit
      qref.custom "CNOT"() %26, %27 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %26 : !qref.bit
      qref.custom "PauliX"() %26 : !qref.bit
      qref.custom "CNOT"() %24, %25 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %24 : !qref.bit
      qref.custom "PauliX"() %24 : !qref.bit
      qref.custom "CNOT"() %22, %23 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %22 : !qref.bit
      qref.custom "PauliX"() %22 : !qref.bit
      qref.custom "CNOT"() %20, %21 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %20 : !qref.bit
      qref.custom "PauliX"() %20 : !qref.bit
      qref.custom "CNOT"() %18, %19 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %18 : !qref.bit
      qref.custom "PauliX"() %18 : !qref.bit
      qref.custom "CNOT"() %16, %17 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %16 : !qref.bit
      qref.custom "PauliX"() %16 : !qref.bit
      qref.custom "CNOT"() %14, %15 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %14 : !qref.bit
      qref.custom "PauliX"() %14 : !qref.bit
      qref.custom "CNOT"() %12, %13 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %12 : !qref.bit
      qref.custom "PauliX"() %12 : !qref.bit
      qref.custom "CNOT"() %10, %11 : !qref.bit, !qref.bit
      qref.custom "Hadamard"() %10 : !qref.bit
      qref.custom "PauliX"() %10 : !qref.bit
      qref.custom "Hadamard"() %8 : !qref.bit
      qref.custom "PauliX"() %8 : !qref.bit
      qref.custom "Hadamard"() %6 : !qref.bit
      qref.custom "Hadamard"() %6 : !qref.bit
      qref.custom "Hadamard"() %6 : !qref.bit
      qref.custom "PauliX"() %6 : !qref.bit
      scf.yield
    } 
    qref.dealloc %0 : !qref.reg<129>
    quantum.device_release
    return
  }
}