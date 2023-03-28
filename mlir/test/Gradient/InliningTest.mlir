// RUN: quantum-opt %s -inline | FileCheck %s

func.func private @fun(f64) -> f64

// CHECK-LABEL: @target
// CHECK-SAME: ([[arg0:%.*]]: f64)
func.func @target(%x: f64) -> tensor<?xf64> {
    // CHECK-NEXT: [[v0:%.*]] = arith.constant
    // CHECK-NEXT: [[v1:%.*]] = gradient.adjoint @fun([[arg0]]) size([[v0]])
    %0 = func.call @source(%x) : (f64) -> (tensor<?xf64>)
    // CHECK-NEXT: return [[v1]]
    return %0 : tensor<?xf64>
}

// CHECK-NOT: @source
func.func private @source(%x: f64) -> tensor<?xf64> {
    %s = arith.constant 1 : index
    %0 = gradient.adjoint @fun(%x) size(%s) : (f64) -> tensor<?xf64>
    return %0 : tensor<?xf64>
}
