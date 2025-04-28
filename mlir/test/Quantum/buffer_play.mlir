// RUN: echo "hello world!"
// ../../build/bin/quantum-opt buffer_play.mlir --one-shot-bufferize
module @set_state {
  func.func @foo(%arg0: tensor<2xcomplex<f64>>, %q0 : !quantum.bit) {
    // COM: old CHECK: quantum.set_state(%{{.*}}) %{{.*}} : (memref<2xcomplex<f64>>, !quantum.bit) -> !quantum.bit
    %0 = quantum.set_state(%arg0) %q0 : (tensor<2xcomplex<f64>>, !quantum.bit) -> !quantum.bit
    return
  }
}


// expected:
// module @set_state {
//   func.func @foo(%arg0: tensor<2xcomplex<f64>>, %arg1: !quantum.bit) {
//     %0 = bufferization.to_memref %arg0 : memref<2xcomplex<f64>>
//     %1 = bufferization.to_memref %arg0 : memref<2xcomplex<f64>>
//     %2 = quantum.set_state(%1) %arg1 : (memref<2xcomplex<f64>>, !quantum.bit) -> !quantum.bit
//     return
//   }
// }

// got: (at a5a99d1942070f2192dbe6cef6d19f9dd2b41534)
// module @set_state {
//   func.func @foo(%arg0: tensor<2xcomplex<f64>>, %arg1: !quantum.bit) {
//     %0 = bufferization.to_memref %arg0 : memref<2xcomplex<f64>>
//     %1 = quantum.set_state(%0) %arg1 : (memref<2xcomplex<f64>>, !quantum.bit) -> !quantum.bit
//     return
//   }
// }

