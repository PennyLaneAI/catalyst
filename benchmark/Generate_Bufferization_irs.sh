../mlir/build/bin/quantum-opt 3_1_QuantumCompilationPass.mlir --one-shot-bufferize=dialect-filter=memref -o O32/3_2_1_one-shot-bufferize.mlir
../mlir/build/bin/quantum-opt O32/3_2_1_one-shot-bufferize.mlir --inline -o O32/3_2_2_inline.mlir
../mlir/build/bin/quantum-opt O32/3_2_2_inline.mlir --gradient-bufferize -o O32/3_2_3_gradient-bufferize.mlir
../mlir/build/bin/quantum-opt O32/3_2_3_gradient-bufferize.mlir --scf-bufferize -o O32/3_2_4_scf-bufferize.mlir
../mlir/build/bin/quantum-opt O32/3_2_4_scf-bufferize.mlir --convert-tensor-to-linalg -o O32/3_2_5_convert-tensor-to-linalg.mlir
../mlir/build/bin/quantum-opt O32/3_2_5_convert-tensor-to-linalg.mlir --convert-elementwise-to-linalg -o O32/3_2_6_convert-elementwise-to-linalg.mlir
../mlir/build/bin/quantum-opt O32/3_2_6_convert-elementwise-to-linalg.mlir --arith-bufferize -o O32/3_2_7_arith-bufferize.mlir
../mlir/build/bin/quantum-opt O32/3_2_7_arith-bufferize.mlir --empty-tensor-to-alloc-tensor -o O32/3_2_8_empty-tensor-to-alloc-tensor.mlir
../mlir/build/bin/quantum-opt O32/3_2_8_empty-tensor-to-alloc-tensor.mlir --bufferization-bufferize --tensor-bufferize -o O32/3_2_9_bufferization-bufferize.mlir
../mlir/build/bin/quantum-opt O32/3_2_9_bufferization-bufferize.mlir --catalyst-bufferize -o O32/3_2_10_catalyst-bufferize.mlir
../mlir/build/bin/quantum-opt O32/3_2_10_catalyst-bufferize.mlir --linalg-bufferize --tensor-bufferize -o O32/3_2_11_linalg-bufferize.mlir
../mlir/build/bin/quantum-opt O32/3_2_11_linalg-bufferize.mlir --quantum-bufferize -o O32/3_2_12_quantum-bufferize.mlir
../mlir/build/bin/quantum-opt O32/3_2_12_quantum-bufferize.mlir --func-bufferize -o O32/3_2_13_func-bufferize.mlir
../mlir/build/bin/quantum-opt O32/3_2_13_func-bufferize.mlir --finalizing-bufferize --buffer-hoisting --buffer-loop-hoisting --buffer-deallocation -o O32/3_2_14_finalizing-bufferize.mlir
../mlir/build/bin/quantum-opt O32/3_2_14_finalizing-bufferize.mlir --convert-arraylist-to-memref -o O32/3_2_15_convert-arraylist-to-memref.mlir
../mlir/build/bin/quantum-opt O32/3_2_15_convert-arraylist-to-memref.mlir --convert-bufferization-to-memref -o O32/3_2_16_convert-bufferization-to-memref.mlir
../mlir/build/bin/quantum-opt O32/3_2_16_convert-bufferization-to-memref.mlir --canonicalize -o O32/3_2_17_canonicalize.mlir
../mlir/build/bin/quantum-opt O32/3_2_17_canonicalize.mlir --cp-global-memref -o O32/3_2_18_cp-global-memref.mlir
