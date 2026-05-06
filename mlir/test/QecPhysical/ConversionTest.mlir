// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt %s \
// RUN:   --finalize-memref-to-llvm \
// RUN:   --convert-qecp-to-llvm \
// RUN:   --reconcile-unrealized-casts \
// RUN:   --split-input-file -verify-diagnostics \
// RUN: | FileCheck %s

module {
    memref.global "private" constant @__constant_8xi32 : memref<8xi32> = dense<[3, 3, 4, 4, 0, 1, 1, 2]> {alignment = 64 : i64}
    memref.global "private" constant @__constant_6xi32 : memref<6xi32> = dense<[0, 1, 3, 4, 6, 8]> {alignment = 64 : i64}
    // CHECK-LABEL: llvm.func @test_assemble_tanner_static()
    func.func @test_assemble_tanner_static() -> !qecp.tanner_graph<8, 6, i32> {
        // CHECK: [[rowidx0:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        // CHECK: [[rowidx1:%.+]] = llvm.insertvalue {{%.+}}, [[rowidx0:%.+]][0] : !llvm.struct
        // CHECK: [[rowidx2:%.+]] = llvm.insertvalue {{%.+}}, [[rowidx1:%.+]][1] : !llvm.struct
        // CHECK: [[rowidx3:%.+]] = llvm.insertvalue {{%.+}}, [[rowidx2:%.+]][2] : !llvm.struct
        // CHECK: [[rowidx4:%.+]] = llvm.insertvalue {{%.+}}, [[rowidx3:%.+]][3, 0] : !llvm.struct
        // CHECK: [[rowidx5:%.+]] = llvm.insertvalue {{%.+}}, [[rowidx4:%.+]][4, 0] : !llvm.struct
        %row_idx = memref.get_global @__constant_8xi32 : memref<8xi32>

        // CHECK: [[colptr0:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        // CHECK: [[colptr1:%.+]] = llvm.insertvalue {{%.+}}, [[colptr0:%.+]][0] : !llvm.struct
        // CHECK: [[colptr2:%.+]] = llvm.insertvalue {{%.+}}, [[colptr1:%.+]][1] : !llvm.struct
        // CHECK: [[colptr3:%.+]] = llvm.insertvalue {{%.+}}, [[colptr2:%.+]][2] : !llvm.struct
        // CHECK: [[colptr4:%.+]] = llvm.insertvalue {{%.+}}, [[colptr3:%.+]][3, 0] : !llvm.struct
        // CHECK: [[colptr5:%.+]] = llvm.insertvalue {{%.+}}, [[colptr4:%.+]][4, 0] : !llvm.struct
        %col_ptr = memref.get_global @__constant_6xi32 : memref<6xi32>

        //      CHECK: [[tg0:%.+]] = llvm.mlir.undef : !llvm.struct<"TannerGraph", (
        // CHECK-SAME:     struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>,
        // CHECK-SAME:     struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>
        //      CHECK: [[tg1:%.+]] = llvm.insertvalue [[rowidx5]], [[tg0]][0] : !llvm.struct<"TannerGraph"
        //      CHECK: [[tg2:%.+]] = llvm.insertvalue [[colptr5]], [[tg1]][1] : !llvm.struct<"TannerGraph"
        %tanner = qecp.assemble_tanner %row_idx, %col_ptr : memref<8xi32>, memref<6xi32> -> !qecp.tanner_graph<8, 6, i32>

        // llvm.return [[tg2]] : !llvm.struct<"TannerGraph"
        func.return %tanner : !qecp.tanner_graph<8, 6, i32>
    }
}

// -----

module {
    // CHECK-LABEL: llvm.func @test_assemble_tanner_dynamic(
    func.func @test_assemble_tanner_dynamic(
        %row_idx : memref<8xi32>, %col_ptr : memref<6xi32>
    ) -> !qecp.tanner_graph<8, 6, i32> {
        // CHECK: [[colptr0:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        // CHECK: [[colptr1:%.+]] = llvm.insertvalue {{%.+}}, [[colptr0:%.+]][0] : !llvm.struct
        // CHECK: [[colptr2:%.+]] = llvm.insertvalue {{%.+}}, [[colptr1:%.+]][1] : !llvm.struct
        // CHECK: [[colptr3:%.+]] = llvm.insertvalue {{%.+}}, [[colptr2:%.+]][2] : !llvm.struct
        // CHECK: [[colptr4:%.+]] = llvm.insertvalue {{%.+}}, [[colptr3:%.+]][3, 0] : !llvm.struct
        // CHECK: [[colptr5:%.+]] = llvm.insertvalue {{%.+}}, [[colptr4:%.+]][4, 0] : !llvm.struct

        // CHECK: [[rowidx0:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        // CHECK: [[rowidx1:%.+]] = llvm.insertvalue {{%.+}}, [[rowidx0:%.+]][0] : !llvm.struct
        // CHECK: [[rowidx2:%.+]] = llvm.insertvalue {{%.+}}, [[rowidx1:%.+]][1] : !llvm.struct
        // CHECK: [[rowidx3:%.+]] = llvm.insertvalue {{%.+}}, [[rowidx2:%.+]][2] : !llvm.struct
        // CHECK: [[rowidx4:%.+]] = llvm.insertvalue {{%.+}}, [[rowidx3:%.+]][3, 0] : !llvm.struct
        // CHECK: [[rowidx5:%.+]] = llvm.insertvalue {{%.+}}, [[rowidx4:%.+]][4, 0] : !llvm.struct

        //      CHECK: [[tg0:%.+]] = llvm.mlir.undef : !llvm.struct<"TannerGraph", (
        // CHECK-SAME:     struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>,
        // CHECK-SAME:     struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>
        //      CHECK: [[tg1:%.+]] = llvm.insertvalue [[rowidx5]], [[tg0]][0] : !llvm.struct<"TannerGraph"
        //      CHECK: [[tg2:%.+]] = llvm.insertvalue [[colptr5]], [[tg1]][1] : !llvm.struct<"TannerGraph"
        %tanner = qecp.assemble_tanner %row_idx, %col_ptr : memref<8xi32>, memref<6xi32> -> !qecp.tanner_graph<8, 6, i32>

        // CHECK: llvm.return [[tg2]] : !llvm.struct<"TannerGraph"
        func.return %tanner : !qecp.tanner_graph<8, 6, i32>
    }
}

// -----

module {
    //       CHECK: llvm.func @__catalyst__qecp__decode_physical_measurements(!llvm.ptr, !llvm.ptr)
    // CHECK-LABEL: llvm.func @test_decode_physical_meas(
    func.func @test_decode_physical_meas(%pmeas : memref<7xi1>) -> memref<1xi1> {
        %lmeas = memref.alloc() : memref<1xi1>

        // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64) : i64
        // CHECK: [[lmeas_ptr:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        // CHECK: [[c1:%.+]] = llvm.mlir.constant(1 : i64) : i64
        // CHECK: [[pmeas_ptr:%.+]] = llvm.alloca [[c1]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        // CHECK: llvm.store {{%.+}}, [[pmeas_ptr]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
        // CHECK: llvm.store [[lmeas_val:%.+]], [[lmeas_ptr]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
        // CHECK: llvm.call @__catalyst__qecp__decode_physical_measurements([[pmeas_ptr]], [[lmeas_ptr]]) : (!llvm.ptr, !llvm.ptr) -> ()
        qecp.decode_physical_meas %pmeas in(%lmeas : memref<1xi1>) : memref<7xi1>

        // CHECK: llvm.return [[lmeas_val]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        func.return %lmeas : memref<1xi1>
    }
}
