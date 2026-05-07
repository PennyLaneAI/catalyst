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
// RUN:   --one-shot-bufferize \
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
    // CHECK: llvm.func @__catalyst__qecp__lut_decoder(!llvm.ptr, !llvm.ptr, !llvm.ptr) 
    memref.global "private" constant @__constant_3xi1 : memref<3xi1> = dense<[1, 1, 0]> {alignment = 64 : i64}
    memref.global "private" constant @__constant_8xi32 : memref<8xi32> = dense<[3, 3, 4, 4, 0, 1, 1, 2]> {alignment = 64 : i64}
    memref.global "private" constant @__constant_6xi32 : memref<6xi32> = dense<[0, 1, 3, 4, 6, 8]> {alignment = 64 : i64}
    // CHECK-LABEL: llvm.func @test_psudo_qec_cycle()
    func.func @test_psudo_qec_cycle() {
        %row_idx = memref.get_global @__constant_8xi32 : memref<8xi32>
        %col_ptr = memref.get_global @__constant_6xi32 : memref<6xi32>

        //      CHECK: [[TG0:%.+]] = llvm.mlir.undef : !llvm.struct<"TannerGraph", (
        // CHECK-SAME:     struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>,
        // CHECK-SAME:     struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>
        %tanner = qecp.assemble_tanner %row_idx, %col_ptr : memref<8xi32>, memref<6xi32> -> !qecp.tanner_graph<8, 6, i32>

        %esm = memref.get_global @__constant_3xi1 : memref<3xi1>

        %err_buf = memref.alloc() : memref<2xindex>
        // CHECK-NOT: builtin.unrealized_conversion_cast
        // CHECK: llvm.store {{.+}}, [[tanner:%.+]] : !llvm.struct<"TannerGraph",
        // CHECK: llvm.call @__catalyst__qecp__lut_decoder([[tanner:%.+]], {{.+}}, {{.+}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
        qecp.decode_esm_css(%tanner : !qecp.tanner_graph<8, 6, i32>) %esm in(%err_buf : memref<2xindex>) : memref<3xi1>
        memref.dealloc %err_buf : memref<2xindex>

        func.return
    }

}

// -----

module {
    // CHECK: llvm.func @__catalyst__qecp__lut_decoder(!llvm.ptr, !llvm.ptr, !llvm.ptr)
    // CHECK: llvm.mlir.global private constant @__constant_3xi1(dense<[false, true, false]> : tensor<3xi1>)
    // CHECK: llvm.mlir.global private constant @__constant_11xi32
    // CHECK: llvm.mlir.global private constant @__constant_24xi32
    // CHECK-LABEL: llvm.func @test_tanner_decode_integration()
    func.func @test_tanner_decode_integration() {
        %row_idx = arith.constant dense<[7, 7, 8, 7, 8, 9, 7, 9, 8, 8, 9, 9, 0, 1, 2, 3, 1, 2, 4, 5, 2, 3, 5, 6]> : tensor<24xi32>
        %col_ptr = arith.constant dense<[0, 1, 3, 6, 8, 9, 11, 12, 16, 20, 24]> : tensor<11xi32>
        // CHECK: [[tanner:%.+]] = llvm.mlir.undef : !llvm.struct<"TannerGraph"
        %tanner = qecp.assemble_tanner %row_idx, %col_ptr : tensor<24xi32>, tensor<11xi32> -> !qecp.tanner_graph<24, 11, i32>
        %esm = arith.constant dense<[0, 1, 0]> : tensor<3xi1>
        // CHECK: llvm.store {{.+}}, {{.+}} : !llvm.struct<"TannerGraph"
        // CHECK-NEXT: llvm.call @__catalyst__qecp__lut_decoder({{.+}}, {{.+}}, {{.+}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
        %err_idx = qecp.decode_esm_css(%tanner : !qecp.tanner_graph<24, 11, i32>) %esm : tensor<3xi1> -> tensor<2xindex>

        func.return
    }
}

// -----

module{
    // CHECK: llvm.func @__catalyst__qecp__lut_decoder(!llvm.ptr, !llvm.ptr, !llvm.ptr)
    // CHECK-LABEL: llvm.func @get_tanner_graph() -> !llvm.struct<"TannerGraph"
    func.func private @get_tanner_graph()->!qecp.tanner_graph<24, 11, i32>{
        %row_idx = arith.constant dense<[7, 7, 8, 7, 8, 9, 7, 9, 8, 8, 9, 9, 0, 1, 2, 3, 1, 2, 4, 5, 2, 3, 5, 6]> : tensor<24xi32>
        %col_ptr = arith.constant dense<[0, 1, 3, 6, 8, 9, 11, 12, 16, 20, 24]> : tensor<11xi32>

        %tanner = qecp.assemble_tanner %row_idx, %col_ptr : tensor<24xi32>, tensor<11xi32> -> !qecp.tanner_graph<24, 11, i32>
        func.return %tanner : !qecp.tanner_graph<24, 11, i32>
    }
    // CHECK-LABEL: llvm.func @test_outline_tanner_decode_integration() {
    func.func @test_outline_tanner_decode_integration(){
        // CHECK: [[tanner_value: %.+]] = llvm.call @get_tanner_graph() : () -> !llvm.struct<"TannerGraph"
        // CHECK: [[tanner_ptr: %.+]] = llvm.alloca {{.+}} x !llvm.struct<"TannerGraph"
        %tanner = func.call @get_tanner_graph():() -> !qecp.tanner_graph<24, 11, i32>
        %esm = arith.constant dense<[0, 1, 0]> : tensor<3xi1>
        // CHECK: lvm.store [[tanner_value:%.+]], [[tanner_ptr:%.+]] : !llvm.struct<"TannerGraph"
        // CHECK: llvm.call @__catalyst__qecp__lut_decoder([[tanner_ptr:%.+]], {{.+}}, {{.+}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
        %err_idx = qecp.decode_esm_css(%tanner : !qecp.tanner_graph<24, 11, i32>) %esm : tensor<3xi1> -> tensor<2xindex>

        func.return
    }
}

// -----

module{
    // CHECK: llvm.func @__catalyst__qecp__lut_decoder(!llvm.ptr, !llvm.ptr, !llvm.ptr)
    // CHECK-LABEL: llvm.func @get_tanner_graph() -> !llvm.struct<"TannerGraph"
    func.func private @get_tanner_graph()->!qecp.tanner_graph<24, 11, i32>{
        %row_idx = arith.constant dense<[7, 7, 8, 7, 8, 9, 7, 9, 8, 8, 9, 9, 0, 1, 2, 3, 1, 2, 4, 5, 2, 3, 5, 6]> : tensor<24xi32>
        %col_ptr = arith.constant dense<[0, 1, 3, 6, 8, 9, 11, 12, 16, 20, 24]> : tensor<11xi32>

        %tanner = qecp.assemble_tanner %row_idx, %col_ptr : tensor<24xi32>, tensor<11xi32> -> !qecp.tanner_graph<24, 11, i32>
        func.return %tanner : !qecp.tanner_graph<24, 11, i32>
    }
    // CHECK-LABEL: llvm.func @test_decode_with_tanner_from_arg(%arg0: !llvm.struct<"TannerGraph",
    func.func private @test_decode_with_tanner_from_arg(%tanner:!qecp.tanner_graph<24, 11, i32>){
        %esm = arith.constant dense<[0, 1, 0]> : tensor<3xi1>
        // CHECK: lvm.store [[tanner_value:%.+]], [[tanner_ptr:%.+]] : !llvm.struct<"TannerGraph"
        // CHECK: llvm.call @__catalyst__qecp__lut_decoder([[tanner_ptr:%.+]], {{.+}}, {{.+}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
        %err_idx = qecp.decode_esm_css(%tanner : !qecp.tanner_graph<24, 11, i32>) %esm : tensor<3xi1> -> tensor<2xindex>

        func.return
    }

    // CHECK-LABEL: llvm.func @test_psudo_qec() {
    func.func @test_psudo_qec(){
        // CHECK: [[tanner:%.+]] = llvm.call @get_tanner_graph() : () -> !llvm.struct<"TannerGraph",
        %tanner = func.call @get_tanner_graph():() -> !qecp.tanner_graph<24, 11, i32>
        // CHECK-NEXT: llvm.call @test_decode_with_tanner_from_arg([[tanner:%.+]]) : (!llvm.struct<"TannerGraph"
        func.call @test_decode_with_tanner_from_arg(%tanner):(!qecp.tanner_graph<24, 11, i32>)->()
        func.return
    }
}
