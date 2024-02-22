// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --emit-catalyst-py-interface --split-input-file %s | FileCheck %s

llvm.func @foo() -> () attributes {llvm.emit_c_interface} 

// Test name change
// CHECK-LABEL: @_catalyst_ciface_foo
llvm.func @_mlir_ciface_foo() {
    llvm.call @foo() : () -> ()
    llvm.return
}

// -----

llvm.func @foo() -> () attributes {llvm.emit_c_interface} 

// Test new function exists
// CHECK-LABEL: @_catalyst_pyface_foo
llvm.func @_mlir_ciface_foo() {
    llvm.call @foo() : () -> ()
    llvm.return
}

// -----

llvm.func @foo() -> () attributes {llvm.emit_c_interface} 

// Test opaque pointers when no input and no return
// CHECK-LABEL: @_catalyst_pyface_foo(%arg0: !llvm.ptr, %arg1: !llvm.ptr)
llvm.func @_mlir_ciface_foo() {
    llvm.call @foo() : () -> ()
    llvm.return
}

// -----

llvm.func @foo(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface} 

// Test that the return argument type remains unchanged in wrapper function
// CHECK-LABEL: llvm.func @_catalyst_pyface_foo(%arg0: !llvm.ptr, %arg1: !llvm.ptr)  

llvm.func @_mlir_ciface_foo(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr,  i64)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64)> 
    %4 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    %5 = llvm.extractvalue %4[0] : !llvm.struct<(ptr, ptr, i64)> 
    %6 = llvm.extractvalue %4[1] : !llvm.struct<(ptr, ptr, i64)> 
    %7 = llvm.extractvalue %4[2] : !llvm.struct<(ptr, ptr, i64)> 
    %8 = llvm.call @foo(%1, %2, %3, %5, %6, %7) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.store %8, %arg0 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    llvm.return
}

// -----

llvm.func @foo(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface} 

// Test that the return argument is passed without modification to the wrapped function
// CHECK-LABEL: llvm.call @_catalyst_ciface_foo(%arg0

llvm.func @_mlir_ciface_foo(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64)> 
    %4 = llvm.load %arg2 : !llvm.ptr ->  !llvm.struct<(ptr, ptr, i64)>
    %5 = llvm.extractvalue %4[0] : !llvm.struct<(ptr, ptr, i64)> 
    %6 = llvm.extractvalue %4[1] : !llvm.struct<(ptr, ptr, i64)> 
    %7 = llvm.extractvalue %4[2] : !llvm.struct<(ptr, ptr, i64)> 
    %8 = llvm.call @foo(%1, %2, %3, %5, %6, %7) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.store %8, %arg0 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    llvm.return
}

// -----

llvm.func @foo(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: i64) -> () attributes {llvm.emit_c_interface} 

// Test that the input argument is wrapped around a structure and a pointer
// CHECK-LABEL: llvm.func @_catalyst_pyface_foo(%arg0: !llvm.ptr, %arg1: !llvm.ptr)

llvm.func @_mlir_ciface_foo(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64)> 
    %4 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    %5 = llvm.extractvalue %4[0] : !llvm.struct<(ptr, ptr, i64)> 
    %6 = llvm.extractvalue %4[1] : !llvm.struct<(ptr, ptr, i64)> 
    %7 = llvm.extractvalue %4[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.call @foo(%1, %2, %3, %5, %6, %7) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
}

// -----

llvm.func @foo(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: i64) -> () attributes {llvm.emit_c_interface} 

// Test that the input argument is correctly translated to the one needed by _mlir_ciface_foo

// CHECK-LABEL: llvm.func @_catalyst_pyface_foo
// CHECK:    [[var0:%.+]] = llvm.load %arg1
// CHECK:    [[var1:%.+]] = llvm.extractvalue [[var0]][0]
// CHECK:    [[var2:%.+]] = llvm.extractvalue [[var0]][1]
// CHECK:    llvm.call @_catalyst_ciface_foo([[var1]], [[var2]])

llvm.func @_mlir_ciface_foo(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64)> 
    %4 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    %5 = llvm.extractvalue %4[0] : !llvm.struct<(ptr, ptr, i64)> 
    %6 = llvm.extractvalue %4[1] : !llvm.struct<(ptr, ptr, i64)> 
    %7 = llvm.extractvalue %4[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.call @foo(%1, %2, %3, %5, %6, %7) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
}

// -----

llvm.func @foo(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64) -> () attributes {llvm.emit_c_interface} 

// Test that the input argument is wrapped around a structure and a pointer for only one
// CHECK-LABEL: llvm.func @_catalyst_pyface_foo(%arg0: !llvm.ptr, %arg1: !llvm.ptr)
llvm.func @_mlir_ciface_foo(%arg0: !llvm.ptr) {
    %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.call @foo(%1, %2, %3) : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
}

// -----

llvm.func @foo(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64) -> () attributes {llvm.emit_c_interface} 

// Test that the input argument is correctly translated to the one needed by _mlir_ciface_foo
// For only one argument

// CHECK-LABEL: llvm.func @_catalyst_pyface_foo
// CHECK:    [[var0:%.+]] = llvm.load %arg1
// CHECK:    [[var1:%.+]] = llvm.extractvalue [[var0]][0]
// CHECK:    llvm.call @_catalyst_ciface_foo([[var1]])

llvm.func @_mlir_ciface_foo(%arg0: !llvm.ptr) {
    %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.call @foo(%1, %2, %3) : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
}


