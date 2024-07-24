; Copyright 2022-2023 Xanadu Quantum Technologies Inc.

; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at

;     http://www.apache.org/licenses/LICENSE-2.0

; Unless required by applicable law or agreed to in writing, software
; distributed under the License is distributed on an "AS IS" BASIS,
; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
; See the License for the specific language governing permissions and
; limitations under the License.

; ModuleID = 'grad_qfunc'
source_filename = "grad_qfunc"
target triple = "x86_64-pc-linux-gnu"

%Result = type opaque
%Qubit = type opaque
%Array = type opaque

%struct.MemRefT = type { double*, double*, i64, [1 x i64], [1 x i64] }

@.str = private constant [15 x i8] c"grad[%d] = %f\0A\00", align 1
@rtd_lib = internal constant [33 x i8] c"../build/lib/librtd_lightning.so\00"
@rtd_name = internal constant [19 x i8] c"LightningSimulator\00"
@rtd_kwargs = internal constant [11 x i8] c"{shots: 0}\00"

declare i8* @aligned_alloc(i64, i64)

declare i32 @printf(i8*, ...)

declare void @free(i8*)

declare void @__catalyst__rt__device_init(i8*, i8*, i8*)

declare void @__catalyst__rt__initialize(i32*)

declare void @__catalyst__rt__finalize()

declare void @__catalyst__rt__print_state()

declare void @__catalyst__rt__toggle_recorder(i8)

declare i8* @__catalyst__rt__array_get_element_ptr_1d(%Array*, i64)

declare %Array* @__catalyst__rt__qubit_allocate_array(i64)

declare void @__catalyst__qis__RY(%Qubit*, double, i8)

declare void @__catalyst__qis__RZ(%Qubit*, double, i8)

declare void @__catalyst__qis__Hadamard(%Qubit*, i8)

declare i64 @__catalyst__qis__NamedObs(i8, %Qubit*)

declare double @__catalyst__qis__Expval(i64)

declare void @__catalyst__qis__Gradient(i64, ...)


; Print jacobian results at index %1
define void @print_jacobian_at(double* %0, i64 %1) {
  %3 = getelementptr inbounds double, double* %0, i64 %1
  %4 = load double, double* %3, align 8
  %5 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str, i64 0, i64 0), i64 %1, double %4)
  ret void
}

; A simple quantum circuit
define double @circuit(%Qubit* %0) {
  call void @__catalyst__qis__Hadamard(%Qubit* %0, i8 0)
  call void @__catalyst__qis__RZ(%Qubit* %0, double 0.3, i8 0)
  call void @__catalyst__qis__RY(%Qubit* %0, double 0.7, i8 0)
  call void @__catalyst__qis__RZ(%Qubit* %0, double 0.4, i8 0)
  %2 = call i64 @__catalyst__qis__NamedObs(i8 1, %Qubit* %0)
  %3 = call double @__catalyst__qis__Expval(i64 %2)
  ret double %3
}

define i32 @main() {
  ; Initialize quantum runtime
  call void @__catalyst__rt__initialize(i32* null)
  call void @__catalyst__rt__device_init(i8* getelementptr ([33 x i8], [33 x i8]* @rtd_lib, i64 0, i64 0), i8* getelementptr ([19 x i8], [19 x i8]* @rtd_name, i64 0, i64 0), i8* getelementptr ([11 x i8], [11 x i8]* @rtd_kwargs, i64 0, i64 0))

  ; Allocate 2 qubits
  %1 = call %Array* @__catalyst__rt__qubit_allocate_array(i64 2)
  %2 = call i8* @__catalyst__rt__array_get_element_ptr_1d(%Array* %1, i64 0)
  %3 = bitcast i8* %2 to %Qubit**
  %4 = load %Qubit*, %Qubit** %3, align 8

  ; Activate the recorder
  call void @__catalyst__rt__toggle_recorder(i8 1)

  ; Call a quantum circuit
  %5 = call double @circuit(%Qubit* %4)

  ; Allocate the buffer enough for two qubits
  %buffer_allocated = call i8* @aligned_alloc(i64 8, i64 16)
  %buffer_cast = bitcast i8* %buffer_allocated to double*

  ; Insert buffers into result structure
  %t0 = insertvalue %struct.MemRefT undef, double* %buffer_cast, 0
  %t1 = insertvalue %struct.MemRefT %t0, double* %buffer_cast, 1
  %t2 = insertvalue %struct.MemRefT %t1, i64 0, 2
  %t3 = insertvalue %struct.MemRefT %t2, i64 3, 3, 0
  %memref = insertvalue %struct.MemRefT %t3, i64 1, 4, 0
  %memref_ptr = alloca %struct.MemRefT, i64 1, align 8
  store %struct.MemRefT %memref, %struct.MemRefT* %memref_ptr, align 8

  ; Call the Gradient function
  call void (i64, ...) @__catalyst__qis__Gradient(i64 1, %struct.MemRefT* %memref_ptr)

  ; Deactivate the recorder
  call void @__catalyst__rt__toggle_recorder(i8 0)

  ; Print results
  call void @print_jacobian_at(double* %buffer_cast, i64 0)
  call void @print_jacobian_at(double* %buffer_cast, i64 1)
  call void @print_jacobian_at(double* %buffer_cast, i64 2)

  ; Print the updated state-vector
  call void @__catalyst__rt__print_state()

  ; Close the context and free memory
  call void @free(i8* %buffer_allocated)
  call void @__catalyst__rt__finalize()
  ret i32 0
}
