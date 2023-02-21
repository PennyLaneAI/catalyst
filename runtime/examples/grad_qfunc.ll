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

declare i8* @malloc(i64)

declare i32 @printf(i8*, ...)

declare void @free(i8*)

declare void @__quantum__rt__initialize()

declare void @__quantum__rt__finalize()

declare void @__quantum__rt__print_state()

declare void @__quantum__rt__toggle_recorder(i8)

declare i8* @__quantum__rt__array_get_element_ptr_1d(%Array*, i64)

declare %Array* @__quantum__rt__qubit_allocate_array(i64)

declare void @__quantum__qis__RY(%Qubit*, double, i8)

declare void @__quantum__qis__RZ(%Qubit*, double, i8)

declare void @__quantum__qis__Hadamard(%Qubit*, i8)

declare i64 @__quantum__qis__NamedObs(i8, %Qubit*)

declare double @__quantum__qis__Expval(i64)

declare void @__quantum__qis__Gradient(i64, ...)


; Print jacobian results at index %1
define void @print_jacobian_at(double* %0, i64 %1) {
  %3 = getelementptr inbounds double, double* %0, i64 %1
  %4 = load double, double* %3, align 8
  %5 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str, i64 0, i64 0), i64 %1, double %4)
  ret void
}

; A simple quantum circuit
define double @circuit(%Qubit* %0) {
  call void @__quantum__qis__Hadamard(%Qubit* %0, i8 0)
  call void @__quantum__qis__RZ(%Qubit* %0, double 0.3, i8 0)
  call void @__quantum__qis__RY(%Qubit* %0, double 0.7, i8 0)
  call void @__quantum__qis__RZ(%Qubit* %0, double 0.4, i8 0)
  %2 = call i64 @__quantum__qis__NamedObs(i8 1, %Qubit* %0)
  %3 = call double @__quantum__qis__Expval(i64 %2)
  ret double %3
}

define i32 @main() {
  ; Initialize quantum runtime
  call void @__quantum__rt__initialize()

  ; Allocate 2 qubits
  %1 = call %Array* @__quantum__rt__qubit_allocate_array(i64 2)
  %2 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %1, i64 0)
  %3 = bitcast i8* %2 to %Qubit**
  %4 = load %Qubit*, %Qubit** %3, align 8

  ; Activate the recorder
  call void @__quantum__rt__toggle_recorder(i8 1)

  ; Call a quantum circuit
  %5 = call double @circuit(%Qubit* %4)

  ; Call the gradient instruction
  %6 = call i8* @malloc(i64 40)
  %7 = bitcast i8* %6 to %struct.MemRefT*

  call void (i64, ...) @__quantum__qis__Gradient(i64 1, %struct.MemRefT* %7)

  %8 = getelementptr %struct.MemRefT, %struct.MemRefT* %7, i32 0, i32 0
  %9 = load double*, double** %8, align 8

  ; Deactivate the recorder
  call void @__quantum__rt__toggle_recorder(i8 0)

  ; Print results
  call void @print_jacobian_at(double* %9, i64 0)
  call void @print_jacobian_at(double* %9, i64 1)
  call void @print_jacobian_at(double* %9, i64 2)

  ; Print the updated state-vector
  call void @__quantum__rt__print_state()

  ; Close the context and free memory
  call void @free(i8* %6)
  call void @__quantum__rt__finalize()
  ret i32 0
}
