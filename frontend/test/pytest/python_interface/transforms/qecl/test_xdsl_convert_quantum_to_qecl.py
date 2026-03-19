# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test module for the convert-quantum-to-qecl dialect-conversion transform."""

import pennylane as qml
import pytest

from catalyst.python_interface.transforms.qecl import (
    ConvertQuantumToQecLogicalPass,
    convert_quantum_to_qecl_pass,
)

pytestmark = pytest.mark.xdsl


class TestQuantumToQecLogicalPass:
    """Unit tests for the convert-quantum-to-qecl pass."""

    def test_alloc_k_1(self, run_filecheck):
        """Test that `quantum.alloc` ops are converted to their corresponding `qecl.alloc` ops for
        various initial numbers of qubits in the register.

        Perform the conversion for k = 1.
        """
        program = """
        func.func @test_program() {
            // CHECK: qecl.alloc() : !qecl.hyperreg<1 x 1>
            // CHECK-NOT: quantum.alloc
            %0 = quantum.alloc(1) : !quantum.reg

            // CHECK: qecl.alloc() : !qecl.hyperreg<2 x 1>
            // CHECK-NOT: quantum.alloc
            %1 = quantum.alloc(2) : !quantum.reg

            // CHECK: qecl.alloc() : !qecl.hyperreg<3 x 1>
            // CHECK-NOT: quantum.alloc
            %2 = quantum.alloc(3) : !quantum.reg

            // CHECK: qecl.alloc() : !qecl.hyperreg<4 x 1>
            // CHECK-NOT: quantum.alloc
            %3 = quantum.alloc(4) : !quantum.reg

            return
        }
        """
        pipeline = (ConvertQuantumToQecLogicalPass(k=1),)
        run_filecheck(program, pipeline)

    @pytest.mark.xfail(reason="Only k = 1 is supported", raises=NotImplementedError)
    def test_alloc_k_2(self, run_filecheck):
        """Test that `quantum.alloc` ops are converted to their corresponding `qecl.alloc` ops for
        various initial numbers of qubits in the register.

        Perform the conversion for k = 2.
        """
        program = """
        func.func @test_program() {
            // CHECK: qecl.alloc() : !qecl.hyperreg<1 x 2>
            // CHECK-NOT: quantum.alloc
            %0 = quantum.alloc(1) : !quantum.reg

            // CHECK: qecl.alloc() : !qecl.hyperreg<1 x 2>
            // CHECK-NOT: quantum.alloc
            %1 = quantum.alloc(2) : !quantum.reg

            // CHECK: qecl.alloc() : !qecl.hyperreg<2 x 2>
            // CHECK-NOT: quantum.alloc
            %2 = quantum.alloc(3) : !quantum.reg

            // CHECK: qecl.alloc() : !qecl.hyperreg<2 x 2>
            // CHECK-NOT: quantum.alloc
            %3 = quantum.alloc(4) : !quantum.reg

            return
        }
        """
        pipeline = (ConvertQuantumToQecLogicalPass(k=2),)
        run_filecheck(program, pipeline)

    @pytest.mark.xfail(reason="dynamic register size not supported", raises=NotImplementedError)
    def test_alloc_k_1_dyn(self, run_filecheck):
        """Test that `quantum.alloc` ops are converted to their corresponding `qecl.alloc` ops for a
        dynamic initial number of qubits in the register.

        Perform the conversion for k = 1.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[n_cst:%.+]] = arith.constant 1 : i64
            %0 = arith.constant 1 : i64

            // CHECK: qecl.alloc([[n_cst]]) : !qecl.hyperreg<? x 1>
            // CHECK-NOT: quantum.alloc
            %1 = quantum.alloc(%0) : !quantum.reg

            return
        }
        """
        pipeline = (ConvertQuantumToQecLogicalPass(k=1),)
        run_filecheck(program, pipeline)

    def test_extract_k_1_width_1(self, run_filecheck):
        """Test that `quantum.extract` ops are converted to their corresponding `qecl.extract_block`
        ops for registers with width = 1 and for k = 1. Also check that `qecl.encode[zero]` and
        `qecl.qec` ops are inserted after the `qecl.extract_block` op.

        In this case, the extract index is static.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = qecl.alloc() : !qecl.hyperreg<1 x 1>
            %0 = quantum.alloc(1) : !quantum.reg

            // CHECK: [[cb0:%.+]] = qecl.extract_block [[hreg0]][0] : !qecl.hyperreg<1 x 1> -> !qecl.codeblock<1>
            // CHECK: [[cb1:%.+]] = qecl.encode[zero] [[cb0]] : !qecl.codeblock<1>
            // CHECK: [[cb2:%.+]] = qecl.qec [[cb1]] : !qecl.codeblock<1>
            %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

            return
        }
        """
        pipeline = (ConvertQuantumToQecLogicalPass(k=1),)
        run_filecheck(program, pipeline)

    def test_extract_k_1_width_1_dyn_idx(self, run_filecheck):
        """Test that `quantum.extract` ops are converted to their corresponding `qecl.extract_block`
        ops for registers with width = 1 and for k = 1.

        In this case, the extract index is dynamic.
        """
        program = """
        func.func @test_program(%arg0 : i64) {
            // CHECK: [[hreg0:%.+]] = qecl.alloc() : !qecl.hyperreg<1 x 1>
            %0 = quantum.alloc(1) : !quantum.reg

            // CHECK: [[idx:%.+]] = arith.index_cast %arg0 : i64 to index
            // CHECK: [[cb0:%.+]] = qecl.extract_block [[hreg0]][[[idx]]] : !qecl.hyperreg<1 x 1> -> !qecl.codeblock<1>
            // CHECK: [[cb1:%.+]] = qecl.encode[zero] [[cb0]] : !qecl.codeblock<1>
            // CHECK: [[cb2:%.+]] = qecl.qec [[cb1]] : !qecl.codeblock<1>
            %1 = quantum.extract %0[%arg0] : !quantum.reg -> !quantum.bit

            return
        }
        """
        pipeline = (ConvertQuantumToQecLogicalPass(k=1),)
        run_filecheck(program, pipeline)

    def test_extract_k_1_width_2(self, run_filecheck):
        """Test that `quantum.extract` ops are converted to their corresponding `qecl.extract_block`
        ops for registers with width = 2 and for k = 1. Also check that `qecl.encode[zero]` and
        `qecl.qec` ops are inserted after each `qecl.extract_block` op.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = qecl.alloc() : !qecl.hyperreg<2 x 1>
            %0 = quantum.alloc(2) : !quantum.reg

            // CHECK: [[cb0:%.+]] = qecl.extract_block [[hreg0]][0] : !qecl.hyperreg<2 x 1> -> !qecl.codeblock<1>
            // CHECK: [[cb1:%.+]] = qecl.encode[zero] [[cb0]] : !qecl.codeblock<1>
            // CHECK: [[cb2:%.+]] = qecl.qec [[cb1]] : !qecl.codeblock<1>
            %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

            // CHECK: [[cb3:%.+]] = qecl.extract_block [[hreg0]][1] : !qecl.hyperreg<2 x 1> -> !qecl.codeblock<1>
            // CHECK: [[cb4:%.+]] = qecl.encode[zero] [[cb3]] : !qecl.codeblock<1>
            // CHECK: [[cb5:%.+]] = qecl.qec [[cb4]] : !qecl.codeblock<1>
            %2 = quantum.extract %0[1] : !quantum.reg -> !quantum.bit

            return
        }
        """
        pipeline = (ConvertQuantumToQecLogicalPass(k=1),)
        run_filecheck(program, pipeline)

    @pytest.mark.xfail(reason="Only k = 1 is supported", raises=NotImplementedError)
    def test_extract_k_2_width_1(self, run_filecheck):
        """Test that `quantum.extract` ops are converted to their corresponding `qecl.extract_block`
        ops for registers with width = 1 and for k = 2.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = qecl.alloc() : !qecl.hyperreg<1 x 2>
            %0 = quantum.alloc(1) : !quantum.reg

            // CHECK: [[cb0:%.+]] = qecl.extract_block [[hreg0]][0] : !qecl.hyperreg<1 x 2> -> !qecl.codeblock<2>
            // CHECK: [[cb1:%.+]] = qecl.encode[zero] [[cb0]] : !qecl.codeblock<2>
            // CHECK: [[cb2:%.+]] = qecl.qec [[cb1]] : !qecl.codeblock<2>
            %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

            return
        }
        """
        pipeline = (ConvertQuantumToQecLogicalPass(k=2),)
        run_filecheck(program, pipeline)

    @pytest.mark.xfail(reason="Only k = 1 is supported", raises=NotImplementedError)
    def test_extract_k_2_width_2(self, run_filecheck):
        """Test that `quantum.extract` ops are converted to their corresponding `qecl.extract_block`
        ops for registers with width = 2 and for k = 2.

        Here, we extract two abstract qubits from the quantum register, but because k=2, we only
        extract a single codeblock from the allocated hyper-register at the QEC logical layer.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = qecl.alloc() : !qecl.hyperreg<1 x 2>
            %0 = quantum.alloc(2) : !quantum.reg

            // CHECK: [[cb0:%.+]] = qecl.extract_block [[hreg0]][0] : !qecl.hyperreg<1 x 2> -> !qecl.codeblock<2>
            // CHECK: [[cb1:%.+]] = qecl.encode[zero] [[cb0]] : !qecl.codeblock<2>
            // CHECK: [[cb2:%.+]] = qecl.qec [[cb1]] : !qecl.codeblock<2>
            // CHECK-NOT: qecl.extract_block
            // CHECK-NOT: qecl.encode
            // CHECK-NOT: qecl.qec
            %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
            %2 = quantum.extract %0[1] : !quantum.reg -> !quantum.bit

            return
        }
        """
        pipeline = (ConvertQuantumToQecLogicalPass(k=2),)
        run_filecheck(program, pipeline)

    def test_extract_k_1_width_1_multiple(self, run_filecheck):
        """Test that `quantum.extract` ops are converted to their corresponding `qecl.extract_block`
        ops for registers with width = 1 and for k = 1, and that the `qecl.encode[zero]` and
        `qecl.qec` ops are only inserted after the initial alloc+extract_block op and not after
        subsequent extract_block ops.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = qecl.alloc() : !qecl.hyperreg<1 x 1>
            %0 = quantum.alloc(1) : !quantum.reg

            // CHECK: [[cb0:%.+]] = qecl.extract_block [[hreg0]][0] : !qecl.hyperreg<1 x 1> -> !qecl.codeblock<1>
            // CHECK: [[cb1:%.+]] = qecl.encode[zero] [[cb0]] : !qecl.codeblock<1>
            // CHECK: [[cb2:%.+]] = qecl.qec [[cb1]] : !qecl.codeblock<1>
            %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

            // CHECK: [[hreg1:%.+]] = qecl.insert_block [[hreg0]][0], [[cb2]] : !qecl.hyperreg<1 x 1>, !qecl.codeblock<1>
            %2 = quantum.insert %0[0], %1 : !quantum.reg, !quantum.bit

            // CHECK: [[cb3:%.+]] = qecl.extract_block [[hreg1]][0] : !qecl.hyperreg<1 x 1> -> !qecl.codeblock<1>
            // CHECK-NOT: qecl.encode
            // CHECK-NOT: qecl.qec
            %3 = quantum.extract %2[0] : !quantum.reg -> !quantum.bit

            return
        }
        """
        pipeline = (ConvertQuantumToQecLogicalPass(k=1),)
        run_filecheck(program, pipeline)

    def test_insert_k_1_width_1(self, run_filecheck):
        """Test that `quantum.insert` ops are converted to their corresponding `qecl.insert_block`
        ops for a registers with width = 1 and for k = 1.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = qecl.alloc() : !qecl.hyperreg<1 x 1>
            %0 = quantum.alloc(1) : !quantum.reg

            // CHECK: [[cb0:%.+]] = qecl.extract_block [[hreg0]][0] : !qecl.hyperreg<1 x 1> -> !qecl.codeblock<1>
            // CHECK: [[cb1:%.+]] = qecl.encode[zero] [[cb0]] : !qecl.codeblock<1>
            // CHECK: [[cb2:%.+]] = qecl.qec [[cb1]] : !qecl.codeblock<1>
            %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

            // CHECK: [[hreg1:%.+]] = qecl.insert_block [[hreg0]][0], [[cb2]] : !qecl.hyperreg<1 x 1>, !qecl.codeblock<1>
            %2 = quantum.insert %0[0], %1 : !quantum.reg, !quantum.bit

            return
        }
        """
        pipeline = (ConvertQuantumToQecLogicalPass(k=1),)
        run_filecheck(program, pipeline)

    def test_dealloc_k_1_width_1(self, run_filecheck):
        """Test that `quantum.dealloc` ops are converted to their corresponding `qecl.dealloc` ops
        for a registers with width = 1 and for k = 1.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = qecl.alloc() : !qecl.hyperreg<1 x 1>
            %0 = quantum.alloc(1) : !quantum.reg

            // CHECK: qecl.dealloc [[hreg0]]  : !qecl.hyperreg<1 x 1>
            quantum.dealloc %0 : !quantum.reg

            return
        }
        """
        pipeline = (ConvertQuantumToQecLogicalPass(k=1),)
        run_filecheck(program, pipeline)

    def test_dealloc_k_1_width_2(self, run_filecheck):
        """Test that `quantum.dealloc` ops are converted to their corresponding `qecl.dealloc` ops
        for a registers with width = 2 and for k = 1.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = qecl.alloc() : !qecl.hyperreg<2 x 1>
            %0 = quantum.alloc(2) : !quantum.reg

            // CHECK: qecl.dealloc [[hreg0]]  : !qecl.hyperreg<2 x 1>
            quantum.dealloc %0 : !quantum.reg

            return
        }
        """
        pipeline = (ConvertQuantumToQecLogicalPass(k=1),)
        run_filecheck(program, pipeline)

    @pytest.mark.xfail(reason="Only k = 1 is supported", raises=NotImplementedError)
    def test_dealloc_k_2_width_1(self, run_filecheck):
        """Test that `quantum.dealloc` ops are converted to their corresponding `qecl.dealloc` ops
        for a registers with width = 1 and for k = 2.
        """
        program = """
        func.func @test_program() {
            // CHECK: [[hreg0:%.+]] = qecl.alloc() : !qecl.hyperreg<1 x 2>
            %0 = quantum.alloc(1) : !quantum.reg

            // CHECK: qecl.dealloc [[hreg0]]  : !qecl.hyperreg<1 x 2>
            quantum.dealloc %0 : !quantum.reg

            return
        }
        """
        pipeline = (ConvertQuantumToQecLogicalPass(k=2),)
        run_filecheck(program, pipeline)

    def test_gate_hadamard(self, run_filecheck):
        program = """
        func.func @test_program() {
            // CHECK: qecl.alloc() : !qecl.hyperreg<1 x 1>
            // CHECK: qecl.extract_block
            // CHECK: qecl.encode[zero]
            // CHECK: [[cb0:%.+]] = qecl.qec
            %0 = quantum.alloc(1) : !quantum.reg
            %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

            // CHECK: [[cb1:%.+]] = qecl.hadamard [[cb0]][0] : !qecl.codeblock<1>
            %2 = quantum.custom "Hadamard"() %1 : !quantum.bit
            return
        }
        """
        pipeline = (ConvertQuantumToQecLogicalPass(k=1),)
        run_filecheck(program, pipeline)

    def test_gate_s(self, run_filecheck):
        program = """
        func.func @test_program() {
            // CHECK: qecl.alloc() : !qecl.hyperreg<1 x 1>
            // CHECK: qecl.extract_block
            // CHECK: qecl.encode[zero]
            // CHECK: [[cb0:%.+]] = qecl.qec
            %0 = quantum.alloc(1) : !quantum.reg
            %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

            // CHECK: [[cb1:%.+]] = qecl.s [[cb0]][0] : !qecl.codeblock<1>
            %2 = quantum.custom "S"() %1 : !quantum.bit

            // CHECK: [[cb2:%.+]] = qecl.s [[cb1]][0] adj : !qecl.codeblock<1>
            %3 = quantum.custom "S"() %2 adj : !quantum.bit
            return
        }
        """
        pipeline = (ConvertQuantumToQecLogicalPass(k=1),)
        run_filecheck(program, pipeline)

    def test_gate_cnot(self, run_filecheck):
        program = """
        func.func @test_program() {
            // CHECK: qecl.alloc() : !qecl.hyperreg<2 x 1>
            // CHECK: qecl.extract_block
            // CHECK: qecl.encode[zero]
            // CHECK: [[cb0:%.+]] = qecl.qec
            // CHECK: qecl.extract_block
            // CHECK: qecl.encode[zero]
            // CHECK: [[cb1:%.+]] = qecl.qec
            %0 = quantum.alloc(2) : !quantum.reg
            %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
            %2 = quantum.extract %0[1] : !quantum.reg -> !quantum.bit

            // CHECK: [[cb2:%.+]], [[cb3:%.+]] = qecl.cnot [[cb0]][0], [[cb1]][0] : !qecl.codeblock<1>, !qecl.codeblock<1>
            %3, %4 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
            return
        }
        """
        pipeline = (ConvertQuantumToQecLogicalPass(k=1),)
        run_filecheck(program, pipeline)

    def test_measure(self, run_filecheck):
        program = """
        func.func @test_program() {
            // CHECK: qecl.alloc() : !qecl.hyperreg<1 x 1>
            // CHECK: qecl.extract_block
            // CHECK: qecl.encode[zero]
            // CHECK: [[cb0:%.+]] = qecl.qec
            %0 = quantum.alloc(1) : !quantum.reg
            %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

            // CHECK: [[mres:%.+]], [[cb1:%.+]] = qecl.measure [[cb0]][0] : i1, !qecl.codeblock<1>
            %mres, %2 = quantum.measure %1 : i1, !quantum.bit
            return
        }
        """
        pipeline = (ConvertQuantumToQecLogicalPass(k=1),)
        run_filecheck(program, pipeline)


class TestQuantumToQecLogicalPassIntegration:
    """Integration lit tests for the convert-quantum-to-qecl pass"""

    @pytest.mark.usefixtures("use_capture")
    def test_ghz_circuit(self, run_filecheck_qjit):
        dev = qml.device("null.qubit", wires=3)

        @qml.qjit(target="mlir")
        @convert_quantum_to_qecl_pass(k=1)
        @qml.qnode(dev, shots=1)
        def circuit():
            # CHECK: qecl.alloc() : !qecl.hyperreg<3 x 1>
            # CHECK: qecl.extract_block
            # CHECK: qecl.encode[zero]
            # CHECK: qecl.qec
            # CHECK: qecl.hadamard
            # CHECK: qecl.extract_block
            # CHECK: qecl.encode[zero]
            # CHECK: qecl.qec
            # CHECK: qecl.cnot
            # CHECK: qecl.extract_block
            # CHECK: qecl.encode[zero]
            # CHECK: qecl.qec
            # CHECK: qecl.cnot
            # CHECK: qecl.measure
            # CHECK: qecl.measure
            # CHECK: qecl.measure
            qml.H(0)
            qml.CNOT([0, 1])
            qml.CNOT([1, 2])
            m0 = qml.measure(0)
            m1 = qml.measure(1)
            m2 = qml.measure(2)
            return qml.sample([m0, m1, m2])

        run_filecheck_qjit(circuit)
