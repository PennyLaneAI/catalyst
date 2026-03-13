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

"""Test module for the quantum-to-qecl dialect-conversion transform."""

import pytest

from catalyst.python_interface.transforms.qecl import ConvertQuantumToQecLogicalPass

pytestmark = pytest.mark.xdsl


class TestQuantumToQecLogicalPass:
    """Unit tests for the quantum-to-qecl pass."""

    def test_alloc_k_1(self, run_filecheck):
        program = """
        func.func @test_alloc_k_1() {
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

    def test_alloc_k_2(self, run_filecheck):
        program = """
        func.func @test_alloc_k_2() {
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
        program = """
        func.func @test_alloc_k_2() {
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
