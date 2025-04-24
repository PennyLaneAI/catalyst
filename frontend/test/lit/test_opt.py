# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for opt"""

# RUN: %PYTHON %s | FileCheck %s

from catalyst.compiler import _quantum_opt, canonicalize, to_llvmir


def test_opt_happy_path():
    """Test functionality of opt-tool"""

    mlir = """
    module {
        // CHECK: llvm.func @foo
        func.func @foo() {
            return
        }
    }
    """
    output = _quantum_opt(stdin=mlir)
    print(output)


test_opt_happy_path()


def test_opt_canonicalize():
    """Test functionality of canonicalization"""

    mlir = """
    module {
        // CHECK: func.func @foo
        func.func @foo() {
            return
        }
    }
    """
    output = canonicalize(stdin=mlir)
    print(output)


test_opt_canonicalize()


def test_to_llvmir():
    """Test functionality of lowering to llvm"""

    mlir = """
    module {
        // CHECK: define void @foo
        func.func @foo() {
            return
        }
    }
    """
    output = to_llvmir(stdin=mlir)
    print(output)


test_to_llvmir()
