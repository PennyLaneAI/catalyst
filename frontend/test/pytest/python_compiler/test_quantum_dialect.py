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

"""Unit test module for catalyst/python_compiler/quantum_dialect.py"""


import pytest

import pennylane as qml

from catalyst import qjit

import pytest
from catalyst.python_compiler.quantum_dialect import QuantumDialect

name = QuantumDialect.name
all_ops = list(QuantumDialect.operations)
all_attrs = list(QuantumDialect.attributes)

expected_ops_names = {
    "AdjointOp": "quantum.adjoint",
    "AllocOp": "quantum.alloc",
    "ComputationalBasisOp": "quantum.compbasis",
    "CountsOp": "quantum.counts",
    "CustomOp": "quantum.custom",
    "DeallocOp": "quantum.dealloc",
    "DeviceInitOp": "quantum.device_init",
    "DeviceReleaseOp": "quantum.device_release",
    "ExpvalOp": "quantum.expval",
    "ExtractOp": "quantum.extract",
    "FinalizeOp": "quantum.finalize",
    "GlobalPhaseOp": "quantum.gphase",
    "HamiltonianOp": "quantum.hamiltonian",
    "HermitianOp": "quantum.hermitian",
    "InitializeOp": "quantum.init",
    "InsertOp": "quantum.insert",
    "MeasureOp": "quantum.measure",
    "MultiRZOp": "quantum.multirz",
    "NamedObsOp": "quantum.namedobs",
    "ProbsOp": "quantum.probs",
    "QubitUnitaryOp": "quantum.unitary",
    "SampleOp": "quantum.sample",
    "SetBasisStateOp": "quantum.set_basis_state",
    "SetStateOp": "quantum.set_state",
    "StateOp": "quantum.state",
    "TensorOp": "quantum.tensor",
    "VarianceOp": "quantum.var",
    "YieldOp": "quantum.yield",
}

expected_attrs_names = {
    "ObservableType": "quantum.obs",
    "QubitType": "quantum.bit",
    "QuregType": "quantum.reg",
    "ResultType": "quantum.res",
    "NamedObservableAttr": "quantum.named_observable",
}


def test_quantum_dialect_name():
    """Test that the QuantumDialect name is correct."""
    assert name == "quantum"


@pytest.mark.parametrize("op", all_ops)
def test_all_operations_names(op):
    """Test that all operations have the expected name."""
    op_class_name = op.__name__
    expected_name = expected_ops_names.get(op_class_name)
    assert (
        expected_name is not None
    ), f"Unexpected operation {op_class_name} found in QuantumDialect"
    assert op.name == expected_name


@pytest.mark.parametrize("attr", all_attrs)
def test_all_attributes_names(attr):
    """Test that all attributes have the expected name."""
    attr_class_name = attr.__name__
    expected_name = expected_attrs_names.get(attr_class_name)
    assert (
        expected_name is not None
    ), f"Unexpected attribute {attr_class_name} found in QuantumDialect"
    assert attr.name == expected_name
