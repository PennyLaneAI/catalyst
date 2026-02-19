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
"""Unit tests for the xDSL Quantum dialect."""
from io import StringIO

import pytest
from xdsl.context import Context
from xdsl.dialects.builtin import (
    ArrayAttr,
    ComplexType,
    Float64Type,
    IntegerAttr,
    StringAttr,
    TensorType,
    UnitAttr,
    i1,
    i64,
)
from xdsl.dialects.test import Test, TestOp
from xdsl.ir import AttributeCovT, Block, Operation, OpResult, Region
from xdsl.irdl import ConstraintContext
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import ParseError, VerifyException

from catalyst.python_interface.dialects import Quantum, quantum
from catalyst.python_interface.dialects.quantum import (
    CustomOp,
    NamedObservableAttr,
    ObservableType,
    QubitLevel,
    QubitRole,
    QubitType,
    QubitTypeConstraint,
    QuregType,
    QuregTypeConstraint,
)

pytestmark = pytest.mark.xdsl

all_ops = list(Quantum.operations)
all_attrs = list(Quantum.attributes)

expected_ops_names = {
    "AdjointOp": "quantum.adjoint",
    "AllocOp": "quantum.alloc",
    "AllocQubitOp": "quantum.alloc_qb",
    "ComputationalBasisOp": "quantum.compbasis",
    "CountsOp": "quantum.counts",
    "CustomOp": "quantum.custom",
    "DeallocOp": "quantum.dealloc",
    "DeallocQubitOp": "quantum.dealloc_qb",
    "DeviceInitOp": "quantum.device",
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
    "NumQubitsOp": "quantum.num_qubits",
    "PauliRotOp": "quantum.paulirot",
    "PCPhaseOp": "quantum.pcphase",
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

TestOp.__test__ = False
"""Setting this attribute silences the PytestCollectionWarning that TestOp can not be collected
for testing, because it is a class with __init__ method."""


# Test function taken from xdsl/utils/test_value.py
def create_ssa_value(t: AttributeCovT) -> OpResult[AttributeCovT]:
    """Create a single SSA value with the given type for testing purposes."""
    op = TestOp(result_types=(t,))
    return op.results[0]


q0 = create_ssa_value(QubitType())
q1 = create_ssa_value(QubitType())
q2 = create_ssa_value(QubitType())
qreg = create_ssa_value(QuregType())
theta = create_ssa_value(Float64Type())
dim = create_ssa_value(Float64Type())
pauli_x = NamedObservableAttr("PauliX")
obs = create_ssa_value(ObservableType())
bool_ssa = create_ssa_value(i1)
matrix = create_ssa_value(TensorType(element_type=Float64Type(), shape=(2, 2)))
coeffs = create_ssa_value(TensorType(Float64Type(), shape=(10,)))
samples = create_ssa_value(TensorType(Float64Type(), shape=(8, 7)))
basis_state = create_ssa_value(TensorType(i1, shape=(8,)))
state = create_ssa_value(TensorType(ComplexType(Float64Type()), shape=(16,)))
pauli_word = ArrayAttr([StringAttr("X"), StringAttr("Y"), StringAttr("Z")])
int_ssa = create_ssa_value(i64)
int_attr = IntegerAttr(0, 32)

expected_ops_init_kwargs = {
    "AdjointOp": [
        {
            "qreg": qreg,
            "region": Region(Block((CustomOp(gate_name="CNOT", in_qubits=(q0, q1)),))),
        }
    ],
    "AllocOp": [{"nqubits": 3}, {"nqubits": int_ssa}, {"nqubits": IntegerAttr(3, 64)}],
    "AllocQubitOp": [{}],
    "ComputationalBasisOp": [{"operands": (q0, None), "result_types": (obs,)}],
    "CountsOp": [
        {
            "operands": (obs, int_ssa, None, None),
            "result_types": (TensorType(Float64Type(), shape=(1,)), TensorType(i64, shape=(1,))),
        }
    ],
    "CustomOp": [
        {
            "gate_name": "RX",
            "params": (theta,),
            "in_qubits": (q0, q1),
            "in_ctrl_qubits": (q2,),
            "in_ctrl_values": (bool_ssa,),
            "adjoint": True,
        },
        {
            "gate_name": StringAttr("RX"),
            "params": theta,
            "in_qubits": q0,
            "in_ctrl_qubits": q2,
            "in_ctrl_values": bool_ssa,
        },
    ],
    "DeallocOp": [{"qreg": qreg}],
    "DeallocQubitOp": [{"qubit": q0}],
    "DeviceInitOp": [
        {
            "operands": (int_ssa,),
            "properties": {"lib": StringAttr("lib"), "device_name": StringAttr("my_device")},
        }
    ],
    "DeviceReleaseOp": [{}],
    "ExpvalOp": [{"obs": obs}],
    "ExtractOp": [
        {"qreg": qreg, "idx": int_ssa},
        {"qreg": qreg, "idx": int_attr},
        {"qreg": qreg, "idx": 0},
    ],
    "FinalizeOp": [{}],
    "GlobalPhaseOp": [
        {"params": theta, "in_ctrl_qubits": (q0,), "in_ctrl_values": (bool_ssa,)},
        {"params": theta, "in_ctrl_qubits": q0, "in_ctrl_values": bool_ssa},
    ],
    "HamiltonianOp": [{"operands": (coeffs, (obs,)), "result_types": (obs,)}],
    "HermitianOp": [{"operands": (matrix, (q0, q1)), "result_types": (obs,)}],
    "InitializeOp": [{}],
    "InsertOp": [
        {"in_qreg": qreg, "idx": int_ssa, "qubit": q1},
        {"in_qreg": qreg, "idx": int_attr, "qubit": q1},
        {"in_qreg": qreg, "idx": 0, "qubit": q1},
    ],
    "MeasureOp": [
        {"in_qubit": q0, "postselect": int_ssa},
        {"in_qubit": q0},
        {"in_qubit": q0, "postselect": 1},
    ],
    "MultiRZOp": [
        {
            "theta": theta,
            "in_qubits": (q1, q0),
            "in_ctrl_qubits": (q2,),
            "in_ctrl_values": (bool_ssa,),
            "adjoint": UnitAttr(),
        },
        {
            "theta": theta,
            "in_qubits": q1,
            "in_ctrl_qubits": q2,
            "in_ctrl_values": bool_ssa,
            "adjoint": UnitAttr(),
        },
    ],
    "NamedObsOp": [{"qubit": q0, "obs_type": pauli_x}],
    "NumQubitsOp": [{"result_types": (int_ssa,)}],
    "PauliRotOp": [
        {"angle": theta, "pauli_product": "XYZ", "in_qubits": (q0,)},
        {"angle": theta, "pauli_product": ["X", "Y", "Z"], "in_qubits": (q0,)},
        {"angle": theta, "pauli_product": pauli_word, "in_qubits": (q0,)},
        {
            "angle": theta,
            "pauli_product": pauli_word,
            "in_qubits": q0,
            "in_ctrl_qubits": q2,
            "in_ctrl_values": bool_ssa,
        },
    ],
    "PCPhaseOp": [
        {
            "theta": theta,
            "dim": dim,
            "in_qubits": (q1, q0),
            "in_ctrl_qubits": (q2,),
            "in_ctrl_values": (bool_ssa,),
            "adjoint": False,
        },
        {
            "theta": theta,
            "dim": dim,
            "in_qubits": q1,
            "in_ctrl_qubits": q2,
            "in_ctrl_values": bool_ssa,
        },
    ],
    "ProbsOp": [
        {
            "operands": (obs, int_ssa, None),
            "result_types": (TensorType(Float64Type(), shape=(8,)),),
        }
    ],
    "QubitUnitaryOp": [
        {"matrix": matrix, "in_qubits": (q2,), "adjoint": True},
        {"matrix": matrix, "in_qubits": q2, "in_ctrl_qubits": q1, "in_ctrl_values": bool_ssa},
    ],
    "SampleOp": [{"operands": (obs, int_ssa, samples), "result_types": (samples,)}],
    "SetBasisStateOp": [{"operands": (basis_state, (q0, q2)), "result_types": ((q1, q2),)}],
    "SetStateOp": [{"operands": (state, (q0, q1)), "result_types": ((q0, q1),)}],
    "StateOp": [{"operands": (obs, int_ssa, state), "result_types": (state,)}],
    "TensorOp": [{"operands": ((obs, obs),), "result_types": (obs,)}],
    "VarianceOp": [{"obs": (obs,)}],
    "YieldOp": [{"operands": (qreg,)}],
}


class TestDialectBasics:
    """Unit tests for basic checks for the Quantum dialect."""

    def test_quantum_dialect_name(self):
        """Test that the QuantumDialect name is correct."""
        assert Quantum.name == "quantum"

    @pytest.mark.parametrize("op", all_ops)
    def test_all_operations_names(self, op):
        """Test that all operations have the expected name."""
        op_class_name = op.__name__
        expected_name = expected_ops_names.get(op_class_name)
        assert (
            expected_name is not None
        ), f"Unexpected operation {op_class_name} found in QuantumDialect"
        assert op.name == expected_name

    def test_only_existing_operations_are_expected(self):
        """Test that the expected operations above only contain existing operations."""
        existing_ops_names = {op.__name__ for op in all_ops}
        assert existing_ops_names == set(expected_ops_names)

    @pytest.mark.parametrize("op", all_ops)
    def test_operation_construction(self, op):
        """Test the constructors of operations in the Quantum dialect."""
        kwargs_list = expected_ops_init_kwargs[op.__name__]
        for kwargs in kwargs_list:
            cloned_kwargs = {
                k: v.clone() if isinstance(v, (Operation, Region)) else v for k, v in kwargs.items()
            }
            _ = op(**cloned_kwargs)

    @pytest.mark.parametrize("attr", all_attrs)
    def test_all_attributes_names(self, attr):
        """Test that all attributes have the expected name."""
        attr_class_name = attr.__name__
        expected_name = expected_attrs_names.get(attr_class_name)
        assert (
            expected_name is not None
        ), f"Unexpected attribute {attr_class_name} found in QuantumDialect"
        assert attr.name == expected_name

    def test_only_existing_attributes_are_expected(self):
        """Test that the expected attributes above only contain existing attributes."""
        existing_attrs_names = {attr.__name__ for attr in all_attrs}
        assert existing_attrs_names == set(expected_attrs_names)


class TestQubitType:
    """Unit tests for QubitType and its associated QubitTypeConstraint."""

    @pytest.mark.parametrize(
        "level,expected_level",
        [
            (None, StringAttr("abstract")),
            ("abstract", StringAttr("abstract")),
            ("pbc", StringAttr("pbc")),
            ("physical", StringAttr("physical")),
            ("logical", StringAttr("logical")),
        ],
    )
    @pytest.mark.parametrize(
        "role,expected_role",
        [
            (None, StringAttr("null")),
            ("null", StringAttr("null")),
            ("data", StringAttr("data")),
            ("xcheck", StringAttr("xcheck")),
            ("zcheck", StringAttr("zcheck")),
        ],
    )
    def test_constructor(self, level, expected_level, role, expected_role):
        """Test that the parameters of QubitType are correct with defaults."""
        args = {}
        if level is not None:
            args["level"] = level
        if role is not None:
            args["role"] = role

        ty = QubitType(**args)
        assert ty.level == expected_level
        assert ty.role == expected_role

    @pytest.mark.parametrize(
        "args,error",
        [
            ({}, None),
            ({"level": "physical", "role": "xcheck"}, None),
            ({"level": "foo"}, "Invalid value foo for 'QubitType.level'"),
            ({"role": "bar"}, "Invalid value bar for 'QubitType.role'"),
        ],
    )
    def test_verify(self, args, error):
        """Test that QubitType verifies correctly."""
        if error:
            with pytest.raises(VerifyException, match=error):
                QubitType(**args)
        else:
            QubitType(**args)

    @pytest.mark.parametrize(
        "ty,expected",
        [
            (QubitType(), "!quantum.bit"),
            (QubitType(level="abstract", role="null"), "!quantum.bit"),
            (QubitType(role="xcheck"), "!quantum.bit<xcheck>"),
            (QubitType(level="physical"), "!quantum.bit<physical>"),
            (QubitType(level="logical", role="data"), "!quantum.bit<logical, data>"),
        ],
    )
    @pytest.mark.parametrize("generic", [True, False])
    def test_printing(self, ty, expected, generic):
        """Test that QubitType is printed correctly."""
        buf = StringIO()
        printer = Printer(stream=buf, print_generic_format=generic)
        printer.print_attribute(ty)
        assert buf.getvalue() == expected

    @pytest.mark.parametrize(
        "input_str,expected_level,expected_role",
        [
            ('%0 = "test.op"() : () -> !quantum.bit', "abstract", "null"),
            ('%0 = "test.op"() : () -> !quantum.bit<abstract>', "abstract", "null"),
            ('%0 = "test.op"() : () -> !quantum.bit<null>', "abstract", "null"),
            ('%0 = "test.op"() : () -> !quantum.bit<abstract, null>', "abstract", "null"),
            ('%0 = "test.op"() : () -> !quantum.bit<logical>', "logical", "null"),
            ('%0 = "test.op"() : () -> !quantum.bit<data>', "abstract", "data"),
            ('%0 = "test.op"() : () -> !quantum.bit<physical, xcheck>', "physical", "xcheck"),
        ],
    )
    def test_parsing(self, input_str, expected_level, expected_role):
        """Test that QubitType is parsed correctly."""
        ctx = Context()
        ctx.load_dialect(Quantum)
        ctx.load_dialect(Test)
        op = Parser(ctx, input=input_str).parse_op()

        ty = op.results[0].type
        assert ty.level.data == expected_level
        assert ty.role.data == expected_role

    def test_parsing_error(self):
        """Test that an error is raised if a qubit being parsed has an invalid
        number of parameters"""
        input_str = '%0 = "test.op"() : () -> !quantum.bit<abstract, data, foo>'
        ctx = Context()
        ctx.load_dialect(Quantum)
        ctx.load_dialect(Test)

        with pytest.raises(ParseError, match="Expected 2 or fewer parameters"):
            _ = Parser(ctx, input=input_str).parse_op()

    @pytest.mark.parametrize(
        "constr,can_infer",
        [
            (QubitTypeConstraint(), True),
            (QubitTypeConstraint(level_constr=["logical"]), True),
            (QubitTypeConstraint(role_constr=["data"]), True),
            (QubitTypeConstraint(level_constr=["physical"], role_constr=["xcheck"]), True),
            (QubitTypeConstraint(level_constr=["abstract", "logical"]), False),
            (QubitTypeConstraint(role_constr=["null", "data"]), False),
            (
                QubitTypeConstraint(
                    level_constr=["abstract", "logical"], role_constr=["null", "data"]
                ),
                False,
            ),
        ],
    )
    def test_constraint_can_infer(self, constr, can_infer):
        """Test that QubitTypeConstraint can infer the type correctly if possible."""
        assert constr.can_infer({}) == can_infer

    @pytest.mark.parametrize(
        "constr,expected_type",
        [
            (QubitTypeConstraint(), QubitType()),
            (QubitTypeConstraint(level_constr=["logical"]), QubitType(level="logical")),
            (QubitTypeConstraint(role_constr=["data"]), QubitType(role="data")),
            (
                QubitTypeConstraint(level_constr=["physical"], role_constr=["xcheck"]),
                QubitType(level="physical", role="xcheck"),
            ),
        ],
    )
    def test_constraint_infer(self, constr, expected_type):
        """Test that QubitTypeConstraint infers the correct type based on its constraints."""
        ty = constr.infer(ConstraintContext())
        assert ty == expected_type

    @pytest.mark.parametrize(
        "constr,ty,error",
        [
            (QubitTypeConstraint(), QubitType(), None),
            (QubitTypeConstraint(), QubitType(level="logical", role="xcheck"), None),
            (QubitTypeConstraint(level_constr=["logical"]), QubitType(level="logical"), None),
            (
                QubitTypeConstraint(level_constr=["logical"]),
                QubitType(level="physical"),
                'Unexpected attribute "physical"',
            ),
            (QubitTypeConstraint(role_constr=["data"]), QubitType(role="data"), None),
            (
                QubitTypeConstraint(role_constr=["data"]),
                QubitType(role="xcheck"),
                'Unexpected attribute "xcheck"',
            ),
            (
                QubitTypeConstraint(level_constr=["physical"], role_constr=["xcheck"]),
                QubitType(level="physical", role="xcheck"),
                None,
            ),
            (
                QubitTypeConstraint(level_constr=["physical"], role_constr=["xcheck"]),
                QubitType(level="physical", role="null"),
                'Unexpected attribute "null"',
            ),
            (
                QubitTypeConstraint(level_constr=["physical"], role_constr=["xcheck"]),
                QubitType(level="logical", role="xcheck"),
                'Unexpected attribute "logical"',
            ),
            (
                QubitTypeConstraint(level_constr=["abstract", "logical"]),
                QubitType(level="logical"),
                None,
            ),
            (
                QubitTypeConstraint(level_constr=["abstract", "logical"]),
                QubitType(level="physical"),
                'Unexpected attribute "physical"',
            ),
            (QubitTypeConstraint(role_constr=["null", "data"]), QubitType(role="null"), None),
            (
                QubitTypeConstraint(role_constr=["null", "data"]),
                QubitType(role="xcheck"),
                'Unexpected attribute "xcheck"',
            ),
            (
                QubitTypeConstraint(
                    level_constr=["abstract", "logical"], role_constr=["null", "data"]
                ),
                QubitType(level="logical", role="data"),
                None,
            ),
            (
                QubitTypeConstraint(
                    level_constr=["abstract", "logical"], role_constr=["null", "data"]
                ),
                QubitType(level="physical", role="data"),
                'Unexpected attribute "physical"',
            ),
            (
                QubitTypeConstraint(
                    level_constr=["abstract", "logical"], role_constr=["null", "data"]
                ),
                QubitType(level="logical", role="xcheck"),
                'Unexpected attribute "xcheck"',
            ),
        ],
    )
    def test_constraint_verify(self, constr, ty, error):
        """Test that QubitTypeConstraint verifies correctly."""
        if error:
            with pytest.raises(VerifyException, match=error):
                constr.verify(ty, ConstraintContext())
        else:
            constr.verify(ty, ConstraintContext())


class TestQuregType:
    """Unit tests for QuregType and its associated QuregTypeConstraint."""

    @pytest.mark.parametrize(
        "level,expected_level",
        [
            (None, StringAttr("abstract")),
            ("abstract", StringAttr("abstract")),
            ("pbc", StringAttr("pbc")),
            ("physical", StringAttr("physical")),
            ("logical", StringAttr("logical")),
        ],
    )
    def test_constructor(self, level, expected_level):
        """Test that the parameters of QuregType are correct with defaults."""
        args = {"level": level} if level is not None else {}

        ty = QuregType(**args)
        assert ty.level == expected_level

    @pytest.mark.parametrize(
        "level,error",
        [
            (None, None),
            ("physical", None),
            ("foo", "Invalid value foo for 'QuregType.level'"),
        ],
    )
    def test_verify(self, level, error):
        """Test that QuregType verifies correctly."""
        args = {"level": level} if level is not None else {}
        if error:
            with pytest.raises(VerifyException, match=error):
                QuregType(**args)
        else:
            QuregType(**args)

    @pytest.mark.parametrize(
        "ty,expected",
        [
            (QuregType(), "!quantum.reg"),
            (QuregType(level="abstract"), "!quantum.reg"),
            (QuregType(level="physical"), "!quantum.reg<physical>"),
        ],
    )
    @pytest.mark.parametrize("generic", [True, False])
    def test_printing(self, ty, expected, generic):
        """Test that QuregType is printed correctly."""
        buf = StringIO()
        printer = Printer(stream=buf, print_generic_format=generic)
        printer.print_attribute(ty)
        assert buf.getvalue() == expected

    @pytest.mark.parametrize(
        "input_str,expected_level",
        [
            ('%0 = "test.op"() : () -> !quantum.reg', "abstract"),
            ('%0 = "test.op"() : () -> !quantum.reg<abstract>', "abstract"),
            ('%0 = "test.op"() : () -> !quantum.reg<logical>', "logical"),
        ],
    )
    def test_parsing(self, input_str, expected_level):
        """Test that QuregType is parsed correctly."""
        ctx = Context()
        ctx.load_dialect(Quantum)
        ctx.load_dialect(Test)
        op = Parser(ctx, input=input_str).parse_op()

        ty = op.results[0].type
        assert ty.level.data == expected_level

    def test_parsing_error(self):
        """Test that an error is raised if a register being parsed has an invalid
        number of parameters"""
        input_str = '%0 = "test.op"() : () -> !quantum.reg<abstract, foo>'
        ctx = Context()
        ctx.load_dialect(Quantum)
        ctx.load_dialect(Test)

        with pytest.raises(ParseError, match="Expected 1 or fewer parameters"):
            _ = Parser(ctx, input=input_str).parse_op()

    @pytest.mark.parametrize(
        "constr,can_infer",
        [
            (QuregTypeConstraint(), True),
            (QuregTypeConstraint(level_constr=["logical"]), True),
            (QuregTypeConstraint(level_constr=["abstract", "logical"]), False),
            (QuregTypeConstraint(level_constr=["abstract", "logical", "physical", "pbc"]), True),
        ],
    )
    def test_constraint_can_infer(self, constr, can_infer):
        """Test that QuregTypeConstraint can infer the type correctly if possible."""
        assert constr.can_infer({}) == can_infer

    @pytest.mark.parametrize(
        "constr,expected_type",
        [
            (QuregTypeConstraint(), QuregType()),
            (QuregTypeConstraint(level_constr=["logical"]), QuregType(level="logical")),
            (
                QuregTypeConstraint(level_constr=["abstract", "logical", "pbc", "physical"]),
                QuregType(level="abstract"),
            ),
        ],
    )
    def test_constraint_infer(self, constr, expected_type):
        """Test that QuregTypeConstraint infers the correct type based on its constraints."""
        ty = constr.infer(ConstraintContext())
        assert ty == expected_type

    @pytest.mark.parametrize(
        "constr,ty,error",
        [
            (QuregTypeConstraint(), QuregType(), None),
            (QuregTypeConstraint(), QuregType(level="logical"), None),
            (QuregTypeConstraint(level_constr=["logical"]), QuregType(level="logical"), None),
            (
                QuregTypeConstraint(level_constr=["logical"]),
                QuregType(level="physical"),
                'Unexpected attribute "physical"',
            ),
            (
                QuregTypeConstraint(level_constr=["abstract", "logical"]),
                QuregType(level="logical"),
                None,
            ),
            (
                QuregTypeConstraint(level_constr=["abstract", "logical"]),
                QuregType(level="physical"),
                'Unexpected attribute "physical"',
            ),
        ],
    )
    def test_constraint_verify(self, constr, ty, error):
        """Test that QuregTypeConstraint verifies correctly."""
        if error:
            with pytest.raises(VerifyException, match=error):
                constr.verify(ty, ConstraintContext())
        else:
            constr.verify(ty, ConstraintContext())


class TestCustomVerifiers:
    """Unit tests for operations and attributes that have custom verification."""

    def test_valid_paulirot(self):
        """Test that a valid PauliRotOp passes verification."""
        op = quantum.PauliRotOp(angle=theta, pauli_product="XYZ", in_qubits=(q0, q1, q2))
        op.verify()

    def test_invalid_paulirot(self):
        """Test that invalid PauliRotOps raise an error during verification."""
        # Invalid pauli string
        op = quantum.PauliRotOp(angle=theta, pauli_product="WYZ", in_qubits=(q0, q1, q2))
        with pytest.raises(ValueError, match="is not a valid Pauli operator"):
            op.verify()

        # Invalid pauli string length
        op = quantum.PauliRotOp(angle=theta, pauli_product="XY", in_qubits=(q0, q1, q2))
        with pytest.raises(ValueError, match="The length of the Pauli word"):
            op.verify()


@pytest.mark.parametrize(
    "pretty_print", [pytest.param(True, id="pretty_print"), pytest.param(False, id="generic_print")]
)
class TestAssemblyFormat:
    """Lit tests for assembly format of operations/attributes in the Quantum
    dialect."""

    def test_qubit_qreg_operations(self, run_filecheck, pretty_print):
        """Test that the assembly format for operations for allocation/deallocation of
        qubits/quantum registers works correctly."""

        # Tests for allocation/deallocation ops: AllocOp, DeallocOp, AllocQubitOp, DeallocQubitOp
        # Tests for extraction/insertion ops: ExtractOp, InsertOp
        program = """
        ////////////////// **Allocation of register with dynamic number of wires** //////////////////
        // CHECK: [[NQUBITS:%.+]] = "test.op"() : () -> i64
        // CHECK: [[QREG_DYN:%.+]] = quantum.alloc([[NQUBITS]]) : !quantum.reg
        %nqubits = "test.op"() : () -> i64
        %qreg_dynamic = quantum.alloc(%nqubits) : !quantum.reg

        ////////////////// **Deallocation of dynamic register** //////////////////
        // CHECK: quantum.dealloc [[QREG_DYN]] : !quantum.reg
        quantum.dealloc %qreg_dynamic : !quantum.reg

        ////////////////// **Allocation of register with static number of wires** //////////////////
        // CHECK: [[QREG_STATIC:%.+]] = quantum.alloc(10) : !quantum.reg
        %qreg_static = quantum.alloc(10) : !quantum.reg

        {{%.+}} = quantum.alloc(10) : !quantum.reg<logical>
        %qreg_logical = quantum.alloc(10) : !quantum.reg<logical>
        {{%.+}} = quantum.alloc(10) : !quantum.reg
        %qreg_logical = quantum.alloc(10) : !quantum.reg<abstract>

        ////////////////// **Deallocation of static register** //////////////////
        // CHECK: quantum.dealloc [[QREG_STATIC]] : !quantum.reg
        quantum.dealloc %qreg_static : !quantum.reg

        ////////////////// **Dynamic qubit allocation** //////////////////
        // CHECK: [[DYN_QUBIT:%.+]] = quantum.alloc_qb : !quantum.bit
        %dyn_qubit = quantum.alloc_qb : !quantum.bit

        ////////////////// **Dynamic qubit deallocation** //////////////////
        // CHECK: quantum.dealloc_qb [[DYN_QUBIT]] : !quantum.bit
        quantum.dealloc_qb %dyn_qubit : !quantum.bit

        //////////////////////////////////////////////////////
        ////////////////// Quantum register //////////////////
        //////////////////////////////////////////////////////
        // CHECK: [[QREG:%.+]] = "test.op"() : () -> !quantum.reg
        %qreg = "test.op"() : () -> !quantum.reg

        ////////////////// **Static qubit extraction** //////////////////
        // CHECK: [[STATIC_QUBIT:%.+]] = quantum.extract [[QREG]][[[STATIC_INDEX:0]]] : !quantum.reg -> !quantum.bit
        %static_qubit = quantum.extract %qreg[0] : !quantum.reg -> !quantum.bit

        ////////////////// **Dynamic qubit extraction** //////////////////
        // CHECK: [[DYN_INDEX:%.+]] = "test.op"() : () -> i64
        // CHECK: [[DYN_QUBIT1:%.+]] = quantum.extract [[QREG]][[[DYN_INDEX]]] : !quantum.reg -> !quantum.bit
        %dyn_index = "test.op"() : () -> i64
        %dyn_qubit1 = quantum.extract %qreg[%dyn_index] : !quantum.reg -> !quantum.bit

        ////////////////// **Static qubit insertion** //////////////////
        // CHECK: [[QREG1:%.+]] = quantum.insert [[QREG]][[[STATIC_INDEX]]], [[STATIC_QUBIT]] : !quantum.reg, !quantum.bit
        %qreg1 = quantum.insert %qreg[0], %static_qubit : !quantum.reg, !quantum.bit

        ////////////////// **Dynamic qubit insertion** //////////////////
        // CHECK: quantum.insert [[QREG1]][[[DYN_INDEX]]], [[DYN_QUBIT1]] : !quantum.reg, !quantum.bit
        %qreg2 = quantum.insert %qreg1[%dyn_index], %dyn_qubit1 : !quantum.reg, !quantum.bit

        //////////////////////////////////////////////////////////////////
        //////////////Hierarchical qubits/quregs testing//////////////////
        //////////////////////////////////////////////////////////////////

        /////////////////////// **QuregType** ///////////////////////
        {{%.+}} = "test.op"() : () -> !quantum.reg
        %qreg_abstract0 = "test.op"() : () -> !quantum.reg
        {{%.+}} = "test.op"() : () -> !quantum.reg
        %qreg_abstract1 = "test.op"() : () -> !quantum.reg<abstract>
        {{%.+}} = "test.op"() : () -> !quantum.reg<logical>
        %qreg_logical = "test.op"() : () -> !quantum.reg<logical>

        /////////////////////// **QubitType** ///////////////////////
        // Defaults
        {{%.+}} = "test.op"() : () -> !quantum.bit
        %qb_abstract_null0 = "test.op"() : () -> !quantum.bit
        {{%.+}} = "test.op"() : () -> !quantum.bit
        %qb_abstract_null1 = "test.op"() : () -> !quantum.bit<abstract>
        {{%.+}} = "test.op"() : () -> !quantum.bit
        %qb_abstract_null2 = "test.op"() : () -> !quantum.bit<null>
        {{%.+}} = "test.op"() : () -> !quantum.bit
        %qb_abstract_null3 = "test.op"() : () -> !quantum.bit<abstract, null>

        //// Single arg ////
        // Levels
        {{%.+}} = "test.op"() : () -> !quantum.bit<logical>
        %qb_level0 = "test.op"() : () -> !quantum.bit<logical>
        {{%.+}} = "test.op"() : () -> !quantum.bit<physical>
        %qb_level1 = "test.op"() : () -> !quantum.bit<physical>
        {{%.+}} = "test.op"() : () -> !quantum.bit<pbc>
        %qb_level2 = "test.op"() : () -> !quantum.bit<pbc>

        // Roles
        {{%.+}} = "test.op"() : () -> !quantum.bit<data>
        %qb_role0 = "test.op"() : () -> !quantum.bit<data>
        {{%.+}} = "test.op"() : () -> !quantum.bit<xcheck>
        %qb_role1 = "test.op"() : () -> !quantum.bit<xcheck>
        {{%.+}} = "test.op"() : () -> !quantum.bit<zcheck>
        %qb_role2 = "test.op"() : () -> !quantum.bit<zcheck>

        // Multiple args
        {{%.+}} = "test.op"() : () -> !quantum.bit<logical, data>
        %qb_mul0 = "test.op"() : () -> !quantum.bit<logical, data>
        {{%.+}} = "test.op"() : () -> !quantum.bit<physical, xcheck>
        %qb_mul0 = "test.op"() : () -> !quantum.bit<physical, xcheck>
        """

        run_filecheck(program, roundtrip=True, verify=True, pretty_print=pretty_print)

    def test_quantum_ops(self, run_filecheck, pretty_print):
        """Test that the assembly format for quantum non-terminal operations works correctly."""

        # Tests for CustomOp, GlobalPhaseOp, MeasureOp, MultiRZOp, QubitUnitaryOp
        program = """
        ////////////////////////////////////////////////////////////////////////
        ////////////////// Qubits, params, and control values //////////////////
        ////////////////////////////////////////////////////////////////////////
        ////////////////// **Qubits** //////////////////
        // CHECK: [[Q0:%.+]] = "test.op"() : () -> !quantum.bit
        // CHECK: [[Q1:%.+]] = "test.op"() : () -> !quantum.bit
        // CHECK: [[Q2:%.+]] = "test.op"() : () -> !quantum.bit
        // CHECK: [[Q3:%.+]] = "test.op"() : () -> !quantum.bit
        %q0 = "test.op"() : () -> !quantum.bit
        %q1 = "test.op"() : () -> !quantum.bit
        %q2 = "test.op"() : () -> !quantum.bit
        %q3 = "test.op"() : () -> !quantum.bit

        ////////////////// **Params** //////////////////
        // CHECK: [[PARAM1:%.+]] = "test.op"() : () -> f64
        // CHECK: [[PARAM2:%.+]] = "test.op"() : () -> f64
        // CHECK: [[MAT_TENSOR:%.+]] = "test.op"() : () -> tensor<4x4xcomplex<f64>>
        // CHECK: [[MAT_MEMREF:%.+]] = "test.op"() : () -> memref<4x4xcomplex<f64>>
        %param1 = "test.op"() : () -> f64
        %param2 = "test.op"() : () -> f64
        %mat_tensor = "test.op"() : () -> tensor<4x4xcomplex<f64>>
        %mat_memref = "test.op"() : () -> memref<4x4xcomplex<f64>>

        ////////////////// **Control values** //////////////////
        // CHECK: [[TRUE_CST:%.+]] = "test.op"() : () -> i1
        // CHECK: [[FALSE_CST:%.+]] = "test.op"() : () -> i1
        %true_cst = "test.op"() : () -> i1
        %false_cst = "test.op"() : () -> i1

        ///////////////////////////////////////////////////////////////////////
        ///////////////////////// **Operation tests** /////////////////////////
        ///////////////////////////////////////////////////////////////////////

        ////////////////// **CustomOp tests** //////////////////
        // No params, no control wires
        // CHECK: {{%.+}}, {{%.+}} = quantum.custom "Gate"() [[Q0]], [[Q1]] : !quantum.bit, !quantum.bit
        %qc1, %qc2 = quantum.custom "Gate"() %q0, %q1 : !quantum.bit, !quantum.bit

        // Params, no control wires
        // CHECK: {{%.+}}, {{%.+}} = quantum.custom "ParamGate"([[PARAM1]], [[PARAM2]]) [[Q0]], [[Q1]] : !quantum.bit, !quantum.bit
        %qc3, %qc4 = quantum.custom "ParamGate"(%param1, %param2) %q0, %q1 : !quantum.bit, !quantum.bit

        // Control wires and values
        // CHECK: {{%.+}}, {{%.+}} = quantum.custom "ControlledGate"() [[Q0]] ctrls([[Q1]]) ctrlvals([[TRUE_CST]]) : !quantum.bit ctrls !quantum.bit
        %qc5, %qc6 = quantum.custom "ControlledGate"() %q0 ctrls(%q1) ctrlvals(%true_cst) : !quantum.bit ctrls !quantum.bit

        // Adjoint
        // CHECK: {{%.+}} = quantum.custom "AdjGate"() [[Q0]] adj : !quantum.bit
        %qc8 = quantum.custom "AdjGate"() %q0 adj : !quantum.bit

        ////////////////// **GlobalPhaseOp tests** //////////////////
        // No control wires
        // CHECK: quantum.gphase([[PARAM1]]) :
        quantum.gphase(%param1) :

        // Control wires and values
        // CHECK: {{%.+}}, {{%.+}} = quantum.gphase([[PARAM1]]) ctrls([[Q0]], [[Q1]]) ctrlvals([[FALSE_CST]], [[TRUE_CST]]) : ctrls !quantum.bit, !quantum.bit
        %qg1, %qg2 = quantum.gphase(%param1) ctrls(%q0, %q1) ctrlvals(%false_cst, %true_cst) : ctrls !quantum.bit, !quantum.bit

        // Adjoint
        // CHECK: {{%.+}} = quantum.gphase([[PARAM1]]) {adjoint} ctrls([[Q0]]) ctrlvals([[TRUE_CST]]) : ctrls !quantum.bit
        %qg3 = quantum.gphase(%param1) {adjoint} ctrls(%q0) ctrlvals(%true_cst) : ctrls !quantum.bit

        ////////////////// **MultiRZOp tests** //////////////////
        // No control wires
        // CHECK: {{%.+}}, {{%.+}} = quantum.multirz([[PARAM1]]) [[Q0]], [[Q1]] : !quantum.bit, !quantum.bit
        %qm1, %qm2 = quantum.multirz(%param1) %q0, %q1 : !quantum.bit, !quantum.bit

        // Control wires and values
        // CHECK: {{%.+}}, {{%.+}}, {{%.+}} = quantum.multirz([[PARAM1]]) [[Q0]], [[Q1]] ctrls([[Q2]]) ctrlvals([[TRUE_CST]]) : !quantum.bit, !quantum.bit
        %qm3, %qm4, %qm5 = quantum.multirz(%param1) %q0, %q1 ctrls(%q2) ctrlvals(%true_cst) : !quantum.bit, !quantum.bit ctrls !quantum.bit

        // Adjoint
        // CHECK: {{%.+}}, {{%.+}} = quantum.multirz([[PARAM1]]) [[Q0]], [[Q1]] adj : !quantum.bit, !quantum.bit
        %qm6, %qm7 = quantum.multirz(%param1) %q0, %q1 adj : !quantum.bit, !quantum.bit

        ////////////////// **PauliRotOp tests** //////////////////
        // No control wires
        // CHECK: {{%.+}}, {{%.+}}, {{%.+}} = quantum.paulirot ["X", "Y", "Z"]([[PARAM1]]) [[Q0]], [[Q1]], [[Q2]] : !quantum.bit, !quantum.bit, !quantum.bit
        %qpr1, %qpr2, %qpr3 = quantum.paulirot ["X", "Y", "Z"](%param1) %q0, %q1, %q2 : !quantum.bit, !quantum.bit, !quantum.bit

        // Control wires and values
        // CHECK: {{%.+}}, {{%.+}}, {{%.+}} = quantum.paulirot ["X", "Y"]([[PARAM1]]) [[Q0]], [[Q1]] ctrls([[Q2]]) ctrlvals([[TRUE_CST]]) : !quantum.bit, !quantum.bit ctrls !quantum.bit
        %qpr4, %qpr5, %qpr6 = quantum.paulirot ["X", "Y"](%param1) %q0, %q1 ctrls(%q2) ctrlvals(%true_cst) : !quantum.bit, !quantum.bit ctrls !quantum.bit

        // Adjoint
        // CHECK: {{%.+}}, {{%.+}} = quantum.paulirot ["X", "Y"]([[PARAM1]]) [[Q0]], [[Q1]] adj : !quantum.bit, !quantum.bit
        %qpr7, %qpr8 = quantum.paulirot ["X", "Y"](%param1) %q0, %q1 adj : !quantum.bit, !quantum.bit

        ////////////////// **PCPhaseOp tests** //////////////////
        // No control wires
        // CHECK: {{%.+}}, {{%.+}}, {{%.+}} = quantum.pcphase([[PARAM1]], [[PARAM2]]) [[Q0]], [[Q1]], [[Q2]] : !quantum.bit, !quantum.bit, !quantum.bit
        %qp1, %qp2, %qp3 = quantum.pcphase(%param1, %param2) %q0, %q1, %q2 : !quantum.bit, !quantum.bit, !quantum.bit

        // Control wires and values
        // CHECK: {{%.+}}, {{%.+}}, {{%.+}} = quantum.pcphase([[PARAM1]], [[PARAM2]]) [[Q0]], [[Q1]] ctrls([[Q2]]) ctrlvals([[TRUE_CST]]) : !quantum.bit, !quantum.bit ctrls !quantum.bit
        %qp4, %qp5, %qp6 = quantum.pcphase(%param1, %param2) %q0, %q1 ctrls(%q2) ctrlvals(%true_cst) : !quantum.bit, !quantum.bit ctrls !quantum.bit

        // Adjoint
        // CHECK: {{%.+}}, {{%.+}} = quantum.pcphase([[PARAM1]], [[PARAM2]]) [[Q0]], [[Q1]] adj : !quantum.bit, !quantum.bit
        %qp7, %qp8 = quantum.pcphase(%param1, %param2) %q0, %q1 adj : !quantum.bit, !quantum.bit

        ////////////////// **QubitUnitaryOp tests** //////////////////
        // No control wires
        // CHECK: {{%.+}}, {{%.+}} = quantum.unitary([[MAT_TENSOR]] : tensor<4x4xcomplex<f64>>) [[Q0]], [[Q1]] : !quantum.bit, !quantum.bit
        %qb1, %qb2 = quantum.unitary(%mat_tensor : tensor<4x4xcomplex<f64>>) %q0, %q1 : !quantum.bit, !quantum.bit

        // Control wires and values
        // CHECK: {{%.+}}, {{%.+}} {{%.+}} = quantum.unitary([[MAT_TENSOR]] : tensor<4x4xcomplex<f64>>) [[Q0]], [[Q1]] ctrls([[Q2]]) ctrlvals([[FALSE_CST]]) : !quantum.bit, !quantum.bit ctrls !quantum.bit
        %qb3, %qb4, %qb5 = quantum.unitary(%mat_tensor : tensor<4x4xcomplex<f64>>) %q0, %q1 ctrls(%q2) ctrlvals(%false_cst) : !quantum.bit, !quantum.bit ctrls !quantum.bit

        // Adjoint
        // CHECK: {{%.+}}, {{%.+}} = quantum.unitary([[MAT_TENSOR]] : tensor<4x4xcomplex<f64>>) [[Q0]], [[Q1]] adj : !quantum.bit, !quantum.bit
        %qb6, %qb7 = quantum.unitary(%mat_tensor : tensor<4x4xcomplex<f64>>) %q0, %q1 adj : !quantum.bit, !quantum.bit

        // MemRef
        // CHECK: {{%.+}}, {{%.+}} = quantum.unitary([[MAT_MEMREF]] : memref<4x4xcomplex<f64>>) [[Q0]], [[Q1]] : !quantum.bit, !quantum.bit
        %qb8, %qb9 = quantum.unitary(%mat_memref : memref<4x4xcomplex<f64>>) %q0, %q1 : !quantum.bit, !quantum.bit

        ////////////////// **MeasureOp tests** //////////////////
        // No postselection
        // CHECK: {{%.+}}, {{%.+}} = quantum.measure [[Q0]] : i1, !quantum.bit
        %mres1, %mqubit1 = quantum.measure %q0 : i1, !quantum.bit

        // Postselection
        // CHECK: {{%.+}}, {{%.+}} = quantum.measure [[Q1]] postselect 0 : i1, !quantum.bit
        // CHECK: {{%.+}}, {{%.+}} = quantum.measure [[Q2]] postselect 1 : i1, !quantum.bit
        %mres2, %mqubit2 = quantum.measure %q1 postselect 0 : i1, !quantum.bit
        %mres3, %mqubit3 = quantum.measure %q2 postselect 1 : i1, !quantum.bit
        """

        run_filecheck(program, roundtrip=True, verify=True, pretty_print=pretty_print)

    def test_state_prep(self, run_filecheck, pretty_print):
        """Test that the assembly format for state prep operations works correctly."""

        # Tests for SetBasisStateOp, SetStateOp
        program = """
        ////////////////////////////////////////////
        ////////////////// Qubits //////////////////
        ////////////////////////////////////////////
        // CHECK: [[Q0:%.+]] = "test.op"() : () -> !quantum.bit
        // CHECK: [[Q1:%.+]] = "test.op"() : () -> !quantum.bit
        %q0 = "test.op"() : () -> !quantum.bit
        %q1 = "test.op"() : () -> !quantum.bit

        ////////////////// **SetBasisStateOp tests** //////////////////
        // Basis state containers
        // CHECK: [[BASIS_TENSOR:%.+]] = "test.op"() : () -> tensor<2xi1>
        // CHECK: [[BASIS_MEMREF:%.+]] = "test.op"() : () -> memref<2xi1>
        %basis_tensor = "test.op"() : () -> tensor<2xi1>
        %basis_memref = "test.op"() : () -> memref<2xi1>

        // Basis state operations
        // CHECK: [[Q2:%.+]], [[Q3:%.+]] = quantum.set_basis_state([[BASIS_TENSOR]]) [[Q0]], [[Q1]] : (tensor<2xi1>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        // CHECK: [[Q4:%.+]], [[Q5:%.+]] = quantum.set_basis_state([[BASIS_MEMREF]]) [[Q2]], [[Q3]] : (memref<2xi1>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        %q2, %q3 = quantum.set_basis_state(%basis_tensor) %q0, %q1 : (tensor<2xi1>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        %q4, %q5 = quantum.set_basis_state(%basis_memref) %q2, %q3 : (memref<2xi1>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)

        ////////////////// **SetStateOp tests** //////////////////
        // State vector containers
        // CHECK: [[STATE_TENSOR:%.+]] = "test.op"() : () -> tensor<4xcomplex<f64>>
        // CHECK: [[STATE_MEMREF:%.+]] = "test.op"() : () -> memref<4xcomplex<f64>>
        %state_tensor = "test.op"() : () -> tensor<4xcomplex<f64>>
        %state_memref = "test.op"() : () -> memref<4xcomplex<f64>>

        // State prep operations
        // CHECK: [[Q6:%.+]], [[Q7:%.+]] = quantum.set_state([[STATE_TENSOR]]) [[Q4]], [[Q5]] : (tensor<4xcomplex<f64>>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        // CHECK: quantum.set_state([[STATE_MEMREF]]) [[Q6]], [[Q7]] : (memref<4xcomplex<f64>>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        %q6, %q7 = quantum.set_state(%state_tensor) %q4, %q5 : (tensor<4xcomplex<f64>>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        %q8, %q9 = quantum.set_state(%state_memref) %q6, %q7 : (memref<4xcomplex<f64>>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        """

        run_filecheck(program, roundtrip=True, verify=True, pretty_print=pretty_print)

    def test_observables(self, run_filecheck, pretty_print):
        """Test that the assembly format for observable operations works correctly."""

        # Tests for observables: ComputationalBasisOp, HamiltonianOp, HermitianOp,
        #                        NamedObsOp, TensorOp
        program = """
        //////////////////////////////////////////////////////
        //////////// Quantum register  and qubits ////////////
        //////////////////////////////////////////////////////
        // CHECK: [[QREG:%.+]] = "test.op"() : () -> !quantum.reg
        %qreg = "test.op"() : () -> !quantum.reg

        // CHECK: [[Q0:%.+]] = "test.op"() : () -> !quantum.bit
        // CHECK: [[Q1:%.+]] = "test.op"() : () -> !quantum.bit
        // CHECK: [[Q2:%.+]] = "test.op"() : () -> !quantum.bit
        // CHECK: [[Q3:%.+]] = "test.op"() : () -> !quantum.bit
        // CHECK: [[Q4:%.+]] = "test.op"() : () -> !quantum.bit
        %q0 = "test.op"() : () -> !quantum.bit
        %q1 = "test.op"() : () -> !quantum.bit
        %q2 = "test.op"() : () -> !quantum.bit
        %q3 = "test.op"() : () -> !quantum.bit
        %q4 = "test.op"() : () -> !quantum.bit

        //////////////////////////////////////////////
        //////////// **Observable tests** ////////////
        //////////////////////////////////////////////

        //////////// **NamedObsOp** ////////////
        // CHECK: [[X_OBS:%.+]] = quantum.namedobs [[Q0]][PauliX] : !quantum.obs
        // CHECK: [[Y_OBS:%.+]] = quantum.namedobs [[Q1]][PauliY] : !quantum.obs
        // CHECK: [[Z_OBS:%.+]] = quantum.namedobs [[Q2]][PauliZ] : !quantum.obs
        // CHECK: [[H_OBS:%.+]] = quantum.namedobs [[Q3]][Hadamard] : !quantum.obs
        // CHECK: [[I_OBS:%.+]] = quantum.namedobs [[Q4]][Identity] : !quantum.obs
        %x_obs = quantum.namedobs %q0[PauliX] : !quantum.obs
        %y_obs = quantum.namedobs %q1[PauliY] : !quantum.obs
        %z_obs = quantum.namedobs %q2[PauliZ] : !quantum.obs
        %h_obs = quantum.namedobs %q3[Hadamard] : !quantum.obs
        %i_obs = quantum.namedobs %q4[Identity] : !quantum.obs

        //////////// **HermitianOp** ////////////
        // Create tensor/memref
        // CHECK: [[HERM_TENSOR:%.+]] = "test.op"() : () -> tensor<2x2xcomplex<f64>>
        // CHECK: [[HERM_MEMREF:%.+]] = "test.op"() : () -> memref<2x2xcomplex<f64>>
        %herm_tensor = "test.op"() : () -> tensor<2x2xcomplex<f64>>
        %herm_memref = "test.op"() : () -> memref<2x2xcomplex<f64>>

        // Create Hermitians
        // CHECK: [[HERM1:%.+]] = quantum.hermitian([[HERM_TENSOR]] : tensor<2x2xcomplex<f64>>) [[Q0]] : !quantum.obs
        // CHECK: [[HERM2:%.+]] = quantum.hermitian([[HERM_MEMREF]] : memref<2x2xcomplex<f64>>) [[Q1]] : !quantum.obs
        %herm1 = quantum.hermitian(%herm_tensor : tensor<2x2xcomplex<f64>>) %q0 : !quantum.obs
        %herm2 = quantum.hermitian(%herm_memref : memref<2x2xcomplex<f64>>) %q1 : !quantum.obs

        //////////// **TensorOp** ////////////
        // CHECK: [[TENSOR_OBS:%.+]] = quantum.tensor [[X_OBS]], [[HERM2]], [[I_OBS]] : !quantum.obs
        %tensor_obs = quantum.tensor %x_obs, %herm2, %i_obs : !quantum.obs

        //////////// **HamiltonianOp** ////////////
        // Create tensor/memref
        // CHECK: [[HAM_TENSOR:%.+]] = "test.op"() : () -> tensor<3xf64>
        // CHECK: [[HAM_MEMREF:%.+]] = "test.op"() : () -> memref<3xf64>
        %ham_tensor = "test.op"() : () -> tensor<3xf64>
        %ham_memref = "test.op"() : () -> memref<3xf64>

        // Create Hamiltonians
        // CHECK: {{%.+}} = quantum.hamiltonian([[HAM_TENSOR]] : tensor<3xf64>) [[TENSOR_OBS]], [[X_OBS]], [[HERM1]] : !quantum.obs
        // CHECK: {{%.+}} = quantum.hamiltonian([[HAM_MEMREF]] : memref<3xf64>) [[TENSOR_OBS]], [[X_OBS]], [[HERM1]] : !quantum.obs
        %ham1 = quantum.hamiltonian(%ham_tensor : tensor<3xf64>) %tensor_obs, %x_obs, %herm1 : !quantum.obs
        %ham2 = quantum.hamiltonian(%ham_memref : memref<3xf64>) %tensor_obs, %x_obs, %herm1 : !quantum.obs

        //////////// **ComputationalBasisOp** ////////////
        // CHECK: {{%.+}} = quantum.compbasis qubits [[Q0]], [[Q1]] : !quantum.obs
        // CHECK: {{%.+}} = quantum.compbasis qreg [[QREG]] : !quantum.obs
        %cb_01 = quantum.compbasis qubits %q0, %q1 : !quantum.obs
        %cb_all = quantum.compbasis qreg %qreg : !quantum.obs
        """

        run_filecheck(program, roundtrip=True, verify=True, pretty_print=pretty_print)

    def test_measurements(self, run_filecheck, pretty_print):
        """Test that the assembly format for measurement operations works correctly."""

        # Tests for measurements: CountsOp, ExpvalOp, MeasureOp, ProbsOp, SampleOp,
        #                         StateOp, VarianceOp
        program = """
        ///////////////////////////////////////////////////
        //////////// Observables and constants ////////////
        ///////////////////////////////////////////////////
        // CHECK: [[Q0:%.+]], [[Q1:%.+]], [[Q2:%.+]] = "test.op"() : () -> (!quantum.bit
        %q0, %q1, %q2 = "test.op"() : () -> (!quantum.bit, !quantum.bit, !quantum.bit)
        // CHECK: [[QREG:%.+]] = "test.op"() : () -> !quantum.reg
        %qreg = "test.op"() : () -> !quantum.reg

        // CHECK: [[X_OBS:%.+]] = quantum.namedobs [[Q0]][PauliX] : !quantum.obs
        %x_obs = quantum.namedobs %q0[PauliX] : !quantum.obs
        // CHECK: [[C_OBS:%.+]] = quantum.compbasis qubits [[Q0]], [[Q1]], [[Q2]] : !quantum.obs
        %c_obs = quantum.compbasis qubits %q0, %q1, %q2 : !quantum.obs
        // CHECK: [[C_OBS_ALL:%.+]] = quantum.compbasis qreg [[QREG]] : !quantum.obs
        %c_obs_all = quantum.compbasis qreg %qreg : !quantum.obs

        // CHECK: [[DYN_WIRES:%.+]] = "test.op"() : () -> i64
        %dyn_wires = "test.op"() : () -> i64
        // CHECK: [[DYN_SHOTS:%.+]] = "test.op"() : () -> i64
        %dyn_shots = "test.op"() : () -> i64

        ///////////////////////////////////////////////
        //////////// **Measurement tests** ////////////
        ///////////////////////////////////////////////

        ///////////////////// **ExpvalOp** /////////////////////
        // CHECK: {{%.+}} = quantum.expval [[X_OBS]] : f64
        %expval = quantum.expval %x_obs : f64

        ///////////////////// **VarianceOp** /////////////////////
        // CHECK: {{%.+}} = quantum.var [[X_OBS]] : f64
        %var = quantum.var %x_obs : f64

        ///////////////////// **CountsOp** /////////////////////
        // Counts with static shape
        // CHECK: {{%.+}}, {{%.+}} = quantum.counts [[X_OBS]] : tensor<2xf64>, tensor<2xi64>
        %eigvals1, %counts1 = quantum.counts %x_obs : tensor<2xf64>, tensor<2xi64>

        // Counts with dynamic shape
        // CHECK: {{%.+}}, {{%.+}} = quantum.counts [[C_OBS_ALL]] shape [[DYN_WIRES]] : tensor<?xf64>, tensor<?xi64>
        %eigvals2, %counts2 = quantum.counts %c_obs_all shape %dyn_wires : tensor<?xf64>, tensor<?xi64>

        // Counts with no results (mutate memref in-place)
        // CHECK: [[EIGVALS_IN:%.+]] = "test.op"() : () -> memref<8xf64>
        // CHECK: [[COUNTS_IN:%.+]] = "test.op"() : () -> memref<8xi64>
        // CHECK: quantum.counts [[C_OBS]] in([[EIGVALS_IN]] : memref<8xf64>, [[COUNTS_IN]] : memref<8xi64>)
        %eigvals_in = "test.op"() : () -> memref<8xf64>
        %counts_in = "test.op"() : () -> memref<8xi64>
        quantum.counts %c_obs in(%eigvals_in : memref<8xf64>, %counts_in : memref<8xi64>)

        ///////////////////// **ProbsOp** /////////////////////
        // Probs with static shape
        // CHECK: {{%.+}} = quantum.probs [[C_OBS]] : tensor<8xf64>
        %probs1 = quantum.probs %c_obs : tensor<8xf64>

        // Probs with dynamic shape
        // CHECK: {{%.+}} = quantum.probs [[C_OBS_ALL]] shape [[DYN_WIRES]] : tensor<?xf64>
        %probs2 = quantum.probs %c_obs_all shape %dyn_wires : tensor<?xf64>

        // Probs with no results (mutate memref in-place)
        // CHECK: [[PROBS_IN:%.+]] = "test.op"() : () -> memref<8xf64>
        // CHECK: quantum.probs [[C_OBS]] in([[PROBS_IN]] : memref<8xf64>)
        %probs_in = "test.op"() : () -> memref<8xf64>
        quantum.probs %c_obs in(%probs_in : memref<8xf64>)

        ///////////////////// **StateOp** /////////////////////
        // State with static shape
        // CHECK: {{%.+}} = quantum.state [[C_OBS_ALL]] : tensor<8xcomplex<f64>>
        %state1 = quantum.state %c_obs_all : tensor<8xcomplex<f64>>

        // State with dynamic shape
        // CHECK: {{%.+}} = quantum.state [[C_OBS_ALL]] shape [[DYN_WIRES]] : tensor<?xcomplex<f64>>
        %state2 = quantum.state %c_obs_all shape %dyn_wires : tensor<?xcomplex<f64>>

        // State with no results (mutate memref in-place)
        // CHECK: [[STATE_IN:%.+]] = "test.op"() : () -> memref<8xcomplex<f64>>
        // CHECK: quantum.state [[C_OBS_ALL]] in([[STATE_IN]] : memref<8xcomplex<f64>>)
        %state_in = "test.op"() : () -> memref<8xcomplex<f64>>
        quantum.state %c_obs_all in(%state_in : memref<8xcomplex<f64>>)

        ///////////////////// **SampleOp** /////////////////////
        // Samples with static shape
        // CHECK: {{%.+}} = quantum.sample [[C_OBS]] : tensor<10x3xf64>
        %samples1 = quantum.sample %c_obs : tensor<10x3xf64>

        // Samples with dynamic wires
        // CHECK: {{%.+}} = quantum.sample [[C_OBS_ALL]] shape [[DYN_WIRES]] : tensor<10x?xf64>
        %samples2 = quantum.sample %c_obs_all shape %dyn_wires : tensor<10x?xf64>

        // Samples with dynamic shots
        // CHECK: {{%.+}} = quantum.sample [[C_OBS]] shape [[DYN_SHOTS]] : tensor<?x3xf64>
        %samples3 = quantum.sample %c_obs shape %dyn_shots : tensor<?x3xf64>

        // Samples with dynamic wires and shots
        // CHECK: {{%.+}} = quantum.sample [[C_OBS_ALL]] shape [[DYN_SHOTS]], [[DYN_WIRES]] : tensor<?x?xf64>
        %samples4 = quantum.sample %c_obs_all shape %dyn_shots, %dyn_wires : tensor<?x?xf64>

        // Samples with no results (mutate memref in-place)
        // CHECK: [[SAMPLES_IN:%.+]] = "test.op"() : () -> memref<7x3xf64>
        // CHECK: quantum.sample [[C_OBS]] in([[SAMPLES_IN]] : memref<7x3xf64>)
        %samples_in = "test.op"() : () -> memref<7x3xf64>
        quantum.sample %c_obs in(%samples_in : memref<7x3xf64>)
        """

        run_filecheck(program, roundtrip=True, verify=True, pretty_print=pretty_print)

    def test_miscellaneous_operations(self, run_filecheck, pretty_print):
        """Test that the assembly format for miscelleneous operations
        works correctly."""

        # Tests for AdjointOp, DeviceInitOp, DeviceReleaseOp, FinalizeOp, InitializeOp,
        # NumQubitsOp, YieldOp
        program = """
        //////////////////////////////////////////
        //////////// Quantum register ////////////
        //////////////////////////////////////////
        // CHECK: [[QREG:%.+]] = "test.op"() : () -> !quantum.reg
        %qreg = "test.op"() : () -> !quantum.reg

        //////////// **AdjointOp and YieldOp tests** ////////////
        // CHECK:      quantum.adjoint([[QREG]]) : !quantum.reg {
        // CHECK-NEXT: ^bb0([[ARG_QREG:%.+]] : !quantum.reg):
        // CHECK-NEXT:   quantum.yield [[ARG_QREG]] : !quantum.reg
        // CHECK-NEXT: }
        %qreg1 = quantum.adjoint(%qreg) : !quantum.reg {
        ^bb0(%arg_qreg: !quantum.reg):
          quantum.yield %arg_qreg : !quantum.reg
        }

        //////////// **DeviceInitOp tests** ////////////
        // Integer SSA value for shots
        // CHECK: [[SHOTS:%.+]] = "test.op"() : () -> i64
        %shots = "test.op"() : () -> i64

        // No auto qubit management
        // CHECK: quantum.device shots([[SHOTS]]) ["foo", "bar", "baz"]
        quantum.device shots(%shots) ["foo", "bar", "baz"]

        // Auto qubit management
        // CHECK: quantum.device shots([[SHOTS]]) ["foo", "bar", "baz"] {auto_qubit_management}
        quantum.device shots(%shots) ["foo", "bar", "baz"] {auto_qubit_management}

        //////////// **DeviceReleaseOp tests** ////////////
        // CHECK: quantum.device_release
        quantum.device_release

        //////////// **FinalizeOp tests** ////////////
        // CHECK: quantum.finalize
        quantum.finalize

        //////////// **InitializeOp tests** ////////////
        // CHECK: quantum.init
        quantum.init

        //////////// **NumQubitsOp tests** ////////////
        // CHECK: quantum.num_qubits : i64
        %nqubits = quantum.num_qubits : i64
        """

        run_filecheck(program, roundtrip=True, verify=True, pretty_print=pretty_print)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
