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

"""This module contains JAX-compatible quantum primitives to support the lowering
of quantum operations, measurements, and observables to reference semantics JAXPR.
"""

import jax
import numpy as np
import pennylane as qml
from jax._src.lib.mlir import ir
from jax.core import AbstractValue
from jax.extend.core import Primitive
from jax.interpreters import mlir
from jaxlib.mlir._mlir_libs import _mlir as _ods_cext
from jaxlib.mlir.dialects.arith import (
    ExtUIOp,
)

# TODO: remove after jax v0.7.2 upgrade
# Mock _ods_cext.globals.register_traceback_file_exclusion due to API conflicts between
# Catalyst's MLIR version and the MLIR version used by JAX. The current JAX version has not
# yet updated to the latest MLIR, causing compatibility issues. This workaround will be removed
# once JAX updates to a compatible MLIR version
# pylint: disable=ungrouped-imports
from catalyst.jax_extras.patches import mock_attributes
from catalyst.jax_primitives import AbstractObs, extract_scalar
from catalyst.utils.patching import Patcher

with Patcher(
    (
        _ods_cext,
        "globals",
        mock_attributes(
            # pylint: disable=c-extension-no-member
            _ods_cext.globals,
            {"register_traceback_file_exclusion": lambda x: None},
        ),
    ),
):
    from mlir_quantum.dialects.qref import (
        AllocOp,
        ComputationalBasisOp,
        DeallocOp,
        GetOp,
    )


#########
# Types #
#########


#
# qubit
#
class QrefQubit(AbstractValue):
    """Abstract Qubit"""

    hash_value = hash("QrefQubit")

    def __eq__(self, other):
        return isinstance(other, QrefQubit)

    def __hash__(self):
        return self.hash_value


def _qref_qubit_lowering(aval):
    assert isinstance(aval, QrefQubit)
    return ir.OpaqueType.get("qref", "bit")


#
# qreg
#
class QrefQreg(AbstractValue):
    """Abstract quantum register."""

    def __init__(self, num_qubits=None):
        self.num_qubits = num_qubits

        if num_qubits is None:
            add_hash = 0
        else:
            add_hash = hash(num_qubits)
        self.hash_value = hash("QrefQreg") + add_hash

    def __eq__(self, other):
        return isinstance(other, QrefQreg) and self.hash_value == other.hash_value

    def __hash__(self):
        return self.hash_value


def _qref_qreg_lowering(aval):
    assert isinstance(aval, QrefQreg)
    if aval.num_qubits is None:
        tag = "?"
    else:
        tag = str(aval.num_qubits)
    return ir.OpaqueType.get("qref", "reg<" + tag + ">")


#
# registration
#
mlir.ir_type_handlers[QrefQubit] = _qref_qubit_lowering
mlir.ir_type_handlers[QrefQreg] = _qref_qreg_lowering


##############
# Primitives #
##############

qref_alloc_p = Primitive("qref_alloc")
qref_dealloc_p = Primitive("qref_dealloc")
qref_dealloc_p.multiple_results = True
qref_get_p = Primitive("qref_get")
qref_compbasis_p = Primitive("compbasis")


#
# qref_alloc_p
#
@qref_alloc_p.def_abstract_eval
def _qref_alloc_abstract_eval(num_qubits=None):
    return QrefQreg(num_qubits)


def _qref_alloc_lowering(jax_ctx: mlir.LoweringRuleContext, size_value: ir.Value):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    if isinstance(size_value.owner, ir.Operation) and size_value.owner.name == "stablehlo.constant":
        size_value_attr = size_value.owner.attributes["value"]
        assert ir.DenseIntElementsAttr.isinstance(size_value_attr)
        size = ir.DenseIntElementsAttr(size_value_attr)[0]
        assert size >= 0

        size_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(64, ctx), size)
        qreg_type = ir.OpaqueType.get("qref", "reg<" + str(size) + ">", ctx)
        return AllocOp(qreg_type, nqubits_attr=size_attr).results
    else:
        size_value = extract_scalar(size_value, "qref_alloc")
        qreg_type = ir.OpaqueType.get("qref", "reg<?>", ctx)
        return AllocOp(qreg_type, nqubits=size_value).results


#
# qref_dealloc_p
#
@qref_dealloc_p.def_abstract_eval
def _qref_dealloc_abstract_eval(qreg):
    return ()


def _qref_dealloc_lowering(jax_ctx: mlir.LoweringRuleContext, qreg):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True
    DeallocOp(qreg)
    return ()


#
# qref_get_p
#
@qref_get_p.def_abstract_eval
def _qref_get_abstract_eval(qreg, qubit_idx):
    assert isinstance(qreg, QrefQreg), f"Expected QrefQreg, got {qreg}"
    return QrefQubit()


def _qref_get_lowering(jax_ctx: mlir.LoweringRuleContext, qreg: ir.Value, qubit_idx: ir.Value):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert ir.OpaqueType.isinstance(qreg.type), qreg.type
    assert ir.OpaqueType(qreg.type).dialect_namespace == "qref"
    assert "reg" in ir.OpaqueType(qreg.type).data

    qubit_idx = extract_scalar(qubit_idx, "wires", "index")
    if not ir.IntegerType.isinstance(qubit_idx.type):
        raise TypeError(f"Operator wires expected to be integers, got {qubit_idx.type}!")

    if ir.IntegerType(qubit_idx.type).width < 64:
        qubit_idx = ExtUIOp(ir.IntegerType.get_signless(64), qubit_idx).result
    elif not ir.IntegerType(qubit_idx.type).width == 64:
        raise TypeError(f"Operator wires expected to be 64-bit integers, got {qubit_idx.type}!")

    qubit_type = ir.OpaqueType.get("qref", "bit", ctx)
    return GetOp(qubit_type, qreg, idx=qubit_idx).results


#
# compbasis observable
#
@qref_compbasis_p.def_abstract_eval
def _compbasis_abstract_eval(*qubits_or_qreg, qreg_available=False):
    if qreg_available:
        qreg = qubits_or_qreg[0]
        assert isinstance(qreg, QrefQreg)
        return AbstractObs(qreg, qref_compbasis_p)
    else:
        qubits = qubits_or_qreg
        for qubit in qubits:
            assert isinstance(qubit, QrefQubit)
        return AbstractObs(len(qubits), qref_compbasis_p)


def _qref_compbasis_lowering(
    jax_ctx: mlir.LoweringRuleContext, *qubits_or_qreg: tuple, qreg_available=False
):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    result_type = ir.OpaqueType.get("quantum", "obs")

    if qreg_available:
        qreg = qubits_or_qreg[0]
        assert ir.OpaqueType.isinstance(qreg.type)
        assert ir.OpaqueType(qreg.type).dialect_namespace == "qref"
        assert "reg" in ir.OpaqueType(qreg.type).data
        return ComputationalBasisOp(result_type, [], qreg=qreg).results

    else:
        qubits = qubits_or_qreg
        for qubit in qubits:
            assert ir.OpaqueType.isinstance(qubit.type)
            assert ir.OpaqueType(qubit.type).dialect_namespace == "qref"
            assert ir.OpaqueType(qubit.type).data == "bit"

        return ComputationalBasisOp(result_type, qubits).results


CUSTOM_LOWERING_RULES = (
    (qref_alloc_p, _qref_alloc_lowering),
    (qref_dealloc_p, _qref_dealloc_lowering),
    (qref_get_p, _qref_get_lowering),
    (qref_compbasis_p, _qref_compbasis_lowering),
)
