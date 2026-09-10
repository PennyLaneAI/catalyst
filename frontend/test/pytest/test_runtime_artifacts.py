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

"""Tests for the ``catalyst.runtime_artifacts`` module attribute: the libraries a local
``runtime_call`` records, and their collection into the compile options."""

import jax.numpy as jnp
import pennylane as qp
from jax._src.lib.mlir import ir

from catalyst import qjit
from catalyst.pipelines import CompileOptions
from catalyst.utils.runtime_artifacts import collect_runtime_artifacts

qp.runtime_declare("test_artifacts_symbol", "(ptr, u32) -> u64")

LIB_A = "/tmp/libtest_artifact_a.so"
LIB_B = "/tmp/libtest_artifact_b.so"


def lower(fn, args):
    """Run the capture and lowering stages of ``fn`` and return the artifacts they collected."""
    compiled = qjit(fn)
    compiled.jaxpr, *_ = compiled.capture(args)
    compiled.generate_ir()
    return compiled.compile_options.runtime_artifacts


def test_local_call_library_is_collected():
    """The library of a local ``runtime_call`` reaches the compile options, named once."""

    def program(session):
        first = qp.runtime_call("test_artifacts_symbol", session, 100, library=LIB_A)
        second = qp.runtime_call("test_artifacts_symbol", session, 200, library=LIB_A)
        return first + second

    assert lower(program, (jnp.uint64(0),)) == (LIB_A,)


def test_library_recorded_inside_a_qnode_is_collected():
    """A call inside a qnode records onto a nested module, which the walk still reaches."""

    @qp.qnode(qp.device("null.qubit", wires=1))
    def circuit(session):
        rounds = qp.runtime_call("test_artifacts_symbol", session, 100, library=LIB_B)
        qp.RX(jnp.float64(rounds) * 0.0, wires=0)
        return qp.expval(qp.PauliZ(0))

    assert lower(circuit, (jnp.uint64(0),)) == (LIB_B,)


def collect_from(mlir_text):
    """Collect the artifacts of a hand-written module, whose ops need not be registered ones."""
    context = ir.Context()
    context.allow_unregistered_dialects = True
    options = CompileOptions()
    with context, ir.Location.unknown():
        collect_runtime_artifacts(ir.Module.parse(mlir_text), options)
    return options.runtime_artifacts


def test_nested_modules_are_collected_and_deduplicated():
    """Every module is reached at any depth, and a library named by two of them appears once."""

    collected = collect_from("""
        module attributes {catalyst.runtime_artifacts = ["/outer.so", "/shared.so"]} {
          module attributes {catalyst.runtime_artifacts = ["/shared.so", "/inner.so"]} {
            module attributes {catalyst.runtime_artifacts = ["/deepest.so"]} { }
          }
        }
        """)

    assert collected == ("/outer.so", "/shared.so", "/inner.so", "/deepest.so")


def test_only_module_operations_are_consulted():
    """The attribute is read off modules alone, and no other operation is traversed."""

    collected = collect_from("""
        module attributes {catalyst.runtime_artifacts = ["/outer.so"]} {
          "some.holder"() ({
            module attributes {catalyst.runtime_artifacts = ["/under_an_op.so"]} { }
            "some.terminator"() : () -> ()
          }) {catalyst.runtime_artifacts = ["/on_an_op.so"]} : () -> ()
        }
        """)

    assert collected == ("/outer.so",)
