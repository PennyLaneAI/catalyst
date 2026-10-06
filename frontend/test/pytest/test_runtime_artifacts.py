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
``runtime_call`` records, the libraries a device records through ``get_runtime_artifacts()``,
and their collection into the compile options."""

import re

import jax.numpy as jnp
import pennylane as qp
import pytest
from jax._src.lib.mlir import ir

from catalyst import qjit
from catalyst.pipelines import CompileOptions
from catalyst.utils.exceptions import CompileError
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


def device_with_artifacts(*paths):
    """A ``null.qubit`` device declaring ``paths`` through ``get_runtime_artifacts()``."""
    dev = qp.device("null.qubit", wires=1)
    dev.get_runtime_artifacts = lambda: paths
    return dev


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


def test_device_runtime_artifacts_are_collected(tmp_path):
    """Libraries a device declares reach the compile options, once each, in declaration order."""
    first = tmp_path / "first.so"
    second = tmp_path / "second.so"
    first.write_bytes(b"")
    second.write_bytes(b"")

    @qp.qnode(device_with_artifacts(first, second, first))
    def circuit():
        return qp.expval(qp.PauliZ(0))

    assert lower(circuit, ()) == (str(first), str(second))

def test_missing_device_runtime_artifact_is_rejected(tmp_path):
    """A declared path that is not a file fails compilation and names the device and the path."""
    missing = tmp_path / "missing.so"

    @qp.qnode(device_with_artifacts(missing))
    def circuit():
        return qp.expval(qp.PauliZ(0))

    message = f"Device 'null.qubit' declares runtime artifact '{missing}', which does not exist."
    with pytest.raises(CompileError, match=re.escape(message)):
        lower(circuit, ())


def test_device_and_call_libraries_are_both_collected(tmp_path):
    """The device library is recorded on the enclosing module, ahead of the nested call's library."""
    device_lib = tmp_path / "device.so"
    device_lib.write_bytes(b"")

    @qp.qnode(device_with_artifacts(device_lib))
    def circuit(session):
        rounds = qp.runtime_call("test_artifacts_symbol", session, 100, library=LIB_A)
        qp.RX(jnp.float64(rounds) * 0.0, wires=0)
        return qp.expval(qp.PauliZ(0))

    assert lower(circuit, (jnp.uint64(0),)) == (str(device_lib), LIB_A)


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
