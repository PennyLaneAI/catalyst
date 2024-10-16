import jax
import numpy as np
import pennylane as qml

import catalyst
from catalyst import qjit
from catalyst.debug import instrumentation

import os

dev = qml.device("lightning.qubit", wires=3)

peephole_pipeline = {"cancel_inverses":{}, "merge_rotations":{}}


# @qjit(seed=37, keep_intermediate=True)
#@qjit(keep_intermediate=True)
"""
@qjit(pipelines = [
    ("0_canonicalize", ["canonicalize"]),
    ("peephole",
    [
    "builtin.module(remove-chained-self-inverse{func-name=circuit})",
    "builtin.module(merge-rotations{func-name=circuit})"
    ]
    ), # peephole
    ("inline_nested_module", ["inline-nested-module"]),
    ], # pipelines =
    #circuit_transform_pipeline = peephole_pipeline,
    autograph=True,
    keep_intermediate=True)
"""
@qjit(autograph=True,
    keep_intermediate=True,
    #circuit_transform_pipeline = peephole_pipeline,
    )
@qml.qnode(dev)
def circuit(theta, loop_size):
    for i in range(loop_size):
        qml.Hadamard(0)
        qml.Hadamard(0)
        qml.RX(theta, wires=1)
        qml.RX(-theta, wires=1)
    return qml.probs()


os.remove("my_toy_circuit.yml")
with instrumentation("my_toy_circuit", filename="my_toy_circuit.yml", detailed=True):
    res = circuit(12.3, 10000000)
    print(res)

with open('my_toy_circuit.yml', 'r') as f:
    print(f.read())

