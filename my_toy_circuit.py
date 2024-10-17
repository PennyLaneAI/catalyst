import datetime
import os
import sys

import jax
import numpy as np
import pennylane as qml

import catalyst
from catalyst import qjit
from catalyst.debug import instrumentation

dev = qml.device("lightning.qubit", wires=2)

peephole_pipeline = {"cancel_inverses": {}, "merge_rotations": {}}


# @qjit(seed=37, keep_intermediate=True)
# @qjit(keep_intermediate=True)
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


@qjit(
    autograph=True,
    keep_intermediate=False,
    # circuit_transform_pipeline = peephole_pipeline,
)
@qml.qnode(dev)
def circuit(theta, loop_size):
    for i in range(loop_size):
        # for j in range(loop_size):
        qml.Hadamard(0)
        qml.Hadamard(0)
        qml.RX(theta, wires=1)
        qml.RX(-theta, wires=1)
    return qml.probs()


num_of_iters = int(sys.argv[1:][0])
os.remove("my_toy_circuit.yml")
with instrumentation("my_toy_circuit", filename="my_toy_circuit.yml", detailed=True):
    res = circuit(12.3, num_of_iters)
    # print(res)

# with open('my_toy_circuit.yml', 'r') as f:
#    print(f.read())


####################### core PL #######################


@qml.qnode(dev)
def circuit_corePL(theta, loop_size):
    for i in range(loop_size):
        # for j in range(loop_size):
        qml.Hadamard(0)
        qml.Hadamard(0)
        qml.RX(theta, wires=1)
        qml.RX(-theta, wires=1)
    return qml.probs()


from matplotlib.patches import Rectangle

draw = False
if draw:
    plt = qml.draw_mpl(circuit_corePL, show_all_wires=True, style="pennylane", label_options={"color": "white"})(1.23, 1)[0]
    plt.text(0.5, 0.06, "repeated $N$ times", fontsize=12, ha="center")
    square = Rectangle(
        (-0.52, -0.5), 1, 2, linestyle="--", linewidth=1, edgecolor="r", facecolor="none"
    )
    plt.gca().add_patch(square)
    plt.savefig("circuit_optimized")


(tape,), _ = qml.workflow.construct_batch(circuit_corePL, level=0)(12.3, num_of_iters)
start1 = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
_ = qml.transforms.cancel_inverses(tape)
end1 = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
start_time = float(start1[start1.rfind(":") + 1 :])
end_time = float(end1[end1.rfind(":") + 1 :])
elapsed_seconds = end_time - start_time
elapsed_ms1 = elapsed_seconds * 1e3


tape = _[0][0]  # <QuantumScript>
start2 = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
_ = qml.transforms.merge_rotations(tape)
end2 = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
start_time = float(start2[start2.rfind(":") + 1 :])
end_time = float(end2[end2.rfind(":") + 1 :])
elapsed_seconds = end_time - start_time
elapsed_ms2 = elapsed_seconds * 1e3

total_elapsed_ms = elapsed_ms1 + elapsed_ms2
with open("core_peephole_time.txt", "w") as f:
    print(total_elapsed_ms, file=f)


# res = circuit_corePL(12.3, num_of_iters)
# print(res)
