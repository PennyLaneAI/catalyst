import pennylane as qml
from pennylane.decomposition.gate_sets import ALL_QUBIT_OPS
from pennylane.ops.op_math.decompositions.controlled_decompositions import (
    decompose_mcx_many_workers_explicit,
    decompose_mcx_with_no_worker,
)

qml.decomposition.enable_graph()


@qml.transforms.decompose(
    gate_set=ALL_QUBIT_OPS,
    num_work_wires=100,
    fixed_decomps={qml.MultiControlledX: decompose_mcx_with_no_worker},
)
@qml.qnode(qml.device("null.qubit"))
def circuit():

    qml.MultiControlledX([0, 1, 2, 3])

    return qml.sample(wires=3)


qml.draw_mpl(circuit)()
