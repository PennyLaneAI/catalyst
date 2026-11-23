from functools import reduce

import pennylane as qp

dev = qp.device("default.qubit")

circuit = qp.tape.qscript.QuantumScript(
    [
        qp.CNOT(["(0, 1)", "(1, 1)"]),
        qp.H("(0, 1)"),
        qp.X("(1, 0)"),
        qp.CNOT(["(1, 1)", "(1, 0)"]),
        qp.H("(0, 1)"),
        qp.CNOT(["(1, 0)", "(0, 0)"]),
        qp.X("(1, 1)"),
        qp.CNOT(["(1, 1)", "(1, 0)"]),
    ]
)

circuit = qp.tape.qscript.QuantumScript([qp.CZ((0, 1)), qp.CZ((0, 2))])

circuit = qp.tape.qscript.QuantumScript(
    [
        qp.CZ((0, 1)),
        qp.H(2),
        qp.CZ((0, 2)),
        qp.H(1),
        qp.H(2),
    ]
)


def greedy_schedule(circuit):
    dag = qp.commutation_dag(circuit)
    nodes = dag.get_nodes()

    time = 0
    moments = []
    scheduled_ids = []

    while len(scheduled_ids) < len(nodes):
        moments.append([])
        moment_ids = []

        # check if we can schedule each node
        for i, node in nodes:
            # check if we already scheduled this node
            if node.node_id in scheduled_ids:
                continue

            # check that there is no predecessor that has not been scheduled already (before this moment)
            dependent = False
            for predecessor_id in node.predecessors:
                if predecessor_id not in scheduled_ids:
                    dependent = True
            if dependent:
                continue

            # check that the wires are available in the current moment
            moment_wires = []
            if moments[time]:
                moment_wires += reduce(
                    lambda acc, next: acc + next, [n.wires for n in moments[time]]
                )

            found = False
            for wire in node.wires:
                if wire in moment_wires:
                    found = True
            if found:
                continue  # we can't schedule it due to a wire collision

            # schedule the op in the current time slice
            moments[time].append(node.op)
            moment_ids.append(node.node_id)

        time += 1
        scheduled_ids += moment_ids

    return moments


schedule = greedy_schedule(circuit)
