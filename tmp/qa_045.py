import pennylane as qp
from pennylane.drawer.label import LabelledOp
from pennylane.fourier.mark import MarkedOp, mark

base = qp.PauliX(15)
op = LabelledOp(base, "one")
op2 = LabelledOp(base, "two")

assert op != op2

op = qp.X(0)
marked_op = mark(op, marker="my-x")
assert isinstance(marked_op, MarkedOp)
assert hasattr(marked_op, "marker")
assert marked_op.marker == "my-x"