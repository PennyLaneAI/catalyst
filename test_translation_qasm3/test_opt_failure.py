import subprocess
import sys
from pathlib import Path
import tempfile

here = Path(__file__).parent.resolve()
sys.path.append(str(here.parent))
sys.path.append(str(here.parent / "mlir/llvm-project/build/tools/mlir/python_packages/mlir_core"))

from qiskit import QuantumCircuit
from qiskit_importer_standalone import QiskitToCatalystImporter

qc = QuantumCircuit(1)
qc.h(0)
importer = QiskitToCatalystImporter(qc)
module = importer.convert()

with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as tmp:
    tmp.write(str(module))
    name = tmp.name

print("Original MLIR:")
print(module)

opt_cmd = ["../mlir/build/bin/quantum-opt", "--pass-pipeline=builtin.module(apply-transform-sequence,canonicalize,merge-rotations)", name]
res = subprocess.run(opt_cmd, capture_output=True, text=True)
print("OPT ERR:", res.stderr)
opt_out = res.stdout
print("OPT MLIR:")
print(opt_out)

with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as tmp2:
    tmp2.write(opt_out)
    name2 = tmp2.name

trans_cmd = ["../mlir/build/bin/quantum-translate", "--mlir-to-qasm3", name2]
res2 = subprocess.run(trans_cmd, capture_output=True, text=True)
print("TRANS OUT:", res2.stdout)
print("TRANS ERR:", res2.stderr)
