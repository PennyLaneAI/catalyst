import subprocess
import sys
from pathlib import Path
import tempfile

here = Path(__file__).parent.resolve()
sys.path.append(str(here.parent))
sys.path.append(str(here.parent / "mlir/llvm-project/build/tools/mlir/python_packages/mlir_core"))

from qiskit import QuantumCircuit
from qiskit_importer_standalone import QiskitToCatalystImporter

qc = QuantumCircuit.from_qasm_file("qasm3_circuits/mid_measurement.qasm")
importer = QiskitToCatalystImporter(qc)
module = importer.convert()

with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
    tmp.write(str(module))
    name = tmp.name

opt_cmd = [
    "../mlir/build/bin/quantum-opt",
    "--pass-pipeline=builtin.module(apply-transform-sequence,canonicalize,merge-rotations)",
    name,
]
res = subprocess.run(opt_cmd, capture_output=True, text=True)
opt_out = res.stdout
print("OPT MLIR:")
print(opt_out)

with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp2:
    tmp2.write(opt_out)
    name2 = tmp2.name

trans_cmd = ["../mlir/build/bin/quantum-translate", "--mlir-to-qasm3", name2]
res2 = subprocess.run(trans_cmd, capture_output=True, text=True)
if res2.returncode != 0:
    print("TRANS ERR: \n", res2.stderr)
else:
    print("SUCCESS")
