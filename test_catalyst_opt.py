import os
import sys
from pathlib import Path
script_dir = Path(__file__).parent.resolve()
root_dir = script_dir
sys.path.append(str(root_dir / "mlir" / "llvm-project" / "build" / "tools" / "mlir" / "python_packages" / "mlir_core"))
from qiskit import QuantumCircuit
from qiskit_importer_standalone import QiskitToCatalystImporter
qc = QuantumCircuit(1)
qc.h(0)
importer = QiskitToCatalystImporter(qc)
module = importer.convert()

from catalyst.compiler import to_mlir_opt
from catalyst.pipelines import CompileOptions

print("Original MLIR:", module)
out = to_mlir_opt(stdin=str(module))
print("Optimized MLIR:", out)
