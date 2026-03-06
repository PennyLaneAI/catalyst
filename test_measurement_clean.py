
import sys
import os
import unittest.mock

if sys.version_info < (3, 13):
    print("WARNING: This script requires Python 3.13 to match the compiled MLIR bindings.")
    print(f"Current version: {sys.version}")
    print("Please run with: /Users/user_name/miniconda3/bin/python3")
    # We don't exit, just warn, in case they magically have compatible bindings? 
    # But usually it fails.


# Mock jax and jaxlib to prevent interference with mlir package
sys.modules["jax"] = unittest.mock.MagicMock()
sys.modules["jaxlib"] = unittest.mock.MagicMock()
sys.modules["jaxlib"].__version__ = "0.7.1"
sys.modules["jaxlib.mlir"] = unittest.mock.MagicMock()
sys.modules["jaxlib.mlir.ir"] = unittest.mock.MagicMock()
sys.modules["jaxlib.mlir._mlir_libs"] = unittest.mock.MagicMock()

import qiskit

# Add paths
sys.path.append(os.path.abspath("frontend"))
sys.path.append(os.path.abspath("mlir/llvm-project/build/tools/mlir/python_packages/mlir_core"))
sys.path.append(os.path.abspath("mlir/build/python_packages/quantum"))

try:
    from qiskit_importer_standalone import QiskitToCatalystImporter
except ImportError as e:
    print(f"Error importing standalone importer: {e}")
    sys.exit(1)

def test_measurement():
    print("1. Creating Qiskit Circuit...")
    qc = qiskit.QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    print(qc)

    print("\n2. converting to Catalyst MLIR...")
    try:
        importer = QiskitToCatalystImporter(qc)
        module = importer.convert()
        mlir_str = str(module)
        print("MLIR Generation Successful:")
        print(mlir_str)
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    mlir_filename = "test_measurement.mlir"
    with open(mlir_filename, "w") as f:
        f.write(mlir_str)

    print(f"\n3. Running quantum-translate on {mlir_filename}...")
    translate_tool = os.path.abspath("mlir/build/bin/quantum-translate")
    
    if not os.path.exists(translate_tool):
        # Fallback to checking location
        pass

    import subprocess
    result = subprocess.run(
        [translate_tool, mlir_filename, "--mlir-to-qasm3"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("Translation Failed:")
        print(result.stderr)
        sys.exit(1)
    
    print("\n4. Translation Output (OpenQASM 3.0):")
    print(result.stdout)
    
    # Verification
    if "measure" in result.stdout and "bit" in result.stdout:
        print("\nSUCCESS: Measurement translation verified.")
    else:
        print("\nFAILURE: Measurement instruction not found in output.")
        sys.exit(1)

if __name__ == "__main__":
    test_measurement()
