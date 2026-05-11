"""
Randomized Circuit Benchmarking for QASM 3 Translation Pipeline

Benchmarks the Qiskit -> MLIR -> quantum-opt -> QASM3 translation pipeline
using randomly generated circuits of varying sizes and complexities.

Usage:
    python benchmark_random_circuits.py --num-qubits 5 --depth 10 --count 20
    python benchmark_random_circuits.py --circuit-type qv --num-qubits 4 --count 10
    python benchmark_random_circuits.py --batch-mode --output results.json
"""

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
script_dir = Path(__file__).parent.resolve()
root_dir = script_dir.parent.parent
sys.path.insert(0, str(script_dir))
sys.path.append(str(root_dir))

# Setup MLIR paths
mlir_core_path = (
    root_dir
    / "mlir"
    / "llvm-project"
    / "build"
    / "tools"
    / "mlir"
    / "python_packages"
    / "mlir_core"
)
if mlir_core_path.exists():
    sys.path.append(str(mlir_core_path))

from qiskit import QuantumCircuit
from random_circuit_generator import RandomCircuitConfig, RandomCircuitGenerator

try:
    from qiskit_importer_standalone import QiskitToCatalystImporter
except ImportError as e:
    print(f"Error: Could not import QiskitToCatalystImporter: {e}")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Results from benchmarking a single circuit."""

    circuit_name: str
    num_qubits: int
    circuit_depth: int
    gate_count: int

    # Timing metrics (seconds)
    mlir_generation_time: float
    optimization_time: float
    translation_time: float
    total_time: float

    # Size metrics
    mlir_lines: int
    mlir_ops: int
    qasm3_lines: int
    qasm3_gates: int

    # Status
    success: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AggregatedResults:
    """Aggregated statistics from multiple benchmark runs."""

    total_circuits: int
    successful: int
    failed: int
    success_rate: float

    avg_total_time: float
    min_total_time: float
    max_total_time: float
    median_total_time: float

    avg_mlir_generation_time: float
    avg_optimization_time: float
    avg_translation_time: float

    avg_gate_count: float
    avg_qasm3_gates: float
    optimization_effectiveness: float  # Percentage of gates reduced

    results: List[BenchmarkResult]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert BenchmarkResult objects to dicts
        data["results"] = [r.to_dict() if hasattr(r, "to_dict") else r for r in self.results]
        return data


class QASM3TranslationBenchmark:
    """Benchmark runner for QASM3 translation pipeline."""

    def __init__(self, verbose: bool = False):
        """
        Initialize benchmark runner.

        Args:
            verbose: Whether to print detailed output
        """
        self.verbose = verbose
        self.root_dir = root_dir
        self.quantum_opt_path = root_dir / "mlir" / "build" / "bin" / "quantum-opt"
        self.quantum_translate_path = root_dir / "mlir" / "build" / "bin" / "quantum-translate"

        # Verify tools exist
        if not self.quantum_opt_path.exists():
            raise FileNotFoundError(f"quantum-opt not found at {self.quantum_opt_path}")
        if not self.quantum_translate_path.exists():
            raise FileNotFoundError(f"quantum-translate not found at {self.quantum_translate_path}")

    def benchmark_circuit(self, circuit: QuantumCircuit, circuit_name: str) -> BenchmarkResult:
        """
        Benchmark a single circuit through the full translation pipeline.

        Args:
            circuit: Qiskit QuantumCircuit to benchmark
            circuit_name: Name identifier for the circuit

        Returns:
            BenchmarkResult with timing and size metrics
        """
        result = BenchmarkResult(
            circuit_name=circuit_name,
            num_qubits=circuit.num_qubits,
            circuit_depth=circuit.depth(),
            gate_count=sum(
                1 for inst in circuit.data if inst.operation.name not in ["measure", "barrier"]
            ),
            mlir_generation_time=0.0,
            optimization_time=0.0,
            translation_time=0.0,
            total_time=0.0,
            mlir_lines=0,
            mlir_ops=0,
            qasm3_lines=0,
            qasm3_gates=0,
            success=False,
        )

        try:
            # Phase 1: Qiskit -> MLIR
            start_time = time.time()
            importer = QiskitToCatalystImporter(circuit)
            module = importer.convert()
            mlir_code = str(module)
            result.mlir_generation_time = time.time() - start_time

            # Count MLIR metrics
            result.mlir_lines = len(mlir_code.split("\n"))
            result.mlir_ops = mlir_code.count("quantum.")

            if self.verbose:
                print(f"  MLIR generation: {result.mlir_generation_time:.3f}s")

            # Phase 2: quantum-opt (optimization)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp_mlir:
                tmp_mlir.write(mlir_code)
                tmp_mlir_path = tmp_mlir.name

            start_time = time.time()
            opt_cmd = [
                str(self.quantum_opt_path),
                "--pass-pipeline=builtin.module(apply-transform-sequence, canonicalize, merge-rotations)",
                tmp_mlir_path,
                "-o",
                tmp_mlir_path,
            ]
            subprocess.run(opt_cmd, capture_output=True, text=True, check=True)
            result.optimization_time = time.time() - start_time

            if self.verbose:
                print(f"  Optimization: {result.optimization_time:.3f}s")

            # Phase 3: MLIR -> QASM3
            start_time = time.time()
            translate_cmd = [str(self.quantum_translate_path), "--mlir-to-qasm3", tmp_mlir_path]
            translate_result = subprocess.run(
                translate_cmd, capture_output=True, text=True, check=True
            )
            qasm3_code = translate_result.stdout
            result.translation_time = time.time() - start_time

            # Cleanup temp file
            Path(tmp_mlir_path).unlink()

            # Count QASM3 metrics
            result.qasm3_lines = len(qasm3_code.split("\n"))
            result.qasm3_gates = sum(
                1
                for line in qasm3_code.split("\n")
                if line.strip()
                and not line.strip().startswith("//")
                and not line.strip().startswith("OPENQASM")
                and not line.strip().startswith("include")
                and not line.strip().startswith("qubit")
                and not line.strip().startswith("bit")
                and "measure" not in line.lower()
            )

            if self.verbose:
                print(f"  Translation: {result.translation_time:.3f}s")

            # Calculate total time
            result.total_time = (
                result.mlir_generation_time + result.optimization_time + result.translation_time
            )
            result.success = True

            if self.verbose:
                print(f"  Total time: {result.total_time:.3f}s")
                print(f"  Gates: {result.gate_count} -> {result.qasm3_gates}")

        except subprocess.CalledProcessError as e:
            result.error_message = f"Subprocess error: {e.stderr}"
            if self.verbose:
                print(f"  Error: {result.error_message}")

        except Exception as e:
            result.error_message = str(e)
            if self.verbose:
                print(f"  Error: {result.error_message}")

        return result

    def benchmark_batch(
        self, circuits: List[QuantumCircuit], prefix: str = "circuit"
    ) -> List[BenchmarkResult]:
        """
        Benchmark a batch of circuits.

        Args:
            circuits: List of circuits to benchmark
            prefix: Name prefix for circuits

        Returns:
            List of BenchmarkResults
        """
        results = []

        for i, circuit in enumerate(circuits):
            circuit_name = f"{prefix}_{i:04d}"
            if self.verbose:
                print(f"\nBenchmarking {circuit_name} ({i+1}/{len(circuits)})...")

            result = self.benchmark_circuit(circuit, circuit_name)
            results.append(result)

        return results

    def aggregate_results(self, results: List[BenchmarkResult]) -> AggregatedResults:
        """
        Aggregate statistics from multiple benchmark results.

        Args:
            results: List of individual benchmark results

        Returns:
            AggregatedResults with statistics
        """
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        total = len(results)
        success_count = len(successful_results)
        fail_count = len(failed_results)

        if success_count == 0:
            # No successful results
            return AggregatedResults(
                total_circuits=total,
                successful=success_count,
                failed=fail_count,
                success_rate=0.0,
                avg_total_time=0.0,
                min_total_time=0.0,
                max_total_time=0.0,
                median_total_time=0.0,
                avg_mlir_generation_time=0.0,
                avg_optimization_time=0.0,
                avg_translation_time=0.0,
                avg_gate_count=0.0,
                avg_qasm3_gates=0.0,
                optimization_effectiveness=0.0,
                results=results,
            )

        # Calculate statistics from successful results
        total_times = [r.total_time for r in successful_results]
        mlir_times = [r.mlir_generation_time for r in successful_results]
        opt_times = [r.optimization_time for r in successful_results]
        trans_times = [r.translation_time for r in successful_results]

        gate_counts = [r.gate_count for r in successful_results]
        qasm3_gates = [r.qasm3_gates for r in successful_results]

        # Calculate optimization effectiveness
        total_input_gates = sum(gate_counts)
        total_output_gates = sum(qasm3_gates)
        opt_effectiveness = (
            (total_input_gates - total_output_gates) / total_input_gates * 100
            if total_input_gates > 0
            else 0.0
        )

        return AggregatedResults(
            total_circuits=total,
            successful=success_count,
            failed=fail_count,
            success_rate=success_count / total * 100,
            avg_total_time=statistics.mean(total_times),
            min_total_time=min(total_times),
            max_total_time=max(total_times),
            median_total_time=statistics.median(total_times),
            avg_mlir_generation_time=statistics.mean(mlir_times),
            avg_optimization_time=statistics.mean(opt_times),
            avg_translation_time=statistics.mean(trans_times),
            avg_gate_count=statistics.mean(gate_counts),
            avg_qasm3_gates=statistics.mean(qasm3_gates),
            optimization_effectiveness=opt_effectiveness,
            results=results,
        )


def print_summary(aggregated: AggregatedResults) -> None:
    """Print a summary of benchmark results."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total circuits:        {aggregated.total_circuits}")
    print(f"Successful:            {aggregated.successful} ({aggregated.success_rate:.1f}%)")
    print(f"Failed:                {aggregated.failed}")
    print()
    print("Timing Statistics (successful circuits):")
    print(f"  Average total time:  {aggregated.avg_total_time:.3f}s")
    print(f"  Min time:            {aggregated.min_total_time:.3f}s")
    print(f"  Max time:            {aggregated.max_total_time:.3f}s")
    print(f"  Median time:         {aggregated.median_total_time:.3f}s")
    print()
    print("Pipeline Breakdown (average):")
    print(f"  MLIR generation:     {aggregated.avg_mlir_generation_time:.3f}s")
    print(f"  Optimization:        {aggregated.avg_optimization_time:.3f}s")
    print(f"  Translation:         {aggregated.avg_translation_time:.3f}s")
    print()
    print("Circuit Statistics:")
    print(f"  Avg input gates:     {aggregated.avg_gate_count:.1f}")
    print(f"  Avg output gates:    {aggregated.avg_qasm3_gates:.1f}")
    print(f"  Optimization effect: {aggregated.optimization_effectiveness:.1f}% reduction")
    print("=" * 70)


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark QASM3 translation pipeline with random circuits"
    )

    parser.add_argument(
        "--num-qubits", "-n", type=int, default=4, help="Number of qubits (default: 4)"
    )
    parser.add_argument("--depth", "-d", type=int, default=10, help="Circuit depth (default: 10)")
    parser.add_argument(
        "--count", "-c", type=int, default=10, help="Number of circuits to generate (default: 10)"
    )
    parser.add_argument(
        "--circuit-type",
        "-t",
        choices=["standard", "qv", "clifford", "custom"],
        default="standard",
        help="Type of random circuit (default: standard)",
    )
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--batch-mode", action="store_true", help="Run predefined batch of different circuit sizes"
    )

    args = parser.parse_args()

    # Initialize generator and benchmark
    generator = RandomCircuitGenerator(seed=args.seed)
    benchmark = QASM3TranslationBenchmark(verbose=args.verbose)

    print(f"QASM3 Translation Pipeline Benchmark")
    print(f"{'='*70}")

    if args.batch_mode:
        # Run comprehensive batch
        print("Running batch mode with multiple circuit sizes...")
        all_results = []

        batch_configs = [
            (2, 5, "standard", 5),  # 2 qubits, depth 5
            (3, 10, "standard", 5),  # 3 qubits, depth 10
            (4, 10, "standard", 5),  # 4 qubits, depth 10
            (5, 15, "standard", 5),  # 5 qubits, depth 15
            (3, 10, "clifford", 5),  # Clifford circuits
            (4, 4, "qv", 5),  # Quantum volume
        ]

        for num_qubits, depth, circuit_type, count in batch_configs:
            print(f"\n--- Testing {circuit_type} circuits: {num_qubits} qubits, depth {depth} ---")

            config = RandomCircuitConfig(
                num_qubits=num_qubits, depth=depth, measure=True, seed=args.seed
            )

            circuits = generator.generate_batch(config, count, circuit_type)
            results = benchmark.benchmark_batch(
                circuits, prefix=f"{circuit_type}_{num_qubits}q_{depth}d"
            )
            all_results.extend(results)

        aggregated = benchmark.aggregate_results(all_results)

    else:
        # Run single configuration
        print(f"Circuit type: {args.circuit_type}")
        print(f"Qubits: {args.num_qubits}, Depth: {args.depth}, Count: {args.count}")
        print(f"Seed: {args.seed}")

        config = RandomCircuitConfig(
            num_qubits=args.num_qubits, depth=args.depth, measure=True, seed=args.seed
        )

        circuits = generator.generate_batch(config, args.count, args.circuit_type)
        results = benchmark.benchmark_batch(circuits, prefix=args.circuit_type)
        aggregated = benchmark.aggregate_results(results)

    # Print summary
    print_summary(aggregated)

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(aggregated.to_dict(), f, indent=2)

        print(f"\nResults saved to: {output_path}")

    # Exit with error code if any failures
    sys.exit(0 if aggregated.failed == 0 else 1)


if __name__ == "__main__":
    main()
