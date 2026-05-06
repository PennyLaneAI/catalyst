# Enhanced Qiskit Importer with improved error handling and diagnostics
# Extends qiskit_importer_standalone with better error messages and logging

import qiskit
import logging
from typing import Optional, List, Dict, Any

try:
    from mlir.dialects import scf, func, arith
    from mlir.ir import (
        Context,
        Module,
        Location,
        InsertionPoint,
        StringAttr,
        SymbolTable,
        Type,
        IntegerType,
        IndexType,
        IntegerAttr,
        FloatAttr,
        Operation,
        DenseI32ArrayAttr,
    )
except ImportError as e:
    raise ImportError(
        f"Could not import MLIR package: {e}\n"
        "Ensure PYTHONPATH includes mlir_core:\n"
        "  export PYTHONPATH=/path/to/mlir/llvm-project/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH"
    )

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class CompileError(Exception):
    """Exception raised for errors during compilation."""

    pass


class UnsupportedOperationError(CompileError):
    """Exception raised when encountering an unsupported quantum operation."""

    def __init__(self, operation: str, suggestion: Optional[str] = None):
        message = f"Unsupported operation: {operation}"
        if suggestion:
            message += f"\nSuggestion: {suggestion}"
        super().__init__(message)


class InvalidCircuitError(CompileError):
    """Exception raised when the circuit is invalid or malformed."""

    pass


class QiskitToCatalystImporter:
    """
    Imports a Qiskit QuantumCircuit into a Catalyst MLIR module.

    Enhanced version with:
    - Better error messages
    - Logging support
    - Input validation
    - Operation tracking
    """

    SUPPORTED_GATES = {
        "h",
        "x",
        "y",
        "z",
        "s",
        "t",
        "sdg",
        "tdg",
        "rx",
        "ry",
        "rz",
        "u1",
        "u2",
        "u3",
        "cx",
        "cz",
        "cy",
        "swap",
        "ccx",
        "cswap",
        "crx",
        "cry",
        "crz",
        "cu1",
        "cu2",
        "cu3",
        "cp",
        "measure",
        "barrier",
        "reset",
    }

    def __init__(self, circuit: qiskit.QuantumCircuit, enable_logging: bool = False):
        """
        Initialize the importer.

        Args:
            circuit: Qiskit QuantumCircuit to convert
            enable_logging: Enable detailed logging of conversion process
        """
        self.circuit = circuit
        self.enable_logging = enable_logging

        if enable_logging:
            logger.setLevel(logging.DEBUG)

        # Validate circuit
        self._validate_circuit()

        # Initialize MLIR context
        self.ctx = Context()
        self.ctx.allow_unregistered_dialects = True
        try:
            self.ctx.load_all_available_dialects()
        except AttributeError:
            logger.warning("Could not load all available dialects")

        self.module = Module.create(Location.unknown(self.ctx))
        self.qubit_map: Dict[qiskit.circuit.Qubit, Any] = {}
        self.clbit_map: Dict[qiskit.circuit.Clbit, Any] = {}
        self.global_qubit_reg = None

        # Statistics
        self.stats = {"gates_processed": 0, "measurements": 0, "control_flow": 0}

        logger.info(
            f"Initialized importer for circuit with {circuit.num_qubits} qubits, "
            f"{circuit.num_clbits} classical bits, {len(circuit.data)} operations"
        )

    def _validate_circuit(self):
        """Validate the input circuit."""
        if self.circuit.num_qubits == 0:
            raise InvalidCircuitError("Circuit must have at least one qubit")

        if self.circuit.num_qubits > 100:
            logger.warning(
                f"Large circuit with {self.circuit.num_qubits} qubits detected. "
                "Translation may be slow."
            )

        # Check for unsupported operations
        unsupported_ops = []
        for instruction in self.circuit.data:
            op_name = instruction.operation.name.lower()
            if (
                op_name not in self.SUPPORTED_GATES
                and not op_name.startswith("if_")
                and not op_name.startswith("for_")
            ):
                unsupported_ops.append(op_name)

        if unsupported_ops:
            unique_ops = set(unsupported_ops)
            logger.warning(f"Circuit contains potentially unsupported operations: {unique_ops}")

    def _emit_quantum_op(
        self,
        name: str,
        operands: List,
        result_types: List,
        loc: Location,
        attrs: Optional[Dict] = None,
    ) -> Operation:
        """
        Emit a quantum operation with error handling.

        Args:
            name: Operation name
            operands: List of operands
            result_types: List of result types
            loc: Location information
            attrs: Optional attributes dictionary

        Returns:
            Created MLIR operation

        Raises:
            CompileError: If operation creation fails
        """
        try:
            return Operation.create(
                name, results=result_types, operands=operands, attributes=attrs, loc=loc
            )
        except Exception as e:
            raise CompileError(
                f"Failed to create operation '{name}': {e}\n"
                f"Operands: {len(operands)}, Result types: {len(result_types)}"
            )

    def convert(self) -> Module:
        """
        Convert the stored Qiskit circuit to an MLIR module.

        Returns:
            MLIR Module representing the quantum circuit

        Raises:
            CompileError: If conversion fails
        """
        try:
            with InsertionPoint(self.module.body):
                loc = Location.unknown(self.ctx)
                func_type = func.FunctionType.get([], [], context=self.ctx)
                func_op = func.FuncOp("main", func_type, loc=loc)

                with InsertionPoint(func_op.add_entry_block()):
                    num_qubits = self.circuit.num_qubits

                    # Allocate quantum register
                    r_type = Type.parse("!quantum.reg", context=self.ctx)
                    idx_type = IntegerType.get_signless(64, self.ctx)

                    val_attr = IntegerAttr.get(idx_type, num_qubits)
                    n_qubits_val = arith.ConstantOp(idx_type, val_attr, loc=loc).result

                    reg = self._emit_quantum_op(
                        "quantum.alloc", [n_qubits_val], [r_type], loc
                    ).result

                    logger.debug(f"Allocated register with {num_qubits} qubits")

                    # Extract all qubits
                    for i, qubit in enumerate(self.circuit.qubits):
                        idx_attr = IntegerAttr.get(idx_type, i)
                        idx_val = arith.ConstantOp(idx_type, idx_attr, loc=loc).result
                        bit_type = Type.parse("!quantum.bit", context=self.ctx)
                        q_bit = self._emit_quantum_op(
                            "quantum.extract", [reg, idx_val], [bit_type], loc
                        ).result
                        self.qubit_map[qubit] = q_bit

                    logger.debug(f"Extracted {len(self.qubit_map)} qubits")

                    # Process instructions
                    self._process_instructions(self.circuit.data, loc)

                    func.ReturnOp([], loc=loc)

            logger.info(f"Conversion complete. Stats: {self.stats}")
            return self.module

        except Exception as e:
            if isinstance(e, CompileError):
                raise
            raise CompileError(f"Unexpected error during conversion: {e}")

    def _process_instructions(self, instructions: List, loc: Location):
        """Process circuit instructions with detailed error reporting."""
        for idx, instruction in enumerate(instructions):
            try:
                op = instruction.operation
                qubits = instruction.qubits

                logger.debug(f"Processing instruction {idx}: {op.name}")

                if op.name == "h":
                    self._emit_gate("h", qubits, [], loc)
                    self.stats["gates_processed"] += 1
                elif op.name == "cx":
                    self._emit_gate("cnot", qubits, [], loc)
                    self.stats["gates_processed"] += 1
                elif op.name == "for_loop":
                    self._emit_for_loop(op, qubits, loc)
                    self.stats["control_flow"] += 1
                elif op.name == "measure":
                    self._emit_measure(qubits, instruction.clbits, loc)
                    self.stats["measurements"] += 1
                elif op.name == "if_else" or type(op).__name__ == "IfElseOp":
                    condition = getattr(op, "condition", None)
                    if condition:
                        clbits = [condition[0]]
                    elif instruction.clbits:
                        clbits = instruction.clbits
                    else:
                        clbits = []
                    self._emit_if_else(op, qubits, clbits, loc)
                    self.stats["control_flow"] += 1
                else:
                    # Generic gate handling
                    self._emit_gate(op.name, qubits, op.params, loc)
                    self.stats["gates_processed"] += 1

            except Exception as e:
                raise CompileError(
                    f"Error processing instruction {idx} ({op.name}): {e}\n"
                    f"Qubits: {qubits}, Params: {getattr(op, 'params', [])}"
                )

    def _emit_measure(self, qubits: List, clbits: List, loc: Location):
        """Emit measurement operation with validation."""
        if len(qubits) > len(clbits):
            raise CompileError(
                f"Insufficient classical bits for measurement: "
                f"{len(qubits)} qubits but only {len(clbits)} classical bits"
            )

        for i, q in enumerate(qubits):
            if q not in self.qubit_map:
                raise CompileError(f"Qubit {q} not found in qubit map")

            val_in = self.qubit_map[q]
            bit_type = Type.parse("!quantum.bit", context=self.ctx)
            i1_type = IntegerType.get_signless(1, self.ctx)

            op = self._emit_quantum_op("quantum.measure", [val_in], [i1_type, bit_type], loc)
            self.qubit_map[q] = op.results[1]
            if i < len(clbits):
                self.clbit_map[clbits[i]] = op.results[0]
                logger.debug(f"Measured qubit {q} into classical bit {clbits[i]}")

    def _emit_gate(self, gate_name: str, qubits: List, params: List, loc: Location):
        """Emit quantum gate with parameter handling."""
        if not qubits:
            raise CompileError(f"Gate {gate_name} requires at least one qubit")

        in_qubits = []
        for q in qubits:
            if q not in self.qubit_map:
                raise CompileError(f"Qubit {q} not found in qubit map for gate {gate_name}")
            in_qubits.append(self.qubit_map[q])

        bit_type = Type.parse("!quantum.bit", context=self.ctx)
        result_types = [bit_type] * len(qubits)

        # Handle parameters
        op_params = []
        f64_type = Type.parse("f64", context=self.ctx)

        with loc:
            for p in params:
                try:
                    val = float(p)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping non-numeric parameter for {gate_name}: {p}")
                    continue

                val_attr = FloatAttr.get(f64_type, val)
                val_op = arith.ConstantOp(f64_type, val_attr, loc=loc)
                op_params.append(val_op.result)

        op_ctrls = []
        op_ctrlvals = []

        all_operands = op_params + in_qubits + op_ctrls + op_ctrlvals

        sizes = [len(op_params), len(in_qubits), len(op_ctrls), len(op_ctrlvals)]
        result_sizes = [len(result_types), 0]

        attrs = {
            "gate_name": StringAttr.get(gate_name, self.ctx),
            "operandSegmentSizes": DenseI32ArrayAttr.get(sizes, context=self.ctx),
            "resultSegmentSizes": DenseI32ArrayAttr.get(result_sizes, context=self.ctx),
        }

        op = self._emit_quantum_op("quantum.custom", all_operands, result_types, loc, attrs)

        results = op.results
        for i, q in enumerate(qubits):
            self.qubit_map[q] = results[i]

        logger.debug(f"Emitted gate {gate_name} on {len(qubits)} qubit(s)")

    def _emit_for_loop(self, op: Any, qubits: List, loc: Location):
        """Emit for loop with bounds checking."""
        indexset, loop_param, body = op.params

        if len(indexset) != 3:
            start, stop, step = 0, 1, 1
        else:
            start, stop, step = indexset

        if step == 0:
            raise CompileError("For loop step cannot be zero")

        idx_type = IntegerType.get_signless(64, self.ctx)
        lb = arith.ConstantOp(idx_type, start, loc=loc).result
        ub = arith.ConstantOp(idx_type, stop, loc=loc).result
        st = arith.ConstantOp(idx_type, step, loc=loc).result

        iter_args_init = [self.qubit_map[q] for q in qubits]

        for_op = scf.ForOp(lb, ub, st, iter_args_init, loc=loc)

        with InsertionPoint(for_op.body):
            ind_var = for_op.induction_var
            block_args = for_op.inner_iter_args

            old_map = self.qubit_map.copy()
            for i, q in enumerate(qubits):
                self.qubit_map[q] = block_args[i]

            self._process_instructions(body.data, loc)

            yield_args = [self.qubit_map[q] for q in qubits]
            scf.YieldOp(yield_args, loc=loc)

            self.qubit_map = old_map

        results = for_op.results
        for i, q in enumerate(qubits):
            self.qubit_map[q] = results[i]

        logger.debug(f"Emitted for loop from {start} to {stop} step {step}")

    def _emit_if_else(self, op: Any, qubits: List, clbits: List, loc: Location):
        """Emit if-else with condition validation."""
        true_body = op.params[0]
        false_body = op.params[1]

        if not clbits:
            raise CompileError("if_else instruction requires classical bits for condition")

        cond_bit = clbits[0]
        if cond_bit not in self.clbit_map:
            raise CompileError(
                f"Classical bit {cond_bit} used in if_else but not measured previously.\n"
                f"Ensure measurement occurs before conditional operation."
            )

        cond_val = self.clbit_map[cond_bit]

        if_op = scf.IfOp(cond_val, [self.qubit_map[q].type for q in qubits], hasElse=True, loc=loc)

        # True branch
        with InsertionPoint(if_op.then_block):
            old_map = self.qubit_map.copy()
            self._process_instructions(true_body.data, loc)
            yield_results = [self.qubit_map[q] for q in qubits]
            scf.YieldOp(yield_results, loc=loc)
            self.qubit_map = old_map

        # False branch
        with InsertionPoint(if_op.else_block):
            if false_body is not None:
                old_map = self.qubit_map.copy()
                self._process_instructions(false_body.data, loc)
                yield_results = [self.qubit_map[q] for q in qubits]
                scf.YieldOp(yield_results, loc=loc)
                self.qubit_map = old_map
            else:
                yield_results = [self.qubit_map[q] for q in qubits]
                scf.YieldOp(yield_results, loc=loc)

        results = if_op.results
        for i, q in enumerate(qubits):
            self.qubit_map[q] = results[i]

        logger.debug(f"Emitted if-else conditional on classical bit {cond_bit}")

    def get_stats(self) -> Dict[str, int]:
        """Return compilation statistics."""
        return self.stats.copy()
