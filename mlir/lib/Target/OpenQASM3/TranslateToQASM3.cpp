
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

// Include Catalyst Dialects
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace {

class QASM3Emitter {
public:
    QASM3Emitter(raw_ostream &os) : os(os) {}

    LogicalResult emitModule(ModuleOp module) {
        os << "OPENQASM 3.0;\ninclude \"stdgates.inc\";\n\n";
        
        // Walk the module
        for (Operation &op : module.getBody()->getOperations()) {
            if (failed(emitOperation(&op)))
                return failure();
        }
        return success();
    }

private:
    raw_ostream &os;
    unsigned qubitCounter = 0;
    
    // Map SSA values to OpenQASM variables (e.g. "q[0]")
    // In a real compiler, we might interpret AllocOp to know the array name.
    // For this task, we assume simple mapping or pass-through.
    DenseMap<Value, std::string> qubitMap;
    // Map SSA values to OpenQASM classical bits (e.g. "c[0]")
    DenseMap<Value, std::string> bitMap;

    LogicalResult emitOperation(Operation *op) {
        return TypeSwitch<Operation *, LogicalResult>(op)
            .Case<ModuleOp>([&](ModuleOp op) { return emitModule(op); })
            .Case<func::FuncOp>([&](func::FuncOp op) { return emitFunction(op); })
            .Case<scf::ForOp>([&](scf::ForOp op) { return emitForLoop(op); })
            .Case<scf::IfOp>([&](scf::IfOp op) { return emitIf(op); })
            .Case<CustomOp>([&](CustomOp op) { return emitCustomGate(op); })
            // Add other quantum ops like Alloc, Measure here
            .Case<AllocOp>([&](AllocOp op) { return emitAlloc(op); })
            .Case<ExtractOp>([&](ExtractOp op) { return emitExtract(op); })
            .Case<InsertOp>([&](InsertOp op) { return emitInsert(op); })
            .Case<MeasureOp>([&](MeasureOp op) { return emitMeasure(op); })
            .Case<scf::YieldOp>([&](scf::YieldOp op) { return success(); }) // Handled by parent
            .Case<func::ReturnOp>([&](func::ReturnOp op) { return success(); })
            .Default([&](Operation *op) {
                // Ignore other ops for now or emit warning
                // os << "// Unhandled op: " << op->getName().getStringRef() << "\n";
                return success();
            });
    }

    LogicalResult emitFunction(func::FuncOp op) {
        // Simple main function or simple scope
        // If it's main, we just emit body.
        if (op.getName() == "main") {
            for (Operation &innerOp : op.getBody().front()) {
                if (failed(emitOperation(&innerOp))) return failure();
            }
            return success();
        }
        
        // Otherwise emit as box or def (simplification)
        os << "def " << op.getName() << "() {\n";
        for (Operation &innerOp : op.getBody().front()) {
            if (failed(emitOperation(&innerOp))) return failure();
        }
        os << "}\n";
        return success();
    }

    LogicalResult emitForLoop(scf::ForOp op) {
        // Convert scf.for to OpenQASM for
        // for i in [start:step:stop] { ... }

        // We need to resolve bounds to constants if possible
        // We need to resolve bounds to constants if possible
        // int64_t start, stop, step;

        // Helper to extract constant integer value
        auto getConst = [](Value v) -> std::optional<int64_t> {
             if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
                 if (auto intAttr = dyn_cast<IntegerAttr>(cOp.getValue())) {
                     return intAttr.getInt();
                 }
             }
             return std::nullopt;
        };

        // Note: Catalyst uses arith.constant for bounds usually.
        // We'll trust the user provided constants for this task logic.

        // Handling loop variable name
        std::string loopVar = "i_" + std::to_string(qubitCounter++); // Simple unique name

        // scf.for uses lower, upper, step
        // We need to print them.
        // For simplicity, let's just assume they are computable or print their values.

        // Actually, we should check if they are constants.
        // If not, we'd need to emit code to compute them, but OpenQASM 3 loops expect ranges.

        int64_t start = 0, stop = 0, step = 1;
        if (auto s = getConst(op.getLowerBound())) start = *s;
        if (auto e = getConst(op.getUpperBound())) stop = *e;
        if (auto st = getConst(op.getStep())) step = *st;

        os << "for " << loopVar << " in [" << start << ":" << step << ":" << stop << "]";
        os << " {\n";

        // Handle body
        // Loop induction var is argument 0
        // Iter args are subsequent args

        // Map block arguments before processing body
        // This ensures qubits carried through loop iterations maintain their names
        auto &loopBody = op.getRegion().front();

        // Map iter_args (initial values) to loop body block arguments
        // Block arg 0 is the induction variable, block args 1+ are iter_args
        auto initArgs = op.getInitArgs();
        for (size_t i = 0; i < initArgs.size(); ++i) {
            Value initVal = initArgs[i];
            Value blockArg = loopBody.getArgument(i + 1); // +1 to skip induction var

            // If this is a qubit value, propagate the mapping
            if (qubitMap.count(initVal)) {
                qubitMap[blockArg] = qubitMap[initVal];
            }
            // Also handle classical bits if needed
            if (bitMap.count(initVal)) {
                bitMap[blockArg] = bitMap[initVal];
            }
        }

        for (Operation &innerOp : *op.getBody()) {
             if (failed(emitOperation(&innerOp))) return failure();
        }

        os << "}\n";

        // Map loop results to qubit names from the loop-carried values
        // scf.for results correspond to the values yielded by the loop body
        // We need to find the yield operation and map its operands to the loop results
        auto *terminator = loopBody.getTerminator();
        if (auto yieldOp = dyn_cast<scf::YieldOp>(terminator)) {
            auto yieldOperands = yieldOp.getOperands();
            auto results = op.getResults();

            for (size_t i = 0; i < results.size() && i < yieldOperands.size(); ++i) {
                Value yieldedVal = yieldOperands[i];
                Value result = results[i];

                // Map result to the same qubit name as the yielded value
                if (qubitMap.count(yieldedVal)) {
                    qubitMap[result] = qubitMap[yieldedVal];
                }
                if (bitMap.count(yieldedVal)) {
                    bitMap[result] = bitMap[yieldedVal];
                }
            }
        }

        return success();
    }

    LogicalResult emitIf(scf::IfOp op) {
        Value cond = op.getCondition();
        std::string condName = "unknown_cond";
        if (bitMap.count(cond)) {
            condName = bitMap[cond];
        }

        os << "if (" << condName << " == 1) {\n";

        for (Operation &innerOp : op.getThenRegion().front()) {
            if (failed(emitOperation(&innerOp))) return failure();
        }
        os << "}";

        if (!op.getElseRegion().empty()) {
            // Check if else block is effectively empty (just yield)
            // But we can just print it.
            os << " else {\n";
            for (Operation &innerOp : op.getElseRegion().front()) {
                if (failed(emitOperation(&innerOp))) return failure();
            }
            os << "}";
        }
        os << "\n";

        // Map scf.if results to qubit names from yielded values
        // The results come from the then/else branches via yield operations
        auto results = op.getResults();
        if (!results.empty()) {
            // Get yield from then branch
            auto &thenBlock = op.getThenRegion().front();
            auto *thenTerminator = thenBlock.getTerminator();

            if (auto thenYield = dyn_cast<scf::YieldOp>(thenTerminator)) {
                auto yieldOperands = thenYield.getOperands();

                for (size_t i = 0; i < results.size() && i < yieldOperands.size(); ++i) {
                    Value yieldedVal = yieldOperands[i];
                    Value result = results[i];

                    // Map result to the same qubit/bit name as the yielded value
                    // In QASM, the if doesn't create new variables, it modifies existing ones
                    if (qubitMap.count(yieldedVal)) {
                        qubitMap[result] = qubitMap[yieldedVal];
                    }
                    if (bitMap.count(yieldedVal)) {
                        bitMap[result] = bitMap[yieldedVal];
                    }
                }
            }
        }

        return success();
    }

    LogicalResult emitAlloc(AllocOp op) {
        // quantum.alloc(n) -> qreg q[n];
        // We need to name it.
        std::string name = "q" + std::to_string(qubitCounter++);
        
        // n_qubits is an operand. Check if constant.

        int64_t n = 1; 
        if (auto cOp = op.getNqubits().getDefiningOp<arith::ConstantOp>()) {
             if (auto intAttr = dyn_cast<IntegerAttr>(cOp.getValue())) {
                 n = intAttr.getInt();
             }
        }
        
        os << "qubit[" << n << "] " << name << ";\n";
        
        // Map the result (register) to this name
        qubitMap[op.getResult()] = name;
        return success();
    }
    
    LogicalResult emitExtract(ExtractOp op) {
        // quantum.extract(reg, idx) -> reg[idx]
        // Map result SSA to string "name[idx]"

        Value reg = op.getQreg();
        Value idx = op.getIdx();

        // Check if register is mapped
        if (!qubitMap.count(reg)) {
            return op.emitError("Cannot emit extract: register operand not mapped");
        }
        std::string regName = qubitMap[reg];
        if (regName.empty()) {
            return op.emitError("Cannot emit extract: register name is empty");
        }

        int64_t i = 0;

        if (idx) {
            if (auto cOp = idx.getDefiningOp<arith::ConstantOp>()) {
                 if (auto intAttr = dyn_cast<IntegerAttr>(cOp.getValue())) {
                     i = intAttr.getInt();
                 }
            }
        } else if (op.getIdxAttr().has_value()) {
            i = op.getIdxAttr().value();
        }

        std::string qName = regName + "[" + std::to_string(i) + "]";
        qubitMap[op.getResult()] = qName;

        return success();
    }

    LogicalResult emitInsert(InsertOp op) {
        // quantum.insert just maintains SSA, OpenQASM modifies in place.
        Value inQreg = op.getInQreg();
        if (!qubitMap.count(inQreg) || qubitMap[inQreg].empty()) {
            return op.emitError("Cannot emit insert: input register not mapped");
        }
        qubitMap[op.getResult()] = qubitMap[inQreg];
        return success();
    }

    LogicalResult emitCustomGate(CustomOp op) {
        // quantum.custom "name" (q1, q2)
        // or quantum.custom "name"(p1) (q1)
        llvm::StringRef gateName = op.getGateName();
        if (gateName == "cnot") {
            os << "cx";
        } else {
            os << gateName;
        }
        
        auto params = op.getParams();
        if (!params.empty()) {
            os << "(";
            for (size_t i = 0; i < params.size(); ++i) {
                 Value p = params[i];
                 // Try to resolve constant
                 if (auto cOp = p.getDefiningOp<arith::ConstantOp>()) {
                     if (auto floatAttr = dyn_cast<FloatAttr>(cOp.getValue())) {
                         os << floatAttr.getValueAsDouble();
                     } else if (auto intAttr = dyn_cast<IntegerAttr>(cOp.getValue())) {
                         os << intAttr.getInt();
                     } else {
                         os << "unknown_param";
                     }
                 } else {
                     os << "unknown_param";
                 }
                 
                 if (i < params.size() - 1) os << ", ";
            }
            os << ")";
        }
        
        os << " ";
        
        auto operands = op.getInQubits();
        for (size_t i = 0; i < operands.size(); ++i) {
            Value q = operands[i];
            // Lookup name
            if (qubitMap.count(q) && !qubitMap[q].empty()) {
                 os << qubitMap[q];
            } else {
                 // Input qubit not mapped - should not happen in well-formed IR
                 llvm::errs() << "ERROR: Gate " << gateName << " input qubit not mapped\n";
                 llvm::errs() << "  Input value: " << q << "\n";
                 return op.emitError("Cannot emit gate: input qubit not mapped to QASM variable");
            }
            if (i < operands.size() - 1) os << ", ";
        }
        os << ";\n";

        // Update map for results
        // In Catalyst, gates return new SSA values for qubits.
        // We must map these new values to the SAME names as inputs if we want QASM style.
        // QASM is not SSA. "h q[0]" modifies q[0].

        // Use the standard MLIR getResults() API which correctly handles multi-result operations
        // (operations that return tuples like %result:2)
        auto results = op->getResults();

        // DEBUG: For multi-result operations, print details
        if (results.size() > 1 && gateName == "cu") {
            llvm::errs() << "DEBUG: cu gate has " << results.size() << " results\n";
            for (size_t i = 0; i < results.size(); ++i) {
                llvm::errs() << "  results[" << i << "] = " << results[i] << "\n";
            }
        }

        // Map each result to the corresponding input qubit name
        if (results.size() != operands.size()) {
            return op.emitError("Mismatch between qubit operands and results");
        }

        for (size_t i = 0; i < results.size(); ++i) {
             Value inQ = operands[i];
             Value outQ = results[i];
             if (qubitMap.count(inQ) && !qubitMap[inQ].empty()) {
                 qubitMap[outQ] = qubitMap[inQ];
                 if (results.size() > 1 && gateName == "cu") {
                     llvm::errs() << "DEBUG: Mapped cu result[" << i << "] to " << qubitMap[inQ] << "\n";
                 }
             }
             // If input not mapped, output won't be mapped either
             // This will cause errors downstream, which is intentional (fail-fast)
        }
        
        return success();
    }

    LogicalResult emitMeasure(MeasureOp op) {
        // quantum.measure(q) -> (bit, q_out)
        // In QASM: bit c = measure q; -> bit c; c = measure q;

        Value measureQubit = op.getInQubit();
        std::string qName;
        if (qubitMap.count(measureQubit) && !qubitMap[measureQubit].empty()) {
            qName = qubitMap[measureQubit];
        } else {
            // Qubit not mapped - should not happen in well-formed IR
            return op.emitError("Cannot emit measure: qubit operand not mapped to QASM variable");
        }

        // Generate a name for the classical outcome
        std::string cName = "m_" + std::to_string(qubitCounter++);

        os << "bit " << cName << ";\n" << cName << " = measure " << qName << ";\n";

        // Update map for the qubit OUT state
        // In Catalyst, measure returns the qubit state as well.
        qubitMap[op.getOutQubit()] = qName;

        // Map the result bit
        bitMap[op.getResult(0)] = cName;

        return success();
    }
};

} // namespace

LogicalResult translateModuleToOpenQASM3(ModuleOp module, raw_ostream &output) {
    QASM3Emitter emitter(output);
    return emitter.emitModule(module);
}

namespace mlir {
void registerToQASM3Translation() {
    static TranslateFromMLIRRegistration registration(
        "mlir-to-qasm3",
        "Translate MLIR to OpenQASM 3.0",
        translateModuleToOpenQASM3,
        [](DialectRegistry &registry) {
            registry.insert<catalyst::quantum::QuantumDialect>();
            registry.insert<scf::SCFDialect>();
            registry.insert<func::FuncDialect>();
            registry.insert<arith::ArithDialect>();
        });
}
}
