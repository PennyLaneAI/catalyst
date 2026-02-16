
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

    LogicalResult emitOperation(Operation *op) {
        return TypeSwitch<Operation *, LogicalResult>(op)
            .Case<ModuleOp>([&](ModuleOp op) { return emitModule(op); })
            .Case<func::FuncOp>([&](func::FuncOp op) { return emitFunction(op); })
            .Case<scf::ForOp>([&](scf::ForOp op) { return emitForLoop(op); })
            .Case<CustomOp>([&](CustomOp op) { return emitCustomGate(op); })
            // Add other quantum ops like Alloc, Measure here
            .Case<AllocOp>([&](AllocOp op) { return emitAlloc(op); })
            .Case<ExtractOp>([&](ExtractOp op) { return emitExtract(op); })
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
        
        // We need to map block args to something.
        
        for (Operation &innerOp : *op.getBody()) {
             if (failed(emitOperation(&innerOp))) return failure();
        }
        
        os << "}\n";
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
        
        os << "qubit " << name << "[" << n << "];\n";
        
        // Map the result (register) to this name
        qubitMap[op.getResult()] = name;
        return success();
    }
    
    LogicalResult emitExtract(ExtractOp op) {
        // quantum.extract(reg, idx) -> reg[idx]
        // Map result SSA to string "name[idx]"
        
        Value reg = op.getQreg();
        Value idx = op.getIdx();
        
        std::string regName = qubitMap[reg];
        
        int64_t i = 0;

        if (auto cOp = idx.getDefiningOp<arith::ConstantOp>()) {
             if (auto intAttr = dyn_cast<IntegerAttr>(cOp.getValue())) {
                 i = intAttr.getInt();
             }
        }
        
        std::string qName = regName + "[" + std::to_string(i) + "]";
        qubitMap[op.getResult()] = qName;
        
        return success();
    }

    LogicalResult emitCustomGate(CustomOp op) {
        // quantum.custom "name" (q1, q2)
        os << op.getGateName() << " ";
        
        auto operands = op.getInQubits();
        for (size_t i = 0; i < operands.size(); ++i) {
            Value q = operands[i];
            // Lookup name
            if (qubitMap.count(q)) {
                 os << qubitMap[q];
            } else {
                 os << "unknown_q";
            }
            if (i < operands.size() - 1) os << ", ";
        }
        os << ";\n";
        
        // Update map for results?
        // In Catalyst, gates return new SSA values for qubits.
        // We must map these new values to the SAME names as inputs if we want QASM style.
        // QASM is not SSA. "h q[0]" modifies q[0].
        
        auto results = op.getOutQubits();
        for (size_t i = 0; i < results.size(); ++i) {
             Value inQ = operands[i];
             Value outQ = results[i];
             qubitMap[outQ] = qubitMap[inQ];
        }
        
        return success();
    }

    LogicalResult emitMeasure(MeasureOp op) {
        // quantum.measure(q) -> (bit, q_out)
        // In QASM: bit c = measure q;
        
        Value measureQubit = op.getInQubit();
        std::string qName = "unknown_q";
        if (qubitMap.count(measureQubit)) {
            qName = qubitMap[measureQubit];
        }
        
        // Generate a name for the classical outcome
        std::string cName = "c" + std::to_string(qubitCounter++); // Reuse counter for simplicity? Or new one.
        // Let's use "m" prefix for measurement results
        cName = "m_" + std::to_string(qubitCounter++);

        os << "bit " << cName << " = measure " << qName << ";\n";
        
        // Update map for the qubit OUT state 
        // In Catalyst, measure returns the qubit state as well.
        qubitMap[op.getOutQubit()] = qName;
        
        // We might want to map the result bit too?
        // But for QASM generation, we just need to emit the statement.
        
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
