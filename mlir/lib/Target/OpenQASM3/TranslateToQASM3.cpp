
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Tools/mlir-translate/Translation.h"

// Include Catalyst Dialects
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace {

class QASM3Emitter {
  public:
    QASM3Emitter(raw_ostream &os) : os(os) {}

    LogicalResult emitModule(ModuleOp module)
    {
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
    // Track which non-stdgates custom gate definitions have been emitted
    llvm::SmallSet<std::string, 8> emittedCustomGates;
    // Track which classical registers (bit[n] c;) have been declared
    llvm::SmallSet<std::string, 4> declaredCregs;

    // Copy a name mapping from one SSA value to another. NEVER write
    // `map[to] = map[from]` directly: the LHS operator[] may insert and
    // rehash the DenseMap, invalidating the RHS reference mid-assignment
    // (C++17 evaluates the RHS first). Symptom: silently empty/garbage
    // names once the map grows past a rehash threshold.
    static void copyMapping(DenseMap<Value, std::string> &map, Value from, Value to)
    {
        auto it = map.find(from);
        if (it == map.end())
            return;
        std::string name = it->second;
        map[to] = name;
    }

    LogicalResult emitOperation(Operation *op)
    {
        return TypeSwitch<Operation *, LogicalResult>(op)
            .Case<ModuleOp>([&](ModuleOp op) { return emitModule(op); })
            .Case<func::FuncOp>([&](func::FuncOp op) { return emitFunction(op); })
            .Case<scf::ForOp>([&](scf::ForOp op) { return emitForLoop(op); })
            .Case<scf::IfOp>([&](scf::IfOp op) { return emitIf(op); })
            .Case<scf::WhileOp>([&](scf::WhileOp op) { return emitWhileLoop(op); })
            .Case<CustomOp>([&](CustomOp op) { return emitCustomGate(op); })
            // Add other quantum ops like Alloc, Measure here
            .Case<AllocOp>([&](AllocOp op) { return emitAlloc(op); })
            .Case<ExtractOp>([&](ExtractOp op) { return emitExtract(op); })
            .Case<InsertOp>([&](InsertOp op) { return emitInsert(op); })
            .Case<MeasureOp>([&](MeasureOp op) { return emitMeasure(op); })
            .Case<scf::YieldOp>([&](scf::YieldOp op) { return success(); }) // Handled by parent
            .Case<func::ReturnOp>([&](func::ReturnOp op) { return success(); })
            .Default([&](Operation *op) {
                llvm::errs() << "WARNING: Unhandled op: " << op->getName().getStringRef() << "\n";
                return success();
            });
    }

    LogicalResult emitFunction(func::FuncOp op)
    {
        // Simple main function or simple scope
        // If it's main, we just emit body.
        if (op.getName() == "main") {
            // Declare all classical registers up front. Measurements may sit
            // inside if/for/while bodies, but QASM3 register declarations must
            // be visible at the outer scope for feedforward conditions.
            emitCregDeclarations(op);
            for (Operation &innerOp : op.getBody().front()) {
                if (failed(emitOperation(&innerOp)))
                    return failure();
            }
            return success();
        }

        // Otherwise emit as box or def (simplification)
        os << "def " << op.getName() << "() {\n";
        // Local register declarations for any creg-tagged measurements here.
        emitCregDeclarations(op);
        for (Operation &innerOp : op.getBody().front()) {
            if (failed(emitOperation(&innerOp)))
                return failure();
        }
        os << "}\n";
        return success();
    }

    void emitCregDeclarations(func::FuncOp fn)
    {
        fn.walk([&](MeasureOp m) {
            auto nameAttr = m->getAttrOfType<StringAttr>("creg_name");
            if (!nameAttr)
                return;
            std::string name = nameAttr.getValue().str();
            if (declaredCregs.count(name))
                return;
            declaredCregs.insert(name);
            int64_t size = 1;
            if (auto sizeAttr = m->getAttrOfType<IntegerAttr>("creg_size"))
                size = sizeAttr.getInt();
            os << "bit[" << size << "] " << name << ";\n";
        });
    }

    LogicalResult emitForLoop(scf::ForOp op)
    {
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
        if (auto s = getConst(op.getLowerBound()))
            start = *s;
        if (auto e = getConst(op.getUpperBound()))
            stop = *e;
        if (auto st = getConst(op.getStep()))
            step = *st;

        os << "for int " << loopVar << " in [" << start << ":" << step << ":" << stop << "]";
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

            copyMapping(qubitMap, initVal, blockArg);
            copyMapping(bitMap, initVal, blockArg);
        }

        for (Operation &innerOp : *op.getBody()) {
            if (failed(emitOperation(&innerOp)))
                return failure();
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
                copyMapping(qubitMap, yieldOperands[i], results[i]);
                copyMapping(bitMap, yieldOperands[i], results[i]);
            }
        }

        return success();
    }

    static bool splitBitName(const std::string &s, std::string &reg, int64_t &idx)
    {
        auto lb = s.find('[');
        auto rb = s.find(']');
        if (lb == std::string::npos || rb == std::string::npos || rb != s.size() - 1 ||
            rb <= lb + 1)
            return false;
        std::string digits = s.substr(lb + 1, rb - lb - 1);
        for (char c : digits)
            if (!isdigit(c))
                return false;
        reg = s.substr(0, lb);
        idx = std::stoll(digits);
        return true;
    }

    // Try to reconstruct a whole-register comparison (reg == k) from a
    // conjunction of per-bit tests (c[i] / !c[i]) over one classical
    // register. Returns the number of matched leaves (0 = no match).
    int matchRegisterEquality(Value v, std::string &regName, int64_t &k)
    {
        if (auto andOp = v.getDefiningOp<arith::AndIOp>()) {
            int l = matchRegisterEquality(andOp.getLhs(), regName, k);
            if (!l)
                return 0;
            int r = matchRegisterEquality(andOp.getRhs(), regName, k);
            return r ? l + r : 0;
        }
        bool negated = false;
        Value leaf = v;
        if (auto xorOp = v.getDefiningOp<arith::XOrIOp>()) {
            auto isTrueConst = [](Value c) {
                if (auto cOp = c.getDefiningOp<arith::ConstantOp>())
                    if (auto intAttr = dyn_cast<IntegerAttr>(cOp.getValue()))
                        return !intAttr.getValue().isZero();
                return false;
            };
            if (isTrueConst(xorOp.getRhs())) {
                negated = true;
                leaf = xorOp.getLhs();
            }
            else if (isTrueConst(xorOp.getLhs())) {
                negated = true;
                leaf = xorOp.getRhs();
            }
        }
        if (!bitMap.count(leaf))
            return 0;
        std::string reg;
        int64_t idx;
        if (!splitBitName(bitMap[leaf], reg, idx))
            return 0;
        if (regName.empty())
            regName = reg;
        else if (regName != reg)
            return 0;
        if (!negated)
            k |= (int64_t(1) << idx);
        return 1;
    }

    // Recursively translate a classical i1/int SSA value into a QASM3
    // condition expression. Supports measurement bits, constants, negation
    // (arith.xori with 1), conjunction/disjunction, and eq/ne comparisons.
    std::string buildCondExpr(Value v)
    {
        if (bitMap.count(v))
            return bitMap[v];

        Operation *def = v.getDefiningOp();
        if (!def) {
            llvm::errs() << "WARNING: condition value is an unmapped block argument\n";
            return "unknown_cond";
        }

        if (auto cOp = dyn_cast<arith::ConstantOp>(def)) {
            // NOTE: i1 constants ("true") sign-extend under getInt() to -1,
            // so read the raw APInt zero-extended instead.
            if (auto intAttr = dyn_cast<IntegerAttr>(cOp.getValue())) {
                if (intAttr.getType().isInteger(1))
                    return intAttr.getValue().isZero() ? "0" : "1";
                return std::to_string(intAttr.getInt());
            }
        }

        if (auto xorOp = dyn_cast<arith::XOrIOp>(def)) {
            // xori(x, 1) is boolean negation; xori(x, 0) is a no-op.
            auto getConst = [](Value v) -> std::optional<bool> {
                if (auto cOp = v.getDefiningOp<arith::ConstantOp>())
                    if (auto intAttr = dyn_cast<IntegerAttr>(cOp.getValue()))
                        return !intAttr.getValue().isZero();
                return std::nullopt;
            };
            Value other;
            std::optional<bool> c;
            if ((c = getConst(xorOp.getRhs())))
                other = xorOp.getLhs();
            else if ((c = getConst(xorOp.getLhs())))
                other = xorOp.getRhs();
            if (c) {
                if (*c) {
                    // Negated register equality reads as a != comparison.
                    // Only folds the importer tagged as equality tests may be
                    // reconstructed; generic conjunctions don't constrain
                    // unmentioned register bits.
                    Operation *innerDef = other.getDefiningOp();
                    if (innerDef && innerDef->hasAttr("qasm3_creg_eq")) {
                        std::string reg;
                        int64_t k = 0;
                        if (matchRegisterEquality(other, reg, k) >= 2)
                            return reg + " != " + std::to_string(k);
                    }
                }
                std::string inner = buildCondExpr(other);
                if (!*c)
                    return inner;
                return inner.find(' ') == std::string::npos ? "!" + inner : "!(" + inner + ")";
            }
        }

        if (auto andOp = dyn_cast<arith::AndIOp>(def)) {
            // Prefer a whole-register comparison over a bit-by-bit conjunction,
            // but only for folds the importer tagged as register-equality
            // tests (`qasm3_creg_eq`) — a generic `c[0] && c[1]` conjunction
            // does not constrain the register's unmentioned bits.
            if (def->hasAttr("qasm3_creg_eq")) {
                std::string reg;
                int64_t k = 0;
                if (matchRegisterEquality(v, reg, k) >= 2)
                    return reg + " == " + std::to_string(k);
            }
            return "(" + buildCondExpr(andOp.getLhs()) + " && " + buildCondExpr(andOp.getRhs()) +
                   ")";
        }

        if (auto orOp = dyn_cast<arith::OrIOp>(def))
            return "(" + buildCondExpr(orOp.getLhs()) + " || " + buildCondExpr(orOp.getRhs()) +
                   ")";

        if (auto cmpOp = dyn_cast<arith::CmpIOp>(def)) {
            auto pred = cmpOp.getPredicate();
            if (pred == arith::CmpIPredicate::eq || pred == arith::CmpIPredicate::ne) {
                std::string lhs = buildCondExpr(cmpOp.getLhs());
                std::string rhs = buildCondExpr(cmpOp.getRhs());
                const char *opStr = (pred == arith::CmpIPredicate::eq) ? " == " : " != ";
                return lhs + opStr + rhs;
            }
        }

        llvm::errs() << "WARNING: unsupported condition op: " << def->getName().getStringRef()
                     << "\n";
        return "unknown_cond";
    }

    LogicalResult emitIf(scf::IfOp op)
    {
        std::string condExpr = buildCondExpr(op.getCondition());

        os << "if (" << condExpr << ") {\n";

        for (Operation &innerOp : op.getThenRegion().front()) {
            if (failed(emitOperation(&innerOp)))
                return failure();
        }
        os << "}";

        if (!op.getElseRegion().empty()) {
            // Check if else block is effectively empty (just yield)
            // But we can just print it.
            os << " else {\n";
            for (Operation &innerOp : op.getElseRegion().front()) {
                if (failed(emitOperation(&innerOp)))
                    return failure();
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

                // Map results to the same qubit/bit names as the yielded
                // values: in QASM the if modifies variables in place.
                for (size_t i = 0; i < results.size() && i < yieldOperands.size(); ++i) {
                    copyMapping(qubitMap, yieldOperands[i], results[i]);
                    copyMapping(bitMap, yieldOperands[i], results[i]);
                }
            }
        }

        return success();
    }

    LogicalResult emitWhileLoop(scf::WhileOp op)
    {
        // scf.while → while (<cond>) { ... }
        // The before region computes the condition from loop-carried values;
        // it is pure classical logic, so we translate it as an expression and
        // never emit its ops as statements. QASM3's named creg bits provide
        // the loop-carry that SSA does in MLIR: the in-body measurement
        // re-assigns the same c[i] the condition reads.
        Block &before = op.getBefore().front();
        auto inits = op.getInits();
        for (size_t i = 0; i < inits.size() && i < before.getNumArguments(); ++i) {
            copyMapping(qubitMap, inits[i], before.getArgument(i));
            copyMapping(bitMap, inits[i], before.getArgument(i));
        }

        // The before region must be pure classical condition computation; any
        // quantum op or side effect there has no place in the emitted text.
        for (Operation &beforeOp : before) {
            if (!isa<scf::ConditionOp>(beforeOp) &&
                beforeOp.getDialect()->getNamespace() != "arith") {
                llvm::errs() << "WARNING: op '" << beforeOp.getName().getStringRef()
                             << "' in scf.while before-region is dropped; only pure "
                             << "condition computation is supported there\n";
            }
        }

        auto condOp = op.getConditionOp();
        std::string condExpr = buildCondExpr(condOp.getCondition());
        if (condExpr.find("m_") != std::string::npos) {
            llvm::errs() << "WARNING: while condition uses an anonymous measurement bit ('"
                         << condExpr << "'); re-measurements inside the loop body will not "
                         << "update it. Measure into a named classical register instead.\n";
        }

        os << "while (" << condExpr << ") {\n";

        // After-region args correspond to scf.condition's forwarded operands,
        // NOT positionally to the inits: canonicalize prunes unused forwarded
        // values, so the two lists can differ.
        Block &after = op.getAfter().front();
        auto fwdArgs = condOp.getArgs();
        for (size_t i = 0; i < fwdArgs.size() && i < after.getNumArguments(); ++i) {
            copyMapping(qubitMap, fwdArgs[i], after.getArgument(i));
            copyMapping(bitMap, fwdArgs[i], after.getArgument(i));
        }

        for (Operation &innerOp : after) {
            if (failed(emitOperation(&innerOp)))
                return failure();
        }

        os << "}\n";

        // Back-propagate names from the body's yielded values onto the
        // before-region args. A loop-carried bit first measured INSIDE the
        // body has a constant as its init (no name), but its yield operand
        // carries the creg name assigned during body emission.
        auto yieldOp = op.getYieldOp();
        auto yieldOperands = yieldOp.getOperands();
        for (size_t j = 0; j < yieldOperands.size() && j < before.getNumArguments(); ++j) {
            Value blockArg = before.getArgument(j);
            if (!qubitMap.count(blockArg))
                copyMapping(qubitMap, yieldOperands[j], blockArg);
            if (!bitMap.count(blockArg))
                copyMapping(bitMap, yieldOperands[j], blockArg);
        }

        // Loop results also correspond to the condition's forwarded operands.
        auto results = op.getResults();
        for (size_t i = 0; i < results.size() && i < fwdArgs.size(); ++i) {
            copyMapping(qubitMap, fwdArgs[i], results[i]);
            copyMapping(bitMap, fwdArgs[i], results[i]);
        }

        return success();
    }

    LogicalResult emitAlloc(AllocOp op)
    {
        // quantum.alloc(n) -> qreg q[n];
        // We need to name it.
        std::string name = "q" + std::to_string(qubitCounter++);

        // The qubit count is either an SSA operand or the nqubits_attr
        // attribute (e.g. `quantum.alloc(5)` literal form).
        int64_t n = 1;
        if (Value nq = op.getNqubits()) {
            if (auto cOp = nq.getDefiningOp<arith::ConstantOp>()) {
                if (auto intAttr = dyn_cast<IntegerAttr>(cOp.getValue())) {
                    n = intAttr.getInt();
                }
            }
        }
        else if (auto nqAttr = op.getNqubitsAttr()) {
            n = *nqAttr;
        }

        os << "qubit[" << n << "] " << name << ";\n";

        // Map the result (register) to this name
        qubitMap[op.getResult()] = name;
        return success();
    }

    LogicalResult emitExtract(ExtractOp op)
    {
        // quantum.extract(reg, idx) -> reg[idx]
        // Map result SSA to string "name[idx]"

        Value reg = op.getQreg();
        Value idx = op.getIdx();

        // Check if register is mapped
        if (!qubitMap.count(reg)) {
            // The register might not be mapped yet if this is a block argument
            // or if it's coming from a complex control flow pattern
            // Try to provide more diagnostic information
            llvm::errs() << "WARNING: ExtractOp register operand not mapped\n";
            llvm::errs() << "  Register value: " << reg << "\n";
            if (auto defOp = reg.getDefiningOp()) {
                llvm::errs() << "  Defined by: " << defOp->getName() << "\n";
            }
            else {
                llvm::errs() << "  (block argument)\n";
            }

            // Try to auto-generate a mapping if this is from an AllocOp
            if (auto allocOp = reg.getDefiningOp<AllocOp>()) {
                // This should have been handled already, but let's be safe
                llvm::errs() << "  Attempting to handle AllocOp retroactively\n";
                if (failed(emitAlloc(allocOp))) {
                    return failure();
                }
            }
            else {
                return op.emitError("Cannot emit extract: register operand not mapped");
            }
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
        }
        else if (op.getIdxAttr().has_value()) {
            i = op.getIdxAttr().value();
        }

        std::string qName = regName + "[" + std::to_string(i) + "]";
        qubitMap[op.getResult()] = qName;

        return success();
    }

    LogicalResult emitInsert(InsertOp op)
    {
        // quantum.insert just maintains SSA, OpenQASM modifies in place.
        Value inQreg = op.getInQreg();
        if (!qubitMap.count(inQreg) || qubitMap[inQreg].empty()) {
            return op.emitError("Cannot emit insert: input register not mapped");
        }
        copyMapping(qubitMap, inQreg, op.getResult());
        return success();
    }

    // Emit the gate body definition for custom gates not in stdgates.inc,
    // but only on first use. This avoids polluting every output file with
    // definitions for gates the circuit does not actually use.
    void maybeEmitCustomGateDef(const std::string &name)
    {
        if (emittedCustomGates.count(name))
            return;
        emittedCustomGates.insert(name);

        if (name == "rzz") {
            os << "gate rzz(theta) a, b {\n";
            os << "  cx a, b;\n";
            os << "  rz(theta) b;\n";
            os << "  cx a, b;\n";
            os << "}\n\n";
        }
        else if (name == "rxx") {
            os << "gate rxx(theta) a, b {\n";
            os << "  h a;\n";
            os << "  h b;\n";
            os << "  cx a, b;\n";
            os << "  rz(theta) b;\n";
            os << "  cx a, b;\n";
            os << "  h a;\n";
            os << "  h b;\n";
            os << "}\n\n";
        }
        else if (name == "ryy") {
            os << "gate ryy(theta) a, b {\n";
            os << "  rx(pi/2) a;\n";
            os << "  rx(pi/2) b;\n";
            os << "  cx a, b;\n";
            os << "  rz(theta) b;\n";
            os << "  cx a, b;\n";
            os << "  rx(-pi/2) a;\n";
            os << "  rx(-pi/2) b;\n";
            os << "}\n\n";
        }
        else if (name == "rccx") {
            os << "gate rccx a, b, c {\n";
            os << "  h c;\n";
            os << "  t c;\n";
            os << "  cx b, c;\n";
            os << "  tdg c;\n";
            os << "  cx a, c;\n";
            os << "  t c;\n";
            os << "  cx b, c;\n";
            os << "  tdg c;\n";
            os << "  h c;\n";
            os << "  t c;\n";
            os << "  cx a, b;\n";
            os << "  t a;\n";
            os << "  tdg b;\n";
            os << "  cx a, b;\n";
            os << "}\n\n";
        }
    }

    LogicalResult emitResetOrBarrier(CustomOp op)
    {
        llvm::StringRef name = op.getGateName();
        auto operands = op.getInQubits();

        SmallVector<std::string> qNames;
        for (Value q : operands) {
            if (!qubitMap.count(q) || qubitMap[q].empty())
                return op.emitError("Cannot emit " + name + ": qubit operand not mapped");
            qNames.push_back(qubitMap[q]);
        }

        if (name == "reset") {
            // One reset statement per qubit.
            for (const std::string &qName : qNames)
                os << "reset " << qName << ";\n";
        }
        else if (qNames.empty()) {
            // Bare form: barrier on everything.
            os << "barrier;\n";
        }
        else {
            os << "barrier " << llvm::join(qNames, ", ") << ";\n";
        }

        // Thread qubit SSA values through, QASM modifies in place.
        auto results = op->getResults();
        size_t numToMap = std::min(results.size(), operands.size());
        for (size_t i = 0; i < numToMap; ++i)
            qubitMap[results[i]] = qNames[i];

        return success();
    }

    LogicalResult emitCustomGate(CustomOp op)
    {
        // quantum.custom "name" (q1, q2)
        // or quantum.custom "name"(p1) (q1)
        llvm::StringRef gateName = op.getGateName();
        std::string qasmGateName;

        // "reset" and "barrier" are represented as quantum.custom markers by
        // the importer (the quantum dialect has no dedicated ops for them).
        // They are statements, not gates: no params, no gate defs.
        if (gateName == "reset" || gateName == "barrier")
            return emitResetOrBarrier(op);

        // Map gate names from QASM 2.0 (qelib1.inc) to QASM 3.0 (stdgates.inc)
        if (gateName == "cnot") {
            qasmGateName = "cx";
        }
        else if (gateName == "cu1") {
            // cu1(lambda) in QASM 2.0 is equivalent to cp(lambda) in QASM 3.0
            qasmGateName = "cp";
        }
        else if (gateName == "u") {
            // Qiskit's u(theta,phi,lambda) is the QASM 3.0 builtin U gate
            // (plain "u" is not defined in stdgates.inc).
            qasmGateName = "U";
        }
        else if (gateName == "rzz" || gateName == "rxx" || gateName == "ryy" ||
                 gateName == "rccx") {
            // These gates are not in stdgates.inc; emit their definition on first use.
            qasmGateName = gateName.str();
            maybeEmitCustomGateDef(qasmGateName);
        }
        else {
            qasmGateName = gateName.str();
        }

        os << qasmGateName;

        auto params = op.getParams();
        if (!params.empty()) {
            os << "(";
            for (size_t i = 0; i < params.size(); ++i) {
                Value p = params[i];
                // Try to resolve constant
                if (auto cOp = p.getDefiningOp<arith::ConstantOp>()) {
                    if (auto floatAttr = dyn_cast<FloatAttr>(cOp.getValue())) {
                        os << floatAttr.getValueAsDouble();
                    }
                    else if (auto intAttr = dyn_cast<IntegerAttr>(cOp.getValue())) {
                        os << intAttr.getInt();
                    }
                    else {
                        os << "unknown_param";
                    }
                }
                else {
                    os << "unknown_param";
                }

                if (i < params.size() - 1)
                    os << ", ";
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
            }
            else {
                // Input qubit not mapped - this should not happen in well-formed IR
                // after proper SSA canonicalization
                llvm::errs() << "ERROR: Gate '" << gateName << "' input qubit not mapped\n";
                llvm::errs() << "  Input value: " << q << "\n";
                if (auto opResult = dyn_cast<OpResult>(q)) {
                    llvm::errs() << "  This is result #" << opResult.getResultNumber()
                                 << " of operation " << opResult.getDefiningOp()->getName() << "\n";
                }
                llvm::errs() << "  Hint: This may be caused by complex SSA patterns from "
                                "quantum-opt canonicalization.\n";
                llvm::errs()
                    << "  Try disabling canonicalization or decomposing the circuit further.\n";
                return op.emitError("Cannot emit gate: input qubit not mapped to QASM variable");
            }
            if (i < operands.size() - 1)
                os << ", ";
        }
        os << ";\n";

        // Update map for results
        // In Catalyst, gates return new SSA values for qubits.
        // We must map these new values to the SAME names as inputs if we want QASM style.
        // QASM is not SSA. "h q[0]" modifies q[0].

        // Use the standard MLIR getResults() API which correctly handles multi-result operations
        // (operations that return tuples like %result:2)
        auto results = op->getResults();

        // Map each result to the corresponding input qubit name
        // Handle mismatch gracefully - some gates might have different numbers of outputs
        size_t numToMap = std::min(results.size(), operands.size());

        for (size_t i = 0; i < numToMap; ++i) {
            Value inQ = operands[i];
            Value outQ = results[i];

            // Check if input is in map
            bool hasMapping = qubitMap.count(inQ) > 0;
            bool hasEmptyMapping = hasMapping && qubitMap[inQ].empty();

            if (hasMapping && !hasEmptyMapping) {
                // IMPORTANT: Copy the mapped name to a local variable FIRST before inserting.
                // Direct assignment like `qubitMap[outQ] = qubitMap[inQ]` can cause issues
                // with LLVM DenseMap when the map is modified during lookup (iterator
                // invalidation).
                std::string mappedName = qubitMap[inQ];
                qubitMap[outQ] = mappedName;
            }
            else {
                // If input not mapped or has empty mapping, skip this result
                // This can happen with complex control flow patterns from quantum-opt
                // canonicalization
                llvm::errs() << "WARNING: Skipping output qubit mapping for gate " << gateName
                             << " operand " << i << " (input "
                             << (hasEmptyMapping ? "has empty mapping" : "not mapped") << ")\n";
            }
        }

        // If there are extra results (shouldn't happen for quantum gates), warn
        if (results.size() > operands.size()) {
            llvm::errs() << "WARNING: Gate " << gateName << " has " << results.size()
                         << " results but " << operands.size() << " operands\n";
        }

        return success();
    }

    LogicalResult emitMeasure(MeasureOp op)
    {
        // quantum.measure(q) -> (bit, q_out)
        // In QASM: bit c = measure q; -> bit c; c = measure q;

        Value measureQubit = op.getInQubit();
        std::string qName;
        if (qubitMap.count(measureQubit) && !qubitMap[measureQubit].empty()) {
            qName = qubitMap[measureQubit];
        }
        else {
            // Qubit not mapped - should not happen in well-formed IR
            return op.emitError("Cannot emit measure: qubit operand not mapped to QASM variable");
        }

        // If the importer tagged this measurement with its classical register,
        // assign into the pre-declared bit[n] register element. Otherwise fall
        // back to a fresh anonymous bit.
        std::string cName;
        if (auto nameAttr = op->getAttrOfType<StringAttr>("creg_name")) {
            int64_t idx = 0;
            if (auto idxAttr = op->getAttrOfType<IntegerAttr>("creg_idx"))
                idx = idxAttr.getInt();
            cName = nameAttr.getValue().str() + "[" + std::to_string(idx) + "]";
            os << cName << " = measure " << qName << ";\n";
        }
        else {
            cName = "m_" + std::to_string(qubitCounter++);
            os << "bit " << cName << ";\n" << cName << " = measure " << qName << ";\n";
        }

        // Update map for the qubit OUT state
        // In Catalyst, measure returns the qubit state as well.
        qubitMap[op.getOutQubit()] = qName;

        // Map the result bit
        bitMap[op.getResult(0)] = cName;

        return success();
    }
};

} // namespace

LogicalResult translateModuleToOpenQASM3(ModuleOp module, raw_ostream &output)
{
    QASM3Emitter emitter(output);
    return emitter.emitModule(module);
}

namespace mlir {
void registerToQASM3Translation()
{
    static TranslateFromMLIRRegistration registration(
        "mlir-to-qasm3", "Translate MLIR to OpenQASM 3.0", translateModuleToOpenQASM3,
        [](DialectRegistry &registry) {
            registry.insert<catalyst::quantum::QuantumDialect>();
            registry.insert<scf::SCFDialect>();
            registry.insert<func::FuncDialect>();
            registry.insert<arith::ArithDialect>();
        });
}
} // namespace mlir
