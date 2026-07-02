#include "Rem.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"

#include "Catalyst/Utils/CallGraph.h"
// Quantum dialect types and ops
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "Quantum/IR/QuantumOps.h"

#define DEBUG_TYPE "mitigation-rem"

using namespace mlir;
using namespace std::string_literals;

namespace catalyst {
namespace mitigation {

namespace {

// Emit one calibration circuit next to the user callee. It reuses the callee's
// `quantum.device` kwargs verbatim (so it inherits the same noise channel) and,
// when `applyPauliX` is true, applies a PauliX gate on every qubit before
// compbasis to produce the all-ones basis state. Returns the calibration result
// values (drops the eigvals half of CountsOp).
ValueRange emitCalibrationCircuit(PatternRewriter &rewriter, Location loc, func::FuncOp calleeOp,
                                  RankedTensorType tensorTyF64, RankedTensorType tensorTyI64,
                                  bool applyPauliX)
{
    const char *tag = applyPauliX ? "(ones)" : "(zeros)";

    const int64_t qubitCount =
        (*calleeOp.getOps<quantum::AllocOp>().begin()).getNqubitsAttr().value_or(0);
    quantum::DeviceInitOp deviceInitOp = *(calleeOp.getOps<quantum::DeviceInitOp>().begin());
    Operation *shots = deviceInitOp.getShots().getDefiningOp();
    StringAttr lib = deviceInitOp.getLibAttr();
    StringAttr name = deviceInitOp.getDeviceNameAttr();
    StringAttr kwargs = deviceInitOp.getKwargsAttr();

    Operation *measurementProcess =
        (*calleeOp.getOps<quantum::MeasurementProcess>().begin()).getOperation();
    const std::string MPName = measurementProcess->getName().getIdentifier().str();

    Operation *shotsLocal = shots->clone();
    rewriter.insert(shotsLocal);
    auto devInit =
        rewriter.create<quantum::DeviceInitOp>(loc, shotsLocal->getResult(0), lib, name, kwargs);
    llvm::dbgs() << "[mitigation.rem] quantum.device initialization " << tag
                 << " OK, op=" << devInit << "\n";

    IntegerAttr qubitCountAttr = rewriter.getI64IntegerAttr(qubitCount);
    Value numberQubitsValue = rewriter.create<arith::ConstantOp>(loc, qubitCountAttr);
    Type qregType = quantum::QuregType::get(rewriter.getContext());
    auto qreg = rewriter.create<quantum::AllocOp>(loc, qregType, numberQubitsValue, qubitCountAttr);
    llvm::dbgs() << "[mitigation.rem] quantum.alloc " << tag << " addition OK, op=" << qreg << "\n";

    Value currentQreg = qreg.getResult();
    if (applyPauliX) {
        // For each qubit: extract -> PauliX -> insert. Each insert produces a
        // fresh register SSA Value so the chain threads correctly.
        for (int64_t i = 0; i < qubitCount; ++i) {
            auto idxAttr = rewriter.getI64IntegerAttr(i);
            auto extracted = rewriter.create<quantum::ExtractOp>(
                loc, rewriter.getType<quantum::QubitType>(), currentQreg, nullptr, idxAttr);
            auto xgate = rewriter.create<quantum::CustomOp>(
                loc, /*gate_name=*/"PauliX", mlir::ValueRange({extracted.getResult()}));
            auto inserted = rewriter.create<quantum::InsertOp>(
                loc, qreg.getType(), currentQreg, nullptr, idxAttr, xgate.getResult(0));
            currentQreg = inserted.getResult();
        }
    }

    Type obsType = ::catalyst::quantum::ObservableType::get(rewriter.getContext());
    auto compbasis = rewriter.create<quantum::ComputationalBasisOp>(loc, TypeRange{obsType},
                                                                    ValueRange{}, currentQreg);
    llvm::dbgs() << "[mitigation.rem] quantum.compbasis " << tag << " addition OK, op=" << compbasis
                 << "\n";

    ValueRange calibratedResult;
    if (MPName == "quantum.probs"s) {
        auto insertedMP = rewriter.create<quantum::ProbsOp>(
            loc, TypeRange{tensorTyF64}, compbasis.getResult(), Value(), Value());
        llvm::dbgs() << "[mitigation.rem] quantum.probs " << tag
                     << " addition OK, op=" << insertedMP << "\n";
        calibratedResult = insertedMP->getResults();
    }
    else if (MPName == "quantum.counts"s) {
        auto insertedMP =
            rewriter.create<quantum::CountsOp>(loc, TypeRange{tensorTyF64, tensorTyI64},
                                               compbasis.getResult(), Value(), Value(), Value());
        llvm::dbgs() << "[mitigation.rem] quantum.counts " << tag
                     << " addition OK, op=" << insertedMP << "\n";
        // Drop the eigvals half; only the counts tensor feeds calibration.
        calibratedResult = insertedMP->getResults().drop_front();
    }
    else if (MPName == "quantum.sample"s) {
        auto insertedMP = rewriter.create<quantum::SampleOp>(
            loc, TypeRange{tensorTyF64}, compbasis.getResult(), ValueRange{}, Value());
        llvm::dbgs() << "[mitigation.rem] quantum.sample " << tag
                     << " addition OK, op=" << insertedMP << "\n";
        IRMapping postprocMapping;
        uint64_t resultIndex = 0;
        for (auto res : measurementProcess->getResults())
            postprocMapping.map(res, insertedMP->getResult(resultIndex++));
        rewriter.setInsertionPointAfter(insertedMP.getOperation());
        calibratedResult = insertedMP->getResults();
        for (auto postprocOp = measurementProcess->getNextNode();
             postprocOp != nullptr &&
             postprocOp->getName().getIdentifier() != quantum::DeallocOp::getOperationName();
             postprocOp = postprocOp->getNextNode()) {
            llvm::dbgs() << "[mitigation.rem] looking for next op: "
                         << postprocOp->getName().getIdentifier() << "\n";
            Operation *insertedOp = rewriter.clone(*postprocOp, postprocMapping);
            if (!insertedOp->getResults().empty())
                calibratedResult = insertedOp->getResults();
            rewriter.setInsertionPointAfter(insertedOp);
            llvm::dbgs() << "[mitigation.rem] insertedOp: " << insertedOp->getName().getIdentifier()
                         << "\n";
        }
    }

    auto dealloc = rewriter.create<quantum::DeallocOp>(loc, qreg);
    llvm::dbgs() << "[mitigation.rem] quantum.dealloc " << tag << " addition OK, op=" << dealloc
                 << "\n";
    auto deinit = rewriter.create<quantum::DeviceReleaseOp>(loc);
    llvm::dbgs() << "[mitigation.rem] quantum.deinit " << tag << " addition OK, op=" << deinit
                 << "\n";
    return calibratedResult;
}

} // namespace

LogicalResult RemLowering::matchAndRewrite(mitigation::RemOp op, PatternRewriter &rewriter) const
{
    // The Rem lowering will produce three groups of results:
    //  1) the original callee results
    //  2) probabilities from an all-zeroes calibration circuit
    //  3) probabilities from an all-ones calibration circuit
    Location loc = op.getLoc();

    auto runCalibration = op.getRunCalibrationAttr();
    llvm::dbgs() << "[mitigation.rem] runCalibration: " << runCalibration << "\n";
    for (auto v : op.getResultTypes()) {
        llvm::dbgs() << "[mitigation.rem] Result type: " << v << "\n";
    }
    // Resolve callee
    func::FuncOp calleeOp =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());
    if (!calleeOp) {
        return rewriter.notifyMatchFailure(op, "cannot resolve callee");
    }

    // Call the original callee
    SmallVector<Value> callArgs(op.getArgs().begin(), op.getArgs().end());
    auto callOp = rewriter.create<func::CallOp>(loc, calleeOp, callArgs);

    auto results = callOp.getResults();
    // Declare vector of SSA Values representing the total result (callee results + two calibration
    // circuit results, or zero-tensors if no calibration is present)
    SmallVector<Value> completeResults;
    // The type (and shape is part of it) of the return value is known, and we know that we have 3
    // times as many Value objects (because of zeros and ones calibration circuits)
    completeResults.reserve(results.size() * 3);

    const bool doCalib = runCalibration.getValue();
    if (!doCalib) {
        llvm::dbgs() << "[mitigation.rem] doCalibration == false, replacing RemOp with wrapped "
                        "callee circuit function call..."
                     << "\n";
        // The zeros/ones result groups are empty variadics when doCalib is false,
        // so the op is replaced by only the callee's SSA values.
        for (Value v : results)
            completeResults.push_back(v);
        rewriter.replaceOp(op, completeResults);
        return success();
    }

    // Determine quantum device information, specifically number of qubits and shots.
    // Get the number of qubits from quantum.alloc operation inside callee function (FuncOp).
    const int64_t qubitCount =
        (*calleeOp.getOps<quantum::AllocOp>().begin()).getNqubitsAttr().value_or(0);
    // Get the device using quantom.device opeartion  inside callee function (FuncOp)
    quantum::DeviceInitOp deviceInitOp = *(calleeOp.getOps<quantum::DeviceInitOp>().begin());
    // shots value could be a result of some operation, and not just a literal value, which is why
    // pointer to the operation is stored
    Operation *shots = deviceInitOp.getShots().getDefiningOp();
    llvm::dbgs() << "[mitigation.rem] qubitCount: " << qubitCount << ", shots: " << *shots
                 << ", lib: " << deviceInitOp.getLibAttr()
                 << ", name: " << deviceInitOp.getDeviceNameAttr() << ", kwargs"
                 << deviceInitOp.getKwargsAttr() << "\n";

    // check for MP type: Sample, Probs or Counts
    auto measurementProcesses = calleeOp.getOps<quantum::MeasurementProcess>();
    if (measurementProcesses.empty()) {
        llvm::errs() << "[mitigation.rem] No valid MP found.\n";
        return failure();
    }
    const std::string MPName =
        (*measurementProcesses.begin()).getOperation()->getName().getIdentifier().str();
    llvm::dbgs() << "[mitigation.rem] MP name: " << MPName << "\n";

    // pre-compute MP and attribute-dependent values
    RankedTensorType tensorTyI64;
    RankedTensorType tensorTyF64;
    if (MPName == "quantum.probs"s || MPName == "quantum.counts"s) {
        // avoid undefined behaviour due to integer overflow
        if (qubitCount >= 63) {
            // in practice, due to exponential memory growth, users will
            // OOM much earlier, at < 30 qubits
            llvm::errs() << "[mitigation.rem] Qubit count" << qubitCount
                         << "exceeds maximum simulated register size (62 qubits).\n";
        }
        const int64_t calibrationTensorShape = static_cast<int64_t>(1ULL << qubitCount);
        tensorTyF64 = RankedTensorType::get({calibrationTensorShape}, rewriter.getF64Type());
        tensorTyI64 = RankedTensorType::get({calibrationTensorShape}, rewriter.getI64Type());
    }
    else if (MPName == "quantum.sample"s) {
        // shots must be a compile-time arith.constant with a non-zero value.
        if (shots->getName().getIdentifier() != arith::ConstantOp::getOperationName()) {
            llvm::errs() << "[mitigation.rem] Dynamic number of shots not supported currently.\n";
            return failure();
        }
        auto shotsAttr = shots->getAttrOfType<IntegerAttr>("value");
        if (!shotsAttr || shotsAttr == 0) {
            llvm::errs()
                << "[mitigation.rem] Shots constant missing non-zero integer value. "
                   "Sample MP not supported for analytic simulation (shots must be > 0).\n";
            return failure();
        }
        const int64_t shotCount = shotsAttr.getInt();
        llvm::dbgs()
            << "[mitigation.rem] Shots is a arth.constant op, shots known at compile time: "
            << shotCount << "\n";
        tensorTyI64 = RankedTensorType::get({shotCount, qubitCount}, rewriter.getI64Type());
        tensorTyF64 = RankedTensorType::get({shotCount, qubitCount}, rewriter.getF64Type());
    }
    else {
        llvm::errs() << "[mitigation.rem] Supported measurement processes are quantum.counts, "
                        "quantum.probs and quantum.sample. "
                     << MPName << " is not supported.\n";
        return failure();
    }

    llvm::dbgs() << "[mitigation.rem] doCalibration == true, start adding all-zeroes and all-ones "
                    "circuits..."
                 << "\n";

    ValueRange calibratedZeroResult =
        emitCalibrationCircuit(rewriter, loc, calleeOp, tensorTyF64, tensorTyI64,
                               /*applyPauliX=*/false);
    ValueRange calibratedOnesResult =
        emitCalibrationCircuit(rewriter, loc, calleeOp, tensorTyF64, tensorTyI64,
                               /*applyPauliX=*/true);

    // 1) original callee results
    for (Value v : results)
        completeResults.push_back(v);
    // 2) zeros calibration result
    completeResults.push_back(*calibratedZeroResult.begin());
    // 3) ones calibration result
    completeResults.push_back(*calibratedOnesResult.begin());

    rewriter.replaceOp(op, completeResults);
    return success();
}

void populateRemLoweringPatterns(RewritePatternSet &patterns)
{
    patterns.add<RemLowering>(patterns.getContext());
}

} // namespace mitigation
} // namespace catalyst
