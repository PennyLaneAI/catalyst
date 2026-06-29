#include "Rem.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "Catalyst/Utils/CallGraph.h"
#include "mlir/IR/BuiltinAttributes.h"
// Quantum dialect types and ops
#include "Quantum/IR/QuantumOps.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "mitigation-rem"

using namespace mlir;
using namespace std::string_literals;

namespace catalyst {
namespace mitigation {

LogicalResult RemLowering::matchAndRewrite(mitigation::RemOp op, PatternRewriter &rewriter) const
{
    // The Rem lowering will produce three groups of results:
    //  1) the original callee results
    //  2) probabilities from an all-zeroes calibration circuit
    //  3) probabilities from an all-ones calibration circuit
    // For the initial implementation we create simple placeholder calibration
    // functions that return a 1-element tensor<f64>. These will be replaced
    // with proper calibration circuits in a subsequent step.
    Location loc = op.getLoc();
    
    auto computeAllZeroesOnes = op.getComputeAllZeroesOnesAttr();
    llvm::dbgs() << "[mitigation.rem] computeAllZeroesOnes: " << computeAllZeroesOnes << "\n";
    for (auto v : op.getResultTypes())
    {
        llvm::dbgs() << "[mitigation.rem] Result type: " << v << "\n";
    }    
    // Resolve callee
    func::FuncOp calleeOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());
    if (!calleeOp) {
        return rewriter.notifyMatchFailure(op, "cannot resolve callee");
    }
    
    // Call the original callee
    SmallVector<Value> callArgs(op.getArgs().begin(), op.getArgs().end());
    auto callOp = rewriter.create<func::CallOp>(loc, calleeOp, callArgs);
    
    // Get SSA Values for wrapped callee function results. Note that these are not actual values of elements, but an abstraction, Value objects and ValueRange: non-owning view of multiple Value objects. These values are not known until runtime of the compiled program (not to be confused with from pass runtime / pass time). But their abstractions can be used to define the flow of the program 
    auto results = callOp.getResults();
    // Declare vector of SSA Values representing the total result (callee results + two calibration circuit results, or zero-tensors if no calibration is present)
    SmallVector<Value> completeResults;
    // The type (and shape is part of it) of the return value is known, and we know that we have 3 times as many Value objects (because of zeros and ones calibration circuits)
    completeResults.reserve(results.size() * 3);
    
    // Determine quantum device information, specifically number of qubits and shots.
    // Get the number of qubits from quantum.alloc operation inside callee function (FuncOp). Note the due to iterator (pointer)
    const int64_t qubitCount = (*calleeOp.getOps<quantum::AllocOp>().begin()).getNqubitsAttr().value_or(0);
    // Get the device using quantom.device opeartion  inside callee function (FuncOp) 
    quantum::DeviceInitOp deviceInitOp = *(calleeOp.getOps<quantum::DeviceInitOp>().begin());
    // shots value could be a result of some operation, and not just a literal value, which is why pointer to the operation is stored
    Operation *shots = deviceInitOp.getShots().getDefiningOp();
    StringAttr lib = deviceInitOp.getLibAttr();
    StringAttr name = deviceInitOp.getDeviceNameAttr();
    StringAttr kwargs = deviceInitOp.getKwargsAttr();
    llvm::dbgs() << "[mitigation.rem] qubitCount: " << qubitCount 
                 << ", shots: " << *shots << ", lib: "<< lib
                 << ", name: " << name << ", kwargs" << kwargs << "\n";
    
    //check for MP type: Sample, Probs or Counts
    auto measurementProcesses = calleeOp.getOps<quantum::MeasurementProcess>();
    if (measurementProcesses.empty()) {
        llvm::errs() << "[mitigation.rem] No valid MP found.\n";
        return failure();
    }
    mlir::Operation* measurementProcess = (*measurementProcesses.begin()).getOperation();
    std::string MPName = measurementProcess->getName().getIdentifier().str();
    llvm::dbgs() << "[mitigation.rem] MP name: " << MPName << "\n";
    
    //pre-compute MP and attribute-dependent values
    bool doCalib = computeAllZeroesOnes.getValue(); //get boolean value of BoolAttr
    mlir::RankedTensorType tensorTyI64;
    mlir::RankedTensorType tensorTyF64;
    int64_t calibrationTensorShape = 0;
    int64_t shotCount = 0;
    if (MPName == "quantum.probs"s || MPName == "quantum.counts"s) {
        // avoid undefined behaviour due to integer overflow
        if (qubitCount >= 63) {
            // in practice, due to exponential memory growth, users will
            // OOM much earlier, at < 30 qubits
            llvm::errs() << "[mitigation.rem] Qubit count" << qubitCount
                         << "exceeds maximum simulated register size (62 qubits).\n";
        }
        calibrationTensorShape = static_cast<int64_t>(1ULL << qubitCount);
        tensorTyF64 = RankedTensorType::get({calibrationTensorShape}, rewriter.getF64Type());
        tensorTyI64 = RankedTensorType::get({calibrationTensorShape}, rewriter.getI64Type());
    }
    else if (MPName == "quantum.sample"s) {
        // check if shots Op is actually a arith.constant. In that case, shots are known at compile time
        if (shots->getName().getIdentifier() == arith::ConstantOp::getOperationName()) {
            auto shotsAttr = shots->getAttrOfType<IntegerAttr>("value");
            if (!shotsAttr || shotsAttr == 0) {
                llvm::errs() << "[mitigation.rem] Shots constant missing non-zero integer value. Sample MP not supported for analytic simulation (shots must be > 0).\n";
                return failure();
            }
            shotCount = shotsAttr.getInt();
            llvm::dbgs() << "[mitigation.rem] Shots is a arth.constant op, shots known at compile time: " << shotCount << "\n";
            calibrationTensorShape = shotCount * qubitCount;
            tensorTyI64 = RankedTensorType::get({shotCount, qubitCount}, rewriter.getI64Type());
            tensorTyF64 = RankedTensorType::get({shotCount, qubitCount}, rewriter.getF64Type());
        }
        else {
            llvm::errs() << "[mitigation.rem] Dynamic number of shots not supported currently.\n";
            return failure();
        }
        
        // SmallVector<double> zeros_vec(bitspaceShape, 0.0); //initialize with zeroes
    }
    else {
        llvm::errs() << "[mitigation.rem] Supported measurement processes are quantum.counts, quantum.probs and quantum.sample. "
                     << MPName << " is not supported.\n";
        return failure();
    }
    if (!doCalib) { //doCalib == false
        llvm::dbgs() << "[mitigation.rem] doCalibration == false, replacing RemOp with wrapped callee circuit function call..." << "\n";
        mlir::RankedTensorType tensorTyConst;
        // Create constant array of type ranked tensor<Nxf64> initialized with 0.0. This is the default value returned when doCalib == false and indicates that no calibration circuits were run
        // auto tensorTy = RankedTensorType::get({bitspaceShape}, rewriter.getF64Type());
        // SmallVector<double> zeros_vec(bitspaceShape, 0.0); //initialize with zeroes
        DenseElementsAttr tensorAttr;
        if (MPName == "quantum.probs"s || MPName == "quantum.counts"s) {
            SmallVector<double> zerosVector(calibrationTensorShape, 0.0); //initialize with zeroes
            // To initialize with other values:
            // for (auto i = 0; i < 4; ++i) {
            //     zeros_vec[i] = static_cast<double>(i);
            // }
            // -----------------------------------------------------------------
            // ArrayRef is a non-owning view and needs to be created because it is often required by MLIR APIs to avoid copies.
            // It is also possible to use this:
            // auto zerosAttr = rewriter.getF64ArrayAttr(zeros_vec); // returns ArrayAttr used by DenseElementsAttr
            // because there is also an overload of DenseElementsAttr which accepts ArrayAttr. Attributes and compile-time constant metadata attached to ops/types in the IR, it requires a static shape
            tensorTyConst = tensorTyF64;
            tensorAttr = DenseElementsAttr::get(tensorTyConst, ArrayRef<double>{zerosVector.begin(), zerosVector.end()});
            //tensorAttr is an attribute attached with the newly created airth.constant operation on the line below. It represents values known at pass-time (compile-time with regards to IR generation, but could be inferred at compile time of the passs from other attributes or something else)
        }
        else if (MPName == "quantum.sample"s) {
            SmallVector<int64_t> zerosVector(calibrationTensorShape, 0); //initialize with zeroes
            tensorTyConst = tensorTyI64;
            tensorAttr = DenseElementsAttr::get(tensorTyConst, ArrayRef<int64_t>{zerosVector.begin(), zerosVector.end()});
        } 
        auto cst_zeros = rewriter.create<arith::ConstantOp>(loc, tensorTyConst, tensorAttr);
        llvm::dbgs() << "[mitigation.rem] cst_zeros = " << cst_zeros << "\n";

        // 1) original callee results
        for (Value v : results)
            completeResults.push_back(v);
        // 2) zeros calibration placeholders (reuse same tensors for now)
        completeResults.push_back(cst_zeros.getResult());
        // 3) ones calibration placeholders 
        completeResults.push_back(cst_zeros.getResult());
        // Replace the results of the original operation (mitigation.rem) with this vector of Values (could be any object castable to ValueRange)
        rewriter.replaceOp(op, completeResults);
        return success();
    }
    // else, doCalib == true
    llvm::dbgs() << "[mitigation.rem] doCalibration == true, start adding all-zeroes and all-ones circuits..." << "\n";
    //
    //DONE: implement adding two calibation circuits (all-zeroes and all-ones) and return appropriate results
    //DONE: Then, add support for other MP, such as CountsMP and change the return type accordingly.
    // Finally, add support for returning Observables, by somehow injecting the mitigation code (applying the confusion matrix) before observable calculation. (out of scope for now)
    // 0. Check how insertion point is managed -- for now it seems to be just fine as it is, but this must be researched in the future
    // ===BEGIN ZEROES QFUNC===
    // 1. Add device creation and allocation (quantum.device and quantum.alloc with previously determined attributes)
    llvm::dbgs() << "[mitigation.rem] doCalibration == true, start adding all-zeroes and all-ones circuits..." << "\n";
    Operation *shotsLocal = shots->clone(); // not sure why this is needed
    rewriter.insert(shotsLocal); // not sure why this is needed
    auto devInit = rewriter.create<quantum::DeviceInitOp>(loc, shotsLocal->getResult(0), lib, name, kwargs);
    Type qregType = quantum::QuregType::get(rewriter.getContext());
    // not sure why empty IntegerAttr is needed, probably need to research more about how quantum.alloc works
    IntegerAttr qubitCountAttr = rewriter.getI64IntegerAttr(qubitCount);
    Value numberQubitsValue = rewriter.create<arith::ConstantOp>(loc, qubitCountAttr);
    // IntegerAttr intAttr{};
    llvm::dbgs() << "[mitigation.rem] quantum.device initialization OK, op=" << devInit << "\n";
    auto qreg = rewriter.create<quantum::AllocOp>(loc, qregType, numberQubitsValue, qubitCountAttr);
    llvm::dbgs() << "[mitigation.rem] quantum.alloc addition OK, op=" << qreg << "\n";
    // 2. Add quantum.compbasis op with the quantum observation result type and
    // the allocated quantum register Value as its operand. Use the dialect's
    // observation type and pass the AllocOp result Value (qreg.getResult())
    // as the operand; passing the wrong type or an incorrect operand can
    // cause an `operandSegmentSizes` attribute to appear in the printed IR.
    Type obsType = ::catalyst::quantum::ObservableType::get(rewriter.getContext());
    Value qregVal = qreg.getResult();
    // ComputationalBasisOp has the signature build(OpBuilder, OperationState, Type obs,
    // ValueRange qubits, optional Value qreg). To create the "qreg" form (measure
    // the entire register), pass an empty qubits ValueRange and the qreg Value as
    // the optional argument.
    // Use the TypeRange overload to ensure the builder picks the overload that
    // treats the first argument as the result type (observable) rather than
    // mistakenly interpreting operands. This avoids accidental operand ordering
    // that can produce explicit operandSegmentSizes or wrong printed form.
    auto compbasis = rewriter.create<quantum::ComputationalBasisOp>(loc, TypeRange{obsType}, ValueRange{}, qregVal);
    llvm::dbgs() << "[mitigation.rem] quantum.compbasis addition OK, op=" << compbasis << "\n";
    // 3. Add quantum.probs op from result of compbasis and get Value of tensor
    // Probs returns a tensor of size 2^nqubits; build the appropriate RankedTensorType
    // const int64_t bitspaceShape = 2 << (qubitCount - 1);
    // auto probsTensorTy = RankedTensorType::get({bitspaceShape}, rewriter.getF64Type());
    // Be explicit about the builder overload to avoid overload-resolution
    // ambiguity that can lead to incorrect operandSegmentSizes being set.
    // Use the TypeRange overload so the result type is unambiguous and the
    // `obs` operand is passed in the correct position.
    // Pass explicit placeholders for the optional `dynamic_shape` and `state_in`
    // operands (as null Values) so that the ODS builder records the correct
    // operandSegmentSizes = {1, 0, 0} and the assembly printer can elide that
    // property producing a clean printed form.
    quantum::MeasurementProcess insertedMP;
    ValueRange calibratedZeroResult;
    if (MPName == "quantum.probs"s) {
        insertedMP = rewriter.create<quantum::ProbsOp>(loc, TypeRange{tensorTyF64}, compbasis.getResult(), Value(), Value());
        llvm::dbgs() << "[mitigation.rem] quantum.probs addition OK, op=" << insertedMP << "\n";
        calibratedZeroResult = insertedMP->getResults();
    }
    else if (MPName == "quantum.counts"s) {
        insertedMP = rewriter.create<quantum::CountsOp>(loc, TypeRange{tensorTyF64, tensorTyI64}, compbasis.getResult(), Value(), Value(), Value());
        llvm::dbgs() << "[mitigation.rem] quantum.counts addition OK, op=" << insertedMP << "\n";
        calibratedZeroResult = insertedMP->getResults().drop_front(); // drop the first element in the iterator (eigvals not used for calibration matrices)
    }
    else if (MPName == "quantum.sample"s) {
        // auto tensorTyFloat = RankedTensorType::get({shotCount, qubitCount}, rewriter.getF64Type());
        insertedMP = rewriter.create<quantum::SampleOp>(loc, TypeRange{tensorTyF64}, compbasis.getResult(), ValueRange{}, Value());// (loc, TypeRange{tensorTy}, compbasis.getResult(), Value(), Value());
        llvm::dbgs() << "[mitigation.rem] quantum.sample addition OK, op=" << insertedMP << "\n";
        IRMapping postprocMapping;
        uint64_t resultIndex = 0;
        for (auto res : measurementProcess->getResults())
            postprocMapping.map(res, insertedMP->getResult(resultIndex++));
        // OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(insertedMP.getOperation());
        calibratedZeroResult = insertedMP->getResults();
        for (auto postprocOp = measurementProcess->getNextNode(); postprocOp != nullptr && postprocOp->getName().getIdentifier() != quantum::DeallocOp::getOperationName(); postprocOp = postprocOp->getNextNode()) {
            llvm::dbgs() << "[mitigation.rem] looking for next op: " << postprocOp->getName().getIdentifier() << "\n";
            Operation *insertedOp = rewriter.clone(*postprocOp, postprocMapping);
            if (!insertedOp->getResults().empty())
                calibratedZeroResult = insertedOp->getResults();
            rewriter.setInsertionPointAfter(insertedOp);
            llvm::dbgs() << "[mitigation.rem] insertedOp: " << insertedOp->getName().getIdentifier() << "\n";
            // loc = insertedOp->getLoc();
        }
    }
    //MPName is already guaranteed to be a valid value, so no else block here. 

    // 4. Deallocate everything
    auto dealloc = rewriter.create<quantum::DeallocOp>(loc, qreg);
    llvm::dbgs() << "[mitigation.rem] quantum.dealloc addition OK, op=" << dealloc << "\n";
    auto deinit = rewriter.create<quantum::DeviceReleaseOp>(loc);
    llvm::dbgs() << "[mitigation.rem] quantum.deinit addition OK, op=" << deinit << "\n";
    // ===END ZEROES QFUNC===
    // 5. Add quantum.custom 'X' gates to implement ones circuit
    // ===BEGIN ONES QFUNC===
    // Reuse numberQubitsValue and qubitCountAttr created above. Clone shots and create a new device/alloc sequence for the ones circuit.
    Operation *shotsLocalOnes = shots->clone();
    rewriter.insert(shotsLocalOnes);
    auto devInitOnes = rewriter.create<quantum::DeviceInitOp>(loc, shotsLocalOnes->getResult(0), lib, name, kwargs);
    llvm::dbgs() << "[mitigation.rem] quantum.device initialization (ones) OK, op=" << devInitOnes << "\n";
    auto qregOnes = rewriter.create<quantum::AllocOp>(loc, qregType, numberQubitsValue, qubitCountAttr);
    llvm::dbgs() << "[mitigation.rem] quantum.alloc (ones) addition OK, op=" << qregOnes << "\n";

    // For each qubit in the register: extract -> apply PauliX -> insert back
    // Use a Value to track the current register SSA value so each insert
    // produces a new SSA Value derived from the previous one. Reusing the
    // previous InsertOp object (or its Op pointer) risks confusing the
    // builder/operands and producing identical SSA uses.
    Value currentQreg = qregOnes.getResult();
    for (int64_t i = 0; i < qubitCount; ++i) {
        auto idxAttr = rewriter.getI64IntegerAttr(i);
        // Extract qubit
        auto extracted = rewriter.create<quantum::ExtractOp>(loc, rewriter.getType<quantum::QubitType>(), currentQreg, nullptr, idxAttr);
        // Apply PauliX (1-qubit gate)
        auto xgate = rewriter.create<quantum::CustomOp>(loc, /*gate_name=*/"PauliX", mlir::ValueRange({extracted.getResult()}));
        // Insert back into the register, producing a new register Value
        auto inserted = rewriter.create<quantum::InsertOp>(loc, qregOnes.getType(), currentQreg, nullptr, idxAttr, xgate.getResult(0));
        currentQreg = inserted.getResult();
    }

    // Measure and get probs for ones circuit
    Value qregValOnes = currentQreg;
    auto compbasisOnes = rewriter.create<quantum::ComputationalBasisOp>(loc, TypeRange{obsType}, ValueRange{}, qregValOnes);
    llvm::dbgs() << "[mitigation.rem] quantum.compbasis (ones) addition OK, op=" << compbasisOnes << "\n";
    //TODO: maybe refactor into a function like in ZNE and call for zeroes and ones
    quantum::MeasurementProcess insertedMPOnes;
    ValueRange calibratedOnesResult{};
    if (MPName == "quantum.probs"s) {
        insertedMPOnes = rewriter.create<quantum::ProbsOp>(loc, TypeRange{tensorTyF64}, compbasisOnes.getResult(), Value(), Value());
        llvm::dbgs() << "[mitigation.rem] quantum.probs (ones) addition OK, op=" << insertedMPOnes << "\n";
        calibratedOnesResult = insertedMPOnes->getResults();
    }
    if (MPName == "quantum.counts"s) {
        insertedMPOnes = rewriter.create<quantum::CountsOp>(loc, TypeRange{tensorTyF64, tensorTyI64}, compbasisOnes.getResult(), Value(), Value(), Value());
        llvm::dbgs() << "[mitigation.rem] quantum.counts (ones) addition OK, op=" << insertedMPOnes << "\n";
        calibratedOnesResult = insertedMPOnes->getResults().drop_front(); // drop the first element in the iterator (eigvals not used for calibration matrices)
    }
    else if ((MPName == "quantum.sample"s)) {
        // auto tensorTyFloat = RankedTensorType::get({shotCount, qubitCount}, rewriter.getF64Type());
        insertedMPOnes = rewriter.create<quantum::SampleOp>(loc, TypeRange{tensorTyF64}, compbasisOnes.getResult(), ValueRange{}, Value());// (loc, TypeRange{tensorTy}, compbasis.getResult(), Value(), Value());
        llvm::dbgs() << "[mitigation.rem] quantum.sample addition OK, op=" << insertedMPOnes << "\n";
        IRMapping postprocMapping;
        uint64_t resultIndex = 0;
        for (auto res : measurementProcess->getResults())
            postprocMapping.map(res, insertedMPOnes->getResult(resultIndex++));
        // OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(insertedMPOnes.getOperation());
        calibratedOnesResult = insertedMPOnes->getResults();
        for (auto postprocOp = measurementProcess->getNextNode(); postprocOp != nullptr && postprocOp->getName().getIdentifier() != quantum::DeallocOp::getOperationName(); postprocOp = postprocOp->getNextNode()) {
            llvm::dbgs() << "[mitigation.rem] looking for next op: " << postprocOp->getName().getIdentifier() << "\n";
            Operation *insertedOp = rewriter.clone(*postprocOp, postprocMapping);
            if (!insertedOp->getResults().empty())
                calibratedOnesResult = insertedOp->getResults();
            rewriter.setInsertionPointAfter(insertedOp);
            // loc = insertedOp->getLoc();
            llvm::dbgs() << "[mitigation.rem] insertedOp: " << insertedOp->getName().getIdentifier() << "\n";
        }        
    }
    auto deallocOnes = rewriter.create<quantum::DeallocOp>(loc, qregOnes);
    llvm::dbgs() << "[mitigation.rem] quantum.dealloc (ones) addition OK, op=" << deallocOnes << "\n";
    auto deinitOnes = rewriter.create<quantum::DeviceReleaseOp>(loc);
    llvm::dbgs() << "[mitigation.rem] quantum.deinit (ones) addition OK, op=" << deinitOnes << "\n";
    // ===END ONES QFUNC===
    // If I'll ever need to create integer constants later, this is how it's done:
    // TypedAttr numberQubitsAttr = rewriter.getI64IntegerAttr(numberQubits);
    // Value numberQubitsValue = rewriter.create<arith::ConstantOp>(loc, numberQubitsAttr);

    // placeholder fallback: return 3 groups (callee, zeros, ones) by
    // replicating the call results. Each returned Value is an SSA use of
    // the same tensor produced by the callee; this yields three separate
    // results in the caller, each referencing the same underlying tensor.

    // 1) original callee results
    for (Value v : results)
        completeResults.push_back(v);
    // 2) zeros calibration result
    completeResults.push_back(*calibratedZeroResult.begin());
    // 3) ones calibration result
    completeResults.push_back(*calibratedOnesResult.begin());
    
    rewriter.replaceOp(op, completeResults);
    return success();

    // // Determine device qubit count if present on the op. This controls the
    // // size of the probability vector returned by the calibration circuits
    // // (2^n_qubits). If absent, fallback to a single-element tensor used
    // // during iterative development.
    // int64_t nqubits = 0;
    // if (auto attr = op->getAttrOfType<IntegerAttr>("deviceNumQubits")) {
    //     nqubits = attr.getInt();
    // }
    // size_t probSize = 1;
    // if (nqubits > 0) {
    //     // cap shifts to avoid UB for large n
    //     if (nqubits < 63)
    //         probSize = (size_t)1 << static_cast<size_t>(nqubits);
    //     else
    //         probSize = 1; // fallback
    // }

    // // Helper to get or create a simple calibration function that returns
    // // tensor<probSize x f64> with a constant one-hot vector. The function
    // // is inserted into the parent module.
    // auto getOrCreateCalib = [&](StringRef suffix, size_t size, double constPos) -> func::FuncOp {
    //     auto module = op->getParentOfType<ModuleOp>();
    //     assert(module && "rem op must be inside a module");
    //     std::string name = (calleeOp.getName().str() + ".rem_calib_") + suffix.str();

    //     // If function already exists, return it.
    //     if (auto existing = module.lookupSymbol<func::FuncOp>(name))
    //         return existing;

    //     // Create function type: () -> tensor<size x f64>
    //     auto f64 = rewriter.getF64Type();
    //     auto tensorTy = RankedTensorType::get({static_cast<int64_t>(size)}, f64);
    //     auto fnType = FunctionType::get(rewriter.getContext(), /*inputs=*/{}, /*results=*/{tensorTy});

    //     OpBuilder::InsertionGuard guard(rewriter);
    // // Insert the new function at the start of the module body
    // rewriter.setInsertionPointToStart(module.getBody());
    // auto func = rewriter.create<func::FuncOp>(loc, name, fnType);
    //     // Add an entry block
    //     auto *block = func.addEntryBlock();

    //     // Create a dense constant vector with a single 1.0 in the requested
    //     // position (constPos interpreted as index position) and zeros elsewhere.
    //     SmallVector<double> elems(size, 0.0);
    //     size_t pos = 0;
    //     if (size > 0) {
    //         if (constPos <= 0.0)
    //             pos = 0;
    //         else
    //             pos = static_cast<size_t>(constPos) < size ? static_cast<size_t>(constPos) : size - 1;
    //     }
    //     elems[pos] = 1.0;
    //     auto attr = DenseElementsAttr::get(tensorTy, rewriter.getF64ArrayAttr(elems));
    //     rewriter.setInsertionPointToStart(block);
    //     auto cst = rewriter.create<arith::ConstantOp>(loc, tensorTy, attr);
    //     Value constValRes = cst.getResult();
    //     rewriter.create<func::ReturnOp>(loc, constValRes);
    //     return func;
    // };

    // // Create/get calibration functions
    // func::FuncOp allZeros = getOrCreateCalib("all_zeroes", 0.0);
    // func::FuncOp allOnes = getOrCreateCalib("all_ones", 1.0);

    // // Call the calibration functions
    // auto callZeros = rewriter.create<func::CallOp>(loc, allZeros, ArrayRef<Value>{});
    // auto callOnes = rewriter.create<func::CallOp>(loc, allOnes, ArrayRef<Value>{});

    // // Aggregate results: original callee results followed by zeros and ones results
    // SmallVector<Value> results;
    // for (auto r : originalCall.getResults())
    //     results.push_back(r);
    // for (auto r : callZeros.getResults())
    //     results.push_back(r);
    // for (auto r : callOnes.getResults())
    //     results.push_back(r);

    // // Decide whether to emit calibration circuits based on the attribute.
    // // If the frontend requested no calibration runs, simply forward the
    // // callee results (preserving the previous behavior).
    // bool doCalib = false;
    // if (computeAllZeroesOnes) {
    //     // computeAllZeroesOnes is a BoolAttr; retrieve its value if present.
    //     doCalib = computeAllZeroesOnes.getValue();
    // }

    // // Otherwise, we will produce callee results + two calibration results.
    // // Replace the operation with the full set of produced values.
    // rewriter.replaceOp(op, results);
    // return success();
}

void populateRemLoweringPatterns(RewritePatternSet &patterns)
{
    patterns.add<RemLowering>(patterns.getContext());
}

} // namespace mitigation
} // namespace catalyst
