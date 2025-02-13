// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <fcntl.h> // file open()
#include <string>
#include <unistd.h> // file close()

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"

#include "Quantum/IR/QuantumOps.h"

#include "capnp/message.h"
#include "capnp/serialize.h"
#include "qpr.capnp.h"

using namespace mlir;
using namespace catalyst;

class QPRSerializer {
    capnp::MallocMessageBuilder message;
    Module::Builder module = nullptr;

    std::vector<func::FuncOp> moduleFunctions;
    std::vector<std::string> moduleStrings;

  public:
    void addFunction(func::FuncOp fun) { moduleFunctions.push_back(fun); }

    void generateModule()
    {
        module = message.initRoot<Module>();

        auto functions = module.initFunctions(moduleFunctions.size());
        unsigned int funIdx = 0;
        for (func::FuncOp fun : moduleFunctions) {
            Function::Builder function = functions[funIdx++];
            generateFunction(fun, function);
        }

        auto strings = module.initStrings(moduleStrings.size());
        unsigned int stringIdx = 0;
        for (std::string &string : moduleStrings) {
            strings.set(stringIdx++, string);
        }
    }

    void generateFunction(func::FuncOp fun, Function::Builder &function)
    {
        // Ordered list of values populated during traversal. They will be copied into value storage
        // in the gathered order. Note that the same value can appear multiple times, contrary to
        // the value map used within a region.
        std::vector<mlir::Value> functionValues;

        moduleStrings.emplace_back(fun.getName());
        function.setName(moduleStrings.size() - 1);

        // populate the function body
        ::Region::Builder region = function.initBody();
        generateRegion(fun.getBody(), region, functionValues);

        // populate the values
        auto values = function.initValues(functionValues.size());
        unsigned int valueIdx = 0;
        for (mlir::Value val : functionValues) {
            ::Type::Builder type = values[valueIdx++].initType();
            generateType(val.getType(), type);
        }
    }

    void generateRegion(mlir::Region &body, ::Region::Builder &region,
                        std::vector<mlir::Value> &functionValues)
    {
        // Map MLIR values to their storage index in the current QPR function.
        llvm::DenseMap<mlir::Value, unsigned int> mlirValueMap;

        auto &mlirOps = body.front().getOperations(); // assume no CFG
        mlir::Operation *terminator = body.back().getTerminator();

        auto operations = region.initOperations(mlirOps.size() - 1); // exclude terminator
        auto sources = region.initSources(body.getNumArguments());
        auto targets = region.initTargets(terminator->getNumOperands());

        // set the sources
        unsigned int sourceIdx = 0;
        for (mlir::Value arg : body.getArguments()) {
            sources.set(sourceIdx++, trackNewValue(arg, functionValues, mlirValueMap));
        }

        // process operations
        unsigned int opIdx = 0;
        for (mlir::Operation &op : mlirOps) {
            if (!op.hasTrait<OpTrait::IsTerminator>()) {
                ::Op::Builder operation = operations[opIdx++];
                generateOp(op, operation, functionValues, mlirValueMap);
            }
        }

        // set the targets
        unsigned int targetIdx = 0;
        for (mlir::Value res : terminator->getOperands()) {
            targets.set(targetIdx++, mlirValueMap[res]);
        }
    }

    void generateOp(mlir::Operation &op, ::Op::Builder &operation,
                    std::vector<mlir::Value> &functionValues,
                    llvm::DenseMap<mlir::Value, unsigned int> &mlirValueMap)
    {
        ::Op::Instruction::Builder instruction = operation.initInstruction();

        if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
            mlir::TypedAttr attribute = constOp.getValue();
            if (auto intAttr = dyn_cast<mlir::IntegerAttr>(attribute)) {
                IntOp::Builder intOp = instruction.initInt();
                intOp.setConst32(intAttr.getInt());
            }
            else if (auto floatAttr = dyn_cast<mlir::FloatAttr>(attribute)) {
                FloatOp::Builder floatOp = instruction.initFloat();
                floatOp.setConst64(floatAttr.getValueAsDouble());
            }
            else {
                llvm::outs() << "Unsupported constant type: " << attribute << ". Abort.\n";
                std::exit(-1);
            }

            defaultSignatureMapping(op, operation, functionValues, mlirValueMap);
        }
        else if (auto allocOp = dyn_cast<quantum::AllocOp>(op)) {
            QuregOp::Builder quregOp = instruction.initQureg();
            quregOp.setAlloc();

            defaultSignatureMapping(op, operation, functionValues, mlirValueMap);
        }
        else if (auto extractOp = dyn_cast<quantum::ExtractOp>(op)) {
            QuregOp::Builder quregOp = instruction.initQureg();
            quregOp.setExtractIndex();

            auto opInputs = operation.initInputs(2);
            opInputs.set(0, mlirValueMap[extractOp.getQreg()]);
            opInputs.set(1, mlirValueMap[extractOp.getIdx()]);

            // The 1st output doesn't exist in Catalyst (non-linear), we just generate a new QPR
            // value and map it to the existing register MLIR Value.
            auto opOutputs = operation.initOutputs(2);
            opOutputs.set(0, trackNewValue(extractOp.getQreg(), functionValues, mlirValueMap));
            opOutputs.set(1, trackNewValue(extractOp.getQubit(), functionValues, mlirValueMap));
        }
        else if (auto gateOp = dyn_cast<quantum::CustomOp>(op)) {
            QubitOp::Builder qubitOp = instruction.initQubit();
            QubitOp::Gate::Builder gate = qubitOp.initGate();

            moduleStrings.emplace_back(gateOp.getGateName());
            gate.setName(moduleStrings.size() - 1);
            gate.setNumQubits(gateOp.getInQubits().size() + gateOp.getInCtrlQubits().size());
            gate.setNumParams(gateOp.getParams().size());

            auto opInputs = operation.initInputs(gate.getNumQubits() + gate.getNumParams());
            unsigned int argIdx = 0;
            for (mlir::Value arg : gateOp.getInQubits()) {
                opInputs.set(argIdx++, mlirValueMap[arg]);
            }
            for (mlir::Value arg : gateOp.getInCtrlQubits()) { // relative location TBD
                opInputs.set(argIdx++, mlirValueMap[arg]);
            }
            for (mlir::Value arg : gateOp.getParams()) {
                opInputs.set(argIdx++, mlirValueMap[arg]);
            }

            auto opOutputs = operation.initOutputs(gate.getNumQubits());
            unsigned int resIdx = 0;
            for (mlir::Value res : gateOp.getOutQubits()) {
                opOutputs.set(resIdx++, trackNewValue(res, functionValues, mlirValueMap));
            }
            for (mlir::Value res : gateOp.getOutCtrlQubits()) { // relative location TBD
                opOutputs.set(resIdx++, trackNewValue(res, functionValues, mlirValueMap));
            }
        }
        else if (auto measureOp = dyn_cast<quantum::MeasureOp>(op)) {
            QubitOp::Builder qubitOp = instruction.initQubit();
            qubitOp.setMeasureNd();

            auto opInputs = operation.initInputs(1);
            opInputs.set(0, mlirValueMap[measureOp.getInQubit()]);

            auto opOutputs = operation.initOutputs(2);
            opOutputs.set(0, trackNewValue(measureOp.getOutQubit(), functionValues, mlirValueMap));
            opOutputs.set(1, trackNewValue(measureOp.getMres(), functionValues, mlirValueMap));
        }
        else if (auto insertOp = dyn_cast<quantum::InsertOp>(op)) {
            QuregOp::Builder quregOp = instruction.initQureg();
            quregOp.setInsertIndex();

            auto opInputs = operation.initInputs(3);
            opInputs.set(0, mlirValueMap[insertOp.getInQreg()]);
            opInputs.set(1, mlirValueMap[insertOp.getQubit()]);
            opInputs.set(2, mlirValueMap[insertOp.getIdx()]);

            auto opOutputs = operation.initOutputs(1);
            opOutputs.set(0, trackNewValue(insertOp.getOutQreg(), functionValues, mlirValueMap));
        }
        else if (auto deallocOp = dyn_cast<quantum::DeallocOp>(op)) {
            QuregOp::Builder quregOp = instruction.initQureg();
            quregOp.setFree();

            defaultSignatureMapping(op, operation, functionValues, mlirValueMap);
        }
        else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
            ScfOp::Builder scfOp = instruction.initScf();
            ::Region::Builder body = scfOp.initFor();

            auto opInputs = operation.initInputs(forOp.getNumRegionIterArgs() + 1);
            opInputs.set(0, mlirValueMap[forOp.getUpperBound()]); // TODO: need extra instructions
            unsigned int argIdx = 1;
            for (mlir::Value arg : forOp.getInitArgs()) {
                opInputs.set(argIdx++, mlirValueMap[arg]);
            }

            // QPR nested regions do not support accessing outside values, that is QPR regions
            // are IsolatedFromAbove, which MLIR SCF regions are not.
            generateRegion(forOp.getBodyRegion(), body, functionValues);

            auto opOutputs = operation.initOutputs(forOp.getNumRegionIterArgs());
            unsigned int resIdx = 0;
            for (mlir::Value res : forOp.getResults()) {
                opOutputs.set(resIdx++, trackNewValue(res, functionValues, mlirValueMap));
            }
        }
        else {
            llvm::outs() << "Unsupported operation: " << op.getName() << ". Abort.\n";
            std::exit(-1);
        }
    }

    void defaultSignatureMapping(mlir::Operation &op, ::Op::Builder &operation,
                                 std::vector<mlir::Value> &functionValues,
                                 llvm::DenseMap<mlir::Value, unsigned int> &mlirValueMap)
    {
        auto opInputs = operation.initInputs(op.getNumOperands());
        unsigned int argIdx = 0;
        for (mlir::Value arg : op.getOperands()) {
            opInputs.set(argIdx++, mlirValueMap[arg]);
        }

        auto opOutputs = operation.initOutputs(op.getNumResults());
        unsigned int resIdx = 0;
        for (mlir::Value res : op.getResults()) {
            opOutputs.set(resIdx++, trackNewValue(res, functionValues, mlirValueMap));
        }
    }

    unsigned int trackNewValue(mlir::Value val, std::vector<mlir::Value> &functionValues,
                               llvm::DenseMap<mlir::Value, unsigned int> &mlirValueMap)
    {
        unsigned int valIdx = functionValues.size();
        functionValues.push_back(val);
        mlirValueMap[val] = valIdx;
        return valIdx;
    }

    void generateType(mlir::Type val, ::Type::Builder &type)
    {
        if (isa<quantum::QubitType>(val)) {
            type.setQubit();
        }
        else if (isa<quantum::QuregType>(val)) {
            type.setQureg();
        }
        else if (auto idxVal = dyn_cast<mlir::IndexType>(val)) {
            type.setInt(64);
        }
        else if (auto intVal = dyn_cast<mlir::IntegerType>(val)) {
            type.setInt(intVal.getWidth());
        }
        else if (auto floatVal = dyn_cast<mlir::FloatType>(val)) {
            if (floatVal.isF64()) {
                type.setFloat(FloatPrecision::FLOAT64);
            }
            else if (floatVal.isF32()) {
                type.setFloat(FloatPrecision::FLOAT32);
            }
            else {
                llvm::outs() << "Unsupported float width: " << floatVal.getWidth() << ". Abort.\n";
                std::exit(-1);
            }
        }
        else {
            llvm::outs() << "Unsupported type: " << val << ". Abort.\n";
            std::exit(-1);
        }
    }

    void writeOut()
    {
        int fd = open("catalyst.qpr", O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            llvm::outs() << "Could not open file for writing. Abort.\n";
            std::exit(-1);
        }
        capnp::writeMessageToFd(fd, message);
        close(fd);
    }
};

struct QPRExportPass : public PassWrapper<QPRExportPass, OperationPass<ModuleOp>> {

    StringRef getArgument() const final { return "qpr-export"; }

    void runOnOperation() final
    {
        QPRSerializer serializer;

        ModuleOp module = getOperation();
        for (func::FuncOp fun : module.getOps<func::FuncOp>()) {
            serializer.addFunction(fun);
        }

        serializer.generateModule();
        serializer.writeOut();
    }
};

std::unique_ptr<Pass> createQPRExportPass() { return std::make_unique<QPRExportPass>(); }
