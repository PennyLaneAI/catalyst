// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <exception>
#include <iostream>
#include <string>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeRange.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h" // for automatic string conversion
#include "nanobind/stl/vector.h" // for automatic vector conversion

#include "Quantum/IR/QuantumInterfaces.h"

#include "PythonDriverUtils.hpp"

#define DEBUG_TYPE "[QPD] "

namespace nb = nanobind;

nb::object getPyvalFromTypeRange(mlir::TypeRange typerange)
{
    nb::list pyTypes;
    for (auto type : typerange) {
        std::string typestr;
        llvm::raw_string_ostream ss(typestr);
        type.print(ss);
        pyTypes.append(typestr);
    }
    return pyTypes;
}

nb::object getPyvalFromMlirAttribute(mlir::Attribute attr)
{
    return llvm::TypeSwitch<mlir::Attribute, nb::object>(attr)
        .Case<mlir::DictionaryAttr>([](auto dictAttr) {
            nb::dict outDict;
            for (auto namedAttr : dictAttr) {
                outDict[namedAttr.getName().str().c_str()] =
                    getPyvalFromMlirAttribute(namedAttr.getValue());
            }
            return outDict;
        })
        .Case<mlir::ArrayAttr>([](auto arrAttr) {
            nb::list outTuple;
            for (auto val : arrAttr) {
                outTuple.append(getPyvalFromMlirAttribute(val));
            }
            return outTuple;
        })
        .Case<mlir::StringAttr>([](auto strAttr) { return nb::cast(strAttr.getValue().str()); })
        .Default([](auto attr) { return nb::cast("placeholder"); });
}

std::string pythonRuleLowering(catalyst::quantum::DecomposableGate op)
{
    std::cout << "using new python lowering for op with the following data:\n";
    std::cout << "op name: " << op.getPlName() << "\n";
    std::cout << "op ID: " << op.getGraphOpId() << "\n";

    std::cout << "dynamic data shape:\n";
    for (auto type : op.getDynamicShape()) {
        type.dump();
    }

    std::cout << "wire lens:\n";
    for (auto len : op.getWireLens()) {
        std::cout << len << ", ";
    }
    std::cout << "\n";

    std::cout << "static data:\n";
    for (auto namedAttr : op.getStaticData()) {
        std::cout << namedAttr.getName().getValue().str();
        namedAttr.getValue().dump();
        std::cout << "\n";
    }

    std::cout << "trying nanobind now\n";

    QuantumPythonDecompositions::PyInterpreterGuard guard;
    std::string mlirText = guard.withGil([&] -> std::string {
        const char *moduleName = "catalyst.device.python_decompositions";
        const char *functionName = "python_decomposition_wrapper";

        try {
            nb::module_ wrapperModule = nb::module_::import_(moduleName);
            nb::object wrapperFunction = wrapperModule.attr(functionName);

            nb::object pythonResult = wrapperFunction(
                op.getPlName(), op.getGraphOpId(), getPyvalFromTypeRange(op.getDynamicShape()),
                op.getWireLens(), getPyvalFromMlirAttribute(op.getStaticData()));

            return nb::borrow<nb::str>(pythonResult).c_str();
        }
        catch (const nb::python_error &error) {
            throw QuantumPythonDecompositions::TracingError(moduleName, functionName,
                                                            op.getGraphOpId(), error.what());
        }
        catch (const std::exception &error) {
            throw;
        }
    });

    return mlirText;
}

extern "C" __attribute__((visibility("default"))) void *getPythonRuleLoweringFunction()
{
    return reinterpret_cast<void *>(pythonRuleLowering);
}
