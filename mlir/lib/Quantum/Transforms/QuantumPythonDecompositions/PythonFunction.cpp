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

#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h" // for type conversion

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringMapEntry.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/DebugStringHelper.h"

#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/QPDLoader.h"

#include "PythonDriverUtils.hpp"
#include "StringConstants.h"
#include "longobject.h"
#include "pyerrors.h"
#include "pytypedefs.h"

#define DEBUG_TYPE "[QPD] "

namespace nb = nanobind;

namespace {

using catalyst::quantum::DecomposableOp;
using BaseOperations = llvm::StringMap<DecomposableOp>;
// The dicts below are built in operand order (Python dicts keep insertion order), which the
// frontend uses as the operand layout of the rules it compiles. Do not reorder them.
static nb::dict getPyvalFromDynamicShape(const catalyst::quantum::DynamicShape &dynamicShape) {
    nb::dict name_to_shape;
    for (const auto &[name, types] : dynamicShape) {
        nb::list pyTypes;
        for (auto type : types) {
            if (!type) {
                continue;
            }
            std::string typestr;
            llvm::raw_string_ostream ss(typestr);
            type.print(ss);
            pyTypes.append(typestr);
        }
        name_to_shape[nb::str(name.c_str())] = pyTypes;
    }
    return name_to_shape;
}

static nb::dict getPyvalFromWireLens(const catalyst::quantum::WireLens &wireLens) {
    nb::dict dict;
    for (const auto &[name, len] : wireLens) {
        dict[nb::str(name.c_str())] = len;
    }
    return dict;
}

// Read an integer the way MLIR's own printer does, so a value that crosses into the frontend and is
// printed back into a graphOpId keeps the spelling it arrived with: unsigned types zero-extend,
// while signed and signless ones sign-extend (a signless negative prints as `-5:i64`).
static nb::object getPyvalFromAPInt(const llvm::APInt &value, bool isUnsigned) {
    // APInt only surrenders its value 64 bits at a time, so a wider one goes through its decimal
    // spelling. Python integers have no width of their own to overflow.
    if (isUnsigned ? value.getActiveBits() > 64 : value.getSignificantBits() > 64) {
        llvm::SmallString<40> digits;
        value.toString(digits, /*Radix=*/10, /*Signed=*/!isUnsigned);
        PyObject *pyInt = PyLong_FromString(digits.c_str(), /*pend=*/nullptr, /*base=*/10);
        if (!pyInt) {
            throw nb::python_error();
        }
        return nb::steal(pyInt);
    }

    if (isUnsigned) {
        return nb::cast(value.getZExtValue());
    }
    return nb::cast(value.getSExtValue());
}

static nb::object getPyvalFromIntegerAttribute(mlir::IntegerAttr intAttr) {
    // An `index` attribute is an IntegerAttr with no signedness of its own; reading it as signed is
    // what IntegerAttr::getInt does.
    return getPyvalFromAPInt(intAttr.getValue(), intAttr.getType().isUnsignedInteger());
}

// Name the NumPy dtype an element type came from, mirroring the dtypes the frontend lowers to dense
// attributes in `_dense_attribute_from_array`.
static std::string getNumpyDtypeName(mlir::Type elementType) {
    unsigned width = 0;
    llvm::StringRef kind;
    if (auto intType = llvm::dyn_cast<mlir::IntegerType>(elementType)) {
        // NumPy has no 1-bit integer; an `i1` array is how a boolean array lowers.
        if (intType.getWidth() == 1) {
            return "bool";
        }
        width = intType.getWidth();
        kind = intType.isUnsigned() ? "uint" : "int";
    } else if (auto floatType = llvm::dyn_cast<mlir::FloatType>(elementType)) {
        width = floatType.getWidth();
        kind = "float";
    }

    if (width != 8 && width != 16 && width != 32 && width != 64) {
        throw QuantumPythonDecompositions::QPDError(
            "Cannot convert an array of " + mlir::debugString(elementType) +
            " to a Python value for graph decomposition, no matching NumPy dtype.");
    }
    return (kind + llvm::Twine(width)).str();
}

// Convert a dense attribute into an equivalent NumPy array. Elements arrive in row-major order,
// which is the order NumPy fills the requested shape in, and a splat attribute expands back into
// its full set of elements.
static nb::object getPyvalFromDenseAttribute(mlir::DenseIntOrFPElementsAttr denseAttr) {
    mlir::ShapedType shapedType = denseAttr.getType();
    mlir::Type elementType = shapedType.getElementType();
    std::string dtypeName = getNumpyDtypeName(elementType);

    nb::list elements;
    if (llvm::isa<mlir::IntegerType>(elementType)) {
        bool isUnsigned = elementType.isUnsignedInteger();
        for (const llvm::APInt &element : denseAttr.getValues<llvm::APInt>()) {
            elements.append(getPyvalFromAPInt(element, isUnsigned));
        }
    } else {
        for (llvm::APFloat element : denseAttr.getValues<llvm::APFloat>()) {
            // Widen to double regardless of the element type; NumPy narrows it back when it builds
            // the array with the dtype the elements came from.
            bool losesInfo = false;
            element.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven,
                            &losesInfo);
            elements.append(element.convertToDouble());
        }
    }

    nb::list shape;
    for (int64_t dim : shapedType.getShape()) {
        shape.append(dim);
    }

    nb::module_ numpy = nb::module_::import_("numpy");
    nb::object array = numpy.attr("array")(elements, nb::arg("dtype") = dtypeName.c_str());
    return array.attr("reshape")(shape);
}

// Convert an MLIR attribute into an equivalent Python value. Generally should represent the
// inverse of `get_mlir_attribute_from_pyval` from the frontend direction.
static nb::object getPyvalFromMlirAttribute(mlir::Attribute attr) {
    if (auto typeAttr = llvm::dyn_cast<mlir::TypeAttr>(attr);
        typeAttr && llvm::isa<mlir::NoneType>(typeAttr.getValue())) {
        return nb::none();
    }
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
            nb::list outList;
            for (auto val : arrAttr) {
                outList.append(getPyvalFromMlirAttribute(val));
            }
            return outList;
        })
        .Case<mlir::StringAttr>([](auto strAttr) { return nb::cast(strAttr.getValue().str()); })
        // Needs to be before IntegerAttr, since bools are also integers (i1).
        .Case<mlir::BoolAttr>([](auto boolAttr) { return nb::cast(boolAttr.getValue()); })
        .Case<mlir::IntegerAttr>([](auto intAttr) { return getPyvalFromIntegerAttribute(intAttr); })
        .Case<mlir::FloatAttr>(
            [](auto floatAttr) { return nb::cast(floatAttr.getValueAsDouble()); })
        .Case<mlir::DenseIntOrFPElementsAttr>(
            [](auto denseAttr) { return getPyvalFromDenseAttribute(denseAttr); })
        .Default([](mlir::Attribute attr) -> nb::object {
            throw QuantumPythonDecompositions::QPDError(
                "Cannot convert the MLIR attribute " + mlir::debugString(attr) +
                " to a Python value for graph decomposition, unknown attribute type.");
        });
}

// Resource IDs can include modifiers inherited from regions and subroutine callers.
static const llvm::StringMapEntry<DecomposableOp> &
findBaseOperation(const std::string &opId, const BaseOperations &baseOps,
                  const llvm::StringSet<> &opNames, const nb::object &parseId) {
    for (const auto &name : opNames) {
        try {
            nb::tuple parsed(parseId(name.getKey().str(), opId));
            auto match = baseOps.find(nb::cast<std::string>(parsed[0]));
            if (match != baseOps.end()) {
                return *match;
            }
        } catch (const nb::python_error &error) {
            // A modified ID for another base name cannot be unwrapped.
            if (!error.matches(PyExc_ValueError)) {
                throw;
            }
        }
    }
    throw QuantumPythonDecompositions::QPDError("No operation matches resource graph ID " + opId);
}

static nb::dict getBaseMetadata(DecomposableOp op) {
    auto dynamicShape = op.getDynamicShape();
    // Control lowering can promote custom gates to OperatorOp, retaining numeric parameter names.
    bool positionalParameters =
        !dynamicShape.empty() && llvm::all_of(dynamicShape, [](const auto &parameter) {
            return !parameter.first.empty() && llvm::all_of(parameter.first, llvm::isDigit);
        });

    // Cache only base arguments, independently of which modifier variant came first.
    auto wireLens = op.getWireLens();
    llvm::erase_if(wireLens,
                   [](const auto &entry) { return entry.first == catalyst::CTRL_WORK_WIRES; });
    mlir::NamedAttrList staticData(op.getStaticData());
    staticData.erase(catalyst::CTRL_WORK_WIRE_TYPE);

    nb::dict data;
    data["op_name"] = nb::cast(op.getOperatorName());
    data["dynamic_shape"] = getPyvalFromDynamicShape(dynamicShape);
    data["wire_lens"] = getPyvalFromWireLens(wireLens);
    data["static_data"] = getPyvalFromMlirAttribute(staticData.getDictionary(op->getContext()));
    data["is_custom_op"] = nb::bool_(isa<catalyst::quantum::CustomOp>(op) || positionalParameters);
    return data;
}

static nb::list buildRuleRequests(llvm::ArrayRef<DecomposableOp> ops,
                                  llvm::ArrayRef<std::string> requestedIds,
                                  const nb::object &parseId, std::string &tracingContext) {
    // Index base identities once. The original operations supply operand order, which cannot
    // be recovered from their sorted graph IDs.
    BaseOperations baseOps;
    llvm::StringSet<> opNames;
    for (auto op : ops) {
        baseOps.try_emplace(op.getGraphOpIdExplicitModifiers(false, 0, 0, false), op);
        opNames.insert(op.getOperatorName());
    }

    llvm::StringMap<nb::dict> metadata;
    nb::list requests;
    for (const auto &opId : requestedIds) {
        tracingContext = opId;
        const auto &baseOp = findBaseOperation(opId, baseOps, opNames, parseId);
        auto [entry, inserted] = metadata.try_emplace(baseOp.getKey());
        if (inserted) {
            entry->second = getBaseMetadata(baseOp.getValue());
        }
        // Modifier variants share cached metadata, but each request retains its full ID.
        nb::dict request = nb::borrow<nb::dict>(entry->second.attr("copy")());
        request["op_id"] = nb::cast(opId);
        requests.append(request);
    }
    return requests;
}

} // namespace

bool pythonRuleLowering(llvm::ArrayRef<catalyst::quantum::DecomposableOp> ops,
                        llvm::ArrayRef<std::string> requestedIds,
                        llvm::ArrayRef<mlir::DictionaryAttr> availableRules, std::string &mlirText,
                        std::string &errorMessage) {
    try {
        QuantumPythonDecompositions::PyInterpreterGuard guard;
        mlirText = guard.withGil([&] -> std::string {
            const char *moduleName = "catalyst.decomposition.on_demand_rules";
            const char *functionName = "compile_on_demand_decomposition_rules";
            std::string tracingContext;

            try {
                nb::module_ wrapperModule = nb::module_::import_(moduleName);
                nb::object wrapperFunction = wrapperModule.attr(functionName);

                nb::module_ modifierModule =
                    nb::module_::import_("catalyst.decomposition.modifier_state");
                nb::object parseId = modifierModule.attr("ModifierState").attr("from_graph_op_id");
                nb::list requests = buildRuleRequests(ops, requestedIds, parseId, tracingContext);

                tracingContext = llvm::join(requestedIds, ", ");
                nb::list availableEntries;
                for (auto entry : availableRules) {
                    availableEntries.append(getPyvalFromMlirAttribute(entry));
                }
                nb::object pythonResult = wrapperFunction(requests, availableEntries);
                return nb::borrow<nb::str>(pythonResult).c_str();
            } catch (const std::exception &error) {
                throw QuantumPythonDecompositions::TracingError(moduleName, functionName,
                                                                tracingContext, error.what());
            }
        });
        return true;
    } catch (const std::exception &error) {
        errorMessage = error.what();
        return false;
    }
}

extern "C" __attribute__((visibility("default"))) void *getPythonRuleLoweringFunction() {
    return reinterpret_cast<void *>(pythonRuleLowering);
}
