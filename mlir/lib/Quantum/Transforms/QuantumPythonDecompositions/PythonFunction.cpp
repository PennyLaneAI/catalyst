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
#include <exception>
#include <iostream>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/DebugStringHelper.h"

#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"

#include "PythonDriverUtils.hpp"

#define DEBUG_TYPE "[QPD] "

namespace nb = nanobind;

namespace {

static nb::dict
getPyvalFromDynamicShape(const llvm::StringMap<llvm::SmallVector<mlir::Type>> &map) {
    nb::dict name_to_shape;
    for (const auto &entry : map) {
        nb::list types;
        for (auto type : entry.getValue()) {
            if (!type) {
                continue;
            }
            std::string typestr;
            llvm::raw_string_ostream ss(typestr);
            type.print(ss);
            types.append(typestr);
        }
        name_to_shape[nb::str(entry.getKey().str().c_str())] = types;
    }
    return name_to_shape;
}

static nb::dict getPyvalFromWireLens(const llvm::StringMap<size_t> map) {
    nb::dict dict;
    for (const auto &entry : map) {
        dict[nb::str(entry.getKey().str().c_str())] = entry.getValue();
    }
    return dict;
}

static nb::object getPyvalFromTypeRange(mlir::TypeRange typerange) {
    nb::list pyTypes;
    for (auto type : typerange) {
        std::string typestr;
        llvm::raw_string_ostream ss(typestr);
        type.print(ss);
        pyTypes.append(typestr);
    }
    return pyTypes;
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

} // namespace

std::string pythonRuleLowering(catalyst::quantum::DecomposableGate op) {
    QuantumPythonDecompositions::PyInterpreterGuard guard;
    std::string mlirText = guard.withGil([&] -> std::string {
        const char *moduleName = "catalyst.decomposition.decomposition_rules";
        const char *functionName = "compile_reachable_decomposition_rules_wrapper";

        try {
            auto tmp = op.getDynamicShape();
            nb::module_ wrapperModule = nb::module_::import_(moduleName);
            nb::object wrapperFunction = wrapperModule.attr(functionName);

            nb::object pythonResult = wrapperFunction(
                op.getOperatorName(), op.getGraphOpId(),
                getPyvalFromDynamicShape(op.getDynamicShape()),
                getPyvalFromWireLens(op.getWireLens()),
                getPyvalFromMlirAttribute(op.getStaticData()), nb::arg("extra_data") = nb::none(),
                nb::arg("is_custom_op") = isa<catalyst::quantum::CustomOp>(op));
            return nb::borrow<nb::str>(pythonResult).c_str();
        } catch (const nb::python_error &error) {
            throw QuantumPythonDecompositions::TracingError(moduleName, functionName,
                                                            op.getGraphOpId(), error.what());
        } catch (const std::exception &error) {
            throw;
        }
    });

    return mlirText;
}

extern "C" __attribute__((visibility("default"))) void *getPythonRuleLoweringFunction() {
    return reinterpret_cast<void *>(pythonRuleLowering);
}
