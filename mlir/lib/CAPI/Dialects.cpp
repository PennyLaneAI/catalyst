// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Quantum-c/Dialects.h"

#include "Gradient/IR/GradientDialect.h"
#include "Quantum/IR/QuantumDialect.h"
#include "mlir/CAPI/Registration.h"

#include "Quantum/IR/QuantumEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "Quantum/IR/QuantumAttributes.h.inc"

// #include "Quantum/IR/QuantumEnums.cpp.inc"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Quantum, quantum, catalyst::quantum::QuantumDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Gradient, gradient, catalyst::gradient::GradientDialect)

using namespace mlir;

bool mlirTypeIsAQuregType(MlirType type) { return isa<catalyst::quantum::QuregType>(unwrap(type)); }

MlirType mlirQuantumQuregTypeGet(MlirContext ctx)
{
    return wrap(catalyst::quantum::QuregType::get(unwrap(ctx)));
}

bool mlirTypeIsAQubitType(MlirType type)
{
    return mlir::isa<catalyst::quantum::QubitType>(unwrap(type));
}

MlirType mlirQuantumQubitTypeGet(MlirContext ctx)
{
    return wrap(catalyst::quantum::QubitType::get(unwrap(ctx)));
}

bool mlirTypeIsAObservableType(MlirType type)
{
    return mlir::isa<catalyst::quantum::ObservableType>(unwrap(type));
}

MlirType mlirQuantumObservableTypeGet(MlirContext ctx)
{
    return wrap(catalyst::quantum::ObservableType::get(unwrap(ctx)));
}

bool mlirAttributeIsANamedObservableAttr(MlirAttribute attr)
{
    return isa<catalyst::quantum::NamedObservableAttr>(unwrap(attr));
}

MlirAttribute mlirNamedObservableAttrGet(MlirContext ctx, const char *observableType)
{
    auto observable = catalyst::quantum::symbolizeNamedObservable(observableType);
    assert(observable.has_value() && "Received invalid named observable");
    return wrap(catalyst::quantum::NamedObservableAttr::get(unwrap(ctx), observable.value()));
}
