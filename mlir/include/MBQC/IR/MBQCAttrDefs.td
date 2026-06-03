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

#ifndef MBQC_ATTR_DEFS
#define MBQC_ATTR_DEFS

include "mlir/IR/EnumAttr.td"

include "MBQC/IR/MBQCDialect.td"

//===----------------------------------------------------------------------===//
// MBQC dialect enums and attributes.
//===----------------------------------------------------------------------===//

def MeasurementPlane : I32EnumAttr<"MeasurementPlane",
    "Planes in the Bloch sphere representation with support for arbitrary-basis measurements",
    [
        I32EnumAttrCase<"XY", 0>,
        I32EnumAttrCase<"YZ", 1>,
        I32EnumAttrCase<"ZX", 2>,
    ]> {
    let cppNamespace = "catalyst::mbqc";
    let genSpecializedAttr = 0;
}

def MeasurementPlaneAttr : EnumAttr<MBQC_Dialect, MeasurementPlane, "measurement_plane">;

#endif // MBQC_ATTR_DEFS
