// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ION_PASSES
#define ION_PASSES

include "mlir/Pass/PassBase.td"

def GatesToPulsesPass : Pass<"gates-to-pulses"> {
    let summary = "Lower quantum gates to ionic pulses.";

    let options = [
    Option<"DeviceTomlLoc", "device-toml-loc",
           "std::string", /*default=*/"\"\"",
           "Toml file location for the ion hardware device parameters.">,
    Option<"QubitTomlLoc", "qubit-toml-loc",
           "std::string", /*default=*/"\"\"",
           "Toml file location for the ion hardware qubit parameters.">,
    Option<"Gate2PulseDecompTomlLoc", "gate-to-pulse-toml-loc",
           "std::string", /*default=*/"\"\"",
           "Toml file location for the ion hardware gate-to-pulse decomposition parameters.">,
    Option<"LoadIon", "load-ion",
           "bool", /*default=*/"true",
           "Whether to load the physical parameters for the ion (e.g. mass, charge, spin) into the IR.">,
    ];

    let dependentDialects = [
        "quantum::QuantumDialect",
        "ion::IonDialect",
        "scf::SCFDialect"
    ];

    let constructor = "catalyst::createGatesToPulsesPass()";
}

def IonConversionPass : Pass<"convert-ion-to-llvm"> {
    let summary = "Perform a dialect conversion from ion to LLVM";

    let dependentDialects = [
       "mlir::func::FuncDialect",
       "mlir::LLVM::LLVMDialect",
       "catalyst::quantum::QuantumDialect",
    ];

    let constructor = "catalyst::createIonConversionPass()";
}

#endif // ION_PASSES
