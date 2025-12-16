// Copyright 2025 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef RTIO_PASSES
#define RTIO_PASSES

include "mlir/Pass/PassBase.td"

def RTIOEventToARTIQPass : Pass<"convert-rtio-event-to-artiq", "mlir::ModuleOp"> {
    let summary = "Convert RTIO event-based operations directly to ARTIQ-compatible LLVM IR";
    let description = [{
        Mapping:
        - rtio.pulse -> at_mu(operand) DDS config + TTL on + delay + TTL off + now_mu()
        - rtio.sync -> maximum the timestamp of the input events + at_mu() + now_mu()
        - rtio.channel -> channel ID
        - rtio.empty -> just return now_mu()

        And this pass generates LLVM IR that directly calls ARTIQ runtime functions
    }];

    let dependentDialects = [
        "rtio::RTIODialect",
        "mlir::LLVM::LLVMDialect",
        "mlir::arith::ArithDialect",
        "mlir::math::MathDialect",
        "mlir::scf::SCFDialect",
        "mlir::func::FuncDialect"
    ];
}

#endif // RTIO_PASSES


