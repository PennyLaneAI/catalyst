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

#ifndef QREF_PASSES
#define QREF_PASSES

include "mlir/Pass/PassBase.td"

def ValueSemanticsConversionPass : Pass<"convert-to-value-semantics"> {
    let summary = "Convert to value semantics quantum dialect";

    let dependentDialects = [
       "catalyst::quantum::QuantumDialect",
       "catalyst::CatalystDialect",
       "math::MathDialect"
    ];
}

def VerifyNoQuantumUseAfterFreePass : Pass<"verify-no-quantum-use-after-free"> {
    let summary = "Verify that the quantum program has no uses of qubits or registers after they have been deallocated";
}

def DecomposeLoweringPass : Pass<"decompose-lowering"> {
    let summary = "Replace quantum operations with compiled decomposition rules.";

    let options = [
        ListOption<
            /*C++ name*/"targetRulesOption",
            /*CLI name*/"target-rules",
            /*Type*/"std::string",
            /*Description*/"The set of decomposition rules to apply. If empty, applies all rules.">,
            Option<
            /*C++ name*/"inlineRuleBody",
            /*CLI name*/"inline-rule-body",
            /*Type*/"bool",
            /*Default*/[{ false }],
            /*Description*/"Whether to inline the rule body. Defaults to false.">
    ];

    let dependentDialects = [
       "catalyst::quantum::QuantumDialect",
       "catalyst::CatalystDialect"
    ];
}

def AdjointLoweringPass : Pass<"adjoint-lowering"> {
    let summary = "Lower qref.adjoint regions using reference semantics.";

    let dependentDialects = [
        "catalyst::qref::QRefDialect",
        "catalyst::CatalystDialect",
        "arith::ArithDialect",
        "index::IndexDialect",
        "memref::MemRefDialect",
        "tensor::TensorDialect",
        "complex::ComplexDialect",
        "scf::SCFDialect"
    ];
}

def CtrlLoweringPass : Pass<"ctrl-lowering"> {
    let summary = "Distribute qref.ctrl controls over the enclosed reference-semantics gates.";

    let dependentDialects = [
        "scf::SCFDialect",
        "catalyst::qref::QRefDialect"
    ];
}

def ModifiersLoweringPass : Pass<"modifiers-lowering"> {
    let summary = "Lower quantum.ctrl and quantum.adjoint regions to op-level modifiers.";

    let description = [{
        Converts value semantics to reference semantics on entry and back to value semantics on
        exit. The standalone ctrl-lowering and adjoint-lowering passes use reference semantics
        directly and do not perform semantics conversion.

        Runs the ctrl-lowering and adjoint-lowering rewrite patterns together under a single greedy
        driver until a fixed point. This allows arbitrarily nested control and adjoint regions to be
        resolved. Each pattern defers (returns a match failure) while its region still contains the
        other modifier, so the greedy worklist naturally interleaves them: an inner adjoint region is
        reduced, then the enclosing control lowers on a re-try, and vice versa. This reaches the same
        fixed point as manually alternating the standalone ctrl-lowering and adjoint-lowering passes,
        in a single pass and regardless of nesting depth or order.
    }];

    let dependentDialects = [
        "catalyst::qref::QRefDialect",
        "catalyst::quantum::QuantumDialect",
        "catalyst::CatalystDialect",
        "arith::ArithDialect",
        "index::IndexDialect",
        "memref::MemRefDialect",
        "tensor::TensorDialect",
        "complex::ComplexDialect",
        "scf::SCFDialect"
    ];
}

#endif // QREF_PASSES
