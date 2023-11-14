include "mlir/IR/DialectBase.td"

def OpenQASMDialect : Dialect {
    let summary = "An OpenQASM 3 dialect in MLIR.";
    let description = [{
        ...
    }];

    /// This is the namespace of the dialect in MLIR, which is used as a prefix for types and ops.
    let name = "oq";

    /// This is the C++ namespace that the dialect, and all sub-components, get placed in.
    let cppNamespace = "::catalyst::openqasm";

    /// Use the default type printing/parsing hooks, otherwise we would explicitly define them.
    let useDefaultTypePrinterParser = 1;
}

include "mlir/IR/AttrTypeBase.td"

class OpenQASM_Type<string name, string nameInIR> : TypeDef<OpenQASMDialect, name, []> {
    let mnemonic = nameInIR;
}

def QubitType : OpenQASM_Type<"Qubit", "qubit"> {
    let summary = "A single quantum bit reference.";
}

include "mlir/IR/OpBase.td"

class OpenQASM_Op<string nameInIR> : Op<OpenQASMDialect, nameInIR, []>;

def RZGate : OpenQASM_Op<"RZ"> {
    let summary = "A single-qubit rotation around the Z-axis by an angle θ.";

    let arguments = (ins
        F64:$theta,
        QubitType:$qubit
    );

    let results = (outs
    );

    let assemblyFormat = [{
        `(` $theta `)` $qubit attr-dict `:` type($qubit)
    }];
}
