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

#ifndef RTIO_DIALECT
#define RTIO_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// RTIO Dialect Definition
//===----------------------------------------------------------------------===//

def RTIO_Dialect : Dialect {
    let summary = "Real-Time I/O dialect for FPGA quantum control";
    let description = [{
        The RTIO dialect provides operations for precise timing control
        and hardware signal generation on FPGAs for quantum computing.

        This dialect supports two levels of abstraction:
        1. Event-Based IR (high-level): Declarative operations with explicit event dependencies

        // TODO: Do we need the separate Timeline IR for artiq family?
        2. Timeline IR (low-level): Stateful operations with implicit time cursor
    }];

    let name = "rtio";
    let cppNamespace = "::catalyst::rtio";
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 0;

    let extraClassDeclaration = [{
        mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser,
                                       mlir::Type type) const override;

        void printAttribute(mlir::Attribute attr,
                            mlir::DialectAsmPrinter &printer) const override;
    }];
}

//===----------------------------------------------------------------------===//
// RTIO dialect types.
//===----------------------------------------------------------------------===//

class RTIO_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<RTIO_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

// Channel type with variadic parameters
def RTIOChannelType : RTIO_Type<"Channel", "channel"> {
    let summary = "A hardware I/O channel with logical and physical identification";
    let description = [{
        Represents a virtual hardware channel for RTIO operations

        Syntax:
        ```
        !rtio.channel<KIND, ?>                 // Dynamic: kind only
        !rtio.channel<KIND, N>                 // Static: kind + channel ID
        !rtio.channel<KIND, [QUALIFIERS], ?>   // Dynamic: with qualifiers
        !rtio.channel<KIND, [QUALIFIERS], N>   // Static: with qualifiers + channel ID
        ```

        Examples:
        ```mlir

        // Simple DDS, channel ID TBD (to be resolved during channel resolution)
        !rtio.channel<"dds", ?>

        // DDS channel with qualifiers:
        // Qualifiers distinguish different logical channels of the same kind.
        // Example: ion 0, transition 0 -> will be distinguished from other transitions of
        //          the same ion. The mapping of the logical channel to the hardware channel will
        //          be resolved during the lowering pass from given dialect to RTIO dialect.
        //          User need to specifiy the mapping logic.
        !rtio.channel<"dds", [0, "transition_0"], ?>

        // DDS on hardware channel 0
        !rtio.channel<"dds", 0>

        // === Channel Resolution During Compilation ===

        // Before channel resolution: Dynamic channels
        !rtio.channel<"dds", [0], ?>

        // After channel resolution: Resolved to hardware channel 1
        !rtio.channel<"dds", [0], 1>

        // Note: The qualifiers are used to provide additional information to the channel type.
        // And they will be used to distinguish different channels with the same kind.
        ```
    }];

    let parameters = (ins
        StringRefParameter<"channel kind">:$kind,
        OptionalParameter<"mlir::ArrayAttr">:$qualifiers,
        OptionalParameter<"mlir::IntegerAttr">:$channelId
    );

    let assemblyFormat = [{
        `<` custom<ChannelTypeBody>($kind, $qualifiers, $channelId) `>`
    }];

    let extraClassDeclaration = [{
        bool hasQualifiers() const {
            return getQualifiers() && !getQualifiers().empty();
        }

        size_t getNumQualifiers() const {
            return getQualifiers() ? getQualifiers().size() : 0;
        }

        mlir::Attribute getQualifier(size_t index) const {
            if (!getQualifiers() || index >= getQualifiers().size())
                return nullptr;
            return getQualifiers()[index];
        }

        bool isStatic() const {
            auto channelId = getChannelId();
            if (!channelId) {
                return false;
            }
            return channelId.getInt() >= 0;
        }

        bool isDynamic() const {
            return !isStatic();
        }
    }];
}

// Event handle type
def RTIOEventType : RTIO_Type<"Event", "event"> {
    let summary = "A handle to a pending RTIO event";
    let description = [{
        Represents a handle to a pending RTIO event (e.g., pulse, sync).

        Example:
        ```mlir
        %event0 = rtio.pulse %ch0 duration(%dur) frequency(%freq) phase(%phase) : !rtio.event
        %event1 = rtio.pulse %ch1 duration(%dur) frequency(%freq) phase(%phase) : !rtio.event
        %sync = rtio.sync %event0, %event1 : !rtio.event
        ```
    }];
}

//===----------------------------------------------------------------------===//
// RTIO Operation Base
//===----------------------------------------------------------------------===//

class RTIO_Op<string mnemonic, list<Trait> traits = []> :
    Op<RTIO_Dialect, mnemonic, traits>;

#endif // RTIO_DIALECT

