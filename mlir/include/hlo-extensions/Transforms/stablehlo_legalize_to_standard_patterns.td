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


// This file is taken from the
//   tensorflow/mlir-hlo
// repository, under the Apache 2.0 License, at
//   https://github.com/tensorflow/mlir-hlo/blob/a5529d99fc4d1132b0c282a053d26c11e6636b3a/mhlo/transforms/legalize_to_standard/legalize_to_standard_patterns.td
// with the following copyright notice:

    /* Copyright 2019 The OpenXLA Authors.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    ==============================================================================*/



// This is the legalization pattern definition file for Stablehlo to StandardOps.

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Math/IR/MathOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "stablehlo/dialect/StablehloOps.td"

//===----------------------------------------------------------------------===//
// Nullary op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(StableHLO_ConstantOp ElementsAttr:$value),
          (Arith_ConstantOp $value)>;

//===----------------------------------------------------------------------===//
// Binary op patterns.
//===----------------------------------------------------------------------===//

def IsSameSizePred : CPred<
    "cast<ShapedType>($0.getType()).getShape() "
    "== cast<ShapedType>($1.getType()).getShape()">;
def IsSameSizeConstraint : Constraint<IsSameSizePred, "inputs are same size">;
def createFastMathNone : NativeCodeCall<
    "::mlir::arith::FastMathFlagsAttr::get("
      "$_builder.getContext(), ::mlir::arith::FastMathFlags::none"
    ")">;
def createOverflowNone : NativeCodeCall<
    "::mlir::arith::IntegerOverflowFlagsAttr::get("
      "$_builder.getContext(), ::mlir::arith::IntegerOverflowFlags::none"
    ")">;
def createDenormalIEEE : NativeCodeCall<
    "::mlir::arith::DenormalModeAttr::get("
      "$_builder.getContext(), ::mlir::arith::DenormalMode::ieee"
    ")">;
def createExactNone : NativeCodeCall<"::mlir::UnitAttr()">;


// Unary Lowering Patterns.
def : Pat<(StableHLO_CeilOp HLO_FpTensor:$i), (Math_CeilOp $i, (createFastMathNone ))>;

// Binary Lowering Patterns.
def : Pat<(StableHLO_AndOp HLO_IntTensor:$l, HLO_IntTensor:$r),
          (Arith_AndIOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(StableHLO_OrOp HLO_IntTensor:$l, HLO_IntTensor:$r),
          (Arith_OrIOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(StableHLO_AddOp HLO_FpTensor:$l, HLO_FpTensor:$r),
          (Arith_AddFOp $l, $r, (createFastMathNone )),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(StableHLO_SubtractOp HLO_FpTensor:$l, HLO_FpTensor:$r),
          (Arith_SubFOp $l, $r, (createFastMathNone )),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(StableHLO_MulOp HLO_FpTensor:$l, HLO_FpTensor:$r),
          (Arith_MulFOp $l, $r, (createFastMathNone )),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(StableHLO_DivOp HLO_FpTensor:$l, HLO_FpTensor:$r),
          (Arith_DivFOp $l, $r, (createFastMathNone )),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(StableHLO_RemOp HLO_FpTensor:$l, HLO_FpTensor:$r),
          (Arith_RemFOp $l, $r, (createFastMathNone )),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(StableHLO_AddOp HLO_IntTensor:$l, HLO_IntTensor:$r),
          (Arith_AddIOp $l, $r, (createOverflowNone )),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(StableHLO_SubtractOp HLO_IntTensor:$l, HLO_IntTensor:$r),
          (Arith_SubIOp $l, $r, (createOverflowNone )),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(StableHLO_MulOp HLO_IntTensor:$l, HLO_IntTensor:$r),
          (Arith_MulIOp $l, $r, (createOverflowNone )),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(StableHLO_DivOp HLO_IntTensor:$l, HLO_IntTensor:$r),
          (Arith_DivSIOp $l, $r, (createExactNone )),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(StableHLO_RemOp HLO_IntTensor:$l, HLO_IntTensor:$r),
          (Arith_RemSIOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;

def : Pat<(StableHLO_SelectOp $pred, $tv, $fv),
          (SelectOp $pred, $tv, $fv),
          [(IsSameSizeConstraint $pred, $tv), (IsSameSizeConstraint $tv, $fv)]>;
