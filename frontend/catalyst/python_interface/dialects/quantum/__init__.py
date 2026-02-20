# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The xDSL Quantum dialect, which mirrors Catalyst's MLIR Quantum dialect."""

from xdsl.ir import Dialect

from .attributes import (
    NamedObservable,
    NamedObservableAttr,
    ObservableSSAValue,
    ObservableType,
    PauliWord,
    QubitLevel,
    QubitRole,
    QubitSSAValue,
    QubitType,
    QubitTypeConstraint,
    QuregSSAValue,
    QuregType,
    QuregTypeConstraint,
    ResultType,
)
from .operations import (
    AdjointOp,
    AllocOp,
    AllocQubitOp,
    ComputationalBasisOp,
    CountsOp,
    CustomOp,
    DeallocOp,
    DeallocQubitOp,
    DeviceInitOp,
    DeviceReleaseOp,
    ExpvalOp,
    ExtractOp,
    FinalizeOp,
    GateOp,
    GlobalPhaseOp,
    HamiltonianOp,
    HermitianOp,
    InitializeOp,
    InsertOp,
    MeasureOp,
    MultiRZOp,
    NamedObsOp,
    NumQubitsOp,
    ObservableOp,
    PauliRotOp,
    PCPhaseOp,
    ProbsOp,
    QubitUnitaryOp,
    SampleOp,
    SetBasisStateOp,
    SetStateOp,
    StateOp,
    TensorOp,
    TerminalMeasurementOp,
    VarianceOp,
    YieldOp,
)

Quantum = Dialect(
    "quantum",
    [
        AdjointOp,
        AllocOp,
        AllocQubitOp,
        ComputationalBasisOp,
        CountsOp,
        CustomOp,
        DeallocOp,
        DeallocQubitOp,
        DeviceInitOp,
        DeviceReleaseOp,
        ExpvalOp,
        ExtractOp,
        FinalizeOp,
        GlobalPhaseOp,
        HamiltonianOp,
        HermitianOp,
        InitializeOp,
        InsertOp,
        MeasureOp,
        MultiRZOp,
        NamedObsOp,
        NumQubitsOp,
        PauliRotOp,
        PCPhaseOp,
        ProbsOp,
        QubitUnitaryOp,
        SampleOp,
        SetBasisStateOp,
        SetStateOp,
        StateOp,
        TensorOp,
        VarianceOp,
        YieldOp,
    ],
    [ObservableType, QubitType, QuregType, ResultType, NamedObservableAttr],
)

__all__ = [
    # Main dialect
    "Quantum",
    # Attributes
    "NamedObservable",
    "NamedObservableAttr",
    "ObservableType",
    "QubitType",
    "QuregType",
    "ResultType",
    # Type aliases
    "ObservableSSAValue",
    "PauliWord",
    "QubitSSAValue",
    "QuregSSAValue",
    # Operation bases
    "GateOp",
    "ObservableOp",
    "TerminalMeasurementOp",
    # Gates
    "CustomOp",
    "GlobalPhaseOp",
    "MultiRZOp",
    "PauliRotOp",
    "PCPhaseOp",
    "QubitUnitaryOp",
    # State preparation
    "SetBasisStateOp",
    "SetStateOp",
    # Observables
    "ComputationalBasisOp",
    "HamiltonianOp",
    "HermitianOp",
    "NamedObsOp",
    "TensorOp",
    # Mid-circuit measurements
    "MeasureOp",
    # Terminal measurements
    "CountsOp",
    "ExpvalOp",
    "ProbsOp",
    "SampleOp",
    "StateOp",
    "VarianceOp",
    # Qubit operations
    "AllocOp",
    "AllocQubitOp",
    "DeallocOp",
    "DeallocQubitOp",
    "ExtractOp",
    "InsertOp",
    # Miscellaneous
    "AdjointOp",
    "DeviceInitOp",
    "DeviceReleaseOp",
    "FinalizeOp",
    "InitializeOp",
    "NumQubitsOp",
    "YieldOp",
    # Qubit/Qureg parameters
    "QubitLevel",
    "QubitRole",
    "QubitTypeConstraint",
    "QuregTypeConstraint",
]
