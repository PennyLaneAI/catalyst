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
"""Operations for the xDSL Quantum dialect, which mirrors Catalyst's MLIR
Quantum dialect."""

from .gate_ops import (
    CustomOp,
    GateOp,
    GlobalPhaseOp,
    MultiRZOp,
    PauliRotOp,
    PCPhaseOp,
    QubitUnitaryOp,
    SetBasisStateOp,
    SetStateOp,
)
from .measurement_ops import (
    CountsOp,
    ExpvalOp,
    MeasureOp,
    ProbsOp,
    SampleOp,
    StateOp,
    TerminalMeasurementOp,
    VarianceOp,
)
from .miscellaneous_ops import (
    AdjointOp,
    DeviceInitOp,
    DeviceReleaseOp,
    FinalizeOp,
    InitializeOp,
    NumQubitsOp,
    YieldOp,
)
from .observable_ops import (
    ComputationalBasisOp,
    HamiltonianOp,
    HermitianOp,
    NamedObsOp,
    ObservableOp,
    TensorOp,
)
from .qubit_ops import AllocOp, AllocQubitOp, DeallocOp, DeallocQubitOp, ExtractOp, InsertOp

__all__ = [
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
]
