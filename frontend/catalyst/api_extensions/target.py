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

"""Tag a PennyLane device as a separate cross-compilation target.
"""

from dataclasses import dataclass
from typing import Optional

import pennylane as qp

_TARGET_ATTR = "_catalyst_target"


@dataclass(frozen=True)
class Target:
    """Cross-compilation target spec attached to a device by :func:`target`.

    Args:
        backend: Optional backend name, recorded as metadata on the target module.
        pipeline: Optional name of a lowering pipeline registered with compiler.
        triple: Optional LLVM target triple. Defaults to the host triple.
    """

    backend: Optional[str] = None
    pipeline: Optional[str] = None
    triple: Optional[str] = None


def target(
    device,
    *,
    backend: Optional[str] = None,
    pipeline: Optional[str] = None,
    triple: Optional[str] = None,
):
    """Tag a PennyLane device as a separate cross-compilation target and return it.

    Any QNode wrapping the returned device is kept as a separate compilation unit which carries
    ``catalyst.target = {backend, pipeline, triple}`` and is cross-compiled to a standalone 
    object file rather than being inlined into the host module.

    Args:
        device: A PennyLane device.
        backend: Optional backend name, recorded as metadata on the target module.
        pipeline: Optional lowering-pipeline name registered with the compiler.
        triple: Optional LLVM target triple. Defaults to the host triple.

    Returns:
        The same device, now tagged with target metadata.
    """
    setattr(device, _TARGET_ATTR, Target(backend=backend, pipeline=pipeline, triple=triple))
    return device


def get_target(device) -> Optional[Target]:
    """Return the :class:`Target` previously attached via :func:`target`, or ``None``."""
    return getattr(device, _TARGET_ATTR, None)
