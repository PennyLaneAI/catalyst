# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains public API functions that provide control for the 
user to input what MLIR compiler passes to run. 

Currently, each pass has its own user-facing decorator. In the future, 
a unified user interface for all the passes is necessary. 

Note that the pass pipeline table does not need to modify the qnode in 
any way. Its only purpose it to mark down the passes the user wants to 
run on each qnode, and let send this information to 
frontend/catalyst/compiler.py to handle the actual running of the passes. 
"""

import pennylane as qml


## PASS PIPELINE TABLE ##
class QUANTUM_CIRCUIT_TRANSFORM_PASS_TABLE:
    """
    A class that records the active compilation passes on a qnode.
    Note that the ordering of passes of course matters.
    """

    def __init__(self):
        self._table = {}

    def __repr__(self):
        return str(self.table)

    @property
    def table(self):
        """
        Return the table object.
        """
        return self._table

    def add_pass_on_qnode(self, qnode, pass_):
        """
        qnode (QNODE): the qnode object to compile
        pass_ (str):   the compiler pass to run on this object.
                       At each call of add_pass, the pass is added to the end
                       of the current pipeline for that qnode.

        """
        if qnode not in self.table:
            self.table[qnode] = [pass_]
        else:
            self.table[qnode].append(pass_)

    def query(self, qnode):
        """
        Look up a qnode from the table and return the list of passes to run on it.
        """
        return self.table[qnode]


active_passes = QUANTUM_CIRCUIT_TRANSFORM_PASS_TABLE()


def get_quantum_circuit_transform_pass_table():
    """
    To be called in compiler.py to retrieve the pipeline table.
    """
    return active_passes


## API ##
def cancel_inverses(fn=None):
    """
    The top-level `catalyst.cancel_inverses` decorator.
    !!! TODO: add documentation here !!!
    """
    global active_passes  # pylint: disable

    if not isinstance(fn, qml.QNode):
        raise TypeError(f"A QNode is expected, got the classical function {fn}")
    active_passes.add_pass_on_qnode(fn, "remove-chained-self-inverse")
    return fn
