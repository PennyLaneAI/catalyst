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
"""Test callbacks"""

import pennylane as qml

from catalyst.pennylane_extensions import callback


def test_callback_no_returns_no_params(capsys):
    """Test callback no parameters no returns"""

    def my_callback():
        print("Hello erick")

    @qml.qjit
    def cir():
        callback(my_callback, [])
        return None

    cir()
    captured = capsys.readouterr()
    assert captured.out.strip() == "Hello erick"


def test_callback_twice(capsys):
    """Test callback no parameters no returns"""

    def my_callback():
        print("Hello erick")

    @qml.qjit
    def cir():
        callback(my_callback, [])
        return None

    cir()
    captured = capsys.readouterr()
    assert captured.out.strip() == "Hello erick"

    @qml.qjit
    def cir2():
        callback(my_callback, [])
        return None

    cir2()
    captured = capsys.readouterr()
    assert captured.out.strip() == "Hello erick"
