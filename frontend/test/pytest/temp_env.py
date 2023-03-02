# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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
Temporarily modify environment variables.
"""
import os

# pylint: disable=missing-class-docstring
class TempEnv:

    def __init__(self, **kwargs):
        self.backup = os.environ
        self.kwargs = kwargs

    def __enter__(self):
        for key, value in self.kwargs.items():
            os.environ[key] = value

    def __exit__(self, exc_t, exc_v, exc_tb):
        for key in self.kwargs:
            del os.environ[key]
        for key, value in self.backup.items():
            os.environ[key] = value
