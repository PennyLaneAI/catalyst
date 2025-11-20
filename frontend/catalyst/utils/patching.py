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
Patcher module.
"""


class DictPatchWrapper:
    """A wrapper to enable dictionary item patching using attribute-like access.

    This allows the Patcher class to patch dictionary items by wrapping the dictionary
    and key into an object where the item can be accessed as an attribute.

    Args:
        dictionary: The dictionary to wrap
        key: The key to access in the dictionary
    """

    def __init__(self, dictionary, key):
        self.dictionary = dictionary
        self.key = key

    def __getattr__(self, name):
        if name in ("dictionary", "key"):  # pragma: no cover
            return object.__getattribute__(self, name)
        if name == "value":
            return self.dictionary[self.key]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in ("dictionary", "key"):
            object.__setattr__(self, name, value)
        elif name == "value":
            self.dictionary[self.key] = value
        else:  # pragma: no cover
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class Patcher:
    """Patcher, a class to replace object attributes.

    Args:
        patch_data: List of triples. The first element in the triple corresponds to the object
        whose attribute is to be replaced. The second element is the attribute name. The third
        element is the new value assigned to the attribute.
    """

    def __init__(self, *patch_data):
        self.backup = {}
        self.patch_data = patch_data

        assert all(len(data) == 3 for data in patch_data)

    def __enter__(self):
        for obj, attr_name, fn in self.patch_data:
            self.backup[(obj, attr_name)] = getattr(obj, attr_name)
            setattr(obj, attr_name, fn)

    def __exit__(self, _type, _value, _traceback):
        for obj, attr_name, _ in self.patch_data:
            setattr(obj, attr_name, self.backup[(obj, attr_name)])
