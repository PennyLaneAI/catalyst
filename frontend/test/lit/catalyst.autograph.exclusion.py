"""This module can be used to test autograph exclusion/inclusion mechanisms in Catalyst.

The naming of the module is such that it should be excluded by Catalyst's default filters."""

import catalyst
from catalyst.utils.patching import Patcher


def dummy_func(x):
    """Simple function with if statements for testing the 'auto_include' option of @qjit.
    The parent 'catalyst' module is excluded for autograph conversion by default, hence
    adding this module explicitly to the inclusion list will override that restriction"""

    with Patcher((catalyst, "compile_without_static_conditionals", False)):
        if x > 5:
            y = x**2
        else:
            y = x**3
    return y
