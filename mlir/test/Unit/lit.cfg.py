# -*- Python -*-

# Configuration file for the 'lit' test runner.

##############################################################################
# The following subsection of code                                           #
# was taken from the CIRCT project: https://github.com/llvm/circt            #
# Small alteration were made for Catalyst                                    #
# The CIRCT project has the following license:                               #
#                                                                            #
#   The LLVM Project is under the Apache License v2.0 with LLVM Exceptions:  #
#   As an incubator project with ambition to become part of the LLVM Project,#
#   CIRCT is under the same license.                                         #
#                                                                            #
#  https://github.com/llvm/circt/blob                                        #
#  /a50540ecdbb2db641d14ffb682c9165c206dea26/test/Unit/lit.cfg.py            #
##############################################################################

import os
import subprocess

import lit.formats

# name: The name of this test suite.
config.name = "Catalyst-Unit"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = []

# test_source_root: The root path where tests are located.
# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.proj_obj_root, "unittests")
config.test_source_root = config.test_exec_root

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, "Tests")

# Propagate the temp directory. Windows requires this because it uses \Windows\
# if none of these are present.
if "TMP" in os.environ:
    config.environment["TMP"] = os.environ["TMP"]
if "TEMP" in os.environ:
    config.environment["TEMP"] = os.environ["TEMP"]

# Propagate HOME as it can be used to override incorrect homedir in passwd
# that causes the tests to fail.
if "HOME" in os.environ:
    config.environment["HOME"] = os.environ["HOME"]

# Propagate path to symbolizer for ASan/MSan.
for symbolizer in ["ASAN_SYMBOLIZER_PATH", "MSAN_SYMBOLIZER_PATH"]:
    if symbolizer in os.environ:
        config.environment[symbolizer] = os.environ[symbolizer]
