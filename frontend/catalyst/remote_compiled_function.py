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

"""Remote shared-object manager"""

import logging

# pylint: disable=c-extension-no-member

from catalyst.utils import remote_driver as driver
from catalyst.logging import debug_logger_init


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class RemoteSharedObjectManager:
    """Remote shared-object manager.

    Args:
        shared_object_file (str): path to the kernel object file
        func_name (str): name of the compiled function
        remote (str): address of the remote executor
    """

    @debug_logger_init
    def __init__(self, shared_object_file, func_name, remote):
        self.shared_object_file = shared_object_file
        self.func_name = func_name
        self.function = None
        self.setup = None
        self.teardown = None
        self.mem_transfer = None

        self.remote = remote
        self.session = None
        self.open()

    def open(self):
        """Open the remote session and load symbols."""
        self.session = driver.open(self.shared_object_file, self.remote)
        self.function, self.setup, self.teardown, self.mem_transfer = self.load_symbols()

    def load_symbols(self):
        """Load symbols necessary for the remote execution of the compiled function.

        Returns:
            RemoteFunc: handle holding (session, pyface_addr)
            Callable: remote setup function
            Callable: remote teardown function
            None: no mem_transfer for remote execution
        """
        sess = self.session
        pyface_addr = driver.lookup(sess, "_catalyst_pyface_" + self.func_name)
        setup_addr = driver.lookup(sess, "setup")
        teardown_addr = driver.lookup(sess, "teardown")

        function = driver.RemoteFunc(sess, pyface_addr)

        def setup(_argc, argv):
            driver.run_as_main(sess, setup_addr, argv)

        def teardown():
            driver.run_as_main(sess, teardown_addr, [])

        return function, setup, teardown, None

    def __enter__(self):
        params_to_setup = ["jitted-function"]
        self.setup(len(params_to_setup), params_to_setup)
        return self

    def __exit__(self, _type, _value, _traceback):
        self.teardown()
