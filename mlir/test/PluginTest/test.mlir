// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: machine="$(%PYTHON -c 'import platform; print(platform.system(), platform.machine())')" ; \
// RUN:	if [ "${machine}" == "Linux x86_64" ] ; then \
// RUN:		quantum-opt \
// RUN:			--load-dialect-plugin=$(dirname %s)/StandalonePlugin.so \
// RUN:			--load-pass-plugin=$(dirname %s)/StandalonePlugin.so \
// RUN:			--pass-pipeline='builtin.module(standalone-switch-bar-foo)' %s \
// RUN:		| FileCheck %s ; \
// RUN:		exit $? ; \
// RUN:	fi ; \
// RUN:	exit 1


module {
    // CHECK-LABEL: func.func @foo()
    func.func @bar() { 
     return
    }
}
