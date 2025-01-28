// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>

#include "OQDRuntimeCAPI.h"

extern "C" {

void __catalyst__oqd__greetings() { std::cout << "Hello OQD world!" << std::endl; }

void __catalyst__oqd__ion(Ion *ion)
{
    std::cout << "Hello Ion! " << (ion == nullptr) << std::endl;
    std::cout << "ion ptr: " << ion << std::endl;
    std::cout << "ion name: " << ion->name << std::endl;
    std::cout << ion->mass << std::endl;
    std::cout << ion->charge << std::endl;
    std::cout << (ion->position)[0]  << " , "
    			<< (ion->position)[1] << " , "
    			<< (ion->position)[2] << std::endl;
    std::cout << ion->levels[0].spin << std::endl;
    std::cout << ion->transitions[2].einstein_a << std::endl;
}

} // extern "C"
