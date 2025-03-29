// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <dlfcn.h>
#include <string_view>

#include "Exception.hpp"

/**
 * @brief A utility struct to handle opening, closing and retrieving symbols
 *        from dynamic shared objects.
 */
struct DynamicLibraryLoader {
    void *handle;

    DynamicLibraryLoader(std::string_view library_name, int mode = RTLD_LAZY | RTLD_NODELETE)
    {
        // Load the shared library
        handle = dlopen(library_name.data(), mode);
        if (!handle) {
            const char *err_msg = dlerror();
            RT_FAIL(err_msg);
        }
    }

    ~DynamicLibraryLoader()
    {
        if (handle) {
            // TODO: This is non-sensical.
            // We are using RTLD_NODELETE, why would calling dlclose have a side-effect?
            // Worst of all, the side-effect is not in our code.
            // When we have dlclose, everything works well the first time.
            // However, when trying to compile a second time, we will find that jaxlib will now
            // raise a StopIteration exception. This doesn't really make any sense.
            // My guess is that somehow dlclosing here will unload a the StopIteration symbol (?)
            // rebind it with another equivalent (but with different id?)
            // and then the MLIR python bindings are unable to catch it and stop the iteration and
            // it gets propagated upwards.
            //
            // Is not calling dlclose bad?
            // A little bit, although dlclose implies intent and does not create any requirements
            // upon the implementation. See here:
            // https://pubs.opengroup.org/onlinepubs/000095399/functions/dlclose.html
            // https://github.com/pybind/pybind11/blob/75e48c5f959b4f0a49d8c664e059b6fb4b497102/include/pybind11/detail/internals.h#L108-L113
            //
#ifndef __APPLE__
            dlclose(handle);
#endif
        }
    }

    // Get symbol from library
    template <typename T> T getSymbol(std::string_view symbol_name)
    {
        // Clear any existing errors
        dlerror();

        // Retrieve symbol
        T symbol = reinterpret_cast<T>(dlsym(handle, symbol_name.data()));
        const char *err_msg = dlerror();
        if (err_msg != nullptr) {
            RT_FAIL(err_msg);
        }
        return symbol;
    }
};
