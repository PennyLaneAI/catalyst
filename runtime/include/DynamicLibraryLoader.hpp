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

    DynamicLibraryLoader(std::string_view library_name, int mode = RTLD_LAZY)
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
            // dlclose(handle); TODO: investigate
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
