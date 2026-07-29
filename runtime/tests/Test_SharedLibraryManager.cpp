// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_string.hpp"

#include "Exception.hpp"
#include "ExecutionContext.hpp"

using namespace Catalyst::Runtime;
using Catch::Matchers::ContainsSubstring;

#ifdef __APPLE__
constexpr const char *kPlatformExt = ".dylib";
constexpr const char *kWrongExt = ".so";
#else
constexpr const char *kPlatformExt = ".so";
constexpr const char *kWrongExt = ".dylib";
#endif

static const std::string kNullQubitAbs = RTD_NULL_QUBIT_LIB;
static const std::string kNullQubitBasename = std::string("librtd_null_qubit") + kPlatformExt;

TEST_CASE("SharedLibraryManager loads from absolute path", "[shared_lib]")
{
    REQUIRE_NOTHROW(SharedLibraryManager{kNullQubitAbs});
}

TEST_CASE("SharedLibraryManager rewrites wrong extension", "[shared_lib]")
{
    std::string mangled = kNullQubitAbs;
    auto dot = mangled.find_last_of('.');
    REQUIRE(dot != std::string::npos);
    mangled.resize(dot);
    mangled += kWrongExt;

    REQUIRE_NOTHROW(SharedLibraryManager{mangled});
}

TEST_CASE("SharedLibraryManager falls back when directory is stale", "[shared_lib]")
{
    std::string mangled = "/this/path/does/not/exist/" + kNullQubitBasename;
    REQUIRE_NOTHROW(SharedLibraryManager{mangled});
}

TEST_CASE("SharedLibraryManager throws with original filename on hard failure", "[shared_lib]")
{
    const std::string bogus = "lib_not_real.so";
    REQUIRE_THROWS_WITH(SharedLibraryManager{bogus}, ContainsSubstring(bogus));
}

TEST_CASE("SharedLibraryManager resolves a known symbol", "[shared_lib]")
{
    SharedLibraryManager mgr{kNullQubitAbs};
    REQUIRE(mgr.getSymbol("NullQubitFactory") != nullptr);
}
