// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Helpers shared by the transport C API tests.

#pragma once

#include <iostream>
#include <sstream>
#include <string>

// Redirects std::cerr into a string for the lifetime of the object.
class CerrCapture {
  public:
    CerrCapture() : old_(std::cerr.rdbuf(buf_.rdbuf())) {}
    ~CerrCapture() { std::cerr.rdbuf(old_); }
    CerrCapture(const CerrCapture &) = delete;
    CerrCapture &operator=(const CerrCapture &) = delete;
    std::string str() const { return buf_.str(); }

  private:
    std::ostringstream buf_;
    std::streambuf *old_;
};
