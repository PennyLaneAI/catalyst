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

#include "Driver/HighResolutionOutputStrategy.h"
#include "llvm/Support/Format.h"

using namespace mlir;

constexpr llvm::StringLiteral kTimingDescription = "... Execution time report ...";

void HighResolutionOutputStrategy::printHeader(const TimeRecord &total)
{
    // Figure out how many spaces to description name.
    unsigned padding = (80 - kTimingDescription.size()) / 2;
    os << "===" << std::string(73, '-') << "===\n";
    os.indent(padding) << kTimingDescription << '\n';
    os << "===" << std::string(73, '-') << "===\n";

    // Print the total time followed by the section headers.
    os << llvm::format("  Total Execution Time: %.9f seconds\n\n", total.wall);
    if (total.user != total.wall)
        os << "  ----User Time----";
    os << "  ----Wall Time----  ----Name----\n";
}

void HighResolutionOutputStrategy::printFooter() { os.flush(); }

void HighResolutionOutputStrategy::printTime(const TimeRecord &time, const TimeRecord &total)
{
    if (total.user != total.wall) {
        os << llvm::format("  %10.9f (%5.1f%%)", time.user, 100.0 * time.user / total.user);
    }
    os << llvm::format("  %10.9f (%5.1f%%)  ", time.wall, 100.0 * time.wall / total.wall);
}
void HighResolutionOutputStrategy::printListEntry(llvm::StringRef name, const TimeRecord &time,
                                                  const TimeRecord &total, bool lastEntry)
{
    printTime(time, total);
    os << name << "\n";
}

void HighResolutionOutputStrategy::printTreeEntry(unsigned indent, llvm::StringRef name,
                                                  const TimeRecord &time, const TimeRecord &total)
{
    printTime(time, total);
    os.indent(indent) << name << "\n";
}

void HighResolutionOutputStrategy::printTreeEntryEnd(unsigned indent, bool lastEntry) {}
