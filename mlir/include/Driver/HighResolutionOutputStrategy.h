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

#pragma once

#include "mlir/Support/Timing.h"
#include "llvm/Support/raw_ostream.h"

class HighResolutionOutputStrategy : public mlir::OutputStrategy {
  public:
    HighResolutionOutputStrategy(llvm::raw_ostream &os) : mlir::OutputStrategy(os) {}
    void printTime(const mlir::TimeRecord &time, const mlir::TimeRecord &total) override;
    void printListEntry(llvm::StringRef name, const mlir::TimeRecord &time,
                        const mlir::TimeRecord &total, bool lastEntry) override;
    void printTreeEntry(unsigned indent, llvm::StringRef name, const mlir::TimeRecord &time,
                        const mlir::TimeRecord &total) override;
    void printTreeEntryEnd(unsigned indent, bool lastEntry) override;
    void printHeader(const mlir::TimeRecord &total) override;
    void printFooter() override;
};
