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

#include "AffineSchema.hpp"

using namespace catalyst::phase_folding;

IdxList AffineSchema::getFreeLocs(size_t n) const {
    IdxList locs;
    locs.reserve(n);

    size_t k = std::min(n, recycledLocs.size());
    locs.insert(locs.end(), recycledLocs.end() - k, recycledLocs.end());
    recycledLocs.resize(recycledLocs.size() - k);

    for (int i = n - k; i > 0; --i) {
        locs.push_back(++maxLoc);
    }

    return locs;
}

BitLocation AffineSchema::getFreeLoc() const {
    if (!recycledLocs.empty()) {
        BitLocation loc = recycledLocs.back();
        recycledLocs.pop_back();
        return loc;
    }
    return ++maxLoc;
}

/*
    Print:
*/
namespace catalyst::phase_folding {

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const AffineSchema &schema) {
    os << " PreVars: " << schema.preVars << "\n";
    os << " AuxVars: " << schema.auxVars << "\n";
    os << " AffineValue: " << schema.affVal << "\n";

    os << " Recycled Locs: " << schema.getRecycledLocs() << "\n";
    os << " Max Idx: " << schema.getMaxLoc() << "\n";
    os << "------\n";
    return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const RelationSchema &relSchm) {
    os << " PostVars: " << relSchm.postVars << "\n";
    const AffineSchema &schema = relSchm;
    os << schema;
    return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const JoinSchema &joinSchm) {
    os << " L_PostVars: " << joinSchm.lPostVars << "\n";
    os << " L_PreVars: " << joinSchm.lPreVars << "\n";
    // os << " L_AuxVars: " << joinSchm.lAuxVars << "\n";
    os << " L_AffineValue: " << joinSchm.lAffVal << "\n";

    const RelationSchema &relSchm = joinSchm;
    os << relSchm;
    return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const CompositionSchema &cmpSchm) {
    os << " ProjVars: " << cmpSchm.projVars << "\n";

    const RelationSchema &relSchm = cmpSchm;
    os << relSchm;
    return os;
}
} // namespace catalyst::phase_folding
