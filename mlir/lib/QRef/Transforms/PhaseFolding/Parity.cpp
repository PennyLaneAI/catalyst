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

#include "Parity.hpp"

#include "llvm/ADT/Hashing.h"

using namespace catalyst::phase_folding;

BitLocation BitLocation::operator++() {
    bit++;
    if (bit == BLOCK_SIZE) {
        bit = 0;
        block++;
    }
    return *this;
}

BitLocation BitLocation::operator++(int) {
    BitLocation p = *this;
    ++(*this);
    return p;
}

/*
    Constructors:
*/
Parity Parity::eVec(size_t blockNum, BitLocation oneLoc) {
    assert(oneLoc.block <= blockNum);

    Parity res = Parity(blockNum);
    res.setBitAtLoc(oneLoc);
    return res;
}

/*
    Operators:
*/
bool Parity::operator==(const Parity &rhs) const {
    if (state != rhs.state) {
        return false;
    }
    if (state != State::Valid) {
        return true;
    }
    return isEquivalentWithFromBlock(rhs, 0);
}

Parity &Parity::operator+=(const Parity &rhs) {
    size_t minBlockNum = std::min(bits.size(), rhs.bits.size());

    for (size_t i = 0; i < minBlockNum; ++i) {
        bits[i] ^= rhs.bits[i];
    }

    if (bits.size() < rhs.bits.size()) {
        bits.insert(bits.end(), rhs.bits.begin() + minBlockNum, rhs.bits.end());
    }
    return *this;
}

Parity Parity::operator+(const Parity &rhs) const {
    Parity res = *this;
    res += rhs;
    return res;
}

std::string Parity::toString() const {
    std::string res = "";
    for (size_t i = 0; i < bits.size(); ++i) {
        for (size_t j = 0; j < BitLocation::BLOCK_SIZE; j++) {
            res += (getBitAtLoc(BitLocation(i, j)) ? '1' : '0');
        }
    }
    return res;
}

/*
    Setters:
*/
void Parity::assignBitAtLoc(BitLocation loc, bool value) {
    if (loc.block >= bits.size()) {
        bits.resize(loc.block + 1, 0);
    }

    uint64_t mask = 1ULL << loc.bit;
    bits[loc.block] = (bits[loc.block] & ~mask) | (static_cast<uint64_t>(value) << loc.bit);
}

void Parity::setBitAtLoc(BitLocation loc) {
    if (loc.block >= bits.size()) {
        bits.resize(loc.block + 1, 0);
    }

    bits[loc.block] |= (1ULL << loc.bit);
}

void Parity::clearBitAtLoc(BitLocation loc) {
    if (loc.block >= bits.size()) {
        bits.resize(loc.block + 1, 0);
        return;
    }

    bits[loc.block] &= ~(1ULL << loc.bit);
}

void Parity::flipBitAtLoc(BitLocation loc) {
    if (loc.block >= bits.size()) {
        bits.resize(loc.block + 1, 0);
    }

    bits[loc.block] ^= (1ULL << loc.bit);
}

/*
    Methods:
*/
void Parity::extendBitsFor(size_t newMaxBlock) {
    const size_t newBlockNum = newMaxBlock + 1;
    if (bits.size() < newBlockNum) {
        bits.resize(newBlockNum, 0);
    }
}
/*
    Checks:
*/

bool Parity::isTrivialInBlocks(size_t fstBlock, size_t lstBlock) const {
    for (size_t i = fstBlock; i < lstBlock; ++i) {
        if (bits[i] != 0) {
            return false;
        }
    }
    return true;
}

// check if they are the same up to adding some 0 bits in the end.
// (so they can be considered equal if we had added some path or new vaiables and now their 0)
bool Parity::isEquivalentWithFromBlock(const Parity &rhs, size_t fstBlock) const {
    size_t minBlockNum = std::min(bits.size(), rhs.bits.size());

    for (size_t i = fstBlock; i < minBlockNum; ++i) {
        if (bits[i] != rhs.bits[i]) {
            return false;
        }
    }

    const llvm::SmallVector<uint64_t, 8> &longerBits =
        (bits.size() > rhs.bits.size()) ? bits : rhs.bits;
    for (size_t i = minBlockNum; i < longerBits.size(); ++i) {
        if (longerBits[i] != 0) {
            return false;
        }
    }
    return true;
}

/*
    Print:
*/
namespace catalyst::phase_folding {

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const BitLocation &bitLoc) {
    os << "(" << bitLoc.block << ", " << bitLoc.bit << ")";
    return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const IdxList &idxMap) {
    os << "[";
    for (size_t i = 0; i < idxMap.size(); ++i) {
        os << idxMap[i];
        if (i < idxMap.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, IdxView idxView) {
    os << "[";
    for (size_t i = 0; i < idxView.size(); ++i) {
        os << idxView[i];
        if (i < idxView.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Parity &par) {
    os << par.toString();
    return os;
}
} // namespace catalyst::phase_folding
