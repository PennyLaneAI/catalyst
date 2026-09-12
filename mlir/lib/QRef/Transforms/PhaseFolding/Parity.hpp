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

#include <cassert>
#include <functional>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace catalyst::phase_folding {

struct BitLocation {
    size_t block;
    size_t bit;

    // Constructors
    BitLocation() = default;
    constexpr BitLocation(size_t block, size_t bit) : block(block), bit(bit) {}
    constexpr BitLocation(size_t pos) : BitLocation(pos / BLOCK_SIZE, pos % BLOCK_SIZE) {}

    // Static Constants:
    static const size_t BLOCK_SIZE = 64;
    static const BitLocation AFFINE_VALUE;

    // Operators
    BitLocation operator++();
    BitLocation operator++(int);

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const BitLocation &bitLoc);
    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                         const std::vector<BitLocation> &idxList);
    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                         llvm::ArrayRef<BitLocation> idxView);

    static size_t requiredBlockNum(size_t varNum);

    [[nodiscard]] size_t toPos() const;
};

inline constexpr BitLocation BitLocation::AFFINE_VALUE{0, 0}; // it's LSB. for MSB would be varNum.

inline size_t BitLocation::toPos() const { return block * BLOCK_SIZE + bit; }
inline size_t BitLocation::requiredBlockNum(size_t varNum) { return (varNum / BLOCK_SIZE) + 1; }

using IdxList = std::vector<BitLocation>;
using IdxView = llvm::ArrayRef<BitLocation>;

class Parity {
  public:
    // Constructors
    Parity() = default;
    Parity(const llvm::SmallVector<uint64_t, 8> &bits) : bits(bits) {}
    explicit Parity(size_t blockNum) : bits(blockNum, 0) {}

    // Static Factories
    static Parity eVec(size_t blockNum, BitLocation oneLoc);
    static Parity Trivial(size_t varNum);
    static Parity Unsat(size_t varNum);

    // Operators
    bool operator==(const Parity &rhs) const;
    Parity &operator+=(const Parity &rhs);
    Parity operator+(const Parity &rhs) const; // XOR

    friend struct std::hash<Parity>;
    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Parity &par);

    template <typename ColOrderRange>
    [[nodiscard]] std::string toStringWithOrder(ColOrderRange order) const;
    [[nodiscard]] std::string toString() const;

    // Getters
    [[nodiscard]] const llvm::SmallVector<uint64_t, 8> &getBits() const;
    [[nodiscard]] bool getBitAtLoc(BitLocation ind) const;
    [[nodiscard]] bool getBitAtPos(size_t pos) const;

    // Setters
    void reset(std::optional<size_t> newNumBlocks = 1);
    void mkBasis(BitLocation oneLoc, size_t numBlocks);
    void mkTrivial(std::optional<size_t> newNumBlocks = 1);
    void mkUnsat(BitLocation affVal, std::optional<size_t> newNumBlocks = 1);
    void assignBitAtLoc(BitLocation ind, bool value);
    void assignBitAtPos(size_t pos, bool value);
    void setBitAtLoc(BitLocation ind);
    void setBitAtPos(size_t pos);
    void clearBitAtLoc(BitLocation ind);
    void flipBitAtLoc(BitLocation ind);

    // Checks & Inspections
    [[nodiscard]] bool isIdenticalWith(const Parity &rhs) const;
    template <typename ColOrderRange>
    [[nodiscard]] bool isTrivialInRange(ColOrderRange checkedRange) const;
    [[nodiscard]] bool isTrivial() const;
    [[nodiscard]] bool isUnsat(BitLocation affValLoc) const;

    // Methods:
    void extendBitsFor(size_t newMaxBlock);
    template <typename ColOrderRange>
    void mapBitsFrom(const Parity &srcPar, ColOrderRange srcLocs, ColOrderRange trgtLocs);
    void mapBitFrom(const Parity &srcPar, BitLocation srcLoc, BitLocation trgtLoc);

  private:
    llvm::SmallVector<uint64_t, 8> bits;

    // DenseMap Helpers
    enum class State : uint8_t { Valid, Empty, Tombstone } state = State::Valid;
    explicit Parity(State s) : state(s) {}
    friend struct llvm::DenseMapInfo<Parity>;

    // Helper Methods
    bool isTrivialFromBlock(size_t fstBlock) const;
    bool isTrivialInBlocks(size_t fstBlock, size_t lstBlock) const;
    bool isEquivalentWithFromBlock(const Parity &rhs, size_t fstBlock) const;
};

template <typename ColOrderRange> std::string Parity::toStringWithOrder(ColOrderRange order) const {
    std::string res = "";
    for (BitLocation loc : order) {
        res += (getBitAtLoc(loc) ? '1' : '0');
    }
    return res;
}

template <typename ColOrderRange>
void Parity::mapBitsFrom(const Parity &srcPar, ColOrderRange srcLocs, ColOrderRange trgtLocs) {
    for (auto srcIt = srcLocs.begin(), trgtIt = trgtLocs.begin(); srcIt != srcLocs.end();
         ++srcIt, ++trgtIt) {
        mapBitFrom(srcPar, *srcIt, *trgtIt);
    }
} // inefficient

template <typename ColOrderRange> bool Parity::isTrivialInRange(ColOrderRange checkedRange) const {
    for (BitLocation loc : checkedRange) {
        if (getBitAtLoc(loc)) {
            return false;
        }
    }
    return true;
} // inefficient

inline Parity Parity::Trivial(size_t varNum) {
    return Parity(BitLocation::requiredBlockNum(varNum));
}

inline Parity Parity::Unsat(size_t varNum) {
    return eVec(BitLocation::requiredBlockNum(varNum), BitLocation(0, 0));
}

// Getters:
inline const llvm::SmallVector<uint64_t, 8> &Parity::getBits() const { return bits; }

inline bool Parity::getBitAtLoc(BitLocation loc) const {
    return (loc.block < bits.size()) ? (bits[loc.block] & (1ULL << loc.bit)) : 0;
}

inline bool Parity::getBitAtPos(size_t pos) const { return getBitAtLoc(BitLocation(pos)); }

// Setters:
inline void Parity::mkBasis(BitLocation oneLoc, size_t numBlocks) {
    reset(numBlocks);
    setBitAtLoc(oneLoc);
}

inline void Parity::mkTrivial(std::optional<size_t> newNumBlocks) { reset(newNumBlocks); }

inline void Parity::mkUnsat(BitLocation affVal, std::optional<size_t> newNumBlocks) {
    mkBasis(affVal, newNumBlocks.value_or(1));
}

inline void Parity::reset(std::optional<size_t> newNumBlocks) {
    bits.assign(newNumBlocks.value(), 0);
}

inline void Parity::assignBitAtPos(size_t pos, bool value) {
    assignBitAtLoc(BitLocation(pos), value);
}

inline void Parity::setBitAtPos(size_t pos) { setBitAtLoc(BitLocation(pos)); }

// Checks & Inspections
inline bool Parity::isTrivial() const { return isTrivialFromBlock(0); }

inline bool Parity::isTrivialFromBlock(size_t fstBlock) const {
    return isTrivialInBlocks(fstBlock, bits.size());
}

inline bool Parity::isUnsat(BitLocation affValLoc) const {
    assert(affValLoc.block < bits.size());
    return (bits[affValLoc.block] != 1 << affValLoc.bit)
               ? false
               : isTrivialInBlocks(0, affValLoc.block) && isTrivialFromBlock(affValLoc.block + 1);
} // currently unsat is 0..01, but we can change it to empty bits.

inline bool Parity::isIdenticalWith(const Parity &rhs) const { return bits == rhs.bits; }

// Methods:
inline void Parity::mapBitFrom(const Parity &srcPar, BitLocation srcLoc, BitLocation trgtLoc) {
    if (srcPar.getBitAtLoc(srcLoc)) {
        setBitAtLoc(trgtLoc);
    }
}

} // namespace catalyst::phase_folding

namespace llvm {
template <> struct DenseMapInfo<catalyst::phase_folding::Parity> {
    static inline catalyst::phase_folding::Parity getEmptyKey() {
        return catalyst::phase_folding::Parity(catalyst::phase_folding::Parity::State::Empty);
    }

    static inline catalyst::phase_folding::Parity getTombstoneKey() {
        return catalyst::phase_folding::Parity(catalyst::phase_folding::Parity::State::Tombstone);
    }

    static unsigned getHashValue(const catalyst::phase_folding::Parity &val) {
        if (val.state != catalyst::phase_folding::Parity::State::Valid) {
            return 0;
        }
        return static_cast<unsigned>(llvm::hash_combine_range(val.bits.begin(), val.bits.end()));
    }

    static bool isEqual(const catalyst::phase_folding::Parity &lhs,
                        const catalyst::phase_folding::Parity &rhs) {
        return lhs == rhs;
    }
};
} // namespace llvm
