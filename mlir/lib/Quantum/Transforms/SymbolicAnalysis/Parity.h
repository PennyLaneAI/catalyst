#pragma once

// #include <iosfwd>
#include <string>
#include <utility>
#include <functional>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMapInfo.h" // Needed for DenseMapInfo
#include "llvm/ADT/Hashing.h"      // Needed for hash_combine_range
#include "llvm/Support/raw_ostream.h"

class Parity {
public:
    // Constructors
    Parity() = default;
    Parity(size_t varNum, llvm::SmallVector<uint64_t, 8>& bits) :
        varNum(varNum), bits(bits) {}
    explicit Parity(size_t varNum) :
        varNum(varNum), bits(requiredBlockNum(), 0) {}
    explicit Parity(const std::string& parityStr);
    Parity(const std::string& linearPart, bool affineValue) :
        Parity(linearPart + (affineValue ? "1" : "0")) {}
    
    // Static Factories
    static Parity eVec(size_t varNum, size_t pos);

    // Operators
    bool operator==(const Parity& rhs) const;    // their affine values are being checked too.
    Parity& operator+=(const Parity& rhs);
    Parity operator+(const Parity& rhs) const;   // XOR

    friend struct std::hash<Parity>;
    friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Parity& par);
    std::string algebraicView(size_t qubitNum) const;

    // Checks & Inspections
    [[nodiscard]] bool isIdenticalWith(const Parity& rhs) const;
    [[nodiscard]] bool isLinearEquivalentWith(const Parity& rhs) const;
    [[nodiscard]] bool isUnsat() const;
    [[nodiscard]] bool isLinearZero() const;

    // Getters
    [[nodiscard]] size_t getVarNum() const;
    [[nodiscard]] size_t getLen() const;
    [[nodiscard]] const llvm::SmallVector<uint64_t, 8>& getBits() const;
    [[nodiscard]] llvm::SmallVector<uint64_t, 8> getLinearPart() const;  // inefficient, but is only for testing
    [[nodiscard]] std::string getLinearPartString() const;      // inefficient, but is only for testing
    [[nodiscard]] bool getBitAt(size_t pos) const;
    [[nodiscard]] bool getAffineValue() const;

    // Setters
    void setBitAt(size_t pos, bool value);
    void setAffineValue(bool value);
    void onBitAt(size_t pos);
    void offBitAt(size_t pos);
    void clearAffineValue();
    void flipBitAt(size_t pos);
    void flipAffineValue();
    void extendBitsAtWith(size_t pos, bool value);
    
private:
    using Index = std::pair<size_t, size_t>;   // block index, bit index

    static const size_t BLOCK_SIZE = 64;

    size_t varNum;  // should be removed
    llvm::SmallVector<uint64_t, 8> bits;
    // Least significant bit (pos = 0) is the affine value. pos = i corresponds to x_i

    // DenseMap Helpers
    enum class State : uint8_t { Valid, Empty, Tombstone } state = State::Valid;
    explicit Parity(State s) : state(s) {}
    friend struct llvm::DenseMapInfo<Parity>;

    // Helper Methods
    [[nodiscard]] size_t requiredBlockNum() const;
    [[nodiscard]] Index getIndex(size_t pos) const;
    [[nodiscard]] bool getBitAtBlock(Index ind) const;
    [[nodiscard]] bool isEquivalentWithFromBlock(const Parity& rhs, size_t fstBlock) const;
    void setBitAtBlock(Index ind, bool value);
    void onBitAtBlock(Index ind);
    void offBitAtBlock(Index ind);
    void flipBitAtBlock(Index ind);
    void extendBitsTo(size_t newVarNum);
};

inline size_t Parity::getVarNum() const {
    return varNum;
}

inline size_t Parity::getLen() const { 
    return varNum + 1; 
}

inline const llvm::SmallVector<uint64_t, 8>& Parity::getBits() const {
    return bits;
}

inline size_t Parity::requiredBlockNum() const {
    return (varNum / BLOCK_SIZE) + 1;
}

inline Parity::Index Parity::getIndex(size_t pos) const {
    return {pos / BLOCK_SIZE, pos % BLOCK_SIZE};
}

namespace llvm {
template <>
struct DenseMapInfo<Parity> {
    static inline Parity getEmptyKey() {
        return Parity(Parity::State::Empty);
    }

    static inline Parity getTombstoneKey() {
        return Parity(Parity::State::Tombstone);
    }

    static unsigned getHashValue(const Parity& val) {
        if (val.state != Parity::State::Valid) {
            return 0; 
        }
        return static_cast<unsigned>(
            llvm::hash_combine_range(val.bits.begin(), val.bits.end())
        );
    }

    static bool isEqual(const Parity& lhs, const Parity& rhs) {
        return lhs == rhs;
    }
};
} // namespace llvm
