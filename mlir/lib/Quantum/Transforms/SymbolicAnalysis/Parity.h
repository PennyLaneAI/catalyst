#pragma once

#include <iosfwd>
#include <vector>
#include <string>
#include <utility>
#include <functional>

class Parity {
public:
    // Constructors
    Parity() = default;
    Parity(size_t varNum, std::vector <uint64_t>& bits) :
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
    friend std::ostream& operator<<(std::ostream& os, const Parity& par);

    // Checks & Inspections
    [[nodiscard]] bool isIdenticalWith(const Parity& rhs) const;
    [[nodiscard]] bool isUnsat() const;
    [[nodiscard]] bool isLinearZero() const;

    // Getters
    [[nodiscard]] size_t getVarNum() const;
    [[nodiscard]] size_t getLen() const;
    [[nodiscard]] const std::vector<uint64_t>& getBits() const;
    [[nodiscard]] std::vector<uint64_t> getLinearPart() const;  // inefficient, but is only for testing
    [[nodiscard]] std::string getLinearPartString() const;      // inefficient, but is only for testing
    [[nodiscard]] bool getBitAt(size_t pos) const;
    [[nodiscard]] bool getAffineValue() const;

    // Setters
    void setBitAt(size_t pos, bool value);
    void setAffineValue(bool value);
    void onBitAt(size_t pos);
    void offBitAt(size_t pos);
    void flipBitAt(size_t pos);
    void flipAffineValue();
    void extendBitsAtWith(size_t pos, bool value);
    
private:
    using Index = std::pair<size_t, size_t>;   // block index, bit index

    static const size_t BLOCK_SIZE = 64;

    size_t varNum;
    std::vector <uint64_t> bits; // small_vector in mlir
    // Least significant bit (pos = 0) is the affine value. pos = i correspods to x_i

    // Helper Methods
    [[nodiscard]] size_t requiredBlockNum() const;
    [[nodiscard]] Index getIndex(size_t pos) const;
    [[nodiscard]] bool getBitAtBlock(Index ind) const;
    void setBitAtBlock(Index ind, bool value);
    void onBitAtBlock(Index ind);
    void offBitAtBlock(Index ind);
    void flipBitAtBlock(Index ind);
    void extendBitsTo(size_t newVarNum);
    [[nodiscard]] std::string algebraicView() const;
};

inline size_t Parity::getVarNum() const {
    return varNum;
}

inline size_t Parity::getLen() const { 
    return varNum + 1; 
}

inline const std::vector<uint64_t>& Parity::getBits() const {
    return bits;
}

inline size_t Parity::requiredBlockNum() const {
    return (varNum / BLOCK_SIZE) + 1;
}

inline Parity::Index Parity::getIndex(size_t pos) const {
    return {pos / BLOCK_SIZE, pos % BLOCK_SIZE};
}

namespace std {
    template <>
    struct hash<Parity> {
        size_t operator()(const Parity& p) const;
    };
}