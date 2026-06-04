// #include <ostream>
#include <vector>
#include <cassert>
#include "Parity.h"

// using namespace std;
using Index = std::pair<size_t, size_t>;   // block index, bit index

static const Index AFFINE_VALUE_INDEX = {0, 0};   // it's LSB. for MSB would be varNum.

/*.................
    Constructors:
...................*/
// parityStr[0] corresponds to x_1 and parityStr[n-1] corresponds to x_n. Not the most efficient way, but only for testing purpose.
// Precondition: parityStr should not be empty.
Parity::Parity(const std::string& parityStr) : Parity(parityStr.size() - 1) {
    for (size_t i = 0; i < parityStr.size(); i++) {
        if (parityStr[i] == '1') {
            onBitAt(i + 1);
        }
    }
}

Parity Parity::eVec(size_t varNum, size_t pos) {    // e_i, starting from 1 (0 is affVal).
    assert(pos <= varNum);
    
    Parity res = Parity(varNum);
    res.onBitAt(pos);
    return res;
}

/*.................
    Operators:
...................*/
// check if they are the same up to adding some 0 bits in the end. 
// (so they can be considered equal if we had added some path vaiables and now their 0)
bool Parity::operator==(const Parity& rhs) const {
    size_t minBlockNum = std::min(bits.size(), rhs.bits.size());

    for (size_t i = 0; i < minBlockNum; i++) {
        if (bits[i] != rhs.bits[i]) {
            return false;
        }
    }

    const std::vector<uint64_t>& longerBits = (bits.size() > rhs.bits.size()) ? bits : rhs.bits;
    for (size_t i = minBlockNum; i < longerBits.size(); i++) {
        if (longerBits[i] != 0) {
            return false;
        }
    }
    return true;    
}

Parity& Parity::operator+=(const Parity& rhs) {
    size_t minBlockNum = std::min(bits.size(), rhs.bits.size());
    varNum = std::max(varNum, rhs.varNum);

    for (size_t i = 0; i < minBlockNum; i++) {
        bits[i] ^= rhs.bits[i];
    }

    if (bits.size() < rhs.bits.size()) {
        bits.insert(bits.end(), rhs.bits.begin() + minBlockNum, rhs.bits.end());
    }
    return *this;
}

Parity Parity::operator+(const Parity& rhs) const {
    Parity res = *this;
    res += rhs;
    return res;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Parity& par) {
    os << par.getLinearPartString() << " " << par.getAffineValue();
    // os << "  (" << par.algebraicView() << ")";
    return os;
}

// TODO: check
size_t std::hash<Parity>::operator()(const Parity& p) const {
    // FNV-1a constants for 64-bit systems
    size_t res = 0xcbf29ce484222325; 
    
    // Loop directly through the raw uint64_t values
    for (uint64_t block : p.getBits()) {
        res ^= block;
        res *= 0x100000001b3; // Multiply by the FNV prime
    }
    return res;
}

/*.................
    Getters:
...................*/
// It's not efficient, but is only for testing right now.
std::vector<uint64_t> Parity::getLinearPart() const {
    Parity linearPart = *this;
    linearPart.setAffineValue(0);
    return linearPart.getBits();
}

// very inefficient. only for testing now.
std::string Parity::getLinearPartString() const {
    std::string res;
    res.reserve(varNum);

    for (size_t i = 1; i <= varNum; i++) {
        res.push_back(getBitAt(i) ? '1' : '0');
    }
    return res;
}

bool Parity::getBitAt(size_t pos) const {
    assert(pos <= varNum);
    return getBitAtBlock(getIndex(pos));
}

bool Parity::getAffineValue() const {
    return getBitAtBlock(AFFINE_VALUE_INDEX);
}

/*.................
    Setters:
...................*/
void Parity::setBitAt(size_t pos, bool value) {
    assert(pos <= varNum);
    setBitAtBlock(getIndex(pos), value);
}

void Parity::setAffineValue(bool value) {
    setBitAtBlock(AFFINE_VALUE_INDEX, value);
}

void Parity::onBitAt(size_t pos) {
    assert(pos <= varNum);
    onBitAtBlock(getIndex(pos));
}

void Parity::offBitAt(size_t pos) {
    assert(pos > varNum);
    offBitAtBlock(getIndex(pos));
}

void Parity::flipBitAt(size_t pos) {
    assert(pos > varNum);
    flipBitAtBlock(getIndex(pos));
}

void Parity::flipAffineValue() {
    flipBitAtBlock(AFFINE_VALUE_INDEX);
}

void Parity::extendBitsAtWith(size_t pos, bool value) {
    assert(pos > varNum);
    extendBitsTo(pos);
    setBitAt(pos, value);
}

/*.................
    Checks:
...................*/
bool Parity::isIdenticalWith(const Parity& rhs) const {
    return varNum == rhs.varNum && bits == rhs.bits;
}

bool Parity::isUnsat() const {
    assert(bits.size() > 0);
    
    size_t affineBlockInd = AFFINE_VALUE_INDEX.first;
    for (size_t i = 0; i < bits.size(); i++) {
        if (i == affineBlockInd) {
            if (bits[i] != 1) { // affine value is LSB
                return false;
            }
        }
        else {
            if (bits[i] != 0) {
                return false;
            }
        }
    }
    return true;
    
    // return bits.size() > 0 && getAffineValue() && isLinearZero();
}   // currently unsat is 0..01, but we can change it to empty bits.

bool Parity::isLinearZero() const {
    assert(bits.size() > 0);
    
    size_t affineBlockInd = AFFINE_VALUE_INDEX.first;
    for (size_t i = 0; i < bits.size(); i++) {
        if (i == affineBlockInd) {
            if (bits[i] > 1) { // affine value is LSB
                return false;
            }
        }
        else {
            if (bits[i] != 0) {
                return false;
            }
        }
    }
    return true;
}

/*.................
    Helper Methods: 
...................*/
bool Parity::getBitAtBlock(Index ind) const {
    auto [blockInd, bitInd] = ind;
    assert(blockInd < bits.size());
    
    return bits[blockInd] & (1ULL << bitInd);
}

void Parity::setBitAtBlock(Index ind, bool value) {
    auto [blockInd, bitInd] = ind;
    assert(blockInd < bits.size());

    uint64_t mask = 1ULL << bitInd;
    bits[blockInd] = (bits[blockInd] & ~mask) | (static_cast<uint64_t>(value) << bitInd);
}

void Parity::onBitAtBlock(Index ind) {
    auto [blockInd, bitInd] = ind;
    assert(blockInd < bits.size());

    bits[blockInd] |= (1ULL << bitInd);
}

void Parity::offBitAtBlock(Index ind) {
    auto [blockInd, bitInd] = ind;
    assert(blockInd < bits.size());

    bits[blockInd] &= ~(1ULL << bitInd);
}

void Parity::flipBitAtBlock(Index ind) {
    auto [blockInd, bitInd] = ind;
    assert(blockInd < bits.size());
    
    bits[blockInd] ^= (1ULL << bitInd);
}

void Parity::extendBitsTo(size_t newVarNum) {
    varNum = newVarNum;
    size_t requiredBlocks = requiredBlockNum();
    if (bits.size() < requiredBlocks) {
        bits.resize(requiredBlocks, 0);
    }
}

std::string Parity::algebraicView(size_t qubitNum) const {
    std::string res = "";
    size_t n = 0;
    for (size_t i = 1; i <= varNum; i++) {
        if (getBitAt(i)) {
            if (n > 0) {
                res += " + ";
            }
            if (i <= qubitNum) {
                res += ("x" + std::to_string(i));
            } else {
                res += ("y" + std::to_string(i - qubitNum));
            }
            n++;
        }
    }
    bool c = getAffineValue();
    if (c) {
        res += (n > 0) ? " + 1" : "1";
    }
    return res;
}

// TODO: other isomorphism that ignores affine values.