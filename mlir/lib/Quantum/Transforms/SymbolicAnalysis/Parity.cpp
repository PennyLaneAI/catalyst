#include "Parity.h"

#include <cassert>

#include "llvm/ADT/Hashing.h"

using Index = std::pair<size_t, size_t>; // block index, bit index

static const Index AFFINE_VALUE_INDEX = {0, 0}; // it's LSB. for MSB would be varNum.

/*
    Constructors:
*/
// parityStr[0] corresponds to x_1 and parityStr[n-1] corresponds to x_n. Not the most efficient
// way, but only for testing purpose. Precondition: parityStr should not be empty.
Parity::Parity(const std::string &parityStr) : Parity(parityStr.size() - 1)
{
    for (size_t i = 0; i < parityStr.size(); i++) {
        if (parityStr[i] == '1') {
            setBitAt(i + 1);
        }
    }
}

Parity Parity::eVec(size_t varNum, size_t pos)
{
    assert(pos <= varNum);

    Parity res = Parity(varNum);
    res.setBitAt(pos);
    return res;
}

/*
    Operators:
*/
bool Parity::operator==(const Parity &rhs) const
{
    if (state != rhs.state) {
        return false;
    }
    if (state != State::Valid) {
        return true;
    }
    return isEquivalentWithFromBlock(rhs, 0);
}

Parity &Parity::operator+=(const Parity &rhs)
{
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

Parity Parity::operator+(const Parity &rhs) const
{
    Parity res = *this;
    res += rhs;
    return res;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Parity &par)
{
    os << par.getLinearPartString() << " | " << par.getAffineValue();
    // os << "  (" << par.algebraicView() << ")";
    return os;
}

/*
    Getters:
*/
std::string Parity::getLinearPartString() const
{
    std::string res;
    res.reserve(varNum);

    for (size_t i = 1; i <= varNum; i++) {
        res.push_back(getBitAt(i) ? '1' : '0');
    }
    return res;
}

bool Parity::getBitAt(size_t pos) const
{
    assert(pos <= varNum);
    return getBitAtBlock(getIndex(pos));
}

bool Parity::getAffineValue() const { return getBitAtBlock(AFFINE_VALUE_INDEX); }

/*
    Setters:
*/
void Parity::reset() { bits.assign(1, 0); }

void Parity::assignBitAt(size_t pos, bool value)
{
    assert(pos <= varNum);
    assignBitAtBlock(getIndex(pos), value);
}

void Parity::assignAffineValue(bool value) { assignBitAtBlock(AFFINE_VALUE_INDEX, value); }

void Parity::setBitAt(size_t pos)
{
    assert(pos <= varNum);
    setBitAtBlock(getIndex(pos));
}

void Parity::clearAffineValue() { clearBitAtBlock(AFFINE_VALUE_INDEX); }

void Parity::flipAffineValue() { flipBitAtBlock(AFFINE_VALUE_INDEX); }

void Parity::extendBitsAtWith(size_t pos, bool value)
{
    assert(pos > varNum);
    extendBitsTo(pos);
    assignBitAt(pos, value);
}

/*
    Checks:
*/
bool Parity::isLinearEquivalentWith(const Parity &rhs) const
{
    uint64_t blockL = bits.empty() ? 0 : bits[0];
    uint64_t blockR = rhs.bits.empty() ? 0 : rhs.bits[0];

    return ((blockL ^ blockR) > 1) ? false : isEquivalentWithFromBlock(rhs, 1);
}

bool Parity::isTrivial() const { return isTrivialFromBlock(0); }

bool Parity::isUnsat() const
{
    assert(bits.size() > 0);
    size_t affineBlockInd = AFFINE_VALUE_INDEX.first;
    return (bits[affineBlockInd] != 1) ? false : isTrivialFromBlock(1); // affine value is LSB
} // currently unsat is 0..01, but we can change it to empty bits.

bool Parity::isLinearZero() const
{
    assert(bits.size() > 0);
    size_t affineBlockInd = AFFINE_VALUE_INDEX.first;
    return (bits[affineBlockInd] > 1) ? false : isTrivialFromBlock(1); // affine value is LSB
}

/*
    Helper Methods:
*/
bool Parity::isTrivialFromBlock(size_t fstBlock) const
{
    for (size_t i = fstBlock; i < bits.size(); i++) {
        if (bits[i] != 0) {
            return false;
        }
    }
    return true;
}

// check if they are the same up to adding some 0 bits in the end.
// (so they can be considered equal if we had added some path or new vaiables and now their 0)
bool Parity::isEquivalentWithFromBlock(const Parity &rhs, size_t fstBlock) const
{
    size_t minBlockNum = std::min(bits.size(), rhs.bits.size());

    for (size_t i = fstBlock; i < minBlockNum; i++) {
        if (bits[i] != rhs.bits[i]) {
            return false;
        }
    }

    const llvm::SmallVector<uint64_t, 8> &longerBits =
        (bits.size() > rhs.bits.size()) ? bits : rhs.bits;
    for (size_t i = minBlockNum; i < longerBits.size(); i++) {
        if (longerBits[i] != 0) {
            return false;
        }
    }
    return true;
}

bool Parity::getBitAtBlock(Index ind) const
{
    auto [blockInd, bitInd] = ind;
    assert(blockInd < bits.size());

    return bits[blockInd] & (1ULL << bitInd);
}

void Parity::assignBitAtBlock(Index ind, bool value)
{
    auto [blockInd, bitInd] = ind;
    assert(blockInd < bits.size());

    uint64_t mask = 1ULL << bitInd;
    bits[blockInd] = (bits[blockInd] & ~mask) | (static_cast<uint64_t>(value) << bitInd);
}

void Parity::setBitAtBlock(Index ind)
{
    auto [blockInd, bitInd] = ind;
    assert(blockInd < bits.size());

    bits[blockInd] |= (1ULL << bitInd);
}

void Parity::clearBitAtBlock(Index ind)
{
    auto [blockInd, bitInd] = ind;
    assert(blockInd < bits.size());

    bits[blockInd] &= ~(1ULL << bitInd);
}

void Parity::flipBitAtBlock(Index ind)
{
    auto [blockInd, bitInd] = ind;
    assert(blockInd < bits.size());

    bits[blockInd] ^= (1ULL << bitInd);
}

void Parity::extendBitsTo(size_t newVarNum)
{
    varNum = newVarNum;
    size_t requiredBlocks = requiredBlockNum();
    if (bits.size() < requiredBlocks) {
        bits.resize(requiredBlocks, 0);
    }
}

std::string Parity::algebraicView(size_t qubitNum) const
{ // wrong
    std::string res = "";
    size_t n = 0;
    for (size_t i = 1; i <= varNum; i++) {
        if (getBitAt(i)) {
            if (n > 0) {
                res += " + ";
            }
            if (i <= qubitNum) {
                res += ("x" + std::to_string(i));
            }
            else {
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
