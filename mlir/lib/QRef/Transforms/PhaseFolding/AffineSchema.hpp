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

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "Parity.hpp"

namespace catalyst::phase_folding {

template <typename Derived> struct OrderedSchema {
    auto getOrder() const { return static_cast<const Derived &>(*this).orderImpl(); }
};

template <typename Derived> struct ProjectableSchema {
    template <typename ColOrderRange> auto getProjOrder(ColOrderRange projRange) const {
        return static_cast<const Derived &>(*this).projOrderImpl(projRange);
    }
};

struct AffineSchema : OrderedSchema<AffineSchema> {
    IdxList preVars;
    IdxList auxVars;
    BitLocation affVal = BitLocation::AFFINE_VALUE;

    // Constructors
    AffineSchema() = default;
    virtual ~AffineSchema() = default;

    // Operators
    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const AffineSchema &schema);

    // Getters
    size_t maxBlock() const;
    BitLocation getMaxLoc() const;
    IdxView getRecycledLocs() const;
    IdxList takeRecycledLocs();

    // Stats
    size_t numQubits() const;
    size_t numAuxVars() const;
    virtual size_t numVars() const;
    virtual size_t numCols() const;

    // Methods
    BitLocation allocPreVar();
    BitLocation allocAuxVar();
    IdxView allocPreVars(size_t num);
    IdxView allocAuxVars(size_t num);
    void growAuxVars(size_t num);

    template <typename ColOrderRange> void recycleLocs(ColOrderRange locs) const;

    auto orderImpl() const;

  protected:
    mutable IdxList recycledLocs;
    mutable BitLocation maxLoc = BitLocation::AFFINE_VALUE;
    ;

    AffineSchema(IdxList preVars, IdxList auxVars, BitLocation affVal, IdxList recycledLocs,
                 BitLocation maxLoc)
        : preVars(std::move(preVars)), auxVars(std::move(auxVars)), affVal(affVal),
          recycledLocs(std::move(recycledLocs)), maxLoc(maxLoc) {}

    void growPreVars(size_t num);
    [[nodiscard]] BitLocation getFreeLoc() const;
    [[nodiscard]] IdxList getFreeLocs(size_t n) const;
};

struct TransformSchema : AffineSchema {
    // Constructors
    TransformSchema() : AffineSchema() {}
    explicit TransformSchema(size_t numQubits, std::optional<size_t> numAuxVars = 0)
        : AffineSchema() {
        preVars = getFreeLocs(numQubits);
        if (numAuxVars.has_value()) {
            auxVars = getFreeLocs(numAuxVars.value());
        }
    }
    explicit TransformSchema(IdxList preVars, IdxList auxVars, BitLocation affVal,
                             IdxList recycledLocs, BitLocation maxLoc)
        : AffineSchema(std::move(preVars), std::move(auxVars), affVal, std::move(recycledLocs),
                       maxLoc) {}
};

struct RelationSchema : AffineSchema,
                        OrderedSchema<RelationSchema>,
                        ProjectableSchema<RelationSchema> {
    IdxList postVars;

    // Constructors
    RelationSchema() : AffineSchema() {}
    explicit RelationSchema(size_t numQubits, std::optional<size_t> numAuxVars = 0)
        : AffineSchema() {
        postVars = getFreeLocs(numQubits);
        preVars = getFreeLocs(numQubits);
        if (numAuxVars.has_value()) {
            auxVars = getFreeLocs(numAuxVars.value());
        }
    }
    explicit RelationSchema(TransformSchema &&transSchm) : AffineSchema(std::move(transSchm)) {
        postVars = getFreeLocs(transSchm.numQubits());
    }

    // Operators
    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const RelationSchema &relSchm);

    // Stats
    size_t numVars() const override;

    // Methods
    BitLocation allocPostVar();
    IdxView allocPostVars(size_t num);
    TransformSchema toTransformSchema();

    using OrderedSchema<RelationSchema>::getOrder;
    auto orderImpl() const;

    using ProjectableSchema<RelationSchema>::getProjOrder;
    template <typename ColOrderRange> auto projOrderImpl(ColOrderRange projRange) const;

  protected:
    void growPostVars(size_t num);

    RelationSchema(IdxList postVars, IdxList preVars, IdxList auxVars, BitLocation affVal,
                   IdxList recycledLocs, BitLocation maxLoc)
        : AffineSchema(std::move(preVars), std::move(auxVars), affVal, std::move(recycledLocs),
                       maxLoc),
          postVars(std::move(postVars)) {}
};

struct RelationSchemaView {
    const IdxView postVars;
    const IdxView preVars;
    const IdxView auxVars;
    const BitLocation affVal;

    const size_t maxBlock;

    RelationSchemaView(const IdxView postVars, const IdxView preVars, const IdxView auxVars,
                       const BitLocation affVal, const size_t maxBlock)
        : postVars(postVars), preVars(preVars), auxVars(auxVars), affVal(affVal),
          maxBlock(maxBlock) {}
    explicit RelationSchemaView(const RelationSchema &relSchm)
        : postVars(relSchm.postVars), preVars(relSchm.preVars), auxVars(relSchm.auxVars),
          affVal(relSchm.affVal), maxBlock(relSchm.maxBlock()) {}
};

template <typename Derived> struct TempSchema : RelationSchema, ProjectableSchema<Derived> {
    using ProjectableSchema<Derived>::getProjOrder;

    template <typename ColOrderRange> auto projOrderImpl(ColOrderRange projRange) const;

  protected:
    TempSchema(IdxList postVars, IdxList preVars, IdxList auxVars, BitLocation affVal,
               IdxList recycledLocs, BitLocation maxLoc)
        : RelationSchema(std::move(postVars), std::move(preVars), std::move(auxVars), affVal,
                         std::move(recycledLocs), maxLoc) {}
};

// Precondition: auxVars have been projected out before meet.
struct MeetSchema : TempSchema<MeetSchema> {
    MeetSchema(RelationSchema &&lhs, const RelationSchema &rhs)
        : TempSchema<MeetSchema>(std::move(lhs.postVars), std::move(lhs.preVars), {}, lhs.affVal,
                                 lhs.takeRecycledLocs(), lhs.getMaxLoc()) {}
};

// Precondition: auxVars have been projected out before join.
struct JoinSchema : TempSchema<JoinSchema>, OrderedSchema<JoinSchema> {
    IdxList lPostVars;
    IdxList lPreVars;
    BitLocation lAffVal;

    JoinSchema(RelationSchema &&lhs, const RelationSchema &rhs)
        : TempSchema<JoinSchema>(std::move(lhs.postVars), std::move(lhs.preVars), {}, lhs.affVal,
                                 lhs.takeRecycledLocs(), lhs.getMaxLoc()),
          lPostVars(getFreeLocs(numQubits())), lPreVars(getFreeLocs(numQubits())),
          lAffVal(getFreeLoc()) {}

    // Operators
    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const JoinSchema &joinSchm);

    // Stats
    size_t numVars() const override;
    size_t numCols() const override;

    // Methods
    auto getProjRange() const;

    using OrderedSchema<JoinSchema>::getOrder;
    auto orderImpl() const;
};

struct CompositionSchema : TempSchema<CompositionSchema>, OrderedSchema<CompositionSchema> {
    IdxList projVars;

    CompositionSchema(RelationSchema &&lhs, const RelationSchema &rhs)
        : TempSchema<CompositionSchema>({}, std::move(lhs.preVars), std::move(lhs.auxVars),
                                        lhs.affVal, lhs.takeRecycledLocs(), lhs.getMaxLoc()),
          projVars(std::move(lhs.postVars)) {
        postVars = getFreeLocs(rhs.numQubits());
        growAuxVars(rhs.numAuxVars());
    }

    // Operators
    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const CompositionSchema &cmpSchm);

    // Stats
    size_t numVars() const override;

    using OrderedSchema<CompositionSchema>::getOrder;
    auto orderImpl() const;
};

struct PropagateSchema : CompositionSchema {
    IdxView concretizerVars;

    PropagateSchema(RelationSchema &&lhs, const RelationSchema &rhs)
        : CompositionSchema(std::move(lhs), rhs), concretizerVars(allocAuxVars(numQubits())) {}
};

// Getters:
inline IdxView AffineSchema::getRecycledLocs() const { return recycledLocs; }

inline IdxList AffineSchema::takeRecycledLocs() { return std::move(recycledLocs); }

inline BitLocation AffineSchema::getMaxLoc() const { return maxLoc; }

inline size_t AffineSchema::maxBlock() const { return maxLoc.block; }

inline auto AffineSchema::orderImpl() const {
    return llvm::concat<const BitLocation>(llvm::ArrayRef(preVars), llvm::ArrayRef(auxVars),
                                           llvm::ArrayRef(affVal));
}

inline auto RelationSchema::orderImpl() const {
    return llvm::concat<const BitLocation>(llvm::ArrayRef(postVars), llvm::ArrayRef(preVars),
                                           llvm::ArrayRef(auxVars), llvm::ArrayRef(affVal));
}

inline auto JoinSchema::orderImpl() const {
    return llvm::concat<const BitLocation>(llvm::ArrayRef(lPostVars), llvm::ArrayRef(lPreVars),
                                           llvm::ArrayRef(lAffVal), llvm::ArrayRef(postVars),
                                           llvm::ArrayRef(preVars), llvm::ArrayRef(affVal));
}

inline auto CompositionSchema::orderImpl() const {
    return llvm::concat<const BitLocation>(llvm::ArrayRef(projVars), llvm::ArrayRef(postVars),
                                           llvm::ArrayRef(preVars), llvm::ArrayRef(auxVars),
                                           llvm::ArrayRef(affVal));
}

// Precondition: projRange is auxVars. could be extended to arbitrary range, but not necessary for
// now.
template <typename ColOrderRange>
inline auto RelationSchema::projOrderImpl(ColOrderRange projRange) const {
    return llvm::concat<const BitLocation>(llvm::ArrayRef(auxVars), llvm::ArrayRef(postVars),
                                           llvm::ArrayRef(preVars), llvm::ArrayRef(affVal));
}

template <typename Derived>
template <typename ColOrderRange>
inline auto TempSchema<Derived>::projOrderImpl(ColOrderRange projRange) const {
    return static_cast<const Derived &>(*this).getOrder();
}

inline auto JoinSchema::getProjRange() const {
    return llvm::concat<const BitLocation>(llvm::ArrayRef(lPostVars), llvm::ArrayRef(lPreVars),
                                           llvm::ArrayRef(lAffVal));
}

// Stats:
inline size_t AffineSchema::numQubits() const { return preVars.size(); }

inline size_t AffineSchema::numAuxVars() const { return auxVars.size(); }

inline size_t AffineSchema::numVars() const { return preVars.size() + auxVars.size(); }

inline size_t RelationSchema::numVars() const { return AffineSchema::numVars() + postVars.size(); }

inline size_t JoinSchema::numVars() const { return RelationSchema::numVars() * 2; }

inline size_t CompositionSchema::numVars() const {
    return RelationSchema::numVars() + projVars.size();
}

inline size_t AffineSchema::numCols() const { return numVars() + 1; }

inline size_t JoinSchema::numCols() const { return numVars() + 2; }

// Methods
inline BitLocation AffineSchema::allocPreVar() {
    BitLocation idx = getFreeLoc();
    preVars.push_back(idx);
    return idx;
}

inline IdxView AffineSchema::allocPreVars(size_t n) {
    growPreVars(n);
    return IdxView(preVars).take_back(n);
}

inline void AffineSchema::growPreVars(size_t n) {
    IdxList ids = getFreeLocs(n);
    preVars.reserve(preVars.size() + n);
    preVars.insert(preVars.end(), ids.begin(), ids.end());
}

inline BitLocation AffineSchema::allocAuxVar() {
    BitLocation idx = getFreeLoc();
    auxVars.push_back(idx);
    return idx;
}

inline IdxView AffineSchema::allocAuxVars(size_t n) {
    growAuxVars(n);
    return IdxView(auxVars).take_back(n);
}

inline void AffineSchema::growAuxVars(size_t n) {
    IdxList ids = getFreeLocs(n);
    auxVars.reserve(auxVars.size() + n);
    auxVars.insert(auxVars.end(), ids.begin(), ids.end());
}

inline BitLocation RelationSchema::allocPostVar() {
    BitLocation idx = getFreeLoc();
    postVars.push_back(idx);
    return idx;
}

inline IdxView RelationSchema::allocPostVars(size_t n) {
    growPostVars(n);
    return IdxView(postVars).take_back(n);
}

inline void RelationSchema::growPostVars(size_t n) {
    IdxList ids = getFreeLocs(n);
    postVars.reserve(postVars.size() + n);
    postVars.insert(postVars.end(), ids.begin(), ids.end());
}

inline TransformSchema RelationSchema::toTransformSchema() {
    return TransformSchema(std::move(preVars), std::move(auxVars), affVal, takeRecycledLocs(),
                           getMaxLoc());
}

template <typename ColOrderRange> void AffineSchema::recycleLocs(ColOrderRange locs) const {
    if (llvm::adl_begin(locs) != llvm::adl_end(locs)) {
        recycledLocs.reserve(recycledLocs.size() + llvm::range_size(locs));
        llvm::append_range(recycledLocs, locs);
    }
} // pass size

} // namespace catalyst::phase_folding
