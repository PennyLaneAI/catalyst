// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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

#include <cstring>

#include <Exception.hpp>

extern "C" {
void *_mlir_memref_to_llvm_alloc(size_t size);
void *_mlir_memref_to_llvm_aligned_alloc(size_t alignment, size_t size);
void _mlir_memref_to_llvm_free(void *ptr);
}

template <typename T, size_t R> struct MemRefT {
    T *data_allocated;
    T *data_aligned;
    size_t offset;
    size_t sizes[R];
    size_t strides[R];
};

/**
 * A multi-dimensional view for MemRef<T, R> types.
 *
 * @tparam T The underlying data type
 * @tparam R The Rank (R >= 0)
 *
 * @note A forward iterator is implemented in this view for traversing over the entire
 * elements of MemRef types rank-by-rank starting from the last dimension (R-1). For example,
 * The MemRefView iterator for MemRef<T, 2> starts from index (0, 0) and traverses elements
 * in the following order:
 * (0, 0), ..., (0, sizes[1]-1), (1, 0), ..., (1, sizes[1]-1), ... (sizes[0]-1, sizes[1]-1).
 */
template <typename T, size_t R> class MemRefView {
  private:
    const MemRefT<T, R> *buffer;
    size_t tsize; // total size

  public:
    class MemRefIter {
      private:
        const MemRefT<T, R> *buffer;
        int64_t loc; // physical index
        size_t indices[R] = {0};

      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T *;
        using reference = T &;

        MemRefIter(const MemRefT<T, R> *_buffer, int64_t begin_idx)
            : buffer(_buffer), loc(begin_idx)
        {
        }
        pointer operator->() const { return &buffer->data_aligned[loc]; }
        reference operator*() const { return buffer->data_aligned[loc]; }
        MemRefIter &operator++()
        {
            int64_t next_axis = -1;
            for (int64_t i = R - 1; i >= 0; --i) {
                if (indices[i]++ < buffer->sizes[i] - 1) {
                    next_axis = i;
                    break;
                }
                if (!i) {
                    break;
                }
                indices[i] = 0;
                loc -= (buffer->sizes[i] - 1) * buffer->strides[i];
            }

            loc = next_axis == -1 ? -1 : loc + buffer->strides[next_axis];
            return *this;
        }
        MemRefIter operator++(int)
        {
            auto tmp = *this;
            int64_t next_axis = -1;
            for (int64_t i = R - 1; i >= 0; --i) {
                if (indices[i]++ < buffer->sizes[i] - 1) {
                    next_axis = i;
                    break;
                }
                if (!i) {
                    break;
                }
                indices[i] = 0;
                loc -= (buffer->sizes[i] - 1) * buffer->strides[i];
            }

            loc = next_axis == -1 ? -1 : loc + buffer->strides[next_axis];
            return tmp;
        }
        bool operator==(const MemRefIter &other) const
        {
            return (loc == other.loc && buffer == other.buffer);
        }
        bool operator!=(const MemRefIter &other) const { return !(*this == other); }
    };

    explicit MemRefView(const MemRefT<T, R> *_buffer, size_t _size) : buffer(_buffer), tsize(_size)
    {
        RT_FAIL_IF(!buffer, "[Class: MemRefView] Error in Catalyst Runtime: Cannot create a view "
                            "for uninitialized MemRefT<T, R>");
    }

    [[nodiscard]] auto get() const -> const MemRefT<T, R> & { return *buffer; }

    [[nodiscard]] auto empty() const -> bool { return tsize == 0; }

    [[nodiscard]] auto size() const -> size_t { return tsize; }

    template <typename... I> T &operator()(I... idxs) const
    {
        static_assert(sizeof...(idxs) == R,
                      "[Class: MemRefView] Error in Catalyst Runtime: Wrong number of indices");
        size_t indices[] = {static_cast<size_t>(idxs)...};

        if (R == 0) {
            // 0-rank memref
            return buffer->data_aligned[buffer->offset];
        }

        size_t loc = buffer->offset;
        for (size_t axis = 0; axis < R; axis++) {
            RT_ASSERT(indices[axis] < buffer->sizes[axis]);
            loc += indices[axis] * buffer->strides[axis];
        }
        return buffer->data_aligned[loc];
    }

    MemRefIter begin() { return MemRefIter{buffer, static_cast<int64_t>(buffer->offset)}; }

    MemRefIter end() { return MemRefIter{buffer, -1}; }
};
