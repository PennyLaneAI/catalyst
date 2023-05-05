// Copyright 2023 Xanadu Quantum Technologies Inc.

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

#include <Exception.hpp>

/**
 * A multi-dimensional view for MemRef-like and std::vector<T> types.
 *
 * @tparam T The underlying data type
 * @tparam R The Rank (R > 0)
 *
 * @note A forward iterator is implemented in this view for traversing over the entire
 * elements of MemRef types rank-by-rank starting from the last dimension (R-1). For example,
 * The DataView iterator for MemRef<T, 2> starts from index (0, 0) and traverses elements
 * in the following order:
 * (0, 0), ..., (0, sizes[1]-1), (1, 0), ..., (1, sizes[1]-1), ... (sizes[0]-1, sizes[1]-1).
 */
template <typename T, size_t R> class DataView {
  private:
    T *data_aligned;
    size_t offset;
    size_t sizes[R] = {0};
    size_t strides[R] = {0};

  public:
    class iterator {
      private:
        const DataView<T, R> &view;

        int64_t loc; // physical index
        size_t indices[R] = {0};

      public:
        using iterator_category = std::forward_iterator_tag; // LCOV_EXCL_LINE
        using value_type = T;                                // LCOV_EXCL_LINE
        using difference_type = std::ptrdiff_t;              // LCOV_EXCL_LINE
        using pointer = T *;                                 // LCOV_EXCL_LINE
        using reference = T &;                               // LCOV_EXCL_LINE

        iterator(const DataView<T, R> &_view, int64_t begin_idx) : view(_view), loc(begin_idx) {}
        pointer operator->() const { return &view.data_aligned[loc]; }
        reference operator*() const { return view.data_aligned[loc]; }
        iterator &operator++()
        {
            int64_t next_axis = -1;
            int64_t idx;
            for (int64_t i = R; i > 0; --i) {
                idx = i - 1;
                if (indices[idx]++ < view.sizes[idx] - 1) {
                    next_axis = idx;
                    break;
                }
                indices[idx] = 0;
                loc -= (view.sizes[idx] - 1) * view.strides[idx];
            }

            loc = next_axis == -1 ? -1 : loc + view.strides[next_axis];
            return *this;
        }
        iterator operator++(int)
        {
            auto tmp = *this;
            int64_t next_axis = -1;
            int64_t idx;
            for (int64_t i = R; i > 0; --i) {
                idx = i - 1;
                if (indices[idx]++ < view.sizes[idx] - 1) {
                    next_axis = idx;
                    break;
                }
                indices[idx] = 0;
                loc -= (view.sizes[idx] - 1) * view.strides[idx];
            }

            loc = next_axis == -1 ? -1 : loc + view.strides[next_axis];
            return tmp;
        }
        bool operator==(const iterator &other) const
        {
            return (loc == other.loc && view.data_aligned == other.view.data_aligned);
        }
        bool operator!=(const iterator &other) const { return !(*this == other); }
    };

    explicit DataView(std::vector<T> &buffer) : data_aligned(buffer.data()), offset(0)
    {
        static_assert(R == 1, "[Class: DataView] Assertion: R == 1");
        sizes[0] = buffer.size();
        strides[0] = 1;
    }

    explicit DataView(T *_data_aligned, size_t _offset, size_t *_sizes, size_t *_strides)
        : data_aligned(_data_aligned), offset(_offset)
    {
        static_assert(R > 0, "[Class: DataView] Assertion: R > 0");
        if (_sizes && _strides) {
            for (size_t i = 0; i < R; i++) {
                sizes[i] = _sizes[i];
                strides[i] = _strides[i];
            }
        } // else sizes = {0}, strides = {0}
    }

    [[nodiscard]] auto size() const -> size_t
    {
        if (!data_aligned) {
            return 0;
        }

        size_t tsize = 1;
        for (size_t i = 0; i < R; i++) {
            tsize *= sizes[i];
        }
        return tsize;
    }

    template <typename... I> T &operator()(I... idxs) const
    {
        static_assert(sizeof...(idxs) == R,
                      "[Class: DataView] Error in Catalyst Runtime: Wrong number of indices");
        size_t indices[] = {static_cast<size_t>(idxs)...};

        size_t loc = offset;
        for (size_t axis = 0; axis < R; axis++) {
            RT_ASSERT(indices[axis] < sizes[axis]);
            loc += indices[axis] * strides[axis];
        }
        return data_aligned[loc];
    }

    iterator begin() { return iterator{*this, static_cast<int64_t>(offset)}; }

    iterator end() { return iterator{*this, -1}; }
};
