// Copyright 2023 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <csignal>
#include <nanobind/nanobind.h>

// TODO: Periodically check and increment version.
// https://endoflife.date/numpy
#define NPY_NO_DEPRECATED_API NPY_1_24_API_VERSION

#include "numpy/ndarrayobject.h"

namespace nb = nanobind;

struct memref_beginning_t {
    char *allocated;
    char *aligned;
    size_t offset;
};

size_t memref_size_based_on_rank(size_t rank)
{
    size_t allocated = sizeof(void *);
    size_t aligned = sizeof(void *);
    size_t offset = sizeof(size_t);
    size_t sizes = rank * sizeof(size_t);
    size_t strides = rank * sizeof(size_t);
    return allocated + aligned + offset + sizes + strides;
}

size_t *to_sizes(char *base, size_t rank)
{
    if (rank == 0) {
        return NULL;
    }

    size_t allocated = sizeof(void *);
    size_t aligned = sizeof(void *);
    size_t offset = sizeof(size_t);
    size_t bytes_offset = allocated + aligned + offset;
    return reinterpret_cast<size_t *>(base + bytes_offset);
}

size_t *to_strides(char *base, size_t rank)
{
    if (rank == 0) {
        return NULL;
    }

    size_t allocated = sizeof(void *);
    size_t aligned = sizeof(void *);
    size_t offset = sizeof(size_t);
    size_t sizes = rank * sizeof(size_t);
    size_t bytes_offset = allocated + aligned + offset + sizes;
    return reinterpret_cast<size_t *>(base + bytes_offset);
}

void free_wrap(PyObject *capsule)
{
    void *obj = PyCapsule_GetPointer(capsule, NULL);
    free(obj);
}

const npy_intp *npy_get_dimensions(char *memref, size_t rank)
{
    size_t *sizes = to_sizes(memref, rank);
    const npy_intp *dims = reinterpret_cast<npy_intp *>(sizes);
    return dims;
}

const npy_intp *npy_get_strides(char *memref, size_t element_size, size_t rank)
{
    size_t *strides = to_strides(memref, rank);
    for (unsigned int idx = 0; idx < rank; idx++) {
        // memref strides are in terms of elements.
        // numpy strides are in terms of bytes.
        // Therefore multiply by element size.
        strides[idx] *= element_size;
    }

    npy_intp *npy_strides = reinterpret_cast<npy_intp *>(strides);
    return npy_strides;
}

nb::list move_returns(void *memref_array_ptr, nb::object result_desc, nb::object transfer,
                      nb::dict numpy_arrays)
{
    nb::list returns;
    if (result_desc.is_none()) {
        return returns;
    }

    auto ctypes = nb::module_::import_("ctypes");
    using f_ptr_t = bool (*)(void *);
    f_ptr_t f_transfer_ptr = *((f_ptr_t *)nb::cast<size_t>(ctypes.attr("addressof")(transfer)));

    /* Data from the result description */
    auto ranks = result_desc.attr("_ranks_");
    auto etypes = result_desc.attr("_etypes_");
    auto sizes = result_desc.attr("_sizes_");

    size_t memref_len = nb::cast<size_t>(ranks.attr("__len__")());
    size_t offset = 0;

    char *memref_array_bytes = reinterpret_cast<char *>(memref_array_ptr);

    for (size_t idx = 0; idx < memref_len; idx++) {
        unsigned int rank_i = nb::cast<unsigned int>(ranks.attr("__getitem__")(idx));
        char *memref_i_beginning = memref_array_bytes + offset;
        offset += memref_size_based_on_rank(rank_i);

        struct memref_beginning_t *memref =
            reinterpret_cast<struct memref_beginning_t *>(memref_i_beginning);
        bool is_in_rt_heap = f_transfer_ptr(memref->allocated);

        if (!is_in_rt_heap) {
            // This case is guaranteed by the compiler to be the following:
            // 1. When an input tensor is sent to as an output
            // 2. When an output tensor is aliased with with another output tensor
            // and one of them has already been transferred.
            //
            // The first case is guaranteed by the use of the flag --cp-global-memref
            //
            // Use the numpy_arrays dictionary which sets up the following map:
            // integer (memory address) -> nb::object (numpy array)
            auto array_object =
                numpy_arrays.attr("__getitem__")(reinterpret_cast<size_t>(memref->allocated));
            returns.append(array_object);
            continue;
        }

        const npy_intp *dims = npy_get_dimensions(memref_i_beginning, rank_i);

        size_t element_size = nb::cast<size_t>(sizes.attr("__getitem__")(idx));
        const npy_intp *strides = npy_get_strides(memref_i_beginning, element_size, rank_i);

        auto etype_i = etypes.attr("__getitem__")(idx);
        PyArray_Descr *descr = PyArray_DescrFromTypeObject(etype_i.ptr());
        if (!descr) {
            throw std::runtime_error("PyArray_Descr failed.");
        }

        void *aligned = memref->aligned;
        PyObject *new_array =
            PyArray_NewFromDescr(&PyArray_Type, descr, rank_i, dims, strides, aligned, 0, NULL);
        if (!new_array) {
            throw std::runtime_error("PyArray_NewFromDescr failed.");
        }

        PyObject *capsule = PyCapsule_New(memref->allocated, NULL,
                                          reinterpret_cast<PyCapsule_Destructor>(&free_wrap));
        if (!capsule) {
            throw std::runtime_error("PyCapsule_New failed.");
        }

        int retval = PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(new_array), capsule);
        bool success = 0 == retval;
        if (!success) {
            throw std::runtime_error("PyArray_SetBaseObject failed.");
        }

        returns.append(nb::borrow(new_array)); // nb::borrow increments ref count by 1

        // Now we insert the array into the dictionary.
        // This dictionary is a map of the type:
        // integer (memory address) -> nb::object (numpy array)
        //
        // Upon first entry into this function, it holds the numpy.arrays
        // sent as an input to the generated function.
        // Upon following entries it is extended with the numpy.arrays
        // which are the output of the generated function.
        PyObject *pyLong = PyLong_FromLong(reinterpret_cast<size_t>(memref->allocated));
        if (!pyLong) {
            throw std::runtime_error("PyLong_FromLong failed.");
        }

        numpy_arrays[pyLong] = nb::borrow(new_array); // nb::borrow increments ref count by 1

        // Decrement reference counts.
        // The final ref count of `new_array` should be 2: one for the `returns` list and one for
        // the `numpy_arrays` dict.
        Py_DecRef(pyLong);
        Py_DecRef(new_array);
    }
    return returns;
}

nb::list wrap(nb::object func, nb::tuple py_args, nb::object result_desc, nb::object transfer,
              nb::dict numpy_arrays)
{
    // Install signal handler to catch user interrupts (e.g. CTRL-C).
    signal(SIGINT, [](int code) { throw std::runtime_error("KeyboardInterrupt (SIGINT)"); });

    nb::list returns;

    size_t length = nb::cast<size_t>(py_args.attr("__len__")());
    if (length != 2) {
        throw std::invalid_argument("Invalid number of arguments.");
    }

    auto ctypes = nb::module_::import_("ctypes");
    using f_ptr_t = void (*)(void *, void *);
    f_ptr_t f_ptr = *reinterpret_cast<f_ptr_t *>(nb::cast<size_t>(ctypes.attr("addressof")(func)));

    auto value0 = py_args.attr("__getitem__")(0);
    void *value0_ptr =
        *reinterpret_cast<void **>(nb::cast<size_t>(ctypes.attr("addressof")(value0)));
    auto value1 = py_args.attr("__getitem__")(1);
    void *value1_ptr =
        *reinterpret_cast<void **>(nb::cast<size_t>(ctypes.attr("addressof")(value1)));

    {
        nb::gil_scoped_release lock;
        f_ptr(value0_ptr, value1_ptr);
    }
    returns = move_returns(value0_ptr, result_desc, transfer, numpy_arrays);

    return returns;
}

NB_MODULE(wrapper, m)
{
    m.doc() = "wrapper module";
    // We have to annotate all the arguments to `wrap` to allow `result_desc` to be None
    // See https://nanobind.readthedocs.io/en/latest/functions.html#none-arguments
    m.def("wrap", &wrap, "A wrapper function.", nb::arg("func"), nb::arg("py_args"),
          nb::arg("result_desc").none(), nb::arg("transfer"), nb::arg("numpy_arrays"));
    int retval = _import_array();
    bool success = retval >= 0;
    if (!success) {
        throw nb::import_error("Could not import numpy array C-API.");
    }
}
