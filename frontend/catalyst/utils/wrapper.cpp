#include <iostream>
#include <pybind11/pybind11.h>

// Emit a compile time warning if we are using numpy
// functions that are not compatible with numpy 1.22
// Why 1.22?
//
// As of time of this writing 1.21.* will be deprecated
// next month. See here: https://endoflife.date/numpy
#define NPY_NO_DEPRECATED_API NPY_1_22_API_VERSION

#include "numpy/arrayobject.h"

namespace py = pybind11;

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
    return (size_t *)(base + bytes_offset);
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
    return (size_t *)(base + bytes_offset);
}

void free_wrap(PyObject *capsule)
{
    void *obj = PyCapsule_GetPointer(capsule, NULL);
    free(obj);
}

const npy_intp *npy_get_dimensions(char *memref, size_t rank)
{
    size_t *sizes = to_sizes(memref, rank);
    const npy_intp *dims = (npy_intp *)sizes;
    return dims;
}

const npy_intp *npy_get_strides(char *memref, size_t element_size, size_t rank)
{
    size_t *strides = to_strides(memref, rank);
    for (unsigned int idx = 0; idx < rank; idx++) {
        // memref strides are in terms of elements.
        // numpy strides are in terms of bytes.
        // Therefore multiply by element size.
        strides[idx] *= (size_t)element_size;
    }

    npy_intp *npy_strides = (npy_intp *)strides;
    return npy_strides;
}

py::list move_returns(void *memref_array_ptr, py::object result_desc, py::object transfer,
                      py::dict numpy_arrays)
{
    py::list returns;
    if (result_desc.is_none()) {
        return returns;
    }

    auto ctypes = py::module::import("ctypes");
    using f_ptr_t = bool (*)(void *);
    f_ptr_t f_transfer_ptr = *((f_ptr_t *)ctypes.attr("addressof")(transfer).cast<size_t>());

    /* Data from the result description */
    auto ranks = result_desc.attr("_ranks_");
    auto etypes = result_desc.attr("_etypes_");
    auto sizes = result_desc.attr("_sizes_");

    size_t memref_len = ranks.attr("__len__")().cast<size_t>();
    size_t offset = 0;

    char *memref_array_bytes = (char *)(memref_array_ptr);

    for (size_t idx = 0; idx < memref_len; idx++) {
        unsigned int rank_i = ranks.attr("__getitem__")(idx).cast<unsigned int>();
        char *memref_i_beginning = memref_array_bytes + offset;
        offset += memref_size_based_on_rank(rank_i);

        struct memref_beginning_t *memref = (struct memref_beginning_t *)memref_i_beginning;
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
            // integer (memory address) -> py::object (numpy array)
            auto array_object = numpy_arrays.attr("__getitem__")((size_t)memref->allocated);
            returns.append(array_object);
            continue;
        }

        const npy_intp *dims = npy_get_dimensions(memref_i_beginning, rank_i);

        size_t element_size = sizes.attr("__getitem__")(idx).cast<size_t>();
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

        PyObject *capsule =
            PyCapsule_New(memref->allocated, NULL, (PyCapsule_Destructor)&free_wrap);
        if (!capsule) {
            throw std::runtime_error("PyCapsule_New failed.");
        }

        int retval = PyArray_SetBaseObject((PyArrayObject *)new_array, capsule);
        bool success = 0 == retval;
        if (!success) {
            throw std::runtime_error("PyArray_SetBaseObject failed.");
        }

        returns.append(new_array);

        // Now we insert the array into the dictionary.
        // This dictionary is a map of the type:
        // integer (memory address) -> py::object (numpy array)
        //
        // Upon first entry into this function, it holds the numpy.arrays
        // sent as an input to the generated function.
        // Upon following entries it is extended with the numpy.arrays
        // which are the output of the generated function.
        PyObject *pyLong = PyLong_FromLong((size_t)memref->allocated);
        if (!pyLong) {
            throw std::runtime_error("PyLong_FromLong failed.");
        }

        numpy_arrays[pyLong] = new_array;

        Py_DECREF(pyLong);
        Py_DECREF(new_array);
    }
    return returns;
}

py::list wrap(py::object func, py::tuple py_args, py::object result_desc, py::object transfer,
              py::dict numpy_arrays)
{
    py::list returns;

    size_t length = py_args.attr("__len__")().cast<size_t>();
    if (length != 2) {
        throw std::invalid_argument("Invalid number of arguments.");
    }

    auto ctypes = py::module::import("ctypes");
    using f_ptr_t = void (*)(void *, void *);
    f_ptr_t f_ptr = *reinterpret_cast<f_ptr_t *>(ctypes.attr("addressof")(func).cast<size_t>());

    auto value0 = py_args.attr("__getitem__")(0);
    void *value0_ptr = *reinterpret_cast<void **>(ctypes.attr("addressof")(value0).cast<size_t>());
    auto value1 = py_args.attr("__getitem__")(1);
    void *value1_ptr = *reinterpret_cast<void **>(ctypes.attr("addressof")(value1).cast<size_t>());

    f_ptr(value0_ptr, value1_ptr);
    returns = move_returns(value0_ptr, result_desc, transfer, numpy_arrays);

    return returns;
}

PYBIND11_MODULE(wrapper, m)
{
    m.doc() = "wrapper module";
    m.def("wrap", &wrap, "A wrapper function.");
    int retval = _import_array();
    bool success = retval >= 0;
    if (!success) {
        throw pybind11::import_error("Couldn't import numpy array C-API.");
    }
}
