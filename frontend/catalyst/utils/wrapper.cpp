#include <iostream>
#include <pybind11/pybind11.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NUMPY_WARN_IF_NO_MEM_POLICY 1
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

size_t *to_sizes(char *base)
{
    size_t allocated = sizeof(void *);
    size_t aligned = sizeof(void *);
    size_t offset = sizeof(size_t);
    size_t bytes_offset = allocated + aligned + offset;
    return (size_t *)(base + bytes_offset);
}

size_t *to_strides(char *base, size_t rank)
{
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
    void *allocated = obj;
    free(allocated);
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

    auto ranks = result_desc.attr("_ranks_");
    auto etypes = result_desc.attr("_etypes_");
    size_t memref_len = ranks.attr("__len__")().cast<size_t>();
    size_t offset = 0;

    auto numpy = py::module::import("numpy");
    char *memref_array_bytes = (char *)(memref_array_ptr);

    for (size_t idx = 0; idx < memref_len; idx++) {
        unsigned int rank_i = ranks.attr("__getitem__")(idx).cast<unsigned int>();
        char *memref_i_beginning = memref_array_bytes + offset;
        offset += memref_size_based_on_rank(rank_i);
        struct memref_beginning_t *memref = (struct memref_beginning_t *)memref_i_beginning;
        bool is_in_rt_heap = f_transfer_ptr(memref->allocated);

        if (!is_in_rt_heap) {
            // This case is guaranteed by the compiler to be the following:
            // when an input tensor is sent to as an output as well.
            //
            // It is guaranteed by the use of the flag --cp-global-memref
            //
            // Use the numpy_arrays dictionary which sets up the following map:
            // integer (memory address) -> py::object (numpy array)
            auto array_object = numpy_arrays.attr("__getitem__")((size_t)memref->allocated);
            returns.append(array_object);
            continue;
        }

        // This is the case where the returned tensor is managed by the runtime
        // and we need to construct a new numpy array.
        py::object etype_i = etypes.attr("__getitem__")(idx);
        auto dtype = numpy.attr("dtype")(etype_i);
        size_t element_size = dtype.attr("itemsize").cast<size_t>();

        PyArray_Descr *descr = PyArray_DescrFromTypeObject(etype_i.ptr());
        if (!descr) {
            throw std::runtime_error("PyArray_Descr failed.");
        }

        size_t *sizes = to_sizes(memref_i_beginning);
        const npy_intp *dims = (npy_intp *)sizes;

        size_t *strides = to_strides(memref_i_beginning, rank_i);
        for (unsigned int jdx = 0; jdx < rank_i; jdx++) {
            // memref strides are in terms of elements.
            // numpy strides are in terms of bytes.
            // Therefore multiply by element size.
            strides[jdx] *= (size_t)element_size;
        }

        npy_intp *npy_strides = (npy_intp *)strides;

        void *aligned = memref->aligned;
        PyObject *new_array =
            PyArray_NewFromDescr(&PyArray_Type, descr, rank_i, dims, npy_strides, aligned, 0, NULL);
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
    _import_array();
}
