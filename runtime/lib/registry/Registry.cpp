// Copyright 2024 Xanadu Quantum Technologies Inc.
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

#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <string>
#include <unordered_map>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

// From PyBind11's documentation:
//
//     Do you have any global variables that are pybind11 objects or invoke pybind11 functions in
//     either their constructor or destructor? You are generally not allowed to invoke any Python
//     function in a global static context. We recommend using lazy initialization and then
//     intentionally leaking at the end of the program.
//
// https://pybind11.readthedocs.io/en/stable/advanced/misc.html#common-sources-of-global-interpreter-lock-errors
std::unordered_map<int64_t, nb::callable> *references;

std::string libmlirpath;

struct UnrankedMemrefType {
    int64_t rank;
    void *descriptor;
};

class LibraryManager {
    void *_handle;

  public:
    LibraryManager(std::string path)
    {
        this->_handle = dlopen(path.c_str(), RTLD_LAZY);
        if (!this->_handle) {
            throw nb::value_error(dlerror());
        }
    }

    ~LibraryManager()
    {
        if (this->_handle) {
            dlclose(this->_handle);
        }
    }

    void operator()(long elementSize, UnrankedMemrefType *src, UnrankedMemrefType *dst)
    {
        void *f_ptr = dlsym(this->_handle, "memrefCopy");
        if (!f_ptr) {
            throw nb::value_error(dlerror());
        }
        typedef void (*memrefCopy_t)(int64_t, void *, void *);
        void (*memrefCopy)(int64_t, void *, void *);
        memrefCopy = (memrefCopy_t)(f_ptr);
        return memrefCopy(elementSize, src, dst);
    }
};

inline const char *ext()
{
#ifdef __APPLE__
    return ".dylib";
#elif __linux__
    return ".so";
#else
#error "Only apple and linux are currently supported";
#endif
}

std::string library_name(std::string name) { return name + ext(); }

void convertResult(nb::handle tuple)
{
    nb::object unrankedMemrefPtrSizeTuple = tuple.attr("__getitem__")(0);

    nb::object unranked_memref = unrankedMemrefPtrSizeTuple.attr("__getitem__")(0);
    nb::object element_size = unrankedMemrefPtrSizeTuple.attr("__getitem__")(1);
    nb::object unranked_memref_ptr_int = unranked_memref.attr("value");

    void *unranked_memref_ptr = reinterpret_cast<void *>(nb::cast<long>(unranked_memref_ptr_int));
    long e_size = nb::cast<long>(element_size);

    nb::object dest = tuple.attr("__getitem__")(1);

    long destAsLong = nb::cast<long>(dest);
    void *destAsPtr = (void *)(destAsLong);

    UnrankedMemrefType *src = (UnrankedMemrefType *)unranked_memref_ptr;
    UnrankedMemrefType destMemref = {src->rank, destAsPtr};

    std::string libpath = libmlirpath + library_name("/libmlir_c_runner_utils");
    LibraryManager memrefCopy(libpath);
    memrefCopy(e_size, src, &destMemref);
}

void convertResults(nb::list results, nb::list allocated)
{
    auto builtins = nb::module_::import_("builtins");
    auto zip = builtins.attr("zip");
    for (nb::handle obj : zip(results, allocated)) {
        convertResult(obj);
    }
}

extern "C" {
[[gnu::visibility("default")]] void callbackCall(int64_t identifier, int64_t count, int64_t retc,
                                                 va_list args)
{
    nb::gil_scoped_acquire lock;
    auto it = references->find(identifier);
    if (it == references->end()) {
        throw std::invalid_argument("Callback called with invalid identifier");
    }
    auto lambda = it->second;

    nb::list flat_args;
    for (int i = 0; i < count; i++) {
        int64_t ptr = va_arg(args, int64_t);
        flat_args.append(ptr);
    }

    nb::list flat_results = nb::list(lambda(flat_args));

    // We have a flat list of return values.
    // These returns **may** be array views to
    // the very same memrefs that we passed as inputs.
    // As a first prototype, let's copy these values.
    // I think it is best to always copy them because
    // of aliasing. Let's just copy them to guarantee
    // no aliasing issues. We can revisit this as an optimization
    // and allowing these to alias.
    nb::list flat_returns_allocated_compiler;
    for (int i = 0; i < retc; i++) {
        int64_t ptr = va_arg(args, int64_t);
        flat_returns_allocated_compiler.append(ptr);
    }
    convertResults(flat_results, flat_returns_allocated_compiler);
}
}

void setMLIRLibPath(std::string path) { libmlirpath = path; }

auto registerImpl(nb::callable f)
{
    // Do we need to see if it is already present or can we just override it? Just override is fine.
    // Does python reuse id's? Yes.
    // But only after they have been garbaged collected.
    // So as long as we maintain a reference to it, then they won't be garbage collected.
    // Inserting the function into the unordered map increases the reference by one.
    int64_t id = reinterpret_cast<int64_t>(f.ptr());
    references->insert({id, f});
    return id;
}

NB_MODULE(catalyst_callback_registry, m)
{
    if (references == nullptr) {
        references = new std::unordered_map<int64_t, nb::callable>();
    }
    m.doc() = "Callbacks";
    m.def("register", &registerImpl, "Call a python function registered in a map.");
    m.def("set_mlir_lib_path", &setMLIRLibPath, "Set location of mlir's libraries.");
}
