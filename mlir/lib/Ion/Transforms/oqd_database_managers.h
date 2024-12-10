#include <dlfcn.h>
#include <iostream>

//#include <nanobind/nanobind.h>
//#include <nanobind/eval.h>
//#include <nanobind/stl/string.h>

using namespace std;

void oqd_hi(){
	std::string path_to_this_file = __FILE__;
    std::cout << "Current working file: " << path_to_this_file << std::endl;
    std::string to_remove = "mlir/lib/Ion/Transforms/./oqd_database_managers.h";

    size_t pos = path_to_this_file.find(to_remove);

    std::string manager_so_location = path_to_this_file.replace(pos, to_remove.length(),
        	"frontend/catalyst/third_party/oqd/oqd_database_managers.so");


    std::cout << ".so location: " << manager_so_location << std::endl;


	void *handle;
    void (*libfunc)();

    handle = dlopen((manager_so_location).c_str(), RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "Error loading library: %s\n", dlerror());
        return;
    }

    libfunc = (void (*)())dlsym(handle, "lib_oqd_hi");

    (*libfunc)();
};
