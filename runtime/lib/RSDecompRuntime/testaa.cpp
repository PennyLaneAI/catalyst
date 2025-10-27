#include <vector>
#include <cstdint> // For int64_t
#include <cstring>
#include <iostream>

// 1D memref descriptor (unchanged)
struct MemRef1D {
    int64_t* allocated; 
    int64_t* aligned;   
    int64_t offset;
    int64_t size;
    int64_t stride;
};

// RuntimeData struct is no longer needed.

extern "C" {

// Function 1: Returns the memref
MemRef1D some_func_get_gates_0(double param) {
    std::cout << "guess what, i'm in testtaaAAAAA!!!";
    std::vector<int64_t> gates_data = {3, 2, 1, 0, 1, 2, 3}; 
    int64_t num_gates = gates_data.size();
    int64_t* heap_data = new int64_t[num_gates];
    memcpy(heap_data, gates_data.data(), num_gates * sizeof(int64_t));

    std::cout << "param received = " << param << "\n";
    std::cout << "some_func_0_get_gates called\n";
    std::cout << "heap_data address: " << static_cast<void*>(heap_data) << "\n";

    MemRef1D result; 
    result.allocated = heap_data;
    result.aligned = heap_data;
    result.offset = 0;
    result.size = num_gates;
    result.stride = 1;
    return result;
}

// Function 2: Returns the first double
double some_func_get_val1_0() {
    return 1.23;
}

// Function 3: Returns the second double
double some_func_get_val2_0() {
    return 4.56;
}

// FIX: This function MUST match the unbundled 5-argument
// signature that LLVM is calling.
void free_memref_0(int64_t* allocated, int64_t* aligned, int64_t offset, int64_t size, int64_t stride) {
    // Mark other args as unused to prevent compiler warnings
    (void)aligned;
    (void)offset;
    (void)size;
    (void)stride;

    std::cout << "free_memref_0 called\n";
    std::cout << "deleting heap_data at: " << static_cast<void*>(allocated) << "\n";
    delete[] allocated;
}

} // extern "C"
