add_library(catalyst_python_interpreter SHARED Python.cpp)
target_include_directories(catalyst_python_interpreter PRIVATE ${util_includes})

