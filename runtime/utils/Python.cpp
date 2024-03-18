#include "Python.hpp"

std::mutex python_mutex;
std::mutex &getPythonMutex() { return python_mutex; }
