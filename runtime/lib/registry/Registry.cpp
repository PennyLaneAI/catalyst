#include <cstdint>
#include <unordered_map>

class Environment
{
private:
	std::unordered_map<uintptr_t, void*> _function_map;
public:
	uintptr_t insert(void*);
	void erase(void*);
};

uintptr_t
Environment::insert(void *fptr)
{
	uintptr_t retval = (uintptr_t)fptr;
	_function_map.insert({retval, fptr});
	return retval;
}

