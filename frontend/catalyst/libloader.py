import ctypes
import numpy as np
import mlir_quantum.ir as ir


double_ptr = ctypes.POINTER(ctypes.c_double)
float_ptr = ctypes.POINTER(ctypes.c_float)


class MemRefDescriptor(ctypes.Structure):
    freed = False
    nparr = None

    def to_numpy(self):
        assert not self.freed, "Memory was freed"
        if not self.nparr:
            self.nparr = np.ctypeslib.as_array(self.aligned, self.shape).copy()
        return self.nparr

    def free(self):
        assert not self.freed, "Memory was already freed"
        self.freed = True


class F64Descriptor1D(MemRefDescriptor):
    _fields_ = [
        ("allocated", double_ptr),
        ("aligned", double_ptr),
        ("offset", ctypes.c_longlong),
        ("size", ctypes.c_longlong),
        ("stride", ctypes.c_longlong),
    ]

    @property
    def shape(self):
        return [self.size]


def _mlir_type_to_ctype(typ: ir.Type, as_return=False):
    if ir.IntegerType.isinstance(typ):
        int_type = ir.IntegerType(typ)
        if int_type.width == 64:
            return [ctypes.c_longlong]
        elif int_type.width == 32:
            return [ctypes.c_long]
    if ir.F64Type.isinstance(typ):
        return [ctypes.c_double]
    if ir.F32Type.isinstance(typ):
        return [ctypes.c_float]
    if ir.RankedTensorType.isinstance(typ):
        tensor_type = ir.RankedTensorType(typ)
        el_type = _mlir_type_to_ctype(tensor_type.element_type)[0]
        ptr_type = np.ctypeslib.ndpointer(
            dtype=el_type, ndim=tensor_type.rank, flags="C_CONTIGUOUS"
        )

        if as_return:
            sizes = [(f"size_{i}", ctypes.c_longlong) for i in range(tensor_type.rank)]
            strides = [(f"stride_{i}", ctypes.c_longlong) for i in range(tensor_type.rank)]

            class ReturnDescriptor(MemRefDescriptor):
                _fields_ = (
                    [
                        ("allocated", ctypes.POINTER(el_type)),
                        ("aligned", ctypes.POINTER(el_type)),
                        ("offset", ctypes.c_long),
                    ]
                    + sizes
                    + strides
                )

                @property
                def shape(self):
                    return [getattr(self, f"size_{i}") for i in range(tensor_type.rank)]

            return [ReturnDescriptor]
        else:
            # 1 for offset, the rank many sizes and strides
            return [ptr_type, ptr_type] + [ctypes.c_longlong] * (1 + tensor_type.rank * 2)

    assert False, f"Unhandled type {typ}"


def convert_argument(arg):
    if isinstance(arg, (int, float, np.int64, np.float64)):
        return (arg,)
    assert isinstance(arg, np.ndarray), f"Unexpected argument type: '{type(arg)}'"
    return (arg, arg, 0) + arg.shape + tuple(stride // arg.itemsize for stride in arg.strides)


def struct_to_tuple(s):
    if isinstance(s, (int, float)):
        return s
    elif isinstance(s, MemRefDescriptor):
        return s.to_numpy()
    descriptors = (getattr(s, field[0]) for field in s._fields_)
    return ((desc if isinstance(desc, float) else desc.to_numpy()) for desc in descriptors)


class SharedLibraryManager:
    def __init__(self, shared_object_file: str, func_name: str, func_type: ir.FunctionType) -> None:
        self.shared_object_file = shared_object_file
        self.func_name = func_name
        self.func_type = func_type

    @staticmethod
    def load_symbols(shared_object_file, func_name):
        """Load symbols necessary for for execution of the compiled function.

        Args:
            shared_object_file: path to shared object file
            func_name: name of compiled function to be executed

        Returns:
            shared_object: in memory shared object
            function: function handle
            setup: handle to the setup function, which initializes the device
            teardown: handle to the teardown function, which tears down the device
        """
        shared_object = ctypes.CDLL(shared_object_file)

        setup = shared_object.setup
        setup.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        setup.restypes = ctypes.c_int

        teardown = shared_object.teardown
        teardown.argtypes = None
        teardown.restypes = None

        jit_func = shared_object[func_name]

        mem_transfer = shared_object["_mlir_memory_transfer"]

        return shared_object, jit_func, setup, teardown, mem_transfer

    def _exec(self, *args):
        shared_object, jit_func, setup, teardown, mem_transfer = SharedLibraryManager.load_symbols(
            self.shared_object_file, self.func_name
        )

        params_to_setup = [b"jitted-function"]
        argc = len(params_to_setup)
        array_of_char_ptrs = (ctypes.c_char_p * len(params_to_setup))()
        array_of_char_ptrs[:] = params_to_setup

        # What are these arguments to the setup function?
        setup(ctypes.c_int(argc), array_of_char_ptrs)
        assert len(self.func_type.results) == 1, "Expected jit function to return 1 result"

        jit_func.argtypes = [el for typ in self.func_type.inputs for el in _mlir_type_to_ctype(typ)]
        jit_func.restype = _mlir_type_to_ctype(self.func_type.results[0], as_return=True)[0]
        return struct_to_tuple(jit_func(*[el for arg in args for el in convert_argument(arg)]))

    def __call__(self, *args, **kwargs):
        return self._exec(*args)
