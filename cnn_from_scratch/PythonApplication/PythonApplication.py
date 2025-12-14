import ctypes
from pathlib import Path
import numpy as np
import os

# ---------------------------------------------------------------------
# 1. Load the DLL
# ---------------------------------------------------------------------
here = Path(__file__).resolve().parent
root = here.parent  # ...\cnn_from_scratch\cnn_from_scratch

dll_name = "Dll.dll"  # whatever VS produced
dll_path = root / "x64" / "Debug" / dll_name

print("Loading:", dll_path)
os.add_dll_directory(str(dll_path.parent))
cnn = ctypes.CDLL(str(dll_path))

# ---------------------------------------------------------------------
# 2. Mirror the C struct in Python
#
# typedef struct {
#     int   num_channels;
#     int*  channel_dims;
#     float* data;
# } tensor_t;
# ---------------------------------------------------------------------
class Tensor(ctypes.Structure):
    _fields_ = [
        ("num_channels", ctypes.c_int),
        ("channel_dims", ctypes.POINTER(ctypes.c_int)),
        ("total_size", ctypes.c_int),  
        ("data", ctypes.POINTER(ctypes.c_float)),
    ]

# ---------------------------------------------------------------------
# 3. Declare function signatures to ctypes
#
# tensor_t* create_tensor(int num_channels, int* channel_dims);
# void      free_tensor(tensor_t* t);
# ---------------------------------------------------------------------
cnn.create_tensor.argtypes = [
    ctypes.c_int,                     # num_channels
    ctypes.POINTER(ctypes.c_int),     # channel_dims
]
cnn.create_tensor.restype = ctypes.POINTER(Tensor)

cnn.free_tensor.argtypes = [ctypes.POINTER(Tensor)]
cnn.free_tensor.restype = None

# ---------------------------------------------------------------------
# 4. Simple test: create a [2, 3] tensor and poke at it
# ---------------------------------------------------------------------
def main():
    # shape = [2, 3]  (just a tiny example)
    dims = np.array([2, 3], dtype=np.int32)
    num_channels = dims.size

    # get int* to pass into C
    dims_ptr = dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    # call C: tensor_t* t = create_tensor(num_channels, channel_dims)
    t_ptr = cnn.create_tensor(num_channels, dims_ptr)
    if not t_ptr:
        raise RuntimeError("create_tensor returned NULL")

    t = t_ptr.contents  # dereference pointer to get the Tensor struct

    print("num_channels:", t.num_channels)
    shape_from_c = [t.channel_dims[i] for i in range(t.num_channels)]
    print("shape from C:", shape_from_c)

    # compute total elements
    total = 1
    for s in shape_from_c:
        total *= s
    print("total elements:", total)

    # Wrap data pointer as a numpy array (view into C memory)
    buf_type = ctypes.c_float * total
    data_buf = ctypes.cast(t.data, ctypes.POINTER(buf_type)).contents
    np_view = np.frombuffer(data_buf, dtype=np.float32)

    print("initial data:", np_view.copy())

    # Modify from Python - this writes into C's tensor->data
    np_view[:] = 1.23
    print("modified data:", np_view.copy())

    # Clean up
    cnn.free_tensor(t_ptr)
    print("freed tensor (no crash is a good sign)")

if __name__ == "__main__":
    main()
