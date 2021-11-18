#!/usr/bin/env python3
import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')
from mitsuba.core import UInt32, Vector2f, Vector3f, Vector2u, Vector3u, Float32
import enoki as ek


# Convert flat array into a vector of arrays (will be included in next enoki release)
def ravel(buf, dim = 3):
    idx = dim * UInt32.arange(ek.slices(buf) // dim)
    if dim == 2:
        return Vector2f(ek.gather(buf, idx), ek.gather(buf, idx + 1))
    elif dim == 3:
        return Vector3f(ek.gather(buf, idx), ek.gather(buf, idx + 1), ek.gather(buf, idx + 2))

def ravel_int(buf, dim = 3):
    idx = dim * UInt32.arange(ek.slices(buf) // dim)
    if dim == 2:
        return Vector2u(ek.gather(buf, idx), ek.gather(buf, idx + 1))
    elif dim == 3:
        return Vector3u(ek.gather(buf, idx), ek.gather(buf, idx + 1), ek.gather(buf, idx + 2))

# Return contiguous flattened array (will be included in next enoki release)
def unravel(source, target, dim = 3):
    idx = UInt32.arange(ek.slices(source))
    for i in range(dim):
        ek.scatter(target, source[i], dim * idx + i)

def to_each_col(buf, ek_func, dim = 3):
    res = []
    ek_type = type(buf)
    for i in range(dim):
        res.append(ek_func(buf[i]))
    res = ek_type(*res)
    return res
