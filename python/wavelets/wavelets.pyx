import numpy as np
from cython cimport view
from libcpp.string cimport string

cdef extern from "python.wavelets.h" namespace "sopt::pyWavelets":
    ctypedef unsigned t_uint
    void direct[T](T * signal, T* out, const string & name,
                   t_uint level, t_uint nrow, t_uint ncol) except +
    void indirect[T](T * signal, T* out, const string & name,
                     t_uint level, t_uint nrow, t_uint ncol) except +


cdef _dwt(input, name, level, inverse=False):
    if input.ndim > 2:
        raise ValueError("Expect 1D or 2D arrays")
    nrow, ncol = input.shape if input.ndim == 2 else (input.size, 1)
    output = np.zeros(input.shape, dtype=input.dtype)
    cdef:
        long input_data = input.ctypes.data
        long output_data = output.ctypes.data
        string cname = name.encode("UTF-8")

    if inverse:
        if input.dtype == "float64":
            indirect[double](
                <double*>input_data, <double*>output_data,
                cname, <unsigned>level, <unsigned>nrow, <unsigned>ncol
            )
        elif input.dtype == "complex128":
            indirect[complex](
                <double complex*>input_data, <double complex*>output_data,
                cname, <unsigned>level, <unsigned>nrow, <unsigned>ncol
            )
    else:
        if input.dtype == "float64":
            direct[double](
                <double*>input_data, <double*>output_data,
                cname, <unsigned>level, <unsigned>nrow, <unsigned>ncol
            )
        elif input.dtype == "complex128":
            direct[complex](
                <double complex*>input_data, <double complex*>output_data,
                cname, <unsigned>level, <unsigned>nrow, <unsigned>ncol
            )
    return output


def dwt(input, name, level, inverse=False):
    """ direct/inverse Daubechies wavelet transform

    Parameters:
    ------------
    inputs: numpy array
        1D/2D numerical input signal
    name: string
        Daubechies wavelets coefficients, e.g. "DB1" through "DB38"
    level: int
        Wavelets transform level
    inverse: bool
        True - indirect transform
        False - direct transform

    Returns
    ------------
    numpy array
        Approximation matrix in the same size as input

    Notes
    ------------
    * Input will be converted to "float64" or "complex128" automatically
    * To avoid extra copies, please use those types exclusively
    * Size of signal must be a multiple of 2^levels

    Examples
    -----------
    signal = np.random.random((64,64))
    coefficient = wv.dwt(signal, "DB4", 2) # direct transform
    recover = wv.dwt(coefficient, "DB4", 2, inverse = True) # inverse transform

    """
    from numpy import iscomplex, isreal, require, all
    is_complex = all(iscomplex(input))
    if (not is_complex) and not all(isreal(input)):
        raise ValueError("Incorrect array type")
    dtype = "complex128" if is_complex else "float64"
    normalized_input = require(input, requirements=['C'], dtype=dtype)
    return _dwt(normalized_input, name, level, inverse=inverse).astype(input.dtype)
