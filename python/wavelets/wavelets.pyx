import numpy as np
from cython cimport view
from libcpp.string cimport string

cdef extern from "python.wavelets.h" namespace "sopt::pyWavelets":
    void direct[T](T* signal, T* out, string name,
                   unsigned long level, int nrow, int ncol) except +
    void indirect[T](T* signal, T* out, string name,
                     unsigned long level, int nrow, int ncol) except +


def _getInDim(input):
    """ convert input to 1D vector and return dimention information"""
    if input.ndim == 1:
        nrow = input.size
        ncol = 1
    elif input.ndim == 2:
        nrow, ncol = input.shape
        input = input.reshape(nrow * ncol)
    else:
        raise ValueError('input dimension should be either 1D or 2D')
    return input, nrow, ncol


def _dwt(input, name, level, inverse=False):
    in_ndim = input.ndim
    input, nrow, ncol = _getInDim(input)
    output = np.zeros((nrow, ncol), dtype=input.dtype)
    cdef:
        long input_data = input.ctypes.data
        long output_data = output.ctypes.data

    if inverse:
        if input.dtype == "float64":
            indirect[double](
                <double*>input_data, <double*>output_data,
                <string>name, <unsigned long>level, <int>nrow, <int>ncol
            )
        elif input.dtype == "complex128":
            indirect[complex](
                <double complex*>input_data, <double complex*>output_data,
                <string>name, <unsigned long>level, <int>nrow, <int>ncol
            )
    else:
        if input.dtype == "float64":
            direct[double](
                <double*>input_data, <double*>output_data,
                <string>name, <unsigned long>level, <int>nrow, <int>ncol
            )
        elif input.dtype == "complex128":
            direct[complex](
                <double complex*>input_data, <double complex*>output_data,
                <string>name, <unsigned long>level, <int>nrow, <int>ncol
            )
    return output


def dwt(input, name, level, inverse=False):
    """ direct/inverse Daubechies wavelet transform

    Parameters:
    ------------
    inputs: array_like
        1D/2D input signal which can be either float64, int64, or complex128
    name: string
        Daubechies wavelets coefficients, e.g. "DB3"
    level: int
        Wavlets transform level
    inverse: bool
        True - indirect transform
        False - direct transform

    Returns
    ------------
    array_like
        Approximation matrix in the same size as input.

    Notes
    ------------
    * Input that is 'int64' will be converted to 'float64' automatically.
    * Size of signal must be number a multiple of 2^levels.

    Examples
    -----------
    signal = np.random.random((64,64))
    coefficient = wv.dwt(signal, "DB4", 2) # direct transform
    recover = wv.dwt(coefficient, "DB4", 2, inverse = True) # inverse transform

    """
    if input.dtype == "float64" or input.dtype == "complex128":
        return _dwt(input, name, level, inverse=inverse)
    elif input.dtype == "int64":  # convert int to float64
        input = input.astype("float64")
        return _dwt(input, name, level, inverse=inverse)
    else:
        raise ValueError("input data type should be either \
                         'float64' or 'int64' or 'complex128'.")
