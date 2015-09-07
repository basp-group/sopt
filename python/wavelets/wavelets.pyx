import numpy as np
from cython cimport view
from libcpp.string cimport string

cdef extern from "python.wavelets.h" namespace "sopt::pyWavelets":
    void direct[T](T *signal, T* out, string name, 
                   unsigned long level, int nrow, int ncol) except +
    void indirect[T](T *signal, T* out, string name, 
                     unsigned long level, int nrow, int ncol) except +


def _getInDim(input):
    """ convert input to 1D vector and return 
    dimention information"""
    if input.ndim == 1:
        nrow = input.size
        ncol = 1
    elif input.ndim == 2:
        nrow, ncol = input.shape
        input = input.reshape(nrow * ncol)
    else:
        raise ValueError('input dimension should be either 1D or 2D')
    return input, nrow, ncol

def _rdwt(input, name, level, inverse = False):
    in_ndim = input.ndim
    input, nrow, ncol = _getInDim(input)
    output = np.zeros(nrow * ncol, dtype=input.dtype)
    cdef:
        double[:] input_view = input
        double[:] output_view = output
        double *inptr = &input_view[0]
        double *outptr = &output_view[0]
        int c_nrow = nrow
        int c_ncol = ncol
        unsigned long c_level = level 
        string c_name = name

    if inverse: 
        indirect(inptr, outptr, c_name, c_level, c_nrow, c_ncol)
    else: 
        direct(inptr, outptr, c_name, c_level, c_nrow, c_ncol)
    if in_ndim == 1:# vector case
        return output.reshape(nrow)
    else:
        return output.reshape(nrow, ncol)

def _cdwt(input, name, level, inverse = False):
    in_ndim = input.ndim
    input, nrow, ncol = _getInDim(input)
    output = np.zeros(nrow * ncol, dtype=input.dtype)
    cdef:
        double complex[:] input_view = input
        double complex[:] output_view = output
        double complex *inptr = &input_view[0]
        double complex *outptr = &output_view[0]
        int c_nrow = nrow
        int c_ncol = ncol
        unsigned long c_level = level 
        string c_name = name
    if inverse: 
        indirect(inptr, outptr, c_name, c_level, c_nrow, c_ncol)
    else:
        direct(inptr, outptr, c_name, c_level, c_nrow, c_ncol)
    if in_ndim == 1:    
        return output.reshape(nrow)
    else:
        return output.reshape(nrow, ncol)

def dwt(input, name, level, inverse = False):
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

    if input.dtype == "float64":
        return _rdwt(input, name, level, inverse = inverse)
    elif input.dtype == "complex128":
        return _cdwt(input, name, level, inverse = inverse)
    elif input.dtype == "int64":#convert int to float64
        input = np.array(input, dtype = "float64")
        return _rdwt(input, name, level, inverse = inverse) 
    else:
        raise ValueError("input data type should be either 'float64' or 'int64' or 'complex128'.")


