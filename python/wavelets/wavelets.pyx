import numpy as np
from cython cimport view
from libcpp.string cimport string

cdef extern from "python.wavelets.h" namespace "sopt::pyWavelets":
    void direct[T](T *signal, T* out, string name, 
                   unsigned long level, int nrow, int ncol);
    void indirect[T](T *signal, T* out, string name, 
                   unsigned long level, int nrow, int ncol);


def _getInDim(input):
    """ initialise output """
    if input.ndim == 1:
        nrow = input.size
        ncol = 1
    elif input.ndim == 2:
        nrow, ncol = input.shape
        input = input.reshape(nrow * ncol)
    else:
        raise ValueError('input dimension should be either 1D or 2D')
    return input, nrow, ncol

def rdwt(input, name, level, inverse = False):
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
    if in_ndim == 1:
        return output.reshape(nrow)
    else:
        return output.reshape(nrow, ncol)

def cdwt(input, name, level, inverse = False):
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
    if input.dtype == "float64":
        return rdwt(input, name, level, inverse = inverse)
    elif input.dtype == "complex128":
        return cdwt(input, name, level, inverse = inverse)
    else:
        raise ValueError("input data type should be either 'float64' or 'complex128'.")


