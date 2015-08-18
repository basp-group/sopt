import numpy as np
from cython cimport view
from libcpp.string cimport string

cdef extern from "python.wavelets.h" namespace "sopt::pyWavelets":
    void direct[T](T *signal, T* out, string name, 
                   unsigned long level, int nrow, int ncol);
    void direct[T](T *signal, T *coefficient, T *out,
                   string name, unsigned long level, int nrow,
                   int ncol);
    void indirect[T](T *signal, T *out, string name,\
                     unsigned long level, int nrow, int ncol);
    void indirect[T](T *signal, T *coefficient, T *out,
                   string name, unsigned long level, int nrow,
                   int ncol);

def _getInDim(input):
    """ initialise output """
    assert input.ndim < 3
    if input.ndim == 1:
        nrow = input.size
        ncol = 1
    if input.ndim == 2:
        nrow, ncol = input.shape
        input = input.reshape(nrow * ncol)
    return input, nrow, ncol

def dwt(input, name, level, inverse = False, coefficient = None):
    
    input, nrow, ncol = _getInDim(input)
    output = np.zeros(nrow * ncol, dtype=input.dtype)
       
    if coefficient is None:
        coefficient = np.zeros(1, dtype=input.dtype)#dummy coefficient
        WITH_COEFFICIENT = False 
    else:
        coefficient, nrow_cft, ncol_cft = _getInDim(coefficient)
        if nrow_cft != nrow or ncol_cft != ncol:
            raise ValueError('The dimension of coefficient should agree with that of input.\n')
        WITH_COEFFICIENT = True
        
    cdef:
        double[:] input_view = input
        double[:] output_view = output
        double[:] coefficient_view = coefficient
        double *inptr = &input_view[0]
        double *outptr = &output_view[0]
        double *cftptr = &coefficient_view[0]
        int c_nrow = nrow
        int c_ncol = ncol
        unsigned long c_level = level 
        string c_name = name
    
    if inverse is False:
        if WITH_COEFFICIENT:
            direct(inptr, cftptr, outptr, c_name, c_level,
                   c_nrow, c_ncol)
        else:
            direct(inptr, outptr, c_name, c_level, c_nrow, c_ncol)
    elif inverse is True:
        if WITH_COEFFICIENT:
            indirect(inptr, cftptr, outptr, c_name, c_level,
                c_nrow, c_ncol)
        else:
            indirect(inptr, outptr, c_name, c_level, c_nrow, c_ncol)
    else:
        raise ValueError('argument inverse should be True/False')
    
    return output.reshape(nrow,ncol)



def cdwt(input, name, level, inverse = False, coefficient = None):
    """ initialise output """
    assert input.ndim < 3
    if input.ndim == 1:
        nrow = input.size
        ncol = 1
    if input.ndim == 2:
        nrow, ncol = input.shape
        input = input.reshape(nrow * ncol)
    WITH_COEFFICIENT = True
    output = np.zeros(nrow * ncol, dtype=input.dtype)
    if coefficient is None:
        coefficient = np.zeros(1, dtype=input.dtype)
        WITH_COEFFICIENT = False 

    cdef:
        double complex[:] input_view = input
        double complex[:] output_view = output
        double complex *inptr = &input_view[0]
        double complex *outptr = &output_view[0]
        int c_nrow = nrow
        int c_ncol = ncol
        unsigned long c_level = level 
        string c_name = name
    if WITH_COEFFICIENT is True:
        print 'with'
    else:
        if inverse is False:
            direct(inptr, outptr, c_name, c_level, c_nrow, c_ncol)
        elif inverse is True:
            indirect(inptr, outptr, c_name, c_level, c_nrow, c_ncol)
        else:
            raise ValueError('argument inverse should be True/False')
    return output.reshape(nrow,ncol)


