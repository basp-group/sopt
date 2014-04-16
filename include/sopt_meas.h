//
//  sopt_meas.h
//  
//
//  Created by Rafael Carrillo on 9/4/12.
//
//

#ifndef SOPT_MEAS
#define SOPT_MEAS
#include "sopt_config.h"

#include <complex.h>
/*!
 * Parameters for the uniform random sampling operator.
 */
typedef struct {
    /*! Number of measurements. */
    int nmeas;
    /*! Dimension of the original signal. */
    int nx;
    /*! Reality flag. 1 for real data, 0 for complex data. */
    int real_flag;
    /*! Indexes for the selection. */
    int *perm;
} sopt_meas_usparam;

/*!
 * Parameters for the complex dense matrix measurement operator.
 */
typedef struct {
    /*! Number of measurements. */
    int nmeas;
    /*! Dimension of the original signal. */
    int nx;
    /*! Pointer to the Gaussian sampling matrix stored in 
    vector form (columnwise). */
    complex double *mat;
} sopt_meas_dcmatparam;

/*!
 * Parameters for the real dense matrix measurement operator.
 */
typedef struct {
    /*! Number of measurements. */
    int nmeas;
    /*! Dimension of the original signal. */
    int nx;
    /*! Pointer to the Gaussian sampling matrix stored in 
    vector form (columnwise). */
    double *mat;
} sopt_meas_drmatparam;

void sopt_meas_urandsamp(void *out, void *in, void **data);
void sopt_meas_urandsampadj(void *out, void *in, void **data);
void sopt_meas_identity(void *out, void *in, void **data);
void sopt_meas_densematc(void *out, void *in, void **data);
void sopt_meas_densematadjc(void *out, void *in, void **data);
void sopt_meas_densematr(void *out, void *in, void **data);
void sopt_meas_densematadjr(void *out, void *in, void **data);
void sopt_meas_sparsematc(void *out, void *in, void **data);
void sopt_meas_sparsematadjc(void *out, void *in, void **data);
void sopt_meas_sparsematr(void *out, void *in, void **data);
void sopt_meas_sparsematadjr(void *out, void *in, void **data);

#ifdef SOPT_FFTW_INSTALLED
void sopt_meas_fsamp_c2c(void *out, void *in, void **data);
void sopt_meas_fsampadj_c2c(void *out, void *in, void **data);
void sopt_meas_fft_r2c_full(void *out, void *in, void **data);
void sopt_meas_fft_r2c(void *out, void *in, void **data);
void sopt_meas_fft_c2r(void *out, void *in, void **data);
void sopt_meas_fftw_c2c(void *out, void *in, void **data);
#endif 



#endif
