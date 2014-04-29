//
//  sopt_meas.c
//  
//
//  Created by Rafael Carrillo on 9/4/12.
//
//
#include "sopt_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include SOPT_BLAS_H
#ifdef SOPT_FFTW_INSTALLED
  #include <fftw3.h>
#endif
#include "sopt_ran.h"
#include "sopt_error.h"
#include "sopt_sparsemat.h"
#include "sopt_meas.h"

/*!
 * Random uniform sampling. Takes nmeas samples 
 * uniformly at random from the vector x of dimension
 * nx.
 *
 * \param[out] out Output vector of dimension nmeas.
 * \param[in] in Input vector of dimension nx.
 * \param[in] data Data structure with the information
 *            of the sampling process.
 */
void sopt_meas_urandsamp(void *out, void *in, void **data){
    
    int i;
    complex double *yc;
    complex double *xc;
    double *yr;
    double *xr;
    sopt_meas_usparam *param;

    //Cast input pointers
    param = (sopt_meas_usparam*)data[0];

    if (param->real_flag == 1){
      yr = (double*)out;
      xr = (double*)in;
    }
    else {
      yc = (complex double*)out;
      xc = (complex double*)in;
    }

    if (param->real_flag == 1){
      for (i=0; i < param->nmeas; i++)
        yr[i] = xr[param->perm[i]];
    }
    else {
      for (i=0; i < param->nmeas; i++)
        yc[i] = xc[param->perm[i]];
    }
}

/*!
 * Random uniform sampling. Adjoint opeartor.
 *
 * \param[out] out Output vector of dimension nx.
 * \param[in] in Input vector of dimension nmeas.
 * \param[in] data Data structure with the information
 *            of the sampling process.
 */
void sopt_meas_urandsampadj(void *out, void *in, void **data){
    
    int i;
    complex double *yc;
    complex double *xc;
    double *yr;
    double *xr;
    sopt_meas_usparam *param;

    //Cast input pointers
    param = (sopt_meas_usparam*)data[0];

    if (param->real_flag == 1){
      yr = (double*)in;
      xr = (double*)out;
    }
    else {
      yc = (complex double*)in;
      xc = (complex double*)out;
    }

    if (param->real_flag == 1){
      for (i=0; i < param->nx; i++)
          xr[i] = 0.0;
      for (i=0; i < param->nmeas; i++)
          xr[param->perm[i]] = yr[i];
    }
    else {
      for (i=0; i < param->nx; i++)
          xc[i] = 0.0 +0.0*I;
      for (i=0; i < param->nmeas; i++)
          xc[param->perm[i]] = yc[i];
    }
}

/*!
 * Identity operator. Adjoint same as forward.
 *
 * \param[out] out Output vector of dimension nx.
 * \param[in] in Input vector of dimension nx.
 * \param[in] data Data structure with the information
 *            of the sampling process.
 *            data[0] nx dimension of the array.
 *            data[1] real_flag reality flag. 1 for real data, 0 for complex.
 */
void sopt_meas_identity(void *out, void *in, void **data){
    
    int *nx;
    int *real_flag;
    //Cast input pointers
    nx = (int*)data[0];
    real_flag = (int*)data[1];
    //Copy
    if (*real_flag == 1) {
      cblas_dcopy(*nx, (double*)in, 1, (double*)out, 1);
    }
    else {
      cblas_zcopy(*nx, in, 1, out, 1);
    } 
}

/*!
 * Complex dense matrix measurement operator. Takes nmeas  
 * measurements of the vector x of dimension nx.
 *
 * \param[out] out Output vector of dimension nmeas.
 * \param[in] in Input vector of dimension nx.
 * \param[in] data Data structure with the information
 *            of the sampling process.
 */
void sopt_meas_densematc(void *out, void *in, void **data){

    complex double alpha;
    complex double beta;

    alpha = 1.0 + 0.0*I;
    beta = 0.0 + 0.0*I;
    
    sopt_meas_dcmatparam *param;

    //Cast input pointers
    param = (sopt_meas_dcmatparam*)data[0];
    
    cblas_zgemv(CblasColMajor, CblasNoTrans, 
                 param->nmeas, param->nx,
                 (void*)&alpha, (void*)param->mat,
                 param->nmeas, in,
                 1, (void*)&beta,
                 out, 1);    

} 

/*!
 * Complex dense matrix measurement adjoint operator.
 *
 * \param[out] out Output vector of dimension nx.
 * \param[in] in Input vector of dimension nmeas.
 * \param[in] data Data structure with the information
 *            of the sampling process.
 */
void sopt_meas_densematadjc(void *out, void *in, void **data){

    complex double alpha;
    complex double beta;

    alpha = 1.0 + 0.0*I;
    beta = 0.0 + 0.0*I;
    
    sopt_meas_dcmatparam *param;

    //Cast input pointers
    param = (sopt_meas_dcmatparam*)data[0];
    
    cblas_zgemv(CblasColMajor, CblasTrans, 
                 param->nmeas, param->nx,
                 (void*)&alpha, (void*)param->mat,
                 param->nmeas, in,
                 1, (void*)&beta,
                 out, 1);    

} 

/*!
 * Real dense matrix measurement operator. Takes nmeas  
 * measurements of the vector x of dimension nx.
 *
 * \param[out] out Output vector of dimension nmeas.
 * \param[in] in Input vector of dimension nx.
 * \param[in] data Data structure with the information
 *            of the sampling process.
 */
void sopt_meas_densematr(void *out, void *in, void **data){

    sopt_meas_drmatparam *param;

    //Cast input pointers
    param = (sopt_meas_drmatparam*)data[0];
    
    cblas_dgemv(CblasColMajor, CblasNoTrans, 
                 param->nmeas, param->nx,
                 1.0, param->mat,
                 param->nmeas, (double*)in,
                 1, 0.0,
                 (double*)out, 1);    

} 

/*!
 * Real dense matrix measurement adjoint operator.
 *
 * \param[out] out Output vector of dimension nx.
 * \param[in] in Input vector of dimension nmeas.
 * \param[in] data Data structure with the information
 *            of the sampling process.
 */
void sopt_meas_densematadjr(void *out, void *in, void **data){

    sopt_meas_drmatparam *param;

    //Cast input pointers
    param = (sopt_meas_drmatparam*)data[0];
    
    cblas_dgemv(CblasColMajor, CblasTrans, 
                 param->nmeas, param->nx,
                 1.0, param->mat,
                 param->nmeas, (double*)in,
                 1, 0.0,
                 (double*)out, 1);    
} 

/*!
 * Complex sparse matrix measurement operator. Takes nmeas  
 * measurements of the vector x of dimension nx.
 *
 * \param[out] out Output vector of dimension nmeas.
 * \param[in] in Input vector of dimension nx.
 * \param[in] data Data structure with the information
 *            of the sampling process.
 *            data[0] real_flag. Reality flag. 1 real data, 0 complex data.
 *            data[1] structure containing the sparse matrix info
 */
void sopt_meas_sparsematc(void *out, void *in, void **data){

  sopt_sparsemat *param;

  //Cast input pointers
  
  param = (sopt_sparsemat*)data[0];

  sopt_sparsemat_fwd_complex((complex double*)out, (complex double*)in, param);

}

/*!
 * Complex sparse matrix measurement adjoint operator.
 *
 * \param[out] out Output vector of dimension nx.
 * \param[in] in Input vector of dimension nmeas.
 * \param[in] data Data structure with the information
 *            of the sampling process.
 */
void sopt_meas_sparsematadjc(void *out, void *in, void **data){
  
  sopt_sparsemat *param;

  //Cast input pointers
  
  param = (sopt_sparsemat*)data[0];

  sopt_sparsemat_adj_complex((complex double*)out, (complex double*)in, param);
 
}

/*!
 * Complex sparse matrix measurement operator. Takes nmeas  
 * measurements of the vector x of dimension nx.
 *
 * \param[out] out Output vector of dimension nmeas.
 * \param[in] in Input vector of dimension nx.
 * \param[in] data Data structure with the information
 *            of the sampling process.
 */
void sopt_meas_sparsematr(void *out, void *in, void **data){

  
  sopt_sparsemat *param;

  //Cast input pointers
  param = (sopt_sparsemat*)data[0];

  sopt_sparsemat_fwd_real((double*)out, (double*)in, param);
  
}

/*!
 * Complex sparse matrix measurement adjoint operator.
 *
 * \param[out] out Output vector of dimension nx.
 * \param[in] in Input vector of dimension nmeas.
 * \param[in] data Data structure with the information
 *            of the sampling process.
 */
void sopt_meas_sparsematadjr(void *out, void *in, void **data){
  
  sopt_sparsemat *param;

  //Cast input pointers
  
  param = (sopt_sparsemat*)data[0];

  sopt_sparsemat_adj_real((double*)out, (double*)in, param);

}

#ifdef SOPT_FFTW_INSTALLED
/*!
 * Fourier sampling operator. It takes samples of the Fourier transform
 * of the input signal according to the subset perm.
 * Compute normalized forward Fouier transform of a complex signal. 
 * 
 * \param[out] out (complex double*) Forward Fourier transform of input signal.
 * \param[in] in (complex double*) Real input signal.
 * \param[in] data 
 * - data[0] (fftw_plan*): The real-to-complex FFTW plan to use when
 *      computing the Fourier transform (passed as an input so that the
 *      FFTW can be FFTW_MEASUREd beforehand).
 */
void sopt_meas_fsamp_c2c(void *out, void *in, void **data){
    
    int i;
    double scale;
    int *nx;
    int *nmeas;
    complex double *yc;
    complex double *xc;
    int *perm;
    complex double *temp;
    fftw_plan *plan;

    //Cast input pointers
    plan = (fftw_plan*)data[0];
    perm = (int*)data[1];
    nmeas = (int*)data[2];
    nx = (int*)data[3];
    temp = (complex double*)data[4];

    yc = (complex double*)out;
    xc = (complex double*)in;

    fftw_execute_dft(*plan, xc, temp);

    scale = 1.0/sqrt((double)*nx);

    for (i=0; i < *nmeas; i++)
      *(yc + i) = *(temp + *(perm + i))*scale;
}

#endif 

#ifdef SOPT_FFTW_INSTALLED
/*!
 * Adjoint Fourier sampling operator. It takes samples of the Fourier transform
 * of the input signal according to the subset perm.
 * Compute normalized inverse Fouier transform of a complex signal. 
 * 
 * \param[out] out (complex double*) Forward Fourier transform of input signal.
 * \param[in] in (complex double*) Real input signal.
 * \param[in] data 
 * - data[0] (fftw_plan*): The real-to-complex FFTW plan to use when
 *      computing the Fourier transform (passed as an input so that the
 *      FFTW can be FFTW_MEASUREd beforehand).
 */
void sopt_meas_fsampadj_c2c(void *out, void *in, void **data){
    
    int i;
    double scale;
    int *nx;
    int *nmeas;
    complex double *yc;
    complex double *xc;
    int *perm;
    complex double *temp;
    fftw_plan *plan;

    //Cast input pointers
    plan = (fftw_plan*)data[0];
    perm = (int*)data[1];
    nmeas = (int*)data[2];
    nx = (int*)data[3];
    temp = (complex double*)data[4];
    
    yc = (complex double*)out;
    xc = (complex double*)in;

    scale = 1.0/sqrt((double)*nx);

    for (i=0; i < *nx; i++)
        *(temp + i) = 0.0 +0.0*I;
    for (i=0; i < *nmeas; i++)
        *(temp + *(perm + i)) = *(xc + i)*scale;
    
    fftw_execute_dft(*plan, temp, yc);
  
}

#endif 


#ifdef SOPT_FFTW_INSTALLED
/*!
 * Compute forward Fouier transform of real signal.  A real-to-complex
 * FFT is used (for speed optimisation) but the complex output signal
 * is filled to its full size through conjugate symmetry.
 * 
 * \param[out] out (complex double*) Forward Fourier transform of input signal.
 * \param[in] in (double*) Real input signal.
 * \param[in] data 
 * - data[0] (int): The size of the image in the x dimension (nx).
 * - data[1] (int): The size of the image in the y dimension (ny).
 * - data[2] (fftw_plan*): The real-to-complex FFTW plan to use when
 *      computing the Fourier transform (passed as an input so that the
 *      FFTW can be FFTW_MEASUREd beforehand).
 */
void sopt_meas_fft_r2c_full(void *out, void *in, void **data) {

  fftw_plan *plan;
  int *nx, *ny;
  double complex *y, *y_half;
  int iu, iv, ind, ind_half;
  int iu_neg, iv_neg, ind_neg;

  // Cast intput pointers.
  y = (double complex*)out;
  nx = (int*)data[0];
  ny = (int*)data[1];
  plan = (fftw_plan*)data[2];

  // Allocate space for output of real-to-complex FFT before compute
  // full plane through conjugate symmetry.
  y_half = (complex double*)malloc((*nx) * (*ny) * sizeof(complex double));
  SOPT_ERROR_MEM_ALLOC_CHECK(y_half);

  // Perform real-to-complex FFT.
  fftw_execute_dft_r2c(*plan, 
		       (double*)in, 
		       y_half);

  // Compute other half of complex plane through conjugate symmetry.
  for (iu = 0; iu < (*nx); iu++) {
    for (iv = 0; iv < (*ny)/2+1; iv++) {

      ind_half = iu*((*ny)/2+1) + iv;
      ind = iu * (*ny) + iv;

      // Copy current data element.
      y[ind] = y_half[ind_half];

      // Compute reflected data through conjugate symmetry if
      // necessary.
      if (iu == 0 && iv == 0) {
	// Do nothing for DC component.
      } 
      else if (iu == 0) {
	// Reflect along line iu = 0.
	iv_neg = (*ny) - iv;
	ind_neg = iu * (*ny) + iv_neg;
	if (ind != ind_neg) y[ind_neg] = conj(y_half[ind_half]);
      }
      else if (iv == 0) {
	// Reflect along line iu = 0.
	iu_neg = (*nx) - iu;
	ind_neg = iu_neg * (*ny) + iv;
	if (ind != ind_neg) y[ind_neg] = conj(y_half[ind_half]);
      }
      else {
	// Reflect along diagonal.
	iv_neg = (*ny) - iv;
	iu_neg = (*nx) - iu;
	ind_neg = iu_neg * (*ny) + iv_neg;
	if (ind != ind_neg) y[ind_neg] = conj(y_half[ind_half]);
      }
    }
  }
  
  // Free temporary memory.
  free(y_half);

}
#endif


#ifdef SOPT_FFTW_INSTALLED
/*!
 * Compute forward Fouier transform of real signal.  A real-to-complex
 * FFT is used (for speed optimisation), with the output signal only
 * defined on the half complex plane.
 * 
 * \param[out] out (complex double*) Forward Fourier transform of input signal.
 * \param[in] in (double*) Real input signal.
 * \param[in] data 
 * - data[0] (fftw_plan*): The real-to-complex FFTW plan to use when
 *      computing the Fourier transform (passed as an input so that the
 *      FFTW can be FFTW_MEASUREd beforehand).
 */
void sopt_meas_fft_r2c(void *out, void *in, void **data) {

  fftw_plan *plan;
  double complex *y;

  // Cast intput pointers.
  y = (double complex*)out;
  plan = (fftw_plan*)data[0];

  // Perform real-to-complex FFT.
  fftw_execute_dft_r2c(*plan, 
		       (double*)in, 
		       y);

}
#endif


#ifdef SOPT_FFTW_INSTALLED
/*!
 * Compute inverse Fouier transform of real signal.  A complex-to-real
 * FFT is used (for speed optimisation).
 * 
 * \param[out] out (complex double*) Inverse Fourier transform of input signal.
 * \param[in] in (complex double*) Complex input signal.
 * \param[in] data 
 * - data[0] (fftw_plan*): The complex-to-real FFTW plan to use when
 *      computing the Fourier transform (passed as an input so that the
 *      FFTW can be FFTW_MEASUREd beforehand).
 */
void sopt_meas_fft_c2r(void *out, void *in, void **data) {

  fftw_plan *plan;
  double *x;

  // Cast intput pointers.
  x = (double*)out;
  plan = (fftw_plan*)data[0];

  // Perform complex-to-real FFT.
  fftw_execute_dft_c2r(*plan, 
		       (complex double*)in, 
		       x);

}
#endif


#ifdef SOPT_FFTW_INSTALLED
/*!
 * Generic complex-to-complex FFTW transform (forward or inverse).
 *
 * \param[out] out (complex double*) Fourier transform (forward or inverse) of
 * input signal.
 * \param[in] in (complex double*) Complex input signal.
 * \param[in] data 
 * - data[0] (fftw_plan*): The complex-to-complex FFTW plan to use when
 *      computing the Fourier transform (passed as an input so that the
 *      FFTW can be FFTW_MEASUREd beforehand).
 */
void sopt_meas_fftw_c2c(void *out, void *in, 
		      void **data) {

  fftw_plan *plan;

  plan = (fftw_plan*)data[0];
  fftw_execute_dft(*plan, 
		   (complex double*)in, 
		   (complex double*)out);

}
#endif

