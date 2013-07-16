//
//  sopt_sara.c
//  
//
//  Created by Rafael Carrillo on 10/26/12.
//
//

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#elif __unix__
#include <cblas.h>
#else
#include <cblas.h>
#endif
#include "sopt_error.h"
#include "sopt_wavelet.h"
#include "sopt_sara.h"

/*!
 * This function initializes the parameters for the SARA dictionary.
 *
 * \param[out] param Structure with the necessary parameters for the 
 *            operator.
 * \param[in] nx1 Number of rows in the input image.
 * \param[in] nx2 Number of rows in the input image.
 * \param[in] nb_levels Number of levels in the wavelet decomposition.
 * \param[in] dict_types array containing the list of basis 
 *            used in the operator.
 */
void sopt_sara_initop(sopt_sara_param *param, int nx1, int nx2, int nb_levels, 
		      sopt_wavelet_type *dict_types){
    
  int i;

  //Allocate memory for the wavelet parameter list
  param->wav_params = malloc(param->ndict * sizeof(sopt_wavelet_param));
  SOPT_ERROR_MEM_ALLOC_CHECK(param->wav_params);

  for (i = 0; i < param->ndict; i++){
    param->wav_params[i].nx1 = nx1;
    param->wav_params[i].nx2 = nx2;
    param->wav_params[i].nb_levels = nb_levels;
    param->wav_params[i].type = dict_types[i];
    //Initialize wavelet parameters
    sopt_wavelet_initwav(&param->wav_params[i]);
  }

}

/*!
 * This function frees all memory used to store the 
 * SARA operator parameters.
 *
 * \param[in] param Structure with the necessary parameters for the 
 *            operator.
 */
void sopt_sara_free(sopt_sara_param *param) {

  int i;

  if (param->wav_params != NULL){
		
    for (i = 0; i < param->ndict; i++){
      if (param->wav_params[i].h != NULL) free(param->wav_params[i].h);
    }

    free(param->wav_params);
  }

}

/*!
 * This function computes the analysis operator for concatenation of bases.
 *
 * \param[out] out Output data.
 * \param[in] in Input data.
 * \param[in] data array containing the necessary parameters for the 
 *            operator.
 *            data[0] Parameter structure for the operator
 *            of type sopt_sara_param.
 */
void sopt_sara_analysisop(void *out, void *in, void **data){

  int i,d;
  int nx;
  double sc;
  sopt_sara_param *param;
  complex double *tempc;
  double *tempr;
  complex double *xinc;
  double *xinr;
  complex double *xoutc;
  double *xoutr;


  //cast input pointer for the data
  param = (sopt_sara_param*)data[0];
  if (param->real == 1){
    xinr = (double*)in;
    xoutr = (double*)out;
  }
  else {
    xinc = (complex double*)in;
    xoutc = (complex double*)out;
  }

  // Image dimension.
  nx = param->wav_params[0].nx1*param->wav_params[0].nx2;

  // Scale for the operator.
  sc = 1.0/sqrt((double)param->ndict);

  // Main loop.
  #pragma omp parallel default(none) \
  shared(i, nx, param, xoutr, xoutc, xinr, xinc) \
  private(d, tempr, tempc)
  {
    #pragma omp for
    for (i = 0; i < param->ndict; i++){
		
      d = i*nx;
      if (param->real == 1){
	       tempr = xoutr + d;
	       sopt_wavelet_gdfwtr(tempr, xinr, param->wav_params[i]);
      }
      else {
	       tempc = xoutc + d;
	       sopt_wavelet_gdfwtc(tempc, xinc, param->wav_params[i]);
      }
    }
  }

  // Scaling the result.
  if (param->real == 1){
    cblas_dscal(param->ndict*nx, sc, xoutr,1);
  }
  else {
    cblas_zdscal(param->ndict*nx, sc, (void*)xoutc,1);
  }
    
}

/*!
 * This function computes the synthesis operator for concatenation of bases.
 *
 * \warning Input data is overwritten on output.
 *
 * \param[out] out Output data, dimension nx.
 * \param[in,out] in Input data, dimension q*nx. Overwritten on output.
 * \param[in] data array containing the necessary parameters for the 
 *            operator.
 *            data[0] Parameter structure for the operator
 */
void sopt_sara_synthesisop(void *out, void *in, void **data){

  int i,d;
  int nx;
  double sc;
  complex double alpha;
  sopt_sara_param *param;
  complex double *tempc;
  double *tempr;
  complex double *xinc;
  double *xinr;
  complex double *xoutc;
  double *xoutr;

  // Cast input pointer for the data.
  param = (sopt_sara_param*)data[0];
  if (param->real == 1){
    xinr = (double*)in;
    xoutr = (double*)out;
  }
  else {
    xinc = (complex double*)in;
    xoutc = (complex double*)out;
  }

  // Signal dimension.
  nx = param->wav_params[0].nx1*param->wav_params[0].nx2;

  // Scale for the operator.
  sc = 1.0/sqrt((double)param->ndict);
    
  // Initialize solution to zero.
  if (param->real == 1){
    for (d = 0; d < nx; d++)
      *(xoutr + d) = 0.0;
  }
  else {
    alpha = 0.0 + I*0.0;
    for (d = 0; d < nx; d++)
      *(xoutc + d) = alpha;
  }

  // Main loop.
  #pragma omp parallel default(none) \
  shared(i, nx, param, xoutr, xoutc, xinr, xinc) \
  private(d, tempr, tempc)
  {
    #pragma omp for
    for (i = 0; i < param->ndict; i++){
		
      d = i*nx;
      if (param->real == 1){		
	       tempr = xinr + d;
	       sopt_wavelet_gdiwtr(tempr, tempr, param->wav_params[i]);	
      }
      else {			
	       tempc = xinc + d;
	       sopt_wavelet_gdiwtc(tempc, tempc, param->wav_params[i]);
      }

    }
  }

  // Reduce data by summing over contribution from each dictionary
  // and scaling it.
  // (This could be openmp reduced in Fortran but not supported in C.)
  alpha = sc + I*0.0;
  for (i = 0; i < param->ndict; i++){
    d = i*nx;	
    if (param->real == 1) {
      tempr = xinr + d;
      cblas_daxpy(nx, sc, tempr, 1, xoutr, 1);
    }
    else {
      tempc = xinc + d;
      cblas_zaxpy(nx, (void*)&alpha, (void*)tempc, 1, (void*)xoutc, 1);
    }
  }

}
