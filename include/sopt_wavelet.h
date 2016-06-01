//
//  sopt_wavelet.h
//  
//
//  Created by Rafael Carrillo on 10/17/12.
//
//

#ifndef sopt_wavelet
#define sopt_wavelet
#include "sopt_config.h"
#include "sopt_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!  
 * Options for the basis to be included in the
 * SARA operator.
 * As for now we support the Dirac basis and the first
 * 10 Daubechies orthornormal wavelet basis.
 */
typedef enum {
  
  SOPT_WAVELET_Dirac,
  SOPT_WAVELET_DB1,
  SOPT_WAVELET_DB2,
  SOPT_WAVELET_DB3,
  SOPT_WAVELET_DB4,
  SOPT_WAVELET_DB5,
  SOPT_WAVELET_DB6,
  SOPT_WAVELET_DB7,
  SOPT_WAVELET_DB8,
  SOPT_WAVELET_DB9,
  SOPT_WAVELET_DB10,

} sopt_wavelet_type;


/*!  
 * General parameters for the wavelet decomposition.
 */
typedef struct {
	/*! Number of elements in the first dimension.*/
	int nx1;
    /*! Number of elements in the second dimension.*/
	int nx2;
	/*! Wavelet type.*/	
	sopt_wavelet_type type;
	/*! Wavelet filter.*/	
	double *h;
	/*! Size of wavelet filter.*/
	int h_size;
	/*! Decomposition level.*/
	int nb_levels;
	
} sopt_wavelet_param;



void sopt_wavelet_convd(double *out_l, double *out_h, double *in, 
		                double *h0, double *h1, int length_sig, int length_filt);

void sopt_wavelet_convu(double *out, double *in_l, double *in_h, 
		                double *h0, double *h1, int length_sig, int length_filt);

void sopt_wavelet_gdfwtr(double *w, double *x, sopt_wavelet_param param_wav);

void sopt_wavelet_gdiwtr(double *x, double *w, sopt_wavelet_param param_wav);

void sopt_wavelet_gdfwtc(sopt_complex_double *w, sopt_complex_double *x, sopt_wavelet_param param_wav);

void sopt_wavelet_gdiwtc(sopt_complex_double *x, sopt_complex_double *w, sopt_wavelet_param param_wav);

void sopt_wavelet_initwav(sopt_wavelet_param *param);


#ifdef __cplusplus
}
#endif
#endif
