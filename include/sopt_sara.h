//
//  sopt_sara.h
//  
//
//  Created by Rafael Carrillo on 10/26/12.
//
//

#ifndef sopt_sara
#define sopt_sara
#include "sopt_config.h"
#include "sopt_wavelet.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!  
 * Data structure containing the parameters for the
 * SARA dictionary.
 */
typedef struct {
	/*! Number of bases in the dictionary.*/
	int ndict;	
	/*! Flag for real or complex data. 1 real, 0 complex.*/
	int real;
    /*! List with parameters  for the wavelet transforms.*/
	sopt_wavelet_param *wav_params;
	
} sopt_sara_param;

void sopt_sara_initop(sopt_sara_param *param, int nx1, int nx2, int nb_levels, 
	                  sopt_wavelet_type *dict_types);
void sopt_sara_free(sopt_sara_param *param);
void sopt_sara_analysisop(void *out, void *in, void **data);
void sopt_sara_synthesisop(void *out, void *in, void **data);


#ifdef __cplusplus
}
#endif
#endif
