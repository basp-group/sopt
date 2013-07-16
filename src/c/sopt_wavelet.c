//
//  sopt_wavelet.c
//  
//
//  Created by Rafael Carrillo on 10/17/12.
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

#define max(A,B) (A > B ? A : B)

/*!
 * Convolve the signal "in" with "h0" and "h1".
 *
 * \param[out] out_l Low-pass filtered signal. 
 * \param[out] out_h High-pass filtered signal. 
 * \param[in] in Input data.
 * \param[in] h0 Aproximation filter coefficients.
 * \param[in] h1 Detail filter coefficients.
 * \param[in] length_sig Input signal length.
 * \param[in] length_filt Filter coefficients length.
 */ 
void sopt_wavelet_convd(double *out_l, double *out_h, double *in, 
		   double *h0, double *h1, int length_sig, int length_filt)
{
	int i,j;
	
	for(i=0; i<length_sig; i++)
	{
		*(out_l+i) = 0; *(out_h+i) = 0;
		for(j=0; j<length_filt; j++)
		{
			if(length_sig<=(i+j))
			{
				*(out_l+i) += *(in+i+j-length_sig) * *(h0+length_filt-j-1);
				*(out_h+i) += *(in+i+j-length_sig) * *(h1+length_filt-j-1);
			}
			else
			{
				*(out_l+i) += *(in+i+j) * *(h0+length_filt-j-1);
				*(out_h+i) += *(in+i+j) * *(h1+length_filt-j-1);
			}
		}
	}	
}

/*!
 * Convolve signals "in_l" and "in_h" with "h0" and
 * "h1" respectively, and add them in out.
 *
 * \param[out] out Output signal.
 * \param[in] in_l Low-pass input signal. 
 * \param[in] in_h High-pass input signal. 
 * \param[in] h0 Aproximation filter coefficients.
 * \param[in] h1 Detail filter coefficients.
 * \param[in] length_sig Input signal length.
 * \param[in] length_filt Filter coefficients length.
 */  
void sopt_wavelet_convu(double *out, double *in_l, double *in_h, 
		                double *h0, double *h1, int length_sig, int length_filt)
{
	int i,j;
	
	for(i=0; i<length_sig; i++)
	{
		*(out+(i+length_filt-1)%length_sig) = 0;
		for(j=0; j<length_filt; j++)
		{
			if(length_sig<=(i+j))
			{
				*(out+(i+length_filt-1)%length_sig) += *(in_l+i+j-length_sig) * *(h0+length_filt-j-1) + 
					*(in_h+i+j-length_sig) * *(h1+length_filt-j-1);
			}	
			else
			{
				*(out+(i+length_filt-1)%length_sig) += *(in_l+i+j) * *(h0+length_filt-j-1) + 
					*(in_h+i+j) * *(h1+length_filt-j-1);
			}
				
		}
	}	
}


/*!
 * Compute wavelets coefficients of a real 2D signal
 * x stored in column major order.
 *
 * \param[out] w Output wavelet coefficients.
 * \param[in] x Input 2D signal. 
 * \param[in] param_wav Structure with the necessary parameters
 *            for the wavelet decomposition. 
 */   
void sopt_wavelet_gdfwtr(double *w, double *x, sopt_wavelet_param param_wav) 
{
	double *h0, *h1;
	double *in, *out_l, *out_h;
	int i, j, decomp_level, dim_max;
	
    if (param_wav.type == SOPT_WAVELET_Dirac){

    	i = param_wav.nx1*param_wav.nx2;
		cblas_dcopy(i, x, 1, w, 1);

    }
    else{

    	/* Allocate memory */
		dim_max = max(param_wav.nx1, param_wav.nx2);
		in= (double *) malloc(dim_max * sizeof(double));
		SOPT_ERROR_MEM_ALLOC_CHECK(in);
		out_l= (double *) malloc(dim_max * sizeof(double));
		SOPT_ERROR_MEM_ALLOC_CHECK(out_l);
		out_h= (double *) malloc(dim_max * sizeof(double));
		SOPT_ERROR_MEM_ALLOC_CHECK(out_h);
		h0 = (double *) malloc(param_wav.h_size * sizeof(double));
		SOPT_ERROR_MEM_ALLOC_CHECK(h0);
		h1 = (double *) malloc(param_wav.h_size * sizeof(double));
		SOPT_ERROR_MEM_ALLOC_CHECK(h1);
	
		/* Lowpass and highpass analysis (decomposition) filters */
		for (i=0; i<param_wav.h_size; i++)
		{
			*(h0+i) = *(param_wav.h+param_wav.h_size-i-1);
			*(h1+i) = *(param_wav.h+i);
		}
		for (i=0; i<param_wav.h_size; i+=2)
			*(h1+i) = -*(h1+i);
	
		/* Main loop */
		for (decomp_level = 0; decomp_level<param_wav.nb_levels; decomp_level++ )
		{
			/* Transform along rows */
			for (i=0; i<param_wav.nx1/pow(2, decomp_level); i++){
				/* Copy input */
				if(decomp_level==0){
					for(j=0; j<param_wav.nx2/pow(2, decomp_level); j++)
					*(in+j) = *(x+i+param_wav.nx1*j);
				}
				else{
					for(j=0; j<param_wav.nx2/pow(2, decomp_level); j++)
					*(in+j) = *(w+i+param_wav.nx1*j);
				}
			
				/* Perform filtering lowpass and highpass */ 
				sopt_wavelet_convd(out_l, out_h, in, h0, h1, param_wav.nx2/pow(2, decomp_level), param_wav.h_size);
				/* Downsample and copy output */ 
				for(j=0; j<param_wav.nx2/pow(2, decomp_level+1); j++)
				{
					*(w+i+param_wav.nx1*j) = *(out_l+2*j);
				}
				for(j=param_wav.nx2/pow(2, decomp_level+1); j<param_wav.nx2/pow(2, decomp_level); j++)
				{
					dim_max = param_wav.nx2/pow(2, decomp_level+1);
					*(w+i+param_wav.nx1*j) = *(out_h+2*(j-dim_max));
				}
			}
		
			/* Transform along columns*/
			for (j=0; j<param_wav.nx2/pow(2, decomp_level); j++){
				/* Copy input */
				for(i=0; i<param_wav.nx1/pow(2, decomp_level); i++)
				*(in+i) = *(w+i+param_wav.nx1*j);

				/* Perform filtering lowpass and highpass */ 
				sopt_wavelet_convd(out_l, out_h, in, h0, h1, param_wav.nx1/pow(2, decomp_level), param_wav.h_size);
				/* Downsample and copy output */ 
				for(i=0; i<param_wav.nx1/pow(2, decomp_level+1); i++)
				{
					*(w+i+param_wav.nx1*j) = *(out_l+2*i);
				}
				for(i=param_wav.nx1/pow(2, decomp_level+1); i<param_wav.nx1/pow(2, decomp_level); i++)
				{
					dim_max = param_wav.nx1/pow(2, decomp_level+1);
					*(w+i+param_wav.nx1*j) = *(out_h+2*(i-dim_max));
				}

			}
		}
	
		/* Free memory */
		free(in); free(out_h); free(out_l);
		free(h0); free(h1);
    }
}

/*!
 * Inverse wavelet transform (on real 2D signal).
 *
 * \param[out] x Output 2D signal. 
 * \param[in] w Input wavelet coefficients. 
 * \param[in] param_wav Structure with the necessary parameters
 *            for the wavelet decomposition. 
 */   
void sopt_wavelet_gdiwtr(double *x, double *w, sopt_wavelet_param param_wav) 
{
	double *h0, *h1;
	double *in_l, *in_h, *out;
	int i, j, decomp_level, dim_max;

	if (param_wav.type == SOPT_WAVELET_Dirac){

		i = param_wav.nx1*param_wav.nx2;
		cblas_dcopy(i, w, 1, x, 1);

    }
    else{

    	/* Allocate memory */
		dim_max = max(param_wav.nx1, param_wav.nx2);
		in_l= (double *) malloc(dim_max * sizeof(double));
		SOPT_ERROR_MEM_ALLOC_CHECK(in_l);
		in_h= (double *) malloc(dim_max * sizeof(double));
		SOPT_ERROR_MEM_ALLOC_CHECK(in_h);
		out= (double *) malloc(dim_max * sizeof(double));
		SOPT_ERROR_MEM_ALLOC_CHECK(out);
		h0 = (double *) malloc(param_wav.h_size * sizeof(double));
		SOPT_ERROR_MEM_ALLOC_CHECK(h0);
		h1 = (double *) malloc(param_wav.h_size * sizeof(double));
		SOPT_ERROR_MEM_ALLOC_CHECK(h1);
	
		/* Lowpass and highpass synthesis (reconstruction) filters */
		for (i=0; i<param_wav.h_size; i++)
		{
			*(h0+i) = *(param_wav.h+i);
			*(h1+i) = *(param_wav.h+param_wav.h_size-i-1);
		}
		for (i=1; i<param_wav.h_size; i+=2)
			*(h1+i) = -*(h1+i);
	
		/* Initialization - Copy all data in the output vector */
		cblas_dcopy(param_wav.nx1*param_wav.nx2, w, 1, x, 1);
	
		/* Main loop */
		for (decomp_level = param_wav.nb_levels; decomp_level>0; decomp_level--)
		{
			/* Transform along rows */
			for (i=0; i<param_wav.nx1/pow(2, decomp_level-1); i++){

				/* Copy inputs and upsample*/
				/* low frequency part */
				for (j=0; j<param_wav.nx2/pow(2, decomp_level); j++){

					*(in_l+2*j) = *(x+i+param_wav.nx1*j);
					if((2*j+1)<param_wav.nx2)
						*(in_l+2*j+1) = 0.0;
				}
				/* High frequency part */
				dim_max = param_wav.nx2/pow(2, decomp_level);
				for(j=param_wav.nx2/pow(2, decomp_level); j<param_wav.nx2/pow(2, decomp_level-1); j++){
					*(in_h+2*(j-dim_max)) = *(x+i+param_wav.nx1*j);
					if((2*(j-dim_max)+1)<param_wav.nx2)
						*(in_h+2*(j-dim_max)+1) = 0.0;
				}
				/* Perform filtering lowpass and highpass */
				sopt_wavelet_convu(out, in_l, in_h, h0, h1, param_wav.nx2/pow(2, decomp_level-1), param_wav.h_size);
				/* Copy output */
				for(j=0; j<param_wav.nx2/pow(2, decomp_level-1); j++)
					*(x+i+param_wav.nx1*j) = *(out+j);
			}

			/* Transform along columns */
			for (j=0; j<param_wav.nx2/pow(2, decomp_level-1); j++){

				/* Copy inputs and upsample */
				/* Low frequency part */
				for(i=0; i<param_wav.nx1/pow(2, decomp_level); i++)
				{
					*(in_l+2*i) = *(x+i+param_wav.nx1*j);
					if((2*i+1)<param_wav.nx1)
						*(in_l+2*i+1) = 0.0;
				}
				/* High frequency part */
				dim_max = param_wav.nx1/pow(2, decomp_level);
				for(i=param_wav.nx1/pow(2, decomp_level); i<param_wav.nx1/pow(2, decomp_level-1); i++)
				{
					*(in_h+2*(i-dim_max)) = *(x+i+param_wav.nx1*j);
					if((2*(i-dim_max)+1)<param_wav.nx1)
						*(in_h+2*(i-dim_max)+1) = 0.0;
				}
				/* Perform filtering lowpass and highpass */
				sopt_wavelet_convu(out, in_l, in_h, h0, h1, param_wav.nx1/pow(2, decomp_level-1), param_wav.h_size);
				/* Copy output */
				for(i=0; i<param_wav.nx1/pow(2, decomp_level-1); i++)
					*(x+i+param_wav.nx1*j) = *(out+i);

			}
		}
	
		/* Free memory */
		free(in_l); free(in_h); free(out);
		free(h0); free(h1);
    	
    }
}


/*!
 * Compute wavelets coefficients of a complex 2D signal
 * x stored in column major order.
 *
 * \param[out] w Output wavelet complex coefficients.
 * \param[in] x Input 2D complex signal. 
 * \param[in] param_wav Structure with the necessary parameters
 *            for the wavelet decomposition. 
 */    
void sopt_wavelet_gdfwtc(complex double *w, complex double *x, sopt_wavelet_param param_wav) 
{
	double *temp_in, *temp_wr, *temp_wi;
	int i;

	if (param_wav.type == SOPT_WAVELET_Dirac){

		i = param_wav.nx1*param_wav.nx2;
		cblas_zcopy(i, (void*)x, 1, (void*)w, 1);

    }
    else{
    	/* Allocate memory */
		temp_in = (double *) malloc(param_wav.nx1*param_wav.nx2 * sizeof(double));
		SOPT_ERROR_MEM_ALLOC_CHECK(temp_in);
		temp_wr = (double *) malloc(param_wav.nx1*param_wav.nx2 * sizeof(double));
		SOPT_ERROR_MEM_ALLOC_CHECK(temp_wr);
		temp_wi = (double *) malloc(param_wav.nx1*param_wav.nx2 * sizeof(double));
		SOPT_ERROR_MEM_ALLOC_CHECK(temp_wi);
	
		/* Copy real part and do wavelet decomposition */
		for (i=0; i < param_wav.nx1*param_wav.nx2; i++){
			*(temp_in+i) = creal(*(x+i));
		}
		sopt_wavelet_gdfwtr(temp_wr, temp_in, param_wav);
	
		/* Copy imaginary part and do wavelet decomposition */
		for (i=0; i < param_wav.nx1*param_wav.nx2; i++){
			*(temp_in+i) = cimag(*(x+i));
		}
		sopt_wavelet_gdfwtr(temp_wi, temp_in, param_wav);

		for (i=0; i < param_wav.nx1*param_wav.nx2; i++){
			*(w+i) = *(temp_wr+i) + I*(*(temp_wi+i));
		}
	
		/* Free memory */
		free(temp_wr);
		free(temp_wi);
		free(temp_in);	
    }
}


/*!
 * Inverse wavelet transform (on complex 2D signal).
 *
 * \param[out] x Output 2D complex signal. 
 * \param[in] w Input complex wavelet coefficients. 
 * \param[in] param_wav Structure with the necessary parameters
 *            for the wavelet decomposition. 
 */   

void sopt_wavelet_gdiwtc(complex double *x, complex double *w, sopt_wavelet_param param_wav) 
{
	double *temp_in, *temp_wr, *temp_wi;
	int i;

	if (param_wav.type == SOPT_WAVELET_Dirac){

        i = param_wav.nx1*param_wav.nx2;
		cblas_zcopy(i, (void*)w, 1, (void*)x, 1);

    }
    else{
    	/* Allocate memory */
		temp_in = (double *) malloc(param_wav.nx1*param_wav.nx2 * sizeof(double));
		SOPT_ERROR_MEM_ALLOC_CHECK(temp_in);
		temp_wr = (double *) malloc(param_wav.nx1*param_wav.nx2 * sizeof(double));
		SOPT_ERROR_MEM_ALLOC_CHECK(temp_wr);
		temp_wi = (double *) malloc(param_wav.nx1*param_wav.nx2 * sizeof(double));
		SOPT_ERROR_MEM_ALLOC_CHECK(temp_wi);
	
		/* Copy real part and do inverse wavelet transform */
		for (i=0; i < param_wav.nx1*param_wav.nx2; i++){
			*(temp_in+i) = creal(*(w+i));
		}
		sopt_wavelet_gdiwtr(temp_wr, temp_in, param_wav);
	
		/* Copy imaginary part and do inverse wavelet transform */
		for (i=0; i < param_wav.nx1*param_wav.nx2; i++){
			*(temp_in+i) = cimag(*(w+i));
		}
		sopt_wavelet_gdiwtr(temp_wi, temp_in, param_wav);

		for (i=0; i < param_wav.nx1*param_wav.nx2; i++){
			*(x+i) = *(temp_wr+i) + I*(*(temp_wi+i));
		}
	
		/* Free memory */
		free(temp_wr);
		free(temp_wi);
		free(temp_in);	
    }
	
	
}

/*!
 * This function initializes the filter 
 * coefficients for the wavelet transform.
 *
 * \param[out] param Structure with the necessary parameters
 *                for the wavelet decomposition. 
 */

void sopt_wavelet_initwav(sopt_wavelet_param *param){
      
    
    double hfilt1[2] = {0.707106781186548, 0.707106781186548};
    double hfilt2[4] = {0.482962913144690, 0.836516303737469, 0.224143868041857, -0.129409522550921};
    double hfilt3[6] = {0.332670552950957, 0.806891509313339, 0.459877502119331, -0.135011020010391, -0.085441273882241, 0.035226291882101};
    double hfilt4[8] = {0.230377813308855, 0.714846570552542, 0.630880767929590, -0.027983769416984, -0.187034811718881, 0.030841381835987, 0.032883011666983, -0.010597401784997};
    double hfilt5[10] = {0.160102397974125, 0.603829269797473, 0.724308528438574, 0.138428145901103, -0.242294887066190, -0.032244869585030, 0.077571493840065, -0.006241490213012, -0.012580751999016, 0.003335725285002};
    double hfilt6[12] = {0.111540743350080, 0.494623890398385, 0.751133908021578, 0.315250351709243, -0.226264693965169, -0.129766867567096, 0.097501605587079, 0.027522865530016, -0.031582039318031, 0.000553842200994, 0.004777257511011, -0.001077301084996};
    double hfilt7[14] = {0.077852054085062, 0.396539319482306, 0.729132090846555, 0.469782287405359, -0.143906003929106, -0.224036184994166, 0.071309219267050, 0.080612609151066, -0.038029936935035, -0.016574541631016, 0.012550998556014, 0.000429577973005, -0.001801640704000, 0.000353713800001};
    double hfilt8[16] = {0.054415842243082, 0.312871590914466, 0.675630736298013, 0.585354683654869, -0.015829105256024, -0.284015542962428, 0.000472484573998, 0.128747426620186, -0.017369301002022, -0.044088253931065, 0.013981027917016, 0.008746094047016, -0.004870352993011, -0.000391740372996, 0.000675449405999, -0.000117476784002};
    double hfilt9[18] = {0.038077947363167, 0.243834674637667, 0.604823123676779, 0.657288078036639, 0.133197385822089, -0.293273783272587, -0.096840783220879, 0.148540749334760, 0.030725681478323, -0.067632829059524, 0.000250947114992, 0.022361662123515, -0.004723204757895, -0.004281503681905, 0.001847646882961, 0.000230385763995, -0.000251963188998, 0.000039347319995};
    double hfilt10[20] = {0.026670057900951, 0.188176800077621, 0.527201188930920, 0.688459039452592, 0.281172343660426, -0.249846424326489, -0.195946274376597, 0.127369340335743, 0.093057364603807, -0.071394147165861, -0.029457536821946, 0.033212674058933, 0.003606553566988, -0.010733175482980, 0.001395351746994, 0.001992405294991, -0.000685856695005, -0.000116466854994, 0.000093588670001, -0.000013264203002};

   

    switch (param->type){

    	case SOPT_WAVELET_Dirac:
    	     param->h_size = 1;
    	     param->h = (double*)malloc(param->h_size * sizeof(double));
             SOPT_ERROR_MEM_ALLOC_CHECK(param->h);
             param->h[0] = 1.0;
    	     break;
    	case SOPT_WAVELET_DB1:
    	     param->h_size = 2;
    	     param->h = (double*)malloc(param->h_size * sizeof(double));
             SOPT_ERROR_MEM_ALLOC_CHECK(param->h);
             cblas_dcopy(param->h_size, hfilt1, 1, param->h, 1);
    	     break;
    	case SOPT_WAVELET_DB2:
    	     param->h_size = 4;
    	     param->h = (double*)malloc(param->h_size * sizeof(double));
             SOPT_ERROR_MEM_ALLOC_CHECK(param->h);
             cblas_dcopy(param->h_size, hfilt2, 1, param->h, 1);
    	     break;
    	case SOPT_WAVELET_DB3:
    	     param->h_size = 6;
    	     param->h = (double*)malloc(param->h_size * sizeof(double));
             SOPT_ERROR_MEM_ALLOC_CHECK(param->h);
             cblas_dcopy(param->h_size, hfilt3, 1, param->h, 1);
    	     break;
    	case SOPT_WAVELET_DB4:
    	     param->h_size = 8;
    	     param->h = (double*)malloc(param->h_size * sizeof(double));
             SOPT_ERROR_MEM_ALLOC_CHECK(param->h);
             cblas_dcopy(param->h_size, hfilt4, 1, param->h, 1);
    	     break;
    	case SOPT_WAVELET_DB5:
    	     param->h_size = 10;
    	     param->h = (double*)malloc(param->h_size * sizeof(double));
             SOPT_ERROR_MEM_ALLOC_CHECK(param->h);
             cblas_dcopy(param->h_size, hfilt5, 1, param->h, 1);
    	     break;
    	case SOPT_WAVELET_DB6:
    	     param->h_size = 12;
    	     param->h = (double*)malloc(param->h_size * sizeof(double));
             SOPT_ERROR_MEM_ALLOC_CHECK(param->h);
             cblas_dcopy(param->h_size, hfilt6, 1, param->h, 1);
    	     break;
    	case SOPT_WAVELET_DB7:
    	     param->h_size = 14;
    	     param->h = (double*)malloc(param->h_size * sizeof(double));
             SOPT_ERROR_MEM_ALLOC_CHECK(param->h);
             cblas_dcopy(param->h_size, hfilt7, 1, param->h, 1);
    	     break;
    	case SOPT_WAVELET_DB8:
    	     param->h_size = 16;
    	     param->h = (double*)malloc(param->h_size * sizeof(double));
             SOPT_ERROR_MEM_ALLOC_CHECK(param->h);
             cblas_dcopy(param->h_size, hfilt8, 1, param->h, 1);
    	     break;
    	case SOPT_WAVELET_DB9:
    	     param->h_size = 18;
    	     param->h = (double*)malloc(param->h_size * sizeof(double));
             SOPT_ERROR_MEM_ALLOC_CHECK(param->h);
             cblas_dcopy(param->h_size, hfilt9, 1, param->h, 1);
    	     break;
    	case SOPT_WAVELET_DB10:
    	     param->h_size = 20;
    	     param->h = (double*)malloc(param->h_size * sizeof(double));
             SOPT_ERROR_MEM_ALLOC_CHECK(param->h);
             cblas_dcopy(param->h_size, hfilt10, 1, param->h, 1);
    	     break;

    }

}





