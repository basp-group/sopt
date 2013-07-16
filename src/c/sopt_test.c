/*
 *  test.c
 *  
 *
 *  Created by Rafael Carrillo on 8/28/12.
 *  Copyright 2012 EPFL. All rights reserved.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <fftw3.h> 
#include <math.h>
#ifdef __APPLE__
  #include <Accelerate/Accelerate.h>
#elif __unix__
  #include <cblas.h>
#else
  #include <cblas.h>
#endif
#include "sopt_utility.h"
#include "sopt_error.h"
#include "sopt_prox.h" 
#include "sopt_ran.h"
#include "sopt_meas.h"
#include "sopt_tv.h"
#include "sopt_l1.h"
#include "sopt_wavelet.h"
#include "sopt_sara.h"
#include "sopt_sparsemat.h"
#include "sopt_image.h"


int main(int argc, char *argv[]) {
  
    int dim1;
    int dim2;
    int Nx;
    int Nr;
    int Ny;
    int seedn=54;
    int seedmat;
    int flag;
    double sigma;
    double a;
    double mse;
    double snr;
    
    
    int i;


    //Structures for the different operators
    sopt_wavelet_type *dict_types;
    sopt_sara_param param5;
    void *datas[3];
    void *datafwd[5];
    void *dataadj[5];
    
    double *temp1;
    int *perm;
    int *permc;
    complex double *fft_temp1;
    complex double *fft_temp2;
    fftw_plan planfwd;
    fftw_plan planadj;
    

    double *xin;
    double *xout;
    double *error;
    double *w;
    complex double *y0;
    complex double *y;
    complex double *noise;
    complex double *xinc;
    complex double *xoutc;


    double scale = 1.0/255.0, offset = 0.0;
    int fail = 0;

    printf("**********************\n");
    printf("Image module test\n");
    printf("**********************\n");
  
    //Read input image
    fail = spot_image_tiff_read(&xin, &dim2, &dim1, "images/cameraman256.tiff", scale, offset);
    if(fail == 1)
      SOPT_ERROR_GENERIC("Error reading image");
    else
         printf("Image module test past\n"); 

    
    
    Nx=dim1*dim2;
    Nr=8*Nx;
    Ny=(int)(0.3*Nx);
    
    
    xout = (double*)malloc((Nx) * sizeof(double));
    SOPT_ERROR_MEM_ALLOC_CHECK(xout);
    error = (double*)malloc((Nx) * sizeof(double));
    SOPT_ERROR_MEM_ALLOC_CHECK(error);
    y = (complex double*)malloc((Ny) * sizeof(complex double));
    SOPT_ERROR_MEM_ALLOC_CHECK(y);
    y0 = (complex double*)malloc((Ny) * sizeof(complex double));
    SOPT_ERROR_MEM_ALLOC_CHECK(y0);
    noise = (complex double*)malloc((Ny) * sizeof(complex double));
    SOPT_ERROR_MEM_ALLOC_CHECK(noise);
    w = (double*)malloc((Nr) * sizeof(double));
    SOPT_ERROR_MEM_ALLOC_CHECK(w);
    xinc = (complex double*)malloc((Nx) * sizeof(complex double));
    SOPT_ERROR_MEM_ALLOC_CHECK(xinc);
    xoutc = (complex double*)malloc((Nx) * sizeof(complex double));
    SOPT_ERROR_MEM_ALLOC_CHECK(xoutc);

    printf("**********************\n");
    printf("Wavelet module test\n");
    printf("**********************\n");

    //SARA structure initialization
    
    temp1 = (double*)malloc(Nx * sizeof(double));
    SOPT_ERROR_MEM_ALLOC_CHECK(temp1);
    

    param5.ndict = 8;
    param5.real = 1;

    dict_types = malloc(param5.ndict * sizeof(sopt_wavelet_type));
    SOPT_ERROR_MEM_ALLOC_CHECK(dict_types);


    dict_types[0] = SOPT_WAVELET_DB1;
    dict_types[1] = SOPT_WAVELET_DB2;
    dict_types[2] = SOPT_WAVELET_DB3;
    dict_types[3] = SOPT_WAVELET_DB4;
    dict_types[4] = SOPT_WAVELET_DB5;
    dict_types[5] = SOPT_WAVELET_DB6;
    dict_types[6] = SOPT_WAVELET_DB7;
    dict_types[7] = SOPT_WAVELET_DB8;


    sopt_sara_initop(&param5, dim1, dim2, 4, dict_types);

    datas[0] = (void*)&param5;
    datas[1] = (void*)temp1;

    double *image;

    image = (double*)malloc((Nr) * sizeof(double));
    SOPT_ERROR_MEM_ALLOC_CHECK(image);

    sopt_sara_analysisop((void*)image, (void*)xin, datas);

    fail = spot_image_tiff_write(image, dim2, param5.ndict*dim1, "images/wavedec.tiff", 1.0/scale, offset);
    if(fail == 1)
      SOPT_ERROR_GENERIC("Error writing image");

    sopt_sara_synthesisop((void*)xout, (void*)image, datas);

    //Check
    for (i=0; i < Nx; i++) {
        error[i] = xout[i] - xin[i];
    }
    mse = cblas_dnrm2(Nx, error, 1);
    
    printf("MSE of synthesized image: %f\n", mse);

    printf("Measurement module test past\n"); 
    
    
    printf("**********************\n");
    printf("Measurement module test\n");
    printf("**********************\n");

    //Fourier measurement operator initialization
    //Structure for the measurement operator
   
    seedmat = 634;

    perm = (int*)malloc(Ny * sizeof(int));
    SOPT_ERROR_MEM_ALLOC_CHECK(perm);
    
    flag = sopt_ran_knuthshuffle(perm, Ny, Nx, seedmat);
    //Check random permutation.
    if (flag!=0) {
        SOPT_ERROR_GENERIC("Could not generate the permutation");
    }

    permc = (int*)malloc(Ny * sizeof(int));
    SOPT_ERROR_MEM_ALLOC_CHECK(permc);

    for (i =0; i < Ny; i++){
        permc[i] = perm[i];
    }

    fft_temp1 = (complex double*)malloc((Nx) * sizeof(complex double));
    SOPT_ERROR_MEM_ALLOC_CHECK(fft_temp1);
    fft_temp2 = (complex double*)malloc((Nx) * sizeof(complex double));
    SOPT_ERROR_MEM_ALLOC_CHECK(fft_temp2);

    planfwd = fftw_plan_dft_2d(dim2, dim1, 
              xinc, fft_temp1, 
              FFTW_FORWARD, FFTW_MEASURE);

    planadj = fftw_plan_dft_2d(dim2, dim1, 
              fft_temp2, xinc, 
              FFTW_BACKWARD, FFTW_MEASURE);

    

    //Data cast for the measurement operator
    datafwd[0] = (void*)&planfwd;
    datafwd[1] = (void*)perm;
    datafwd[2] = (void*)&Ny;
    datafwd[3] = (void*)&Nx;
    datafwd[4] = (void*)fft_temp1;

    dataadj[0] = (void*)&planadj;
    dataadj[1] = (void*)permc;
    dataadj[2] = (void*)&Ny;
    dataadj[3] = (void*)&Nx;
    dataadj[4] = (void*)fft_temp2;

    for (i = 0; i < Nx; i++){
        xinc[i] = xin[i] + 0.0*I;
    }
    
    //Measurement operator
    sopt_meas_fsamp_c2c((void*)y0, (void*)xinc, datafwd);
    
    //Noise realization
    //Input snr
    snr = 30.0;
    a = cblas_dznrm2(Ny, (void*)y0, 1)/sqrt(Ny);
    sigma = a*pow(10.0,-(snr/20.0));
    
    for (i=0; i < Ny; i++) {
        noise[i] = (sopt_ran_gasdev2(seedn) + sopt_ran_gasdev2(seedn)*I)*(sigma/sqrt(2));
        y[i] = y0[i] + noise[i];
    }


    //Backprojected image
    sopt_meas_fsampadj_c2c((void*)xinc, (void*)y, dataadj);

    for (i=0; i < Nx; i++) {
        xout[i] = creal(xinc[i]);
        if (xout[i] < 0.0)
            xout[i] = 0.0;
    }

    //SNR
    for (i=0; i < Nx; i++) {
        error[i] = xout[i] - xin[i];
    }
    mse = cblas_dnrm2(Nx, error, 1);
    a = cblas_dnrm2(Nx, xin, 1);
    mse = 20.0*log10(a/mse);
    printf("Backprojected image SNR: %f dB\n\n", mse);

    //Write Dirty image
    fail = spot_image_tiff_write(xout, dim2, dim1, "images/cameraFS_Dirty.tiff", 1.0/scale, offset);
    if(fail == 1)
      SOPT_ERROR_GENERIC("Error writing image");
    else
        printf("Measurement module test past\n"); 

   

    
    free(xin);
    free(xout);
    free(w);
    free(y);
    free(y0);
    free(noise);
    free(error);
    free(xinc);
    free(xoutc);

    free(temp1);
    free(dict_types);

    free(perm);
    free(permc);
    free(fft_temp1);
    free(fft_temp2);
    fftw_destroy_plan(planfwd);
    fftw_destroy_plan(planadj);

    sopt_sara_free(&param5);
    free(image);
        
    return 0;
}
