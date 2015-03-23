/*
 *  sopt_demo2.c
 *  
 *
 *  Created by Rafael Carrillo on 8/28/12.
 *  Copyright 2012 EPFL. All rights reserved.
 *
 */
#include "sopt_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <fftw3.h> 
#include <math.h>
#include <assert.h>
#include <time.h>
#ifdef _OPENMP 
  #include <omp.h>
#endif
#include SOPT_BLAS_H
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

    #ifdef _OPENMP
        double start, stop;
    #else
        clock_t start, stop;
    #endif
    double t = 0.0;

    //Structures for the different operators
    sopt_wavelet_type *dict_types;
    sopt_prox_l1param param;
    sopt_l1_param param2;
    sopt_prox_l2bparam param3;
    sopt_l1_rwparam param4;
    sopt_sara_param param5;
    sopt_prox_tvparam param7;
    sopt_tv_param param8;

    void *datas[1];
    void *datafwd[5];
    void *dataadj[5];
    
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


    double scale = 1.0/255.0, offset = 0.0;
    int fail = 0;

    printf("**********************\n");
    printf("Fourier sampling demo\n");
    printf("**********************\n");
  
    //Read input image
    fail = sopt_image_tiff_read(&xin, &dim2, &dim1, "images/cameraman256.tiff", scale, offset);
    if(fail == 1)
      SOPT_ERROR_GENERIC("Error reading image");
    
    
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


    
    //Measurement operator initialization
    //Structure for the measurement operator
   
    seedmat = 31;

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
    fail = sopt_image_tiff_write(xout, dim2, dim1, "images/cameraFS_Dirty_temp.tiff", 1.0/scale, offset);
    if(fail == 1)
      SOPT_ERROR_GENERIC("Error writing image");

    printf("**********************\n");
    printf("TV reconstruction\n");
    printf("**********************\n");


    //Structure for the TV prox
    param7.verbose = 1;
    param7.max_iter = 50;
    param7.rel_obj = 0.0001;

    //Structure for the l2 ball prox
    param3.verbose = 1;
    param3.max_iter = 50;
    param3.tol = 0.001;
    param3.nu = 1.0;
    param3.tight = 1;
    param3.pos = 1;
    param3.real = 0;

    //Structure for the TV solver    
    param8.verbose = 2;
    param8.max_iter = 200;
    param8.gamma = 0.1;
    param8.rel_obj = 0.0005;
    param8.epsilon = sqrt(Ny + 2*sqrt(Ny))*sigma;
    param8.real_out = 1;
    param8.real_meas = 0;
    param8.paramtv = param7;
    param8.paraml2b = param3;

    //Initial solution
    for (i=0; i < Nx; i++) {
        xout[i] = 0.0;
    }

    
    assert((start = clock())!=-1);
    sopt_tv_solver((void*)xout, dim1, dim2,
                   &sopt_meas_fsamp_c2c,
                   datafwd,
                   &sopt_meas_fsampadj_c2c,
                   dataadj,
                   (void*)y, Ny, param8);
    stop = clock();
    t = (double) (stop-start)/CLOCKS_PER_SEC;
    
    printf("Time TV: %f \n\n", t); 

    //SNR
    for (i=0; i < Nx; i++) {
        error[i] = xout[i] - xin[i];
    }
    mse = cblas_dnrm2(Nx, error, 1);
    a = cblas_dnrm2(Nx, xin, 1);
    mse = 20.0*log10(a/mse);
    printf("SNR: %f dB\n\n", mse);
    
    //Write output image
    fail = sopt_image_tiff_write(xout, dim2, dim1, "images/cameraFS_TV_temp.tiff", 1.0/scale, offset);
    if(fail == 1)
      SOPT_ERROR_GENERIC("Error writing image");
    


    printf("**********************\n");
    printf("L1 reconstruction\n");
    printf("**********************\n");

    //Sparsity averaging structure initialization
    //Analysis operator is the concatenation of the first eight Daubechies 
    //wavelet basis

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

     //Structure for the L1 prox
    param.verbose = 1;
    param.max_iter = 10;
    param.rel_obj = 0.01;
    param.nu = 1;
    param.tight = 0;
    param.pos = 0;
    
    //Structure for the L1 solver    
    param2.verbose = 2;
    param2.max_iter = 200;
    param2.gamma = 0.8;
    param2.rel_obj = 0.0005;
    param2.epsilon = sqrt(Ny + 2*sqrt(Ny))*sigma;
    param2.real_out = 1;
    param2.real_meas = 0;
    param2.paraml1 = param;
    param2.paraml2b = param3;
    
   
    //Weights
    for (i=0; i < Nr; i++) {
        w[i] = 1.0;
    }
    
    //Initial solution
    for (i=0; i < Nx; i++) {
        xout[i] = 0.0;
    }

    #ifdef _OPENMP
        start = omp_get_wtime();
    #else
        assert((start = clock())!=-1);
    #endif
        sopt_l1_solver((void*)xout, Nx,
                    &sopt_meas_fsamp_c2c,
                    datafwd,
                    &sopt_meas_fsampadj_c2c,
                    dataadj,
                    &sopt_sara_synthesisop,
                    datas,
                    &sopt_sara_analysisop,
                    datas,
                    Nr,
                    (void*)y, Ny, w, param2);
    #ifdef _OPENMP
        stop = omp_get_wtime();
        t = stop - start;
    #else    
        stop = clock();
        t = (double) (stop-start)/CLOCKS_PER_SEC;
    #endif    

    printf("Time SA L1: %f \n\n", t); 

    //SNR
    for (i=0; i < Nx; i++) {
        error[i] = xout[i] - xin[i];
    }
    mse = cblas_dnrm2(Nx, error, 1);
    a = cblas_dnrm2(Nx, xin, 1);
    mse = 20.0*log10(a/mse);
    printf("SNR: %f dB\n\n", mse);

    //Write output image
    fail = sopt_image_tiff_write(xout, dim2, dim1, "images/cameraFS_L1_temp.tiff", 1.0/scale, offset);
    if(fail == 1)
      SOPT_ERROR_GENERIC("Error writing image");


    printf("**********************\n");
    printf("SARA reconstruction\n");
    printf("**********************\n");
    
    
    //Structure for the RWL1 solver    
    param4.verbose = 2;
    param4.max_iter = 5;
    param4.rel_var = 0.001;
    param4.sigma = sigma*sqrt(Ny/(8*Nx));
    param4.init_sol = 1;

    #ifdef _OPENMP
        start = omp_get_wtime();
    #else
        assert((start = clock())!=-1);
    #endif    
        sopt_l1_rwmin((void*)xout, Nx,
                    &sopt_meas_fsamp_c2c,
                    datafwd,
                    &sopt_meas_fsampadj_c2c,
                    dataadj,
                    &sopt_sara_synthesisop,
                    datas,
                    &sopt_sara_analysisop,
                    datas,
                    Nr,
                    (void*)y, Ny, param2, param4);
    #ifdef _OPENMP
        stop = omp_get_wtime();
        t = stop - start;
    #else 
        stop = clock();
        t = (double) (stop-start)/CLOCKS_PER_SEC;
    #endif
    
    printf("Time SARA: %f \n\n", t); 

    //SNR
    for (i=0; i < Nx; i++) {
        error[i] = xout[i] - xin[i];
    }
    mse = cblas_dnrm2(Nx, error, 1);
    a = cblas_dnrm2(Nx, xin, 1);
    mse = 20.0*log10(a/mse);
    printf("SNR: %f dB\n\n", mse);

    fail = sopt_image_tiff_write(xout, dim2, dim1, "images/cameraFS_SARA_temp.tiff", 1.0/scale, offset);
    if(fail == 1)
      SOPT_ERROR_GENERIC("Error writing image");

    
    free(xin);
    free(xout);
    free(w);
    free(y);
    free(y0);
    free(noise);
    free(error);
    free(xinc);

    free(dict_types);

    free(perm);
    free(permc);
    free(fft_temp1);
    free(fft_temp2);
    fftw_destroy_plan(planfwd);
    fftw_destroy_plan(planadj);

    sopt_sara_free(&param5);
        
    return 0;
}
