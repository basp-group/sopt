/*
 *  sopt_demo1.c
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
    int Ny;
    int Nr;
    int seedn=54;
    int seedmat;
    int flag;
    double sigma;
    double a;
    double mse;
    double snr;
    
    
    int i;

    clock_t start, stop;
    double t = 0.0;

    //Structures for the different operators
    
    sopt_meas_usparam param1;
    sopt_prox_tvparam param2;
    sopt_prox_l2bparam param3;
    sopt_tv_param param4;
    sopt_wavelet_type *dict_types;
    sopt_sara_param param5;
    sopt_prox_l1param param6;
    sopt_l1_param param7;
    

    void *datam[1];
    void *datas[1];
    
    
    double *xin;
    double *xout;
    double *error;
    double *y0;
    double *y;
    double *noise;
    double *w;
    

    double scale = 1.0/255.0, offset = 0.0;
    int fail = 0;

    printf("**********************\n");
    printf("Inpainting demo\n");
    printf("**********************\n");
  
    //Read input image
    fail = spot_image_tiff_read(&xin, &dim2, &dim1, "images/cameraman256.tiff", scale, offset);
    if(fail == 1)
      SOPT_ERROR_GENERIC("Error reading image");
    
    
    Nx=dim1*dim2;
    Ny=(int)(0.5*Nx);
    Nr=Nx;
    
    
    xout = (double*)malloc((Nx) * sizeof(double));
    SOPT_ERROR_MEM_ALLOC_CHECK(xout);
    error = (double*)malloc((Nx) * sizeof(double));
    SOPT_ERROR_MEM_ALLOC_CHECK(error);
    y = (double*)malloc((Ny) * sizeof(double));
    SOPT_ERROR_MEM_ALLOC_CHECK(y);
    y0 = (double*)malloc((Ny) * sizeof(double));
    SOPT_ERROR_MEM_ALLOC_CHECK(y0);
    noise = (double*)malloc((Ny) * sizeof(double));
    SOPT_ERROR_MEM_ALLOC_CHECK(noise);
    w = (double*)malloc((Nr) * sizeof(double));
    SOPT_ERROR_MEM_ALLOC_CHECK(w);


    //Measurement operator initialization
    //Structure for the measurement operator
    param1.nmeas = Ny;
    param1.nx = Nx;
    param1.real_flag = 1;


    seedmat = 234;

    param1.perm = (int*)malloc( param1.nmeas * sizeof(int));
    SOPT_ERROR_MEM_ALLOC_CHECK(param1.perm);
    
    flag = sopt_ran_knuthshuffle(param1.perm, param1.nmeas, param1.nx, seedmat);
    //Check random permutation.
    if (flag!=0) {
        SOPT_ERROR_GENERIC("Could not generate the permutation");
    }

    //Data cast for the measurement operator
    datam[0] = (void*)&param1;
    
    //Measurement operator
    sopt_meas_urandsamp((void*)y0, (void*)xin, datam);
    
    //Noise realization
    //Input snr
    snr = 30.0;
    a = cblas_dnrm2(Ny, y0, 1)/sqrt(Ny);
    sigma = a*pow(10.0,-(snr/20.0));
    
    for (i=0; i < Ny; i++) {
        noise[i] = sigma*sopt_ran_gasdev2(seedn);
        y[i] = y0[i] + noise[i];
    }

    
    //Backprojected image
    sopt_meas_urandsampadj((void*)xout, (void*)y, datam);

    //Write Dirty image
    fail = spot_image_tiff_write(xout, dim2, dim1, "images/cameraIP_Dirty_temp.tiff", 1.0/scale, offset);
    if(fail == 1)
      SOPT_ERROR_GENERIC("Error writing image");

    //SNR
    for (i=0; i < Nx; i++) {
        error[i] = xout[i] - xin[i];
    }
    mse = cblas_dnrm2(Nx, error, 1);
    a = cblas_dnrm2(Nx, xin, 1);
    mse = 20.0*log10(a/mse);
    printf("Backprojected image SNR: %f dB\n\n", mse);

    
    printf("**********************\n");
    printf("TV reconstruction\n");
    printf("**********************\n");

    
    //Structure for the TV prox
    param2.verbose = 1;
    param2.max_iter = 50;
    param2.rel_obj = 0.0001;

    //Structure for the l2 ball prox
    param3.verbose = 1;
    param3.max_iter = 50;
    param3.tol = 0.001;
    param3.nu = 1;
    param3.tight = 1;
    param3.pos = 1;
    param3.real = 0;
    

    //Structure for the TV solver    
    param4.verbose = 2;
    param4.max_iter = 200;
    param4.gamma = 0.1;
    param4.rel_obj = 0.0005;
    param4.epsilon = sqrt(Ny + 2*sqrt(Ny))*sigma;
    param4.real_out = 1;
    param4.real_meas = 1;
    param4.paramtv = param2;
    param4.paraml2b = param3;

    //Initial solution for TV reconstruction
    for (i=0; i < Nx; i++) {
        xout[i] = 0.0;
    }

    assert((start = clock())!=-1);
    sopt_tv_solver((void*)xout, dim1, dim2,
                   &sopt_meas_urandsamp,
                   datam,
                   &sopt_meas_urandsampadj,
                   datam,
                   (void*)y, Ny, param4);
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

    fail = spot_image_tiff_write(xout, dim2, dim1, "images/cameraIP_TV_temp.tiff", 1.0/scale, offset);
    if(fail == 1)
      SOPT_ERROR_GENERIC("Error writing image");

    printf("**********************************\n");
    printf("L1-Db8 reconstruction\n");
    printf("**********************************\n");

    //SARA structure initialization

    param5.ndict = 1;
    param5.real = 1;

    dict_types = malloc(param5.ndict * sizeof(sopt_wavelet_type));
    SOPT_ERROR_MEM_ALLOC_CHECK(dict_types);


    dict_types[0] = SOPT_WAVELET_DB8;


    sopt_sara_initop(&param5, dim1, dim2, 4, dict_types);

    datas[0] = (void*)&param5;

    //Structure for the L1 prox
    param6.verbose = 1;
    param6.max_iter = 50;
    param6.rel_obj = 0.01;
    param6.nu = 1;
    param6.tight = 1;
    param6.pos = 0;
    
    //Structure for the L1 solver    
    param7.verbose = 2;
    param7.max_iter = 200;
    param7.gamma = 0.1;
    param7.rel_obj = 0.0005;
    param7.epsilon = sqrt(Ny + 2*sqrt(Ny))*sigma;
    param7.real_out = 1;
    param7.real_meas = 1;
    param7.paraml1 = param6;
    param7.paraml2b = param3;
    
   
    //Weights
    for (i=0; i < Nr; i++) {
        w[i] = 1.0;
    }

    //Initial solution for L1 recosntruction
    for (i=0; i < Nx; i++) {
        xout[i] = 0.0;
    }

    #ifdef _OPENMP
        start = omp_get_wtime();
    #else
        assert((start = clock())!=-1);
    #endif
    sopt_l1_solver((void*)xout, Nx,
                   &sopt_meas_urandsamp,
                   datam,
                   &sopt_meas_urandsampadj,
                   datam,
                   &sopt_sara_synthesisop,
                   datas,
                   &sopt_sara_analysisop,
                   datas,
                   Nr,
                   (void*)y, Ny, w, param7);
    #ifdef _OPENMP
        stop = omp_get_wtime();
        t = stop - start;
    #else    
        stop = clock();
        t = (double) (stop-start)/CLOCKS_PER_SEC;
    #endif    

    printf("Time L1: %f \n\n", t); 

    //SNR
    for (i=0; i < Nx; i++) {
        error[i] = xout[i] - xin[i];
    }
    mse = cblas_dnrm2(Nx, error, 1);
    a = cblas_dnrm2(Nx, xin, 1);
    mse = 20.0*log10(a/mse);
    printf("SNR: %f dB\n\n", mse);

    //Write output image
    fail = spot_image_tiff_write(xout, dim2, dim1, "images/cameraIP_Db8_temp.tiff", 1.0/scale, offset);
    if(fail == 1)
      SOPT_ERROR_GENERIC("Error writing image");

   

    


    
    free(xin);
    free(xout);
    free(y);
    free(y0);
    free(noise);
    free(error);
    free(w);
    free(param1.perm);
    free(dict_types);
    sopt_sara_free(&param5);
     
    return 0;
}

