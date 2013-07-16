


#include "mex.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <assert.h>
#include <time.h> 
#ifdef _OPENMP 
  #include <omp.h>
#endif 
#ifdef __APPLE__
  #include <Accelerate/Accelerate.h>
#elif __unix__
  #include <cblas.h>
#else
  #include <cblas.h>
#endif


#include <sopt_utility.h>
#include <sopt_error.h>
#include <sopt_prox.h> 
#include <sopt_ran.h>
#include <sopt_meas.h>
#include <sopt_tv.h>
#include <sopt_l1.h>
#include <sopt_wavelet.h>
#include <sopt_sara.h>
#include <sopt_sparsemat.h>
#include <sopt_image.h>


/**
 * L1 solver.
 *
 * Usage: 
 * xsol = sopt_solver_l1_mex(y, xsol, ny, nx, nr, A, At, Psi, Psit, ...
 *   Weights, Analysis, ...
 *   ParamMaxIter, ParamGamma, ParamRelObj, ...
 *   ParamEpsilon, ParamRealOut, ParamRealMeas, ...
 *   ParamL1ProxMaxIter, ParamL1ProxRelObj, ParamL1ProxNu, ...
 *   ParamL1ProxTight, ParamL1ProxPositivity, ...
 *   ParamL2BallMaxIter, ParamL2BallTol, ParamL2BallNu, ...
 *   ParamL2BallTight, ParamL2BallPositivity, ...
 *   ParamL2BallReality);
 */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{

  int i, j, iin, iout;

  int y_is_complex;
  double *y_real = NULL, *y_imag = NULL;
  double *yr = NULL;
  complex double *yc = NULL;
  int y_m, y_n;

  int xsol_is_complex;
  double *xsol_real = NULL, *xsol_imag = NULL;
  double *xsolr = NULL;
  complex double *xsolc = NULL;
  int xsol_m, xsol_n;




  // Check number of arguments. 
  if (nrhs != 28) {
    mexErrMsgIdAndTxt("sopt_solver_l1_mex:InvalidInput:nrhs",
          "Require 28 inputs.");
  }
  if (nlhs != 1) {
    mexErrMsgIdAndTxt("sopt_solver_l1_mex:InvalidOutput:nlhs",
          "Require one output.");
  }

  // Parse y.
  iin = 0;
  y_m = mxGetM(prhs[iin]);
  y_n = mxGetN(prhs[iin]);  
  if (!mxIsDouble(prhs[iin]) || (y_m != 1 && y_n != 1))
    mexErrMsgIdAndTxt("sopt_solver_l1_mex:InvalidInput:y",
          "y must be double array.");
  y_real = mxGetPr(prhs[iin]);  
  y_is_complex = mxIsComplex(prhs[iin]);
  y_imag = y_is_complex ? mxGetPi(prhs[iin]) : NULL;
  if (!y_is_complex) {
    yr = (double*)malloc(y_m * y_n * sizeof(double));
    for(i=0; i<y_m; i++)
      for(j=0; j<y_n; j++)
        yr[i*y_n + j] = y_real[i*y_n + j];
  }
  else {
    yc = (complex double*)malloc(y_m * y_n * sizeof(complex double));
    for(i=0; i<y_m; i++)
      for(j=0; j<y_n; j++)
        yc[i*y_n + j] = y_real[i*y_n + j]        
          + I * (y_is_complex ? y_imag[i*y_n + j] : 0.0);
  }

  // Parse xsol.
  iin = 1;
  xsol_m = mxGetM(prhs[iin]);
  xsol_n = mxGetN(prhs[iin]);  
  if (!mxIsDouble(prhs[iin]) || (xsol_m != 1 && xsol_n != 1))
    mexErrMsgIdAndTxt("sopt_solver_l1_mex:InvalidInput:xsol",
          "xsol must be double array.");
  xsol_real = mxGetPr(prhs[iin]);  
  xsol_is_complex = mxIsComplex(prhs[iin]);
  xsol_imag = xsol_is_complex ? mxGetPi(prhs[iin]) : NULL;
  if (!xsol_is_complex) {
    xsolr = (double*)malloc(xsol_m * xsol_n * sizeof(double));
    for(i=0; i<xsol_m; i++)
      for(j=0; j<xsol_n; j++)
        xsolr[i*xsol_n + j] = xsol_real[i*xsol_n + j];
  }
  else {
    xsolc = (complex double*)malloc(xsol_m * xsol_n * sizeof(complex double));
    for(i=0; i<xsol_m; i++)
      for(j=0; j<xsol_n; j++)
        xsolc[i*xsol_n + j] = xsol_real[i*xsol_n + j]        
          + I * (xsol_is_complex ? xsol_imag[i*xsol_n + j] : 0.0);
  }



  // if (f_is_complex && reality)
  //   mexWarnMsgTxt("Running real transform but input appears to be complex (ignoring imaginary component).");
  // if (!f_is_complex && !reality)
  //   mexWarnMsgTxt("Running complex transform on real signal (set reality flag to improve performance).");


    double *xout;
    double *error;
    double *y0;
    double *y;
    double *noise;
    double *w;



    void *datam[1];
    void *datas[1];

    int flag;
    double sigma;
    double a;
    double mse;
    double snr;

    int Nx, Ny, Nr;
    int seedn=54;
    int seedmat;

    Nx=256*256;
    Ny=(int)(0.5*Nx);
    Nr=1*Nx;
    
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
    sopt_meas_usparam param1;
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
    sopt_meas_urandsamp((void*)y0, (void*)yr, datam);
    
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
    sopt_meas_urandsampadj((void*)xsolr, (void*)y, datam);


//     sopt_prox_tvparam param2;
    sopt_prox_l2bparam param3;    
//     sopt_tv_param param4;


// //Structure for the TV prox
//     param2.verbose = 1;
//     param2.max_iter = 50;
//     param2.rel_obj = 0.0001;

    //Structure for the l2 ball prox
    param3.verbose = 1;
    param3.max_iter = 50;
    param3.tol = 0.001;
    param3.nu = 1;
    param3.tight = 1;
    param3.pos = 1;
    param3.real = 0;
    

//     //Structure for the TV solver    
//     param4.verbose = 2;
//     param4.max_iter = 200;
//     param4.gamma = 0.1;
//     param4.rel_obj = 0.0005;
//     param4.epsilon = sqrt(Ny + 2*sqrt(Ny))*sigma;
//     param4.real_out = 1;
//     param4.real_meas = 1;
//     param4.paramtv = param2;
//     param4.paraml2b = param3;

//     //Initial solution for TV reconstruction
//     for (i=0; i < Nx; i++) {
//         xsolr[i] = 0.0;
//     }

//     sopt_tv_solver((void*)xsolr, 256, 256,
//                    &sopt_meas_urandsamp,
//                    datam,
//                    &sopt_meas_urandsampadj,
//                    datam,
//                    (void*)y, Ny, param4);


    sopt_wavelet_type *dict_types;
    sopt_sara_param param5;
    sopt_prox_l1param param6;
    sopt_l1_param param7;
    

    param5.ndict = 1;
    param5.real = 1;

    dict_types = malloc(param5.ndict * sizeof(sopt_wavelet_type));
    SOPT_ERROR_MEM_ALLOC_CHECK(dict_types);


    dict_types[0] = SOPT_WAVELET_DB8;


    sopt_sara_initop(&param5, 256, 256, 4, dict_types);

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
        xsolr[i] = 0.0;
    }

    sopt_l1_solver((void*)xsolr, Nx,
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









  // Copy xsol to output.  
  iout = 0;
  if (!xsol_is_complex) {
    plhs[iout] = mxCreateDoubleMatrix(xsol_m * xsol_n, 1, mxREAL);
    xsol_real = mxGetPr(plhs[iout]);    
    for(i=0; i<xsol_m; i++)
      for(j=0; j<xsol_n; j++)
        xsol_real[i*xsol_n + j] = xsolr[i*xsol_n + j];
  }
  else {
    plhs[iout] = mxCreateDoubleMatrix(xsol_m * xsol_n, 1, mxCOMPLEX);
    xsol_real = mxGetPr(plhs[iout]);
    xsol_imag = mxGetPi(plhs[iout]);
    for(i=0; i<xsol_m; i++)
      for(j=0; j<xsol_n; j++) {
        xsol_real[i*xsol_n + j] = creal(xsolc[i*xsol_n + j]);
        xsol_imag[i*xsol_n + j] = cimag(xsolc[i*xsol_n + j]);
      }
  }

  // Free memory. 
  if (!y_is_complex)
    free(yr);
  else
    free(yc);
  if (!xsol_is_complex)
    free(xsolr);
  else
    free(xsolc);







    free(xout);
    free(y);
    free(y0);
    free(noise);
    free(error);
    free(w);
    free(param1.perm);

    free(dict_types);
    sopt_sara_free(&param5);
    



}
