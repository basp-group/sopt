/*
 *  sopt_utility.h
 *  
 *
 *  Created by Rafael Carrillo on 8/27/12.
 *  Copyright 2012 EPFL. All rights reserved.
 *
 */

#ifndef SOPT_UTILITY
#define SOPT_UTILITY
#include "sopt_config.h"
#include "sopt_types.h"

#ifdef __cplusplus
extern "C" {
#endif

sopt_complex_double sopt_utility_softthc(sopt_complex_double x, double T);

double sopt_utility_softthr(double x, double T);

double sopt_utility_l1normc(sopt_complex_double *x, double *w, int dim);

double sopt_utility_l1normr(double *x, double *w, int dim);

double sopt_utility_sql2normc(sopt_complex_double *x, double *w, int dim);

double sopt_utility_sql2normr(double *x, double *w, int dim);

double sopt_utility_tvnormc(sopt_complex_double *x, int dim1, int dim2);

double sopt_utility_tvnormr(double *x, int dim1, int dim2);

double sopt_utility_wtvnormc(sopt_complex_double *x, double *wt_dx, double *wt_dy, int dim1, int dim2);

double sopt_utility_wtvnormr(double *x, double *wt_dx, double *wt_dy, int dim1, int dim2);

void sopt_utility_gradientc(sopt_complex_double *dx, sopt_complex_double *dy, 
                   sopt_complex_double *xin, int dim1, int dim2);

void sopt_utility_gradientr(double *dx, double *dy, double *xin, int dim1, int dim2);

void sopt_utility_divergencec(sopt_complex_double *xout, sopt_complex_double *dx, 
                     sopt_complex_double *dy, int dim1, int dim2);

void sopt_utility_divergencer(double *xout, double *dx, 
                     double *dy, int dim1, int dim2);

void sopt_utility_projposc(sopt_complex_double *xout, sopt_complex_double *xin, int dim);

void sopt_utility_projposr(double *xout, double *xin, int dim);

void sopt_utility_projreal(sopt_complex_double *xout, sopt_complex_double *xin, int dim);

sopt_complex_double sopt_utility_dotpc(sopt_complex_double *xin, sopt_complex_double *yin, int dim);

double sopt_utility_dotpr(double *xin, double *yin, int dim);

void sopt_utility_cgsolc(sopt_complex_double *xout,
                            sopt_complex_double *xin, 
                            sopt_complex_double *v,
                            sopt_complex_double *r,
                            sopt_complex_double *p,
                            sopt_complex_double *ap,
                            void (*A)(void *out, void *in, void **data), 
                            void **A_data,
                            void (*At)(void *out, void *in, void **data), 
                            void **At_data,
                            int nx, 
                            int ny,
                            double tol,
                            int nit,
                            int verbose);

void sopt_utility_cgsolr(double *xout,
                            double *xin, 
                            double *v,
                            double *r,
                            double *p,
                            double *ap,
                            void (*A)(void *out, void *in, void **data), 
                            void **A_data,
                            void (*At)(void *out, void *in, void **data), 
                            void **At_data,
                            int nx, 
                            int ny,
                            double tol,
                            int nit,
                            int verbose);

double sopt_utility_btrackc(sopt_complex_double *v,
                          sopt_complex_double *r, 
                          sopt_complex_double *w, 
                          sopt_complex_double *dummy,
                          sopt_complex_double *xaux,  
                          sopt_complex_double *xout, 
                          sopt_complex_double *xin, 
                          void (*A)(void *out, void *in, void **data), 
                          void **A_data,
                          void (*At)(void *out, void *in, void **data), 
                          void **At_data,
                          int nx, 
                          int ny,
                          double nu,
                          double epsilon);

double sopt_utility_btrackr(double *v,
                          double *r, 
                          double *w, 
                          double *dummy,
                          double *xaux,  
                          double *xout, 
                          double *xin, 
                          void (*A)(void *out, void *in, void **data), 
                          void **A_data,
                          void (*At)(void *out, void *in, void **data), 
                          void **At_data,
                          int nx, 
                          int ny,
                          double nu,
                          double epsilon);



#ifdef __cplusplus
}
#endif
#endif
