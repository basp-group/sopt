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
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

complex double sopt_utility_softthc(complex double x, double T);

double sopt_utility_softthr(double x, double T);

double sopt_utility_l1normc(complex double *x, double *w, int dim);

double sopt_utility_l1normr(double *x, double *w, int dim);

double sopt_utility_sql2normc(complex double *x, double *w, int dim);

double sopt_utility_sql2normr(double *x, double *w, int dim);

double sopt_utility_tvnormc(complex double *x, int dim1, int dim2);

double sopt_utility_tvnormr(double *x, int dim1, int dim2);

double sopt_utility_wtvnormc(complex double *x, double *wt_dx, double *wt_dy, int dim1, int dim2);

double sopt_utility_wtvnormr(double *x, double *wt_dx, double *wt_dy, int dim1, int dim2);

void sopt_utility_gradientc(complex double *dx, complex double *dy, 
                   complex double *xin, int dim1, int dim2);

void sopt_utility_gradientr(double *dx, double *dy, double *xin, int dim1, int dim2);

void sopt_utility_divergencec(complex double *xout, complex double *dx, 
                     complex double *dy, int dim1, int dim2);

void sopt_utility_divergencer(double *xout, double *dx, 
                     double *dy, int dim1, int dim2);

void sopt_utility_projposc(complex double *xout, complex double *xin, int dim);

void sopt_utility_projposr(double *xout, double *xin, int dim);

void sopt_utility_projreal(complex double *xout, complex double *xin, int dim);

complex double sopt_utility_dotpc(complex double *xin, complex double *yin, int dim);

double sopt_utility_dotpr(double *xin, double *yin, int dim);

void sopt_utility_cgsolc(complex double *xout,
                            complex double *xin, 
                            complex double *v,
                            complex double *r,
                            complex double *p,
                            complex double *ap,
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

double sopt_utility_btrackc(complex double *v,
                          complex double *r, 
                          complex double *w, 
                          complex double *dummy,
                          complex double *xaux,  
                          complex double *xout, 
                          complex double *xin, 
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
