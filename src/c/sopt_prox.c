/*
 *  sopt_prox.c
 *  
 *
 *  Created by Rafael Carrillo on 8/23/12.
 *  Copyright 2012 EPFL. All rights reserved.
 *
 */
#include "sopt_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h> 
#include <string.h>
#include <math.h>
#include SOPT_BLAS_H
#include "sopt_error.h"
#include "sopt_utility.h"
#include "sopt_prox.h"

#define min(a,b) (a<b?a:b)
#define max(a,b) (a<b?b:a)

/*!
 * This function computes the prox operator of the l1
 * norm for the input vector \f$ x \f$. It solves the problem:
 * \f[
 * min_{z} 0.5||x - z||_2^2 + \lambda ||\Psi^\dagger z||_1,
 * \f]
 * where \f$ \Psi \in C^{N_x \times N_r} \f$ is the sparsifying operator.
 * The solution is denoted \f$x^\star \in C^{N_x}\f$.
 * \param[out] xout Output solution (\f$ x^\star \f$).
 * \param[in] xin Input vector (\f$ x \f$).
 * \param[in] nx Dimension of the signal (\f$N_x\f$).
 * \param[in] nr Dimension of the signal in the representation domain
 * (\f$N_r\f$).
 * \param[in] Psi Pointer to the synthesis sparsity operator \f$\Psi\f$.
 * \param[in] Psi_data Data structure associated to \f$\Psi\f$.
 * \param[in] Psit Pointer to the analysis sparsity operator
 * \f$\Psi^\dagger\f$.
 * \param[in] Psit_data Data structure associated to \f$\Psi^\dagger\f$.
 * \param[in] weights weights vector.
 * \param[in] lambda Convergence parameter \f$\lambda\f$.
 * \param[in] real_data Flag for real or complex data. 1 if real 0 complex.
 * \param[in] param Data structure with the parameters of
 *            the optimization.
 * \param[in] dummy Auxiliary vector. Dimension: nr.
 * \param[in] sol Auxiliary vector. Dimension: nx.
 * \param[in] u Auxiliary vector. Dimension: nr.
 * \param[in] v Auxiliary vector. Dimension: nr.
 */
void sopt_prox_l1(void *xout, 
                  void *xin,
                  int nx,
                  int nr,
                  void (*Psi)(void *out, void *in, void **data), 
                  void **Psi_data,
                  void (*Psit)(void *out, void *in, void **data), 
                  void **Psit_data,
                  double *weights,
                  double lambda,
                  int real_data,
                  sopt_prox_l1param param,
                  void *dummy,
                  void *sol,
                  void *u,
                  void *v){
    
     int i;
     int iter;
     double obj;
     double prev_ob;
     double rel_ob;
     double mu;
     double temp;
     double t;
     double told;
     char crit[8];
     complex double alpha;
     
     mu = 1.0/param.nu;
     
     
     if ((param.tight==1)&&(param.pos==0)){
         //Tight frame case
         Psit(dummy, xin, Psit_data);
         if (real_data == 1){
            for (i=0; i < nr; i++) {
                *((double*)dummy + i) = 
                sopt_utility_softthr(*((double*)dummy + i),lambda*param.nu*weights[i]) - *((double*)dummy + i);
            }
            Psi(xout, dummy, Psi_data);
            //xout = xin + mu*xout;
            alpha = 1.0 + 0.0*I;
            cblas_dscal(nx, mu, (double*)xout,1);
            cblas_daxpy(nx, 1.0, (double*)xin, 1, (double*)xout, 1);
            //Objective evaluation
            Psit(dummy, xout, Psit_data);
            temp = sopt_utility_l1normr((double*)dummy, weights, nr);
            cblas_dcopy(nx, (void*)xout, 1, (double*)sol, 1);
            cblas_daxpy(nx, -1.0, (double*)xin, 1, (double*)sol, 1);
            obj = cblas_dnrm2(nx, (double*)sol, 1);
         }
         else{
            for (i=0; i < nr; i++) {
                *((complex double*)dummy + i) = 
                sopt_utility_softthc(*((complex double*)dummy + i),lambda*param.nu*weights[i]) - *((complex double*)dummy + i);
            }
            Psi(xout, dummy, Psi_data);
            //xout = xin + mu*xout;
            alpha = 1.0 + 0.0*I;
            cblas_zdscal(nx, mu, (void*)xout,1);
            cblas_zaxpy(nx, (void*)&alpha, (void*)xin, 1, (void*)xout, 1);
            //Objective evaluation
            Psit(dummy, xout, Psit_data);
            temp = sopt_utility_l1normc((complex double*)dummy, weights, nr);
            cblas_zcopy(nx, (void*)xout, 1, (void*)sol, 1);
            alpha = -1.0 + 0.0*I;
            cblas_zaxpy(nx, (void*)&alpha, (void*)xin, 1, (void*)sol, 1);
            obj = cblas_dznrm2(nx, (void*)sol, 1);
         }
         
         obj = 0.5*obj*obj + lambda*temp;
         iter = 1;
         strcpy(crit, "REL_OBJ");
     }
     else{
         //Initializations non tight frame case
         
         obj = 0.0;
         prev_ob = 0.0;
         rel_ob = 1.0;
         iter = 0;
         alpha = 0.0 + 0.0*I;
         told = 1.0;

          if (real_data == 1){
            //xout = xin
            cblas_dcopy(nx, (double*)xin, 1, (double*)xout, 1);
             //u = 0
            for (i=0; i < nr; i++) {
                *((double*)u + i) = 0.0;
            }
          }
          else{
            //xout = xin
            cblas_zcopy(nx, xin, 1, xout, 1);
             //u = 0
            for (i=0; i < nr; i++) {
                *((complex double*)u + i) = alpha;
            }
          }
         
         if (param.verbose > 1){
             printf("Proximal L1 operator:\n ");
         }
         while (1){
             //Objective evaluation
             Psit(dummy, xout, Psit_data);
             if (real_data == 1){
                temp = sopt_utility_l1normr((double*)dummy, weights, nr);
                cblas_dcopy(nx, (double*)xout, 1, (double*)sol, 1);
                alpha = -1.0 + 0.0*I;
                cblas_daxpy(nx, -1.0, (double*)xin, 1, (double*)sol, 1);
                obj = cblas_dnrm2(nx, (double*)sol, 1);
             }
             else {
                temp = sopt_utility_l1normc((complex double*)dummy, weights, nr);
                cblas_zcopy(nx, xout, 1, sol, 1);
                alpha = -1.0 + 0.0*I;
                cblas_zaxpy(nx, (void*)&alpha, xin, 1, sol, 1);
                obj = cblas_dznrm2(nx, sol, 1);
             }
             
             obj = 0.5*obj*obj + lambda*temp;
             rel_ob = fabs(obj-prev_ob)/obj;
             //Log
             if (param.verbose > 1){
                 printf("Iter %i: obj = %e, rel obj = %e \n ", iter, obj, rel_ob);
             }
             //Stopping criteria
             if (rel_ob < param.rel_obj){
                 strcpy(crit, "REL_OBJ");
                 break;
             }
             if (iter > param.max_iter){
                 strcpy(crit, "MAX_ITE");
                 break;
             }
             t = (1.0+sqrt(1.0+4.0*told*told))/2.0;
             temp = (told-1.0)/t;
             //Dual forward-backward
             if (real_data == 1){
                //v = u
                //v = param.nu*v + dummy
                cblas_dcopy(nr, (double*)u, 1, (double*)v, 1);
                cblas_dscal(nr, param.nu, (double*)v,1);
                cblas_daxpy(nr, 1.0, (double*)dummy, 1, (double*)v, 1);
                //Soft thresolding
                for (i=0; i < nr; i++) {
                    *((double*)dummy + i) = 
                    sopt_utility_softthr(*((double*)v + i),lambda*param.nu*weights[i]);
                }
                //v = mu*(v - dummy)
                cblas_daxpy(nr, -1.0, (double*)dummy, 1, (double*)v, 1);
                cblas_dscal(nr, mu, (double*)v,1);
                //FISTA update
                //u = v + temp*(v - u)
                cblas_dscal(nr, -temp, (double*)u,1);
                cblas_daxpy(nr, 1.0 + temp, (double*)v, 1, (double*)u, 1);
                //xout = xin - Psi(u)
                cblas_dcopy(nr, (double*)u, 1, (double*)dummy, 1);
                Psi(xout, dummy, Psi_data);
                cblas_dscal(nx, -1.0, (double*)xout,1);
                cblas_daxpy(nx, 1.0, (double*)xin, 1, (double*)xout, 1);
                //Positivity constraint
                if (param.pos==1){
                    sopt_utility_projposr((double*)xout, (double*)xout, nx);
                }
             }
             else {
                //v = u
                cblas_zcopy(nr, u, 1, v, 1);
                //v = param.nu*v + dummy
                alpha = 1.0 + 0.0*I;
                cblas_zdscal(nr, param.nu, v,1);
                cblas_zaxpy(nr, (void*)&alpha, dummy, 1, v, 1);
                //Soft thresolding
                for (i=0; i < nr; i++) {
                    *((complex double*)dummy + i) = 
                    sopt_utility_softthc(*((complex double*)v + i),lambda*param.nu*weights[i]);
                }
                //u = mu*(u - dummy)
                alpha = -1.0 + 0.0*I;
                cblas_zaxpy(nr, (void*)&alpha, dummy, 1, v, 1);
                cblas_zdscal(nr, mu, v,1);
                //FISTA update
                //u = v + temp*(v - u)
                cblas_zdscal(nr, -temp, u,1);
                alpha = 1.0 + temp + 0.0*I;
                cblas_zaxpy(nr, (void*)&alpha, v, 1, u, 1);
                //xout = xin - Psi(u)
                cblas_zcopy(nr, u, 1, dummy, 1);
                Psi(xout, dummy, Psi_data);
                alpha = 1.0 + 0.0*I;
                cblas_zdscal(nx, -1.0, xout,1);
                cblas_zaxpy(nx, (void*)&alpha, xin, 1, xout, 1);
                //Positivity constraint
                if (param.pos==1){
                    sopt_utility_projposc((complex double*)xout, (complex double*)xout, nx);
                }
             }
             //Update
             iter++;
             prev_ob = obj; 
             told = t; 
         }
     }
    if (param.verbose > 0){
        printf("Prox L1: objective = %e, %s, num. iter = %i \n ", obj, crit, iter);
    }   
}
 


/*!
 * This function computes the prox operator of the TV
 * norm for the input image \f$ x \f$. It solves the problem:
 * \f[
 * min_{z} 0.5||x - z||_2^2 + \lambda||z||_{TV}.
 * \f]
 *The solution is denoted \f$x^\star \in C^{N_{x1}\times N_{x2}}\f$.
 * \param[out] xout Output solution (\f$ x^\star \f$).
 * \param[in] xin Input image (\f$ x \f$).
 * \param[in] nx1 Number of columns of the input image
 * \param[in] nx2 Number of rows of the input image
 * \param[in] lambda Convergence parameter \f$\lambda\f$.
 * \param[in] real_data Flag for real or complex data. 1 if real 0 complex.
 * \param[in] param Data structure with the parameters of
 *            the optimization.
 * \param[in] r Auxiliary vector. Dimension: nx=nx1*nx2.
 * \param[in] s Auxiliary vector. Dimension: nx.
 * \param[in] pold Auxiliary vector. Dimension: nx.
 * \param[in] qold Auxiliary vector. Dimension: nx.
 * \param[in] dx Auxiliary vector. Dimension: nx.
 * \param[in] dy Auxiliary vector. Dimension: nx.
 */
void sopt_prox_tv(void *xout, 
                  void *xin,
                  int nx1,
                  int nx2,
                  double lambda,
                  int real_data,
                  sopt_prox_tvparam param,
                  void *r,
                  void *s,
                  void *pold,
                  void *qold,
                  void *dx,
                  void *dy){
    
    int i;
    int nx;
    int iter;
    double obj;
    double prev_ob;
    double rel_ob;
    double temp;
    double t;
    double told;
    double mu;
    double dummyr;
    complex double dummy;
    char crit[8];
    complex double alpha;
    
      
    //Initializations
    nx = nx1*nx2;
    obj = 0.0;
    prev_ob = 0.0;
    rel_ob = 1.0;
    mu = 1.0/(8.0*lambda);
    told = 1.0;
    iter = 1;
    strcpy(crit, "MAX_ITE");
        
    alpha = 0.0 + 0.0*I;
    
    if (real_data == 1){
        for (i = 0; i < nx; i++){
            *((double*)r + i) = 0.0;
            *((double*)s + i) = 0.0;
            *((double*)pold + i) = 0.0;
            *((double*)qold + i) = 0.0;
        }
    }
    else{
        for (i = 0; i < nx; i++){
            *((complex double*)r + i) = alpha;
            *((complex double*)s + i) = alpha;
            *((complex double*)pold + i) = alpha;
            *((complex double*)qold + i) = alpha;
        }
    }
    
    if (param.verbose > 1){
        printf("Proximal TV operator:\n ");
    }
    
    
    //Main loop
    for (iter=1; iter <= param.max_iter; iter++){
        //Current solution
        if (real_data == 1){
            sopt_utility_divergencer((double*)dx, (double*)r, (double*)s, nx1, nx2);
            //xout = xin - lambda*dx
            cblas_dcopy(nx, (double*)xin, 1, (double*)xout, 1);
            cblas_daxpy(nx, -lambda, (double*)dx, 1, (double*)xout, 1);
        }
        else {
            sopt_utility_divergencec((complex double*)dx, (complex double*)r, (complex double*)s, nx1, nx2);
            //xout = xin - lambda*dx
            cblas_zcopy(nx, xin, 1, xout, 1);
            alpha = -lambda + 0.0*I;
            cblas_zaxpy(nx, (void*)&alpha, dx, 1, xout, 1);
        }
        
        //Objective function evaluation
        if (real_data == 1){
            temp = sopt_utility_tvnormr((double*)xout, nx1, nx2);
            //dx = xout - xin
            cblas_dcopy(nx, (double*)xout, 1, (double*)dx, 1);
            alpha = -1.0 + 0.0*I;
            cblas_daxpy(nx, -1.0, (double*)xin, 1, (double*)dx, 1);
            obj = cblas_dznrm2(nx, (double*)dx, 1);
        }
        else{
            temp = sopt_utility_tvnormc((complex double*)xout, nx1, nx2);
            //dx = xout - xin
            cblas_zcopy(nx, xout, 1, dx, 1);
            alpha = -1.0 + 0.0*I;
            cblas_zaxpy(nx, (void*)&alpha, xin, 1, dx, 1);
            obj = cblas_dznrm2(nx, dx, 1);
        }
        obj = 0.5*obj*obj + lambda*temp;
        
        if (obj > 0.0){
            rel_ob = fabs(obj-prev_ob)/obj;
        }
        else if ((fabs(obj-prev_ob) == 0)&&(iter > 1)){
            rel_ob = 0.0;
        } 
        else{
            rel_ob = 1.0;
        }
        //Log
        if (param.verbose > 1){
            printf("Iter %i: obj = %e, rel obj = %e \n ", iter, obj, rel_ob);
        }
        //Stopping criteria
        if (rel_ob < param.rel_obj){
            strcpy(crit, "REL_OBJ");
            break;
        }
        
        //Update divergence and project
        if (real_data == 1){
            sopt_utility_gradientr((double*)dx, (double*)dy, (double*)xout, nx1, nx2);
            for (i=0; i < nx; i++) {
                *((double*)r + i) += - mu**((double*)dx + i);
                *((double*)s + i) += - mu**((double*)dy + i);
                temp = sqrt(*((double*)r + i)**((double*)r + i) + *((double*)s + i)**((double*)s + i));
                temp = max(1.0,temp);
                *((double*)r + i) = *((double*)r + i)/temp;
                *((double*)s + i) = *((double*)s + i)/temp;
            }
            //FISTA update
            t = (1.0+sqrt(1.0+4.0*told*told))/2.0;
            temp = (told-1.0)/t;
            for (i=0; i < nx; i++) {
                dummyr = *((double*)r + i);
                *((double*)r + i) += temp*(*((double*)r + i) - *((double*)pold + i));
                *((double*)pold + i) = dummyr;
                dummyr = *((double*)s + i);
                *((double*)s + i) += temp*(*((double*)s + i) - *((double*)qold + i));
                *((double*)qold + i) = dummyr;
            }
        }
        else{
            sopt_utility_gradientc((complex double*)dx, (complex double*)dy, (complex double*)xout, nx1, nx2);
            for (i=0; i < nx; i++) {
                *((complex double*)r + i) += - mu**((complex double*)dx + i);
                *((complex double*)s + i) += - mu**((complex double*)dy + i);
                temp = sqrt(creal(*((complex double*)r + i))*creal(*((complex double*)r + i)) 
                       + cimag(*((complex double*)r + i))*cimag(*((complex double*)r + i))
                       + creal(*((complex double*)s + i))*creal(*((complex double*)s + i)) 
                       + cimag(*((complex double*)s + i))*cimag(*((complex double*)s + i)));
                temp = max(1.0,temp);
                *((complex double*)r + i) = *((complex double*)r + i)/temp;
                *((complex double*)s + i) = *((complex double*)s + i)/temp;
            }
            //FISTA update
            t = (1.0+sqrt(1.0+4.0*told*told))/2.0;
            temp = (told-1.0)/t;
            for (i=0; i < nx; i++) {
                dummy = *((complex double*)r + i);
                *((complex double*)r + i) += temp*(*((complex double*)r + i) - *((complex double*)pold + i));
                *((complex double*)pold + i) = dummy;
                dummy = *((complex double*)s + i);
                *((complex double*)s + i) += temp*(*((complex double*)s + i) - *((complex double*)qold + i));
                *((complex double*)qold + i) = dummy;
            }
        }
        

        told = t;
        prev_ob = obj;
        
    }
    //Log
    if (param.verbose > 0){
        printf("Prox TV: objective = %e, %s, num. iter = %i \n ", obj, crit, iter);
    }
       
}

/*!
 * This function computes the prox operator of the weighted TV
 * norm for the input image \f$ x \f$. It solves the problem:
 * \f[
 * min_{z} 0.5||x - z||_2^2 + \lambda||z||_{WTV}.
 * \f]
 *The solution is denoted \f$x^\star \in C^{N_{x1}\times N_{x2}}\f$.
 * \param[out] xout Output solution (\f$ x^\star \f$).
 * \param[in] xin Input image (\f$ x \f$).
 * \param[in] nx1 Number of columns of the input image
 * \param[in] nx2 Number of rows of the input image
 * \param[in] lambda Convergence parameter \f$\lambda\f$.
 * \param[in] real_data Flag for real or complex data. 1 if real 0 complex.
 * \param[in] param Data structure with the parameters of
 *            the optimization.
 * \param[in] wt_dx Weight vector in x direction. Dimension: nx.
 * \param[in] wt_dy Weight vector in y direction. Dimension: nx.
 * \param[in] r Auxiliary vector. Dimension: nx.
 * \param[in] s Auxiliary vector. Dimension: nx.
 * \param[in] xaux Auxiliary vector. Dimension: nx.
 * \param[in] pold Auxiliary vector. Dimension: nx.
 * \param[in] qold Auxiliary vector. Dimension: nx.
 * \param[in] dx Auxiliary vector. Dimension: nx.
 * \param[in] dy Auxiliary vector. Dimension: nx.
 */
void sopt_prox_wtv(void *xout, 
                  void *xin,
                  int nx1,
                  int nx2,
                  double lambda,
                  int real_data,
                  sopt_prox_tvparam param,
                  double *wt_dx,
                  double *wt_dy,
                  void *r,
                  void *s,
                  void *xaux,
                  void *pold,
                  void *qold,
                  void *dx,
                  void *dy){
    
    int i;
    int nx;
    int iter;
    double obj;
    double prev_ob;
    double rel_ob;
    double temp;
    double t;
    double told;
    double mu;
    double dummyr;
    complex double dummy;
    char crit[8];
    complex double alpha;
    
      
    //Initializations
    nx = nx1*nx2;
    obj = 0.0;
    prev_ob = 0.0;
    rel_ob = 1.0;
    mu = 1.0/(8.0*lambda);
    told = 1.0;
    iter = 1;
    strcpy(crit, "MAX_ITE");
        
    alpha = 0.0 + 0.0*I;
    
    if (real_data == 1){
        for (i = 0; i < nx; i++){
            *((double*)r + i) = 0.0;
            *((double*)s + i) = 0.0;
            *((double*)pold + i) = 0.0;
            *((double*)qold + i) = 0.0;
        }
    }
    else{
        for (i = 0; i < nx; i++){
            *((complex double*)r + i) = alpha;
            *((complex double*)s + i) = alpha;
            *((complex double*)pold + i) = alpha;
            *((complex double*)qold + i) = alpha;
        }
    }
    
    if (param.verbose > 1){
        printf("Proximal TV operator:\n ");
    }
    
    
    //Main loop
    for (iter=1; iter <= param.max_iter; iter++){
        //Current solution
        if (real_data == 1){
            for (i = 0; i < nx; i++){
                *((double*)dx + i) = *((double*)r + i)**(wt_dx + i);
                *((double*)dy + i) = *((double*)s + i)**(wt_dy + i);
            }
            sopt_utility_divergencer((double*)xaux, (double*)dx, (double*)dy, nx1, nx2);
            //xout = xin - lambda*xaux
            cblas_dcopy(nx, (double*)xin, 1, (double*)xout, 1);
            cblas_daxpy(nx, -lambda, (double*)xaux, 1, (double*)xout, 1);
        }
        else {
            for (i = 0; i < nx; i++){
                *((complex double*)dx + i) = *((complex double*)r + i)**(wt_dx + i);
                *((complex double*)dy + i) = *((complex double*)s + i)**(wt_dy + i);
            }
            sopt_utility_divergencec((complex double*)xaux, (complex double*)dx, (complex double*)dy, nx1, nx2);
            //xout = xin - lambda*xaux
            cblas_zcopy(nx, xin, 1, xout, 1);
            alpha = -lambda + 0.0*I;
            cblas_zaxpy(nx, (void*)&alpha, xaux, 1, xout, 1);
        }
        
        //Objective function evaluation
        if (real_data == 1){
            temp = sopt_utility_wtvnormr((double*)xout, wt_dx, wt_dy, nx1, nx2);
            //dx = xout - xin
            cblas_dcopy(nx, (double*)xout, 1, (double*)dx, 1);
            alpha = -1.0 + 0.0*I;
            cblas_daxpy(nx, -1.0, (double*)xin, 1, (double*)dx, 1);
            obj = cblas_dznrm2(nx, (double*)dx, 1);
        }
        else{
            temp = sopt_utility_wtvnormc((complex double*)xout, wt_dx, wt_dy, nx1, nx2);
            //dx = xout - xin
            cblas_zcopy(nx, xout, 1, dx, 1);
            alpha = -1.0 + 0.0*I;
            cblas_zaxpy(nx, (void*)&alpha, xin, 1, dx, 1);
            obj = cblas_dznrm2(nx, dx, 1);
        }
        obj = 0.5*obj*obj + lambda*temp;
        
        if (obj > 0.0){
            rel_ob = fabs(obj-prev_ob)/obj;
        }
        else if ((fabs(obj-prev_ob) == 0)&&(iter > 1)){
            rel_ob = 0.0;
        } 
        else{
            rel_ob = 1.0;
        }
        //Log
        if (param.verbose > 1){
            printf("Iter %i: obj = %e, rel obj = %e \n ", iter, obj, rel_ob);
        }
        //Stopping criteria
        if (rel_ob < param.rel_obj){
            strcpy(crit, "REL_OBJ");
            break;
        }
        
        //Update divergence and project
        if (real_data == 1){
            sopt_utility_gradientr((double*)dx, (double*)dy, (double*)xout, nx1, nx2);
            for (i = 0; i < nx; i++){
                *((double*)dx + i) = *((double*)dx + i)**(wt_dx + i);
                *((double*)dy + i) = *((double*)dy + i)**(wt_dy + i);
            }
            for (i=0; i < nx; i++) {
                *((double*)r + i) += - mu**((double*)dx + i);
                *((double*)s + i) += - mu**((double*)dy + i);
                temp = sqrt(*((double*)r + i)**((double*)r + i) + *((double*)s + i)**((double*)s + i));
                temp = max(1.0,temp);
                *((double*)r + i) = *((double*)r + i)/temp;
                *((double*)s + i) = *((double*)s + i)/temp;
            }
            //FISTA update
            t = (1.0+sqrt(1.0+4.0*told*told))/2.0;
            temp = (told-1.0)/t;
            for (i=0; i < nx; i++) {
                dummyr = *((double*)r + i);
                *((double*)r + i) += temp*(*((double*)r + i) - *((double*)pold + i));
                *((double*)pold + i) = dummyr;
                dummyr = *((double*)s + i);
                *((double*)s + i) += temp*(*((double*)s + i) - *((double*)qold + i));
                *((double*)qold + i) = dummyr;
            }
        }
        else{
            sopt_utility_gradientc((complex double*)dx, (complex double*)dy, (complex double*)xout, nx1, nx2);
            for (i = 0; i < nx; i++){
                *((complex double*)dx + i) = *((complex double*)dx + i)**(wt_dx + i);
                *((complex double*)dy + i) = *((complex double*)dy + i)**(wt_dy + i);
            }
            for (i=0; i < nx; i++) {
                *((complex double*)r + i) += - mu**((complex double*)dx + i);
                *((complex double*)s + i) += - mu**((complex double*)dy + i);
                temp = sqrt(creal(*((complex double*)r + i))*creal(*((complex double*)r + i)) 
                       + cimag(*((complex double*)r + i))*cimag(*((complex double*)r + i))
                       + creal(*((complex double*)s + i))*creal(*((complex double*)s + i)) 
                       + cimag(*((complex double*)s + i))*cimag(*((complex double*)s + i)));
                temp = max(1.0,temp);
                *((complex double*)r + i) = *((complex double*)r + i)/temp;
                *((complex double*)s + i) = *((complex double*)s + i)/temp;
            }
            //FISTA update
            t = (1.0+sqrt(1.0+4.0*told*told))/2.0;
            temp = (told-1.0)/t;
            for (i=0; i < nx; i++) {
                dummy = *((complex double*)r + i);
                *((complex double*)r + i) += temp*(*((complex double*)r + i) - *((complex double*)pold + i));
                *((complex double*)pold + i) = dummy;
                dummy = *((complex double*)s + i);
                *((complex double*)s + i) += temp*(*((complex double*)s + i) - *((complex double*)qold + i));
                *((complex double*)qold + i) = dummy;
            }
        }
        

        told = t;
        prev_ob = obj;
        
    }
    //Log
    if (param.verbose > 0){
        printf("Prox TV: objective = %e, %s, num. iter = %i \n ", obj, crit, iter);
    }
       
}


/*!
 * This function computes the projection onto the epsilon
 * l2 ball for the input vector \f$ x \f$. It solves the problem:
 * \f[
 * min_{z} ||x - z||_2^2 \quad \mbox{s.t.} \quad ||y - A z||_2 < epsilon.
 * \f]
 * where \f$A \in C^{N_y \times N_x}\f$ is the measurement operator,
 * \f$\epsilon\f$ is the ball radius and \f$y\in C^{Ny}\f$ is the
 * measurement vector.  The solution is denoted \f$x^\star \in
 * C^{N_x}\f$.
 * \param[out] xout Output solution (\f$ x^\star \f$).
 * \param[in] xin Input vector (\f$ x \f$).
 * \param[in] nx Dimension of the signal (\f$N_x\f$).
 * \param[in] y Measurement vector (\f$y\f$).
 * \param[in] ny Dimension of measurement vector, i.e. number of
 *            measurements (\f$N_y\f$).
 * \param[in] A Pointer to the measurement operator \f$A\f$.
 * \param[in] A_data Data structure associated to \f$A\f$.
 * \param[in] At Pointer to the the adjoint of the measurement operator 
 * \f$A^\dagger\f$.
 * \param[in] At_data Data structure associated to \f$A^\dagger\f$.
 * \param[in] epsilon Radius of the l2 ball \f$\epsilon\f$.
 * \param[in] real_data Flag for real or complex data. 1 if real 0 complex.
 * \param[in] param Data structure with the parameters of
 *            the optimization.
 * \param[in] dummy Auxiliary vector. Dimension: ny.
 * \param[in] u Auxiliary vector. Dimension: ny.
 * \param[in] v Auxiliary vector. Dimension: ny.
 */
void sopt_prox_l2b(void *xout, 
                   void *xin,
                   int nx,
                   void *y,
                   int ny,
                   void (*A)(void *out, void *in, void **data), 
                   void **A_data,
                   void (*At)(void *out, void *in, void **data), 
                   void **At_data,
                   double epsilon,
                   int real_data,
                   sopt_prox_l2bparam param,
                   void *dummy,
                   void *u,
                   void *v){
    
    int i;
    int iter;
    int flag;
    double norm_res;
    double mu;
    double temp;
    double t;
    double told;
    double epsilon_up;
    double epsilon_dwn;
    char crit[8];
    complex double nu;
    complex double alpha;
    
    nu = param.nu + 0.0*I;
    mu = param.nu;

    void *r;
    void *w;
    void *xaux;

    if (real_data == 1) {
        r = malloc(ny * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(r);
        w = malloc(ny * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(w);
        xaux = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(xaux);

    }
    else{
        r = malloc(ny * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(r);
        w = malloc(ny * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(w);
        xaux = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(xaux);
    }
    
    
    if ((param.tight==1)&&(param.pos==0 && param.real==0)){
        //Tight frame case
        
        if (real_data == 1) {
            //Residual: Ax-y
            A(dummy, xin, A_data);
            cblas_daxpy(ny, -1.0, (double*)y, 1, (double*)dummy, 1);
            temp = cblas_dnrm2(ny, (double*)dummy, 1);
            temp = min(1.0,epsilon/temp);
            temp = temp - 1.0;
            //dummy = temp*dummy
            cblas_dscal(ny, temp, (double*)dummy,1);
            //Solution
            //xout = xin + mu*At(dummy)
            At(xout, dummy, At_data);
            cblas_dscal(nx, mu, (double*)xout,1);
            cblas_daxpy(nx, 1.0, (double*)xin, 1, (double*)xout, 1);
        }
        else{
            //Residual: Ax-y
            A(dummy, xin, A_data);
            alpha = -1.0 +0.0*I;
            cblas_zaxpy(ny, (void*)&alpha, (void*)y, 1, (void*)dummy, 1);
            temp = cblas_dznrm2(ny, (void*)dummy, 1);
            temp = min(1.0,epsilon/temp);
            temp = temp - 1.0;
            //dummy = temp*dummy
            cblas_zdscal(ny, temp, (void*)dummy,1);
            //Solution
            //xout = xin + mu*At(dummy)
            At(xout, dummy, At_data);
            alpha = 1.0 + 0.0*I;
            cblas_zdscal(nx, mu, (void*)xout,1);
            cblas_zaxpy(nx, (void*)&alpha, (void*)xin, 1, (void*)xout, 1);
        }
        
        //Parameters for log
        iter = 1;
        strcpy(crit, "TOL_EPS");
        
        if (param.verbose > 0){
            //Residual
            A((void*)dummy, (void*)xout, A_data);
            if (real_data == 1) {
                cblas_daxpy(ny, -1.0, (double*)y, 1, (double*)dummy, 1);    
                norm_res = cblas_dnrm2(ny, (double*)dummy, 1);
            }
            else {
                alpha = -1.0 +0.0*I;
                cblas_zaxpy(ny, (void*)&alpha, y, 1, dummy, 1);    
                norm_res = cblas_dznrm2(ny, dummy, 1);
            }
        }  
    }
    else {
        //Non tight frame case
        //Initalizations
        flag = 1;
        told = 1.0;
        epsilon_up = (1.0+param.tol)*epsilon;
        epsilon_dwn = (1.0-param.tol)*epsilon;
        strcpy(crit, "MAX_ITE");
        iter = 1;

        if (real_data == 1) {
            //Set u = v = 0
            for (i = 0; i < ny; i++){
                *((double*)u + i) = 0.0;
                *((double*)v + i) = 0.0;
            }
            //Check if we are in the epsilon ball
            //xout = xin
            cblas_dcopy(nx, (double*)xin, 1, (double*)xout, 1);
            //Residual
            A(dummy, xout, A_data);
            cblas_daxpy(ny, -1.0, (double*)y, 1, (double*)dummy, 1);
            norm_res = cblas_dnrm2(ny, (double*)dummy, 1);

        }
        else {
            //Set u = v = 0
            alpha = 0.0 + 0.0*I;
            for (i = 0; i < ny; i++){
                *((complex double*)u + i) = alpha;
                *((complex double*)v + i) = alpha;
            }
            //Check if we are in the epsilon ball
            //xout = xin
            cblas_zcopy(nx, xin, 1, xout, 1);
            //Residual
            A(dummy, xout, A_data);
            alpha = -1.0 +0.0*I;
            cblas_zaxpy(ny, (void*)&alpha, y, 1, dummy, 1);
            norm_res = cblas_dznrm2(ny, dummy, 1);
        }
        
        
        if ((norm_res<=epsilon_up)&&(param.pos==0 && param.real==0)){
            flag = 0;
            strcpy(crit, "TOL_EPS");
        }
        
        if (param.verbose > 1 && flag==1){
            printf("Projection L2 ball: \n ");
        }
        
        while (flag==1){
            //Log
            if (param.verbose > 1){
                printf("Iter %i: epsilon = %e, residual = %e, \n ", iter, epsilon, norm_res);
            }
            //Stopping criteria
            if ( norm_res >= epsilon_dwn && norm_res <= epsilon_up){
                flag = 0;
                strcpy(crit, "TOL_EPS");
                break;
            }
            else if (iter>=param.max_iter){
                flag = 0;
                break;
            }
            //Dual forward-backward
            if (real_data == 1) {
                //Copy of u and dummy
                //r = dummy
                cblas_dcopy(ny, (double*)dummy, 1, (double*)r, 1);
                //w = u
                cblas_dcopy(ny, (double*)u, 1, (double*)w, 1);
                //Rescaling
                //dummy = nu*u + dummy
                cblas_daxpy(ny, param.nu, (double*)u, 1, (double*)dummy, 1);
                //Projection onto the l2 ball
                norm_res = cblas_dnrm2(ny, (double*)dummy, 1);
                norm_res = min(1.0,epsilon/norm_res);
                t = (1.0+sqrt(1.0+4.0*told*told))/2.0;
                temp = (told-1.0)/t;

                //FISTA update
                //u = v
                //v = mu*(1.0 - norm_res)*dummy
                //u = v + temp*(v - u)
                cblas_dcopy(ny, (double*)v, 1, (double*)u, 1);
                cblas_dcopy(ny, (double*)dummy, 1, (double*)v, 1);
                cblas_dscal(ny, (1.0 - norm_res)/mu, (double*)v,1);
                //Backtraking
                if (param.bcktrk == 1 && param.pos == 0){
                    mu = sopt_utility_btrackr((double*)v,
                                    (double*)r, 
                                    (double*)w, 
                                    (double*)dummy,
                                    (double*)xaux,  
                                    (double*)xout, 
                                    (double*)xin, 
                                    A, 
                                    A_data,
                                    At, 
                                    At_data,
                                    nx, 
                                    ny,
                                    mu,
                                    epsilon);
                }
                cblas_dscal(ny, -temp, (double*)u,1);
                cblas_daxpy(ny, (1.0 + temp), (double*)v, 1, (double*)u, 1);

                //Current estimate
                // xout = xin - At(u)
                At(xout, u, At_data);
                cblas_dscal(nx, -1.0, (double*)xout,1);
                cblas_daxpy(nx, 1.0, (double*)xin, 1, (double*)xout, 1); 
                //Positivity constraint
                if (param.pos==1){
                    sopt_utility_projposr((double*)xout, (double*)xout, nx);
                }
                //Residual norm
                A(dummy, xout, A_data);
                cblas_daxpy(ny, -1.0, (double*)y, 1, (double*)dummy, 1);
                norm_res = cblas_dnrm2(ny, (double*)dummy, 1);
            }
            else {
                //Copy of u and dummy
                //r = dummy
                cblas_zcopy(ny, dummy, 1, r, 1);
                //w = u
                cblas_zcopy(ny, u, 1, w, 1);
                //Rescaling
                //dummy = nu*u + dummy
                nu = mu + 0.0*I;
                cblas_zaxpy(ny, (void*)&nu, u, 1, dummy, 1);
                //Projection onto the l2 ball
                norm_res = cblas_dznrm2(ny, dummy, 1);
                norm_res = min(1.0,epsilon/norm_res);
                t = (1.0+sqrt(1.0+4.0*told*told))/2.0;
                temp = (told-1.0)/t;

                //FISTA update
                //u = v
                //v = mu*(1.0 - norm_res)*dummy
                //u = v + temp*(v - u)
                cblas_zcopy(ny, v, 1, u, 1);
                cblas_zcopy(ny, dummy, 1, v, 1);
                cblas_zdscal(ny, (1.0 - norm_res)/mu, v,1);
                //Backtraking
                if (param.bcktrk == 1 && param.real == 0 && param.pos == 0){
                    mu = sopt_utility_btrackc((complex double*)v,
                                    (complex double*)r, 
                                    (complex double*)w, 
                                    (complex double*)dummy,
                                    (complex double*)xaux,  
                                    (complex double*)xout, 
                                    (complex double*)xin, 
                                    A, 
                                    A_data,
                                    At, 
                                    At_data,
                                    nx, 
                                    ny,
                                    mu,
                                    epsilon);
                }
                alpha = (1.0 + temp) + 0.0*I;
                cblas_zdscal(ny, -temp, u,1);
                cblas_zaxpy(ny, (void*)&alpha, v, 1, u, 1);

                //Current estimate
                // xout = xin - At(u)
                At(xout, u, At_data);
                alpha = 1.0 + 0.0*I;
                cblas_zdscal(nx, -1.0, xout,1);
                cblas_zaxpy(nx, (void*)&alpha, xin, 1, xout, 1); 
                //Reality constraint
                if (param.real==1){
                    sopt_utility_projreal((complex double*)xout, (complex double*)xout, nx);
                }
                //Positivity constraint
                if (param.pos==1){
                    sopt_utility_projposc((complex double*)xout, (complex double*)xout, nx);
                }
                //Residual norm
                A(dummy, xout, A_data);
                alpha = -1.0 +0.0*I;
                cblas_zaxpy(ny, (void*)&alpha, y, 1, dummy, 1);
                norm_res = cblas_dznrm2(ny, dummy, 1);
            }
            
            //Update variables
            iter++;
            told = t;
        }
         
    }
    free(r);
    free(w);
    free(xaux);
    //Log
    if (param.verbose > 0){
        printf("Projection L2 ball: epsilon = %e, residual = %e, %s, num. iter = %i \n ", epsilon, norm_res, crit, iter);
    }
    
}

/*!
 * This function computes the prox operator of the weighted l2
 * norm for the input vector \f$ x \f$. It solves the problem:
 * \f[
 * min_{z} 0.5||x - z||_2^2 + \lambda ||W\Psi^\dagger z||_2,
 * \f]
 * where \f$ \Psi \in C^{N_x \times N_r} \f$ is the sparsifying operator.
 * The solution is denoted \f$x^\star \in C^{N_x}\f$.
 * \param[out] xout Output solution (\f$ x^\star \f$).
 * \param[in] xin Input vector (\f$ x \f$).
 * \param[in] nx Dimension of the signal (\f$N_x\f$).
 * \param[in] nr Dimension of the signal in the representation domain
 * (\f$N_r\f$).
 * \param[in] Psi Pointer to the synthesis sparsity operator \f$\Psi\f$.
 * \param[in] Psi_data Data structure associated to \f$\Psi\f$.
 * \param[in] Psit Pointer to the analysis sparsity operator
 * \f$\Psi^\dagger\f$.
 * \param[in] Psit_data Data structure associated to \f$\Psi^\dagger\f$.
 * \param[in] weights weights vector.
 * \param[in] lambda Convergence parameter \f$\lambda\f$.
 * \param[in] real_data Flag for real or complex data. 1 if real 0 complex.
 * \param[in] param Data structure with the parameters of
 *            the optimization.
 * \param[in] dummy Auxiliary vector. Dimension: nr.
 * \param[in] sol Auxiliary vector. Dimension: nx.
 * \param[in] u Auxiliary vector. Dimension: nr.
 */
void sopt_prox_wl2(void *xout, 
                  void *xin,
                  int nx,
                  int nr,
                  void (*Psi)(void *out, void *in, void **data), 
                  void **Psi_data,
                  void (*Psit)(void *out, void *in, void **data), 
                  void **Psit_data,
                  double *weights,
                  double lambda,
                  int real_data,
                  sopt_prox_wl2param param,
                  void *dummy,
                  void *sol,
                  void *u){
    
     int i;
     int iter;
     double obj;
     double prev_ob;
     double rel_ob;
     double mu;
     double temp;
     char crit[8];
     complex double alpha;
     //complex double beta;
     
     mu = 1.0/param.nu;
     
     
     
     if ((param.tight==1)&&(param.pos==0)){
         //Tight frame case
         Psit(dummy, xin, Psit_data);
         if (real_data == 1){
            for (i=0; i < nr; i++) {
                *((double*)dummy + i) = 
                ((1.0/(1.0 + 2.0*lambda*param.nu*weights[i])) - 1.0)**((double*)dummy + i);
            }
            Psi(xout, dummy, Psi_data);
            //xout = xin + mu*xout;
            alpha = 1.0 + 0.0*I;
            cblas_dscal(nx, mu, (double*)xout,1);
            cblas_daxpy(nx, 1.0, (double*)xin, 1, (double*)xout, 1);
            //Objective evaluation
            Psit(dummy, xout, Psit_data);
            temp = sopt_utility_l1normr((double*)dummy, weights, nr);
            cblas_dcopy(nx, (void*)xout, 1, (double*)sol, 1);
            cblas_daxpy(nx, -1.0, (double*)xin, 1, (double*)sol, 1);
            obj = cblas_dnrm2(nx, (double*)sol, 1);
         }
         else{
            for (i=0; i < nr; i++) {
                *((complex double*)dummy + i) = 
                ((1.0/(1.0 + 2.0*lambda*param.nu*weights[i])) - 1.0)**((complex double*)dummy + i);
            }
            Psi(xout, dummy, Psi_data);
            //xout = xin + mu*xout;
            alpha = 1.0 + 0.0*I;
            cblas_zdscal(nx, mu, (void*)xout,1);
            cblas_zaxpy(nx, (void*)&alpha, (void*)xin, 1, (void*)xout, 1);
            //Objective evaluation
            Psit(dummy, xout, Psit_data);
            temp = sopt_utility_l1normc((complex double*)dummy, weights, nr);
            cblas_zcopy(nx, (void*)xout, 1, (void*)sol, 1);
            alpha = -1.0 + 0.0*I;
            cblas_zaxpy(nx, (void*)&alpha, (void*)xin, 1, (void*)sol, 1);
            obj = cblas_dznrm2(nx, (void*)sol, 1);
         }
         obj = 0.5*obj*obj + lambda*temp*temp;
         iter = 1;
         strcpy(crit, "REL_OBJ");
     }
     else{
         //Initializations non tight frame case
         
         obj = 0.0;
         prev_ob = 0.0;
         rel_ob = 1.0;
         iter = 0;
         alpha = 0.0 + 0.0*I;
         
         if (real_data == 1){
            //xout = xin
            cblas_dcopy(nx, (double*)xin, 1, (double*)xout, 1);
            //u = 0
            for (i=0; i < nr; i++) {
                *((double*)u + i) = 0.0;
            }
         }
         else{
            //xout = xin
            cblas_zcopy(nx, xin, 1, xout, 1);
            //u = 0
            for (i=0; i < nr; i++) {
                *((complex double*)u + i) = alpha;
            }
         }
         if (param.verbose > 1){
             printf("Proximal L2 operator:\n ");
         }
         while (1){
             //Objective evaluation
             Psit(dummy, xout, Psit_data);
             if (real_data == 1){
                temp = sopt_utility_sql2normr((double*)dummy, weights, nr);
                cblas_dcopy(nx, (double*)xout, 1, (double*)sol, 1);
                alpha = -1.0 + 0.0*I;
                cblas_daxpy(nx, -1.0, (double*)xin, 1, (double*)sol, 1);
                obj = cblas_dnrm2(nx, (double*)sol, 1);
             }
             else {
                temp = sopt_utility_sql2normc((complex double*)dummy, weights, nr);
                cblas_zcopy(nx, xout, 1, sol, 1);
                alpha = -1.0 + 0.0*I;
                cblas_zaxpy(nx, (void*)&alpha, xin, 1, sol, 1);
                obj = cblas_dznrm2(nx, sol, 1);
             }
             obj = 0.5*obj*obj + lambda*temp;
             rel_ob = fabs(obj-prev_ob)/obj;
             //Log
             if (param.verbose > 1){
                 printf("Iter %i: obj = %e, rel obj = %e \n ", iter, obj, rel_ob);
             }
             //Stopping criteria
             if (rel_ob < param.rel_obj){
                 strcpy(crit, "REL_OBJ");
                 break;
             }
             if (iter > param.max_iter){
                 strcpy(crit, "MAX_ITE");
                 break;
             }
             //Dual forward-backward
             if (real_data == 1){
                //u = param.nu*u + dummy
                cblas_dscal(nr, param.nu, (double*)u,1);
                cblas_daxpy(nr, 1.0, (double*)dummy, 1, (double*)u, 1);
                //L2 prox
                for (i=0; i < nr; i++) {
                    *((double*)dummy + i) = 
                    *((double*)u + i)/(1.0 + 2.0*lambda*param.nu*weights[i]);
                }
                //u = mu*(u - dummy)
                cblas_daxpy(nr, -1.0, (double*)dummy, 1, (double*)u, 1);
                cblas_dscal(nr, mu, (double*)u,1);
                //xout = xin - Psi(u)
                Psi(xout, u, Psi_data);
                cblas_dscal(nx, -1.0, (double*)xout,1);
                cblas_daxpy(nx, 1.0, (double*)xin, 1, (double*)xout, 1);
                //Positivity constraint
                if (param.pos==1){
                    sopt_utility_projposr((double*)xout, (double*)xout, nx);
                }
             }
             else {
                //u = param.nu*u + dummy
                alpha = 1.0 + 0.0*I;
                cblas_zdscal(nr, param.nu, (void*)u,1);
                cblas_zaxpy(nr, (void*)&alpha, (void*)dummy, 1, (void*)u, 1);
                //L2 prox
                for (i=0; i < nr; i++) {
                    *((complex double*)dummy + i) = 
                    *((complex double*)u + i)/(1.0 + 2.0*lambda*param.nu*weights[i]);
                }
                //u = mu*(u - dummy)
                alpha = -1.0 + 0.0*I;
                cblas_zaxpy(nr, (void*)&alpha, (void*)dummy, 1, (void*)u, 1);
                cblas_zdscal(nr, mu, (void*)u,1);
                //xout = xin - Psi(u)
                Psi(xout, u, Psi_data);
                alpha = 1.0 + 0.0*I;
                cblas_zdscal(nx, -1.0, (void*)xout,1);
                cblas_zaxpy(nx, (void*)&alpha, (void*)xin, 1, (void*)xout, 1);
                //Positivity constraint
                if (param.pos==1){
                    sopt_utility_projposc((complex double*)xout, (complex double*)xout, nx);
                }
             }
             //Update
             iter++;
             prev_ob = obj;  
         } 
     }
    if (param.verbose > 0){
        printf("Prox L2: objective = %e, %s, num. iter = %i \n ", obj, crit, iter);
    }
    
}




