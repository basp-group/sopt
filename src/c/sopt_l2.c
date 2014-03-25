//
//  sopt_l2.c
//  
//
//  Created by Rafael Carrillo on 10/11/12.
//
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include "sopt_utility.h"
#include "sopt_error.h"
#include "sopt_l2.h"

/*!
 * This function solves the problem:
 * \f[
 * \min_{x} ||W \Psi^\dagger x||_2 \quad \mbox{s.t.} \quad ||y - A x||_2 < \epsilon,
 * \f]
 * where \f$ \Psi \in C^{N_x \times N_r} \f$ is the sparsifying
 * operator, \f$ W  \in R_{+}^{N_x}\f$ is the diagonal weight matrix,
 * \f$A \in C^{N_y \times N_x}\f$ is the measurement operator,
 * \f$\epsilon\f$ is a noise tolerance and \f$y\in C^{Ny}\f$ is the
 * measurement vector.  The solution is denoted \f$x^\star \in
 * C^{N_x}\f$.
 *
 * \note 
 * The solver can be used to solve the analysis based problem, as
 * written, or the synthesis based problem by mapping \f$A \rightarrow
 * A \Psi\f$ and \f$\Psi^\dagger \rightarrow I\f$, where \f$I \f$ is
 * the identity matrix.
 *
 * \param[in,out] xsol Solution (\f$ x^\star \f$). It stores the initial solution,
 *               which should be set as an input to the function, 
 *               and it is modified when the finall solution is found.
 * \param[in] nx Dimension of the signal (\f$N_x\f$).
 * \param[in] A Pointer to the measurement operator \f$A\f$.
 * \param[in] A_data Data structure associated to measurement operator
 * \f$A\f$.
 * \param[in] At Pointer to the adjoint of the measurement operator
 * \f$A^\dagger\f$.
 * \param[in] At_data Data structure associated to the adjoint of the
 * measurement operator \f$A^\dagger\f$.
 * \param[in] Psi Pointer to the synthesis sparsity operator \f$\Psi\f$.
 * \param[in] Psi_data Data structure associated to the synthesis
 * sparsity operator \f$\Psi\f$.
 * \param[in] Psit Pointer to the analysis sparsity operator
 * \f$\Psi^\dagger\f$.
 * \param[in] Psit_data Data structure associated to the analysis
 * sparisty operator \f$\Psi^\dagger\f$.
 * \param[in] nr Dimension of the signal in the representation domain
 * (\f$N_r\f$).
 * \param[in] y Measurement vector (\f$y\f$).
 * \param[in] ny Dimension of measurement vector, i.e. number of
 *             measurements (\f$N_y\f$).
 * \param[in] weights Weights for the weighted L1 problem. Vector storing
 *            the main diagonal of \f$ W  \f$.
 * \param[in] param Data structure with the parameters of
 *            the optimization (including \f$\epsilon\f$).
 */
void sopt_l2_solver(void *xsol,
                    int nx,
                    void (*A)(void *out, void *in, void **data), 
                    void **A_data,
                    void (*At)(void *out, void *in, void **data), 
                    void **At_data,
                    void (*Psi)(void *out, void *in, void **data), 
                    void **Psi_data,
                    void (*Psit)(void *out, void *in, void **data), 
                    void **Psit_data,
                    int nr,
                    void *y,
                    int ny,
                    double *weights,
                    sopt_wl2_param param) {
    
    int i;
    int iter;
    double obj;
    double prev_ob;
    double rel_ob;
    char crit[8];
    complex double alpha;
    
    //Local memory
    void *dummy;
    void *xhat;
    void *temp;
    void *res;

    //Memory for L2 prox
    void *sol1;
    void *u1;

    //Memory for L2b prox
    void *u2;
    void *v2;
    
     if (param.real_out == 1){
        //Local
        dummy = malloc(nr * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(dummy);
        xhat = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(xhat);
        //L1 prox
        u1 = malloc(nr * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(u1);
    }
    else{
        //Local
        dummy = malloc(nr * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(dummy);
        xhat = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(xhat);
        //L1 prox
        u1 = malloc(nr * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(u1);
    }

    if (param.real_out == 1 && param.real_meas == 1) {
        temp = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(temp);
        sol1 = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(sol1);
    }
    else {
        temp = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(temp);
        sol1 = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(sol1);
    }

    if (param.real_meas == 1){
        //Local
        res = malloc(ny * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(res);
        //L2b prox
        u2 = malloc(ny * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(u2);
        v2 = malloc(ny * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(v2); 
    }
    else{
        //Local
        res = malloc(ny * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(res);
        //L2b prox
        u2 = malloc(ny * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(u2);
        v2 = malloc(ny * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(v2);
    }

        
    //Initializations
    iter = 1;
    prev_ob = 0.0;
    //xhat = xsol
    if (param.real_out == 1){
        cblas_dcopy(nx, (double*)xsol, 1, (double*)xhat, 1);
    }
    else{
        cblas_zcopy(nx, xsol, 1, xhat, 1);
    }
    
    //Log
    if (param.verbose > 1){
        printf("L1 solver: \n ");
    }
    while (1){
        //Log
        if (param.verbose > 1){
            printf("Iteration %i:\n ", iter);
        }
        //Projection onto the L2 ball
        if (param.real_out == 1 && param.real_meas == 0){
            for (i = 0; i < nx; i++){
                *((complex double*)sol1 + i) = *((double*)xhat + i) + 0.0*I;
            }
            sopt_prox_l2b(temp, sol1, nx, y, ny,
                      A, A_data, At, At_data,
                      param.epsilon, param.real_meas,
                      param.paraml2b, res, u2, v2);
            for (i = 0; i < nx; i++){
                 *((double*)xsol + i) = creal(*((complex double*)temp + i));
            }
        }
        else {
            sopt_prox_l2b(xsol, xhat, nx, y, ny,
                      A, A_data, At, At_data,
                      param.epsilon, param.real_meas,
                      param.paraml2b, res, u2, v2);
        }
        
        //Objective evaluation
        Psit(dummy, xsol, Psit_data);
        if (param.real_out == 1){
            obj = sopt_utility_sql2normr((double*)dummy, weights, nr);
        }
        else {
            obj = sopt_utility_sql2normc((complex double*)dummy, weights, nr);
        }
    
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
            printf("Objective: obj value = %e, rel obj = %e \n ", obj, rel_ob);
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
        //Proximal L1 operator and DR recursion
        //xhat = 2*xsol - xhat 
        alpha = 2.0 + 0.0*I;
        if (param.real_out == 1){
            cblas_dscal(nx, -1.0, (double*)xhat,1);
            cblas_daxpy(nx, 2.0, (double*)xsol, 1, (double*)xhat, 1);
        }
        else{
            cblas_zdscal(nx, -1.0, xhat,1);
            cblas_zaxpy(nx, (void*)&alpha, xsol, 1, xhat, 1);
        }
        
        //Prox L2
        sopt_prox_wl2(temp, xhat, nx, nr,
                     Psi, Psi_data, Psit, Psit_data,
                     weights, param.gamma, param.real_out,
                     param.paramwl2, dummy, sol1, u1);
        //xhat = temp + xsol - xhat
        alpha = 1.0 + 0.0*I;
        if (param.real_out == 1){
            cblas_daxpy(nx, 1.0, (double*)xsol, 1, (double*)temp, 1);
            cblas_dscal(nx, -1.0, (double*)xhat,1);
            cblas_daxpy(nx, 1.0, (double*)temp, 1, (double*)xhat, 1);
        }
        else {
            cblas_zaxpy(nx, (void*)&alpha, xsol, 1, temp, 1);
            cblas_zdscal(nx, -1.0, xhat,1);
            cblas_zaxpy(nx, (void*)&alpha, temp, 1, xhat, 1);
        }
        
        //Update
        iter++;
        prev_ob = obj;
    }
    
    //Log
    if (param.verbose > 0){
        //L2 norm
        printf("Solution found \n");
        printf("Final L2 norm: %e\n ", obj);
        //Residual
        if (param.real_out == 1 && param.real_meas == 0){
            for (i = 0; i < nx; i++){
                *((complex double*)sol1 + i) = *((double*)xsol + i) + 0.0*I;
            }
            A(res, sol1, A_data);
        }
        else{
            A(res, xsol, A_data);
        }
        
        alpha = -1.0 + 0.0*I;
        if (param.real_meas == 1){
            cblas_daxpy(ny, -1.0, (double*)y, 1, (double*)res, 1);
            prev_ob = cblas_dnrm2(ny, (double*)res, 1);
        }
        else {
            cblas_zaxpy(ny, (void*)&alpha, y, 1, res, 1);
            prev_ob = cblas_dznrm2(ny, res, 1);
        }
        printf("epsilon = %e, residual = %e\n", param.epsilon, prev_ob);
        //Stopping criteria
        printf("%i iterations\n", iter);
        printf("Stopping criterion: %s \n\n ", crit);
    }
    
    //Free temporary memory
    free(dummy);
    free(xhat);
    free(temp);
    free(res);

    free(sol1);
    free(u1);

    free(u2);
    free(v2);

}


