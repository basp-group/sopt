#include "sopt_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include SOPT_BLAS_H
#include "sopt_utility.h"
#include "sopt_error.h"
#include "sopt_l1.h"

#define min(a,b) (a<b?a:b)
#define max(a,b) (a<b?b:a)

/*!
 * This function solves the problem:
 * \f[
 * \min_{x} ||W \Psi^\dagger x||_1 \quad \mbox{s.t.} \quad ||y - A x||_2 < \epsilon,
 * \f]
 * where \f$ \Psi \in C^{N_x \times N_r} \f$ is the sparsifying
 * operator, \f$ W  \in R_{+}^{N_x}\f$ is the diagonal weight matrix,
 * \f$A \in C^{N_y \times N_x}\f$ is the measurement operator,
 * \f$\epsilon\f$ is a noise tolerance and \f$y\in C^{N_y}\f$ is the
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
 * (\f$\Psi^\dagger x\f$).
 * \param[in] y Measurement vector (\f$y\f$).
 * \param[in] ny Dimension of measurement vector, i.e. number of
 *             measurements (\f$N_y\f$).
 * \param[in] weights Weights for the weighted L1 problem. Vector storing
 *            the main diagonal of \f$ W  \f$.
 * \param[in] param Data structure with the parameters of
 *            the optimization (including \f$\epsilon\f$).
 */
void sopt_l1_solver(void *xsol,
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
                    sopt_l1_param param) {
    
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

    //Memory for L1 prox
    void *sol1;
    void *u1;
    void *v1;

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
        v1 = malloc(nr * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(v1);
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
        v1 = malloc(nr * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(v1);
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
            obj = sopt_utility_l1normr((double*)dummy, weights, nr);
        }
        else {
            obj = sopt_utility_l1normc((complex double*)dummy, weights, nr);
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
        
        //Prox L1
        sopt_prox_l1(temp, xhat, nx, nr,
                     Psi, Psi_data, Psit, Psit_data,
                     weights, param.gamma, param.real_out,
                     param.paraml1, dummy, sol1, u1, v1);
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
        //L1 norm
        printf("Solution found \n");
        printf("Final L1 norm: %e\n ", obj);
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
    free(v1);

    free(u2);
    free(v2);

}

/*!
 * Reweighted L1 minimization function that uses an homotopy
 * continuation method to approximate the L0 norm. It solves at each
 * iteration the following problem::
 * \f[
 * \min_{x} ||W_t \Psi^\dagger x||_1 \quad \mbox{s.t.} \quad ||y - A x||_2 < \epsilon,
 * \f]
 * where \f$ \Psi \in C^{N_x \times N_r} \f$ is the sparsifying
 * operator, \f$ W  \in R_{+}^{N_x}\f$ is the diagonal weight matrix,
 * \f$A \in C^{N_y \times N_x}\f$ is the measurement operator,
 * \f$\epsilon\f$ is a noise tolerance, \f$y\in C^{N_y}\f$ is the
 * measurement vector and \f$ W_t \f$ is a diagonal weight matrix that changes 
 * at every iteration. The solution is denoted \f$x^\star \in C^{N_x}\f$.
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
 * (\f$\Psi^\dagger x\f$).
 * \param[in] y Measurement vector (\f$y\f$).
 * \param[in] ny Dimension of measurement vector, i.e. number of
 *             measurements (\f$N_y\f$).
 * \param[in] paraml1 Data structure with the parameters for
 *            the L1 solver (including \f$\epsilon\f$).
 * \param[in] paramrwl1 Data structure with the parameters for
 *            the reweighted L1 solver.
 */

 void sopt_l1_rwmin(void *xsol,
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
                    sopt_l1_param paraml1,
                    sopt_l1_rwparam paramrwl1){

    int iter;
    int i;
    double dist;
    double rel_dist;
    double delta;
    complex double alpha;
    char crit[8];
    double *weights;
    void *dummy;
    void *sol_old;

    weights = (double*)malloc(nr * sizeof(double));
    SOPT_ERROR_MEM_ALLOC_CHECK(weights);

    if (paraml1.real_out == 1){
        dummy = malloc(nr * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(dummy);
        sol_old = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(sol_old);

    }
    else {
        dummy = malloc(nr * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(dummy);
        sol_old = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(sol_old);
    }
    
    strcpy(crit, "MAX_ITE");
    
    //Initial solution
    if (paramrwl1.init_sol == 0){
        //xsol = 0;
        alpha = 0.0 + 0.0*I;
        if (paraml1.real_out == 1){
            for (i=0; i < nx; i++){
                *((double*)xsol + i) = 0.0;
            }
        }
        else {
            for (i=0; i < nx; i++){
                *((complex double*)xsol + i)  = alpha;
            }
        }
        //weights = 1;
        for (i = 0; i < nr; i++){
            weights[i] = 1.0; 
        }
        sopt_l1_solver(xsol, nx,
                   A, A_data, At, At_data,
                   Psi, Psi_data, Psit, Psit_data,
                   nr, y, ny, weights, paraml1);

    }
    //Setting delta
    Psit(dummy, xsol, Psit_data);
    delta = 0.0;
    alpha = 0.0 + 0.0*I;
    
    if (paraml1.real_out == 1){
        for (i=0; i < nr; i++){
            delta += *((double*)dummy + i);
        }
    }
    else {
        for (i=0; i < nr; i++){
            alpha += *((complex double*)dummy + i);
        }
    }

    delta = delta/nr;
    alpha = alpha/nr;
    if (paraml1.real_out == 1){
        for (i=0; i < nr; i++){
            *((double*)dummy + i) += -delta;
        }
    }
    else {
        for (i=0; i < nr; i++){
            *((complex double*)dummy + i)  += -alpha;
        }
    }

    if (paraml1.real_out == 1){
        delta = cblas_dnrm2(nr, (double*)dummy, 1);
    }
    else{
        delta = cblas_dznrm2(nr, dummy, 1);
    }
    
    delta = delta/sqrt(nr);

    for (iter=1; iter <= paramrwl1.max_iter; iter++){
        //Verbose and check delta
        delta = max(paramrwl1.sigma, delta);

        //Update weights
        Psit(dummy, xsol, Psit_data);
        if (paraml1.real_out == 1){
            for (i=0; i < nr; i++){
                *(weights + i) = delta/(delta + fabs(*((double*)dummy + i)));
            }
        }
        else {
            for (i=0; i < nr; i++){
                *(weights + i) = delta/(delta + cabs(*((complex double*)dummy + i)));
            }
        }
        
        //sol_old = xsol
        if (paraml1.real_out == 1){
            cblas_dcopy(nx, (double*)xsol, 1, (double*)sol_old, 1);
        }
        else{
            cblas_zcopy(nx, xsol, 1, sol_old, 1);
        }

        //Log
        if (paramrwl1.verbose > 1){
            printf("RW iteration: %i\n", iter);
        }

        //Solve l1 problem
        sopt_l1_solver(xsol, nx,
                   A, A_data, At, At_data,
                   Psi, Psi_data, Psit, Psit_data,
                   nr, y, ny, weights, paraml1);

        //Update variables and relative distance computation
        delta = delta/10;
        alpha = -1.0 + 0.0*I;
        if (paraml1.real_out == 1){
            rel_dist = cblas_dnrm2(nx, (double*)sol_old, 1);
            cblas_daxpy(nx, -1.0, (double*)xsol, 1, (double*)sol_old, 1);
            dist = cblas_dnrm2(nx, (double*)sol_old, 1);
        }
        else{
            rel_dist = cblas_dznrm2(nx, sol_old, 1);
            cblas_zaxpy(nx, (void*)&alpha, xsol, 1, sol_old, 1);
            dist = cblas_dznrm2(nx, sol_old, 1);
        }
        rel_dist = dist/rel_dist;
        //Log
        if (paramrwl1.verbose > 1){
            printf("Relative distance: %e \n\n ", rel_dist);
        }
        //Stopping criteria
        if (rel_dist < paramrwl1.rel_var){
            strcpy(crit, "REL_DIS");
            break;
        }

    }
    
    free(weights);
    free(dummy);
    free(sol_old);

    if (paramrwl1.verbose > 0){
             printf("Solution found \n");
             //Stopping criteria
             printf("%i RW iterations\n", iter);
             printf("Stopping criterion: %s \n\n ", crit);
    }

 }

 /*!
 * This function solves the problem:
 * \f[
 * \min_{x} ||W \Psi^\dagger x||_1 \quad \mbox{s.t.} \quad ||y - A x||_2 < \epsilon\ \mbox{and}\ x \geq 0,
 * \f]
 * where \f$ \Psi \in C^{N_x \times N_r} \f$ is the sparsifying
 * operator, \f$ W  \in R_{+}^{N_x}\f$ is the diagonal weight matrix,
 * \f$A \in C^{N_y \times N_x}\f$ is the measurement operator,
 * \f$\epsilon\f$ is a noise tolerance and \f$y\in C^{N_y}\f$ is the
 * measurement vector.  The solution is denoted \f$x^\star \in
 * C^{N_x}\f$. The Simultaneous Direction Method of Multipliers (SDMM)
 * is used to solve the optimization problem.
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
 * (\f$\Psi^\dagger x\f$).
 * \param[in] y Measurement vector (\f$y\f$).
 * \param[in] ny Dimension of measurement vector, i.e. number of
 *             measurements (\f$N_y\f$).
 * \param[in] weights Weights for the weighted L1 problem. Vector storing
 *            the main diagonal of \f$ W  \f$.
 * \param[in] param Data structure with the parameters of
 *            the optimization (including \f$\epsilon\f$).
 */
void sopt_l1_sdmm(void *xsol,
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
                    sopt_l1_sdmmparam param){
    
    int i;
    int iter;
    double obj;
    double prev_ob;
    double rel_ob;
    double epsilon_up, epsilon_down;
    double res;
    char crit[8];
    complex double alpha;
    
    //Local memory
    void *x1;
    void *s1;
    void *z1;
    void *x2;
    void *s2;
    void *z2;
    void *x3;
    void *s3;
    void *z3;
    
    if (param.real_data == 1){
        //L1
        x1 = malloc(nr * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(x1);
        s1 = malloc(nr * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(s1);
        z1 = malloc(nr * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(z1);
        
        //L2 ball constraint
        x2 = malloc(ny * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(x2);
        s2 = malloc(ny * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(s2);
        z2 = malloc(ny * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(z2);
        
        //Positivity constraint
        x3 = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(x3);
        s3 = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(s3);
        z3 = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(z3);
        
    }
    else{
        //L1
        x1 = malloc(nr * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(x1);
        s1 = malloc(nr * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(s1);
        z1 = malloc(nr * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(z1);
        
        //L2 ball constraint
        x2 = malloc(ny * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(x2);
        s2 = malloc(ny * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(s2);
        z2 = malloc(ny * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(z2);
        
        //Positivity constraint
        x3 = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(x3);
        s3 = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(s3);
        z3 = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(z3);
        
    }

    
    
        
    //Initializations
    iter = 1;
    prev_ob = 0.0;
    epsilon_up = (1.0 + param.epsilon_tol)*param.epsilon;
    epsilon_down = (1.0 - param.epsilon_tol)*param.epsilon;
    //Lagrange multipliers and dual variables
    if (param.real_data == 1){
        for (i = 0; i < nr; i++){
            *((double*)z1 + i) = 0.0;
        }
        for (i = 0; i < ny; i++){
            *((double*)z2 + i) = 0.0;
        }
        for (i = 0; i < nx; i++){
            *((double*)z3 + i) = 0.0;
        }
        sopt_utility_projposr((double*)x3, (double*)xsol, nx);
        A(x2, x3, A_data);
        Psit(x1, x3, Psit_data);
    }
    else{
        for (i = 0; i < nr; i++){
            *((complex double*)z1 + i) = 0.0 + 0.0*I;
        }
        for (i = 0; i < ny; i++){
            *((complex double*)z2 + i) = 0.0 + 0.0*I;
        }
        for (i = 0; i < nx; i++){
            *((complex double*)z3 + i) = 0.0 + 0.0*I;
        }
        sopt_utility_projposc((complex double*)x3, (complex double*)xsol, nx);
        A(x2, x3, A_data);
        Psit(x1, x3, Psit_data);
    }
    
    //Log
    if (param.verbose > 1){
        printf("L1 SDMM solver: \n ");
    }
    while (1){
        //Log
        if (param.verbose > 1){
            printf("Iteration %i:\n ", iter);
        }
        //Mixing step
        if (param.real_data == 1){
            //x1=x1-z1, x2=x2-z2, x3=x3-z3
            cblas_daxpy(nr, -1.0, (double*)z1, 1, (double*)x1, 1);
            cblas_daxpy(ny, -1.0, (double*)z2, 1, (double*)x2, 1);
            cblas_daxpy(nx, -1.0, (double*)z3, 1, (double*)x3, 1);
            //x3=Psi(x1)+At(x2)+x3
            At(s3, x2, At_data);
            cblas_daxpy(nx, 1.0, (double*)s3, 1, (double*)x3, 1);
            Psi(s3, x1, Psi_data);
            cblas_daxpy(nx, 1.0, (double*)s3, 1, (double*)x3, 1);
            //Conjugate gradient solver
            sopt_utility_cgsolr((double*)xsol,
                            (double*)x3, 
                            (double*)x2,
                            (double*)x1,
                            (double*)s1,
                            (double*)s3,
                            A, 
                            A_data,
                            At, 
                            At_data,
                            nx, 
                            ny,
                            param.cg_tol,
                            param.cg_max_iter,
                            param.verbose);
            
        }
        else {
            //x1=x1-z1, x2=x2-z2, x3=x3-z3
            alpha = -1.0 + 0.0*I;
            cblas_zaxpy(nr, (void*)&alpha, z1, 1, x1, 1);
            cblas_zaxpy(ny, (void*)&alpha, z2, 1, x2, 1);
            cblas_zaxpy(nx, (void*)&alpha, z3, 1, x3, 1);
            //x3=Psi(x1)+At(x2)+x3
            At(s3, x2, At_data);
            alpha = 1.0 + 0.0*I;
            cblas_zaxpy(nx, (void*)&alpha, s3, 1, x3, 1);
            Psi(s3, x1, Psi_data);
            cblas_zaxpy(nx, (void*)&alpha, s3, 1, x3, 1);
            //Conjugate gradient solver
            sopt_utility_cgsolc((complex double*)xsol,
                            (complex double*)x3, 
                            (complex double*)x2,
                            (complex double*)x1,
                            (complex double*)s1,
                            (complex double*)s3,
                            A, 
                            A_data,
                            At, 
                            At_data,
                            nx, 
                            ny,
                            param.cg_tol,
                            param.cg_max_iter,
                            param.verbose);
            
        }
        
        //Objective and residual evaluation
        //x1=Psit(xsol)
        Psit(x1, xsol, Psit_data);
        if (param.real_data == 1){
            obj = sopt_utility_l1normr((double*)x1, weights, nr);
            cblas_dcopy(nr, (double*)x1, 1, (double*)s1, 1);
        }
        else {
            obj = sopt_utility_l1normc((complex double*)x1, weights, nr);
            cblas_zcopy(nr, x1, 1, s1, 1);
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
        //Residual
        //x2=A(xsol)-y
        A(x2, xsol, A_data);
        alpha = -1.0 + 0.0*I;
        if (param.real_data == 1){
            cblas_dcopy(ny, (double*)x2, 1, (double*)s2, 1);
            cblas_daxpy(ny, -1.0, (double*)y, 1, (double*)x2, 1);
            res = cblas_dnrm2(ny, (double*)x2, 1);
        }
        else {
            cblas_zcopy(ny, x2, 1, s2, 1);
            cblas_zaxpy(ny, (void*)&alpha, y, 1, x2, 1);
            res = cblas_dznrm2(ny, x2, 1);
        }
        //Log
        if (param.verbose > 1){
            printf("Objective: L1 norm = %e, rel obj = %e \n ", obj, rel_ob);
            printf("Residual: epsilon = %e, residual = %e \n ", param.epsilon, res);
        }
        //Stopping criteria
        if (rel_ob < param.rel_obj && res >= epsilon_down && res <= epsilon_up){
            strcpy(crit, "REL_OBJ");
            break;
        }
        if (iter > param.max_iter){
            strcpy(crit, "MAX_ITE");
            break;
        }

        //L1 update
        //x1=Psit(xsol)
        if (param.real_data == 1){
            //z1=z1+x1
            cblas_daxpy(nr, 1.0, (double*)x1, 1, (double*)z1, 1);
            //x1=prox_L1(z1), Soft-thresholding
            for (i=0; i < nr; i++) {
                *((double*)x1 + i) = 
                sopt_utility_softthr(*((double*)z1 + i), param.gamma*weights[i]);
            }
            //z1=z1-x1
            cblas_daxpy(nr, -1.0, (double*)x1, 1, (double*)z1, 1);  
            //Log
            if (param.verbose > 1){
                cblas_daxpy(nr, -1.0, (double*)x1, 1, (double*)s1, 1);
                res = cblas_dnrm2(nr, (double*)s1, 1);
                printf("Primal residual r1 = %e \n ", res);
            } 
        }
        else {
            //z1=z1+x1
            alpha = 1.0 + 0.0*I;
            cblas_zaxpy(nr, (void*)&alpha, x1, 1, z1, 1);
            //x1=prox_L1(z1). Soft-thresholding
            for (i=0; i < nr; i++) {
                *((complex double*)x1 + i) = 
                sopt_utility_softthc(*((complex double*)z1 + i), param.gamma*weights[i]);
            }
            //z1=z1-x1
            alpha = -1.0 + 0.0*I;
            cblas_zaxpy(nr, (void*)&alpha, x1, 1, z1, 1);
            //Log
            if (param.verbose > 1){
                alpha = -1.0 + 0.0*I;
                cblas_zaxpy(nr, (void*)&alpha, x1, 1, s1, 1);
                res = cblas_dznrm2(nr, s1, 1);
                printf("Primal residual r1 = %e \n ", res);
            }
            
        }
        //L2 ball constraint update
        //x2=A(xsol)-y
        
        if (param.real_data == 1){
            //z2=z2+x2
            cblas_daxpy(ny, 1.0, (double*)x2, 1, (double*)z2, 1);
            
            //x2=prox_L2b(z2), projection onto the epsilon L2
            //constraint
            //x2=y+res*z2
            res = cblas_dnrm2(ny, (double*)z2, 1);
            res = min(1.0,param.epsilon/res);
            cblas_dcopy(ny, (double*)y, 1, (double*)x2, 1);
            cblas_daxpy(ny, res, (double*)z2, 1, (double*)x2, 1);
            //remove y to get z2=z2+A(xsol)
            cblas_daxpy(ny, 1.0, (double*)y, 1, (double*)z2, 1);

            //z2=z2-x2
            cblas_daxpy(ny, -1.0, (double*)x2, 1, (double*)z2, 1); 
            //Log
            if (param.verbose > 1){
                cblas_daxpy(ny, -1.0, (double*)x2, 1, (double*)s2, 1);
                res = cblas_dnrm2(ny, (double*)s2, 1);
                printf("Primal residual r2 = %e \n ", res);
            }
   
        }
        else {
            //z2=z2+x2
            alpha = 1.0 + 0.0*I;
            cblas_zaxpy(ny, (void*)&alpha, x2, 1, z2, 1);
            //x2=prox_L2b(z2), projection onto the epsilon L2
            //constraint
            //x2=y+res*z2
            res = cblas_dznrm2(ny, z2, 1);
            res = min(1.0,param.epsilon/res);
            cblas_zcopy(ny, y, 1, x2, 1);
            alpha = res + 0.0*I;
            cblas_zaxpy(ny, (void*)&alpha, z2, 1, x2, 1);
            //remove y to get z2=z2+A(xsol)
            alpha = 1.0 + 0.0*I;
            cblas_zaxpy(ny, (void*)&alpha, y, 1, z2, 1);
            //z2=z2-x2
            alpha = -1.0 + 0.0*I;
            cblas_zaxpy(ny, (void*)&alpha, x2, 1, z2, 1); 
            //Log
            if (param.verbose > 1){
                alpha = -1.0 + 0.0*I;
                cblas_zaxpy(ny, (void*)&alpha, x2, 1, s2, 1);
                res = cblas_dznrm2(ny, s2, 1);
                printf("Primal residual r2 = %e \n ", res);
            }
            
        }
        //Positivity constraint update
        if (param.real_data == 1){
            //z3=z3+xsol
            cblas_daxpy(nx, 1.0, (double*)xsol, 1, (double*)z3, 1);
            //x3=prox_p(z2), projection onto the positive orthant
            sopt_utility_projposr((double*)x3, (double*)z3, nx);
            //z3=z3-x3
            cblas_daxpy(nx, -1.0, (double*)x1, 1, (double*)z1, 1); 
            //Log
            if (param.verbose > 1){
                cblas_dcopy(nx, (double*)xsol, 1, (double*)s3, 1);
                cblas_daxpy(nx, -1.0, (double*)x3, 1, (double*)s3, 1);
                res = cblas_dnrm2(nx, (double*)s3, 1);
                printf("Primal residual r3 = %e \n ", res);
            }
            
        }
        else {
             //z3=z3+xsol
            alpha = 1.0 + 0.0*I;
            cblas_zaxpy(nx, (void*)&alpha, xsol, 1, z3, 1);
            //x3=prox_p(z2), projection onto the positive orthant
            sopt_utility_projposc((complex double*)x3, (complex double*)z3, nx);
            //z3=z3-x3
            alpha = -1.0 + 0.0*I;
            cblas_zaxpy(nx, (void*)&alpha, x3, 1, z3, 1); 
            //Log
            if (param.verbose > 1){
                cblas_zcopy(nx, xsol, 1, s3, 1);
                alpha = -1.0 + 0.0*I;
                cblas_zaxpy(nx, (void*)&alpha, x3, 1, s3, 1);
                res = cblas_dznrm2(nx, s3, 1);
                printf("Primal residual r3 = %e \n ", res);
            }
            
        }
        
        //Update
        iter++;
        prev_ob = obj;
    }
    
    //Log
    if (param.verbose > 0){
        //L1 norm
        printf("Solution found \n");
        printf("Final L1 norm: %e\n ", obj);
        //Residual
        printf("epsilon = %e, residual = %e\n", param.epsilon, res);
        //Stopping criteria
        printf("%i iterations\n", iter);
        printf("Stopping criterion: %s \n\n ", crit);
    }
    
    //Free temporary memory
    free(x1);
    free(x2);
    free(x3);
    free(s1);
    free(s2);
    free(s3);
    free(z1);
    free(z2);
    free(z3);

}

/*!
 * Reweighted L1 minimization function that uses an homotopy
 * continuation method to approximate the L0 norm. It solves at each
 * iteration the following problem::
 * \f[
 * \min_{x} ||W_t \Psi^\dagger x||_1 \quad \mbox{s.t.} \quad ||y - A x||_2 < \epsilon\ \mbox{and}\ x \geq 0,
 * \f]
 * where \f$ \Psi \in C^{N_x \times N_r} \f$ is the sparsifying
 * operator, \f$ W  \in R_{+}^{N_x}\f$ is the diagonal weight matrix,
 * \f$A \in C^{N_y \times N_x}\f$ is the measurement operator,
 * \f$\epsilon\f$ is a noise tolerance, \f$y\in C^{N_y}\f$ is the
 * measurement vector and \f$ W_t \f$ is a diagonal weight matrix that changes 
 * at every iteration. The solution is denoted \f$x^\star \in C^{N_x}\f$.
 * The Simultaneous Direction Method of Multipliers (SDMM)
 * is used to solve the optimization problem.
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
 * (\f$\Psi^\dagger x\f$).
 * \param[in] y Measurement vector (\f$y\f$).
 * \param[in] ny Dimension of measurement vector, i.e. number of
 *             measurements (\f$N_y\f$).
 * \param[in] paraml1 Data structure with the parameters for
 *            the L1 solver (including \f$\epsilon\f$).
 * \param[in] paramrwl1 Data structure with the parameters for
 *            the reweighted L1 solver.
 */
 void sopt_l1_rwsdmm(void *xsol,
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
                    sopt_l1_sdmmparam paraml1,
                    sopt_l1_rwparam paramrwl1){

    int iter;
    int i;
    double dist;
    double rel_dist;
    double delta;
    complex double alpha;
    char crit[8];
    double *weights;
    void *dummy;
    void *sol_old;

    weights = (double*)malloc(nr * sizeof(double));
    SOPT_ERROR_MEM_ALLOC_CHECK(weights);

    if (paraml1.real_data == 1){
        dummy = malloc(nr * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(dummy);
        sol_old = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(sol_old);

    }
    else {
        dummy = malloc(nr * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(dummy);
        sol_old = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(sol_old);
    }
    
    strcpy(crit, "MAX_ITE");
    
    //Initial solution
    if (paramrwl1.init_sol == 0){
        //xsol = 0;
        alpha = 0.0 + 0.0*I;
        if (paraml1.real_data == 1){
            for (i=0; i < nx; i++){
                *((double*)xsol + i) = 0.0;
            }
        }
        else {
            for (i=0; i < nx; i++){
                *((complex double*)xsol + i)  = alpha;
            }
        }
        //weights = 1;
        for (i = 0; i < nr; i++){
            weights[i] = 1.0; 
        }
        sopt_l1_sdmm(xsol, nx,
                   A, A_data, At, At_data,
                   Psi, Psi_data, Psit, Psit_data,
                   nr, y, ny, weights, paraml1);

    }
    if (paraml1.real_data == 1){
        sopt_utility_projposr((double*)xsol, (double*)xsol, nx);
    }
    else{
        sopt_utility_projposc((complex double*)xsol, (complex double*)xsol, nx);
    }
    //Setting delta
    Psit(dummy, xsol, Psit_data);
    delta = 0.0;
    alpha = 0.0 + 0.0*I;
    
    if (paraml1.real_data == 1){
        for (i=0; i < nr; i++){
            delta += *((double*)dummy + i);
        }
    }
    else {
        for (i=0; i < nr; i++){
            alpha += *((complex double*)dummy + i);
        }
    }

    delta = delta/nr;
    alpha = alpha/nr;
    if (paraml1.real_data == 1){
        for (i=0; i < nr; i++){
            *((double*)dummy + i) += -delta;
        }
    }
    else {
        for (i=0; i < nr; i++){
            *((complex double*)dummy + i)  += -alpha;
        }
    }

    if (paraml1.real_data == 1){
        delta = cblas_dnrm2(nr, (double*)dummy, 1);
    }
    else{
        delta = cblas_dznrm2(nr, dummy, 1);
    }
    
    delta = delta/sqrt(nr);

    for (iter=1; iter <= paramrwl1.max_iter; iter++){
        //Verbose and check delta
        delta = max(paramrwl1.sigma, delta);

        //Update weights
        Psit(dummy, xsol, Psit_data);
        if (paraml1.real_data == 1){
            for (i=0; i < nr; i++){
                *(weights + i) = delta/(delta + fabs(*((double*)dummy + i)));
            }
        }
        else {
            for (i=0; i < nr; i++){
                *(weights + i) = delta/(delta + cabs(*((complex double*)dummy + i)));
            }
        }
        
        //sol_old = xsol
        if (paraml1.real_data == 1){
            cblas_dcopy(nx, (double*)xsol, 1, (double*)sol_old, 1);
        }
        else{
            cblas_zcopy(nx, xsol, 1, sol_old, 1);
        }

        //Log
        if (paramrwl1.verbose > 1){
            printf("RW iteration: %i\n", iter);
        }

        //Solve l1 problem
        sopt_l1_sdmm(xsol, nx,
                   A, A_data, At, At_data,
                   Psi, Psi_data, Psit, Psit_data,
                   nr, y, ny, weights, paraml1);

        if (paraml1.real_data == 1){
            sopt_utility_projposr((double*)xsol, (double*)xsol, nx);
        }
        else{
            sopt_utility_projposc((complex double*)xsol, (complex double*)xsol, nx);
        }

        //Update variables and relative distance computation
        delta = delta/10;
        alpha = -1.0 + 0.0*I;
        if (paraml1.real_data == 1){
            rel_dist = cblas_dnrm2(nx, (double*)sol_old, 1);
            cblas_daxpy(nx, -1.0, (double*)xsol, 1, (double*)sol_old, 1);
            dist = cblas_dnrm2(nx, (double*)sol_old, 1);
        }
        else{
            rel_dist = cblas_dznrm2(nx, sol_old, 1);
            cblas_zaxpy(nx, (void*)&alpha, xsol, 1, sol_old, 1);
            dist = cblas_dznrm2(nx, sol_old, 1);
        }
        rel_dist = dist/rel_dist;
        //Log
        if (paramrwl1.verbose > 1){
            printf("Relative distance: %e \n\n ", rel_dist);
        }
        //Stopping criteria
        if (rel_dist < paramrwl1.rel_var){
            strcpy(crit, "REL_DIS");
            break;
        }

    }
    
    free(weights);
    free(dummy);
    free(sol_old);

    if (paramrwl1.verbose > 0){
             printf("Solution found \n");
             //Stopping criteria
             printf("%i RW iterations\n", iter);
             printf("Stopping criterion: %s \n\n ", crit);
    }

 }




/*!
 * This function solves the problem:
 * \f[
 * \min_{x} ||W \Psi^\dagger x||_1 \quad \mbox{s.t.} \quad ||y - A x||_2 < \epsilon\ \mbox{and}\ x \geq 0,
 * \f]
 * where \f$ \Psi \in C^{N_x \times N_r} \f$ is the sparsifying
 * operator, \f$ W \in R_{+}^{N_x}\f$ is the diagonal weight matrix,
 * \f$A \in C^{N_y \times N_x}\f$ is the measurement operator,
 * \f$\epsilon\f$ is a noise tolerance and \f$y\in C^{N_y}\f$ is the
 * measurement vector.  The solution is denoted \f$x^\star \in
 * C^{N_x}\f$. The Approximate Alternating Direction Method of
 * Multipliers (AADMM) is used to solve the optimization problem.
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
 * (\f$\Psi^\dagger x\f$).
 * \param[in] y Measurement vector (\f$y\f$).
 * \param[in] ny Dimension of measurement vector, i.e. number of
 *             measurements (\f$N_y\f$).
 * \param[in] weights Weights for the weighted L1 problem. Vector storing
 *            the main diagonal of \f$ W  \f$.
 * \param[in] param Data structure with the parameters of
 *            the optimization (including \f$\epsilon\f$).
 */
void sopt_l1_solver_padmm(void *xsol,
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
			  sopt_l1_param_padmm param){
    
    int i;
    int iter;
    double obj;
    double prev_ob;
    double rel_ob;
    char crit[8];
    double mu;
    double scale;
    double norm_res;

    // Local 
    void *z;
    void *s;
    void *res;
    void *y_temp;
    void *r;
    void *dummy;

    // Memory for L1 prox (so working memory doesn't need to be
    // reallocated for each iteration).
    void *sol1;  // size nx
    void *u1;    // size nr
    void *v1;    // size nr
      
    const complex double complex_unity_minus = -1.0 + 0.0*I;
    const complex double complex_unity = 1.0 + 0.0*I;
    const double tol = 1e-8;


complex double alpha;

    
// TODO: make input parameter
//    double epsilon_tol_scale = 1.001;
//    double beta = 0.9;
    



    
    // Allocate memory

    if (param.real_out == 1) {

      u1 = malloc(nr * sizeof(double));
      SOPT_ERROR_MEM_ALLOC_CHECK(u1);
      v1 = malloc(nr * sizeof(double));
      SOPT_ERROR_MEM_ALLOC_CHECK(v1);

    }
    else {
     
      u1 = malloc(nr * sizeof(complex double));
      SOPT_ERROR_MEM_ALLOC_CHECK(u1);
      v1 = malloc(nr * sizeof(complex double));
      SOPT_ERROR_MEM_ALLOC_CHECK(v1);      

    }

    if (param.real_out == 1 && param.real_meas == 1) {

      sol1 = malloc(nx * sizeof(double));
      SOPT_ERROR_MEM_ALLOC_CHECK(sol1);

      r = malloc(nx * sizeof(double));
      SOPT_ERROR_MEM_ALLOC_CHECK(r);

      dummy = malloc(nr * sizeof(double));
      SOPT_ERROR_MEM_ALLOC_CHECK(dummy);

    }
    else {
    
      sol1 = malloc(nx * sizeof(complex double));
      SOPT_ERROR_MEM_ALLOC_CHECK(sol1);

      r = malloc(nx * sizeof(complex double));
      SOPT_ERROR_MEM_ALLOC_CHECK(r);

      dummy = malloc(nr * sizeof(complex double));
      SOPT_ERROR_MEM_ALLOC_CHECK(dummy);

    }
    
    if (param.real_meas == 1) {

      z = calloc(ny, sizeof(double));  // Must be initalised to zero
      SOPT_ERROR_MEM_ALLOC_CHECK(z);
    
      s = malloc(ny * sizeof(double));
      SOPT_ERROR_MEM_ALLOC_CHECK(s);
    
      res = malloc(ny * sizeof(double));
      SOPT_ERROR_MEM_ALLOC_CHECK(res);

      y_temp = malloc(ny * sizeof(double)); 
      SOPT_ERROR_MEM_ALLOC_CHECK(y_temp);

    }
    else {

      z = calloc(ny, sizeof(complex double));  // Must be initalised to zero
      SOPT_ERROR_MEM_ALLOC_CHECK(z);
    
      s = malloc(ny * sizeof(complex double));
      SOPT_ERROR_MEM_ALLOC_CHECK(s);
    
      res = malloc(ny * sizeof(complex double));
      SOPT_ERROR_MEM_ALLOC_CHECK(res);

      y_temp = malloc(ny * sizeof(complex double)); 
      SOPT_ERROR_MEM_ALLOC_CHECK(y_temp);

    }

    // Initialise solution: xsol =  1/param.nu*At(y)
    At(sol1, y, At_data);
    assert(fabs(param.nu) > tol);
    mu = 1.0 / param.nu;
    if (param.real_out == 1 && param.real_meas == 1)
        cblas_dscal(nx, mu, (double*)sol1, 1);
    else
        cblas_zdscal(nx, mu, sol1, 1);

    // Initalise residuals: res = A * x_sol - y.
    A(res, sol1, A_data);
alpha = -1.0 + 0.0*I;    
    if (param.real_meas == 1) 
      cblas_daxpy(ny, -1.0, (double*)y, 1, (double*)res, 1);
    else
      cblas_zaxpy(ny, (void*)&alpha, y, 1, res, 1);
//cblas_zaxpy(ny, &complex_unity_minus, y, 1, res, 1);

    // Compute objective
    // dummy = Psit * xsol
    Psit(dummy, sol1, Psit_data);
    if (param.real_out == 1 && param.real_meas == 1)
      obj = sopt_utility_l1normr((double*)dummy, weights, nr);
    else
      obj = sopt_utility_l1normc((complex double*)dummy, weights, nr);
    
    // Initializations
    iter = 1;
    prev_ob = 0.0;
        
    // Log
    if (param.verbose > 1)
        printf("L1 solver: \n ");
    
    //Main loop
    while (1) {
      
        // Log
        if (param.verbose > 1)
            printf("Iteration %i:\n ", iter);
	
	// Slack variable update
	if (param.real_meas == 1) {

	  // s = z
	  cblas_dcopy(ny, (double*)z, 1, (double*)s, 1);	
	  // s = s + res
	  cblas_daxpy(ny, 1.0, (double*)res, 1, (double*)s, 1); 
	  // s = -s
	  cblas_dscal(ny, -1.0, (double*)s, 1);
	  //  s = s*min(1.0, epsilon/norm(s))	   
	  scale = cblas_dnrm2(ny, (double*)s, 1);
	  if (fabs(scale) > tol) 	  
	    scale = min(1.0, param.epsilon/scale);
	  else
	    scale = 1.0;	  
	  cblas_dscal(ny, scale, (double*)s, 1);

	}
	else {

	  // s = z
	  cblas_zcopy(ny, z, 1, s, 1);	 
	  // s = s + res
alpha = 1.0 + 0.0*I;    	  
cblas_zaxpy(ny, (void*)&alpha, res, 1, s, 1);
//cblas_zaxpy(ny, &complex_unity, res, 1, s, 1);	  
	  // s = -s
	  cblas_zdscal(ny, -1.0, s, 1);	    
	  //  s = s*min(1.0, epsilon/norm(s))
	  scale = cblas_dznrm2(ny, s, 1);	 
	  if (fabs(scale) > tol) 	  
	    scale = min(1.0, param.epsilon/scale);
	  else
	    scale = 1.0;
	  cblas_zdscal(ny, scale, s, 1);
	  
	}

	// Gradient formulation
	// y_temp = z + res + s
	if (param.real_meas == 1) {
	  cblas_dcopy(ny, (double*)z, 1, (double*)y_temp, 1);
	  cblas_daxpy(ny, 1.0, (double*)res, 1, (double*)y_temp, 1);
	  cblas_daxpy(ny, 1.0, (double*)s, 1, (double*)y_temp, 1);
	}
	else {
alpha = 1.0 + 0.0*I;    	  	  
	  cblas_zcopy(ny, z, 1, y_temp, 1);	 
cblas_zaxpy(ny, (void*)&alpha, res, 1, y_temp, 1);
cblas_zaxpy(ny, (void*)&alpha, s, 1, y_temp, 1);
//cblas_zaxpy(ny, &complex_unity, res, 1, y_temp, 1);
//cblas_zaxpy(ny, &complex_unity, s, 1, y_temp, 1);
	}
    //Todo: can we eliminate y_temp

	// r = At(z + res + s)
	At(r, y_temp, At_data);

	// Gradient descent
	// r = xsol - mu * r
	if (param.real_out == 1 && param.real_meas == 1) {
	  cblas_dscal(nx, -mu, (double*)r, 1);
	  cblas_daxpy(nx, 1.0, (double*)sol1, 1, (double*)r, 1);
	}
	else {
	  cblas_zdscal(nx, -mu, r, 1);
alpha = 1.0 + 0.0*I;    	  	  
cblas_zaxpy(nx, (void*)&alpha, sol1, 1, r, 1);
//cblas_zaxpy(nx, &complex_unity, xsol, 1, r, 1);
	}

    if (param.real_out == 1 && param.real_meas == 0){
        for (i = 0; i < nx; i++){
                *((double*)xsol + i) = creal(*((complex double*)sol1 + i));
        }
    }
    else if (param.real_out == 1 && param.real_meas == 1){

        for (i = 0; i < nx; i++){
                *((double*)xsol + i) = *((double*)sol1 + i);
        }

    }
    else{

        for (i = 0; i < nx; i++){
                *((complex double*)xsol + i) = *((complex double*)sol1 + i);
        }

    }


	// Prox L1 
        sopt_prox_l1(xsol, r, nx, nr,
                     Psi, Psi_data, Psit, Psit_data,
		     weights, param.gamma * mu, param.real_out, // Note mu weighting
                     param.paraml1, dummy, sol1, u1, v1);
	
	// Compute objective
	prev_ob = obj;
	Psit(dummy, xsol, Psit_data);
	if (param.real_out == 1)
	  obj = sopt_utility_l1normr((double*)dummy, weights, nr);
	else
	  obj = sopt_utility_l1normc((complex double*)dummy, weights, nr);
	
	// Residual

    //Copy xsol to sol1 for the different cases
    if (param.real_out == 1 && param.real_meas == 0){
        for (i = 0; i < nx; i++){
                 *((complex double*)sol1 + i) = *((double*)xsol + i) +0.0*I;
        }
    }
    else if (param.real_out == 1 && param.real_meas == 1){

        for (i = 0; i < nx; i++){
                *((double*)sol1 + i) = *((double*)xsol + i);
        }

    }
    else{

        for (i = 0; i < nx; i++){
                *((complex double*)sol1 + i) = *((complex double*)xsol + i);
        }

    }

	// Compute residuals: res = A * x_sol - y.
	A(res, sol1, A_data);
	if (param.real_meas == 1) {
	  cblas_daxpy(ny, -1.0, (double*)y, 1, (double*)res, 1);
	  norm_res = cblas_dnrm2(ny, (double*)res, 1);
	}
	else {
alpha = -1.0 + 0.0*I;    	  	  
cblas_zaxpy(ny, (void*)&alpha, y, 1, res, 1);	  
//cblas_zaxpy(ny, &complex_unity_minus, y, 1, res, 1);
	  norm_res = cblas_dznrm2(ny, res, 1);	 
	}	
	
	// Lagrange multipliers update
	// z = z + beta*(res + s);
	if (param.real_meas == 1) {
	  cblas_daxpy(ny, 1.0, (double*)s, 1, (double*)res, 1);    
	  cblas_dscal(ny, param.lagrange_update_scale, (double*)res, 1);
	  cblas_daxpy(ny, 1.0, (double*)res, 1, (double*)z, 1);    	
	}
	else {
alpha = 1.0 + 0.0*I;    	  	  
cblas_zaxpy(ny, (void*)(&alpha), s, 1, res, 1);
//cblas_zaxpy(ny, &complex_unity, s, 1, res, 1);
	  cblas_zdscal(ny, param.lagrange_update_scale, res, 1);
cblas_zaxpy(ny, (void*)&alpha, res, 1, z, 1);
//cblas_zaxpy(ny, &complex_unity, res, 1, z, 1);
	}

	// Check relative change of objective function
	if (obj > 0.0) 
	  rel_ob = fabs(obj - prev_ob)/obj;
	else
	  rel_ob = fabs(obj - prev_ob);
	
        // Log
        if (param.verbose > 1) {
	    printf("Objective: obj value = %e, rel obj = %e \n ", obj, rel_ob);
	    printf("Residuals: epsilon = %e, residual norm = %e \n ", param.epsilon, norm_res);
        }
	
        // Stopping criteria
        if (rel_ob < param.rel_obj
	    && norm_res <= param.epsilon * param.epsilon_tol_scale) {
            strcpy(crit, "REL_OBJ");
            break;
        }
        if (iter > param.max_iter) {
            strcpy(crit, "MAX_ITE");
            break;
        }

        // Update
        iter++;
        
    }
    
    //Log
    if (param.verbose > 0){
        //L1 norm
        printf("Solution found \n");
        printf("Final L1 norm: %e\n ", obj);
        //Residual        
        printf("epsilon = %e, residual = %e\n", param.epsilon, norm_res);
        //Stopping criteria
        printf("%i iterations\n", iter);
        printf("Stopping criterion: %s \n\n ", crit);
    }
    
    // Free memory

    free(z);
    free(s);
    free(res);
    free(y_temp);
    free(r);
    free(dummy);
            
    free(sol1);
    free(u1);
    free(v1);
    
}
