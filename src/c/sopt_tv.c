
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <string.h>
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
#include "sopt_tv.h"

#define min(a,b) (a<b?a:b)
#define max(a,b) (a<b?b:a)

/*!
 * This function solves the problem:
 * \f[
 * \min_{x} ||x||_{TV} \quad \mbox{s.t.} \quad ||y - A x||_2 < \epsilon,
 * \f]
 * where \f$A \in C^{N_y \times N_x}\f$ is the measurement operator,
 * \f$\epsilon\f$ is a noise tolerance and \f$y\in C^{Ny}\f$ is the
 * measurement vector. We represent the solution image by the vector 
 * \f$x^\star \in C^{N_x}\f$, where \f$N_x=N_{x1}\times N_{x2}\f$.
 *
 * \param[in,out] xsol Solution (\f$ x^\star \f$). It stores the initial solution,
 *               which should be set as an input to the function,
 *               and it is modified when the finall solution is found.
 * \param[in] nx1 Dimension of the signal in the first dimension(\f$N_{x1}\f$).
 * \param[in] nx2 Dimension of the signal in the first dimension(\f$N_{x2}\f$).
 * \param[in] A Pointer to the measurement operator \f$A\f$.
 * \param[in] A_data Data structure associated to measurement operator
 * \f$A\f$.
 * \param[in] At Pointer to the adjoint of the measurement operator
 * \f$A^\dagger\f$.
 * \param[in] At_data Data structure associated to the adjoint of the
 * measurement operator \f$A^\dagger\f$.
 * \param[in] y Measurement vector (\f$y\f$).
 * \param[in] ny Dimension of measurement vector, i.e. number of
 *             measurements (\f$N_y\f$).
 * \param[in] param Data structure with the parameters of
 *            the optimization (including \f$\epsilon\f$).
 */
void sopt_tv_solver(void *xsol,
                    int nx1,
                    int nx2,
                    void (*A)(void *out, void *in, void **data), 
                    void **A_data,
                    void (*At)(void *out, void *in, void **data), 
                    void **At_data,
                    void *y,
                    int ny,
                    sopt_tv_param param){
    
    int i;
    int iter;
    int nx;
    double obj;
    double prev_ob;
    double rel_ob;
    char crit[8];
    complex double alpha;
    
    //Local memory
    void *xhat;
    void *temp;
    void *res;

    //Memory for the TV prox
    void *r;
    void *s;
    void *pold;
    void *qold;
    void *dx;
    void *dy;

    //Memory for L2b prox
    void *u2;
    void *v2;
    
    //Initializations
    nx = nx1*nx2;
    iter = 1;
    prev_ob = 0.0;

    if (param.real_out == 1){
        //Local
        xhat = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(xhat);
        //TV prox
        r = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(r);
        s = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(s);
        pold = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(pold);
        qold = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(qold);
        dy = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(dy);
    }
    else {
        //Local
        xhat = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(xhat);
        //TV prox
        r = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(r);
        s = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(s);
        pold = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(pold);
        qold = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(qold);
        dy = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(dy);
    }

    if (param.real_out == 1 && param.real_meas == 1) {
        temp = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(temp);
        dx = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(dx);
    }
    else {
        temp = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(temp);
        dx = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(dx);
    }

    if (param.real_meas == 1) {
        //Local
        res = malloc(ny * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(res);
        //L2b prox
        u2 = malloc(ny * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(u2);
        v2 = malloc(ny * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(v2);
    }
    else {
        //Local
        res = malloc(ny * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(res);
        //L2b prox
        u2 = malloc(ny * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(u2);
        v2 = malloc(ny * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(v2);
    }

    //xhat = xsol
    if (param.real_out == 1){
        cblas_dcopy(nx, (double*)xsol, 1, (double*)xhat, 1);
    }
    else{
        cblas_zcopy(nx, xsol, 1, xhat, 1);
    }
    
    //Log
    if (param.verbose > 1){
        printf("TV solver: \n ");
    }
    //Main loop
    while (1){
        if (param.verbose > 1){
            printf("Iteration %i:\n ", iter);
        }
        //Projection onto the L2 ball
        if (param.real_out == 1 && param.real_meas == 0){
            for (i = 0; i < nx; i++){
                *((complex double*)dx + i) = *((double*)xhat + i) + 0.0*I;
            }
            sopt_prox_l2b(temp, dx, nx, y, ny,
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
        if (param.real_out == 1){
            obj = sopt_utility_tvnormr((double*)xsol, nx1, nx2);
        }
        else {
            obj = sopt_utility_tvnormc((complex double*)xsol, nx1, nx2);
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
        //Proximal TV operator and DR recursion
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
        
        //Prox TV
        sopt_prox_tv(temp, xhat, nx1, nx2, param.gamma, param.real_out,
                     param.paramtv, r, s, pold, qold, dx, dy);
        
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
        //TV norm
        printf("\n Solution found \n");
        printf("Final TV norm: %e\n ", obj);
        //Residual
        if (param.real_out == 1 && param.real_meas == 0){
            for (i = 0; i < nx; i++){
                *((complex double*)dx + i) = *((double*)xsol + i) + 0.0*I;
            }
            A(res, dx, A_data);
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
    free(xhat);
    free(temp);
    free(res);

    free(r);
    free(s);
    free(pold);
    free(qold);
    free(dx);
    free(dy);

    free(u2);
    free(v2);

}


/*!
 * This function solves the problem:
 * \f[
 * \min_{x} ||x||_{TV} \quad \mbox{s.t.} \quad ||y - A x||_2 < \epsilon and x \geq 0,
 * \f]
 * where \f$A \in C^{N_y \times N_x}\f$ is the measurement operator,
 * \f$\epsilon\f$ is a noise tolerance and \f$y\in C^{Ny}\f$ is the
 * measurement vector. We represent the solution image by the vector 
 * \f$x^\star \in C^{N_x}\f$, where \f$N_x=N_{x1}\times N_{x2}\f$.
 *
 * \param[in,out] xsol Solution (\f$ x^\star \f$). It stores the initial solution,
 *               which should be set as an input to the function,
 *               and it is modified when the finall solution is found.
 * \param[in] nx1 Dimension of the signal in the first dimension(\f$N_{x1}\f$).
 * \param[in] nx2 Dimension of the signal in the first dimension(\f$N_{x2}\f$).
 * \param[in] A Pointer to the measurement operator \f$A\f$.
 * \param[in] A_data Data structure associated to measurement operator
 * \f$A\f$.
 * \param[in] At Pointer to the adjoint of the measurement operator
 * \f$A^\dagger\f$.
 * \param[in] At_data Data structure associated to the adjoint of the
 * measurement operator \f$A^\dagger\f$.
 * \param[in] y Measurement vector (\f$y\f$).
 * \param[in] ny Dimension of measurement vector, i.e. number of
 *             measurements (\f$N_y\f$).
 * \param[in] wt_dx Weight vector in x direction. Dimension: nx.
 * \param[in] wt_dy Weight vector in y direction. Dimension: nx.
 * \param[in] param Data structure with the parameters of
 *            the optimization (including \f$\epsilon\f$).
 */
void sopt_tv_sdmm(void *xsol,
                    int nx1,
                    int nx2,
                    void (*A)(void *out, void *in, void **data), 
                    void **A_data,
                    void (*At)(void *out, void *in, void **data), 
                    void **At_data,
                    void *y,
                    int ny,
                    double *wt_dx,
                    double *wt_dy,
                    sopt_tv_sdmmparam param){
    
    int i;
    int iter;
    int nx;
    double obj;
    double prev_ob;
    double rel_ob;
    double epsilon_up, epsilon_down;
    double res;
    char crit[8];
    complex double alpha;
    
    //Initializations
    nx = nx1*nx2;
    
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

    //Memory for the TV prox
    void *pold;
    void *qold;
    void *dx;
    void *dy;
    void *xaux;
    
    if (param.real_data == 1){
        //TV
        x1 = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(x1);
        s1 = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(s1);
        z1 = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(z1);

        pold = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(pold);
        qold = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(qold);
        dy = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(dy);
        dx = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(dx);
        xaux = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(xaux);
        
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
        //TV
        x1 = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(x1);
        s1 = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(s1);
        z1 = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(z1);

        pold = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(pold);
        qold = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(qold);
        dy = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(dy);
        dx = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(dx);
        xaux = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(xaux);
        
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
        for (i = 0; i < nx; i++){
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
        cblas_dcopy(nx, (double*)x3, 1, (double*)x1, 1);
       
    }
    else{
        for (i = 0; i < nx; i++){
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
        cblas_zcopy(nx, x3, 1, x1, 1);
        
    }
    
    //Log
    if (param.verbose > 1){
        printf("TV SDMM solver: \n ");
    }
    while (1){
        //Log
        if (param.verbose > 1){
            printf("Iteration %i:\n ", iter);
        }
        //Mixing step
        if (param.real_data == 1){
            //x1=x1-z1, x2=x2-z2, x3=x3-z3
            cblas_daxpy(nx, -1.0, (double*)z1, 1, (double*)x1, 1);
            cblas_daxpy(ny, -1.0, (double*)z2, 1, (double*)x2, 1);
            cblas_daxpy(nx, -1.0, (double*)z3, 1, (double*)x3, 1);
            //x3=x1+At(x2)+x3
            At(s3, x2, At_data);
            cblas_daxpy(nx, 1.0, (double*)s3, 1, (double*)x3, 1);
            cblas_daxpy(nx, 1.0, (double*)x1, 1, (double*)x3, 1);
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
            cblas_zaxpy(nx, (void*)&alpha, z1, 1, x1, 1);
            cblas_zaxpy(ny, (void*)&alpha, z2, 1, x2, 1);
            cblas_zaxpy(nx, (void*)&alpha, z3, 1, x3, 1);
            //x3=x1+At(x2)+x3
            At(s3, x2, At_data);
            alpha = 1.0 + 0.0*I;
            cblas_zaxpy(nx, (void*)&alpha, s3, 1, x3, 1);
            cblas_zaxpy(nx, (void*)&alpha, x1, 1, x3, 1);
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
        
        if (param.real_data == 1){
            obj = sopt_utility_wtvnormr((double*)xsol, wt_dx, wt_dy, nx1, nx2);
        }
        else {
            obj = sopt_utility_wtvnormc((complex double*)xsol, wt_dx, wt_dy, nx1, nx2);
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
            printf("Objective: TV norm = %e, rel obj = %e \n ", obj, rel_ob);
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

        //TV update
        //x1=Psit(xsol)
        if (param.real_data == 1){
            //z1=z1+xsol
            cblas_daxpy(nx, 1.0, (double*)xsol, 1, (double*)z1, 1);
            //x1=prox_TV(z1).
            //sopt_prox_tv(x1, z1, nx1, nx2, param.gamma, param.real_data,
                     //param.paramtv, s1, s3, pold, qold, dx, dy);
            sopt_prox_wtv(x1, z1, nx1, nx2, param.gamma, param.real_data,
                         param.paramtv, wt_dx, wt_dy, s1, s3, xaux, pold, qold, dx, dy);
            //z1=z1-x1
            cblas_daxpy(nx, -1.0, (double*)x1, 1, (double*)z1, 1);  
            //Log
            if (param.verbose > 1){
                cblas_dcopy(nx, (double*)xsol, 1, (double*)s1, 1);
                cblas_daxpy(nx, -1.0, (double*)x1, 1, (double*)s1, 1);
                res = cblas_dnrm2(nx, (double*)s1, 1);
                printf("Primal residual r1 = %e \n ", res);
            } 
        }
        else {
            //z1=z1+xsol
            alpha = 1.0 + 0.0*I;
            cblas_zaxpy(nx, (void*)&alpha, xsol, 1, z1, 1);
            //x1=prox_TV(z1). 
            //sopt_prox_tv(x1, z1, nx1, nx2, param.gamma, param.real_data,
                     //param.paramtv, s1, s3, pold, qold, dx, dy);
            sopt_prox_wtv(x1, z1, nx1, nx2, param.gamma, param.real_data,
                         param.paramtv, wt_dx, wt_dy, s1, s3, xaux, pold, qold, dx, dy);
            //z1=z1-x1
            alpha = -1.0 + 0.0*I;
            cblas_zaxpy(nx, (void*)&alpha, x1, 1, z1, 1);
            //Log
            if (param.verbose > 1){
                cblas_zcopy(nx, xsol, 1, s1, 1);
                alpha = -1.0 + 0.0*I;
                cblas_zaxpy(nx, (void*)&alpha, x1, 1, s1, 1);
                res = cblas_dznrm2(nx, s1, 1);
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
        //TV norm
        printf("Solution found \n");
        printf("Final TV norm: %e\n ", obj);
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
    free(pold);
    free(qold);
    free(dx);
    free(dy);
    free(xaux);

}

/*!
 * Reweighted L1 minimization function that uses an homotopy
 * continuation method to approximate the L0 norm of the magnitude of the gradient. 
 * It solves at each iteration the following problem::
 * \f[
 * \min_{x} ||x||_{W_t,TV} \quad \mbox{s.t.} \quad ||y - A x||_2 < \epsilon and x \geq 0,
 * \f]
 * where \f$ W_t  \in R_{+}^{N_x}\f$ is the diagonal weight matrix,
 * \f$A \in C^{N_y \times N_x}\f$ is the measurement operator,
 * \f$\epsilon\f$ is a noise tolerance, \f$y\in C^{N_y}\f$ is the
 * measurement vector and \f$ W_t \f$ is a diagonal weight matrix that changes
 * at every iteration. The solution is denoted \f$x^\star \in C^{N_x}\f$.
 *
 * \note
 * It uses the sdmm algorithm to solve the weighted TV problem.
 * The solver can be used to solve the weighted problem, as
 * written, or the un-weighted problem by setting the weight vectors,
 * wt_dx and wt_dy, both to one.
 *
 * \param[in,out] xsol Solution (\f$ x^\star \f$). It stores the initial solution,
 *               which should be set as an input to the function,
 *               and it is modified when the finall solution is found.
 * \param[in] nx1 Dimension of the signal in the first dimension(\f$N_{x1}\f$).
 * \param[in] nx2 Dimension of the signal in the first dimension(\f$N_{x2}\f$).
 * \param[in] A Pointer to the measurement operator \f$A\f$.
 * \param[in] A_data Data structure associated to measurement operator
 * \f$A\f$.
 * \param[in] At Pointer to the adjoint of the measurement operator
 * \f$A^\dagger\f$.
 * \param[in] At_data Data structure associated to the adjoint of the
 * measurement operator \f$A^\dagger\f$.
 * \param[in] y Measurement vector (\f$y\f$).
 * \param[in] ny Dimension of measurement vector, i.e. number of
 *             measurements (\f$N_y\f$).
 * \param[in] paramtv Data structure with the parameters for
 *            the TV sdmm solver (including \f$\epsilon\f$).
 * \param[in] paramrwtv Data structure with the parameters for
 *            the reweighted TV solver.
 */

void sopt_tv_rwsdmm(void *xsol,
                    int nx1,
                    int nx2,
                    void (*A)(void *out, void *in, void **data),
                    void **A_data,
                    void (*At)(void *out, void *in, void **data),
                    void **At_data,
                    void *y,
                    int ny,
                    sopt_tv_sdmmparam paramtv,
                    sopt_tv_rwparam paramrwtv){
    
    int iter;
    int i;
    int nx = nx1*nx2;
    double dist;
    double rel_dist;
    double delta;
    complex double alpha;
    char crit[8];
    double *wt_dx;
    double *wt_dy;
    void *dx;
    void *dy;
    void *sol_old;
    
    wt_dx = (double*)malloc(nx * sizeof(double));
    SOPT_ERROR_MEM_ALLOC_CHECK(wt_dx);
    wt_dy = (double*)malloc(nx * sizeof(double));
    SOPT_ERROR_MEM_ALLOC_CHECK(wt_dy);

    
    if (paramtv.real_data == 1){
        sol_old = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(sol_old);
        dx = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(dx);
        dy = malloc(nx * sizeof(double));
        SOPT_ERROR_MEM_ALLOC_CHECK(dy);
        
    }
    else {
        sol_old = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(sol_old);
        dx = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(dx);
        dy = malloc(nx * sizeof(complex double));
        SOPT_ERROR_MEM_ALLOC_CHECK(dy);
    }
    
    strcpy(crit, "MAX_ITE");
    
    //Initial solution
    if (paramrwtv.init_sol == 0){
        //xsol = 0;
        alpha = 0.0 + 0.0*I;
        if (paramtv.real_data == 1){
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
        for (i = 0; i < nx; i++){
            wt_dx[i] = 1.0;
            wt_dy[i] = 1.0;
        }
        sopt_tv_sdmm(xsol, nx1, nx2,
                     A, A_data, At, At_data,
                     y, ny, wt_dx, wt_dy, paramtv);
        
    }
    if (paramtv.real_data == 1){
        sopt_utility_projposr((double*)xsol, (double*)xsol, nx);
    }
    else{
        sopt_utility_projposc((complex double*)xsol, (complex double*)xsol, nx);
    }
    //Setting delta
    delta = 0.0;
    alpha = 0.0 + 0.0*I;
    
    if (paramtv.real_data == 1){
        for (i=0; i < nx; i++){
            delta += *((double*)xsol + i);
        }
    }
    else {
        for (i=0; i < nx; i++){
            alpha += *((complex double*)xsol + i);
        }
    }
    
    delta = delta/nx;
    alpha = alpha/nx;
    if (paramtv.real_data == 1){
        for (i=0; i < nx; i++){
            *((double*)sol_old + i) = *((double*)xsol + i) - delta;
        }
    }
    else {
        for (i=0; i < nx; i++){
            *((complex double*)sol_old + i)  = *((complex double*)xsol + i) - alpha;
        }
    }
    
    if (paramtv.real_data == 1){
        delta = cblas_dnrm2(nx, (double*)sol_old, 1);
    }
    else{
        delta = cblas_dznrm2(nx, sol_old, 1);
    }
    
    delta = delta/sqrt(nx);
    
    for (iter=1; iter <= paramrwtv.max_iter; iter++){
        //Verbose and check delta
        delta = max(paramrwtv.sigma, delta);
        
        //Update weights
        if (paramtv.real_data == 1){
            sopt_utility_gradientr((double*)dx, (double*)dy, (double*)xsol, nx1, nx2);
            for (i=0; i < nx; i++){
                *(wt_dx + i) = delta/(delta + fabs(*((double*)dx + i)));
                *(wt_dy + i) = delta/(delta + fabs(*((double*)dy + i)));
            }
        }
        else {
            sopt_utility_gradientc((complex double*)dx, (complex double*)dy, (complex double*)xsol, nx1, nx2);
            for (i=0; i < nx; i++){
                *(wt_dx + i) = delta/(delta + cabs(*((complex double*)dx + i)));
                *(wt_dy + i) = delta/(delta + cabs(*((complex double*)dy + i)));
            }
        }
        
        //sol_old = xsol
        if (paramtv.real_data == 1){
            cblas_dcopy(nx, (double*)xsol, 1, (double*)sol_old, 1);
        }
        else{
            cblas_zcopy(nx, xsol, 1, sol_old, 1);
        }
        
        //Log
        if (paramrwtv.verbose > 1){
            printf("RW iteration: %i\n", iter);
        }
        
        //Solve TV problem
        sopt_tv_sdmm(xsol, nx1, nx2,
                     A, A_data, At, At_data,
                     y, ny, wt_dx, wt_dy, paramtv);
        
        if (paramtv.real_data == 1){
            sopt_utility_projposr((double*)xsol, (double*)xsol, nx);
        }
        else{
            sopt_utility_projposc((complex double*)xsol, (complex double*)xsol, nx);
        }
        
        //Update variables and relative distance computation
        delta = delta/10;
        alpha = -1.0 + 0.0*I;
        if (paramtv.real_data == 1){
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
        if (paramrwtv.verbose > 1){
            printf("Relative distance: %e \n\n ", rel_dist);
        }
        //Stopping criteria
        if (rel_dist < paramrwtv.rel_var){
            strcpy(crit, "REL_DIS");
            break;
        }
        
    }
    
    free(wt_dx);
    free(wt_dy);
    free(dx);
    free(dy);
    free(sol_old);
    
    if (paramrwtv.verbose > 0){
        printf("Solution found \n");
        //Stopping criteria
        printf("%i RW iterations\n", iter);
        printf("Stopping criterion: %s \n\n ", crit);
    }
    
}

