/*
 *  sopt_utility.c
 *  
 *
 *  Created by Rafael Carrillo on 8/27/12.
 *  Copyright 2012 EPFL. All rights reserved.
 *
 */
#include "sopt_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <complex.h> 
#include <string.h>
#include <math.h>
#include SOPT_BLAS_H
#include "sopt_error.h"
#include "sopt_utility.h"

#define min(a,b) (a<b?a:b)
#define max(a,b) (a<b?b:a)

/*! Scalar soft thresolding operator. Complex case.
 * \param[in] x Input value.
 * \param[in] T Thresold.
 * \retval xout Thresolded value.
 */
complex double sopt_utility_softthc(complex double x, double T){
  
  complex double sgn;
  complex double xout;
  double norm;
  
  norm = cabs(x);
  if (norm == 0)
    sgn = 0;
  else 
    sgn = x/norm;
  norm = max(0,norm-T);
  xout = sgn*norm;
  return xout;
}

/*! Scalar soft thresolding operator. Real case.
 * \param[in] x Input value.
 * \param[in] T Thresold.
 * \retval xout Thresolded value.
 */
double sopt_utility_softthr(double x, double T){
  
  double sgn;
  double xout;
  double norm;
  
  norm = fabs(x);
  if (norm == 0)
    sgn = 0;
  else 
    sgn = x/norm;
  norm = max(0,norm-T);
  xout = sgn*norm;
  return xout;
}

/*! Compute the weighted l1 norm of a complex vector.
 * \param[in] x Input vector.
 * \param[in] w Weights vector.
 * \param[in] dim Dimension of the input vector.
 * \retval norm Value of the l1 norm.
 */
double sopt_utility_l1normc(complex double *x, double *w, int dim){
  
  int i;
  double norm;
  
  norm = 0;
  
  for (i = 0; i < dim; i++) {
    norm += w[i]*cabs(x[i]);
  }
  
  return norm;
  
}

/*! Compute the weighted l1 norm of a real vector.
 * \param[in] x Input vector.
 * \param[in] w Weights vector.
 * \param[in] dim Dimension of the input vector.
 * \retval norm Value of the l1 norm.
 */
double sopt_utility_l1normr(double *x, double *w, int dim){
  
  int i;
  double norm;
  
  norm = 0;
  
  for (i = 0; i < dim; i++) {
    norm += w[i]*fabs(x[i]);
  }
  
  return norm;
  
}

/*! Compute the weighted squared l2 norm of a complex vector.
 * \param[in] x Input vector.
 * \param[in] w Weights vector.
 * \param[in] dim Dimention of the input vector. 
 * \retval norm Value of the l2 norm.
 */
double sopt_utility_sql2normc(complex double *x, double *w, int dim){
  
  int i;
  double norm;
  
  norm = 0;
  
  for (i = 0; i < dim; i++) {
    norm += w[i]*w[i]*(creal(x[i])*creal(x[i]) + cimag(x[i])*cimag(x[i]));
  }
  
  return norm;
  
}

/*! Compute the weighted squared l2 norm of a real vector.
 * \param[in] x Input vector.
 * \param[in] w Weights vector.
 * \param[in] dim Dimention of the input vector. 
 * \retval norm Value of the l2 norm.
 */
double sopt_utility_sql2normr(double *x, double *w, int dim){
  
  int i;
  double norm;
  
  norm = 0;
  
  for (i = 0; i < dim; i++) {
    norm += w[i]*w[i]*x[i]*x[i];
  }
  
  return norm;
  
}

/*! Compute the tv norm of a complex discrete image.
 * \param[in] x Input image.
 * \param[in] dim1 Number of elements in the first dimension.
 * \param[in] dim2 Number of elements in the second dimension.
 * \retval norm Value of the tv norm.
 */
double sopt_utility_tvnormc(complex double *x, int dim1, int dim2){
  
  int i;
  double norm, temp;
  complex double *dx, *dy;
  
  dx = (complex double*)malloc((dim1*dim2) * sizeof(complex double));
  SOPT_ERROR_MEM_ALLOC_CHECK(dx);
  dy = (complex double*)malloc((dim1*dim2) * sizeof(complex double));
  SOPT_ERROR_MEM_ALLOC_CHECK(dy);
  
  sopt_utility_gradientc(dx, dy, x, dim1, dim2);
  
  norm = 0;
  
  for (i = 0; i < dim1*dim2; i++) {
    temp = creal(dx[i])*creal(dx[i]) + cimag(dx[i])*cimag(dx[i]);
    temp += creal(dy[i])*creal(dy[i]) + cimag(dy[i])*cimag(dy[i]);
    norm += sqrt(temp);
  }
  
  //Free temporary memory
  free(dx);
  free(dy);
  
  return norm;
  
}

/*! Compute the tv norm of a real discrete image.
 * \param[in] x Input image.
 * \param[in] dim1 Number of elements in the first dimension.
 * \param[in] dim2 Number of elements in the second dimension.
 * \retval norm Value of the tv norm.
 */
double sopt_utility_tvnormr(double *x, int dim1, int dim2){
  
  int i;
  double norm, temp;
  double *dx, *dy;
  
  dx = (double*)malloc((dim1*dim2) * sizeof(double));
  SOPT_ERROR_MEM_ALLOC_CHECK(dx);
  dy = (double*)malloc((dim1*dim2) * sizeof(double));
  SOPT_ERROR_MEM_ALLOC_CHECK(dy);
  
  sopt_utility_gradientr(dx, dy, x, dim1, dim2);
  
  norm = 0;
  
  for (i = 0; i < dim1*dim2; i++) {
    temp = dx[i]*dx[i] + dy[i]*dy[i];
    norm += sqrt(temp);
  }
  
  //Free temporary memory
  free(dx);
  free(dy);
  
  return norm;
  
}

/*! Compute the weighted tv norm of a complex discrete image.
 * \param[in] x Input image.
 * \param[in] wt_dx Weights in the x direction.
 * \param[in] wt_dy Weights in the y direction.
 * \param[in] dim1 Number of elements in the first dimension.
 * \param[in] dim2 Number of elements in the second dimension.
 * \retval norm Value of the tv norm.
 */
double sopt_utility_wtvnormc(complex double *x, double *wt_dx, double *wt_dy, int dim1, int dim2){
  
  int i;
  double norm, temp;
  complex double *dx, *dy;
  
  dx = (complex double*)malloc((dim1*dim2) * sizeof(complex double));
  SOPT_ERROR_MEM_ALLOC_CHECK(dx);
  dy = (complex double*)malloc((dim1*dim2) * sizeof(complex double));
  SOPT_ERROR_MEM_ALLOC_CHECK(dy);
  
  sopt_utility_gradientc(dx, dy, x, dim1, dim2);

  for (i = 0; i < dim1*dim2; i++) {
    dx[i] = wt_dx[i]*dx[i];
    dy[i] = wt_dy[i]*dy[i];
  }
  
  norm = 0;
  
  for (i = 0; i < dim1*dim2; i++) {
    temp = creal(dx[i])*creal(dx[i]) + cimag(dx[i])*cimag(dx[i]);
    temp += creal(dy[i])*creal(dy[i]) + cimag(dy[i])*cimag(dy[i]);
    norm += sqrt(temp);
  }
  
  //Free temporary memory
  free(dx);
  free(dy);
  
  return norm;
  
}

/*! Compute the weighted tv norm of a real discrete image.
 * \param[in] x Input image.
 * \param[in] wt_dx Weights in the x direction.
 * \param[in] wt_dy Weights in the y direction.
 * \param[in] dim1 Number of elements in the first dimension.
 * \param[in] dim2 Number of elements in the second dimension.
 * \retval norm Value of the tv norm.
 */
double sopt_utility_wtvnormr(double *x, double *wt_dx, double *wt_dy, int dim1, int dim2){
  
  int i;
  double norm, temp;
  double *dx, *dy;
  
  dx = (double*)malloc((dim1*dim2) * sizeof(double));
  SOPT_ERROR_MEM_ALLOC_CHECK(dx);
  dy = (double*)malloc((dim1*dim2) * sizeof(double));
  SOPT_ERROR_MEM_ALLOC_CHECK(dy);
  
  sopt_utility_gradientr(dx, dy, x, dim1, dim2);

  for (i = 0; i < dim1*dim2; i++) {
    dx[i] = wt_dx[i]*dx[i];
    dy[i] = wt_dy[i]*dy[i];
  }
  
  norm = 0;
  
  for (i = 0; i < dim1*dim2; i++) {
    temp = dx[i]*dx[i] + dy[i]*dy[i];
    norm += sqrt(temp);
  }
  
  //Free temporary memory
  free(dx);
  free(dy);

  return norm;
  
}


/*! Compute the gradient of a complex discrete image.
 * \param[out] dx Vertical image gradient.
 * \param[out] dy Horizontal image gradient.
 * \param[in] xin Input image.
 * \param[in] dim1 Number of elements in the first dimension.
 * \param[in] dim2 Number of elements in the second dimension.
 */
void sopt_utility_gradientc(complex double *dx, complex double *dy, 
                   complex double *xin, int dim1, int dim2){
  
  int i, j;
  
  for (i = 0; i < dim1-1; i++) {
    for (j = 0; j < dim2-1; j++) {
      dx[i+dim1*j] = xin[(i+1)+dim1*j]-xin[i+dim1*j];
      dy[i+dim1*j] = xin[i+dim1*(j+1)]-xin[i+dim1*j];
    }
  }
  //Last column of dx
  for (i = 0; i < dim1-1; i++) {
    dx[i+dim1*(dim2-1)] = xin[(i+1)+dim1*(dim2-1)]-xin[i+dim1*(dim2-1)];
  }
  
  //Last row of dx all zeros
  for (j = 0; j < dim2; j++) {
    dx[(dim1-1)+dim1*j] = 0;
  }
  
  //Last row of dy
  for (j = 0; j < dim2-1; j++) {
    dy[(dim1-1)+dim1*j] = xin[(dim1-1)+dim1*(j+1)]-xin[(dim1-1)+dim1*j];
  }
  
  //Last column of dy all zeros
  for (i = 0; i < dim1; i++) {
    dy[i+dim1*(dim2-1)] = 0;
  }
  
}

/*! Compute the gradient of a real discrete image.
 * \param[out] dx Vertical image gradient.
 * \param[out] dy Horizontal image gradient.
 * \param[in] xin Input image.
 * \param[in] dim1 Number of elements in the first dimension.
 * \param[in] dim2 Number of elements in the second dimension.
 */
void sopt_utility_gradientr(double *dx, double *dy, double *xin, int dim1, int dim2){
  
  int i, j;
  
  for (i = 0; i < dim1-1; i++) {
    for (j = 0; j < dim2-1; j++) {
      dx[i+dim1*j] = xin[(i+1)+dim1*j]-xin[i+dim1*j];
      dy[i+dim1*j] = xin[i+dim1*(j+1)]-xin[i+dim1*j];
    }
  }
  //Last column of dx
  for (i = 0; i < dim1-1; i++) {
    dx[i+dim1*(dim2-1)] = xin[(i+1)+dim1*(dim2-1)]-xin[i+dim1*(dim2-1)];
  }
  
  //Last row of dx all zeros
  for (j = 0; j < dim2; j++) {
    dx[(dim1-1)+dim1*j] = 0;
  }
  
  //Last row of dy
  for (j = 0; j < dim2-1; j++) {
    dy[(dim1-1)+dim1*j] = xin[(dim1-1)+dim1*(j+1)]-xin[(dim1-1)+dim1*j];
  }
  
  //Last column of dy all zeros
  for (i = 0; i < dim1; i++) {
    dy[i+dim1*(dim2-1)] = 0;
  }
  
}

/*! Compute the divergence of a complex discrete image (adjoint of the gradient).
 * \param[out] xout Output image.
 * \param[in] dx Vertical image gradient.
 * \param[in] dy Horizontal image gradient.
 * \param[in] dim1 Number of elements in the first dimension.
 * \param[in] dim2 Number of elements in the second dimension.
 */
void sopt_utility_divergencec(complex double *xout, complex double *dx, 
                     complex double *dy, int dim1, int dim2){
  
  int i, j;
  
  for (j = 0; j < dim2; j++) {
    xout[dim1*j] = dx[dim1*j];
    for (i = 0; i < dim1-2; i++) {
      xout[(i+1)+dim1*j] = dx[(i+1)+dim1*j]-dx[i+dim1*j];
    }
    xout[(dim1-1)+dim1*j] = -dx[(dim1-2)+dim1*j];
  }
  
  for (i = 0; i < dim1; i++) {
    xout[i] += dy[i];
    for (j = 0; j < dim2-2; j++) {
      xout[i+dim1*(j+1)] += dy[i+dim1*(j+1)]-dy[i+dim1*j];
    }
    xout[i+dim1*(dim2-1)] -= dy[i+dim1*(dim2-2)];
  }
  
}

/*! Compute the divergence of a real discrete image (adjoint of the gradient).
 * \param[out] xout Output image.
 * \param[in] dx Vertical image gradient.
 * \param[in] dy Horizontal image gradient.
 * \param[in] dim1 Number of elements in the first dimension.
 * \param[in] dim2 Number of elements in the second dimension.
 */
void sopt_utility_divergencer(double *xout, double *dx, 
                     double *dy, int dim1, int dim2){
  
  int i, j;
  
  for (j = 0; j < dim2; j++) {
    xout[dim1*j] = dx[dim1*j];
    for (i = 0; i < dim1-2; i++) {
      xout[(i+1)+dim1*j] = dx[(i+1)+dim1*j]-dx[i+dim1*j];
    }
    xout[(dim1-1)+dim1*j] = -dx[(dim1-2)+dim1*j];
  }
  
  for (i = 0; i < dim1; i++) {
    xout[i] += dy[i];
    for (j = 0; j < dim2-2; j++) {
      xout[i+dim1*(j+1)] += dy[i+dim1*(j+1)]-dy[i+dim1*j];
    }
    xout[i+dim1*(dim2-1)] -= dy[i+dim1*(dim2-2)];
  }
  
}

/*! Compute the projection onto the positive orthant of a complex
 * vector x.
 * \param[out] xout Output positive vector.
 * \param[in] xin Input complex vector
 * \param[in] dim Dimension of the input vector
 */
void sopt_utility_projposc(complex double *xout, complex double *xin, int dim){
  
  int i;
  double temp;
  
  for (i = 0; i < dim; i++) {
    temp = creal(xin[i]);
    if (temp < 0.0)
      xout[i] = 0.0 + 0.0*I;
    else 
      xout[i] = temp + 0.0*I;
  }
  
}

/*! Compute the projection onto the positive orthant of a real
 * vector x.
 * \param[out] xout Output positive vector.
 * \param[in] xin Input complex vector
 * \param[in] dim Dimension of the input vector
 */
void sopt_utility_projposr(double *xout, double *xin, int dim){
  
  int i;
  
  for (i = 0; i < dim; i++) {
    if (xin[i] < 0.0)
      xout[i] = 0.0;
    else 
      xout[i] = xin[i];
  }
  
}

/*! Compute the projection onto the real orthant of a complex
 * vector x.
 * \param[out] xout Output real vector.
 * \param[in] xin Input complex vector
 * \param[in] dim Dimension of the input vector
 */
void sopt_utility_projreal(complex double *xout, complex double *xin, int dim){
  
  int i;
  
  for (i = 0; i < dim; i++) {
    xout[i] = (complex double)creal(xin[i]);
  }
  
}

/*! Compute the dot product of two complex
 * vectors.
 * \retval p Dot product output, complex number.
 * \param[in] xin Input complex vector.
 * \param[in] yin Input complex vector.
 * \param[in] dim Dimension of the input vectors.
 */
complex double sopt_utility_dotpc(complex double *xin, complex double *yin, int dim){
  
  complex double p = 0.0 +I*0.0;
  int i;
  
  for (i = 0; i < dim; i++) {
    p += conj(xin[i])*yin[i];
  }

  return p;
  
}

/*! Compute the dot product of two real
 * vectors.
 * \retval p Dot product output, real number.
 * \param[in] xin Input real vector.
 * \param[in] yin Input real vector.
 * \param[in] dim Dimension of the input vectors.
 */
double sopt_utility_dotpr(double *xin, double *yin, int dim){
  
  double p = 0.0;
  int i;
  
  for (i = 0; i < dim; i++) {
    p += xin[i]*yin[i];
  }

  return p;
  
}

/*! Performs a bactracking search for the appropriate
 * Lipchitz constant in sopt_prox_l2b. Complex case.
 * 
 * \param[out] v update vector in the measurement domain. Complex vector of dimension ny.
 * \param[in] r Previuos residual. Complex vector of dimension ny.
 * \param[in] w Previous update point. Complex vector of dimension ny.
 * \param[in] dummy Auxiliary complex vector of dimension ny.
 * \param[in] xaux Auxiliary complex vector of dimension nx.
 * \param[in] xout Previous solution. Complex vector of dimension nx.
 * \param[in] xin Input point. Complex vector of dimension nx.
 * \param[in] A Pointer to the measurement operator \f$A\f$.
 * \param[in] A_data Data structure associated to \f$A\f$.
 * \param[in] At Pointer to the the adjoint of the measurement operator 
 * \f$A^\dagger\f$.
 * \param[in] At_data Data structure associated to \f$A^\dagger\f$.
 * \param[in] nx Dimension of the solution space.
 * \param[in] ny Dimension of the measurement space.
 * \param[in] nu Initial operator norm.
 * \param[in] epsilon Radius of the l2 ball \f$\epsilon\f$.
 */
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
                          double epsilon){
  
  complex double alpha = 0.0 +I*0.0;
  double kappa=nu, temp=0.0;
  double fval=0.0, qval=0.0, faval=0.0;
  int iter = 1;
  int verbose = 0;


  //Computing F and Q.
  //Q = 0.5*||Atu||_2^2 - <u,Axin> + <u-v,Axout> + 0.5*kappa*||v-u||_2^2
  //
  //F = 0.5*||Atv||_2^2 - <v,Axin>

  //faval stores the unchanged portion of Q

  A((void*)dummy, (void*)xout, A_data);
  alpha = sopt_utility_dotpc(w,dummy,ny);
  faval = creal(alpha);
  alpha = sopt_utility_dotpc(v,dummy,ny);
  qval = -creal(alpha);

  A((void*)dummy, (void*)xin, A_data);
  alpha = sopt_utility_dotpc(w,dummy,ny);
  faval -= creal(alpha);
  At((void*)xaux, (void*)w, At_data);
  temp = cblas_dznrm2(nx, (void*)xaux, 1);
  temp = 0.5*temp*temp;
  faval += temp;

  
  alpha = sopt_utility_dotpc(v,dummy,ny); 
  fval = -creal(alpha);
  At((void*)xaux, (void*)v, At_data);
  temp = cblas_dznrm2(nx, (void*)xaux, 1);
  temp = 0.5*temp*temp;
  fval += temp;
  
  cblas_zcopy(ny, (void*)v, 1, (void*)dummy, 1);
  alpha = -1.0 +0.0*I;
  cblas_zaxpy(ny, (void*)&alpha, (void*)w, 1, (void*)dummy, 1);
  temp = cblas_dznrm2(ny, (void*)dummy, 1);
  temp = kappa*0.5*temp*temp;
  qval += faval + temp;

  

  while ((fval > 1.0*qval)&&(iter < 50)){
    kappa = 2*kappa;
    //dummy = r;
    cblas_zcopy(ny, (void*)r, 1, (void*)dummy, 1);
    //Rescaling
    //dummy = kappa*w + dummy
    alpha = kappa + 0.0*I;
    cblas_zaxpy(ny, (void*)&alpha, (void*)w, 1, (void*)dummy, 1);

    //Projection onto the l2 ball
    temp = cblas_dnrm2(ny, (double*)dummy, 1);
    temp = min(1.0,epsilon/temp);

    //v = 1/kappa*(1.0 - norm_res)*dummy
    cblas_zcopy(ny, (void*)dummy, 1, (void*)v, 1);
    cblas_zdscal(ny, (1.0 - temp)/kappa, (void*)v,1);

    //Evaluation of F and Q
    A((void*)dummy, (void*)xout, A_data);
    alpha = sopt_utility_dotpc(v,dummy,ny);
    qval = faval - creal(alpha);
    cblas_zcopy(ny, (void*)v, 1, (void*)dummy, 1);
    alpha = -1.0 +0.0*I;
    cblas_zaxpy(ny, (void*)&alpha, (void*)w, 1, (void*)dummy, 1);
    temp = cblas_dznrm2(ny, (void*)dummy, 1);
    temp = kappa*0.5*temp*temp;
    qval += temp;

    A((void*)dummy, (void*)xin, A_data);
    alpha = sopt_utility_dotpc(v,dummy,ny); 
    fval = -creal(alpha);
    At((void*)xaux, (void*)v, At_data);
    temp = cblas_dznrm2(nx, (void*)xaux, 1);
    temp = 0.5*temp*temp;
    fval += temp;

    iter++;
    //Step back
    //kappa = 2*kappa;

  }
  if (verbose > 0){
            printf("Number of iterations = %i, L = %e \n ", iter-1, kappa);
            printf("F = %e, Q = %e \n ", fval, qval);
  }

  

  return kappa;
  
}

/*! Performs a bactracking search for the appropriate
 * Lipchitz constant in sopt_prox_l2b. Real case.
 * 
 * \param[out] v update vector in the measurement domain. Real vector of dimension ny.
 * \param[in] r Previuos residual. Real vector of dimension ny.
 * \param[in] w Previous update point. Real vector of dimension ny.
 * \param[in] dummy Auxiliary real vector of dimension ny.
 * \param[in] xaux Auxiliary real vector of dimension nx.
 * \param[in] xout Previous solution. Real vector of dimension nx.
 * \param[in] xin Input point. Real vector of dimension nx.
 * \param[in] A Pointer to the measurement operator \f$A\f$.
 * \param[in] A_data Data structure associated to \f$A\f$.
 * \param[in] At Pointer to the the adjoint of the measurement operator 
 * \f$A^\dagger\f$.
 * \param[in] At_data Data structure associated to \f$A^\dagger\f$.
 * \param[in] nx Dimension of the solution space.
 * \param[in] ny Dimension of the measurement space.
 * \param[in] nu Initial operator norm.
 * \param[in] epsilon Radius of the l2 ball \f$\epsilon\f$.
 */
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
                          double epsilon){
  
  double kappa=nu, temp=0.0;
  double fval=0.0, qval=0.0, faval=0.0;
  int iter = 1;


  //Computing F and Q.
  //Q = 0.5*||Atu||_2^2 - <u,Axin> + <u-v,Axout> + 0.5*kappa*||v-u||_2^2
  //
  //F = 0.5*||Atv||_2^2 - <v,Axin>

  //faval stores the unchanged portion of Q

  A(dummy, xout, A_data);
  temp = sopt_utility_dotpr(w,dummy,ny);
  faval = temp;
  temp = sopt_utility_dotpr(v,dummy,ny);
  qval = -temp;

  A(dummy, xin, A_data);
  temp = sopt_utility_dotpr(w,dummy,ny);
  faval -= temp;
  At(xaux, w, At_data);
  temp = cblas_dnrm2(nx, xaux, 1);
  temp = 0.5*temp*temp;
  faval += temp;

  
  temp = sopt_utility_dotpr(v,dummy,ny); 
  fval = -temp;
  At(xaux, v, At_data);
  temp = cblas_dnrm2(nx, xaux, 1);
  temp = 0.5*temp*temp;
  fval += temp;
  
  cblas_dcopy(ny, v, 1, dummy, 1);
  cblas_daxpy(ny, -1.0, w, 1, dummy, 1);
  temp = cblas_dnrm2(ny, dummy, 1);
  temp = kappa*0.5*temp*temp;
  qval += faval + temp;

  while ((fval > qval)&&(iter < 50)){
    //Step back
    kappa = 2*kappa;
    //dummy = r;
    cblas_dcopy(ny, r, 1, dummy, 1);
    //Rescaling
    //dummy = kappa*w + dummy
    cblas_daxpy(ny, kappa, w, 1, dummy, 1);

    //Projection onto the l2 ball
    temp = cblas_dnrm2(ny, dummy, 1);
    temp = min(1.0,epsilon/temp);

    //v = 1/kappa*(1.0 - norm_res)*dummy
    cblas_dcopy(ny, dummy, 1, v, 1);
    cblas_dscal(ny, (1.0 - temp)/kappa, v,1);

    //Evaluation of F and Q
    A(dummy, xout, A_data);
    temp = sopt_utility_dotpr(v,dummy,ny);
    qval = faval - temp;
    cblas_dcopy(ny, v, 1, dummy, 1);
    cblas_daxpy(ny, -1.0, w, 1, dummy, 1);
    temp = cblas_dnrm2(ny, dummy, 1);
    temp = kappa*0.5*temp*temp;
    qval += temp;

    A(dummy, xin, A_data);
    temp = sopt_utility_dotpr(v,dummy,ny); 
    fval = -temp;
    At(xaux, v, At_data);
    temp = cblas_dnrm2(nx, xaux, 1);
    temp = 0.5*temp*temp;
    fval += temp;

    iter++;

  }

  return kappa;
  
}

/*! Conjugate gradient routine to solve the linear system 
 * \f[
 * xin = (A^{T}A + 2I)xout
 * \f]
 * in the SDMM algorithm. Complex case.
 *
 * \param[out] xout Solution of the system. Complex vector of dimension nx.
 * \param[in] xin . Complex vector of dimension nx.
 * \param[in] v Auxiliary complex vector of dimension ny.
 * \param[in] r Auxiliary complex vector of dimension nx.
 * \param[in] p Auxiliary complex vector of dimension nx.
 * \param[in] ap Auxiliary complex vector of dimension nx.
 * \param[in] A Pointer to the measurement operator \f$A\f$.
 * \param[in] A_data Data structure associated to \f$A\f$.
 * \param[in] At Pointer to the the adjoint of the measurement operator 
 * \f$A^\dagger\f$.
 * \param[in] At_data Data structure associated to \f$A^\dagger\f$.
 * \param[in] nx Dimension of the solution space.
 * \param[in] ny Dimension of the measurement space.
 * \param[in] tol tolerance on the relative residual 
 *            \f$||A^{T}Axout-xin||_2^2/||xin||_2^2\f$.
 * \param[in] nit Maximum number of iterations.
 * \param[in] verbose verbosity.
 */
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
                            int verbose){
  int i;
  double temp, rsold, rsnew, alpha, nb, relres;
  complex double beta;
  
  //Check if the input vector is the 
  //zero vector
  temp = cblas_dznrm2(nx, (void*)xin, 1);
  temp = temp*temp;
  nb = temp;
  
  if (temp > 0){
    //r=xin-P(xin)
    //Assumes xout is near xin as a first guess
    //xout = P(xin) = AtA(xin)+2xin
    A(v, xin, A_data);
    At(xout, v, At_data);
    beta = 2.0 + 0.0*I;
    cblas_zaxpy(nx, (void*)&beta, (void*)xin, 1, (void*)xout, 1);
    //r=xin
    cblas_zcopy(nx, (void*)xin, 1, (void*)r, 1);
    //r=r-xout
    beta = -1.0 + 0.0*I;
    cblas_zaxpy(nx, (void*)&beta, (void*)xout, 1, (void*)r, 1);
    //p=r
    cblas_zcopy(nx, (void*)r, 1, (void*)p, 1);
    rsold = cblas_dznrm2(nx, (void*)r, 1);
    rsold = rsold*rsold;
    //xout=xin
    cblas_zcopy(nx, (void*)xin, 1, (void*)xout, 1);
    
    //Main loop
    for (i=1; i <= nit; i++){
      //ap=AtA(p)+2p
      A(v, p, A_data);
      At(ap, v, At_data);
      beta = 2.0 + 0.0*I;
      cblas_zaxpy(nx, (void*)&beta, (void*)p, 1, (void*)ap, 1);
      //alpha=rsold/(p'*ap)
      temp = creal(sopt_utility_dotpc(p, ap, nx));
      alpha = rsold/temp;
      //xout=xout+alpha*p
      beta = alpha + 0.0*I;
      cblas_zaxpy(nx, (void*)&beta, (void*)p, 1, (void*)xout, 1);
      //r=r-alpha*ap
      beta = -alpha + 0.0*I;
      cblas_zaxpy(nx, (void*)&beta, (void*)ap, 1, (void*)r, 1);
      rsnew = cblas_dznrm2(nx, (void*)r, 1);
      rsnew = rsnew*rsnew;
      relres = rsnew/nb;
      if (relres < tol){
        break;
      }
      //p=r+temp*p
      temp = rsnew/rsold;
      //cblas_zcopy(nx, (void*)r, 1, (void*)p, 1);
      cblas_zdscal(nx, temp, (void*)p, 1);
      beta = 1.0 + 0.0*I;
      cblas_zaxpy(nx, (void*)&beta, (void*)r, 1, (void*)p, 1);
      rsold = rsnew;
      //Log
      if (verbose > 2){
        printf("Iter %i: tol = %e, res = %e, rel res = %e\n ", i, tol, rsnew, relres);
      }
    }

  }
  else{
    //If temp = 0 then xout=xin
    cblas_zcopy(nx, (void*)xin, 1, (void*)xout, 1);
    i = 0;
    relres = 1.0;
    rsnew = 0.0;

  }

  //Log
  if (verbose > 1){
    printf("Conjugate gradient: res = %e, rel res = %e, num. iter = %i\n ", rsnew, relres, i);
  }


}

/*! Conjugate gradient routine to solve the linear system 
 * \f[
 * xin = (A^{T}A + 2I)xout
 * \f]
 * in the SDMM algorithm. Real case.
 *
 * \param[out] xout Solution of the system. Real vector of dimension nx.
 * \param[in] xin . Real vector of dimension ny.
 * \param[in] v Auxiliary real vector of dimension ny.
 * \param[in] r Auxiliary real vector of dimension nx.
 * \param[in] p Auxiliary real vector of dimension nx.
 * \param[in] ap Auxiliary real vector of dimension nx.
 * \param[in] A Pointer to the measurement operator \f$A\f$.
 * \param[in] A_data Data structure associated to \f$A\f$.
 * \param[in] At Pointer to the the adjoint of the measurement operator 
 * \f$A^\dagger\f$.
 * \param[in] At_data Data structure associated to \f$A^\dagger\f$.
 * \param[in] nx Dimension of the solution space.
 * \param[in] ny Dimension of the measurement space.
 * \param[in] tol tolerance on the relative residual 
 *            \f$||A^{T}Axout-xin||_2^2/||xin||_2^2\f$.
 * \param[in] nit Maximum number of iterations.
 * \param[in] verbose verbosity.
 */
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
                            int verbose){
  int i;
  double temp, rsold, rsnew, alpha, nb, relres;
  
  //Check if the input vector is the 
  //zero vector
  temp = cblas_dnrm2(nx, xin, 1);
  temp = temp*temp;
  nb = temp;
  
  if (temp > 0){
    //r=xin-P(xin)
    //Assumes xout is near xin as a first guess
    //xout = P(xin) = AtA(xin)+2xin
    A(v, xin, A_data);
    At(xout, v, At_data);
    cblas_daxpy(nx, 2.0, xin, 1, xout, 1);
    //r=xin
    cblas_dcopy(nx, xin, 1, r, 1);
    //r=r-xout
    cblas_daxpy(nx, -1.0, xout, 1, r, 1);
    //p=r
    cblas_dcopy(nx, r, 1, p, 1);
    rsold = cblas_dnrm2(nx, r, 1);
    rsold = rsold*rsold;
    //xout=xin
    cblas_dcopy(nx, xin, 1, xout, 1);
    
    //Main loop
    for (i=1; i <= nit; i++){
      //ap=AtA(p)+2p
      A(v, p, A_data);
      At(ap, v, At_data);
      cblas_daxpy(nx, 2.0, p, 1, ap, 1);
      //alpha=rsold/(p'*ap)
      temp = sopt_utility_dotpr(p, ap, nx);
      alpha = rsold/temp;
      //xout=xout+alpha*p
      cblas_daxpy(nx, alpha, p, 1, xout, 1);
      //r=r-alpha*ap
      cblas_daxpy(nx, -alpha, ap, 1, r, 1);
      rsnew = cblas_dnrm2(nx, r, 1);
      rsnew = rsnew*rsnew;
      relres = rsnew/nb;
      if (relres < tol){
        break;
      }
      //p=r+temp*p
      temp = rsnew/rsold;
      cblas_dscal(nx, temp, p,1);
      cblas_daxpy(nx, 1.0, r, 1, p, 1);
      rsold = rsnew;
      //Log
      if (verbose > 2){
        printf("Iter %i: tol = %e, res = %e, rel res = %e\n ", i, tol, rsnew, relres);
      }
    }

  }
  else{
    //If temp = 0 then xout=xin
    cblas_dcopy(nx, xin, 1, xout, 1);
    i = 0;
    relres = 1.0;
    rsnew = 0.0;

  }

  //Log
  if (verbose > 1){
    printf("Conjugate gradient: res = %e, rel res = %e, num. iter = %i\n ", rsnew, relres, i);
  }

}

