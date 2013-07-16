/*
 *  sopt_prox.h
 *  
 *
 *  Created by Rafael Carrillo on 8/22/12.
 *  Copyright 2012 EPFL. All rights reserved.
 *
 */

#ifndef SOPT_PROX
#define SOPT_PROX

#include <complex.h>

/*!  
 * Parameters for the L1 prox.
 */
typedef struct {
  /*! Verbose flag: 0 no log, 1 a summary at convergence, 
   * 2 print main steps (default: 1). 
   */
  int verbose;
  /*! Maximum number of iterations. */
  int max_iter;
  /*! Minimum relative change of the objective value. */
  double rel_obj;
  /*! Bound on the squared norm of the operator Psi. */
  double nu;
  /*! Tight frame flag: 1 if Psit is a tight frame 
   *or 0 if not (default = 1). 
   */
  int tight;
  /*! Positivity flag: 0 positivity constraint not active, 
   *1 positivity constraint active. 
   */
  int pos;

} sopt_prox_l1param;

/*!  
 * Parameters for the TV prox.
 */
typedef struct {
  /*! Verbose flag: 0 no log, 1 a summary at convergence, 
   * 2 print main steps (default: 1). 
   */
  int verbose;
  /*! Maximum number of iterations. */
  int max_iter;
  /*! Minimum relative change of the objective value. */
  double rel_obj;
  
} sopt_prox_tvparam;

/*!  
 * Parameters for the projection onto the L2 ball.
 */
typedef struct {
  /*! Verbose flag: 0 no log, 1 a summary at convergence, 
   * 2 print main steps (default: 1). 
   */
  int verbose;
  /*! Maximum number of iterations. */
  int max_iter;
  double tol;
  /*! Bound on the squared norm of the operator A. */
  double nu;
  /*! Tight frame flag: 1 if A is a tight frame 
   *or 0 if not (default = 1). 
   */
  int tight;
  /*! Positivity flag: 0 positivity constraint not active, 
   *1 positivity constraint active. 
   */
  int pos;
  /*! Reality flag: 0 reality constraint not active, 
   *1 reality constraint active. 
   */
  int real;
  /*! Backtracking flag: 0 backtracking not active, 
   *1 backtracking active. 
   *Not active if reality or positivity are active.
   */
  int bcktrk;
  
} sopt_prox_l2bparam;

/*!  
 * Parameters for the weighted L2 prox.
 */
typedef struct {
  /*! Verbose flag: 0 no log, 1 a summary at convergence, 
   * 2 print main steps (default: 1). 
   */
  int verbose;
  /*! Maximum number of iterations. */
  int max_iter;
  /*! Minimum relative change of the objective value. */
  double rel_obj;
  /*! Bound on the squared norm of the operator A. */
  double nu;
  /*! Tight frame flag: 1 if Psit is a tight frame 
   *or 0 if not (default = 1). 
   */
  int tight;
  /*! Positivity flag: 0 positivity constraint not active, 
   *1 positivity constraint active. 
   */
  int pos;

} sopt_prox_wl2param;

/*Functions*/

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
                  void *v);

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
                  void *dy);

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
                  void *dy);

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
                   void *v);

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
                   void *u);

#endif
