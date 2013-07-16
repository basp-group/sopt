//
//  sopt_l2.h
//  
//
//  Created by Rafael Carrillo on 10/11/12.
//
//

#ifndef SOPT_L2
#define SOPT_L2

#include "sopt_prox.h"

/*!  
 * Data structure containing the parameters for solving the weighted l2
 * optimisation problem.
 */
typedef struct {
  /*! Verbose flag: 0 no log, 1 a summary at convergence, 
   * 2 print main steps (default: 1). 
   */
  int verbose;
  /*! Maximum number of iterations for the global L1 problem.*/
  int max_iter;
  /*! Convergence parameter for the DR algorithm, gamma>0.*/
  double gamma;
  /*! Convergence criteria. Minimum relative change of the objective value. */
  double rel_obj;
  /*! Radius of the L2 ball. */
  double epsilon;
  /*! Flag for real output signal, i.e. real l2-prox. 
   *  1 if real, 0 for complex.
   */
  int real_out;
  /*! Flag for real measurements, i.e. real prox l2-ball. 
   *  1 if real, 0 for complex.
   */
  int real_meas;
  /*! Parameters for the L1 prox. */
  sopt_prox_wl2param paramwl2;
  /*! Parameters for the projection onto L2 ball. */
  sopt_prox_l2bparam paraml2b;
  
} sopt_wl2_param;

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
                    sopt_wl2_param param);

#endif
