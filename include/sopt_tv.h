

#ifndef SOPT_TV
#define SOPT_TV
#include "sopt_config.h"

#include "sopt_prox.h"

/*!  
 * Data structure containing the parameters for solving the tv
 * optimisation problem using DR.
 */
typedef struct {
  /*! Verbose flag: 0 no log, 1 a summary at convergence, 
   * 2 print main steps (default: 1). 
   */
  int verbose;
  /*! Maximum number of iterations for the global TV problem.*/
  int max_iter;
  /*! Convergence parameter for the DR algorithm, gamma>0.*/
  double gamma;
  /*! Convergence criteria. Minimum relative change of the objective value. */
  double rel_obj;
  /*! Radius of the L2 ball. */
  double epsilon;  
  /*! Flag for real output signal, i.e. real tv-prox. 
   *  1 if real, 0 for complex.
   */
  int real_out;
  /*! Flag for real measurements, i.e. real prox l2-ball. 
   *  1 if real, 0 for complex.
   */
  int real_meas;
  /*! Parameters for the TV prox. */
  sopt_prox_tvparam paramtv;
  /*!Parameters for the projection onto L2 ball. */
  sopt_prox_l2bparam paraml2b;
  
} sopt_tv_param;

/*!  
 * Data structure containing the parameters for solving the tv
 * optimisation problem using SDMM.
 */
typedef struct {
  /*! Verbose flag: 0 no log, 1 a summary at convergence, 
   * 2 print main steps (default: 1). 
   */
  int verbose;
  /*! Maximum number of iterations for the global TV problem.*/
  int max_iter;
  /*! Convergence parameter for the SDMM algorithm, gamma>0.*/
  double gamma;
  /*! Convergence criteria. Minimum relative change of the objective value. */
  double rel_obj;
  /*! Radius of the L2 ball. */
  double epsilon; 
  /*! Relative tolerance on epsilon. Default 1e-3. */ 
  double epsilon_tol;
  /*! Flag for real data. 
   *  1 if real, 0 for complex.
   */
  int real_data;
  /*! Maximum number of iterations for the conjugate gradient solver. */
  int cg_max_iter;
  /*! Relative tolerance for the conjugate gradient solver. */
  double cg_tol;
  /*! Parameters for the TV prox. */
  sopt_prox_tvparam paramtv;
  
} sopt_tv_sdmmparam;

/*!
 * Data structure containing the parameters for solving the reweighted TV
 * optimisation problem.
 */
typedef struct {
    /*! Verbose flag: 0 no log, 1 a summary at convergence,
     * 2 print main steps (default: 1).
     */
    int verbose;
    /*! Maximum number of iterations for the global RW-TV problem.*/
    int max_iter;
    /*! Convergence criteria. Minimum relative variation in the solution. */
    double rel_var;
    /*! Noise standard deviation in the representation domain. */
    double sigma;
    /*! Flag to define a valid initial solution: 0 the solution in xsol
     is not valid and it will run the tv solver with unitary weights
     to find the initial solution. 1 takes xsol as the initial
     solution (warm start). */
    int init_sol;
    
} sopt_tv_rwparam;


void sopt_tv_solver(void *xsol,
                    int nx1,
                    int nx2,
                    void (*A)(void *out, void *in, void **data), 
                    void **A_data,
                    void (*At)(void *out, void *in, void **data), 
                    void **At_data,
                    void *y,
                    int ny,
                    sopt_tv_param param);

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
                  sopt_tv_sdmmparam param);

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
                    sopt_tv_rwparam paramrwtv);

#endif
