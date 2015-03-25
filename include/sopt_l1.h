#ifndef SOPT_L1
#define SOPT_L1
#include "sopt_config.h"

#include "sopt_prox.h"

/*!  
 * Data structure containing the parameters for solving the l1
 * optimisation problem using the Douglas-Rachford algorithm.
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
  /*! Flag for real output signal, i.e. real l1-prox. 
   *  1 if real, 0 for complex.
   */
  int real_out;
  /*! Flag for real measurements, i.e. real prox l2-ball. 
   *  1 if real, 0 for complex.
   */
  int real_meas;
  /*! Parameters for the L1 prox. */
  sopt_prox_l1param paraml1;
  /*! Parameters for the projection onto L2 ball. */
  sopt_prox_l2bparam paraml2b;
  
} sopt_l1_param;

/*!  
 * Data structure containing the parameters for solving the l1
 * optimisation problem using the proximal ADMM algorithm.
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

  /*! Flag for real output signal, i.e. real l1-prox. 
   *  1 if real, 0 for complex.
   */
  int real_out;

  /*! Flag for real measurements, i.e. real prox l2-ball. 
   *  1 if real, 0 for complex.
   */
  int real_meas;

  /*! Parameters for the L1 prox. */
  sopt_prox_l1param paraml1;

  /*! Scale toleranace on epsilon (e.g. 1.001). */
  double epsilon_tol_scale;

  /*! Scale parameter when updating Lagrange multipliers (e.g. 0.9). */
  double lagrange_update_scale;

  /*!  Measurement opertor norm squared. */
  double nu;
  
} sopt_l1_param_padmm;


/*!  
 * Data structure containing the parameters for solving the l1
 * optimisation problem using SDMM.
 */
typedef struct {
  /*! Verbose flag: 0 no log, 1 a summary at convergence, 
   * 2 print main steps, 3 print conjugate gradient steps (default: 1). 
   */
  int verbose;
  /*! Maximum number of iterations for the global L1 problem.*/
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
  
} sopt_l1_sdmmparam;



/*!  
 * Data structure containing the parameters for solving the reweighted l1
 * optimisation problem.
 */
typedef struct {
  /*! Verbose flag: 0 no log, 1 a summary at convergence, 
   * 2 print main steps (default: 1). 
   */
  int verbose;
  /*! Maximum number of iterations for the global RW-L1 problem.*/
  int max_iter;
  /*! Convergence criteria. Minimum relative variation in the solution. */
  double rel_var;
  /*! Noise standard deviation in the representation domain. */
  double sigma;
  /*! Flag to define a valid initial solution: 0 the solution in xsol
      is not valid and it will run the l1 solver with unitary weights
      to find the initial solution. 1 takes xsol as the initial 
      solution (warm start). */
  int init_sol;
  
} sopt_l1_rwparam;

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
		    sopt_l1_param param);

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
                    sopt_l1_rwparam paramrwl1);

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
                    sopt_l1_sdmmparam param);

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
                    sopt_l1_rwparam paramrwl1);

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
			  sopt_l1_param_padmm param);

#endif
