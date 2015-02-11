
#ifndef SOPT_SPARSEMAT
#define SOPT_SPARSEMAT
#include "sopt_config.h"

#include <complex.h>


/*!  
 * Definition of a sparse matrix stored in compressed column storage
 * format.
 *
 * \note
 * See Numerical Recipes for description of compressed column storage.
 */
typedef struct {
  /*! Number of rows. */ 
  int nrows; 
  /*! Number of columns. */
  int ncols; 
  /*! Number of non-zero elements. */
  int nvals; 
  /*! Non-zero elements traversed column by column. Real case. */
  double *vals;
  /*! Non-zero elements traversed column by column. Complex case. */
  complex double *cvals;
  /*! Row index of each non-zero entry. */
  int *rowind;
  /*! Locations in \ref vals and \ref rowind of each new column. */
  int *colptr;
} sopt_sparsemat;




void sopt_sparsemat_free(sopt_sparsemat *mat);
void sopt_sparsemat_explictmat(double **A, sopt_sparsemat *S);
void sopt_sparsemat_fwd_real(double *y, double *x, sopt_sparsemat *A);
void sopt_sparsemat_adj_real(double *y, double *x, sopt_sparsemat *A);
void sopt_sparsemat_fwd_complex(complex double *y, complex double *x,
				  sopt_sparsemat *A);
void sopt_sparsemat_adj_complex(complex double *y, complex double *x,
				  sopt_sparsemat *A);


#endif
